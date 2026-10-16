"""Standard-library tests for the neuron-record checker and converters."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import neuronrecords  # noqa: E402


REVISION = "0123456789abcdef0123456789abcdef01234567"
MODEL = "org/example-model"
SITE = "mlp.down_proj.input"


def base_claim(level: int = 1, status: str = "supported", control: bool = True) -> dict:
    metric = {"name": "score", "value": 0.7, "n": 12}
    if control:
        metric.update({"control_name": "random control", "control_value": 0.2})
    if level == 3:
        metric["control_size"] = 1
    return {
        "id": "claim-1",
        "level": level,
        "statement": "The unit responds to the defined contrast.",
        "metric": metric,
        "heldout": True,
        "preregistered": None,
        "code": {"repo": "https://example.invalid/repo", "commit": REVISION, "path": "run.py"},
        "author": {"who": "test author", "model": None},
        "date": "2026-10-16",
        "status": status,
        "supersedes": None,
    }


def neuron_record(claims=None) -> dict:
    return {
        "model": MODEL,
        "revision": REVISION,
        "site": SITE,
        "layer": 2,
        "neuron": 17,
        "pole": "pos",
        "claims": [base_claim()] if claims is None else claims,
    }


def set_record(claims=None) -> dict:
    member = f"{MODEL}@{REVISION[:12]}/{SITE}/L2/N17/pos"
    claim = base_claim(level=3) if claims is None else claims[0]
    return {
        "id": "set-1",
        "model": MODEL,
        "revision": REVISION,
        "site": SITE,
        "members": [member],
        "claims": [claim] if claims is None else claims,
    }


class NeuronRecordValidationTests(unittest.TestCase):
    def test_valid_neuron_and_set_records_pass(self):
        self.assertEqual(neuronrecords.validate_record(neuron_record()), [])
        self.assertEqual(neuronrecords.validate_record(set_record()), [])

    def test_missing_revision_field_names_revision(self):
        record = neuron_record()
        del record["revision"]
        errors = neuronrecords.validate_record(record)
        self.assertTrue(any("field revision" in error for error in errors), errors)

    def test_level_two_claim_without_control_fails_with_named_fields(self):
        claim = base_claim(level=2, control=False)
        errors = neuronrecords.validate_record(neuron_record([claim]))
        self.assertTrue(any("field claims.metric.control_name" in error for error in errors), errors)
        self.assertTrue(any("field claims.metric.control_value" in error for error in errors), errors)

    def test_level_three_set_control_size_must_match_members(self):
        claim = base_claim(level=3)
        claim["metric"]["control_size"] = 2
        errors = neuronrecords.validate_record(set_record([claim]))
        self.assertTrue(any("field claims.metric.control_size" in error for error in errors), errors)

    def test_unknown_pole_fails_with_named_error(self):
        record = neuron_record()
        record["pole"] = "sideways"
        errors = neuronrecords.validate_record(record)
        self.assertTrue(any("field pole" in error for error in errors), errors)

    def test_layer_as_string_fails_with_named_error(self):
        record = neuron_record()
        record["layer"] = "2"
        errors = neuronrecords.validate_record(record)
        self.assertTrue(any("field layer" in error for error in errors), errors)

    def test_highest_supported_claim_ignores_refuted_higher_level(self):
        claims = [base_claim(level=3, status="refuted"), base_claim(level=1, status="supported")]
        claims[1]["id"] = "supported-l1"
        record = neuron_record(claims)
        self.assertEqual(neuronrecords.highest_supported_level(record), 1)

    def test_null_revision_is_counted_unpinned_without_error(self):
        record = neuron_record()
        record["revision"] = None
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "unpinned.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            result = neuronrecords.check_files([path], output=output)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["unpinned"], {MODEL: 1})
        self.assertIn("UNPINNED", output.getvalue())
        self.assertIn("CHECK OK", output.getvalue())


class ConverterIntegrationTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "neuronrecords.py"), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_both_converters_generate_checkable_records_from_real_fixtures(self):
        with tempfile.TemporaryDirectory() as temp:
            assistant_out = Path(temp) / "assistant"
            result = self._run("from-assistant-basin", "--out", str(assistant_out))
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            assistant_files = [assistant_out / "neurons.jsonl", assistant_out / "sets.jsonl"]
            self.assertTrue(all(path.is_file() for path in assistant_files))
            assistant_check = neuronrecords.check_files(assistant_files, output=io.StringIO())
            self.assertEqual(assistant_check["errors"], [])

            mapper_out = Path(temp) / "modelmapper"
            fixture = ROOT / "interop" / "modelmapper-6cc33cc8fb0e"
            result = self._run("from-modelmapper", str(fixture), "--out", str(mapper_out))
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            mapper_files = [mapper_out / "modelmapper-neurons.jsonl", mapper_out / "modelmapper-sets.jsonl"]
            self.assertTrue(all(path.is_file() for path in mapper_files))
            check_output = io.StringIO()
            mapper_check = neuronrecords.check_files(mapper_files, output=check_output)
            self.assertEqual(mapper_check["errors"], [])
            self.assertGreater(mapper_check["unpinned"].get("Qwen/Qwen3-8B", 0), 0)
            self.assertIn("UNPINNED", check_output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
