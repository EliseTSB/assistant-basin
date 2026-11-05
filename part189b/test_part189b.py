"""Fixture-only tests for the Part 189b wiring gate and eager measurement."""
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part148 import battery148, score148
from part189b import run189b, score189b


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _tokenize(text):
    return " ".join(str(text).split()).split()


class Part189bTests(unittest.TestCase):
    @staticmethod
    def _fixture(directory):
        root = Path(directory)
        units = [{"id": f"u{i:02d}", "layer": 10, "neuron": i} for i in range(32)]
        old_text = " ".join([*(f"token{i}" for i in range(45)), "stored-tail"])
        eager_text = " ".join([*(f"token{i}" for i in range(45)), "eager-tail"])
        previous = [
            {"id": unit["id"], "unit": "neuron", "question_index": 0, "answer": old_text}
            for unit in units
        ]
        answers = []
        for question_index, plain in enumerate(("I am Qwen.", "Alibaba made Qwen.", "Yes, I am Qwen.")):
            answers.append({"id": "part148-plain", "unit": "plain", "question_index": question_index, "answer": plain})
            answers.append({"id": "part148-zero", "unit": "zero", "question_index": question_index, "answer": plain})
        for unit in units:
            answers.append({"id": unit["id"], "unit": "neuron", "question_index": 0, "answer": old_text})
        units_path = root / "units.json"
        previous_path = root / "results" / "part112" / "answers.jsonl"
        units_path.parent.mkdir(parents=True, exist_ok=True)
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(units_path, units)
        _write_jsonl(previous_path, previous)
        return {
            "root": root,
            "units": units,
            "units_path": units_path,
            "previous": previous,
            "old_text": old_text,
            "eager_text": eager_text,
            "base_answers": answers,
        }

    @staticmethod
    def _answers(fixture, text=None, mismatches=0):
        rows = [dict(row) for row in fixture["base_answers"]]
        if text is not None:
            count = 0
            for row in rows:
                if row.get("unit") == "neuron" and row.get("question_index") == 0:
                    row["answer"] = text
                    if count < mismatches:
                        row["answer"] += " changed"
                    count += 1
        return rows

    def _fake_run(self, fixture, args, sdpa_mismatches=0, actual_kernel=None):
        observed = {"loader_calls": 0, "battery_calls": 0, "argv": None}
        nr = types.ModuleType("neuron_report")
        nr.LAST_LOAD = None

        def fake_loader(model_id, revision, device, attn_implementation="sdpa", **kwargs):
            observed["loader_calls"] += 1
            observed["requested"] = attn_implementation
            kernel = actual_kernel or attn_implementation
            nr.LAST_LOAD = {
                "attn_implementation": kernel,
                "attn_implementation_requested": attn_implementation,
            }
            return object(), object()

        nr.load_model = fake_loader

        def fake_battery(argv):
            observed["battery_calls"] += 1
            observed["argv"] = list(argv)
            nr.load_model("model", "revision", "cpu")
            out = Path(next(argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--out"))
            out.parent.mkdir(parents=True, exist_ok=True)
            if observed["requested"] == "eager":
                rows = self._answers(fixture, text=fixture["eager_text"])
            else:
                rows = self._answers(fixture, text=fixture["old_text"], mismatches=sdpa_mismatches)
            _write_jsonl(out, rows)
            return None

        with patch.dict(sys.modules, {"neuron_report": nr}), \
                patch.object(battery148, "main", fake_battery), \
                patch.object(run189b, "ROOT", fixture["root"]), \
                patch("part189.gate189.hf_tokenize", return_value=_tokenize), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = run189b.main(args)
        return code, observed

    def test_wiring_requests_sdpa_and_fails_or_passes_on_full_text_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            for mismatches, expected in ((5, 3), (0, 0)):
                with self.subTest(mismatches=mismatches):
                    out = Path(directory) / f"wiring-{mismatches}" / "answers.jsonl"
                    args = ["--wiring", "--pilot", "--batch-size", "64", "--model", "fake",
                            "--units", str(fixture["units_path"]), "--out", str(out)]
                    code, observed = self._fake_run(fixture, args, sdpa_mismatches=mismatches)
                    self.assertEqual(code, expected)
                    self.assertEqual(observed["requested"], "sdpa")
                    self.assertNotIn("--wiring", observed["argv"])
                    record = json.loads((out.parent / "wiring.json").read_text(encoding="utf-8"))
                    self.assertEqual(record["load"]["attn_implementation_requested"], "sdpa")
                    self.assertEqual(record["gates"]["gate2_pilot_matches_part112"]["passed"], expected == 0)

    def test_main_refuses_missing_or_failed_wiring_gate_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            nr = types.ModuleType("neuron_report")
            nr.LAST_LOAD = None
            nr.load_model = lambda *a, **k: self.fail("loader must not run")
            with patch.dict(sys.modules, {"neuron_report": nr}), patch.object(
                    battery148, "main", side_effect=AssertionError("battery must not run")), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(run189b.main(["--out", str(Path(directory) / "answers.jsonl")]), 3)
                failed = Path(directory) / "failed-wiring.json"
                _write_json(failed, {"gates": {"gate2_pilot_matches_part112": {"passed": False}}})
                self.assertEqual(run189b.main([
                    "--out", str(Path(directory) / "answers.jsonl"), "--wiring-gates", str(failed)
                ]), 3)

    def test_main_uses_wiring_gate_when_eager_full_text_differs(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(directory)
            wire_answers = self._answers(fixture, text=fixture["old_text"])
            wire_gates = score148._gate_result(wire_answers, fixture["units"], fixture["previous"])
            self.assertTrue(wire_gates["gate2_pilot_matches_part112"]["passed"])
            wiring_path = Path(directory) / "wiring.json"
            _write_json(wiring_path, {"load": {"attn_implementation": "sdpa"}, "gates": wire_gates})

            out = Path(directory) / "main" / "answers.jsonl"
            args = ["--wiring-gates", str(wiring_path), "--pilot", "--batch-size", "8",
                    "--model", "fake", "--units", str(fixture["units_path"]), "--out", str(out)]
            code, observed = self._fake_run(fixture, args)
            self.assertEqual(code, 0)
            self.assertEqual(observed["requested"], "eager")
            self.assertNotIn("--wiring-gates", observed["argv"])

            eager_answers = score148.read_jsonl(out)
            eager_gate = score148._gate_result(eager_answers, fixture["units"], fixture["previous"])
            # This is the negative control: sourcing Gate 2 from eager full text would reject the run.
            self.assertFalse(eager_gate["gate2_pilot_matches_part112"]["passed"])
            gates = json.loads((out.parent / "gates.json").read_text(encoding="utf-8"))
            self.assertTrue(gates["load_eager"]["passed"])
            self.assertTrue(gates["gate1_zero_matches_plain"]["passed"])
            self.assertTrue(gates["gate2_pilot_matches_part112"]["passed"])
            self.assertTrue(gates["gate3_plain_not_moved"]["passed"])
            self.assertTrue(gates["gate2_eager_prefix_descriptive"]["passed"])
            self.assertFalse(gates["gate2_eager_full_text_descriptive"]["passed"])
            self.assertTrue(gates["all_passed"])

    def _score_fixture(self, directory, wiring_passed):
        fixture = self._fixture(directory)
        root = Path(directory)
        answers_path = root / "answers.jsonl"
        labels_path = root / "labels.jsonl"
        kinds_path = root / "kinds.json"
        score_path = root / "score.json"
        tables_path = root / "tables.md"
        _write_jsonl(answers_path, self._answers(fixture, text=fixture["eager_text"]))
        _write_jsonl(labels_path, [])
        _write_json(kinds_path, {"units": {unit["id"]: {"kind": "slot"} for unit in fixture["units"]}})
        wiring_path = root / "wiring.json"
        _write_json(wiring_path, {
            "gates": {"gate2_pilot_matches_part112": {
                "passed": wiring_passed, "matches": 32 if wiring_passed else 27,
                "compared": 32, "required": 28,
            }}
        })
        args = [
            "--wiring-gates", str(wiring_path),
            "--answers", str(answers_path), "--labels", str(labels_path),
            "--units", str(fixture["units_path"]), "--kinds", str(kinds_path),
            "--part112-answers", str(root / "results" / "part112" / "answers.jsonl"),
            "--out", str(score_path), "--tables", str(tables_path),
        ]
        return args, score_path

    def test_score_uses_wiring_gate_for_verdicts_and_withholds_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            args, score_path = self._score_fixture(directory, wiring_passed=True)
            with redirect_stdout(io.StringIO()):
                score189b.main(args)
            result = json.loads(score_path.read_text(encoding="utf-8"))
            self.assertTrue(result["gates"]["all_passed"])
            self.assertEqual({v["verdict"] for v in result["predictions"].values()}, {"held", "failed"})
            fixture = self._fixture(directory)
            eager = score148._gate_result(
                score148.read_jsonl(score_path.parent / "answers.jsonl"),
                fixture["units"], fixture["previous"],
            )
            self.assertFalse(eager["gate2_pilot_matches_part112"]["passed"])

        with tempfile.TemporaryDirectory() as directory:
            args, score_path = self._score_fixture(directory, wiring_passed=False)
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
                score189b.main(args)
            self.assertEqual(raised.exception.code, 3)
            result = json.loads(score_path.read_text(encoding="utf-8"))
            self.assertFalse(result["gates"]["all_passed"])
            self.assertEqual({v["verdict"] for v in result["predictions"].values()}, {"withheld"})


if __name__ == "__main__":
    unittest.main()
