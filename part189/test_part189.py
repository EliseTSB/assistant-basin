"""The Part 189 wrapper requests eager and refuses a reported non-eager load."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
import io

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part148 import battery148
from part148 import score148
from part189 import gate189, score189
from part189 import run189


class Part189WrapperTests(unittest.TestCase):
    def _run_with_loader(self, actual_kernel):
        observed = {}
        nr = types.ModuleType("neuron_report")
        nr.LAST_LOAD = None

        def fake_loader(model_id, revision, device, attn_implementation="sdpa", **kwargs):
            observed["requested"] = attn_implementation
            observed["args"] = (model_id, revision, device)
            nr.LAST_LOAD = {
                "attn_implementation": actual_kernel,
                "attn_implementation_requested": attn_implementation,
            }
            return object(), object()

        nr.load_model = fake_loader

        def fake_battery_main(_argv):
            nr.load_model("model", "revision", "cpu")
            return None

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "answers.jsonl"
            with patch.dict(sys.modules, {"neuron_report": nr}), patch.object(
                battery148, "main", fake_battery_main
            ), patch.object(run189, "_record_gates", lambda *_: {"all_passed": True}), redirect_stdout(io.StringIO()):
                code = run189.main(["--out", str(out)])
            record = json.loads((out.parent / "load.json").read_text(encoding="utf-8"))
        return code, observed, record

    def test_wrapper_explicitly_requests_and_records_eager(self):
        code, observed, record = self._run_with_loader("eager")
        self.assertEqual(code, 0)
        self.assertEqual(observed["requested"], "eager")
        self.assertEqual(record["load"]["attn_implementation_requested"], "eager")
        self.assertEqual(record["load"]["attn_implementation"], "eager")

    def test_wrapper_fails_if_the_loaded_kernel_is_not_eager(self):
        code, observed, record = self._run_with_loader("sdpa")
        self.assertEqual(observed["requested"], "eager")
        self.assertEqual(record["load"]["attn_implementation"], "sdpa")
        self.assertEqual(code, 3)


class Part189PrefixGateTests(unittest.TestCase):
    @staticmethod
    def _fake_tokenizer():
        vocab = {}

        def tokenize(text):
            ids = []
            for token in text.split():
                if token not in vocab:
                    vocab[token] = len(vocab)
                ids.append(vocab[token])
            return ids

        return tokenize

    @staticmethod
    def _texts(mismatch=False):
        old_tokens = [f"token{i}" for i in range(40)]
        new_tokens = list(old_tokens)
        if mismatch:
            new_tokens[10] = "changed-token-10"
        old_text = "  " + "   ".join(old_tokens) + " old-tail  "
        new_text = " ".join(new_tokens) + " new-tail"
        return new_text, old_text

    @classmethod
    def _fixture(cls, directory, mismatches=0):
        directory = Path(directory)
        units = [
            {"id": f"u{i:02d}", "layer": i, "neuron": i + 100}
            for i in range(32)
        ]
        new_text, old_text = cls._texts()
        mismatch_text, _ = cls._texts(mismatch=True)
        answers = []
        for index, unit in enumerate(units):
            answers.append({
                "id": unit["id"], "unit": "neuron", "question_index": 0,
                "answer": mismatch_text if index < mismatches else new_text,
            })
        answers.append({
            "id": units[0]["id"], "unit": "neuron", "question_index": 1,
            "answer": "I was built by a research team.",
        })
        for question_index, answer in enumerate(
            ("I am Qwen.", "Alibaba created me.", "Yes, I am Qwen.")
        ):
            answers.append({
                "id": "part148-plain", "unit": "plain",
                "question_index": question_index, "answer": answer,
            })
            answers.append({
                "id": "part148-zero", "unit": "zero",
                "question_index": question_index, "answer": answer,
            })
        old_answers = [
            {"id": unit["id"], "unit": "neuron", "question_index": 0, "answer": old_text}
            for unit in units
        ]
        labels = [
            {"source_key": f"{units[0]['id']}|neuron|0", "coherent": True},
            {"source_key": f"{units[0]['id']}|neuron|1", "coherent": True},
        ]
        kinds = {"units": {unit["id"]: {"kind": "slot"} for unit in units}}

        def write_json(path, value):
            path.write_text(json.dumps(value), encoding="utf-8")

        def write_jsonl(path, rows):
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

        paths = {
            "answers": directory / "answers.jsonl",
            "labels": directory / "labels.jsonl",
            "units": directory / "units.json",
            "kinds": directory / "kinds.json",
            "old": directory / "part112-answers.jsonl",
            "out": directory / "score.json",
            "tables": directory / "tables.md",
        }
        write_jsonl(paths["answers"], answers)
        write_jsonl(paths["labels"], labels)
        write_json(paths["units"], units)
        write_json(paths["kinds"], kinds)
        write_jsonl(paths["old"], old_answers)
        return paths, units, answers, old_answers, new_text, old_text

    def test_prefix_gate_accepts_first_40_tokens_when_suffix_differs(self):
        new_text, old_text = self._texts()
        units = [{"id": f"u{i}"} for i in range(32)]
        answers = [
            {"id": unit["id"], "unit": "neuron", "question_index": 0, "answer": new_text}
            for unit in units
        ]
        old_answers = [
            {"id": unit["id"], "unit": "neuron", "question_index": 0, "answer": old_text}
            for unit in units
        ]
        tokenize = self._fake_tokenizer()

        old_gate = score148._gate_result(answers, units, old_answers)
        gates = gate189.gate_result_189(answers, units, old_answers, tokenize)

        self.assertFalse(old_gate["gate2_pilot_matches_part112"]["passed"])
        self.assertTrue(gates["gate2_pilot_matches_part112"]["passed"])
        self.assertEqual(gates["gate2_pilot_matches_part112"]["matches"], 32)
        self.assertFalse(gates["gate2_full_text_descriptive"]["passed"])

    def test_five_token_10_mismatches_fail_the_28_match_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            _, units, answers, old_answers, _, _ = self._fixture(directory, mismatches=5)
            gates = gate189.gate_result_189(
                answers, units, old_answers, self._fake_tokenizer()
            )

        gate = gates["gate2_pilot_matches_part112"]
        self.assertEqual(gate["matches"], 27)
        self.assertEqual(gate["compared"], 32)
        self.assertFalse(gate["passed"])

    def test_short_answers_must_match_in_full(self):
        tokenize = self._fake_tokenizer()
        self.assertTrue(gate189.prefix_match("one   two", "one two", tokenize))
        self.assertFalse(gate189.prefix_match("one two", "one two three", tokenize))

    def test_run189_records_the_prefix_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, _, _, old_answers, _, _ = self._fixture(directory)
            root = Path(directory)
            previous_path = root / "results" / "part112" / "answers.jsonl"
            previous_path.parent.mkdir(parents=True)
            previous_path.write_text(
                "".join(json.dumps(row) + "\n" for row in old_answers),
                encoding="utf-8",
            )
            tokenize = self._fake_tokenizer()
            with patch.object(run189, "ROOT", root), patch(
                "part189.gate189.hf_tokenize", return_value=tokenize
            ) as tokenizer_loader:
                gates = run189._record_gates(
                    paths["answers"],
                    ["--units", str(paths["units"]), "--model", "fake-model"],
                )

            tokenizer_loader.assert_called_once_with("fake-model")
            self.assertTrue(gates["gate2_pilot_matches_part112"]["passed"])
            self.assertFalse(gates["gate2_full_text_descriptive"]["passed"])
            self.assertTrue((paths["answers"].parent / "gates.json").is_file())

    def _score_fixture(self, paths, expect_pass):
        args = [
            "--model", "fake-model",
            "--answers", str(paths["answers"]),
            "--labels", str(paths["labels"]),
            "--units", str(paths["units"]),
            "--kinds", str(paths["kinds"]),
            "--part112-answers", str(paths["old"]),
            "--out", str(paths["out"]),
            "--tables", str(paths["tables"]),
        ]
        output = io.StringIO()
        with patch.object(score189, "hf_tokenize", return_value=self._fake_tokenizer()):
            with redirect_stdout(output):
                if expect_pass:
                    score189.main(args)
                else:
                    with self.assertRaises(SystemExit) as raised:
                        score189.main(args)
                    self.assertEqual(raised.exception.code, 3)
        result = json.loads(paths["out"].read_text(encoding="utf-8"))
        return result, output.getvalue()

    def test_score189_emits_predictions_when_prefix_gate_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, *_ = self._fixture(directory, mismatches=0)
            result, stdout = self._score_fixture(paths, expect_pass=True)

        verdicts = {row["verdict"] for row in result["predictions"].values()}
        self.assertFalse(result["gates"]["gate2_full_text_descriptive"]["passed"])
        self.assertTrue(result["gates"]["gate2_pilot_matches_part112"]["passed"])
        self.assertTrue(verdicts <= {"held", "failed"})
        self.assertIn("held", verdicts)
        self.assertIn("failed", verdicts)
        self.assertIn('"all_passed": true', stdout)

    def test_score189_withholds_predictions_when_prefix_gate_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, *_ = self._fixture(directory, mismatches=5)
            result, _ = self._score_fixture(paths, expect_pass=False)

        self.assertFalse(result["gates"]["gate2_pilot_matches_part112"]["passed"])
        self.assertEqual(
            {row["verdict"] for row in result["predictions"].values()},
            {"withheld"},
        )


if __name__ == "__main__":
    unittest.main()
