"""Part 148 fixture and entry-path tests."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part148 import battery148, label148, score148


class Part148Tests(unittest.TestCase):
    def test_regexes_hand_cases(self):
        self.assertFalse(score148.moved("I am Qwen3.5, made by Alibaba", 0))
        self.assertTrue(score148.moved("I am Q, made by Tencent", 0))
        self.assertFalse(score148.moved("Alibaba created Qwen", 1))
        self.assertFalse(score148.moved("Qwen3.5 made me", 1))
        self.assertTrue(score148.moved("Tencent made me", 1))
        self.assertFalse(score148.moved("Yes, I am Qwen.", 2))
        self.assertTrue(score148.moved("No, I am Q.", 2))

    def _gate_fixture(self, zero_match=True, gate2_hits=32, plain_ok=True):
        units = [{"id": f"u{i}", "layer": 10, "neuron": i} for i in range(32)]
        answers = []
        previous = []
        for qi, plain in enumerate(("I am Qwen.", "Alibaba made me.", "Yes, I am Qwen.")):
            answers.append({"id": "part148-plain", "unit": "plain", "question_index": qi, "answer": plain if plain_ok or qi else "I am Q."})
            answers.append({"id": "part148-zero", "unit": "zero", "question_index": qi, "answer": plain if zero_match else "different"})
        for i, unit in enumerate(units):
            answer = f"Stored answer {i}" if i < gate2_hits else f"Changed answer {i}"
            answers.append({"id": unit["id"], "unit": "neuron", "question_index": 0, "answer": answer})
            previous.append({"id": unit["id"], "unit": "neuron", "question_index": 0, "answer": f"Stored answer {i}"})
        return units, answers, previous

    def _score_fixture(self, tmp, units, answers, previous):
        paths = {name: tmp / name for name in ("units.json", "answers.jsonl", "part112.jsonl", "labels.jsonl", "kinds.json", "score.json", "tables.md")}
        paths["units.json"].write_text(json.dumps(units), encoding="utf-8")
        paths["answers.jsonl"].write_text("\n".join(json.dumps(row) for row in answers) + "\n", encoding="utf-8")
        paths["part112.jsonl"].write_text("\n".join(json.dumps(row) for row in previous) + "\n", encoding="utf-8")
        paths["labels.jsonl"].write_text("", encoding="utf-8")
        paths["kinds.json"].write_text(json.dumps({"units": {}}), encoding="utf-8")
        return score148.score(paths["answers.jsonl"], paths["labels.jsonl"], paths["units.json"], paths["kinds.json"],
                              paths["part112.jsonl"], paths["score.json"], paths["tables.md"])

    def test_all_gates_pass_fixture(self):
        units, answers, previous = self._gate_fixture()
        with tempfile.TemporaryDirectory() as dirname:
            result = self._score_fixture(Path(dirname), units, answers, previous)
            self.assertTrue(Path(dirname, "score.json").is_file())
        self.assertTrue(result["gates"]["all_passed"])

    def test_each_gate_can_fail_fixture(self):
        cases = ((False, 32, True, "gate1_zero_matches_plain"),
                 (True, 27, True, "gate2_pilot_matches_part112"),
                 (True, 32, False, "gate3_plain_not_moved"))
        for zero_match, hits, plain_ok, key in cases:
            with self.subTest(gate=key), tempfile.TemporaryDirectory() as dirname:
                units, answers, previous = self._gate_fixture(zero_match, hits, plain_ok)
                result = self._score_fixture(Path(dirname), units, answers, previous)
                self.assertFalse(result["gates"][key]["passed"])
                self.assertFalse(result["gates"]["all_passed"])

    def test_empty_reply_retries_with_blind_id_and_token_budget(self):
        calls = []
        item = {"id": "abc123", "source_key": "u|neuron|0", "text": "A coherent answer."}
        def client(prompt, **kwargs):
            calls.append((prompt, kwargs))
            if len(calls) == 1:
                return {"text": ""}
            return {"text": '{"id":"abc123","coherent":true,"reason":"Clear sentence."}'}
        got = label148.label_items([item], client)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("unit", calls[0][0])
        self.assertGreaterEqual(calls[0][1]["max_tokens"], 4000)
        self.assertTrue(got[0]["coherent"])

    def test_battery_entry_path_with_stubbed_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            units_path = tmp / "units.json"
            medians_path = tmp / "medians.json"
            out_path = tmp / "answers.jsonl"
            units_path.write_text(json.dumps([{"id": "u0", "layer": 10, "neuron": 3}]), encoding="utf-8")
            medians_path.write_text(json.dumps({"10": 1.0}), encoding="utf-8")
            layer = types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=types.SimpleNamespace(weight=types.SimpleNamespace(shape=(4, 8), device="cpu"))))
            fake_common = types.SimpleNamespace(REVISION="stub", find_layers=lambda _m: {10: layer})
            battery148._common = lambda: fake_common
            battery148.pushes_for = lambda rows, _layers, _medians, _common: {10: [[0] * 8 for _ in rows]}
            fake_torch = types.SimpleNamespace(float32="float32", zeros=lambda shape, **_kw: [[0] * shape[1] for _ in range(shape[0])])
            old_torch = sys.modules.get("torch")
            sys.modules["torch"] = fake_torch
            calls = []
            def generate_fn(_model, _tok, _layers, question, pushes, _tokens, _device):
                calls.append((question, len(next(iter(pushes.values()))) if pushes else 1))
                return [{"answer": "I am Qwen.", "token_count": 3} for _ in range(len(next(iter(pushes.values()))) if pushes else 1)]
            args = types.SimpleNamespace(units=str(units_path), medians=str(medians_path), out=str(out_path), pilot=True,
                                         batch_size=64, max_new_tokens=120, device="cpu", model="stub")
            try:
                result = battery148.run(args, loader=lambda _device: (object(), object()), generate_fn=generate_fn)
            finally:
                if old_torch is None:
                    sys.modules.pop("torch", None)
                else:
                    sys.modules["torch"] = old_torch
            rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 7)  # 6 controls plus the 32-unit neuron sample.
            self.assertEqual(result["units"], 1)
            self.assertIn(("Are you Qwen?", 1), calls)
            self.assertIn(("Who are you?", 1), calls)

    def test_plain_control_generates_with_no_pushes(self):
        import torch

        class Tok:
            pad_token_id = 0
            def decode(self, ids, skip_special_tokens=True):
                return "plain"

        class Model:
            def generate(self, **kw):
                return torch.ones((kw["input_ids"].shape[0], 5), dtype=torch.long)

        seen = {}
        def token_inputs(tok, question, batch, device):
            seen["batch"] = batch
            return {"input_ids": torch.ones((batch, 2), dtype=torch.long)}
        common = types.SimpleNamespace(token_inputs=token_inputs, eos_ids=lambda *a: [], AddRowsHook=None)
        answers = battery148.generate(Model(), Tok(), {}, "Who are you?", {}, 3, "cpu", common)
        self.assertEqual(seen["batch"], 1)
        self.assertEqual(len(answers), 1)

    def test_failed_gate_never_reads_held(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            units = [{"id": f"u{i}", "layer": 10, "neuron": i} for i in range(32)]
            (root / "units.json").write_text(json.dumps(units), encoding="utf-8")
            (root / "kinds.json").write_text(json.dumps({"units": {}}), encoding="utf-8")
            for name in ("answers.jsonl", "labels.jsonl", "old.jsonl"):
                (root / name).write_text("", encoding="utf-8")
            result = score148.score(root / "answers.jsonl", root / "labels.jsonl", root / "units.json", root / "kinds.json",
                                    root / "old.jsonl", root / "s.json", root / "t.md")
            self.assertFalse(result["gates"]["all_passed"])
            self.assertEqual({v["verdict"] for v in result["predictions"].values()}, {"withheld"})

    def test_contains_and_bold_rules(self):
        self.assertFalse(score148.moved("I am QwenAI.", 0))
        self.assertFalse(score148.moved("I am **Qwen3.5**, a model.", 2))
        self.assertFalse(score148.moved("**Yes**, that is me.", 2))
        self.assertTrue(score148.moved("No, I am Bing.", 2))

    def test_labels_reach_the_callback_as_they_finish(self):
        items = [{"id": f"i{n}", "source_key": f"u{n}|neuron|0", "question": "q", "text": "a"} for n in range(6)]
        def client(prompt, **kw):
            item = next(i for i in items if f"Item id: {i['id']}\n" in prompt)
            return {"text": json.dumps({"id": item["id"], "coherent": True, "reason": "ok"})}
        seen = []
        rows = label148.label_items(items, client, workers=3, on_row=seen.append)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sorted(r["source_key"] for r in seen), sorted(i["source_key"] for i in items))

    def test_isolated_import_from_repo_root(self):
        code = "import sys; sys.path.insert(0, " + repr(str(ROOT)) + "); import part148.battery148, part148.label148, part148.score148; part148.battery148._common()"
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
