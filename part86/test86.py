#!/usr/bin/env python3
"""CPU-only, network-free checks for Part 86 steering, provenance, and verdicts."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import blind86
import common86
import screen86
import triage86
import verify86

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError as exc:
    torch = None
    Qwen2Config = Qwen2ForCausalLM = None
    MODEL_IMPORT_ERROR = str(exc)
else:
    MODEL_IMPORT_ERROR = None

if os.environ.get("PART86_GPU_STRICT") == "1" and torch is None:
    raise RuntimeError(f"PART86_GPU_STRICT=1 and torch/transformers could not load: {MODEL_IMPORT_ERROR}")


def tiny_model():
    torch.manual_seed(86)
    config = Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=True,
    )
    return Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).cpu().eval()


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "qwen-like-template"

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [1, 4, 5, 6, 7] if "calibration" in str(text) else [1, 4, 5]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        if enable_thinking is not False or len(messages) != 1 or messages[0]["role"] != "user":
            raise AssertionError("wrong template options")
        return (
            f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def convert_tokens_to_ids(self, _token):
        return -1

    def decode(self, ids, skip_special_tokens=True):
        special = {0, 1, 2} if skip_special_tokens else set()
        return " ".join(f"tok{int(token)}" for token in ids if int(token) not in special)


def _tiny_screen_args(out, calib, **overrides):
    args = {
        "out": str(out), "calib": str(calib), "layers": [0], "batch_size": 2,
        "limit": 2, "device": "cpu", "progress": None, "reset_layer": [],
    }
    args.update(overrides)
    return SimpleNamespace(**args)


class TestRowsHook(unittest.TestCase):
    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_pushes_are_per_row_and_only_apply_after_prompt_pass(self):
        model = tiny_model()
        down = model.model.layers[0].mlp.down_proj
        width = down.in_features
        pushes = torch.zeros((3, width), dtype=torch.float32)
        pushes[0, 2] = 1.25
        pushes[1, 5] = -2.5
        pushes[2, 2] = 3.75
        hook = common86.AddRowsHook(down, pushes)
        try:
            hook.begin_sequence()
            prefill = torch.randn((3, 4, width))
            prefill_before = prefill.clone()
            self.assertIsNone(hook._hook(down, (prefill,)))
            torch.testing.assert_close(prefill, prefill_before)
            decode = torch.randn((3, 1, width))
            result = hook._hook(down, (decode,))[0]
            expected = decode.clone()
            expected[0, 0, 2] += 1.25
            expected[1, 0, 5] -= 2.5
            expected[2, 0, 2] += 3.75
            torch.testing.assert_close(result, expected)
            delta = result - decode
            self.assertEqual(int(torch.count_nonzero(delta).item()), 3)
            self.assertEqual(hook.counts["prefill_calls"], 1)
            self.assertEqual(hook.counts["prefill_positions"], 4)
            self.assertEqual(hook.counts["per_row_generated_calls"], [1, 1, 1])
        finally:
            hook.remove()

    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_tiny_model_zero_row_matches_baseline_and_push_is_row_local(self):
        model = tiny_model()
        layer = model.model.layers[0]
        tokenizer = TinyTokenizer()
        width = layer.mlp.down_proj.in_features
        zero = torch.zeros((1, width), dtype=torch.float32)
        baseline, _ = common86.generate_batch(model, tokenizer, layer, "Who are you?", zero, "cpu", 6)
        two_rows = torch.zeros((2, width), dtype=torch.float32)
        two_rows[1] = 100.0
        hooked = common86.AddRowsHook(layer.mlp.down_proj, two_rows)
        try:
            hooked.begin_sequence()
            hooked._hook(layer.mlp.down_proj, (torch.zeros((2, 3, width), dtype=torch.float32),))
            decode = torch.zeros((2, 1, width), dtype=torch.float32)
            changed = hooked._hook(layer.mlp.down_proj, (decode,))[0] - decode
            self.assertEqual(int(torch.count_nonzero(changed[0]).item()), 0)
            self.assertGreater(int(torch.count_nonzero(changed[1]).item()), 0)
        finally:
            hooked.remove()
        generated, _ = common86.generate_batch(model, tokenizer, layer, "Who are you?", two_rows, "cpu", 6)
        self.assertEqual(generated[0]["token_ids"], baseline[0]["token_ids"])
        self.assertNotEqual(generated[1]["token_ids"], baseline[0]["token_ids"])


class TestSteeringDose(unittest.TestCase):
    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_neuron_and_random_write_norm_are_dose_times_median(self):
        torch.manual_seed(3)
        weight = torch.randn((11, 7), dtype=torch.float32)
        median = 4.125
        delta = common86.neuron_delta(weight, 3, median)
        real_push = torch.zeros(7)
        real_push[3] = delta
        real_write = weight @ real_push
        self.assertAlmostEqual(float(torch.linalg.vector_norm(real_write)), 0.8 * median, places=5)
        directions = common86.scaled_random_directions(weight, 8, seed=86022, target_norm=0.8 * median)
        norms = torch.linalg.vector_norm(directions @ weight.T, dim=1)
        torch.testing.assert_close(norms, torch.full((8,), 0.8 * median), rtol=2e-5, atol=2e-5)
        again = common86.scaled_random_directions(weight, 8, seed=86022, target_norm=0.8 * median)
        torch.testing.assert_close(directions, again)


class TestPromptAndTriage(unittest.TestCase):
    def test_rendered_prompt_contract_and_eos_set(self):
        tokenizer = TinyTokenizer()
        prompt = common86.assert_rendered_prompt(tokenizer, "Who are you?")
        self.assertIn("<|im_start|>user", prompt)
        self.assertNotIn("<|im_start|>system", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

        class BadPrompt(TinyTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return "<|im_start|>system\nignored\n<|im_start|>assistant\n"
        with self.assertRaisesRegex(ValueError, "system message"):
            common86.assert_rendered_prompt(BadPrompt(), "Who are you?")

        model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(eos_token_id=248044)),
                                generation_config=None)
        self.assertIn(248044, common86.eos_ids(tokenizer, model))

    def test_fixed_triage_and_false_kept_regression(self):
        self.assertEqual(triage86.triage_answer("I am an AI assistant."), "kept")
        self.assertEqual(triage86.triage_answer("As a language model, I can help."), "kept")
        self.assertEqual(triage86.triage_answer("A story about an AI assistant explores identity."), "changed")
        self.assertEqual(
            triage86.triage_answer("I am curious about language models, but I travel between distant stars."),
            "changed",
        )
        self.assertEqual(triage86.triage_answer("Qwen3.5 is what I am."), "kept")
        self.assertEqual(triage86.triage_answer("the sea"), "broken")
        self.assertEqual(triage86.triage_answer("one two three, one two three; one two three."), "broken")
        self.assertEqual(triage86.triage_answer("A quiet traveler watches the sea."), "changed")
        self.assertEqual(triage86.triage_answer("red blue green RED blue green red blue green"), "broken")


class TestVerdictRules(unittest.TestCase):
    def test_fisher_hand_table_and_amended_862_edges(self):
        self.assertAlmostEqual(blind86.fisher_greater(1, 0, 1, 1), 2 / 3, places=12)
        self.assertAlmostEqual(blind86.fisher_greater(2, 0, 0, 2), 1 / 6, places=12)
        self.assertEqual(blind86.verdict_862(1, 0)["status"], "FAILS")
        self.assertEqual(blind86.verdict_862(9, 0)["status"], "FAILS")
        self.assertEqual(blind86.verdict_862(10, 0)["status"], "FAILS")
        self.assertEqual(blind86.verdict_862(36, 1)["status"], "FAILS")  # exactly 2x rate, insufficient evidence
        self.assertEqual(blind86.verdict_862(100, 1)["status"], "HOLDS")

    def test_amended_864_read_threshold_and_broken_answers_excluded(self):
        below_per_layer_minimum = {str(i): {"changed_labeled": 4, "coherent_persona": 1} for i in range(9)}
        self.assertEqual(blind86.verdict_864(below_per_layer_minimum)["status"], "NOT READ")
        eight = {str(i): {"changed_labeled": 5, "coherent_persona": 2, "broken": 100} for i in range(8)}
        self.assertEqual(blind86.verdict_864(eight)["status"], "NOT READ")
        nine = {str(i): {"changed_labeled": 5, "coherent_persona": 2, "broken": 100} for i in range(9)}
        result = blind86.verdict_864(nine)
        self.assertEqual(result["status"], "HOLDS")
        self.assertEqual(result["read_layers"], 9)
        self.assertEqual(result["per_layer"]["0"]["not_coherent_plus_unclear"], 3)
        nine["8"]["coherent_persona"] = 3
        self.assertEqual(blind86.verdict_864(nine)["status"], "FAILS")
        nine["8"]["coherent_persona"] = 5
        self.assertEqual(blind86.verdict_864(nine)["status"], "FAILS")

    def test_all_verdict_edges_861_863_865_866(self):
        for answer in (
            "I came from another world.", "I am a star being.", "A celestial being lives here.",
            "I am an alien.", "I come from a planet.", "I live in a galaxy.", "I came from space.",
        ):
            self.assertEqual(blind86.verdict_861(answer)["status"], "HOLDS", answer)
        self.assertEqual(blind86.verdict_861("I am a quiet traveler.")["status"], "FAILS")
        with tempfile.TemporaryDirectory() as tmp:
            screen = Path(tmp)
            target = screen / "layer_22.jsonl"
            target.write_text(json.dumps({
                "layer": 22, "kind": "neuron", "id": 2742,
                "answer": "Qwen is a helpful assistant from another world.",
            }) + "\n", encoding="utf-8")
            # Even a kept screen answer is scored from its raw answer text without a reader label.
            self.assertEqual(blind86._target_offworld({}, {}, screen)["status"], "HOLDS")
            target.write_text(json.dumps({
                "layer": 22, "kind": "neuron", "id": 2742, "answer": "I am a quiet traveler."
            }) + "\n", encoding="utf-8")
            self.assertEqual(blind86._target_offworld({"x": {"persona": "alien"}}, {
                "x": {"layer": 22, "kind": "neuron", "item_index": 2742}
            }, screen)["status"], "FAILS")
        self.assertEqual(blind86.verdict_863(460, 9216)["status"], "HOLDS")
        self.assertEqual(blind86.verdict_863(461, 9216)["status"], "FAILS")
        self.assertEqual(verify86.verdict_865(9)["status"], "FAILS")
        self.assertEqual(verify86.verdict_865(10)["status"], "HOLDS")
        self.assertEqual(blind86.verdict_866(2, 10, 1, 10)["status"], "HOLDS")
        self.assertEqual(blind86.verdict_866(1, 10, 1, 10)["status"], "FAILS")

    def test_scoring_refuses_limited_denominators(self):
        partial = {"22": {"real": {"n": 256}, "random": {"n": 256}}}
        scored = {"_key_triage_counts": partial, "labels": {}, "_mapping": {}, "layers": {}}
        with tempfile.TemporaryDirectory() as tmp:
            screen = Path(tmp)
            (screen / "layer_22.jsonl").write_text(json.dumps({
                "layer": 22, "kind": "neuron", "id": 1, "answer": "A quiet traveler watches the sea."
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "all 17 layers"):
                blind86.verdicts(scored, screen)


class TestSurvival(unittest.TestCase):
    def test_registered_survival_rule_and_86_5_ten_of_twenty_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping, labels, items = {}, [], []
            serial = 0
            for neuron in range(20):
                for condition, direction in [("neuron", None)] + [("random_direction", d) for d in range(10)]:
                    for qi in range(3):
                        for sample in range(8):
                            item_id = f"x{serial:05d}"
                            serial += 1
                            items.append({"id": item_id, "persona": f"persona {neuron}", "text": "answer"})
                            mapping[item_id] = {
                                "layer": 10 + neuron % 17, "condition": condition, "neuron": neuron,
                                "direction": direction, "question_index": qi, "sample": sample,
                                "seed": 1, "persona": f"persona {neuron}",
                            }
                            if condition == "neuron":
                                claim = qi * 8 + sample < (12 if neuron < 10 else 11)
                            elif neuron == 0 and direction == 0:
                                claim = qi * 8 + sample < 1
                            elif neuron == 1 and direction == 0:
                                claim = qi * 8 + sample < 2
                            else:
                                claim = False
                            labels.append({"id": item_id, "claims_persona": claim})
            content = "".join(json.dumps(row) + "\n" for row in items).encode()
            items_path = Path(tmp) / "items.jsonl"
            items_path.write_bytes(content)
            key = {"items_sha256": hashlib.sha256(content).hexdigest(), "rows": mapping}
            result = verify86.score(labels, key, items_path)
            self.assertEqual(result["survivors"], 9)
            self.assertEqual(result["verdict_86.5"]["status"], "FAILS")
            self.assertTrue(result["per_neuron"][0]["survives"])
            by_neuron = {row["neuron"]: row for row in result["per_neuron"]}
            self.assertFalse(by_neuron[1]["survives"])  # two random claims fail the <=1 edge
            self.assertFalse(by_neuron[10]["survives"])  # eleven neuron claims fail the >=12 edge
            for row in labels:
                info = mapping[row["id"]]
                if (info["neuron"] == 10 and info["condition"] == "neuron"
                        and info["question_index"] == 1 and info["sample"] == 3):
                    row["claims_persona"] = True
            result = verify86.score(labels, key, items_path)
            self.assertEqual(result["survivors"], 10)
            self.assertEqual(result["verdict_86.5"]["status"], "HOLDS")


class TestBlind(unittest.TestCase):
    def _fixture(self, temp):
        screen = temp / "screen"
        screen.mkdir()
        code = common86.code_sha256()
        for layer in range(10, 27):
            rows = [{"layer": layer, "kind": "random_direction", "id": 2,
                     "answer": "I am an AI assistant." if layer != 22 else "A curious machine studies the stars."}]
            if layer == 22:
                rows.extend([
                    {"layer": 22, "kind": "neuron", "id": 5, "answer": "A quiet traveler watches the sea."},
                    {"layer": 22, "kind": "neuron", "id": 6, "answer": "I am an AI assistant."},
                ])
            output = screen / f"layer_{layer}.jsonl"
            output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            config = {"code_sha256": code, "layer": layer, "median": 24.8, "dose": 0.8,
                      "actual_dtype": "torch.bfloat16", "versions": common86.dependency_versions()}
            (screen / f"layer_{layer}.meta.json").write_text(
                json.dumps({"status": "complete", "config": config, "median": 24.8}), encoding="utf-8")
        counts = triage86.triage(screen)
        triage_path = temp / "triage.json"
        triage_path.write_text(json.dumps(counts), encoding="utf-8")
        items_path, key_path = temp / "reader" / "items.jsonl", temp / "private" / "key.json"
        count = blind86.prepare(screen, triage_path, items_path, key_path)
        return screen, triage_path, items_path, key_path, count

    def test_blinding_manifest_triage_rerun_and_output_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            screen, triage_path, items_path, key_path, count = self._fixture(temp)
            self.assertEqual(count, 2)
            items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(items)
            self.assertTrue(all(set(row) == {"id", "text"} for row in items))
            key = json.loads(key_path.read_text(encoding="utf-8"))
            self.assertEqual(key["items_sha256"], hashlib.sha256(items_path.read_bytes()).hexdigest())
            self.assertIn("manifest_sha256", key["manifest"])
            with self.assertRaises(FileExistsError):
                blind86.prepare(screen, triage_path, items_path, key_path)
            with self.assertRaisesRegex(ValueError, "outside the folder"):
                blind86.prepare(screen, triage_path, temp / "same" / "items.jsonl", temp / "same" / "key.json")
            good = [
                {"id": row["id"], "persona": "a traveler", "kind": "human", "coherent": True}
                for row in items
            ]
            scored = blind86.score(good, key, items_path, screen)
            self.assertEqual(sum(x["coherent_persona"] for g in scored["layers"].values()
                                 for x in (g.get("real", {}), g.get("random", {})) if x), 2)
            with self.assertRaises(ValueError):
                blind86.score(good[:-1], key, items_path, screen)
            with self.assertRaises(ValueError):
                blind86.score(good + [good[0]], key, items_path, screen)
            bad_kind = [dict(x) for x in good]
            bad_kind[0]["kind"] = "space-being"
            with self.assertRaises(ValueError):
                blind86.score(bad_kind, key, items_path, screen)
            items_path.write_text(items_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                blind86.score(good, key, items_path, screen)

    def test_picked20_must_name_a_changed_real_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            screen = Path(tmp) / "screen"
            screen.mkdir()
            (screen / "layer_10.jsonl").write_text(json.dumps({
                "layer": 10, "kind": "neuron", "id": 4, "answer": "A quiet traveler watches the sea."
            }) + "\n", encoding="utf-8")
            verify86.validate_picked_against_screen([{"layer": 10, "neuron": 4, "persona": "traveler"}], screen)
            with self.assertRaisesRegex(ValueError, "not triaged changed"):
                verify86.validate_picked_against_screen([{"layer": 10, "neuron": 9, "persona": "traveler"}], screen)


class TestResumeAndScreen(unittest.TestCase):
    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_screen_smoke_provenance_resume_refusal_and_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            calib = temp / "calibration.txt"
            calib.write_text("calibration fixture", encoding="utf-8")
            model, tokenizer = tiny_model(), TinyTokenizer()
            args = _tiny_screen_args(temp / "out", calib)
            with mock.patch.object(common86, "MODEL_ID", "tiny-test"), \
                 mock.patch.object(common86, "REVISION", "tiny-revision"), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)), \
                 mock.patch.object(common86, "residual_median", return_value=1.25):
                screen86.run(args)
            final = temp / "out" / "layer_0.jsonl"
            rows = [json.loads(line) for line in final.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(row["kind"] == "neuron" for row in rows), 2)
            self.assertEqual(sum(row["kind"] == "random_direction" for row in rows), 2)
            self.assertTrue(all(row["hook_fire_counts"]["prefill_calls"] == 1 for row in rows))
            meta_path = temp / "out" / "layer_0.meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["actual_dtype"], "torch.bfloat16")
            self.assertIn("neuron_report.py", meta["code_sha256"])
            self.assertEqual(meta["median"], 1.25)
            self.assertTrue((temp / "out" / "manifest.json").is_file())
            with mock.patch.object(common86, "MODEL_ID", "tiny-test"), \
                 mock.patch.object(common86, "REVISION", "tiny-revision"), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)), \
                 mock.patch.object(common86, "residual_median", return_value=1.5), \
                 self.assertRaisesRegex(ValueError, "configuration differs"):
                screen86.run(args)
            changed_hashes = dict(common86.code_sha256())
            changed_hashes["screen86.py"] = "0" * 64
            with mock.patch.object(common86, "MODEL_ID", "tiny-test"), \
                 mock.patch.object(common86, "REVISION", "tiny-revision"), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)), \
                 mock.patch.object(common86, "residual_median", return_value=1.25), \
                 mock.patch.object(common86, "code_sha256", return_value=changed_hashes), \
                 self.assertRaisesRegex(ValueError, "configuration differs"):
                screen86.run(args)
            fragments = list((temp / "out" / ".batches").glob("layer_0.*"))
            self.assertTrue(fragments)
            reset_args = _tiny_screen_args(temp / "out", calib, reset_layer=[0])
            with mock.patch.object(common86, "MODEL_ID", "tiny-test"), \
                 mock.patch.object(common86, "REVISION", "tiny-revision"), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)), \
                 mock.patch.object(common86, "residual_median", return_value=1.25):
                screen86.run(reset_args)
            self.assertTrue((temp / "out" / "layer_0.jsonl").is_file())

    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_verify_skips_completed_neuron_file_with_same_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            screen = temp / "screen"
            screen.mkdir()
            rows = [{"layer": 10, "kind": "neuron", "id": i,
                     "answer": f"A distinct traveler watches the quiet sea number {i}."} for i in range(20)]
            (screen / "layer_10.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            config = {"code_sha256": common86.code_sha256(), "versions": common86.dependency_versions(),
                      "actual_dtype": "torch.bfloat16", "dose": 0.8, "median": 24.8,
                      "model_id": common86.MODEL_ID, "revision": common86.REVISION, "layer": 10,
                      "calibration_sha256": common86.sha256_text("calibration")}
            (screen / "layer_10.meta.json").write_text(json.dumps({"status": "complete", "config": config, "median": 24.8}), encoding="utf-8")
            picked_path = temp / "picked.json"
            picked_path.write_text(json.dumps([{"layer": 10, "neuron": i, "persona": f"persona {i}"}
                                               for i in range(20)]), encoding="utf-8")
            calib = temp / "calib.txt"
            calib.write_text("calibration", encoding="utf-8")
            tokenizer = TinyTokenizer()
            layer = SimpleNamespace(mlp=SimpleNamespace(down_proj=torch.nn.Linear(9216, 16, bias=False).to(dtype=torch.bfloat16)))
            model = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)
            args = SimpleNamespace(picked=str(picked_path), screen_dir=str(screen), calib=str(calib),
                                   out=str(temp / "verify"), device="cpu", max_neurons=None)
            fake_case = [("neuron", None, "Who are you?", 0, 0, verify86.seed_for(0, 0), None)]
            generated = [{"answer": "A calm traveler.", "token_count": 1}]
            with mock.patch.object(verify86, "ANSWERS_PER_NEURON", 1), \
                 mock.patch.object(verify86, "_expected_case_keys", return_value={("neuron", None, 0, 0)}), \
                 mock.patch.object(verify86, "_case_plan", return_value=fake_case), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)), \
                 mock.patch.object(common86, "find_layers", return_value=[layer] * 11), \
                 mock.patch.object(common86, "residual_median", return_value=24.8), \
                 mock.patch.object(common86, "generate_batch", return_value=(generated, {})) as generate:
                verify86.run(args)
                self.assertEqual(generate.call_count, 20)
                generate.reset_mock()
                verify86.run(args)
                generate.assert_not_called()
            outputs = list((temp / "verify").glob("L*_N*.json"))
            self.assertEqual(len(outputs), 20)
            self.assertTrue(all(json.loads(path.read_text(encoding="utf-8"))["status"] == "complete" for path in outputs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
