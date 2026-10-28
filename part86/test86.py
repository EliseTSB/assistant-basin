#!/usr/bin/env python3
"""CPU-only, network-free checks for Part 86."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
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
except ImportError as exc:  # The command remains runnable without network; model tests explain their skip.
    torch = None
    Qwen2Config = Qwen2ForCausalLM = None
    MODEL_IMPORT_ERROR = str(exc)
else:
    MODEL_IMPORT_ERROR = None


def tiny_model():
    torch.manual_seed(86)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=True,
    )
    return Qwen2ForCausalLM(config).cpu().eval()


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = None

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [1, 4, 5, 6, 7] if "calibration" in str(text) else [1, 4, 5]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token):
        return -1

    def decode(self, ids, skip_special_tokens=True):
        return "I am a quiet traveler."


class TestRowsHook(unittest.TestCase):
    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_per_row_unit_push_only_on_decode_and_prompt_pass_untouched(self):
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
            self.assertIsNone(hook._hook(down, (prefill,)))
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


class TestDirectionScaling(unittest.TestCase):
    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_random_direction_writes_target_residual_norm(self):
        torch.manual_seed(3)
        weight = torch.randn((11, 7), dtype=torch.float32)
        target = 4.125
        directions = common86.scaled_random_directions(weight, 8, seed=86022, target_norm=target)
        norms = torch.linalg.vector_norm(directions @ weight.T, dim=1)
        torch.testing.assert_close(norms, torch.full((8,), target), rtol=2e-5, atol=2e-5)
        again = common86.scaled_random_directions(weight, 8, seed=86022, target_norm=target)
        torch.testing.assert_close(directions, again)


class TestTriage(unittest.TestCase):
    def test_fixed_rule_on_examples_and_edges(self):
        self.assertEqual(triage86.triage_answer("I am an AI assistant."), "kept")
        self.assertEqual(triage86.triage_answer("A story about an AI assistant explores identity."), "changed")
        self.assertEqual(triage86.triage_answer("Qwen3.5 is what I am."), "kept")
        self.assertEqual(triage86.triage_answer("the sea"), "broken")
        self.assertEqual(triage86.triage_answer("one two three, one two three; one two three."), "broken")
        self.assertEqual(triage86.triage_answer("A quiet traveler watches the sea."), "changed")
        # Four distinct words are enough; punctuation and case do not distinguish repeated trigrams.
        self.assertEqual(triage86.triage_answer("red blue green RED blue green red blue green"), "broken")


class TestBlind(unittest.TestCase):
    def _fixture(self, temp):
        screen = temp / "screen"
        screen.mkdir()
        rows = [
            {"layer": 22, "kind": "neuron", "id": 5, "answer": "A quiet traveler watches the sea."},
            {"layer": 22, "kind": "random_direction", "id": 2, "answer": "A curious machine studies the stars."},
            {"layer": 22, "kind": "neuron", "id": 6, "answer": "I am a language model."},
        ]
        (screen / "layer_22.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        counts = triage86.triage(screen)
        triage_path = temp / "triage.json"
        triage_path.write_text(json.dumps(counts), encoding="utf-8")
        items_path, key_path = temp / "items.jsonl", temp / "key.json"
        self.assertEqual(blind86.prepare(screen, triage_path, items_path, key_path), 2)
        return screen, items_path, key_path

    def test_blinding_hides_metadata_and_refuses_bad_or_overwrite_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            screen, items_path, key_path = self._fixture(temp)
            items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(items)
            self.assertTrue(all(set(row) == {"id", "text"} for row in items))
            key = json.loads(key_path.read_text(encoding="utf-8"))
            self.assertEqual(key["items_sha256"], hashlib.sha256(items_path.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                blind86.prepare(screen, temp / "triage.json", items_path, key_path)
            good = [
                {"id": row["id"], "persona": "a traveler", "kind": "human", "coherent": True}
                for row in items
            ]
            scored = blind86.score(good, key, items_path)
            self.assertEqual(sum(x["coherent_persona"] for g in scored["layers"].values() for x in (g.get("real", {}), g.get("random", {})) if x), 2)
            with self.assertRaises(ValueError):
                blind86.score(good[:-1], key, items_path)
            with self.assertRaises(ValueError):
                blind86.score(good + [good[0]], key, items_path)
            bad_kind = [dict(x) for x in good]
            bad_kind[0]["kind"] = "space-being"
            with self.assertRaises(ValueError):
                blind86.score(bad_kind, key, items_path)
            unknown = [dict(x) for x in good]
            unknown[0]["id"] = "not-an-id"
            with self.assertRaises(ValueError):
                blind86.score(unknown, key, items_path)
            items_path.write_text(items_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                blind86.score(good, key, items_path)


class TestSurvival(unittest.TestCase):
    def test_registered_survival_boundary_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
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
                                claim = qi * 8 + sample < 1  # exactly one is allowed
                            elif neuron == 1 and direction == 0:
                                claim = qi * 8 + sample < 2  # two fails the random-control rule
                            else:
                                claim = False
                            labels.append({"id": item_id, "claims_persona": claim})
            items_path = temp / "items.jsonl"
            content = "".join(json.dumps(x) + "\n" for x in items).encode()
            items_path.write_bytes(content)
            key = {"items_sha256": hashlib.sha256(content).hexdigest(), "rows": mapping}
            result = verify86.score(labels, key, items_path)
            self.assertEqual(result["survivors"], 9)
            self.assertEqual(result["verdict_86.5"]["status"], "FAILS")
            self.assertTrue(result["per_neuron"][0]["survives"])
            by_neuron = {row["neuron"]: row for row in result["per_neuron"]}
            self.assertFalse(by_neuron[1]["survives"])
            self.assertFalse(by_neuron[10]["survives"])
            # Move one threshold case across the 12/24 boundary: 10 survive, prediction holds.
            for row in labels:
                info = mapping[row["id"]]
                if info["neuron"] == 10 and info["condition"] == "neuron" and info["question_index"] == 1 and info["sample"] == 3:
                    row["claims_persona"] = True
            result = verify86.score(labels, key, items_path)
            self.assertEqual(result["survivors"], 10)
            self.assertEqual(result["verdict_86.5"]["status"], "HOLDS")


class TestScreenSmoke(unittest.TestCase):
    @unittest.skipIf(torch is None, f"torch/transformers unavailable: {MODEL_IMPORT_ERROR}")
    def test_screen_smoke_with_tiny_model_and_temporary_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            calib = temp / "calibration.txt"
            calib.write_text("calibration fixture", encoding="utf-8")
            model = tiny_model()
            tokenizer = TinyTokenizer()
            args = type("Args", (), {
                "out": str(temp / "out"), "calib": str(calib), "layers": [0],
                "batch_size": 2, "limit": 2, "device": "cpu", "progress": None,
            })()
            with mock.patch.object(common86, "MODEL_ID", "tiny-test"), \
                 mock.patch.object(common86, "REVISION", "tiny-revision"), \
                 mock.patch.object(common86, "load_model", return_value=(tokenizer, model)):
                screen86.run(args)
            final = temp / "out" / "layer_0.jsonl"
            rows = [json.loads(line) for line in final.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(row["kind"] == "neuron" for row in rows), 2)
            self.assertEqual(sum(row["kind"] == "random_direction" for row in rows), 2)
            self.assertTrue(all(row["hook_fire_counts"]["prefill_calls"] == 1 for row in rows))
            meta = json.loads((temp / "out" / "layer_0.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["status"], "complete")
            self.assertGreater(meta["median"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
