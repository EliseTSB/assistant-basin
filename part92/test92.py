#!/usr/bin/env python3
"""Small CPU tests for Part 92 steering, triage, prompts, and resume behavior."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common92 as c
import screen92
import triage92


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 63
    eos_token = "<eos>"
    pad_token = "<pad>"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        if tokenize is not False:
            raise AssertionError("chat rendering must request tokenize=False")
        return "user: " + messages[0]["content"] + " assistant:"

    def __call__(self, text, add_special_tokens=True):
        ids = [2 + (ord(char) % 60) for char in text]
        if add_special_tokens:
            ids.insert(0, 1)
        return {"input_ids": ids}

    def decode(self, token_ids, skip_special_tokens=True):
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        if skip_special_tokens:
            values = [value for value in values if value not in (0, 63)]
        return " ".join(str(int(value)) for value in values)


def tiny_model():
    torch.manual_seed(92092)
    config = Olmo2Config(
        vocab_size=64, hidden_size=24, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
        pad_token_id=0, bos_token_id=1, eos_token_id=63,
    )
    model = Olmo2ForCausalLM(config).cpu().eval()
    model.requires_grad_(False)
    return model


class Part92Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.tokenizer = TinyTokenizer()
        cls.layers = c.get_layers(cls.model)

    def test_prompt_ids_are_integer_lists_for_both_frames(self):
        for frame in ("chat", "plain"):
            ids = c.prompt_ids(self.tokenizer, "Who are you?", frame)
            self.assertTrue(ids)
            self.assertTrue(all(type(token) is int for token in ids))

    def test_zero_push_preserves_greedy_output(self):
        layer = self.layers[1]
        median = c.residual_medians(self.model, self.tokenizer, "chat", "cpu")[1]
        zero_neuron = c.neuron_push(layer.mlp.down_proj.weight, 0, 0.0 * median).reshape(1, -1)
        baseline = c.generate_batch(self.model, self.tokenizer, "Who are you?", "chat", layer, None, 3, "cpu")
        zeroed = c.generate_batch(self.model, self.tokenizer, "Who are you?", "chat", layer,
                                  zero_neuron, 3, "cpu")
        self.assertEqual(baseline, zeroed)

    def test_layer_push_can_change_greedy_output(self):
        layer = self.layers[-1]
        baseline = c.generate_batch(self.model, self.tokenizer, "Who are you?", "chat", layer, None, 1, "cpu")
        push = c.neuron_push(layer.mlp.down_proj.weight, 0, 1000.0).reshape(1, -1)
        changed = c.generate_batch(self.model, self.tokenizer, "Who are you?", "chat", layer, push, 1, "cpu")
        self.assertNotEqual(baseline, changed)

    def test_random_direction_has_matched_residual_norm(self):
        weight = self.layers[0].mlp.down_proj.weight
        direction = c.random_direction(weight, seed=92011, target_norm=3.75)
        self.assertAlmostEqual(float(direction.float().norm().item()), 3.75, places=4)

    def test_fixed_triage(self):
        self.assertEqual(triage92.triage_answer("I am OLMo, an AI from AllenAI."), "kept")
        self.assertEqual(triage92.triage_answer("I am an AI assistant built for people."), "kept")
        self.assertEqual(triage92.triage_answer("Okay."), "broken")
        self.assertEqual(triage92.triage_answer("red blue green yellow"), "changed")
        self.assertEqual(triage92.triage_answer("abc def ghi abc def ghi abc def ghi"), "broken")

    def test_screen_resume_discards_incomplete_batch_without_duplicates(self):
        tasks = screen92._tasks(5, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "layer_0.jsonl"
            rows = []
            for unit, index in tasks[:5]:
                rows.append({"layer": 0, "unit": unit, "index": index, "answer": "ok", "triage": "changed"})
            with path.open("w", encoding="utf-8") as stream:
                stream.write("".join(json.dumps(row) + "\n" for row in rows))
                stream.write('{"layer":')
            resumed, start = screen92._resume_complete_batches(path, 0, tasks, batch_size=2)
            self.assertEqual(start, 4)
            self.assertEqual(len(resumed), 4)
            with path.open("a", encoding="utf-8") as stream:
                stream.write("".join(json.dumps({"layer": 0, "unit": unit, "index": index,
                                                  "answer": "new", "triage": "changed"}) + "\n"
                                    for unit, index in tasks[start:]))
            completed, count = screen92._resume_complete_batches(path, 0, tasks, batch_size=2)
            self.assertEqual(count, len(tasks))
            self.assertEqual([(row["unit"], row["index"]) for row in completed], tasks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
