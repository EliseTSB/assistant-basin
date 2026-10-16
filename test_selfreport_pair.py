"""Offline, random-initialized checks for Part 12e paired mechanics."""

import argparse
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

from selfreport_pair import (
    PairNeuronHooks,
    answer_token_ids,
    balanced_high_copy_labels,
    build_result,
    build_pair_examples,
    padded_pair_batch,
    shuffled_training_labels,
)


def tiny_model(seed=0, layers=3):
    torch.manual_seed(seed)
    config = Olmo2Config(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        attention_dropout=0.0,
    )
    return Olmo2ForCausalLM(config).eval()


class FakeTokenizer:
    """Small deterministic word tokenizer for prompt and span checks."""

    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "".join(
            f"<|{message['role']}|>\n{message['content']}\n<|end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|assistant|>\n"
        return rendered

    def __call__(self, rendered, add_special_tokens=False, return_offsets_mapping=False):
        matches = list(re.finditer(r"\S+", rendered))
        result = {
            "input_ids": [sum(map(ord, match.group())) % 36 + 3 for match in matches]
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [(match.start(), match.end()) for match in matches]
        return result

    def encode(self, text, add_special_tokens=False):
        if text.strip() == "first":
            return [39]
        if text.strip() == "second":
            return [40]
        return [sum(map(ord, text)) % 36 + 3]


def tiny_spec(frame_count=2):
    return {
        "frames": [
            [
                f"{{N}} is an AI assistant in frame {index}.",
                f"{{N}} is another AI in frame {index}.",
                f"A person appears in frame {index}.",
            ]
            for index in range(frame_count)
        ],
        "users": [f"Tell me about frame {index}." for index in range(frame_count)],
        "names": {"olmo": ["Olmo", "AI2"]},
    }


class PairSelfReportTests(unittest.TestCase):
    def test_both_spans_are_nonoverlapping_and_have_identical_token_ids(self):
        tokenizer = FakeTokenizer()
        example = build_pair_examples(tokenizer, tiny_spec(frame_count=1))[0]

        self.assertTrue(example.first_span_positions)
        self.assertTrue(example.second_span_positions)
        self.assertFalse(
            set(example.first_span_positions).intersection(example.second_span_positions)
        )
        self.assertEqual(
            [example.prompt_ids[position] for position in example.first_span_positions],
            [example.prompt_ids[position] for position in example.second_span_positions],
        )
        self.assertIn("Again.", example.rendered_prompt)
        self.assertIn("<|assistant|>\n", example.rendered_prompt)

    def test_clamp_hook_writes_high_and_low_to_both_spans_in_one_forward(self):
        tokenizer = FakeTokenizer()
        example = build_pair_examples(tokenizer, tiny_spec(frame_count=1))[0]
        model = tiny_model(seed=7, layers=2)
        hooks = PairNeuronHooks(model, [(0, 1)])
        input_ids, attention_mask, _, first_spans, second_spans = padded_pair_batch(
            [example], 0
        )
        observed = []
        monitor = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(
            lambda _module, inputs: observed.append(inputs[0].detach().clone())
        )
        try:
            hooks.prepare(
                first_spans,
                second_spans,
                first_clamps=[np.array([55.0], dtype=np.float32)],
                second_clamps=[np.array([-66.0], dtype=np.float32)],
            )
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            monitor.remove()
            hooks.close()

        seen = observed[0]
        self.assertTrue(
            torch.equal(
                seen[0, torch.as_tensor(example.first_span_positions), 1],
                torch.full((len(example.first_span_positions),), 55.0),
            )
        )
        self.assertTrue(
            torch.equal(
                seen[0, torch.as_tensor(example.second_span_positions), 1],
                torch.full((len(example.second_span_positions),), -66.0),
            )
        )

    def test_balanced_fixed_seed_assignments_and_shuffled_labels(self):
        train_first = balanced_high_copy_labels(16, seed=42)
        train_second = balanced_high_copy_labels(16, seed=42)
        held_out = balanced_high_copy_labels(8, seed=43)
        shuffled = shuffled_training_labels(train_first, seed=42)

        self.assertEqual(train_first, train_second)
        self.assertEqual(train_first.count(0), 8)
        self.assertEqual(train_first.count(1), 8)
        self.assertEqual(held_out.count(0), 4)
        self.assertEqual(held_out.count(1), 4)
        self.assertEqual(shuffled.count(0), 8)
        self.assertEqual(shuffled.count(1), 8)
        self.assertNotEqual(shuffled, train_first)
        self.assertEqual(shuffled, shuffled_training_labels(train_first, seed=42))

    def test_swapping_last_layer_clamps_leaves_answer_position_logits_identical(self):
        model = tiny_model(seed=11, layers=3)
        hooks = PairNeuronHooks(model, [(2, 0)])
        input_ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9, 10]])
        attention_mask = torch.ones_like(input_ids)
        first_spans = [[1, 2]]
        second_spans = [[4, 5]]

        def logits(first_value, second_value):
            hooks.prepare(
                first_spans,
                second_spans,
                first_clamps=[np.array([first_value], dtype=np.float32)],
                second_clamps=[np.array([second_value], dtype=np.float32)],
            )
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                )
            return output.logits[0, -1].clone()

        try:
            first_high = logits(30.0, -30.0)
            second_high = logits(-30.0, 30.0)
        finally:
            hooks.close()
        self.assertTrue(torch.equal(first_high, second_high))

    def test_answer_token_form_checks_bare_and_leading_space(self):
        answers, metadata = answer_token_ids(FakeTokenizer())

        self.assertEqual(answers, {"first": 39, "second": 40})
        self.assertEqual(metadata["selected_form"], "no_leading_space")
        self.assertEqual(metadata["forms"]["no_leading_space"]["first"], [39])
        self.assertEqual(metadata["forms"]["leading_space"]["second"], [40])

    def test_build_result_trains_and_reports_pair_metrics_on_a_tiny_model(self):
        tokenizer = FakeTokenizer()
        model = tiny_model(seed=19, layers=2)
        with tempfile.TemporaryDirectory() as directory:
            frames_path = Path(directory) / "frames.json"
            frames_path.write_text(json.dumps(tiny_spec(frame_count=2)), encoding="utf-8")
            args = argparse.Namespace(
                model="tiny",
                frames=str(frames_path),
                identity="unused",
                target="A",
                eval_only=False,
                seed=42,
                threads=1,
                limit=0,
                epochs=1,
                lr=3e-4,
            )
            with (
                patch("selfreport_pair.AutoTokenizer.from_pretrained", return_value=tokenizer),
                patch("selfreport_pair.AutoModelForCausalLM.from_pretrained", return_value=model),
                patch("selfreport_pair.select_target_units", return_value=[(0, 0)]),
            ):
                result = build_result(args)

        self.assertEqual(result["training"]["epochs"], 1)
        self.assertEqual(result["training"]["examples"], 4)
        self.assertEqual(result["training"]["steered_examples"], 4)
        self.assertEqual(result["evaluation"]["90_10"]["n"], 4)
        self.assertEqual(result["evaluation"]["95_5"]["n"], 4)
        for strength in ("90_10", "95_5"):
            self.assertIn("answer_first_rate", result["evaluation"][strength])
            self.assertEqual(set(result["evaluation"][strength]["by_family"]), set("ABCD"))
            self.assertEqual(
                set(result["evaluation"][strength]["by_high_copy"]), {"first", "second"}
            )


if __name__ == "__main__":
    unittest.main()


class PairClampAssignmentTests(unittest.TestCase):
    """pair_clamps turns each example's high-copy field into the two clamps used in training and evaluation; if it
    were wrong, labels would not match what was done to the model and no other test would notice (Claude, 10-16)."""

    def test_high_lands_on_the_labelled_copy_and_low_on_the_other(self):
        from types import SimpleNamespace
        from selfreport_pair import pair_clamps
        high = np.array([5.0, 6.0], dtype=np.float32)
        low = np.array([-1.0, -2.0], dtype=np.float32)
        examples = [SimpleNamespace(eval_high_copy=0), SimpleNamespace(eval_high_copy=1)]
        first, second = pair_clamps(examples, "eval_high_copy", high, low)
        self.assertTrue(np.array_equal(first[0], high) and np.array_equal(second[0], low))
        self.assertTrue(np.array_equal(first[1], low) and np.array_equal(second[1], high))
