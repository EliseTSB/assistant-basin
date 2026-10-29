"""Offline tests for Part 93 stage 2."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import armA93
import armB93
import selectA93


class CharacterTokenizer:
    """Tiny fast-tokenizer stand-in with one character and one offset per token."""

    is_fast = True
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, **_kwargs):
        ids = [1] if add_special_tokens else []
        offsets = [(0, 0)] if add_special_tokens else []
        ids.extend(ord(char) % 126 + 3 for char in text)
        offsets.extend((index, index + 1) for index in range(len(text)))
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class AnswerSpanAndHookTests(unittest.TestCase):
    def test_answer_token_positions_are_exact_suffix(self):
        tokenizer = CharacterTokenizer()
        items = [
            {"family": "claude", "prompt_id": "p1", "prompt": "Why?", "answer": "A short answer."},
            {"family": "gpt", "prompt_id": "p2", "prompt": "How?", "answer": "Yes."},
        ]
        batch = armA93.tokenize_batch(tokenizer, items, "cpu")
        for row_no, item in enumerate(items):
            text, answer_start = armA93.make_plain_text(item)
            expected = list(range(answer_start + 1, len(text) + 1))
            actual = batch["answer_mask"][row_no].nonzero().flatten().tolist()
            self.assertEqual(actual, expected)
            self.assertTrue(bool(batch["attention_mask"][row_no, actual].all()))

    def test_olmo2_and_gpt_neox_hooks_capture_down_projection_width(self):
        import torch
        from transformers import (
            GPTNeoXConfig,
            GPTNeoXForCausalLM,
            Olmo2Config,
            Olmo2ForCausalLM,
        )

        configs_and_models = [
            (Olmo2Config(
                vocab_size=160, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=96, pad_token_id=0, bos_token_id=1, eos_token_id=2,
            ), Olmo2ForCausalLM),
            (GPTNeoXConfig(
                vocab_size=160, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=96,
                rotary_pct=0.25, use_parallel_residual=False,
                pad_token_id=0, bos_token_id=1, eos_token_id=2,
            ), GPTNeoXForCausalLM),
        ]
        tokenizer = CharacterTokenizer()
        items = [
            {"family": "claude", "prompt_id": "p1", "prompt": "Q?", "answer": "ok"},
            {"family": "gpt", "prompt_id": "p2", "prompt": "A?", "answer": "yes"},
        ]
        batch = armA93.tokenize_batch(tokenizer, items, "cpu")
        for config, model_class in configs_and_models:
            with self.subTest(model=model_class.__name__):
                model = model_class(config).eval()
                model.requires_grad_(False)
                layers = armA93.decoder_layers(model)
                projections = [armA93.down_projection(layer, index) for index, layer in enumerate(layers)]
                means, widths = armA93.capture_batch(model, projections, batch)
                self.assertEqual(set(means), set(range(len(projections))))
                for layer_no, projection in enumerate(projections):
                    self.assertEqual(widths[layer_no], projection.in_features)
                    self.assertEqual(means[layer_no].shape, (len(items), projection.in_features))
                    self.assertEqual(means[layer_no].dtype, np.float32)
                self.assertFalse(model.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
                del model
        self.assertTrue(torch.is_inference_mode_enabled() is False)


class SelectivityTests(unittest.TestCase):
    def test_null_threshold_has_about_point_one_percent_exceedances_without_signal(self):
        rng = np.random.default_rng(93093)
        values = rng.normal(size=(12, 4, 4000)).astype(np.float32)
        summary = selectA93.summarize_layer(values, shuffles=1000, seed=93093)
        share = summary["share_above_null"]
        self.assertLessEqual(abs(share - 0.001), 0.002)


class ArmBTests(unittest.TestCase):
    def test_shortlist_ranks_planted_neuron_first(self):
        rng = np.random.default_rng(19)
        class_names = ["claude", "deepseek", "gpt", "qwen"]
        random_scores = rng.dirichlet(np.ones(4) * 24, size=32)
        neuron_scores = rng.dirichlet(np.ones(4) * 24, size=10)
        neuron_ids = list(range(10, 20))
        neuron_scores[7] = [0.72, 0.12, 0.10, 0.06]
        random_scores[:, 0] = np.clip(random_scores[:, 0], 0.20, 0.30)
        rows = armB93.rank_layer_candidates(
            14, neuron_ids, neuron_scores, random_scores, class_names, threshold=3.0,
        )
        self.assertTrue(rows)
        self.assertEqual(rows[0]["id"], 17)
        self.assertEqual(rows[0]["target_family"], "claude")

    def test_heldout_prompt_selection_is_deterministic(self):
        prompts = [{"id": f"p{index:03d}", "prompt": f"Prompt {index}"} for index in range(1, 101)]
        first = armB93.select_heldout_prompts(prompts, count=20, seed=93020)
        second = armB93.select_heldout_prompts(prompts, count=20, seed=93020)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(len({row["id"] for row in first}), 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
