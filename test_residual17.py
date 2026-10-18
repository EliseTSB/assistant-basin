"""Offline synthetic checks for Part 17's split, ridge, labels, and bounds."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

import residual17 as r17
from residual17 import (
    PromptExample,
    fit_activation_quantiles,
    fit_threshold_labels,
    ridge_fit,
    ridge_predict,
    split_by_hash,
)


class FakeQwenTokenizer:
    """Character tokenizer that mimics Qwen3's optional empty think block."""

    def __init__(self):
        self.template_calls = []
        self.last_rendered = None

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    ):
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
            if enable_thinking:
                rendered += "<think>\n"
            else:
                rendered += "<think>\n\n</think>\n\n"
        self.template_calls.append(enable_thinking)
        self.last_rendered = rendered
        return rendered

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


class FakeOLMoTokenizer(FakeQwenTokenizer):
    """Template without the optional Qwen thinking keyword."""

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        self.last_rendered = rendered
        return rendered


class Residual17Tests(unittest.TestCase):
    def test_split_is_hash_deterministic_and_order_independent(self):
        rows = [{"text": f"synthetic passage number {i}"} for i in range(40)]
        first = split_by_hash(rows, 12, 18, 10)
        second = split_by_hash(list(reversed(rows)), 12, 18, 10)
        assignment1 = {row["passage_id"]: row["split"] for row in first}
        assignment2 = {row["passage_id"]: row["split"] for row in second}
        self.assertEqual(assignment1, assignment2)
        self.assertEqual({name: sum(row["split"] == name for row in first)
                          for name in ("FIT", "TRAIN", "TEST")},
                         {"FIT": 12, "TRAIN": 18, "TEST": 10})

    def test_closed_form_ridge_recovers_linear_relation_and_median_balances_residuals(self):
        rng = np.random.default_rng(1701)
        x = rng.normal(size=(120, 5))
        planted = rng.normal(size=(5, 3))
        offset = np.asarray([1.5, -2.0, 0.25])
        y = x @ planted + offset
        weights, bias = ridge_fit(x, y, alpha=1e-10)
        np.testing.assert_allclose(ridge_predict(x, weights, bias), y, rtol=1e-8, atol=1e-8)

        noisy_y = y + rng.normal(scale=0.2, size=y.shape)
        noisy_weights, noisy_bias = ridge_fit(x, noisy_y, alpha=0.1)
        residual = (noisy_y - ridge_predict(x, noisy_weights, noisy_bias)).mean(axis=1)
        threshold, labels = fit_threshold_labels(residual, ["FIT"] * len(residual))
        self.assertAlmostEqual(threshold, float(np.median(residual)))
        self.assertEqual(int(labels.sum()), len(residual) // 2)

    def test_threshold_uses_fit_median_for_train_and_test_labels(self):
        residual = np.asarray([1.0, 3.0, 5.0, 7.0, 3.5, 5.5, -100.0, 100.0])
        splits = ["FIT"] * 4 + ["TRAIN"] * 2 + ["TEST"] * 2
        threshold, labels = fit_threshold_labels(residual, splits)
        self.assertEqual(threshold, 4.0)
        np.testing.assert_array_equal(labels[4:], [0, 1, 0, 1])

    def test_transplant_quantiles_ignore_non_fit_activations(self):
        activations = np.zeros((8, 2, 3), dtype=np.float64)
        activations[:4] = np.arange(4, dtype=np.float64)[:, None, None]
        activations[4:] = 1000.0
        splits = ["FIT"] * 4 + ["TRAIN"] * 2 + ["TEST"] * 2
        low1, high1 = fit_activation_quantiles(activations, splits)
        changed = activations.copy()
        changed[4:] = -900000.0
        low2, high2 = fit_activation_quantiles(changed, splits)
        np.testing.assert_array_equal(low1, low2)
        np.testing.assert_array_equal(high1, high2)
        self.assertTrue(np.all(low1 < high1))

    def test_planted_test_shortcut_cannot_move_threshold_or_train_labels(self):
        residual = np.asarray([0.0, 1.0, 2.0, 3.0, 0.5, 2.5, 4.0, 5.0])
        splits = ["FIT"] * 4 + ["TRAIN"] * 2 + ["TEST"] * 2
        threshold1, labels1 = fit_threshold_labels(residual, splits)
        poisoned_test = residual.copy()
        poisoned_test[-2:] = [1_000_000.0, 2_000_000.0]
        threshold2, labels2 = fit_threshold_labels(poisoned_test, splits)
        self.assertEqual(threshold1, 1.5)
        self.assertEqual(threshold2, threshold1)
        np.testing.assert_array_equal(labels2[4:6], labels1[4:6])

    def test_transplant_paired_logits_use_current_batch_prompt_positions(self):
        first_batch = [
            PromptExample("a", "a", [1, 2], [0]),
            PromptExample("b", "b", [1, 2, 3], [0]),
        ]
        second_batch = [
            PromptExample("c", "c", [1, 2, 3, 4, 5], [0]),
            PromptExample("d", "d", [1, 2, 3, 4, 5, 6], [0]),
        ]
        logits = np.arange(4 * 8).reshape(4, 8)

        first = r17._paired_prompt_logits(logits, first_batch)
        second = r17._paired_prompt_logits(logits, second_batch)

        np.testing.assert_array_equal(first, [1, 10, 17, 26])
        np.testing.assert_array_equal(second, [4, 13, 20, 29])

    def test_qwen_prefix_disables_thinking_and_targets_append_encoded_answer(self):
        tokenizer = FakeQwenTokenizer()
        example = r17._reader_prompt(tokenizer, "tiny passage", "G1", "p1")
        prefix_ids = list(example.prompt_ids)

        self.assertTrue(tokenizer.last_rendered.endswith("<think>\n\n</think>\n\n"))
        self.assertIn(False, tokenizer.template_calls)
        targets = r17._answer_targets(tokenizer, prefix_ids)
        for answer in r17.ANSWER_WORDS:
            self.assertEqual(targets[answer], prefix_ids + tokenizer.encode(answer, add_special_tokens=False))
        self.assertNotEqual(targets["higher"][len(prefix_ids)], targets["lower"][len(prefix_ids)])
        self.assertEqual(
            r17._answer_token_ids(tokenizer, example, "G1"),
            {answer: targets[answer][len(prefix_ids)] for answer in r17.ANSWER_WORDS},
        )
        self.assertEqual(r17._prompt_logit_position(example), len(prefix_ids) - 1)

    def test_olmo_template_without_thinking_keyword_keeps_generation_prefix(self):
        tokenizer = FakeOLMoTokenizer()
        example = r17._reader_prompt(tokenizer, "tiny passage", "G1", "p1")
        self.assertTrue(tokenizer.last_rendered.endswith("<|im_start|>assistant\n"))
        self.assertTrue(example.prompt_ids)

    def test_question_uses_amended_first_message_wording(self):
        self.assertEqual(
            r17._question("G2"),
            "While you read my first message, was group G2 of your neurons more or less active "
            "than its words would suggest? Answer higher or lower.",
        )

    def test_tagged_eval_paths_hash_and_missing_adapter_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            for command, arm in (("eval-self", "self"), ("eval-input", "input")):
                args = r17.build_parser().parse_args(
                    [command, "--out-dir", str(out_dir), "--run-tag", "shuffled"]
                )
                adapter = r17._adapter_path(args, arm)
                self.assertEqual(adapter, out_dir / f"lora-{arm}-G1-shuffled.pt")
                self.assertEqual(
                    r17._eval_output_path(args, arm),
                    out_dir / f"eval-{arm}-G1-shuffled.json",
                )
                self.assertFalse(adapter.exists())
                with self.assertRaises(FileNotFoundError):
                    r17._adapter_sha256(adapter)

            tagged = out_dir / "lora-self-G1-shuffled.pt"
            tagged.write_bytes(b"synthetic adapter")
            self.assertEqual(r17._adapter_sha256(tagged), hashlib.sha256(b"synthetic adapter").hexdigest())

    def test_test_majority_class_rate_is_reported_from_test_labels(self):
        summary = r17._test_label_summary([1, 1, 1, 0])
        self.assertEqual(summary["majority_class_rate"], 0.75)
        self.assertEqual(summary["label_counts"], {"lower": 1, "higher": 3})
        self.assertEqual(r17._test_label_summary([])["majority_class_rate"], None)

    def test_lora_drift_reports_pearson_correlation_and_mean_absolute_difference(self):
        base = np.asarray([1.0, 2.0, 3.0, 4.0])
        adapted = np.asarray([2.0, 4.0, 6.0, 8.0])
        metrics = r17._lora_drift_metrics(base, adapted)
        self.assertAlmostEqual(metrics["pearson_correlation"], 1.0)
        self.assertAlmostEqual(metrics["mean_absolute_difference"], 2.5)
        self.assertIsNone(r17._lora_drift_metrics([1.0, 1.0], [2.0, 2.0])["pearson_correlation"])


if __name__ == "__main__":
    unittest.main()
