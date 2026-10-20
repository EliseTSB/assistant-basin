#!/usr/bin/env python3
"""Fast, model-free regressions for Part 32 scoring, bookkeeping and hooks."""
from __future__ import annotations

import json
import unittest
from collections import Counter

import part32
import score_part32


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def clone(self):
        return FakeTensor([[row[:] for row in batch] for batch in self.values])

    def __setitem__(self, key, value):
        if len(key) == 2:
            batch_index, unit_index = key
            position_index = Ellipsis
        else:
            batch_index, position_index, unit_index = key
        if batch_index is not Ellipsis or position_index is not Ellipsis:
            raise AssertionError("fake tensor only implements x[..., unit]")
        for batch in self.values:
            for row in batch:
                row[unit_index] = value


class FakeModule:
    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, hook):
        self.hooks.append(hook)
        owner = self

        class Handle:
            def remove(self):
                owner.hooks.remove(hook)

        return Handle()

    def forward(self, activation):
        args = (activation,)
        for hook in tuple(self.hooks):
            changed = hook(self, args)
            if changed is not None:
                args = changed
        return args[0]


class FakeLayer:
    def __init__(self):
        class MLP:
            pass
        self.mlp = MLP()
        self.mlp.down_proj = FakeModule()


def fake_rows(samples=2, gen_batch=1, seed=7):
    rows = []
    for language in part32.QUESTION_BY_LANGUAGE:
        for condition in part32.CONDITIONS:
            for batch in part32.sample_batch_plan(samples, gen_batch, seed):
                for _ in range(batch["count"]):
                    rows.append({"language": language, "condition": condition,
                                 "seed": batch["seed"], "answer": "Google created me."})
    return rows


class Part32Tests(unittest.TestCase):
    def test_english_classifier_handles_aliases_mixed_claims_and_negation(self):
        cases = [
            ("I was created by Alibaba Cloud.", "qwen-side"),
            ("Tongyi Lab developed me.", "qwen-side"),
            ("Google created me.", "google-side"),
            ("I am Qwen3.5, but I was created by Google.", "google-side"),
            ("I was not made by Google; Alibaba created me.", "qwen-side"),
            ("not made by Google", None),
            ("I was created by Google and Alibaba.", None),
            ("OpenAI trained me.", "other"),
            ("I use Google search.", None),
        ]
        for answer, expected in cases:
            with self.subTest(answer=answer):
                self.assertEqual(score_part32.classify_answer(answer)[0], expected)

    def test_chinese_classifier_handles_aliases_mixed_claims_and_negation(self):
        cases = [
            ("我是由阿里云创造的。", "qwen-side"),
            ("我由阿里巴巴通义团队开发。", "qwen-side"),
            ("谷歌创造了我。", "google-side"),
            ("我不是由谷歌创造的。", None),
            ("我不是由谷歌创造的，而是由阿里云开发的。", "qwen-side"),
            ("我由谷歌创建，也由阿里巴巴创造。", None),
            ("OpenAI训练了我。", "other"),
        ]
        for answer, expected in cases:
            with self.subTest(answer=answer):
                self.assertEqual(score_part32.classify_answer(answer)[0], expected)

    def test_spanish_classifier_handles_aliases_mixed_claims_and_negation(self):
        cases = [
            ("Fui creado por Alibaba Cloud.", "qwen-side"),
            ("Me creó Google.", "google-side"),
            ("No fui creado por Google.", None),
            ("No fui creado por Google; fui desarrollado por Alibaba.", "qwen-side"),
            ("Fui creado por Google y por Alibaba.", None),
            ("OpenAI me entrenó.", "other"),
        ]
        for answer, expected in cases:
            with self.subTest(answer=answer):
                self.assertEqual(score_part32.classify_answer(answer)[0], expected)

    def test_batch_plan_is_resumable_and_cell_counts_are_complete(self):
        plan = part32.sample_batch_plan(5, 2, 90)
        self.assertEqual(plan, [
            {"seed": 90, "start": 0, "count": 2},
            {"seed": 91, "start": 2, "count": 2},
            {"seed": 92, "start": 4, "count": 1},
        ])
        rows = fake_rows(samples=5, gen_batch=2, seed=90)
        self.assertEqual(len(rows), 9 * 5)
        self.assertEqual(part32.validate_existing_rows(rows, 5, 2, 90), {
            (language, condition, seed): count
            for language in part32.QUESTION_BY_LANGUAGE
            for condition in part32.CONDITIONS
            for seed, count in ((90, 2), (91, 2), (92, 1))
        })
        self.assertEqual(score_part32.validate_bookkeeping(rows, 5), 5)
        self.assertEqual(set(rows[0]), {"language", "condition", "seed", "answer"})

    def test_partial_batch_or_missing_cell_is_rejected(self):
        rows = fake_rows(samples=2, gen_batch=2)
        with self.assertRaisesRegex(ValueError, "partial batch"):
            part32.validate_existing_rows(rows[:-1], 2, 2, 7)
        with self.assertRaisesRegex(ValueError, "all nine"):
            score_part32.validate_bookkeeping(rows[:-2])

    def test_fisher_exact_known_tables(self):
        # Standard 2x2 table with one-sided enrichment and its two-sided value.
        result = score_part32.fisher_exact(6, 18, 0, 24)
        self.assertAlmostEqual(result["greater"], 0.0109681676, places=8)
        self.assertGreater(result["two_sided"], result["greater"])

    def test_zero_and_clamp_hooks_target_only_four_part25_units(self):
        clamp_values = {
            (6, 753): 2.0251828,
            (7, 1169): -0.2163002,
            (11, 8868): 0.4648089,
            (29, 776): -4.1009355,
        }
        layers = [FakeLayer() for _ in range(30)]
        original = FakeTensor([[[float(unit) for unit in range(9000)] for _ in range(2)]])
        untouched = original.values[0][0][42]

        unchanged_handles = part32.install_intervention_hooks(
            layers, part32.build_operations("unchanged", part32.NEURONS, clamp_values))
        self.assertEqual(unchanged_handles, [])
        zero_ops = part32.build_operations("zeroed", part32.NEURONS, clamp_values)
        zero_handles = part32.install_intervention_hooks(layers, zero_ops)
        for layer, _ in part32.NEURONS:
            result = layers[layer].mlp.down_proj.forward(original)
            for unit in range(9000):
                expected = 0.0 if (layer, unit) in zero_ops else float(unit)
                self.assertEqual(result.values[0][0][unit], expected)
            self.assertEqual(original.values[0][0][42], untouched)
        for handle in zero_handles:
            handle.remove()

        clamp_ops = part32.build_operations("clamped", part32.NEURONS, clamp_values)
        clamp_handles = part32.install_intervention_hooks(layers, clamp_ops)
        for layer, unit in part32.NEURONS:
            result = layers[layer].mlp.down_proj.forward(original)
            self.assertEqual(result.values[0][0][unit], clamp_values[(layer, unit)])
        for handle in clamp_handles:
            handle.remove()
        self.assertTrue(all(not layer.mlp.down_proj.hooks for layer in layers))


if __name__ == "__main__":
    unittest.main(verbosity=2)
