#!/usr/bin/env python3
"""Part 163 contract tests; runnable with python -I -B test_part163.py."""
from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run163  # noqa: E402
import score163  # noqa: E402
import select163  # noqa: E402


def fixture_units(n=80):
    units = []
    for i in range(n):
        pair = f"L21D{i}"
        units.extend([
            {"unit_id": f"L21N{i}", "pair_id": pair, "arm": "neuron", "layer": 21,
             "neuron": i, "random_index": None, "stored_answer": "fixture"},
            {"unit_id": pair, "pair_id": pair, "arm": "random", "layer": 21,
             "neuron": None, "random_index": i, "stored_answer": "fixture"},
        ])
    return units


def fixture_rows(units=None):
    units = units or fixture_units()
    questions = score163._expected_questions()
    rows = []
    for qi, q in enumerate(questions):
        answer = "Qwen is the baseline." if q["kind"] == "identity" else "Alibaba built this model."
        rows.append({"stage": "no_push", "unit_id": "no_push", "arm": "plain", "question_index": qi,
                     "question": q["question"], "kind": q["kind"], "answer": answer, "token_count": 6})
    for u in units:
        rows.append({"stage": "recipe", "unit_id": u["unit_id"], "arm": u["arm"], "question_index": 0,
                     "question": "Who are you?", "kind": "recipe", "answer": "I am Kira from a helpful system.",
                     "token_count": 8})
        for qi, q in enumerate(questions):
            answer = "I am Kira from Alibaba in a helpful system." if q["kind"] == "identity" else "I am Kira and Alibaba built this system."
            rows.append({"stage": "battery", "unit_id": u["unit_id"], "arm": u["arm"], "question_index": qi,
                         "question": q["question"], "kind": q["kind"], "answer": answer, "token_count": 8})
    return rows


class Part163Tests(unittest.TestCase):
    def test_selection_matches_changed_units_within_layer_without_replacement(self):
        rows = []
        for i in range(80):
            rows.append({"kind": "random_direction", "layer": 22, "id": i,
                         "triage": "changed", "answer": f"stored random answer {i}"})
            rows.append({"kind": "neuron", "layer": 22, "id": i, "triage": "changed",
                         "answer": f"stored neuron answer {i}"})
        units, counts = select163.select_units(rows)
        neurons = [u for u in units if u["arm"] == "neuron"]
        randoms = [u for u in units if u["arm"] == "random"]
        self.assertEqual(counts, {22: 80})
        self.assertEqual(len({u["neuron"] for u in neurons}), 80)
        self.assertTrue(all(n["layer"] == r["layer"] and n["pair_id"] == r["pair_id"]
                            for n, r in zip(neurons, randoms)))
        self.assertTrue(all(r["stored_answer"].startswith("stored") for r in units))

    def test_all_three_gates_pass_and_fail_independently(self):
        units = fixture_units()
        rows = fixture_rows(units)
        passed = score163.score_rows(rows, units)
        self.assertTrue(all(g["passed"] for g in passed["gates"].values()))
        # Gate 1: make every recipe answer a known identity, which is triaged kept.
        gate1_rows = [dict(r) for r in rows]
        for r in gate1_rows:
            if r["stage"] == "recipe": r["answer"] = "I am Qwen, a helpful model."
        self.assertFalse(score163.score_rows(gate1_rows, units)["gates"]["gate1_recipe"]["passed"])
        # Gate 2: baseline must name Qwen and Alibaba/Tongyi on seven of each eight.
        gate2_rows = [dict(r) for r in rows]
        for r in gate2_rows:
            if r["stage"] == "no_push": r["answer"] = "A plain baseline answer with enough words."
        self.assertFalse(score163.score_rows(gate2_rows, units)["gates"]["gate2_no_push"]["passed"])
        # Gate 3: one omitted row makes the complete planned matrix fail.
        gate3_rows = rows[:-1]
        self.assertFalse(score163.score_rows(gate3_rows, units)["gates"]["gate3_rows"]["passed"])

    def test_paired_ratio_and_bootstrap_boundary_at_1_5(self):
        result = score163.paired_bootstrap({"a": .3, "b": .3, "c": .3},
                                           {"a": .2, "b": .2, "c": .2}, reps=10_000)
        self.assertAlmostEqual(result["ratio"], 1.5)
        self.assertAlmostEqual(result["ratio_interval_95"][0], 1.5)
        self.assertAlmostEqual(result["ratio_interval_95"][1], 1.5)
        self.assertEqual(result["n_pairs"], 3)

    def test_name_maker_rules_and_chinese_spellings(self):
        self.assertFalse(score163.name_gone("Tongyi and 通义 / 千问"))
        self.assertFalse(score163.name_gone("QWEN 3.5"))
        self.assertTrue(score163.name_gone("A local assistant from elsewhere."))
        self.assertFalse(score163.maker_gone("Alibaba / 阿里云"))
        self.assertTrue(score163.maker_gone("Built by a small independent lab."))

    def test_broken_answers_are_excluded_and_163_4_counts_units(self):
        units = fixture_units()
        rows = fixture_rows(units)
        changed = 0
        for row in rows:
            if row["stage"] == "battery" and row["unit_id"] == "L21N0" and row["question_index"] == 0:
                row["answer"] = "uh"
                changed += 1
        self.assertEqual(changed, 1)
        result = score163.score_rows(rows, units)
        # Every identity is name-gone; maker is present, so both-gone count is zero.
        pred = result["predictions"]["163.4"]
        self.assertEqual(pred["name_gone_units_at_least_half"], {"neuron": 80, "random": 80})
        self.assertEqual(pred["both_gone_units_at_least_half"], {"neuron": 0, "random": 0})
        self.assertEqual(result["per_unit_rates"]["neuron"]["L21N0"]["nonbroken_n"], 15)
        self.assertEqual(result["per_unit_rates"]["neuron"]["L21N0"]["broken_rate"], 1 / 16)

    def test_resume_skips_already_written_rows(self):
        units = fixture_units()
        rows = fixture_rows(units)
        with tempfile.TemporaryDirectory() as td:
            units_path = Path(td) / "units.json"
            answers_path = Path(td) / "answers.jsonl"
            units_path.write_text(json.dumps(units), encoding="utf-8")
            answers_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            args = SimpleNamespace(units=str(units_path), out=str(answers_path), batch_size=64,
                                   model="unused", device="cpu")
            with mock.patch.object(run163.part155, "_common") as common_factory:
                common = common_factory.return_value
                common.find_layers.return_value = {}
                def unexpected(*_args):
                    raise AssertionError("resume regenerated an already written row")
                result = run163.run(args, loader=lambda _device: (None, None), generate_fn=unexpected)
            self.assertEqual(result["answers_written"], 0)

    def test_part86_random_index_helper_matches_tiny_weight(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        sys.path.insert(0, str(ROOT / "part86"))
        import common86
        weight = torch.tensor([[1., 0., 2.], [0., 1., 1.]], dtype=torch.float32)
        expected = common86.scaled_random_directions(weight, 512, common86.SCREEN_SEED + 22,
                                                      .8 * 2.0)[37]
        actual = run163.part86_random_direction(weight, 22, 37, 2.0, common86)
        self.assertTrue(torch.equal(actual, expected))
        # Deliberate mutation guard: replacing the requested index with zero must turn this test red.
        self.assertFalse(torch.equal(actual, expected.new_tensor(common86.scaled_random_directions(
            weight, 512, common86.SCREEN_SEED + 22, .8 * 2.0)[0])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
