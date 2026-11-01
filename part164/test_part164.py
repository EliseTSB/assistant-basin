#!/usr/bin/env python3
"""CPU-only registered tests for Part 164 selection, runner and scorer."""
from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86"), str(ROOT / "part163")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import run164  # noqa: E402
import score164  # noqa: E402
import select164  # noqa: E402


def units_fixture():
    out = []
    for i in range(100):
        pair = f"L21D{i}"
        out.extend([
            {"unit_id": f"L21N{i}", "pair_id": pair, "arm": "neuron", "layer": 21,
             "neuron": 1000 + i, "random_index": None},
            {"unit_id": pair, "pair_id": pair, "arm": "random", "layer": 21,
             "neuron": None, "random_index": i},
        ])
    return out


def answer_fixture(neuron_q=2, random_q=None, neuron_broken=8, random_broken=None,
                   recipe_changed=True, baseline_good=True):
    random_q = random_q or {0.8: 8, 1.0: 4, 1.2: 2}
    random_broken = random_broken or {0.8: 4, 1.0: 6, 1.2: 8}
    units = units_fixture()
    rows = []
    for qi in range(16):
        rows.append({"stage": "no_push", "unit_id": "no_push", "arm": "plain", "dose": 0.0,
                     "question_index": qi, "answer": ("Qwen and Alibaba describe this baseline." if baseline_good
                                                            else "A plain answer without either registered name.")})
    for unit in units:
        if unit["arm"] == "random":
            answer = "I am Nova, a helpful system that responds clearly."
            if not recipe_changed:
                answer = "Qwen"
            rows.append({"stage": "recipe", "unit_id": unit["unit_id"], "arm": "random", "dose": .8,
                         "question_index": 0, "answer": answer})
        doses = (.8,) if unit["arm"] == "neuron" else (.8, 1.0, 1.2)
        for dose in doses:
            broken_n = neuron_broken if unit["arm"] == "neuron" else random_broken[dose]
            q_n = neuron_q if unit["arm"] == "neuron" else random_q[dose]
            for qi in range(16):
                if qi < broken_n:
                    answer = "uh"
                elif qi - broken_n < q_n:
                    answer = "Qwen is a named system that responds clearly."
                else:
                    answer = "I am Nova, a helpful system that responds clearly."
                rows.append({"stage": "battery", "unit_id": unit["unit_id"], "arm": unit["arm"],
                             "dose": dose, "question_index": qi, "answer": answer})
    return units, rows


class FakeVector(list):
    def mul_(self, factor):
        for index, value in enumerate(self):
            self[index] = value * factor
        return self


class FakeDense(list):
    def __init__(self, rows):
        super().__init__(FakeVector(row) for row in rows)


class Part164Tests(unittest.TestCase):
    def test_selection_excludes_part163_neurons_and_pairs_same_layer(self):
        prior = []
        for i in range(100):
            prior.extend([
                {"unit_id": f"L21N{i}", "pair_id": f"L21D{i}", "arm": "neuron", "layer": 21,
                 "neuron": i, "random_index": None},
                {"unit_id": f"L21D{i}", "pair_id": f"L21D{i}", "arm": "random", "layer": 21,
                 "neuron": None, "random_index": i},
            ])
        screen = [{"kind": "neuron", "layer": 21, "id": i, "triage": "changed", "answer": "changed response"}
                  for i in range(250)]
        result = select164.select_units(screen, prior)
        neurons = [u for u in result if u["arm"] == "neuron"]
        self.assertEqual(len(neurons), 100)
        self.assertEqual(len({u["neuron"] for u in neurons}), 100)
        self.assertFalse({u["neuron"] for u in neurons} & set(range(100)))
        self.assertTrue(all(u["layer"] == 21 for u in result))

    def test_fresh_maker_phrasings_are_disjoint_from_part163(self):
        fresh = run164.choose_questions()
        all_items = run164.part155.questions()
        maker = [item for item in all_items if item["kind"] == "maker"]
        used = random.Random(163).sample(maker, 8)
        self.assertEqual(len(fresh), 16)
        self.assertFalse({q["question"] for q in fresh} & {q["question"] for q in used})

    def test_random_dose_scaling_uses_fake_part163_direction(self):
        rows = [{"layer": 21, "arm": "random", "dose": 1.0},
                {"layer": 21, "arm": "neuron", "dose": .8}]
        fake = FakeDense([[.8, -.4], [3.0, 4.0]])
        with mock.patch.object(run164.part163, "pushes_for", return_value={21: fake}):
            result = run164.pushes_for(rows, {}, {}, None)
        self.assertEqual(result[21][0], [.8 * 1.25, -.4 * 1.25])
        self.assertEqual(result[21][1], [3.0, 4.0])
        # Deliberate mutation guard: removing the dose factor must make this test red.
        self.assertNotEqual(result[21][0], [.8, -.4])

    def test_qwen_case_rule_and_excluded_alibaba_answer(self):
        self.assertTrue(score164.qwen_hit("Qwen"))
        self.assertFalse(score164.qwen_hit("qwen"))
        self.assertFalse(score164.qwen_eligible("I am Nova, a helpful system from Alibaba today."))
        self.assertFalse(score164.qwen_eligible("I am Nova, a helpful system from Tongyi today."))
        self.assertFalse(score164.qwen_eligible("I am Nova, a helpful system from 通义 today."))

    def test_each_gate_has_pass_and_fail_fixture(self):
        units, rows = answer_fixture()
        self.assertTrue(all(g["passed"] for g in score164.score_rows(rows, units)["gates"].values()))
        gate1 = rows[:-1]
        self.assertFalse(score164.score_rows(gate1, units)["gates"]["gate1_rows"]["passed"])
        _, bad_base = answer_fixture(baseline_good=False)
        self.assertFalse(score164.score_rows(bad_base, units)["gates"]["gate2_no_push"]["passed"])
        _, bad_recipe = answer_fixture(recipe_changed=False)
        self.assertFalse(score164.score_rows(bad_recipe, units)["gates"]["gate3_recipe"]["passed"])
        no_room = [dict(r) for r in rows]
        for row in no_room:
            if row["stage"] == "battery" and row["arm"] == "random" and row["dose"] == .8 and row["answer"].startswith("Qwen"):
                row["answer"] = "I am Nova, a helpful system that responds clearly."
        self.assertFalse(score164.score_rows(no_room, units)["gates"]["gate4_measure_room"]["passed"])

    def test_predictions_pass_and_fail_at_point_estimates(self):
        units, rows = answer_fixture()
        passed = score164.score_rows(rows, units)["predictions"]
        self.assertEqual(passed["164.1"]["verdict"], "held")
        self.assertEqual(passed["164.2"]["verdict"], "held")
        self.assertEqual(passed["164.3"]["verdict"], "held")
        self.assertEqual(passed["164.4"]["verdict"], "held")
        self.assertEqual(passed["164.5"]["verdict"], "held")

        units, fail_08 = answer_fixture(neuron_q=4, random_q={.8: 5, 1.0: 4, 1.2: 2},
                                        neuron_broken=4, random_broken={.8: 4, 1.0: 4, 1.2: 8})
        failed = score164.score_rows(fail_08, units)["predictions"]
        self.assertAlmostEqual(failed["164.1"]["estimate_ratio"], .8)
        self.assertEqual(failed["164.1"]["verdict"], "failed")
        self.assertEqual(failed["164.2"]["verdict"], "failed")
        self.assertEqual(failed["164.3"]["verdict"], "failed")
        self.assertEqual(failed["164.5"]["verdict"], "failed")

        # Broken-rate tie between +1.0 and +1.2 selects +1.0; its Qwen ratio is exactly 1.5.
        units, fail_1644 = answer_fixture(neuron_q=6, random_q={.8: 5, 1.0: 4, 1.2: 2},
                                          neuron_broken=8, random_broken={.8: 4, 1.0: 8, 1.2: 8})
        tie = score164.score_rows(fail_1644, units)["predictions"]["164.4"]
        self.assertEqual(tie["matched_random_dose"], 1.0)
        self.assertAlmostEqual(tie["estimate_ratio"], 1.5)
        self.assertEqual(tie["verdict"], "failed")

    def test_nearest_dose_tie_prefers_lower_dose(self):
        units, rows = answer_fixture(neuron_broken=5, random_broken={.8: 4, 1.0: 6, 1.2: 8})
        pred = score164.score_rows(rows, units)["predictions"]["164.4"]
        # Neuron .3125 is equally distant from random .25 (+.8) and .375 (+1.0).
        self.assertEqual(pred["matched_random_dose"], .8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
