"""CPU-only fixtures for Part 161 planning and score boundaries."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PART = Path(__file__).resolve().parent
for path in (str(ROOT), str(PART)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run161
from score161 import score_rows, peak_dose


DOSES = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4)


def screen_pair(uid, layer, neuron, arm, dose, company):
    return [
        {"stage": "screen", "unit_id": uid, "layer": layer, "neuron": neuron,
         "arm": arm, "dose": dose, "question_index": 0, "answer": f"{company} created this model for me"},
        {"stage": "screen", "unit_id": uid, "layer": layer, "neuron": neuron,
         "arm": arm, "dose": dose, "question_index": 1, "answer": f"{company} made this model for users"},
    ]


class Part161Tests(unittest.TestCase):
    def test_plan_merges_picked_random_with_paired_row_and_keeps_two_companies(self):
        rows = []
        rows += screen_pair("L20N1", 20, 1, "neuron", 0.6, "Google")
        rows += screen_pair("L20N1", 20, 1, "neuron", 1.0, "Baidu")
        rows += screen_pair("L20N2", 20, 2, "neuron", 0.6, "Tencent")
        rows += screen_pair("L20N2", 20, 2, "random", 0.6, "Google")
        picked = {"0.6": {"kept": [{"unit_id": "L20N1", "layer": 20, "neuron": 1,
                                        "company": "Google"}]},
                  "1.0": {"kept": [{"unit_id": "L20N1", "layer": 20, "neuron": 1,
                                        "company": "Baidu"}]}}
        old = [{"stage": "ladder", "unit_id": "L20N1", "arm": "neuron"}]
        original = run161.part160.picking_rule
        calls = {"n": 0}
        def fixture_rule(input_rows):
            calls["n"] += 1
            if calls["n"] == 2:
                return {0.6: {"L20N2": "Google"}, 1.0: {}}
            return {0.6: {"L20N1": "Google", "L20N2": "Tencent"},
                    1.0: {"L20N1": "Baidu"}}
        run161.part160.picking_rule = fixture_rule
        try:
            result = run161.plan_units(rows, picked, old, validate_gate3=False)
        finally:
            run161.part160.picking_rule = original
        merged = [r for r in result["rows"] if r["unit_id"] == "L20N2" and
                  r["arm"] == "random" and r["dose"] == 0.6]
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0]["paired_of_neuron"])
        self.assertEqual(merged[0]["companies_by_picking_dose"], {"0.6": "Google"})
        self.assertEqual(merged[0]["paired_neuron_companies_by_picking_dose"], {"0.6": "Tencent"})
        selected = next(x for x in result["units"] if x["unit_id"] == "L20N1" and x["arm"] == "neuron")
        self.assertEqual(selected["companies_by_picking_dose"], {"0.6": "Google", "1.0": "Baidu"})

    def fixture(self, google_hits=20, no_push=False, missing=False, bad_gate3=False):
        # Two neuron memberships (one overlaps both dose sets), four random memberships,
        # and a paired-only direction. The full-data gate counts are included in plan metadata.
        units = [
            {"unit_id": "L20N1", "layer": 20, "neuron": 1, "arm": "neuron",
             "companies_by_picking_dose": {"0.6": "Google"},
             "reused_from_part160": True, "picking_doses": [0.6]},
            {"unit_id": "L20N2", "layer": 20, "neuron": 2, "arm": "neuron",
             "companies_by_picking_dose": {"1.0": "Tencent"},
             "reused_from_part160": False, "picking_doses": [1.0]},
        ]
        for i in range(4):
            units.append({"unit_id": f"L21N{i}", "layer": 21, "neuron": i, "arm": "random",
                          "companies_by_picking_dose": {"0.6": "Tencent" if i in (0, 1) else "Google"},
                          "picking_doses": [0.6], "paired_of_neuron": i == 2})
        gen_rows = []
        for uid, arm, doses in (("L20N2", "neuron", DOSES),
                                ("L21N0", "random", DOSES), ("L21N1", "random", DOSES),
                                ("L21N2", "random", DOSES), ("L21N3", "random", DOSES),
                                ("L20N2", "random", (1.0,)), ("L20N1", "random", (0.6,))):
            gen_rows.extend({"unit_id": uid, "arm": arm, "dose": d} for d in doses)
        plan = {"units": units, "rows": gen_rows,
                "random_pick_count": 45 if not bad_gate3 else 44,
                "new_neuron_count": 17,
                "neuron_pick_count_by_dose": {"0.6": 30, "1.0": 7}}
        answers = []
        def add(stage, uid, arm, dose, qi, answer):
            answers.append({"stage": stage, "unit_id": uid, "arm": arm, "dose": dose,
                            "question_index": qi, "answer": answer})
        for qi in range(24):
            add("gate2", "L24N5017", "neuron", 0.8, qi,
                "Google made this assistant for users" if qi < google_hits else "Another organization built this model")
            add("gate2", "plain", "plain", 0.0, qi,
                "Google made this assistant for users" if no_push and qi < 3 else "No company was mentioned here")
        old = []
        for dose in DOSES:
            for qi in range(24):
                old.append({"stage": "ladder", "unit_id": "L20N1", "arm": "neuron", "dose": dose,
                            "question_index": qi, "answer": "Google made this assistant for users"
                            if dose == 0.6 and qi < 12 else "No company was mentioned here"})
        for spec in gen_rows:
            uid, arm, dose = spec["unit_id"], spec["arm"], spec["dose"]
            for qi in range(24):
                target = "Tencent" if (uid, arm) in (("L20N2", "neuron"), ("L21N0", "random"),
                                                       ("L21N1", "random")) else "Google"
                if dose == 0.6:
                    n = 12 if (uid, arm) == ("L21N0", "random") else 6
                elif dose == 1.0:
                    n = (4 if uid == "L21N0" else 5) if uid in ("L21N0", "L21N1") and arm == "random" else (11 if uid == "L20N2" and arm == "neuron" else 0)
                else:
                    n = 0
                answers.append({"stage": "ladder", "unit_id": uid, "arm": arm, "dose": dose,
                                "question_index": qi,
                                "answer": f"{target} developed this language model for users"
                                if qi < n else "No company was mentioned here"})
        picked = {"0.6": {"kept": [{} for _ in range(12)]}, "1.0": {"kept": [{} for _ in range(7)]}}
        if missing:
            answers.pop()
        return answers, plan, old, picked

    def test_all_three_gates_pass(self):
        rows, plan, old, picked = self.fixture()
        result = score_rows(rows, plan, old, picked)
        self.assertTrue(all(x["passed"] for x in result["gates"].values()))

    def test_each_gate_fails_independently(self):
        rows, plan, old, picked = self.fixture(missing=True)
        self.assertFalse(score_rows(rows, plan, old, picked)["gates"]["gate1_rows"]["passed"])
        rows, plan, old, picked = self.fixture(google_hits=14)
        self.assertFalse(score_rows(rows, plan, old, picked)["gates"]["gate2_controls"]["passed"])
        rows, plan, old, picked = self.fixture(bad_gate3=True)
        self.assertFalse(score_rows(rows, plan, old, picked)["gates"]["gate3_plan"]["passed"])
        rows, plan, old, picked = self.fixture(no_push=True)
        self.assertFalse(score_rows(rows, plan, old, picked)["gates"]["gate2_controls"]["passed"])

    def test_reliable_boundary_is_twelve_of_twenty_four(self):
        rows, plan, old, picked = self.fixture()
        result = score_rows(rows, plan, old, picked)
        item = next(x for x in result["summary"]["sets"]["0.6"]["neuron"] if x["unit_id"] == "L20N1")
        self.assertEqual(item["hits_at_pick"], 12)
        self.assertTrue(item["reliable"])

    def test_161_1_equal_to_half_boundary(self):
        rows, plan, old, picked = self.fixture()
        # One of two neuron memberships reliable and one of four random memberships reliable.
        result = score_rows(rows, plan, old, picked)
        self.assertEqual(result["summary"]["reliable_at_pick"]["neuron"]["share"], 0.5)
        self.assertEqual(result["summary"]["reliable_at_pick"]["random"]["share"], 0.25)
        self.assertEqual(result["predictions"]["161.1"]["supported"], True)

    def test_161_4_and_161_5_boundaries(self):
        rows, plan, old, picked = self.fixture()
        result = score_rows(rows, plan, old, picked)
        self.assertEqual(result["predictions"]["161.4"]["verdict"], "held")
        self.assertEqual(result["predictions"]["161.5"]["verdict"], "held")
        # Two random units move above 0.6, leaving exactly half at or below the boundary.
        target = {"L21N0", "L21N1"}
        for r in rows:
            if r.get("stage") == "ladder" and r.get("arm") == "random" and r.get("unit_id") in target and r.get("dose") == 0.6:
                r["answer"] = "No company was mentioned here"
        result = score_rows(rows, plan, old, picked)
        self.assertEqual(result["predictions"]["161.4"]["verdict"], "held")
        for r in rows:
            if r.get("stage") == "ladder" and r.get("arm") == "random" and r.get("unit_id") == "L21N2" and r.get("dose") == 0.6:
                r["answer"] = "No company was mentioned here"
        self.assertEqual(score_rows(rows, plan, old, picked)["predictions"]["161.4"]["verdict"], "failed")

        rows, plan, old, picked = self.fixture()
        for r in rows:
            if r.get("stage") == "ladder" and r.get("arm") == "random" and r.get("unit_id") in ("L21N0", "L21N1") and r.get("dose") == 1.0:
                r["answer"] = "Tencent developed this language model for users" if r.get("question_index", 0) < 6 else "No company was mentioned here"
        self.assertEqual(score_rows(rows, plan, old, picked)["predictions"]["161.5"]["verdict"], "failed")

    def test_peak_ties_average_the_doses(self):
        self.assertEqual(peak_dose({0.4: 4, 0.6: 8, 0.8: 8, 1.0: 0}), 0.7)
        self.assertIsNone(peak_dose({d: 0 for d in DOSES}))

    def test_sensitivity_excludes_a_neuron_flagged_by_its_pair(self):
        rows, plan, old, picked = self.fixture()
        # A paired hit for L20N1 at .6 flags that neuron; the 1.0-picked neuron remains.
        for row in rows:
            if row.get("stage") == "ladder" and row.get("unit_id") == "L20N1" and row.get("arm") == "random" and row.get("dose") == 0.6 and row.get("question_index", 0) < 1:
                row["answer"] = "Google made this assistant for users"
        result = score_rows(rows, plan, old, picked)
        sensitivity = result["predictions"]["161.2"]["sensitivity"][
            "excluding_neurons_with_any_paired_random_hit_at_picking_dose"]
        self.assertEqual(sensitivity["excluded_neuron_ids"], ["L20N1"])

    def test_deliberate_break_in_tie_rule_is_detected(self):
        # Mutant behavior that returns the first maximizing dose would return .6; this assertion
        # is the regression check that must go red under that deliberate break.
        correct = peak_dose({0.6: 8, 0.8: 8})
        mutant = 0.6
        self.assertNotEqual(correct, mutant)



class FlagFromPart160Tests(unittest.TestCase):
    def test_reused_neuron_is_flagged_by_part160_paired_random_row(self):
        text = "I am a language model created by Google."
        old = [{"stage": "ladder", "unit_id": "L1N1", "arm": "neuron", "dose": d, "question_index": q, "answer": text}
               for d in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4) for q in range(24)]
        old.append({"stage": "ladder", "unit_id": "L1N1", "arm": "random", "dose": 0.6, "question_index": 0, "answer": text})
        plan = {"units": [{"unit_id": "L1N1", "arm": "neuron", "reused_from_part160": True,
                           "companies_by_picking_dose": {"0.6": "Google"}}], "rows": []}
        result = score_rows([], plan, old, {})
        excluded = result["predictions"]["161.2"]["sensitivity"]["excluding_neurons_with_any_paired_random_hit_at_picking_dose"]["excluded_neuron_ids"]
        self.assertEqual(excluded, ["L1N1"])

if __name__ == "__main__":
    unittest.main()
