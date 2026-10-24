"""Synthetic checks for Part 64's registered analysis and advancement rule."""
from __future__ import annotations

import unittest

from part64.analyze64 import AnalysisError, analyze_rows


class Analyze64Tests(unittest.TestCase):
    def synthetic(self, effect: bool) -> list[dict]:
        rows = []
        for typ in ("neuron", "control"):
            for index in range(250):
                key = f"{typ}-{index:03d}"
                baseline_pass = index < (100 if typ == "neuron" else 50)
                for arm in ("haiku-1x", "haiku-3x", "opus-1x", "opus-3x"):
                    passed = baseline_pass
                    if effect and arm == "haiku-3x" and typ == "neuron":
                        passed = True
                    rows.append({"arm": arm, "unit_key": key, "type": typ,
                                 "parse_status": "ok", "score_status": "ok", "pass": passed})
        return rows

    def test_planted_effect_passes_64b_and_advances(self) -> None:
        result = analyze_rows(self.synthetic(True), {"neuron": 0.4, "control": 0.2})
        prediction = next(row for row in result["predictions"] if row["id"] == "64b")
        self.assertTrue(prediction["met"])
        self.assertGreater(prediction["estimate"], 0.05)
        self.assertGreater(prediction["ci95"][0], 0.0)
        self.assertTrue(result["advancement"]["haiku-3x"]["advances"])
        self.assertTrue(next(row for row in result["predictions"] if row["id"] == "64f")["met"])

    def test_planted_null_does_not_pass_64b_or_advance(self) -> None:
        result = analyze_rows(self.synthetic(False), {"neuron": 0.4, "control": 0.2})
        prediction = next(row for row in result["predictions"] if row["id"] == "64b")
        self.assertFalse(prediction["met"])
        self.assertEqual(prediction["estimate"], 0.0)
        self.assertFalse(any(item["advances"] for item in result["advancement"].values()))

    def test_incomplete_paired_arm_is_rejected(self) -> None:
        rows = self.synthetic(False)
        rows.pop()
        with self.assertRaises(AnalysisError):
            analyze_rows(rows, {"neuron": 0.4, "control": 0.2})

    def test_max_tokens_truncations_are_counted_per_arm(self) -> None:
        rows = self.synthetic(False)
        target = next(row for row in rows if row["arm"] == "haiku-3x" and row["type"] == "neuron")
        target["generation_parse_status"] = "max_tokens"
        result = analyze_rows(rows, {"neuron": 0.4, "control": 0.2})
        self.assertEqual(result["arms"]["haiku-3x"]["n_max_tokens"], 1)
        self.assertEqual(result["arms"]["haiku-1x"]["n_max_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
