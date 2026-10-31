#!/usr/bin/env python3
"""Fixture tests for Part 156's scorer. A deliberate break: flip 156.2's > to >= and the second test goes red."""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import score156 as s


def hits(curves):
    out = {}
    for u, c in curves.items():
        out[u] = {"neuron": dict(zip(s.DOSES, c)), "random": {d: 0 for d in s.DOSES}}
    return out


class Rules(unittest.TestCase):
    base = [0, 0, 12, 20, 12, 0, 0]
    def test_spread_and_contiguous(self):
        p = s.predictions(hits({u: self.base for u in s.UNITS}))
        self.assertEqual(p["156.1"]["verdict"], "held")
        self.assertEqual(p["156.4"]["verdict"], "held")
        self.assertEqual(p["156.3"]["verdict"], "held")
    def test_peak_elsewhere_counts_only_when_higher(self):
        p = s.predictions(hits({u: [0, 0, 20, 20, 0, 0, 0] for u in s.UNITS}))
        self.assertEqual(p["156.2"]["verdict"], "failed")
        p = s.predictions(hits({u: [0, 0, 21, 20, 0, 0, 0] for u in s.UNITS}))
        self.assertEqual(p["156.2"]["verdict"], "held")
    def test_gap_and_width(self):
        p = s.predictions(hits({u: [12, 0, 12, 20, 0, 0, 0] for u in s.UNITS}))
        self.assertEqual(p["156.4"]["verdict"], "failed")
        p = s.predictions(hits({u: [12, 12, 12, 12, 12, 0, 0] for u in s.UNITS}))
        self.assertEqual(p["156.4"]["verdict"], "failed")
    def test_random_quarter(self):
        h = hits({u: self.base for u in s.UNITS})
        h["L24N5017"]["random"][0.8] = 20
        self.assertEqual(s.predictions(h)["156.5"]["verdict"], "failed")
    def test_gate_failure_makes_untestable(self):
        r = s.score([])
        self.assertTrue(all(v["verdict"] == "untestable" for v in r["predictions"].values()))
    def test_gate1_rejects_wrong_row_key_even_with_1368_unique_rows(self):
        rows = []
        for u in s.UNITS:
            for arm in ("neuron", "random"):
                for dose in s.DOSES:
                    for q in range(s.PHRASINGS):
                        rows.append({"unit_id": u, "arm": arm, "dose": dose,
                                     "question_index": q, "answer": "valid answer text"})
        for q in range(s.PHRASINGS):
            rows.append({"unit_id": "plain", "arm": "plain", "dose": 0.0,
                         "question_index": q, "answer": "valid answer text"})
        rows[0]["question_index"] = 24
        result = s.score(rows)
        self.assertEqual(result["gates"]["gate1"]["rows"], 1368)
        self.assertFalse(result["gates"]["gate1"]["matches_expected_keys"])
        self.assertFalse(result["gates"]["gate1"]["57_rows_each_phrasing"])
        self.assertFalse(result["gates"]["gate1"]["passed"])


if __name__ == "__main__":
    unittest.main()
