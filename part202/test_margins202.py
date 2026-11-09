#!/usr/bin/env python3
"""Small tests for Part 202 published-margin arithmetic; runnable with pytest or directly."""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import margins202


class Margins202Tests(unittest.TestCase):
    def test_rate_gap_is_reported_as_equivalent_rows(self):
        self.assertAlmostEqual(margins202.eq_rows(0.05), 80.0)

    def test_cap_count_requires_exactly_eighty(self):
        rows = [
            {"stage": "battery", "arm": "neuron", "token_count": 80},
            {"stage": "battery", "arm": "neuron", "token_count": 79},
            {"stage": "recipe", "arm": "neuron", "token_count": 80},
        ]
        got = margins202.cap_summary(rows)["neuron"]
        self.assertEqual(got["answers"], 2)
        self.assertEqual(got["count_at_80"], 1)
        self.assertEqual(got["mean_token_count"], 79.5)


if __name__ == "__main__":
    unittest.main()
