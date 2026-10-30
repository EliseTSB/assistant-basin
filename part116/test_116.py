"""Tests for Part 116's verdict rules and overlap helper. Written by Claude (Opus 5.5), 2026-10-29.
Deliberate break: make verdicts() treat kstar_act None as failed for 116.1; test_none_kstar_counts_as_larger fails."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score116 as s  # noqa: E402


def test_none_kstar_counts_as_larger():
    v = s.verdicts(None, 100, 10, -3.0, -6.96, 0)
    assert v == {"116.1": "held", "116.2": "held", "116.3": "held", "116.4": "held", "116.5": "held"}


def test_verdicts_fail_the_other_way():
    v = s.verdicts(100, 300, 120, -8.0, -6.96, 3)
    assert set(v.values()) == {"failed"}


def test_top_set_takes_most_negative_attr():
    pt = {"attr": torch.tensor([[0.0, -5.0, -1.0], [-3.0, 0.0, 0.0]])}
    assert s.top_set(pt, 2) == {(0, 1), (1, 0)}
