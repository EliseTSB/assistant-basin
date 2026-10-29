"""Tests for Part 114's row-zeroing hook, controls and log-odds, without a model. Written by Claude (Opus 5.5), 2026-10-29."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part114"))


def test_row_zero_hits_only_each_rows_column():
    import torch
    import qneuron114 as q
    lin = torch.nn.Linear(6, 3, bias=False)
    z = q.RowZero(lin, [2, -1])
    x = torch.ones(2, 4, 6)
    seen = {}
    lin.register_forward_pre_hook(lambda m, args: seen.setdefault("x", args[0].clone()))
    lin(x)
    z.remove()
    assert torch.all(seen["x"][0, :, 2] == 0) and torch.all(seen["x"][1] == 1) and torch.all(x == 1)
    assert torch.count_nonzero(seen["x"][0] == 0) == 4


def test_controls_are_nearest_in_activity_and_exclude_target():
    import torch
    import qneuron114 as q
    mean_abs = {0: torch.tensor([0.10, 0.50, 0.11, 0.09, 0.30])}
    assert q.controls_for(0, 0, mean_abs, n=2) == [2, 3]


def test_logodds():
    import math
    import torch
    import qneuron114 as q
    logits = torch.tensor([math.log(3.0), 0.0])   # p(Q) = 0.75
    assert abs(float(q.q_logodds(logits, 0)) - math.log(3.0)) < 1e-5
