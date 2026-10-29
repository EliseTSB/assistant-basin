"""Tests for Part 112's rows and per-layer pushes, on fake layers (no model). Written by Claude (Opus 5.5), 2026-10-29."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part112"))


class _Lin:
    def __init__(self, w):
        self.weight = w


class _Mlp:
    def __init__(self, w):
        self.down_proj = _Lin(w)


class _Layer:
    def __init__(self, w):
        self.mlp = _Mlp(w)


def test_rows_pair_each_unit_with_a_random_row():
    import battery112 as b
    rows = b.rows_for([{"id": "q112-000", "layer": 3, "neuron": 1}, {"id": "q112-001", "layer": 5, "neuron": 2}])
    assert [(r["id"], r["unit"]) for r in rows] == [("q112-000", "neuron"), ("q112-000", "random"),
                                                   ("q112-001", "neuron"), ("q112-001", "random")]


def test_pushes_are_zero_outside_each_layer_and_write_the_target_norm():
    import torch
    import battery112 as b
    g = torch.Generator().manual_seed(0)
    layers = {3: _Layer(torch.randn(8, 6, generator=g)), 5: _Layer(torch.randn(8, 6, generator=g))}
    medians = {"3": 2.0, "5": 4.0}
    rows = b.rows_for([{"id": "a", "layer": 3, "neuron": 1}, {"id": "b", "layer": 5, "neuron": 2}])
    p = b.pushes_for(rows, layers, medians)
    assert sorted(p) == [3, 5]
    assert torch.count_nonzero(p[3][2:]) == 0 and torch.count_nonzero(p[5][:2]) == 0
    for layer_no, rows_idx in ((3, (0, 1)), (5, (2, 3))):
        w = layers[layer_no].mlp.down_proj.weight
        target = b.DOSE * medians[str(layer_no)]
        for i in rows_idx:
            assert abs(float(torch.linalg.vector_norm(w @ p[layer_no][i])) - target) < 1e-4
    assert torch.count_nonzero(p[3][0]) == 1 and p[3][0][1] > 0   # the neuron row pushes only its own unit
    assert torch.count_nonzero(p[3][1]) == 6                      # the random row is dense


def test_random_direction_is_seeded_by_layer_and_neuron():
    import torch
    import battery112 as b
    w = torch.randn(8, 6, generator=torch.Generator().manual_seed(1))
    layers = {3: _Layer(w)}
    one = b.pushes_for(b.rows_for([{"id": "a", "layer": 3, "neuron": 1}]), layers, {"3": 2.0})[3][1]
    two = b.pushes_for(b.rows_for([{"id": "a", "layer": 3, "neuron": 1}]), layers, {"3": 2.0})[3][1]
    other = b.pushes_for(b.rows_for([{"id": "a", "layer": 3, "neuron": 4}]), layers, {"3": 2.0})[3][1]
    assert torch.equal(one, two) and not torch.equal(one, other)
