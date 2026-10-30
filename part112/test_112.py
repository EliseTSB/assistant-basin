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


def test_kind_rule_matches_part92():
    import score112 as s
    T, F = True, False
    assert s.kind_of([F, T, T, T, T, T]) == "dropped"
    assert s.kind_of([T, T, T, F, F, F]) == "concept"
    assert s.kind_of([T, F, F, F, F, F]) == "slot" and s.kind_of([T, T, F, F, F, F]) == "slot"
    assert s.kind_of([T, F, T, F, F, F]) == "unclassified"


def test_score_groups_and_verdicts(tmp_path):
    import json
    import score112 as s
    units = [{"id": "a", "layer": 1, "neuron": 1, "persona": "Qumin", "kind": "another AI", "q_name": True},
             {"id": "b", "layer": 1, "neuron": 2, "persona": "progress bar", "kind": "object or place", "q_name": False}]
    shows = {"a": [1, 0, 0, 0, 0, 0], "b": [1, 1, 1, 0, 0, 0]}
    labels = [{"id": u, "unit": unit, "question_index": q, "shows_persona": bool(shows[u][q]) if unit == "neuron" else False}
              for u in shows for unit in ("neuron", "random") for q in range(6)]
    lp, up = tmp_path / "l.jsonl", tmp_path / "u.json"
    lp.write_text("\n".join(json.dumps(r) for r in labels) + "\n", encoding="utf-8")
    up.write_text(json.dumps(units), encoding="utf-8")
    r = s.score(lp, up, tmp_path / "s.json")
    assert r["units"]["a"]["kind"] == "slot" and r["units"]["b"]["kind"] == "concept"
    assert r["verdicts"] == {"112.1": True, "112.2": True, "112.3": True, "112.4": True}



def test_missing_labels_follow_amendment_1():
    import score112 as s
    T, F, N = True, False, None
    assert s.kind_with_missing([T, N, F, F, F, F]) == "slot"          # slot either way
    assert s.kind_with_missing([T, F, N, T, F, F]) == "unsorted"      # concept if shown, unclassified if not
    assert s.kind_with_missing([T, T, N, T, F, F]) == "concept"       # concept either way
    assert s.kind_with_missing([F, N, N, F, F, F]) == "dropped"
    assert s.kind_with_missing([T, F, F, F, F, F]) == "slot"          # no gaps: Part 92's rule unchanged


def test_score_leaves_missing_labels_out_of_the_counts(tmp_path):
    import json
    import score112 as s
    units = [{"id": "a", "layer": 1, "neuron": 1, "persona": "Qumin", "kind": "another AI", "q_name": True},
             {"id": "b", "layer": 1, "neuron": 2, "persona": "progress bar", "kind": "object or place", "q_name": False}]
    labels = []
    for u in ("a", "b"):
        for unit in ("neuron", "random"):
            for q in range(6):
                if (u, unit, q) in {("a", "neuron", 2), ("b", "random", 3)}:
                    labels.append({"id": u, "unit": unit, "question_index": q, "error": "no JSON"})
                    continue
                labels.append({"id": u, "unit": unit, "question_index": q,
                               "shows_persona": unit == "neuron" and q == 0})
    lp, up = tmp_path / "l.jsonl", tmp_path / "u.json"
    lp.write_text("\n".join(json.dumps(r) for r in labels) + "\n", encoding="utf-8")
    up.write_text(json.dumps(units), encoding="utf-8")
    r = s.score(lp, up, tmp_path / "s.json")
    assert r["units"]["a"]["kind"] == "unsorted" and r["units"]["a"]["missing"] == [2]
    assert r["units"]["b"]["kind"] == "slot"
    assert r["random_non_self_labels"] == 7 and r["random_non_self_missing"] == 1
    assert r["unsorted_units"] == ["a"]
