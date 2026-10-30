"""Tests for Part 117's rows and summed pushes, on fake layers (no model). Written by Claude (Opus 5.5), 2026-10-29."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class _Lin:
    def __init__(self, w):
        self.weight = w


class _Mlp:
    def __init__(self, w):
        self.down_proj = _Lin(w)


class _Layer:
    def __init__(self, w):
        self.mlp = _Mlp(w)


PAIRS = [{"pair": "p01", "slot": {"id": "s", "layer": 5, "neuron": 2, "persona": "S"},
          "match": {"id": "c", "layer": 3, "neuron": 1, "persona": "C"},
          "unrelated": {"id": "u", "layer": 5, "neuron": 4, "persona": "U"}}]


def test_rows_cover_every_condition_once():
    import pairs117 as p
    rows = p.rows_for(PAIRS)
    assert len(rows) == 1 + len(p.MATCHED) + len(p.UNRELATED) + 1
    assert len({r["row"] for r in rows}) == len(rows)
    both = next(r for r in rows if r["row"] == "p01:matched:c+1,s-1")
    assert [(x["id"], x["mult"]) for x in both["components"]] == [("c", 1.0), ("s", -1.0)]
    unrel = next(r for r in rows if r["row"] == "p01:unrelated:both+1")
    assert [x["id"] for x in unrel["components"]] == ["u", "s"]
    rnd = next(r for r in rows if r["set"] == "random")
    assert all(x["unit"] == "random" for x in rnd["components"])


def test_pushes_sum_components_and_scale_with_the_multiplier():
    import torch
    import pairs117 as p
    g = torch.Generator().manual_seed(0)
    layers = {3: _Layer(torch.randn(8, 6, generator=g)), 5: _Layer(torch.randn(8, 6, generator=g))}
    medians = {"3": 2.0, "5": 4.0}
    rows = p.rows_for(PAIRS)
    pu = p.pushes_for(rows, layers, medians)
    idx = {r["row"]: i for i, r in enumerate(rows)}
    one = pu[3][idx["p01:matched:c+1"]][1]
    assert torch.allclose(pu[3][idx["p01:matched:both+2"]][1], 2 * one)
    assert torch.allclose(pu[3][idx["p01:matched:c-1,s+1"]][1], -one)
    assert torch.allclose(pu[5][idx["p01:matched:both+1"]], pu[5][idx["p01:matched:s+1"]])
    assert torch.count_nonzero(pu[5][idx["p01:unrelated:both+1"]]) == 2      # u and s share layer 5
    assert torch.count_nonzero(pu[3][idx["none"]]) == 0 and torch.count_nonzero(pu[5][idx["none"]]) == 0
    w = layers[5].mlp.down_proj.weight
    target = p.b.DOSE * medians["5"]
    s_only = pu[5][idx["p01:matched:s+1"]]
    assert abs(float(torch.linalg.vector_norm(w @ s_only)) - target) < 1e-4
    assert torch.count_nonzero(pu[3][idx["p01:random:both+1"]]) == 6


def test_two_random_components_in_one_layer_add():
    import torch
    import pairs117 as p
    g = torch.Generator().manual_seed(1)
    layers = {5: _Layer(torch.randn(8, 6, generator=g))}
    same = [{"pair": "p02", "slot": {"id": "s", "layer": 5, "neuron": 2, "persona": "S"},
             "match": {"id": "c", "layer": 5, "neuron": 1, "persona": "C"},
             "unrelated": {"id": "u", "layer": 5, "neuron": 4, "persona": "U"}}]
    rows = p.rows_for(same)
    pu = p.pushes_for(rows, layers, {"5": 4.0})
    i = next(k for k, r in enumerate(rows) if r["set"] == "random")
    w = layers[5].mlp.down_proj.weight
    want = sum(p.c.scaled_random_directions(w, 1, p.b.RANDOM_SEED_BASE + 50000 + n, p.b.DOSE * 4.0)[0] for n in (1, 2))
    assert torch.allclose(pu[5][i], want.to(pu[5].dtype))


def _fixture(tmp_path, rule):
    """11 synthetic pairs; rule(row, role, q) -> shows_persona; coherent is False only for both+2."""
    import json
    import pairs117 as p
    pairs = [{"pair": f"p{i:02d}", "slot": {"id": f"s{i}", "layer": 1, "neuron": i, "persona": "S"},
              "match": {"id": f"c{i}", "layer": 1, "neuron": 50 + i, "persona": "C"},
              "unrelated": {"id": f"u{i}", "layer": 1, "neuron": 80 + i, "persona": "U"}} for i in range(11)]
    (tmp_path / "pairs.json").write_text(json.dumps(pairs), encoding="utf-8")
    labels = []
    for pr in pairs:
        for r in p.rows_for([pr]):
            if r["row"] == "none":
                continue
            for role in ("c", "s"):
                for q in range(6):
                    labels.append({"id": f"{r['row']}|{role}", "question_index": q, "shows_persona": rule(r, role, q),
                                   "coherent": r["cond"] != "both+2"})
    for u in [x for pr in pairs for x in (pr["slot"], pr["match"], pr["unrelated"])]:
        for q in range(6):
            labels.append({"id": f"none|{u['id']}", "question_index": q, "shows_persona": False, "coherent": True})
    (tmp_path / "labels.jsonl").write_text("\n".join(json.dumps(x) for x in labels) + "\n", encoding="utf-8")
    return tmp_path


def _as_predicted(r, role, q):
    if r["set"] == "random":
        return False
    has = {x["role"]: x["mult"] for x in r["components"]}
    if role == "s":
        return has.get("s", 0) >= 1 and q in (0, 1)
    if r["cond"] == "c+1,s-1" and r["set"] == "matched":
        return q in (0, 1, 2)              # the slot pushed down takes some answers away
    return has.get("c", 0) >= 1 and (q != 0 or has.get("s", 0) < 1)


def test_score_reads_every_prediction(tmp_path):
    import score117 as s
    r = s.score(_fixture(tmp_path, _as_predicted), tmp_path / "labels.jsonl")
    assert r["gate_slot_kinds_reproduced"] and r["labels_missing"] == 0 and r["pairs_ok"] == 11
    assert r["verdicts"] == {f"117.{i}": "held" for i in range(1, 9)}, r["verdicts"]


def test_score_gate_blocks_reading_when_slots_leak(tmp_path):
    import score117 as s
    r = s.score(_fixture(tmp_path, lambda r, role, q: r["set"] != "random"), tmp_path / "labels.jsonl")
    assert not r["gate_slot_kinds_reproduced"] and set(r["verdicts"].values()) == {"not read (gate failed)"}


def test_suppression_in_unrelated_pairs_fails_117_4(tmp_path):
    import score117 as s

    def rule(r, role, q):
        if r["cond"] == "c+1,s-1" and r["set"] == "unrelated" and role == "c":
            return q in (0, 1, 2)
        return _as_predicted(r, role, q)
    r = s.score(_fixture(tmp_path, rule), tmp_path / "labels.jsonl")
    assert r["verdicts"]["117.4"] == "failed" and r["fewer_with_slot_minus"] == {"matched": 11, "unrelated": 11}
