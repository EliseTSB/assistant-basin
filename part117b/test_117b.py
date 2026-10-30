"""Tests for Part 117b's rows, pushes and scorer, on fake layers and synthetic labels (no model).
Written by Claude (Opus 5.5), 2026-10-29."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "part117"))


class _Lin:
    def __init__(self, w):
        self.weight = w


class _Mlp:
    def __init__(self, w):
        self.down_proj = _Lin(w)


class _Layer:
    def __init__(self, w):
        self.mlp = _Mlp(w)


def _unit(uid, layer, neuron, persona):
    return {"id": uid, "layer": layer, "neuron": neuron, "persona": persona}


# Two pairs that share their matched concept neuron, as Part 117's p07/p09 and p03/p11 do.
PAIRS = [{"pair": "p01", "slot": _unit("s1", 5, 2, "S1"), "match": _unit("c", 3, 1, "C"), "unrelated": _unit("u1", 5, 4, "U1")},
         {"pair": "p02", "slot": _unit("s2", 5, 3, "S2"), "match": _unit("c", 3, 1, "C"), "unrelated": _unit("u2", 3, 5, "U2")}]


CONTROLS = {"p01": {"layer": 3, "neurons": [0, 2, 4]}, "p02": {"layer": 3, "neurons": [5, 4, 3]}}


def test_rows_cover_every_condition_once_and_concepts_alone_once():
    import rows117b as rb
    rows = rb.rows_for(PAIRS, CONTROLS)
    per_pair = len(rb.PAIR_CONDS) + rb.N_RANDOM + 3
    assert len(rows) == 1 + 2 * per_pair + 3          # concepts alone: c, u1, u2 (c shared by both pairs)
    assert len(rb.rows_for(PAIRS)) == len(rows) - 6   # no control rows without controls
    ctl = [r for r in rows if r["set"] == "control" and r["pair"] == "p02"]
    assert [(x["components"][0]["layer"], x["components"][0]["neuron"], x["components"][0]["mult"]) for x in ctl] == \
        [(3, 5, -1.0), (3, 4, -1.0), (3, 3, -1.0)]
    assert all(x["components"][0]["persona"] is None and x["components"][1]["id"] == "s2" for x in ctl)
    assert len({r["row"] for r in rows}) == len(rows)
    alone = [r for r in rows if r["set"] == "alone"]
    assert sorted(r["components"][0]["id"] for r in alone) == ["c", "u1", "u2"]
    assert all(r["components"][0]["mult"] == -1.0 for r in alone)
    rnd = [r for r in rows if r["set"] == "random" and r["pair"] == "p01"]
    assert [x["components"][0]["rand_index"] for x in rnd] == [0, 1, 2]
    assert all(x["components"][0]["id"] == "c" and x["components"][0]["mult"] == -1.0 for x in rnd)
    assert all(x["components"][1]["id"] == "s1" and x["components"][1]["mult"] == 1.0 for x in rnd)
    u = next(r for r in rows if r["row"] == "p02:u-1,s+1")
    assert [(x["id"], x["mult"]) for x in u["components"]] == [("u2", -1.0), ("s2", 1.0)]


def test_pushes_match_part117_for_the_shared_conditions():
    import torch
    import pairs117 as p117
    import rows117b as rb
    g = torch.Generator().manual_seed(0)
    layers = {3: _Layer(torch.randn(8, 6, generator=g)), 5: _Layer(torch.randn(8, 6, generator=g))}
    medians = {"3": 2.0, "5": 4.0}
    old_rows = p117.rows_for([PAIRS[0]])
    new_rows = rb.rows_for([PAIRS[0]])
    old = p117.pushes_for(old_rows, layers, medians)
    new = rb.pushes_for(new_rows, layers, medians)
    oi = {r["row"]: i for i, r in enumerate(old_rows)}
    ni = {r["row"]: i for i, r in enumerate(new_rows)}
    for ln in (3, 5):
        assert torch.equal(new[ln][ni["p01:c-1,s+1"]], old[ln][oi["p01:matched:c-1,s+1"]])
        assert torch.equal(new[ln][ni["p01:s+1"]], old[ln][oi["p01:matched:s+1"]])
    assert torch.allclose(new[3][ni["p01:c-0.5,s+1"]], 0.5 * new[3][ni["p01:c-1,s+1"]])
    assert torch.allclose(new[5][ni["p01:s+0.5"]], 0.5 * new[5][ni["p01:s+1"]])


def test_random_index_0_is_part112s_direction_and_the_others_differ_at_the_same_size():
    import torch
    import rows117b as rb
    g = torch.Generator().manual_seed(1)
    layers = {3: _Layer(torch.randn(8, 6, generator=g)), 5: _Layer(torch.randn(8, 6, generator=g))}
    medians = {"3": 2.0, "5": 4.0}
    rows = rb.rows_for([PAIRS[0]], CONTROLS)
    pu = rb.pushes_for(rows, layers, medians)
    idx = {r["row"]: i for i, r in enumerate(rows)}
    w = layers[3].mlp.down_proj.weight
    ctl = pu[3][idx["p01:n1-1,s+1"]]                  # control neuron 2 in layer 3, at -1, the concept's push size
    assert torch.count_nonzero(ctl) == 1 and float(ctl[2]) == -rb.c.neuron_delta(w, 2, 2.0, dose=rb.b.DOSE)
    assert abs(float(torch.linalg.vector_norm(w @ ctl)) - rb.b.DOSE * 2.0) < 1e-4
    part112 = rb.c.scaled_random_directions(w, 1, rb.b.RANDOM_SEED_BASE + 10000 * 3 + 1, rb.b.DOSE * 2.0)[0]
    r0, r1, r2 = (pu[3][idx[f"p01:r{k}-1,s+1"]] for k in range(3))
    assert torch.allclose(r0, -part112)
    assert not torch.allclose(r0, r1) and not torch.allclose(r1, r2)
    for r in (r0, r1, r2):     # each writes the concept's push size: 0.8 x the layer median
        assert abs(float(torch.linalg.vector_norm(w @ r)) - rb.b.DOSE * 2.0) < 1e-4
    assert torch.equal(pu[5][idx["p01:r1-1,s+1"]], pu[5][idx["p01:s+1"]])   # the slot part is unchanged


def _fixture(tmp_path, rule, qwen_rule):
    """11 synthetic pairs. rule(row, role, q) -> shows_persona; qwen_rule(row) -> whether the Who answer names Qwen."""
    import rows117b as rb
    import score117b as s
    pairs = [{"pair": f"p{i:02d}", "slot": _unit(f"s{i}", 1, i, "S"), "match": _unit(f"c{i}", 1, 50 + i, "C"),
              "unrelated": _unit(f"u{i}", 1, 80 + i, "U")} for i in range(11)]
    controls = {p["pair"]: {"layer": 1, "neurons": [200 + 3 * i, 201 + 3 * i, 202 + 3 * i]} for i, p in enumerate(pairs)}
    (tmp_path / "control-neurons.json").write_text(json.dumps(controls), encoding="utf-8")
    rows = rb.rows_for(pairs, controls)
    units = s.all_units(pairs)
    labels, answers = [], []
    for r in rows:
        for q in range(6):
            answers.append({"row": r["row"], "pair": r["pair"], "set": r["set"], "cond": r["cond"], "question_index": q,
                            "question": "Who are you?" if q == 0 else "x",
                            "answer": ("I am Qwen." if qwen_rule(r) else "I am Zed.") if q == 0 else "fine"})
            for iid, _ in s.targets(r, units):
                role = iid.split("|")[1]
                labels.append({"id": iid, "question_index": q,
                               "shows_persona": False if r["row"] == "none" else rule(r, role, q), "coherent": True})
    (tmp_path / "answers.jsonl").write_text("\n".join(json.dumps(a) for a in answers) + "\n", encoding="utf-8")
    (tmp_path / "zero-hook-check.json").write_text(json.dumps([{"question": "q", "equal": True}] * 6), encoding="utf-8")
    (tmp_path / "agreement.json").write_text(json.dumps({"rate": 0.95}), encoding="utf-8")
    (tmp_path / "labels.jsonl").write_text("\n".join(json.dumps(x) for x in labels) + "\n", encoding="utf-8")
    return pairs


def _concept_specific(r, role, q):
    """The predicted world: the slot shows on the self questions alone and beside a random push, not beside a concept -1."""
    if role != "s" or q != 0:
        return False
    if r["cond"] in ("s+1", "s+0.85", "s+0.5", "c-0.5,s+1") or r["set"] in ("random", "control"):
        return True
    return False


def _qwen_predicted(r):
    return r["row"] == "none"


def test_score_reads_every_prediction_held(tmp_path):
    import score117b as s
    pairs = _fixture(tmp_path, _concept_specific, _qwen_predicted)
    r = s.score(tmp_path, tmp_path / "labels.jsonl", pairs, part117_answers=tmp_path / "missing.jsonl")
    assert r["gate"]["passed"] and r["labels_missing"] == 0 and r["pairs_ok"] == [11, 11, 11]
    assert r["verdicts"] == {f"117b.{i}": "held" for i in range(1, 11)}, r["verdicts"]
    assert r["reading"].startswith("concept neurons erase")
    assert r["who_slot_shows"]["random"] == [33, 0, 33] and r["who_slot_shows"]["control"] == [33, 0, 33]


def test_control_neurons_erasing_reads_as_any_real_neuron_and_fails_117b_9(tmp_path):
    import score117b as s

    def rule(r, role, q):
        return False if r["set"] == "control" else _concept_specific(r, role, q)
    pairs = _fixture(tmp_path, rule, _qwen_predicted)
    r = s.score(tmp_path, tmp_path / "labels.jsonl", pairs, part117_answers=tmp_path / "missing.jsonl")
    assert r["verdicts"]["117b.9"] == "failed" and r["verdicts"]["117b.3"] == "held"
    assert r["reading"].startswith("a second real neuron's push erases")


def test_random_erasing_reads_as_any_push_and_fails_117b_3(tmp_path):
    import score117b as s

    def rule(r, role, q):
        return False if r["set"] == "random" else _concept_specific(r, role, q)
    pairs = _fixture(tmp_path, rule, _qwen_predicted)
    r = s.score(tmp_path, tmp_path / "labels.jsonl", pairs, part117_answers=tmp_path / "missing.jsonl")
    assert r["verdicts"]["117b.3"] == "failed" and r["reading"].startswith("any second push")


def test_unrelated_keeping_the_slot_reads_as_theme(tmp_path):
    import score117b as s

    def rule(r, role, q):
        return (role == "s" and q == 0) if r["cond"] == "u-1,s+1" else _concept_specific(r, role, q)
    pairs = _fixture(tmp_path, rule, _qwen_predicted)
    r = s.score(tmp_path, tmp_path / "labels.jsonl", pairs, part117_answers=tmp_path / "missing.jsonl")
    assert r["verdicts"]["117b.2"] == "failed" and r["reading"].startswith("the matched concept erases it and the unrelated")


def test_gate_blocks_reading_when_the_slot_alone_does_not_show(tmp_path):
    import score117b as s

    def rule(r, role, q):
        return False if r["cond"] == "s+1" else _concept_specific(r, role, q)
    pairs = _fixture(tmp_path, rule, _qwen_predicted)
    r = s.score(tmp_path, tmp_path / "labels.jsonl", pairs, part117_answers=tmp_path / "missing.jsonl")
    assert not r["gate"]["passed"] and set(r["verdicts"].values()) == {"not read (gate failed)"}
    assert r["reading"] == "not read (gate failed)"


def test_random_erasing_reports_dilution_from_the_085_slot_rows(tmp_path):
    import score117b as s

    def rule(r, role, q):
        if r["set"] == "random" or r["cond"] == "s+0.85":
            return False
        return _concept_specific(r, role, q)
    pairs = _fixture(tmp_path, rule, _qwen_predicted)
    r = s.score(tmp_path, tmp_path / "labels.jsonl", pairs, part117_answers=tmp_path / "missing.jsonl")
    assert r["dilution"] == "consistent with the slot's push being diluted" and r["verdicts"]["117b.10"] == "failed"
    assert r["reading"].endswith("(consistent with the slot's push being diluted)")


def test_incoherent_random_rows_are_not_read_as_erasure(tmp_path):
    import score117b as s

    def rule(r, role, q):
        return False if r["set"] == "random" else _concept_specific(r, role, q)
    pairs = _fixture(tmp_path, rule, _qwen_predicted)
    lab = [json.loads(l) for l in open(tmp_path / "labels.jsonl", encoding="utf-8")]
    for x in lab:
        if ":r" in x["id"]:
            x["coherent"] = False
    (tmp_path / "labels.jsonl").write_text("\n".join(json.dumps(x) for x in lab) + "\n", encoding="utf-8")
    r = s.score(tmp_path, tmp_path / "labels.jsonl", pairs, part117_answers=tmp_path / "missing.jsonl")
    assert r["reading"] == "the random push breaks the answers; not read as erasure" and r["verdicts"]["117b.5"] == "failed"


def test_concept_alone_rows_naming_qwen_fail_117b_4(tmp_path):
    import score117b as s
    pairs = _fixture(tmp_path, _concept_specific, lambda r: r["row"] == "none" or r["set"] == "alone")
    r = s.score(tmp_path, tmp_path / "labels.jsonl", pairs, part117_answers=tmp_path / "missing.jsonl")
    assert r["verdicts"]["117b.4"] == "failed" and r["who_names_qwen"]["alone_c-1"] == [22, 0, 22]


def test_zero_hook_mismatch_or_absence_fails_the_gate(tmp_path):
    import score117b as s
    pairs = _fixture(tmp_path, _concept_specific, _qwen_predicted)
    (tmp_path / "zero-hook-check.json").write_text(json.dumps([{"equal": True}] * 5 + [{"equal": False}]), encoding="utf-8")
    r = s.score(tmp_path, tmp_path / "labels.jsonl", pairs, part117_answers=tmp_path / "missing.jsonl")
    assert not r["gate"]["passed"] and r["reading"] == "not read (gate failed)"
    (tmp_path / "zero-hook-check.json").unlink()
    r = s.score(tmp_path, tmp_path / "labels.jsonl", pairs, part117_answers=tmp_path / "missing.jsonl")
    assert not r["gate"]["zero_hook_equal"] and not r["gate"]["passed"]


def test_absent_agreement_withholds_and_missing_baselines_make_a_unit_uninterpretable(tmp_path):
    import score117b as s
    pairs = _fixture(tmp_path, _concept_specific, _qwen_predicted)
    (tmp_path / "agreement.json").unlink()
    kept = [l for l in open(tmp_path / "labels.jsonl", encoding="utf-8")
            if not ('"none|u3"' in l and ('"question_index": 0' in l or '"question_index": 1' in l or '"question_index": 2' in l))]
    (tmp_path / "labels.jsonl").write_text("".join(kept), encoding="utf-8")
    r = s.score(tmp_path, tmp_path / "labels.jsonl", pairs, part117_answers=tmp_path / "missing.jsonl")
    assert r["reading"] == "withheld (no second-reader agreement)"
    assert r["uninterpretable_units"] == ["u3"] and r["pairs_ok"] == [11, 10, 11]


def test_low_second_reader_agreement_withholds_the_reading(tmp_path):
    import score117b as s
    pairs = _fixture(tmp_path, _concept_specific, _qwen_predicted)
    (tmp_path / "agreement.json").write_text(json.dumps({"rate": 0.8}), encoding="utf-8")
    r = s.score(tmp_path, tmp_path / "labels.jsonl", pairs, part117_answers=tmp_path / "missing.jsonl")
    assert r["reading"].startswith("withheld") and r["verdicts"]["117b.1"] == "held"


def test_missing_labels_are_counted_not_read_as_absent(tmp_path):
    import score117b as s
    pairs = _fixture(tmp_path, _concept_specific, _qwen_predicted)
    kept = [l for l in open(tmp_path / "labels.jsonl", encoding="utf-8") if '"p00:r0-1,s+1|s"' not in l]
    (tmp_path / "labels.jsonl").write_text("".join(kept), encoding="utf-8")
    r = s.score(tmp_path, tmp_path / "labels.jsonl", pairs, part117_answers=tmp_path / "missing.jsonl")
    assert r["labels_missing"] == 6 and r["who_slot_shows"]["random"] == [32, 1, 33]
    assert r["verdicts"]["117b.3"] == "held"


def test_prepare_labels_each_pushed_neuron_and_the_unpushed_row_against_every_unit(tmp_path):
    import score117b as s
    pairs = _fixture(tmp_path, _concept_specific, _qwen_predicted)
    kinds = {u: "slot" for u in s.all_units(pairs)}
    n = s.prepare(tmp_path, pairs, kinds)
    items = [json.loads(l) for l in open(tmp_path / "items.jsonl", encoding="utf-8")]
    assert n == len(items)
    ids = {it["id"] for it in items}
    assert "p00:c-1,s+1|c" in ids and "p00:c-1,s+1|s" in ids and "p00:u-1,s+1|u" in ids
    assert "p00:r0-1,s+1|s" in ids and not any(i.endswith("|r") for i in ids)
    assert "p00:n2-1,s+1|s" in ids and not any(i.endswith("|n") for i in ids)
    assert sum(i.startswith("none|") for i in ids) == 33
