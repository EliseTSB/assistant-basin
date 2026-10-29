"""Tests for part108/score108.py and items108.py on built label fixtures."""
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import items108  # noqa: E402
import score108 as s  # noqa: E402

SLOT = [f"s{i:02d}" for i in range(20)]
CONCEPT = ["c0", "c1", "c2", "c3"]
KINDS = {**{u: "slot" for u in SLOT}, **{u: "concept" for u in CONCEPT}}


def labels(lost_by_tag, incoherent_unpushed=(), incoherent_pushed=None, concept_lost=None):
    """Every slot unit shows at I; each tag loses the listed slot units; concept units show unless listed."""
    out = {}
    for tag in s.TAGS:
        rows = {}
        for u in SLOT:
            shown = u not in lost_by_tag.get(tag, ())
            coh = u not in (incoherent_pushed or {}).get(tag, ())
            rows[(u, "neuron", 0)] = {"shows_persona": shown, "coherent": coh}
            rows[(u, "random", 0)] = {"shows_persona": False, "coherent": True}
            rows[(u, "baseline", 0)] = {"shows_persona": False, "coherent": True}
        for u in CONCEPT:
            rows[(u, "neuron", 0)] = {"shows_persona": u not in (concept_lost or {}).get(tag, ()), "coherent": True}
            rows[(u, "random", 0)] = {"shows_persona": False, "coherent": True}
        first = sorted(KINDS)[0]
        for q in range(6):
            rows[(first, "baseline", q)] = {"shows_persona": False, "coherent": tag not in incoherent_unpushed}
        out[tag] = rows
    return out


def test_fisher_matches_the_hypergeometric_tail():
    assert abs(s.fisher_greater(2, 0, 0, 2) - 1 / comb(4, 2)) < 1e-12
    assert abs(s.fisher_greater(0, 2, 2, 0) - 1.0) < 1e-12
    a, b, c, d = 3, 1, 1, 5
    tail = sum(comb(4, x) * comb(6, 4 - x) for x in range(3, 5)) / comb(10, 4)
    assert abs(s.fisher_greater(a, b, c, d) - tail) < 1e-12


def test_forward_losing_exactly_half_holds_and_reads_F():
    lost = {"S": SLOT[:8], "D": SLOT[:4], "FP": SLOT[10:14], "FR": SLOT[10:12], "XP": SLOT[14:18], "XR": SLOT[14:16]}
    r = s.score(labels(lost), KINDS)
    assert r["B"] == 20 and r["losses"]["S"] == 8 and r["losses"]["FP"] == 4
    assert r["verdicts"]["108.1"] == "held" and r["verdicts"]["108.2"] == "held" and r["verdicts"]["108.3"] == "held"
    assert r["reading"].startswith("F:")
    assert r["overlap"]["FP"]["both"] == 0 and r["verdicts"]["108.4"] == "held"


def test_forward_just_under_half_fails():
    lost = {"S": SLOT[:8], "D": SLOT[:4], "FP": SLOT[10:13], "FR": SLOT[10:12], "XP": SLOT[14:18], "XR": SLOT[14:16]}
    assert s.score(labels(lost), KINDS)["verdicts"]["108.1"] == "failed"


def test_small_forward_losses_with_large_scrambled_losses_read_directional():
    lost = {"S": SLOT[:8], "D": SLOT[:4], "FP": SLOT[:2], "FR": SLOT[:1], "XP": SLOT[:4], "XR": SLOT[:2]}
    r = s.score(labels(lost), KINDS)
    assert r["verdicts"]["108.1"] == "failed" and r["verdicts"]["108.2"] == "failed"
    assert r["reading"].startswith("D:")
    lost["XP"] = SLOT[:3]
    assert s.score(labels(lost), KINDS)["reading"].startswith("D?:")


def test_positive_overlap_fails_108_4():
    lost = {"S": SLOT[:8], "D": SLOT[:4], "FP": SLOT[:8], "FR": SLOT[:4], "XP": SLOT[:8], "XR": SLOT[:4]}
    r = s.score(labels(lost), KINDS)
    assert r["overlap"]["FP"]["both"] == 8 and r["verdicts"]["108.4"] == "failed"


def test_gate_failures_leave_the_prediction_unread():
    lost = {"S": SLOT[:8], "D": SLOT[:4], "FP": SLOT[:8], "FR": SLOT[:4], "XP": SLOT[:8], "XR": SLOT[:4]}
    r = s.score(labels(lost, incoherent_unpushed=("FP",)), KINDS)
    assert r["models"]["FP"]["gate"] is False and r["verdicts"]["108.1"] == "not read" and r["verdicts"]["108.4"] == "not read"
    r = s.score(labels(lost, incoherent_pushed={"FR": SLOT[:3]}), KINDS)
    assert r["models"]["FR"]["gate"] is False and r["verdicts"]["108.2"] == "not read"
    r = s.score(labels(lost, incoherent_pushed={"FR": SLOT[:2]}), KINDS)
    assert r["models"]["FR"]["gate"] is True


def test_concept_and_order_predictions():
    lost = {"S": SLOT[:8], "D": SLOT[:4], "FP": SLOT[:4], "FR": SLOT[:2], "XP": SLOT[:4], "XR": SLOT[:2]}
    r = s.score(labels(lost), KINDS)
    assert r["verdicts"]["108.5"] == "held" and r["verdicts"]["108.6"] == "held"
    assert s.score(labels(lost, concept_lost={"FR": ["c0"]}), KINDS)["verdicts"]["108.5"] == "failed"
    lost_bad = dict(lost, D=SLOT[:9])
    assert s.score(labels(lost_bad), KINDS)["verdicts"]["108.6"] == "failed"


def test_items_keep_pushed_baseline_q0_and_first_unit_unpushed():
    rows = [{"id": u, "unit": unit, "question_index": q} for u in ("a", "b") for unit, qs in
            (("neuron", [0]), ("random", [0]), ("baseline", range(6))) for q in qs]
    kept = items108.keep(rows)
    assert sum(r["unit"] == "baseline" for r in kept) == 6 + 1
    assert {(r["id"], r["question_index"]) for r in kept if r["unit"] == "baseline"} == {("a", q) for q in range(6)} | {("b", 0)}
    assert sum(r["unit"] in ("neuron", "random") for r in kept) == 4


def test_moderate_overlap_holds_108_4():
    lost = {"S": SLOT[:8], "D": SLOT[:4], "FP": SLOT[4:10], "FR": SLOT[:2], "XP": SLOT[:4], "XR": SLOT[:2]}
    r = s.score(labels(lost), KINDS)
    assert r["overlap"]["FP"]["both"] == 4 and 0.05 <= r["overlap"]["FP"]["p_greater"] < 0.5
    assert r["verdicts"]["108.4"] == "held"


def test_forward_between_a_quarter_and_half_reads_mixed():
    lost = {"S": SLOT[:8], "D": SLOT[:4], "FP": SLOT[:3], "FR": SLOT[:1], "XP": SLOT[:4], "XR": SLOT[:2]}
    r = s.score(labels(lost), KINDS)
    assert r["verdicts"]["108.1"] == "failed" and r["reading"].startswith("mixed")
