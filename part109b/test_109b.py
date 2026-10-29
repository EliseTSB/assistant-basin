"""Tests for Part 109b's unit list and Part 109's runner defaults. Written by Claude (Opus 5.5), 2026-10-29."""
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part109b"))
sys.path.insert(0, str(ROOT / "part109"))
sys.path.insert(0, str(ROOT / "part109b"))


def test_units_are_every_concept_neuron():
    import followup109b as b
    kinds = json.loads((ROOT / "results/part92/score92.json").read_text(encoding="utf-8"))["kinds"]
    assert sorted(b.UNITS) == sorted(k for k, v in kinds.items() if v == "concept")
    assert len(b.UNITS) == 18


def test_every_unit_has_a_push_and_a_first_turn():
    import torch
    import followup109b as b
    pushes = torch.load(ROOT / "results/part99-pushes.pt", map_location="cpu")
    first = set()
    for line in open(ROOT / "results/part108/I.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["unit"] == "neuron" and int(r["question_index"]) == 0:
            first.add(r["id"])
    for uid in b.UNITS:
        assert f"{uid}|neuron" in pushes, uid
        assert uid in first, uid


def test_part109_defaults_unchanged():
    import followup109 as f
    assert f.UNITS == ["p92-37", "p92-04", "p92-13", "p92-07", "p92-35", "p92-24"]
    params = inspect.signature(f.main).parameters
    assert params["units"].default is None and params["out"].default is None


def test_scorer_counts_and_verdicts(tmp_path):
    import score109b as s
    concept = tmp_path / "c.jsonl"
    rows = []
    for i in range(18):
        uid = "p92-08" if i == 0 else f"u{i}"
        for arm, shows in (("A", i < 10), ("B", i < 15), ("C", i < 9), ("D", False)):
            rows.append({"id": uid, "unit": arm, "question_index": 0, "shows_persona": shows, "coherent": True})
    concept.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    slot = tmp_path / "s.jsonl"
    slot.write_text("\n".join(json.dumps({"id": f"s{i}", "unit": arm, "question_index": 0, "shows_persona": False,
                                          "coherent": True}) for i in range(6) for arm in "ABCD") + "\n",
                    encoding="utf-8")
    r = s.main(concept, slot, tmp_path / "score.json")
    assert r["concept"]["A"]["shows"] == 10 and r["concept"]["C"]["shows"] == 9
    assert r["verdicts"] == {"109b.1": True, "109b.2": True, "109b.3": False, "109b.4": True}
    assert abs(s.fisher_two_sided(11, 7, 3, 18) - 0.00333) < 1e-4


def test_scorer_skips_error_rows_and_keeps_the_retry(tmp_path):
    import score109b as s
    path = tmp_path / "l.jsonl"
    rows = [{"id": "p92-08", "unit": "A", "question_index": 0, "error": "no JSON"},
            {"id": "p92-08", "unit": "A", "question_index": 0, "shows_persona": True, "coherent": True},
            {"id": "x", "unit": "B", "question_index": 0, "shows_persona": False, "coherent": True}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out, by_unit = s.counts(path)
    assert out["A"] == {"n": 1, "shows": 1, "coherent": 1} and by_unit["p92-08"]["A"] is True
