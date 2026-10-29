"""Tests for Part 109b's unit list and Part 109's runner defaults. Written by Claude (Opus 5.5), 2026-10-29."""
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part109b"))
sys.path.insert(0, str(ROOT / "part109"))


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
