"""Tests for Part 109c's conditions, push hooks and scorer. Written by Claude (Opus 5.5), 2026-10-29."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part109c"))


def test_conditions_cover_the_design():
    import combo109c as m
    ids = [c["id"] for c in m.conditions()]
    assert len(ids) == 24 and len(set(ids)) == 24
    for d in ("-2", "-1", "-0.5", "+0.5", "+1", "+2"):
        assert f"concept{d}" in ids and f"slot{d}" in ids and f"both{d}" in ids
    by = {c["id"]: c["push"] for c in m.conditions()}
    assert by["concept+1_slot-1"] == {"p92-08": 1.0, "p92-37": -1.0}
    assert by["both-2"] == {"p92-08": -2.0, "p92-37": -2.0} and by["none"] == {}


def test_generate_multi_removes_every_hook_even_on_error(monkeypatch):
    import torch
    import combo109c as m
    removed, added = [], []

    class Handle:
        def __init__(self, i):
            self.i = i

        def remove(self):
            removed.append(self.i)

    def fake_hook(layer, rows):
        added.append(tuple(rows.shape))
        return Handle(len(added))

    class Tok:
        pad_token_id = 0
        eos_token_id = 1

    class Model:
        def generate(self, **kw):
            raise RuntimeError("boom")

    monkeypatch.setattr(m.c, "add_row_hook", fake_hook)
    try:
        m.generate_multi(Model(), Tok(), [[5, 6], [7]], [("L4", torch.ones(3)), ("L5", torch.ones(3))])
    except RuntimeError:
        pass
    assert added == [(2, 3), (2, 3)] and removed == [1, 2]


def test_scorer_counts_and_verdicts(tmp_path):
    import score109c as s
    import combo109c as m
    labels, answers = [], []
    for c in m.conditions():
        cid = c["id"]
        for qi in range(7):
            shows = (cid in ("both+1", "concept+1", "concept_slot2+1") and qi < 4) or (cid == "slot+1" and qi < 2) \
                or (cid == "concept+1_slot-1" and qi in (3, 4))
            labels.append({"id": cid, "unit": "combo", "question_index": qi, "shows_persona": shows or (cid == "both+1" and qi == 4),
                           "self_claim": cid == "concept-1_slot+1" and qi == 0, "coherent": cid != "both+2"})
            answers.append({"id": cid, "question_index": qi, "answer": "I am a dog, a good girl." if shows else "Hello."})
    labels.append({"id": "none", "unit": "combo", "question_index": 0, "error": "no JSON"})
    lp, ap = tmp_path / "l.jsonl", tmp_path / "a.jsonl"
    lp.write_text("\n".join(json.dumps(r) for r in labels) + "\n", encoding="utf-8")
    ap.write_text("\n".join(json.dumps(r) for r in answers) + "\n", encoding="utf-8")
    r = s.main(lp, ap, tmp_path / "score.json")
    assert r["conditions"]["both+1"]["shows"] == 5 and r["conditions"]["concept+1"]["dog_words"] == 8
    assert r["verdicts"] == {"109c.1": True, "109c.2": True, "109c.3": True, "109c.4": True, "109c.5": True,
                             "109c.6": True, "109c.7": True}
