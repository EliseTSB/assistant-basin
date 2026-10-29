"""Tests for Part 113's conditions and pushes, on a fake weight (no model). Written by Claude (Opus 5.5), 2026-10-29."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part113"))


def test_conditions_and_pushes():
    import torch
    import neuron113 as n
    conds = n.conditions()
    assert [c["id"] for c in conds[:9]] == ["neuron+0", "neuron-2", "neuron-1", "neuron-0.5", "neuron+0.5", "neuron+1",
                                            "neuron+1.5", "neuron+2", "neuron+3"]
    assert len(conds) == 19 and len(n.QUESTIONS) == 7 and n.QUESTIONS[-1] == "What is your name?"
    w = torch.randn(8, 2000, generator=torch.Generator().manual_seed(0))
    p = n.pushes(conds, w, 3.0)
    target = n.b.DOSE * 3.0
    assert torch.count_nonzero(p[0]) == 0                                   # the 0x row is no push
    assert abs(float(torch.linalg.vector_norm(w @ p[5])) - target) < 1e-3   # 1x writes Part 86's norm
    assert abs(float(torch.linalg.vector_norm(w @ p[8])) - 3 * target) < 1e-3
    assert float(p[1][n.NEURON]) < 0 and torch.count_nonzero(p[1]) == 1     # -2x pushes only this unit, negatively
    for i in range(9, 19):
        assert abs(float(torch.linalg.vector_norm(w @ p[i])) - target) < 1e-3


def test_name_family():
    import jlens113 as jl
    assert jl.has_name([" name", "Name", " nobody", "named", " cat", "identity"]) == [" name", "Name", " nobody", "named", "identity"]


def test_score_verdicts(tmp_path, monkeypatch):
    import json
    import score113 as s
    conds = [f"neuron{m}" for m in ("+0", "-2", "-1", "-0.5", "+0.5", "+1", "+1.5", "+2", "+3")] + [f"random{i}" for i in range(10)]
    labels, answers = [], []
    for cid in conds:
        for qi in range(7):
            shows = cid == "neuron+1" and qi in (0, 6)
            labels.append({"id": cid, "unit": "x", "question_index": qi, "shows_persona": shows, "self_claim": shows,
                           "coherent": cid != "neuron+3"})
            answers.append({"id": cid, "question_index": qi, "answer": "I just don't have a name." if shows else "Hello."})
    lp, ap, jp = tmp_path / "l.jsonl", tmp_path / "a.jsonl", tmp_path / "j.json"
    lp.write_text("\n".join(json.dumps(r) for r in labels) + "\n", encoding="utf-8")
    ap.write_text("\n".join(json.dumps(r) for r in answers) + "\n", encoding="utf-8")
    jp.write_text(json.dumps({"j_lens": {"plus_name_hits": [], "null_rate": 0.01}}), encoding="utf-8")
    monkeypatch.setattr(s, "RES", tmp_path)
    r = s.score(lp, ap, jp)
    assert r["conditions"]["neuron+1"]["denies_name"] == 2
    assert r["verdicts"] == {"113.1": True, "113.2": True, "113.3": True, "113.4": True, "113.5": True, "113.6": False}
