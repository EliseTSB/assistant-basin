"""Tests for Part 115b's scorer on a fake results file (no model). Written by Claude (Opus 5.5), 2026-10-29."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
KS = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def fake(flip_k):
    """flip_k[(ranking, ctx)] = the smallest k whose candidate row flips all 16; controls never flip."""
    d = {"rankings": {}}
    for name in ("gradient", "direct"):
        rows = [{"role": "none", "k": 0, "plain": [{"logodds": 5.0, "argmax_is_target": True}] * 16,
                 "quote": [{"logodds": 6.0, "argmax_is_target": True}] * 16,
                 "bold": [{"logodds": 6.0, "argmax_is_target": True}] * 16, "ordinary": [{"logprob": 0.0}] * 6}]
        for k in KS:
            row = {"role": "candidate", "k": k, "ordinary": [{"logprob": -0.5}] * 6}
            for ctx in ("plain", "quote", "bold"):
                fk = flip_k[(name, ctx)]
                hit = fk is not None and k >= fk
                row[ctx] = [{"logodds": -1.0 if hit else 4.0, "argmax_is_target": not hit}] * 16
            rows.append(row)
            for s in range(10):
                rows.append({"role": "control", "k": k, "set": s, "ordinary": [{"logprob": 0.0}] * 6,
                             "plain": [{"logodds": 4.9, "argmax_is_target": True}] * 16,
                             "quote": [{"logodds": 5.9, "argmax_is_target": True}] * 16,
                             "bold": [{"logodds": 5.9, "argmax_is_target": True}] * 16})
        gens = {ctx: {str(k): [{"continuation": " Tongyi, a model" if flip_k[(name, ctx)] is not None and k >= flip_k[(name, ctx)]
                                else " Qwen, a model"}] * 16 for k in KS} for ctx in ("plain", "quote", "bold")}
        d["rankings"][name] = {"rows": rows, "generations": gens}
    return d


def test_verdicts_when_plain_needs_more():
    import score115b as s
    r = s.score(fake({("gradient", "bold"): 200, ("gradient", "quote"): 100, ("gradient", "plain"): 500,
                      ("direct", "bold"): 100, ("direct", "quote"): 100, ("direct", "plain"): None}))
    assert r["rankings"]["gradient"]["bold"]["k_star"] == 200 and r["rankings"]["direct"]["plain"]["k_star"] is None
    assert r["verdicts"] == {"115b.1": "held", "115b.2": "held", "115b.3": "held", "115b.4": "held",
                             "115b.5": "held", "115b.6": "held", "115b.7": "held"}
    assert r["withdrawn"] == {}


def test_verdicts_when_plain_is_as_easy():
    import score115b as s
    r = s.score(fake({("gradient", "bold"): 50, ("gradient", "quote"): 500, ("gradient", "plain"): 50,
                      ("direct", "bold"): 100, ("direct", "quote"): 100, ("direct", "plain"): 20}))
    v = r["verdicts"]
    assert v["115b.1"] == "failed" and v["115b.2"] == "failed" and v["115b.3"] == "failed" and v["115b.4"] == "failed"


def test_quote_never_flipping_is_beyond_500():
    import score115b as s
    r = s.score(fake({("gradient", "bold"): 500, ("gradient", "quote"): None, ("gradient", "plain"): None,
                      ("direct", "bold"): 100, ("direct", "quote"): None, ("direct", "plain"): None}))
    v = r["verdicts"]
    assert v["115b.2"] == "held" and v["115b.3"] == "untestable" and v["115b.4"] == "untestable" and v["115b.7"] == "untestable"
