"""Tests for Part 115's runner pieces and scorer, without the model. Written by Claude (Opus 5.5), 2026-10-29.
Deliberate break: make MaskZero return args unchanged; test_mask_zero_zeros_only_masked fails."""
import random, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import howmany115 as h  # noqa: E402
import score115 as s  # noqa: E402


def test_mask_zero_zeros_only_masked():
    lin = torch.nn.Linear(4, 2, bias=False)
    x = torch.ones(2, 3, 4)
    mask = torch.ones(2, 4)
    mask[1, 2] = 0
    seen = {}
    hook = h.MaskZero(lin, mask)
    lin.register_forward_pre_hook(lambda m, a: seen.setdefault("x", a[0].clone()))
    lin(x)
    hook.remove()
    assert seen["x"][0].eq(1).all() and seen["x"][1, :, 2].eq(0).all() and seen["x"][1, :, [0, 1, 3]].eq(1).all()


def test_control_sets_match_layers_and_skip_candidates():
    mean_abs = torch.rand(3, 100)
    cands = [(0, 5), (2, 7), (2, 9), (1, 1)]
    out = h.control_set(cands, mean_abs, set(cands), random.Random(0))
    assert [li for li, _ in out] == [li for li, _ in cands]
    assert not set(out) & set(cands) and len(set(out)) == len(out)


def test_build_rows_counts():
    attr = -torch.rand(3, 400)
    cands = h.top_candidates(attr, 500)
    rows = h.build_rows(cands, torch.rand(3, 400))
    assert len(rows) == 1 + len(h.KS) * (1 + h.N_SETS)
    assert all(len(r["units"]) == r["k"] for r in rows)


def fake(flips_by_k, base=6.0):
    def hrow(flipped, lo):
        return {"logodds": lo, "argmax_is_q": not flipped, "top": "Q"}
    rows = [{"role": "none", "k": 0, "heldout": [hrow(False, base)] * 16,
             "ordinary": [{"logprob": -0.1, "argmax_is_target": True}] * 4}]
    gens = {}
    for k in h.KS:
        f = flips_by_k.get(k, 0)
        rows.append({"role": "candidate", "k": k, "heldout": [hrow(i < f, base - 1 - k / 10) for i in range(16)],
                     "ordinary": [{"logprob": -0.3, "argmax_is_target": True}] * 4})
        for _ in range(h.N_SETS):
            rows.append({"role": "control", "k": k, "heldout": [hrow(False, base - 0.1)] * 16,
                         "ordinary": [{"logprob": -0.2, "argmax_is_target": True}] * 4})
        gens[str(k)] = [{"continuation": ("an AI assistant" if i < f else "Qwen3.5**")} for i in range(16)]
    return {"ks": h.KS, "rows": rows, "generations": gens, "gates": {}}


def test_scorer_verdicts():
    out = s.score(fake({20: 8, 50: 12}), -8.42)
    assert out["k_star"] == 20
    assert out["verdicts"] == {"115.1": "held", "115.2": "held", "115.3": "held", "115.4": "held", "115.5": "held",
                               "115.6": "held"}


def test_scorer_no_kstar_is_untestable():
    out = s.score(fake({}), -8.42)
    assert out["k_star"] is None and out["verdicts"]["115.2"] == "failed" and out["verdicts"]["115.4"] == "untestable"


def test_first_word_and_ai_names():
    assert s.first_word("**ChatGPT**, a model") == "ChatGPT" and "chatgpt" in s.AI_NAMES
