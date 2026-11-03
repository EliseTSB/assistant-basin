#!/usr/bin/env python3
"""Counterexamples for the Part 181 review: cases where a gate or a reading passes with something wrong.

Runs the real collector (part181/olmo_atlas_collect.py) on the tiny random OLMo 2 from its own tests, then shows:

  A. Gate 1 (top-1 recompute) passes on a state whose ranks 2 to k are corrupted; top-10 overlap against the true
     state is 0.1 for every unit.
  B. Gate 2 as written (token sha, control sha, commit prefix string) cannot tell two runs of the same weights
     apart: nothing in the run's recorded config fingerprints the weights.
  C. Gate 3 (median overlap of two Instruct runs at least 0.95) passes with 30% of units randomised.
  D. The active-unit rule can manufacture a strictly rising median across checkpoints with no unit changing.
  E. A median of top-10 overlaps takes one of eleven values, so 181.4's strict rise can fail by ties while the
     underlying rate rises, and 181.5's "exceeds" can fail at the ceiling.
  F. With the collector's defaults (checkpoint-every 64), the saturation snapshots at 500, 1,000, 1,500 and 2,000
     documents are never written; only a chunk end that is a multiple of 500 is.

CPU only; no download. Run from the repository root with the test virtualenv:
    python review/part181/counterexamples181.py
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "part181"))
import olmo_atlas_collect as oac  # noqa: E402
import test_olmo_atlas_collect as tt  # noqa: E402  (ToyTokenizer, tiny_model, make_corpus, args_for)

import torch  # noqa: E402


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def load_state(out: Path) -> dict:
    with np.load(out / "state.npz") as s:
        return {k: s[k] for k in s.files}


def top10_overlap(a_doc: np.ndarray, b_doc: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-unit |top-k(a) ∩ top-k(b)| / k on document ids, over a [L, U, K] pair of arrays."""
    L, U, _ = a_doc.shape
    out = np.empty((L, U))
    for l in range(L):
        for u in range(U):
            sa = {int(d) for d in a_doc[l, u, :k] if d >= 0}
            sb = {int(d) for d in b_doc[l, u, :k] if d >= 0}
            out[l, u] = len(sa & sb) / k
    return out


def gate1(state: dict, model, tok, texts, layers, n_pairs: int, seed: int, controls: np.ndarray,
          rel_tol: float = 1e-4) -> dict:
    """The design's Gate 1: recompute n_pairs (unit, top-1 document) activations by a lone forward (batch 1)."""
    rng = np.random.default_rng(seed)
    L, U, _ = state["top_doc"].shape
    picks = [(int(rng.integers(L)), int(rng.integers(U))) for _ in range(n_pairs)]
    failures = 0
    for slot, u in picks:
        doc, pos, act = int(state["top_doc"][slot, u, 0]), int(state["top_pos"][slot, u, 0]), float(state["top_act"][slot, u, 0])
        ids = torch.tensor([tok([texts[doc]], True, 512, False, True)["input_ids"][0]])
        store = {}
        li = layers[slot]
        h = model.model.layers[li].mlp.down_proj.register_forward_pre_hook(lambda _m, a: store.__setitem__("x", a[0][0].clone()))
        with torch.no_grad():
            model(input_ids=ids)
        h.remove()
        x = store["x"]
        value = float(x[pos, u]) if u < x.shape[1] else float(x[pos] @ torch.from_numpy(controls[slot][u - x.shape[1]]))
        if abs(value - act) > rel_tol * max(abs(act), 1e-6):
            failures += 1
    return {"pairs": n_pairs, "failures": failures, "passed": failures == 0}


def example_a_b(tmp: Path) -> dict:
    tt.make_corpus(tmp / "c.jsonl", n=200)  # enough documents that two top-10 sets rarely overlap by chance
    tok, model = tt.ToyTokenizer(), tt.tiny_model()
    texts = oac.read_corpus_texts(tmp / "c.jsonl", None)
    layers = tt.LAYERS
    controls = np.stack([oac.generate_controls(5, tt.WIDTH, 7, li) for li in layers])
    run_a = tmp / "a"
    quiet(oac.collect, tt.args_for(tmp, run_a), tok=tok, model=model)
    true = load_state(run_a)
    # A: corrupt every rank below the top one: rotate ranks 1..k-1 across units so each unit holds another unit's docs.
    bad = {k: v.copy() for k, v in true.items()}
    for key in ("top_doc", "top_pos", "top_act"):
        bad[key][:, :, 1:] = np.roll(true[key][:, :, 1:], shift=1, axis=1)
    gate_true = gate1(true, model, tok, texts, layers, 32, 181, controls)
    gate_bad = gate1(bad, model, tok, texts, layers, 32, 181, controls)
    overlap = top10_overlap(true["top_doc"], bad["top_doc"], k=min(10, len(texts)))
    # B: a second run of the same weights under another label; the recorded config differs only by the label.
    run_b = tmp / "b"
    quiet(oac.collect, tt.args_for(tmp, run_b, model_arm="dpo"), tok=tok, model=model)
    cfg_a = json.loads((run_a / "progress.json").read_text())["config"]
    cfg_b = json.loads((run_b / "progress.json").read_text())["config"]
    differing = sorted(k for k in cfg_a if cfg_a[k] != cfg_b.get(k))
    same_weights_overlap = top10_overlap(true["top_doc"], load_state(run_b)["top_doc"], k=min(10, len(texts)))
    return {"A_gate1_true_state": gate_true, "A_gate1_corrupted_state": gate_bad,
            "A_top10_overlap_corrupted_vs_true_median": float(np.median(overlap)),
            "B_config_keys_that_differ_between_two_runs_of_the_same_weights": differing,
            "B_weight_fingerprint_fields_in_config": [k for k in cfg_a if "sha" in k or "model" in k],
            "B_overlap_between_the_two_runs_median": float(np.median(same_weights_overlap))}


def example_c() -> dict:
    rng = np.random.default_rng(3)
    n = 131_072
    overlap = np.ones(n)
    bad = rng.random(n) < 0.30
    overlap[bad] = rng.integers(0, 3, bad.sum()) / 10.0  # 0.0, 0.1 or 0.2 for the randomised units
    return {"units": n, "randomised_share": 0.30, "median": float(np.median(overlap)), "mean": float(overlap.mean()),
            "gate3_bar": 0.95, "gate3_passes": bool(np.median(overlap) >= 0.95),
            "share_below_0.5": float((overlap < 0.5).mean())}


def example_d() -> dict:
    """Every unit's true overlap with Instruct is fixed over pretraining; only the active set changes."""
    rng = np.random.default_rng(181)
    n = 8192
    true_overlap = np.round(rng.beta(2, 2, n), 1)  # fixed per unit, never changes across checkpoints
    # Low-overlap units are the ones the activity rule drops (a unit with no stable top document is a unit with a
    # small, noisy maximum). Whether the inactive share rises along training (dead units accumulate) or falls (early
    # maxima are uniform and tiny), the median over the surviving units moves with the exclusion, not with any unit.
    names = ["21B", "210B", "1T", "2T", "3T", "4T"]
    order = np.argsort(true_overlap)  # the lowest-overlap units are excluded first
    out = {"per_unit_overlap_changes_across_checkpoints": False, "median_over_all_units": float(np.median(true_overlap))}
    for label, inactive_share in (("inactive_share_rising", [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]),
                                  ("inactive_share_falling", [0.60, 0.50, 0.40, 0.30, 0.20, 0.10])):
        medians = []
        for share in inactive_share:
            active = np.ones(n, dtype=bool)
            active[order[: int(share * n)]] = False
            medians.append(float(np.median(true_overlap[active])))
        out[label] = {"inactive_share": dict(zip(names, inactive_share)),
                      "median_over_active_units": dict(zip(names, medians)),
                      "non_decreasing": all(b >= a for a, b in zip(medians, medians[1:])),
                      "strictly_rising": all(b > a for a, b in zip(medians, medians[1:]))}
    return out


def example_e() -> dict:
    rng = np.random.default_rng(5)
    n = 131_072
    rates = [0.55, 0.58, 0.60, 0.61, 0.62, 0.63]  # a rising true rate of keeping each top document
    medians, means = [], []
    for p in rates:
        o = rng.binomial(10, p, n) / 10.0
        medians.append(float(np.median(o)))
        means.append(float(o.mean()))
    ceiling = rng.binomial(10, 0.97, 1000) / 10.0
    control = rng.binomial(10, 0.95, 256) / 10.0
    return {"true_rates": rates, "medians": medians, "means": [round(m, 4) for m in means],
            "median_strictly_rising": all(b > a for a, b in zip(medians, medians[1:])),
            "mean_strictly_rising": all(b > a for a, b in zip(means, means[1:])),
            "ceiling_pair_neuron_median": float(np.median(ceiling)), "ceiling_pair_control_median": float(np.median(control)),
            "neuron_exceeds_control_at_ceiling": bool(np.median(ceiling) > np.median(control))}


def example_f(tmp: Path) -> dict:
    tt.make_corpus(tmp / "c2.jsonl", n=520)
    tok, model = tt.ToyTokenizer(), tt.tiny_model()
    out = {}
    for every in (64, 50):
        run = tmp / f"snap{every}"
        args = tt.args_for(tmp, run, corpus=str(tmp / "c2.jsonl"), passes=1, checkpoint_every=every, snapshot_every=500)
        quiet(oac.collect, args, tok=tok, model=model)
        out[f"checkpoint_every_{every}"] = sorted(Path(p).name for p in glob.glob(str(run / "snapshot-pass1-docs*.npz")))
    return {"docs": 520, "snapshot_every": 500, "snapshots_written": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "review/part181/counterexamples181.json"))
    a = ap.parse_args(argv)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ce181-") as d:
        tmp = Path(d)
        results = {"A_B": example_a_b(tmp), "C": example_c(), "D": example_d(), "E": example_e(), "F": example_f(tmp)}
    Path(a.out).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
