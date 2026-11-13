#!/usr/bin/env python3
"""score211.py: the gate table, the whole-statistic bootstrap, the noise floor and every verdict of PART211-DESIGN.md
(the frozen design, Journal commit 78228f34a, sha256 6c853ffa...).

Reads only the distance files (results/part211/records/*.npz), never the models.

Uncertainty (design: Uncertainty, "the whole statistic is resampled"). One resample is one draw, with replacement, of
set A's 48 document IDs (seed 211, 1,000 resamples), shared by every checkpoint, run and measure. Inside each resample
everything a prediction reads is recomputed from the per-document records: the distances, the head assignment (refit
by linear_sum_assignment on the resampled cost matrix), the fractions, the persistent crossing step s*, the settling
times with their censoring, the median over data-seed runs, and the layer's verdict. A layer's interval is the 2.5th to
97.5th percentile of that recomputed statistic.

Gates (design: Controls).
  G1 the seed split: D runs equal R0 at step 0; W runs differ from R0 and from each other; and, for every 2-D weight
     matrix, |Pearson r| < 0.05 between W step 0 and R0 step 0, W step 0 and R0 step 143000, and W1 and W2 step 0.
  G2 the measures: zero controls (a repeat gives P, M, O, V of exactly 0; a random within-layer permutation of the
     final heads is matched back at 0), a duplicate-head control (any zero-cost assignment passes), and nonzero
     controls with known values (disjoint rows P = 1 within 1e-6; a doubled OV block V = 1 within 1e-6; independent
     Gaussian features M >= 0.5; rotated and isotropically scaled features M = 0 within 1e-5).
  G3 per prediction: the rules in the design's "Per-prediction rules", each in its own units; "unmeasured" when the
     rule says so.

  python -I -B part211/score211.py --out results/part211 [--json results/part211/score.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import measures211 as MS
import checkpoints211 as C
import text211 as TX
import record211 as RC

N_A_DOCS = TX.N_A                  # set A's documents, the registered verdict set
N_POSITIONS = len(RC.POSITIONS)    # recorded query positions per document (85)

N_BOOT = 1000
N_BOOT_M = 200                 # the design's fixed fallback if the pilot projects the M bootstrap over two hours:
                               # M uses the first 200 of the 1,000 draws, and every statistic pairing P with M uses
                               # those same 200 draws for P (a paired B = 200); P-only statistics keep all 1,000.


def threshold(b: int) -> int:
    """The count a "at least 97.5% of the resamples" rule needs: ceil(0.975 B), 975 of 1,000 or 195 of 200."""
    return int(np.ceil(0.975 * b))


def paired_draws(idx: np.ndarray, b: int = N_BOOT_M) -> np.ndarray:
    """The first b of the shared draws: what any statistic pairing P with M reads, so the pairing is kept. A draw is
    one row of `idx` (one document ID per document); the result is [b, n_docs]."""
    idx = np.asarray(idx)
    if idx.ndim == 1:
        idx = idx.reshape(1, -1)
    return idx[:b]
BOOT_SEED = 211
ALPHA = 0.05
LO, HI = ALPHA / 2, 1 - ALPHA / 2
G3_FRACTION = 0.25             # G3, 211.1/211.2/211.3/211.8: the floor against a quarter of the compared value
G3_FACTOR = 4.0                # G3, 211.4/211.5/211.6/211.7: denominators at or below four times the floor
G1_MAX_R = 0.05                # G1: |Pearson r| below this for every 2-D weight matrix
SHARED_ELIGIBLE = 0.001        # P0: a query keeping less than this outside key 0 is dropped
PYTHIA_RUNS = C.runs_of(C.PYTHIA_FAMILY)
WEIGHT_RUNS = C.runs_with_factor("weights")
DATA_RUNS = C.runs_with_factor("data")
PYTHIA_STEPS = C.STEPS
FINAL = C.FINAL_STEP
THREAD = C.THREAD_STEP
N_LAYERS = 12
N_HEADS = 12
N_OLMO_LAYERS = 16
ALL_RUNS_BAR = len(PYTHIA_RUNS) - 1          # 211.4, 211.5: "5 of the 6 runs"
STEP_16000 = 16000
UNMEASURED = "unmeasured"


# ================================================================ bootstrap machinery

def resample_indices(n_docs: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """The shared document draws: one array [n_boot, n_docs] of document IDs, used for every checkpoint, run and
    measure so paired quantities stay paired (design: Uncertainty)."""
    return np.random.default_rng(seed).integers(0, n_docs, size=(n_boot, n_docs))


def percentile_interval(values, lo: float = LO, hi: float = HI) -> list:
    """The [lo, hi] percentile interval of a recomputed statistic; censored or unmeasured draws are passed as
    np.inf (censored) or None (unmeasured) and handled by the caller's statistic, which returns a float or None."""
    v = np.asarray([x for x in values if x is not None], np.float64)
    if v.size == 0:
        return [None, None]
    return [float(np.quantile(v, lo)), float(np.quantile(v, hi))]


def bootstrap_ci(values, n_boot: int = N_BOOT, seed: int = BOOT_SEED, alpha: float = ALPHA) -> list:
    """The 95% percentile interval of the mean of per-document `values` (a diagnostic; the verdicts use the whole
    statistic)."""
    v = np.asarray(values, np.float64)
    if v.size < 2:
        return [None, None]
    idx = resample_indices(v.size, n_boot, seed)
    means = v[idx].mean(axis=1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def clears(ci, value: float, side: str) -> bool:
    """Does the interval lie entirely on the predicted side ('above' or 'below') of `value`?"""
    if ci is None or ci[0] is None:
        return False
    return ci[0] > value if side == "above" else ci[1] < value


def as_late(x) -> float:
    """A crossing step for comparisons: censored counts as later than any step."""
    return np.inf if x == MS.CENSORED else float(x)


# ================================================================ G1, the seed split

def _learned(state: dict) -> dict:
    """Learned parameters only: float tensors that are not rotary frequencies or attention masks."""
    out = {}
    for k, v in state.items():
        if "rotary" in k or "inv_freq" in k or "mask" in k:
            continue
        if isinstance(v, (int, float, bool)):
            continue
        # kept in its own dtype: float32 -> float64 is exact, so equality is unchanged, and pearson casts per tensor;
        # upcasting every state dict at once would hold about 9 GB for seven Pythia-160m checkpoints
        out[k] = np.asarray(v)
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson's r between two same-shaped arrays (flattened), 0 when either is constant."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def g1_seed_split(state_r0_step0: dict, seed_states: dict, r0_final: dict | None = None) -> dict:
    """G1. `seed_states` maps run -> its step-0 state dict; `r0_final` is R0's step-143000 state dict.

    D runs must equal R0 step 0 in every tensor. W runs must differ from R0 step 0, and from each other, in at least
    one tensor, and for every 2-D weight matrix |Pearson r| must be below G1_MAX_R between W step 0 and R0 step 0,
    between W step 0 and R0 step 143000, and between W1's and W2's step 0. The D half stops 211.2 and 211.3; the W
    half stops 211.1, 211.2 and 211.8 and the headline (design: Controls, gate table)."""
    r0 = _learned(state_r0_step0)
    r0f = _learned(r0_final) if r0_final is not None else None
    w_runs = [r for r in seed_states if r.startswith("W")]
    out = {"pass": True, "d_pass": True, "w_pass": True, "runs": {}}
    for run, state in seed_states.items():
        other = _learned(state)
        same_keys = set(r0) == set(other)
        differing = sorted(k for k in set(r0) & set(other) if not np.array_equal(r0[k], other[k]))
        if run.startswith("D"):
            ok = same_keys and not differing
            out["d_pass"] = out["d_pass"] and bool(ok)
            out["runs"][run] = {"same_keys": same_keys, "n_differing": len(differing), "differing": differing[:10],
                                "pass": bool(ok)}
        else:
            others = [o for o in w_runs if o != run]
            differs_from_w = all(any(not np.array_equal(other[k], _learned(seed_states[o])[k])
                                     for k in set(other) & set(_learned(seed_states[o]))) for o in others)
            corr = w_correlations(run, r0, r0f, {o: _learned(seed_states[o]) for o in others + [run]})
            corr_ok = all(abs(v) < G1_MAX_R for v in corr.values())
            ok = same_keys and bool(differing) and differs_from_w and corr_ok
            out["w_pass"] = out["w_pass"] and bool(ok)
            out["runs"][run] = {"same_keys": same_keys, "n_differing": len(differing), "differing": differing[:10],
                                "correlations": corr, "correlation_max": max((abs(v) for v in corr.values()), default=0.0),
                                "pass": bool(ok)}
        out["pass"] = out["pass"] and bool(out["runs"][run]["pass"])
    return out


def w_correlations(run: str, r0_step0: dict, r0_final: dict | None, w_states: dict) -> dict:
    """|Pearson r| for every 2-D weight matrix of W run `run`: against R0 step 0, R0 step 143000, and the other W
    runs' step 0. The largest is what the gate reads."""
    out = {}
    mine = w_states[run]
    for k, v in mine.items():
        if v.ndim != 2:
            continue
        if k in r0_step0:
            out[f"{k}|r0_step0"] = pearson(v, r0_step0[k])
        if r0_final is not None and k in r0_final:
            out[f"{k}|r0_step143000"] = pearson(v, r0_final[k])
        for o, other in w_states.items():
            if o != run and k in other:
                out[f"{k}|{o}_step0"] = pearson(v, other[k])
    return out


# ================================================================ G2, the measures

def g2_measures(repeat: dict, permuted: dict, duplicate_matched: float | None = None,
                nonzero: dict | None = None) -> dict:
    """G2. `repeat` holds the P, M, O and V of a repeated forward pass (each exactly 0); `permuted` holds, per layer,
    the matched cost between the final heads permuted within the layer and the originals (each exactly 0);
    `duplicate_matched` is the matched cost of a synthetic layer with two identical heads (any zero-cost assignment
    passes); `nonzero` holds the nonzero controls' values, each checked against its known value."""
    out = {"pass": True, "repeat": {}, "permutation": {}, "duplicate": {}, "nonzero": {}}
    for name in ("P", "M", "O", "V"):
        v = float(repeat[name])
        out["repeat"][name] = {"value": v, "pass": v == 0.0}
        out["pass"] = out["pass"] and v == 0.0
    for layer, d in permuted.items():
        v = float(d)
        out["permutation"][str(layer)] = {"matched": v, "pass": v == 0.0}
        out["pass"] = out["pass"] and v == 0.0
    if duplicate_matched is not None:
        ok = float(duplicate_matched) == 0.0
        out["duplicate"] = {"matched": float(duplicate_matched), "pass": ok}
        out["pass"] = out["pass"] and ok
    for name, (value, rule) in (nonzero or {}).items():
        ok = nonzero_ok(float(value), rule)
        out["nonzero"][name] = {"value": float(value), "rule": list(rule), "pass": bool(ok)}
        out["pass"] = out["pass"] and bool(ok)
    return out


def nonzero_ok(value: float, rule: tuple) -> bool:
    """One nonzero control's rule: ("approx", want, tol) passes when |value - want| <= tol; ("at_least", bound) passes
    when value >= bound. A non-finite value fails either. (Alden #13: the earlier (value, want, tol) form chose the
    approximate branch whenever want was finite, so M_independent's (value, 0.5, inf) accepted M = 0.)"""
    if not np.isfinite(value):
        return False
    if rule[0] == "approx":
        _, want, tol = rule
        if not (np.isfinite(want) and np.isfinite(tol) and tol >= 0):
            raise ValueError(f"an approx rule needs a finite want and a finite tolerance: {rule!r}")
        return abs(value - want) <= tol
    if rule[0] == "at_least":
        _, bound = rule
        if not np.isfinite(bound):
            raise ValueError(f"an at_least rule needs a finite bound: {rule!r}")
        return value >= bound
    raise ValueError(f"unknown nonzero-control rule: {rule!r}")


def g2_nonzero_controls(rng=None) -> dict:
    """The nonzero controls (design: Controls, G2), computed on hand-built inputs: {name: (value, rule)}, rule as in
    nonzero_ok.

    disjoint rows give P = 1 within 1e-6; a doubled OV block gives V = 1 within 1e-6; independent Gaussian features
    of one layer's recorded pooled shape (set A's 48 documents times the 85 recorded positions, by 768 features) give
    M of at least 0.5; the same features rotated and isotropically scaled give M = 0 within 1e-5. (The first build used
    85 positions by 768 features: with fewer positions than features, independent Gaussian features have nearly the
    same Gram matrix, so M came out near 0.1, hidden by the infinite tolerance; Alden #13.)"""
    rng = rng or np.random.default_rng(BOOT_SEED)
    T = 256
    a = np.zeros((T, T))
    b = np.zeros((T, T))
    for q in range(T):
        a[q, q] = 1.0
    b[:, 0] = 1.0
    # disjoint support for every query 1..T-1: a puts its mass on key q, b on key 0 (query 0 is left out of P)
    p_disjoint = MS.pattern_distance(a, b)
    d = 768
    ov = rng.normal(size=(d, 64)) @ rng.normal(size=(64, d))
    # one head's OV block doubled: ||2B - B||_F = ||B||_F, against the final checkpoint's OV norm ||B||_F, so V = 1
    v_doubled = MS.ov_distance(2.0 * ov, ov, ov)
    n_pos = N_A_DOCS * N_POSITIONS
    X = rng.normal(size=(n_pos, d))
    Y = rng.normal(size=(n_pos, d))
    m_independent = MS.cka_distance(X, Y)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    m_rotated = MS.cka_distance(X, 3.0 * (X @ Q))
    return {"P_disjoint": (p_disjoint, ("approx", 1.0, 1e-6)), "V_doubled": (v_doubled, ("approx", 1.0, 1e-6)),
            "M_independent": (m_independent, ("at_least", 0.5)),
            "M_rotated_scaled": (m_rotated, ("approx", 0.0, 1e-5))}


def g2_duplicate_control() -> float:
    """G2's duplicate-head control: a synthetic 4-head layer where heads 0 and 1 have identical rows; the matched
    cost must be 0 whichever zero-cost assignment the solver returns."""
    cost = np.array([[0.0, 0.0, 0.5, 0.9],
                     [0.0, 0.0, 0.5, 0.9],
                     [0.5, 0.5, 0.0, 0.9],
                     [0.9, 0.9, 0.9, 0.0]])
    return MS.matched_cost_with_duplicates(cost)


# ================================================================ G3, the per-prediction rules

def g3_quarter(floor: float, compared: float) -> bool:
    """211.1, 211.2, 211.3, 211.8: unmeasured when the floor is at least a quarter of the compared value."""
    return floor >= G3_FRACTION * abs(compared)


def g3_four_times(floor: float, denominator: float) -> bool:
    """211.4 to 211.7: unmeasured when a denominator is at or below four times the floor."""
    return bool(denominator <= G3_FACTOR * floor)            # a Python bool, so `a or b` is the disjunction


# ================================================================ the data each prediction reads
#
# One text set (design: Uncertainty). Every per-document array has documents first, and a resample is a row of
# document counts c [B, n_docs] (counts_of(idx)); the point estimate is c = 1 for every document. Inside a resample
# everything is recomputed from the documents: means are counts-weighted, matched distances are refit on the resampled
# mean matrix, and M is pooled CKA over the resample (cka_from_counts, precomputed per draw as M_boot). An earlier
# version bootstrapped over single documents (np.ravel(idx)) and took quantiles of per-document ratios; every fixture
# was constant across documents, so no test could tell the two apart (found 2026-11-13).
#
# Pythia: data["runs"][r] for r in PYTHIA_RUNS:
#   "P_within" [n_docs, L, n_steps]    head-mean within-run P(s, final); steps are PYTHIA_STEPS, final entry 0
#   "V"        [L, H, n_steps]         V(s, final) per head (no document sample; final entry 0)
#   "M_point"  [L, n_steps], "M_boot" [B, L, n_steps]   M(s, final) over all documents and per draw
#   "cross"    {step: [n_docs, L, H, H]}  P between this run's head i and R0's head j (steps 0, 10000, 143000)
# R0 also: "P_thread" [n_docs, L] (head-mean P(1000, 10000)) and "between_heads" [L] (mean P between different heads
# of a layer in R0's final model). data["floor"] = {"P": [L], "M": [L], "V": [L]} (R0's final against bf16 itself).
# OLMo: data["P"][pair] [n_docs, L] head-mean P for ("mid", "instruct"), ("3T", "4T"), ("21B", "4T"), ("step0", "4T");
#   data["M_point"][pair] [L] and data["M_boot"][pair] [B, L] for ("mid", "instruct") and ("3T", "4T");
#   data["floor"] = {"P": [L], "M": [L]} (4T against bf16 itself).

STEP_INDEX = {s: i for i, s in enumerate(PYTHIA_STEPS)}


def _n_layers(data: dict) -> int:
    """Layers in the data (the loops read the data; score_all checks it against the registered model's 12)."""
    return int(np.asarray(data["runs"]["R0"]["P_within"]).shape[1])


def _n_olmo_layers(olmo: dict) -> int:
    return int(np.asarray(olmo["P"][("3T", "4T")]).shape[1])


def weighted_mean(x: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The counts-weighted mean over documents (the first axis of x) for every draw: [B, ...]."""
    c = np.atleast_2d(np.asarray(counts, np.float64))
    x = np.asarray(x, np.float64)
    return np.tensordot(c, x, axes=(1, 0)) / c.sum(axis=1).reshape((-1,) + (1,) * (x.ndim - 1))


def matched_per_draw(pair_docs: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The matched cross-run P of one layer for every draw, the assignment refit on the draw's mean matrix:
    pair_docs [n_docs, H, H] -> [B]."""
    means = weighted_mean(pair_docs, counts)
    return np.array([MS.matched_distance(m)[0] for m in means])


def point_counts(n_docs: int) -> np.ndarray:
    return np.ones((1, n_docs))


def _hi(values) -> float:
    return float(np.quantile(np.asarray(values, np.float64), HI))


def _lo(values) -> float:
    return float(np.quantile(np.asarray(values, np.float64), LO))


def _times(f_rows) -> list:
    """settling_time per draw from f rows [B, n_steps] (None-able), as floats with censored as inf and unmeasured as
    None."""
    return [MS.settling_time(PYTHIA_STEPS, list(row)) for row in f_rows]


def _fraction_rows(num: np.ndarray, den: np.ndarray, floor: float) -> list:
    """f per draw and checkpoint: num [B, n_steps], den [B]; a draw whose denominator is at or below four times the
    floor is unmeasured in every checkpoint (design: Measures)."""
    return [[MS.settling_fraction(float(v), float(d), floor) for v in row] for row, d in zip(num, den)]


# ================================================================ the predictions

def pred_211_1(data: dict, floor: dict, counts: np.ndarray) -> dict:
    """211.1: per W run b, s* is the first checkpoint at which the layer's mean within-run P(s, final) is below its
    matched cross-run P at 143000 to R0 by more than the floor, and stays below. The layer counts for b when the 97.5th
    percentile of s* over resamples is at or before 16000 (censored as later); counts when it counts for both W runs.
    Unmeasured when the P floor is at least a quarter of the point matched cross-run P."""
    fl = floor["P"]
    n_docs = np.asarray(data["runs"]["R0"]["P_within"]).shape[0]
    pt = point_counts(n_docs)
    good, rows = 0, {}
    per_run = {}
    for b in WEIGHT_RUNS:
        run = data["runs"][b]
        w_pt, w_bt = weighted_mean(run["P_within"], pt)[0], weighted_mean(run["P_within"], counts)  # [L, S], [B, L, S]
        cross = run["cross"][FINAL]
        per_run[b] = (w_pt, w_bt, cross)
    for L in range(_n_layers(data)):
        runs = {}
        for b in WEIGHT_RUNS:
            w_pt, w_bt, cross = per_run[b]
            c_pt = matched_per_draw(cross[:, L], pt)[0]
            c_bt = matched_per_draw(cross[:, L], counts)
            s_pt = MS.crossing_step(PYTHIA_STEPS, w_pt[L], c_pt, fl[L])
            boot = [as_late(MS.crossing_step(PYTHIA_STEPS, w_bt[i, L], c_bt[i], fl[L])) for i in range(len(c_bt))]
            unmeasured = g3_quarter(fl[L], c_pt)
            counts_b = (not unmeasured) and _hi(boot) <= STEP_16000
            runs[b] = {"s_star": s_pt, "cross_final": c_pt, "ci": [_lo(boot), _hi(boot)], "unmeasured": unmeasured,
                       "counts": bool(counts_b)}
        n = sum(1 for r in runs.values() if r["counts"])
        rows[L] = {"runs": runs, "counts": n == len(WEIGHT_RUNS), "n": n}
        good += int(n == len(WEIGHT_RUNS))
    return {"prediction": "211.1", "layers_counting": good, "bar": 9, "pass": good >= 9, "per_layer": rows}


def pred_211_2(data: dict, floor: dict, counts: np.ndarray) -> dict:
    """211.2: per W run k, the median over D of the matched distance at 143000 minus W_k's matched distance, recomputed
    per resample; a layer counts when both differences have a 97.5th percentile below 0 and a point value below minus
    the floor. Unmeasured when the P floor is at least a quarter of either W run's point matched distance."""
    fl = floor["P"]
    n_docs = np.asarray(data["runs"]["R0"]["P_within"]).shape[0]
    pt = point_counts(n_docs)
    good, rows = 0, {}
    for L in range(_n_layers(data)):
        d_pt = np.median([matched_per_draw(data["runs"][b]["cross"][FINAL][:, L], pt)[0] for b in DATA_RUNS])
        d_bt = np.median(np.stack([matched_per_draw(data["runs"][b]["cross"][FINAL][:, L], counts)
                                   for b in DATA_RUNS]), axis=0)
        runs = {}
        for k in WEIGHT_RUNS:
            w_pt = matched_per_draw(data["runs"][k]["cross"][FINAL][:, L], pt)[0]
            w_bt = matched_per_draw(data["runs"][k]["cross"][FINAL][:, L], counts)
            point = float(d_pt - w_pt)
            boot = d_bt - w_bt
            unmeasured = g3_quarter(fl[L], w_pt)
            ok = (not unmeasured) and _hi(boot) < 0 and point < -fl[L]
            runs[k] = {"difference": point, "ci": [_lo(boot), _hi(boot)], "unmeasured": unmeasured, "counts": bool(ok)}
        n = sum(1 for r in runs.values() if r["counts"])
        rows[L] = {"runs": runs, "counts": n == len(WEIGHT_RUNS), "n": n}
        good += int(n == len(WEIGHT_RUNS))
    return {"prediction": "211.2", "layers_counting": good, "bar": 9, "pass": good >= 9, "per_layer": rows}


def pred_211_3(data: dict, floor: dict, counts: np.ndarray, idx: np.ndarray | None = None) -> dict:
    """211.3: a head is same-index compatible when its point gap g_i (over all documents) is at most the P floor;
    the interval of g_i is printed beside it. A layer's 12 heads count as not compatible when the P floor is at least a
    quarter of the mean P between different heads of that layer in R0's final model. Bar: 108 of 144 in each D run.

    Floor policy (BUILD-NOTES 8, frozen 2026-11-13 before any data, Alden #13): the compatibility tolerance for run b
    is the SMALLER of the two compared finals' P floors (b's and R0's), because a larger tolerance passes more heads
    under an at-most rule; the six-final maximum is printed beside it as a sensitivity column and never decides. The
    discrimination gate keeps the six-final maximum, where a larger floor is stricter."""
    fl = floor["P"]
    per_run = floor["per_run"]
    rows, ok = {}, True
    for b in DATA_RUNS:
        cross = np.asarray(data["runs"][b]["cross"][FINAL], np.float64)          # [n_docs, L, H, H]
        n_layers, n_heads = cross.shape[1], cross.shape[-1]
        n, unique, ambiguous, per_layer, n_max = 0, 0, 0, {}, 0
        for L in range(n_layers):
            if g3_quarter(fl[L], data["runs"]["R0"]["between_heads"][L]):
                per_layer[L] = {"compatible": 0, "unique": 0, "ambiguous": n_heads, "unmeasured": True}
                continue
            tol = min(float(per_run[b]["P"][L]), float(per_run["R0"]["P"][L]))
            c = MS.same_index_counts(cross[:, L].mean(axis=0), tol)
            c_max = MS.same_index_counts(cross[:, L].mean(axis=0), fl[L])
            per_layer[L] = {k: c[k] for k in ("compatible", "unique", "ambiguous")} | {
                "unmeasured": False, "tolerance": tol, "compatible_at_max_floor": c_max["compatible"]}
            n_max += c_max["compatible"]
            if idx is not None:
                per_layer[L]["gap_interval"] = MS.same_index_gap_interval(cross[:, L], idx).tolist()
            n += c["compatible"]
            unique += c["unique"]
            ambiguous += c["ambiguous"]
        rows[b] = {"n_compatible": n, "of": n_layers * n_heads, "unique": unique, "ambiguous": ambiguous,
                   "pass": n >= 108, "n_compatible_at_max_floor": n_max, "per_layer": per_layer}
        ok = ok and n >= 108
    return {"prediction": "211.3", "bar": 108, "pass": ok, "per_run": rows}


def _t_p(run: dict, c: np.ndarray, floor: float, L: int) -> list:
    w = weighted_mean(run["P_within"], c)[:, L]                                   # [B, S]
    return _times(_fraction_rows(w, w[:, 0], floor))


def pred_211_4(data: dict, floor: dict, counts: np.ndarray) -> dict:
    """211.4: per run, the layer counts when t_P < t_M holds at the point estimate and in at least ceil(0.975 B) of
    the B resamples, t_P and t_M recomputed in the same draws (M pooled per draw). A draw where either is unmeasured or
    the rule fails counts as a failure. Unmeasured when either point denominator X(step 0, final) is at or below four
    times its floor. A layer counts when it counts for at least 5 of the 6 runs."""
    good, rows = 0, {}
    n_docs = np.asarray(data["runs"]["R0"]["P_within"]).shape[0]
    pt = point_counts(n_docs)
    B = np.atleast_2d(counts).shape[0]
    for L in range(_n_layers(data)):
        runs = {}
        for r in PYTHIA_RUNS:
            run = data["runs"][r]
            m_pt = np.asarray(run["M_point"], np.float64)[L]
            m_bt = np.asarray(run["M_boot"], np.float64)[:B, L]
            if m_bt.shape[0] != B:
                raise ValueError(f"{r}: M_boot has {m_bt.shape[0]} draws, the counts have {B}")
            tp_pt = _t_p(run, pt, floor["P"][L], L)[0]
            tm_pt = _times(_fraction_rows(m_pt[None], m_pt[None, 0], floor["M"][L]))[0]
            tp_bt = _t_p(run, counts, floor["P"][L], L)
            tm_bt = _times(_fraction_rows(m_bt, m_bt[:, 0], floor["M"][L]))
            p0 = float(weighted_mean(run["P_within"], pt)[0, L, 0])
            unmeasured = g3_four_times(floor["P"][L], p0) or g3_four_times(floor["M"][L], float(m_pt[0]))
            holds = sum(1 for a, b in zip(tp_bt, tm_bt) if MS.earlier(a, b))
            ok = (not unmeasured) and MS.earlier(tp_pt, tm_pt) and holds >= threshold(B)
            runs[r] = {"t_P": tp_pt, "t_M": tm_pt, "holds_in": holds, "of": B, "unmeasured": unmeasured,
                       "counts": bool(ok)}
        n = sum(1 for v in runs.values() if v["counts"])
        rows[L] = {"runs": runs, "n": n, "counts": n >= ALL_RUNS_BAR}
        good += int(n >= ALL_RUNS_BAR)
    return {"prediction": "211.4", "layers_counting": good, "bar": 9, "pass": good >= 9, "per_layer": rows}


def pred_211_5(data: dict, floor: dict, counts: np.ndarray) -> dict:
    """211.5: as 211.4 with t_V, the settling time of the ratio of head means of V(s, final), in place of t_M. V has no
    document sample, so t_V is one value and the resamples vary t_P only (design: Uncertainty)."""
    good, rows = 0, {}
    n_docs = np.asarray(data["runs"]["R0"]["P_within"]).shape[0]
    pt = point_counts(n_docs)
    B = np.atleast_2d(counts).shape[0]
    for L in range(_n_layers(data)):
        runs = {}
        for r in PYTHIA_RUNS:
            run = data["runs"][r]
            v = np.asarray(run["V"], np.float64)[L].mean(axis=0)                     # [S]: the ratio of head means
            tv = _times(_fraction_rows(v[None], v[None, 0], floor["V"][L]))[0]
            tp_pt = _t_p(run, pt, floor["P"][L], L)[0]
            tp_bt = _t_p(run, counts, floor["P"][L], L)
            p0 = float(weighted_mean(run["P_within"], pt)[0, L, 0])
            unmeasured = g3_four_times(floor["P"][L], p0) or g3_four_times(floor["V"][L], float(v[0]))
            holds = sum(1 for a in tp_bt if MS.earlier(a, tv))
            ok = (not unmeasured) and MS.earlier(tp_pt, tv) and holds >= threshold(B)
            runs[r] = {"t_P": tp_pt, "t_V": tv, "holds_in": holds, "of": B, "unmeasured": unmeasured,
                       "counts": bool(ok)}
        n = sum(1 for v in runs.values() if v["counts"])
        rows[L] = {"runs": runs, "n": n, "counts": n >= ALL_RUNS_BAR}
        good += int(n >= ALL_RUNS_BAR)
    return {"prediction": "211.5", "layers_counting": good, "bar": 9, "pass": good >= 9, "per_layer": rows}


def pred_211_6(olmo: dict, floor: dict, counts: np.ndarray) -> dict:
    """211.6: per draw, the P ratio (counts-weighted means of P(mid, instruct) over P(3T, 4T)) minus the M ratio (pooled
    M of the same draw); the layer counts when the 97.5th percentile is below 0. Unmeasured when the point P(3T, 4T) or
    M(3T, 4T) is at or below four times its floor, or its 2.5th percentile over draws is at or below the floor."""
    n_docs = np.asarray(olmo["P"][("3T", "4T")]).shape[0]
    pt = point_counts(n_docs)
    B = np.atleast_2d(counts).shape[0]
    good, rows = 0, {}
    pm_pt, pl_pt = weighted_mean(olmo["P"][("mid", "instruct")], pt)[0], weighted_mean(olmo["P"][("3T", "4T")], pt)[0]
    pm_bt, pl_bt = weighted_mean(olmo["P"][("mid", "instruct")], counts), weighted_mean(olmo["P"][("3T", "4T")], counts)
    mm_pt, ml_pt = np.asarray(olmo["M_point"][("mid", "instruct")]), np.asarray(olmo["M_point"][("3T", "4T")])
    mm_bt = np.asarray(olmo["M_boot"][("mid", "instruct")])[:B]
    ml_bt = np.asarray(olmo["M_boot"][("3T", "4T")])[:B]
    for L in range(_n_olmo_layers(olmo)):
        fp, fm = floor["P"][L], floor["M"][L]
        unmeasured = (g3_four_times(fp, pl_pt[L]) or g3_four_times(fm, ml_pt[L])
                      or _lo(pl_bt[:, L]) <= fp or _lo(ml_bt[:, L]) <= fm)
        with np.errstate(divide="ignore", invalid="ignore"):
            diff = pm_bt[:, L] / pl_bt[:, L] - mm_bt[:, L] / ml_bt[:, L]
        hi = _hi(diff) if np.all(np.isfinite(diff)) else np.inf
        ok = (not unmeasured) and hi < 0
        rows[L] = {"ratio_P": float(pm_pt[L] / pl_pt[L]), "ratio_M": float(mm_pt[L] / ml_pt[L]),
                   "difference_ci": [_lo(diff) if np.isfinite(hi) else None, hi], "unmeasured": bool(unmeasured),
                   "counts": bool(ok)}
        fi = floor.get("instruct")
        if fi is not None:
            # reported only (design: Noise floor; Alden #13 asked that it be shown, not used): the mid-training-end
            # floor beside the numerator's point values; no verdict reads it
            rows[L]["numerator_P_mid_instruct"] = float(pm_pt[L])
            rows[L]["numerator_M_mid_instruct"] = float(mm_pt[L])
            rows[L]["instruct_floor"] = {"P": float(fi["P"][L]), "M": float(fi["M"][L])}
        good += int(ok)
    return {"prediction": "211.6", "layers_counting": good, "bar": 12, "pass": good >= 12, "per_layer": rows}


def pred_211_7(olmo: dict, floor: dict, counts: np.ndarray) -> dict:
    """211.7: f_P at 21B = mean P(21B, 4T) / mean P(step 0, 4T) per draw; the layer counts when the 97.5th percentile
    is at most 0.5. Unmeasured when the point P(step 0, 4T) is at or below four times the floor."""
    n_docs = np.asarray(olmo["P"][("step0", "4T")]).shape[0]
    pt = point_counts(n_docs)
    a_pt, z_pt = weighted_mean(olmo["P"][("21B", "4T")], pt)[0], weighted_mean(olmo["P"][("step0", "4T")], pt)[0]
    a_bt, z_bt = weighted_mean(olmo["P"][("21B", "4T")], counts), weighted_mean(olmo["P"][("step0", "4T")], counts)
    good, rows = 0, {}
    for L in range(_n_olmo_layers(olmo)):
        unmeasured = g3_four_times(floor["P"][L], z_pt[L])
        boot = a_bt[:, L] / z_bt[:, L]
        ok = (not unmeasured) and _hi(boot) <= 0.5
        rows[L] = {"f_P_21B": float(a_pt[L] / z_pt[L]), "ci": [_lo(boot), _hi(boot)], "unmeasured": bool(unmeasured),
                   "counts": bool(ok)}
        good += int(ok)
    return {"prediction": "211.7", "layers_counting": good, "bar": 12, "pass": good >= 12, "per_layer": rows}


def pred_211_8(data: dict, floor: dict, counts: np.ndarray) -> dict:
    """211.8: per W run, R0's within-run P(1000, 10000) minus the matched cross-run P at 10000, both recomputed per
    draw; the layer counts when both differences have a 97.5th percentile below 0 and a point value below minus the
    floor. Unmeasured when the P floor is at least a quarter of either point cross-run P.

    Floor policy (BUILD-NOTES 8, frozen 2026-11-13, Alden #13): the compared checkpoints are step 10000, not finals,
    so the floor for W run b is the LARGER of R0's and b's step-10000 P floors (both rules here are stricter with a
    larger floor)."""
    thread = floor["thread"]
    th = np.asarray(data["runs"]["R0"]["P_thread"], np.float64)
    n_docs = th.shape[0]
    pt = point_counts(n_docs)
    th_pt, th_bt = weighted_mean(th, pt)[0], weighted_mean(th, counts)
    good, rows = 0, {}
    for L in range(_n_layers(data)):
        runs = {}
        for b in WEIGHT_RUNS:
            cross = data["runs"][b]["cross"][THREAD][:, L]
            c_pt = matched_per_draw(cross, pt)[0]
            c_bt = matched_per_draw(cross, counts)
            point = float(th_pt[L] - c_pt)
            boot = th_bt[:, L] - c_bt
            f = max(float(thread["R0"]["P"][L]), float(thread[b]["P"][L]))
            unmeasured = g3_quarter(f, c_pt)
            ok = (not unmeasured) and _hi(boot) < 0 and point < -f
            runs[b] = {"difference": point, "ci": [_lo(boot), _hi(boot)], "floor": f, "unmeasured": unmeasured,
                       "counts": bool(ok)}
        n = sum(1 for r in runs.values() if r["counts"])
        rows[L] = {"runs": runs, "n": n, "counts": n == len(WEIGHT_RUNS)}
        good += int(n == len(WEIGHT_RUNS))
    return {"prediction": "211.8", "layers_counting": good, "bar": 9, "pass": good >= 9, "per_layer": rows}


# ================================================================ reported beside the verdicts, never predicted on
# Design, Reading rules: "P0 beside P for 211.1 to 211.3 and 211.8, and the cosine measure and greedy match beside
# every cross-run value. Per-head tables are written beside the per-layer ones." Point values over all documents.

def _nanmean(x, axis=None):
    x = np.asarray(x, np.float64)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(x)
        n = ok.sum(axis=axis)
        return np.where(n > 0, np.where(ok, x, 0.0).sum(axis=axis) / np.where(n > 0, n, 1), np.nan)


def _f(x):
    """A finite float, or None (NaN means no P0 there)."""
    x = float(x)
    return x if np.isfinite(x) else None


def _cross_point(run: dict, step: int, L: int, n_docs: int) -> dict:
    """One run's cross-run values against R0 at one step and layer, every document: P matched (the registered
    Hungarian assignment) and greedy (each head of the run takes its nearest R0 head, repeats allowed); the cosine of
    the matrices flattened over the documents, on the P-matched pairs, Hungarian on cosine and greedy on cosine
    (Bali et al.'s own); P0 on the P-matched pairs, with the pairs that have no P0 counted."""
    pt = point_counts(n_docs)
    m = np.asarray(run["cross"][step][:, L], np.float64).mean(axis=0)            # [H_run, H_R0]
    mp, pairs = MS.matched_distance(m)
    gp, _ = MS.greedy_distance(m.T)                                            # rows R0, columns the run's heads
    out = {"P_matched": mp, "P_greedy": gp, "pairs": pairs}
    if f"cross_cosd" in run and step in run["cross_cosd"]:
        cm = MS.cos_from_parts(run["cross_cosd"][step][:, L], run["cross_cosx"][step][:, L],
                               run["cross_cosy"][step][:, L], pt)[0]
        out.update({"cos_on_P_pairs": float(np.mean([cm[i, j] for i, j in pairs])),
                    "cos_matched": MS.matched_distance(cm)[0], "cos_greedy": MS.greedy_distance(cm.T)[0]})
    if f"cross_P0s" in run and step in run["cross_P0s"]:
        p0, dropped = MS.p0_from_parts(run["cross_P0s"][step][:, L], run["cross_P0k"][step][:, L],
                                       int(run["P0t"]), pt)
        vals = [p0[0][i, j] for i, j in pairs]
        out.update({"P0_on_P_pairs": _f(_nanmean(vals)), "P0_pairs_without": int(sum(1 for v in vals if not np.isfinite(v))),
                    "P0_dropped": float(sum(dropped[0][i, j] for i, j in pairs)),
                    "P0_queries": float(len(pairs) * n_docs * int(run["P0t"]))})
    return out


def _within_p0(run: dict, n_docs: int):
    """A run's within-run P0(s, final), head mean over the heads that have P0: ([L, S] with NaN, dropped [L, S], total
    [L, S])."""
    p0, dropped = MS.p0_from_parts(run["P0s_within"], run["P0k_within"], int(run["P0t"]), point_counts(n_docs))
    p0, dropped = p0[0], dropped[0]                                            # [L, H, S]
    total = n_docs * int(run["P0t"]) * np.ones_like(dropped)
    return _nanmean(p0, axis=1), dropped.sum(axis=1), total.sum(axis=1)


def p0_beside(data: dict, floor: dict) -> dict:
    """P0 beside P for 211.1, 211.2, 211.3 and 211.8: each prediction's per-layer quantity recomputed with P0 (its own
    floor, the P-matched pairs), point values, with the dropped queries printed (design: Noise floor)."""
    runs = data["runs"]
    if "P0s_within" not in runs["R0"]:
        return {"available": False, "reason": "this distance file has no P0 parts"}
    fl0 = np.asarray(floor.get("P0", [np.nan] * _n_layers(data)), np.float64)
    n_docs = np.asarray(runs["R0"]["P_within"]).shape[0]
    nL = _n_layers(data)
    within = {r: _within_p0(runs[r], n_docs) for r in runs}
    cross = {r: {step: {L: _cross_point(runs[r], step, L, n_docs) for L in range(nL)} for step in (THREAD, FINAL)}
             for r in runs if r != "R0"}
    out = {"available": True, "floor_P0": [_f(x) for x in fl0], "211.1": {}, "211.2": {}, "211.3": {}, "211.8": {},
           "dropped_within": {r: [float(x) for x in within[r][1].sum(axis=1)] for r in runs},
           "queries_within": {r: [float(x) for x in within[r][2].sum(axis=1)] for r in runs}}
    for L in range(nL):
        row = {}
        for b in WEIGHT_RUNS:
            c = cross[b][FINAL][L]["P0_on_P_pairs"]
            w = within[b][0][L]
            ok = c is not None and np.isfinite(w).all() and np.isfinite(fl0[L])
            row[b] = {"s_star_P0": MS.crossing_step(PYTHIA_STEPS, w, c, fl0[L]) if ok else "no P0",
                      "cross_final_P0": c}
        out["211.1"][L] = row
        d_vals = [cross[d][FINAL][L]["P0_on_P_pairs"] for d in DATA_RUNS]
        d_med = float(np.median(d_vals)) if all(v is not None for v in d_vals) else None
        out["211.2"][L] = {k: (None if d_med is None or cross[k][FINAL][L]["P0_on_P_pairs"] is None
                               else d_med - cross[k][FINAL][L]["P0_on_P_pairs"]) for k in WEIGHT_RUNS}
        p0_th, _ = MS.p0_from_parts(runs["R0"]["P0s_thread"], runs["R0"]["P0k_thread"], int(runs["R0"]["P0t"]),
                                    point_counts(n_docs))
        th = _nanmean(p0_th[0][L])
        out["211.8"][L] = {k: (None if not np.isfinite(th) or cross[k][THREAD][L]["P0_on_P_pairs"] is None
                               else float(th) - cross[k][THREAD][L]["P0_on_P_pairs"]) for k in WEIGHT_RUNS}
    for b in DATA_RUNS:
        n, no_p0 = 0, 0
        for L in range(nL):
            p0, _ = MS.p0_from_parts(runs[b]["cross_P0s"][FINAL][:, L], runs[b]["cross_P0k"][FINAL][:, L],
                                     int(runs[b]["P0t"]), point_counts(n_docs))
            m = p0[0]
            for i in range(m.shape[0]):
                if not np.isfinite(m[i]).all() or not np.isfinite(fl0[L]):
                    no_p0 += 1
                    continue
                n += int(MS.same_index_compatible(m[i], fl0[L], i)["same_index_compatible"])
        out["211.3"][b] = {"n_compatible_P0": n, "heads_without_P0": no_p0}
    return out


def cross_run_table(data: dict) -> dict:
    """Every cross-run value with the cosine measure and greedy match beside it: every run against R0 at every
    cross step on disk, per layer (design: Reading rules)."""
    runs = data["runs"]
    n_docs = np.asarray(runs["R0"]["P_within"]).shape[0]
    return {r: {int(step): {L: {k: v for k, v in _cross_point(runs[r], step, L, n_docs).items()}
                            for L in range(_n_layers(data))} for step in sorted(runs[r]["cross"])}
            for r in runs if r != "R0"}


def per_head_tables(data: dict) -> dict:
    """Per head (design: Reading rules, "Per-head tables are written beside the per-layer ones"): every run's
    within-run P(s, final) and settling fraction P(s, final) / P(0, final) per head and checkpoint, and at step 143000
    each run's Hungarian assignment to R0's heads with each pair's P."""
    runs = data["runs"]
    out = {"steps": [int(s) for s in PYTHIA_STEPS], "within": {}, "assignment_final": {}}
    for r, run in runs.items():
        if "P_within_heads" not in run:
            continue
        ph = np.asarray(run["P_within_heads"], np.float64).mean(axis=0)            # [L, H, S]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(ph[..., :1] > 0, ph / np.where(ph[..., :1] > 0, ph[..., :1], 1.0), np.nan)
        out["within"][r] = {"P": ph.tolist(), "fraction": frac.tolist()}
    for r, run in runs.items():
        if r == "R0" or FINAL not in run["cross"]:
            continue
        m = np.asarray(run["cross"][FINAL], np.float64).mean(axis=0)               # [L, H, H]
        out["assignment_final"][r] = {L: [[i, j, float(m[L][i, j])] for i, j in MS.matched_distance(m[L])[1]]
                                      for L in range(m.shape[0])}
    return out


def beside(s: dict) -> dict:
    return {"P0": p0_beside(s["pythia"], s["pythia_floor"]), "cross_run": cross_run_table(s["pythia"]),
            "per_head": per_head_tables(s["pythia"])}


PREDICTIONS = {"211.1": pred_211_1, "211.2": pred_211_2, "211.3": pred_211_3, "211.4": pred_211_4,
               "211.5": pred_211_5, "211.6": pred_211_6, "211.7": pred_211_7, "211.8": pred_211_8}
OLMO_PREDICTIONS = ("211.6", "211.7")
G1_D_STOPS = ("211.2", "211.3")
G1_W_STOPS = ("211.1", "211.2", "211.8")


def check_registered_shape(s: dict) -> None:
    """The registered models: Pythia-160m, 12 layers of 12 heads; OLMo 2 1B, 16 layers. Data of any other shape is
    refused, so a mis-assembled file cannot be scored against the registered bars."""
    py = s["pythia"]
    nl = _n_layers(py)
    nh = np.asarray(py["runs"]["D1"]["cross"][FINAL]).shape[-1]
    if (nl, nh) != (N_LAYERS, N_HEADS):
        raise ValueError(f"Pythia data has {nl} layers of {nh} heads; the registered model has {N_LAYERS} of {N_HEADS}")
    if s.get("olmo") is not None and _n_olmo_layers(s["olmo"]) != N_OLMO_LAYERS:
        raise ValueError(f"OLMo data has {_n_olmo_layers(s['olmo'])} layers; the registered model has {N_OLMO_LAYERS}")


def score_all(gates: dict, set_a: dict, set_b: dict, idx_a: np.ndarray, idx_b: np.ndarray,
              registered_shape: bool = True) -> dict:
    """Every verdict. set_a and set_b each hold {"pythia": data, "pythia_floor": floor, "olmo": data or None,
    "olmo_floor": floor}; idx_a and idx_b are the draws ([B, n_docs] of document IDs, seed 211), the same draws every
    M_boot was computed in. A failed G1 D half withholds 211.2 and 211.3, its W half 211.1, 211.2 and 211.8 (and the
    headline); a failed G2 withholds all; the rest are printed."""
    if registered_shape:
        check_registered_shape(set_a)
        check_registered_shape(set_b)
    out = {"gates": gates, "verdicts": {}}
    g2_failed = not gates.get("G2", {}).get("pass", False)
    d_failed = not gates.get("G1", {}).get("d_pass", False)
    w_failed = not gates.get("G1", {}).get("w_pass", False)

    def run_one(name, fn, s, idx):
        n_docs = np.asarray(idx).shape[1]
        counts = MS.counts_of(idx, n_docs)
        if name in OLMO_PREDICTIONS:
            return fn(s["olmo"], s["olmo_floor"], counts)
        if name == "211.3":
            return fn(s["pythia"], s["pythia_floor"], counts, idx=idx)
        return fn(s["pythia"], s["pythia_floor"], counts)

    for name, fn in PREDICTIONS.items():
        withheld = g2_failed or (d_failed and name in G1_D_STOPS) or (w_failed and name in G1_W_STOPS)
        if name in OLMO_PREDICTIONS and (set_a.get("olmo") is None or set_b.get("olmo") is None):
            out["verdicts"][name] = {"withheld": True, "verdict": "withheld", "reason": "no OLMo data"}
            continue
        a = run_one(name, fn, set_a, idx_a)
        b = run_one(name, fn, set_b, idx_b)
        out["verdicts"][name] = {"set_A": a, "set_B": b, "withheld": withheld,
                                 "verdict": "withheld" if withheld else ("held" if a["pass"] else "not held"),
                                 "set_B_agrees": bool(a["pass"] == b["pass"])}
    reverses = any(abs(v["set_A"]["layers_counting"] - v["set_B"]["layers_counting"]) > 1
                   for name, v in out["verdicts"].items() if name in ("211.1", "211.2") and "set_A" in v)
    out["headline_withdrawn"] = bool(g2_failed or d_failed or w_failed or reverses)
    out["beside"] = {"A": beside(set_a), "B": beside(set_b)}
    return out


# ================================================================ reading the distance files

def read_records(out: Path, text_set: str) -> dict:
    """Every complete record of one text set on disk: {checkpoint key: npz arrays}."""
    recs = {}
    for path in sorted((Path(out) / "records").glob(f"*.{text_set}.npz")):
        key = path.name[: -len(f".{text_set}.npz")]
        z = np.load(path, allow_pickle=False)
        if "complete" in z.files and bool(z["complete"]):
            recs[key] = z
    return recs


def report_lines(result: dict) -> list[str]:
    """The printed report (design: Reading rules): each verdict on set A with set B beside it, then every per-layer
    value behind it with its interval and the floor's unmeasured flag. No verdict is led with a mean over layers."""
    lines = []
    g = result["gates"]
    g1, g2 = g.get("G1", {}), g.get("G2", {})
    lines.append(f"G1 seed split: D half {'pass' if g1.get('d_pass') else 'FAIL'}, "
                 f"W half {'pass' if g1.get('w_pass') else 'FAIL'}")
    lines.append(f"G2 measures: {'pass' if g2.get('pass') else 'FAIL'}")
    lines.append(f"headline: {'WITHDRAWN' if result['headline_withdrawn'] else 'stands'}")
    for name, v in result["verdicts"].items():
        if "set_A" not in v:
            lines.append(f"{name}: {v['verdict']} ({v.get('reason', '')})")
            continue
        a, b = v["set_A"], v["set_B"]
        if name == "211.3":
            counts_a = {r: x["n_compatible"] for r, x in a["per_run"].items()}
            counts_b = {r: x["n_compatible"] for r, x in b["per_run"].items()}
            lines.append(f"{name}: {v['verdict']} | set A compatible {counts_a} (bar {a['bar']}) | set B {counts_b}")
            for r, x in a["per_run"].items():
                lines.append(f"  {r}: unique {x['unique']}, ambiguous {x['ambiguous']}")
            continue
        lines.append(f"{name}: {v['verdict']} | set A {a['layers_counting']} layers (bar {a['bar']}) | set B "
                     f"{b['layers_counting']}{'' if v['set_B_agrees'] else ' (disagrees)'}")
        for L, row in a["per_layer"].items():
            lines.append(f"  layer {L}: " + json.dumps(row, default=float, sort_keys=True)[:400])
    lines += beside_lines(result.get("beside", {}).get("A"))
    return lines


def _fmt(x) -> str:
    return "none" if x is None else (x if isinstance(x, str) else f"{float(x):.4g}")


def beside_lines(b: dict | None) -> list[str]:
    """P0 beside 211.1 to 211.3 and 211.8, then every cross-run value with the cosine measure and greedy match, set A;
    per-head tables are in the JSON (beside.A.per_head and beside.B.per_head)."""
    if not b:
        return []
    lines = ["P0 beside P (set A; not predicted on):"]
    p0 = b["P0"]
    if not p0.get("available"):
        lines.append(f"  {p0.get('reason')}")
    else:
        lines.append(f"  P0 floor per layer: {[_fmt(x) for x in p0['floor_P0']]}")
        for L, row in p0["211.1"].items():
            lines.append(f"  211.1 layer {L}: " + ", ".join(f"{r} s*_P0 {_fmt(x['s_star_P0'])} (cross {_fmt(x['cross_final_P0'])})"
                                                            for r, x in row.items()))
        for L, row in p0["211.2"].items():
            lines.append(f"  211.2 layer {L}: D median minus " + ", ".join(f"{r} {_fmt(x)}" for r, x in row.items()))
        lines.append("  211.3: " + ", ".join(f"{r} {x['n_compatible_P0']} compatible, {x['heads_without_P0']} heads "
                                             f"without P0" for r, x in p0["211.3"].items()))
        for L, row in p0["211.8"].items():
            lines.append(f"  211.8 layer {L}: thread minus cross " + ", ".join(f"{r} {_fmt(x)}" for r, x in row.items()))
        lines.append("  dropped P0 queries within-run, per run: " + ", ".join(
            f"{r} {int(sum(p0['dropped_within'][r]))} of {int(sum(p0['queries_within'][r]))}" for r in p0["dropped_within"]))
    lines.append("cross-run to R0 (set A; P matched is the registered value; the rest are reported, not predicted on):")
    for r, steps in b["cross_run"].items():
        for step, layers in steps.items():
            for L, c in layers.items():
                lines.append(f"  {r} step {step} layer {L}: P matched {_fmt(c['P_matched'])}, greedy {_fmt(c['P_greedy'])}"
                             f" | cosine on P pairs {_fmt(c.get('cos_on_P_pairs'))}, matched {_fmt(c.get('cos_matched'))},"
                             f" greedy {_fmt(c.get('cos_greedy'))} | P0 on P pairs {_fmt(c.get('P0_on_P_pairs'))}"
                             f" ({c.get('P0_pairs_without', 0)} pairs without)")
    lines.append("per-head tables: score.json, beside.A.per_head and beside.B.per_head")
    return lines


def main(argv=None):
    import json as _json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/part211")
    ap.add_argument("--json", default=None, help="write the full result here (default <out>/score.json)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    import distances211 as DIST
    gates_path = out / "gates.json"
    if not gates_path.is_file():
        raise SystemExit(f"no {gates_path}: run run211.py --stage gates first")
    paths = {t: out / "distances" / f"{t}.npz" for t in ("A", "B")}
    for t, pth in paths.items():
        if not pth.is_file():
            raise SystemExit(f"no {pth}: run distances211.py first")
    gates = _json.loads(gates_path.read_text(encoding="utf-8"))
    sa, sb = DIST.load(paths["A"]), DIST.load(paths["B"])
    result = score_all(gates, sa, sb, sa["idx"], sb["idx"])
    for line in report_lines(result):
        print(line)
    dest = Path(a.json) if a.json else out / "score.json"
    dest.write_text(_json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"written {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
