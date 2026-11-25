#!/usr/bin/env python3
"""100% Python 3.10+ stdlib-only deterministic scorer for OLMo 3 7B Stage C2
(`OLMO3-TRANSPORT-C2-DESIGN.md` Registered v4, public `main` @ `ecd49b28`).

Requires ZERO third-party packages (no numpy, scipy, or torch) and uses `math.fsum`
for all float reductions so any clean Python 3.10+ machine reproduces `c2_scorecard.json`
and `C2-SCORECARD.md` byte-for-byte via `/usr/bin/python3 -I` + `cmp`.

Fail-closed guarantees:
  - Hard-refuses smoke/truncated/schema-mismatched receipts (ValueError).
  - Refuses empty or non-finite inputs across ALL retained traces (including all `all_24_dlogits`).
  - Enforces the registered BCa containment invariant (`CI_low <= point_estimate <= CI_high`) fail-closed.
  - Evaluates Global Gate `C2.0` (`C2.0.1` 19 shams + liveness, `C2.0.2` telescoping, `C2.0.3` exact C1-count
    ceilings `4/44` & `1/13/6` + diagnostic `fde_gt_075`, `C2.0.4` SHA-256 provenance); if `C2.0` fails,
    confirmatory verdicts are strictly `BLOCKED_BY_C2.0`.
  - Runs the unified `m = 7` reject-the-null Holm-Bonferroni family (`α/(8-k)`) separately from the
    3 registered gates (`CAL-C2.1`, `EQ-C2.2`, `EQ-C2.4c`) and secondary descriptive readouts (`C2.4b`, etc.).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable

EXPECTED_R_LENS_SHA256 = "ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955"
EXPECTED_C2_CONSTANTS_SHA256 = "583117f50cfceeb5cddcfa239c0efa0c32846dd69f6756540cd3bbba5cec5b65"
EXPECTED_C2_BANK_SHA256 = "7124dd565331fd455d6d91d8fe845ba8783c2b0d366f6d11eb533947cb09401d"
EXPECTED_C2_DESIGN_SHA256 = "3764dfee6baf9c7387fc6325eee92c4ca2fdbca5c76ee5b440e0f4f12169d8fc"

PRIMARY_C21_LAYERS = [18, 24, 27]
SWEEP_LAYERS = [18, 21, 24, 27, 30]
PRENORM_LAYERS = [21, 24, 27]
C21_ARMS = ("R_mf", "R_ridge_raw", "R_ridge_centered", "R_ridge_shrink", "logit_lens")

PERM_SEED = 2026112501
PERM_N_FLIPS_24 = 100_000
BCA_SEED = 2026112502
BCA_N_RESAMPLES = 10_000


# ---------------------------------------------------------------------------
# Pure-stdlib I/O, hashing, and fail-closed statistics (`math.fsum`)
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_maybe_gz(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_nonempty_finite(xs: list[float], name: str) -> None:
    if not xs:
        raise ValueError(f"{name}() received empty input")
    for idx, v in enumerate(xs):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ValueError(f"{name}() received non-finite value {v!r} at index {idx}")


def mean(xs: list[float]) -> float:
    _require_nonempty_finite(xs, "mean")
    return math.fsum(xs) / len(xs)


def median(xs: list[float]) -> float:
    _require_nonempty_finite(xs, "median")
    return float(statistics.median(xs))


def quantile(xs: list[float], q: float) -> float:
    _require_nonempty_finite(xs, "quantile")
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"quantile q out of range: {q}")
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (pos - lo) * (s[hi] - s[lo]))


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    if p <= 0.0:
        return -38.0
    if p >= 1.0:
        return 38.0
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def exact_sign_flip_p_one_sided(diffs: list[float]) -> float:
    """Exact 2^N one-sided sign-flip enumeration (for N=12 -> 4,096 or N=16 -> 65,536)."""
    _require_nonempty_finite(diffs, "exact_sign_flip_p_one_sided")
    n = len(diffs)
    if n > 18:
        raise ValueError(f"exact_sign_flip_p_one_sided N={n} too large for exact enumeration")
    obs_sum = math.fsum(diffs)
    total = 1 << n
    # Gray-code / recursive sum enumeration for speed + exactness
    sums = [0.0]
    for d in diffs:
        sums = [s + d for s in sums] + [s - d for s in sums]
    ge = sum(1 for s in sums if s >= obs_sum - 1e-12 * max(1.0, abs(obs_sum)))
    return ge / total


def mc_sign_flip_p_one_sided(diffs: list[float], n_flips: int = PERM_N_FLIPS_24, seed: int = PERM_SEED) -> float:
    """Deterministic 100,000-flip one-sided sign-flip test for N=24 targets."""
    _require_nonempty_finite(diffs, "mc_sign_flip_p_one_sided")
    obs_sum = math.fsum(diffs)
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_flips):
        s = math.fsum(d if rng.getrandbits(1) else -d for d in diffs)
        if s >= obs_sum:
            ge += 1
    return (1 + ge) / (1 + n_flips)


def bca_cluster_ci(
    n_clusters: int,
    stat_fn: Callable[[list[int]], float],
    alphas: tuple[float, ...] = (0.05, 0.95),
    n_boot: int = BCA_N_RESAMPLES,
    seed: int = BCA_SEED,
    enforce_containment_90: bool = True,
) -> dict[str, float]:
    """Deterministic BCa cluster bootstrap with fail-closed 90% CI containment invariant."""
    if n_clusters < 2:
        raise ValueError("bca_cluster_ci requires n_clusters >= 2")
    full_idx = list(range(n_clusters))
    theta_hat = float(stat_fn(full_idx))
    if not math.isfinite(theta_hat):
        raise ValueError(f"Non-finite point estimate in bca_cluster_ci: {theta_hat!r}")

    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n_clusters) for _ in range(n_clusters)]
        v = float(stat_fn(idxs))
        if not math.isfinite(v):
            raise ValueError(f"Non-finite bootstrap replicate in bca_cluster_ci: {v!r}")
        boots.append(v)
    boots.sort()

    if boots[0] == boots[-1] and abs(boots[0] - theta_hat) <= 1e-12 * max(1.0, abs(theta_hat)):
        out = {"point": theta_hat}
        for a in alphas:
            out[f"q_{a}"] = theta_hat
        return out

    less = sum(1 for v in boots if v < theta_hat)
    eq = sum(1 for v in boots if v == theta_hat)
    prop = min(max((less + 0.5 * eq) / n_boot, 1.0 / (2.0 * n_boot)), 1.0 - 1.0 / (2.0 * n_boot))
    z0 = norm_ppf(prop)

    jacks = [float(stat_fn([j for j in range(n_clusters) if j != i])) for i in range(n_clusters)]
    j_bar = mean(jacks)
    num = math.fsum((j_bar - jv) ** 3 for jv in jacks)
    den_sq = math.fsum((j_bar - jv) ** 2 for jv in jacks)
    a_hat = (num / (6.0 * (den_sq ** 1.5))) if den_sq > 0.0 else 0.0

    out = {"point": theta_hat}
    for a in alphas:
        za = norm_ppf(a)
        denom = 1.0 - a_hat * (z0 + za)
        adj_a = a if abs(denom) < 1e-12 else norm_cdf(z0 + (z0 + za) / denom)
        adj_a = min(max(adj_a, 0.0), 1.0)
        out[f"q_{a}"] = quantile(boots, adj_a)

    if enforce_containment_90 and (0.05 in alphas) and (0.95 in alphas):
        lo, hi = out["q_0.05"], out["q_0.95"]
        tol = 1e-12 * max(1.0, abs(theta_hat))
        if lo > theta_hat + tol or hi < theta_hat - tol:
            raise ValueError(f"FAIL-CLOSED BCa CONTAINMENT VIOLATED: [{lo:.6f}, {hi:.6f}] vs point {theta_hat:.6f}")
    return out


def holm_bonferroni_stepdown(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Exact m-test Holm-Bonferroni step-down procedure with explicit stop rule."""
    items = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    res: dict[str, dict[str, Any]] = {}
    stopped = False
    for rank, (name, p) in enumerate(items, start=1):
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"Invalid p-value for {name}: {p!r}")
        thresh = alpha / (m - rank + 1)
        if not stopped and p <= thresh:
            rej = True
        else:
            stopped = True
            rej = False
        res[name] = {"rank": rank, "p": round(p, 8), "alpha_thresh": round(thresh, 6), "rejected": rej}
    return res


def top_right_singular_vector_and_rank1_share(mat: list[list[float]], n_iter: int = 250) -> tuple[list[float], float]:
    """Pure-stdlib exact power iteration on A = M^T M (24x24 PSD) returning (v1_right, s1^2 / ||M||_F^2)."""
    m_rows = len(mat)
    n_cols = len(mat[0])
    frob_sq = math.fsum(v * v for row in mat for v in row)
    if frob_sq <= 1e-18:
        return [0.0] * n_cols, 0.0

    # Form A = M^T M [n_cols x n_cols]
    A = [
        [math.fsum(mat[r][i] * mat[r][j] for r in range(m_rows)) for j in range(n_cols)]
        for i in range(n_cols)
    ]
    # Deterministic non-degenerate initial vector
    v = [1.0 + 0.01 * (i + 1) for i in range(n_cols)]
    nrm = math.sqrt(math.fsum(x * x for x in v))
    v = [x / nrm for x in v]
    lam = 0.0
    for _ in range(n_iter):
        w = [math.fsum(A[i][j] * v[j] for j in range(n_cols)) for i in range(n_cols)]
        nrm = math.sqrt(math.fsum(x * x for x in w))
        if nrm <= 1e-18:
            break
        v = [x / nrm for x in w]
        lam = nrm
    return v, min(1.0, max(0.0, lam / frob_sq))


def ols_slope_and_r2(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Pure-stdlib OLS linear regression slope and R^2 of y on x."""
    _require_nonempty_finite(xs, "ols_x")
    _require_nonempty_finite(ys, "ols_y")
    mx, my = mean(xs), mean(ys)
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    syy = math.fsum((y - my) ** 2 for y in ys)
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 1e-18 or syy <= 1e-18:
        return 0.0, 0.0
    slope = sxy / sxx
    r = sxy / math.sqrt(sxx * syy)
    return slope, r * r


# ---------------------------------------------------------------------------
# Recursive finiteness audit across receipt
# ---------------------------------------------------------------------------
def audit_finiteness(obj: Any, path: str = "receipt") -> list[str]:
    bad: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad.extend(audit_finiteness(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            bad.extend(audit_finiteness(v, f"{path}[{idx}]"))
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(f"{path}={obj!r}")
    return bad


# ---------------------------------------------------------------------------
# Main Scorer
# ---------------------------------------------------------------------------
def score_stage_c2(
    receipt: dict[str, Any],
    c2_constants: dict[str, Any],
    *,
    design_sha256: str | None = None,
    constants_sha256: str | None = None,
    strict_provenance: bool = True,
    n_boot: int = BCA_N_RESAMPLES,
    n_flips_24: int = PERM_N_FLIPS_24,
) -> dict[str, Any]:
    """Score a complete Stage C2 v4 receipt strictly per OLMO3-TRANSPORT-C2-DESIGN.md (v4)."""
    if receipt.get("schema_version") != "olmo3-stage-c2-receipt-v4":
        raise ValueError(f"Unsupported receipt schema_version: {receipt.get('schema_version')!r}")
    meta = receipt.get("meta", {})
    if meta.get("smoke", False):
        raise ValueError("FAIL-CLOSED: Refusing to score a smoke/truncated Stage C2 receipt.")

    targets_24 = c2_constants["targets_24"]
    t_strs = [t["string"] for t in targets_24]
    strata = [t["stratum"] for t in targets_24]
    splits = [t["split"] for t in targets_24]
    prim_idxs = [i for i, sp in enumerate(splits) if sp == "primary_16"]
    ho_idxs = [i for i, sp in enumerate(splits) if sp == "heldout_8"]
    c22_t_idxs = list(c2_constants["c22_target_indices_12"])
    c22_t_strs = [t_strs[i] for i in c22_t_idxs]
    c22_open_strs = [t_strs[i] for i in c22_t_idxs if strata[i] != "function"]
    c22_func_strs = [t_strs[i] for i in c22_t_idxs if strata[i] == "function"]
    prompts_8 = c2_constants["prompts_8_c21"]
    triples_12 = c2_constants["c22_stem_pairs_12"]

    # ------------------------------------------------------------------
    # GLOBAL GATE C2.0
    # ------------------------------------------------------------------
    # C2.0.1: 13 write shams + 6 patch shams == 0.0 exact; every active arm > 0.0
    write_shams = receipt["shams"]["write_13"]
    patch_shams = receipt["shams"]["patch_6"]
    req_w = set(c2_constants["planned_budget"]["write_shams_list"])
    req_p = set(c2_constants["planned_budget"]["patch_shams_list"])
    shams_present_ok = (set(write_shams.keys()) == req_w) and (set(patch_shams.keys()) == req_p)
    max_sham_w = max(abs(float(v)) for v in write_shams.values()) if write_shams else float("inf")
    max_sham_p = max(abs(float(v)) for v in patch_shams.values()) if patch_shams else float("inf")
    active_mins = receipt.get("active_arm_min_max_abs_dlogit", {})
    # Expected active arm families: 15 (C2.1) + 18 (C2.2) + 10 (C2.3) + 18 (C2.4) = 61
    liveness_ok = (len(active_mins) >= 61) and all(float(v) > 0.0 for v in active_mins.values())
    c2_0_1_pass = shams_present_ok and (max_sham_w == 0.0) and (max_sham_p == 0.0) and liveness_ok

    # C2.0.2: Finiteness & 4-term telescoping error (p95 <= 0.08, p99.9 <= 0.15)
    nonfinite_paths = audit_finiteness(receipt)
    tele_errs: list[float] = []
    # Collect telescoping errors across Block 1, Block 2 (uncapped), Block 2b (logit_lens)
    for L in PRIMARY_C21_LAYERS:
        for aname in C21_ARMS:
            for ts in t_strs:
                for rec in receipt["block1_C2_1"][str(L)]["arms"][aname][ts]:
                    tele_errs.append(abs(float(rec["telescoping_err"])))
    for cname in c2_constants["planned_budget"]["c22_conditions_list"]:
        for pidx in range(12):
            for ts in c22_t_strs:
                for aname in ("R_mf", "dm0"):
                    rec = receipt["block2_C2_2"]["traces"][cname][str(pidx)][ts][aname]
                    if not rec.get("capped", False):
                        tele_errs.append(abs(float(rec["telescoping_err"])))
    for L in SWEEP_LAYERS:
        for ts in t_strs:
            for rec in receipt["block2b_C2_3"][str(L)]["logit_lens"][ts]:
                tele_errs.append(abs(float(rec["telescoping_err"])))
    tele_p95 = quantile(tele_errs, 0.95)
    tele_p999 = quantile(tele_errs, 0.999)
    c2_0_2_pass = (len(nonfinite_paths) == 0) and (tele_p95 <= 0.08) and (tele_p999 <= 0.15)

    # C2.0.3: Exact C1-count linearity ceilings (Regime-N N=2,880: 0 cap, <=4 >20%, <=44 >12%;
    # Regime-D C2.2 unpatched N=864: 0 cap, <=1 >20%, <=13 >12%, fde_p95 <= 0.30, <=6 >0.45; + diagnostic fde_gt_075)
    g203 = c2_constants["c2_0_3_named_gate_populations"]
    reg_n_n = reg_n_cap = reg_n_20 = reg_n_12 = 0
    for L in PRIMARY_C21_LAYERS:
        for aname in C21_ARMS:
            for ts in t_strs:
                for rec in receipt["block1_C2_1"][str(L)]["arms"][aname][ts]:
                    reg_n_n += 1
                    if rec.get("capped", False):
                        reg_n_cap += 1
                        continue
                    rc, ri = float(rec["rms_clean"]), float(rec["rms_int"])
                    shrink = abs(ri - rc) / max(rc, 1e-12)
                    if shrink > 0.20:
                        reg_n_20 += 1
                    if shrink > 0.12:
                        reg_n_12 += 1

    reg_d_n = reg_d_cap = reg_d_20 = reg_d_12 = reg_d_fde45 = reg_d_fde75 = 0
    fde_errs: list[float] = []
    for cname in ("unpatched_LongBare", "unpatched_ChatML", "unpatched_ShortBare"):
        for pidx in range(12):
            for ts in c22_t_strs:
                for aname in ("R_mf", "dm0"):
                    rec = receipt["block2_C2_2"]["traces"][cname][str(pidx)][ts][aname]
                    reg_d_n += 1
                    if rec.get("capped", False):
                        reg_d_cap += 1
                        continue
                    rc, ri = float(rec["rms_clean"]), float(rec["rms_int"])
                    shrink = abs(ri - rc) / max(rc, 1e-12)
                    if shrink > 0.20:
                        reg_d_20 += 1
                    if shrink > 0.12:
                        reg_d_12 += 1
                    fe = abs(float(rec["frozen_direct_effect"]) - 3.0)
                    fde_errs.append(fe)
                    if fe > 0.45:
                        reg_d_fde45 += 1
                    if fe > 0.75:
                        reg_d_fde75 += 1
    fde_p95 = quantile(fde_errs, 0.95) if fde_errs else float("inf")
    c2_0_3_pass = (
        (reg_n_n == g203["regime_N_C21_population_N"])
        and (reg_n_cap <= g203["regime_N_max_capped"])
        and (reg_n_20 <= g203["regime_N_max_rmsn_gt_20pct"])
        and (reg_n_12 <= g203["regime_N_max_rmsn_gt_12pct"])
        and (reg_d_n == g203["regime_D_C22_unpatched_population_N"])
        and (reg_d_cap <= g203["regime_D_max_capped"])
        and (reg_d_20 <= g203["regime_D_max_rmsn_gt_20pct"])
        and (reg_d_12 <= g203["regime_D_max_rmsn_gt_12pct"])
        and (fde_p95 <= g203["regime_D_fde_p95_max"])
        and (reg_d_fde45 <= g203["regime_D_max_fde_gt_045"])
    )

    # C2.0.4: Frozen SHA-256 Provenance
    ph = meta.get("provenance_hashes", {})
    c2_0_4_pass = (
        strict_provenance
        and (design_sha256 == EXPECTED_C2_DESIGN_SHA256)
        and (constants_sha256 == EXPECTED_C2_CONSTANTS_SHA256)
        and (ph.get("r_lens_sha256") == EXPECTED_R_LENS_SHA256)
        and (ph.get("c2_constants_sha256") == EXPECTED_C2_CONSTANTS_SHA256)
        and (ph.get("c2_bank_sha256") == EXPECTED_C2_BANK_SHA256)
        and (ph.get("c2_design_sha256") == EXPECTED_C2_DESIGN_SHA256)
    )

    c2_0_pass = c2_0_1_pass and c2_0_2_pass and c2_0_3_pass and c2_0_4_pass

    # Non-gating C0-colon anchor prediction
    anc_raw = receipt["c0_anchor_colon"]
    anc_pairs: dict[str, dict[str, Any]] = {}
    for pid, row in anc_raw.items():
        anc_pairs.setdefault(row["pair_id"], {})[row["ending"]] = row
    r24_all, r27_all, r24_bare, r27_bare = [], [], [], []
    for pair_id, ends in sorted(anc_pairs.items()):
        r24 = float(ends[":"]["pmn_L24"]) / max(float(ends["a"]["pmn_L24"]), 1e-12)
        r27 = float(ends[":"]["pmn_L27"]) / max(float(ends["a"]["pmn_L27"]), 1e-12)
        r24_all.append(r24)
        r27_all.append(r27)
        if ends["a"]["format"] == "bare":
            r24_bare.append(r24)
            r27_bare.append(r27)
    c0_anchor_pred = {
        "pooled_median_ratio_L24": round(median(r24_all), 4),
        "pooled_median_ratio_L27": round(median(r27_all), 4),
        "bare_only_median_ratio_L24": round(median(r24_bare), 4),
        "bare_only_median_ratio_L27": round(median(r27_bare), 4),
        "in_predicted_range_L24": 0.70 <= median(r24_all) <= 0.85,
        "in_predicted_range_L27": 0.65 <= median(r27_all) <= 0.82,
    }

    # ------------------------------------------------------------------
    # PILLAR 1: C2.1 On-Model Arm Comparison (Holm 1-4) & Gate CAL-C2.1
    # ------------------------------------------------------------------
    # Formal F1 definitions: diag_a(i, L), ho_cross_a(i, L) (strictly ABSOLUTE, epsilon=0.25 guard)
    diag_by_arm_i_L: dict[str, dict[int, dict[int, float]]] = {a: {i: {} for i in range(24)} for a in C21_ARMS}
    ho_cross_by_arm_i_L: dict[str, dict[int, dict[int, float]]] = {a: {i: {} for i in prim_idxs} for a in C21_ARMS}
    same_by_arm_i_L: dict[str, dict[int, dict[int, float]]] = {a: {i: {} for i in range(24)} for a in C21_ARMS}
    cross_all_by_arm_i_L: dict[str, dict[int, dict[int, float]]] = {a: {i: {} for i in range(24)} for a in C21_ARMS}
    guarded_sources_by_arm: dict[str, set[int]] = {a: set() for a in C21_ARMS}
    per_prompt_ho_cross: dict[str, dict[int, dict[int, float]]] = {a: {pi: {} for pi in range(8)} for a in C21_ARMS}

    # Also build prompt-averaged 24x24 signed matrices per (L, arm) for SVD / mu_U & CAL-C2.1 regression
    mat24_by_L_arm: dict[int, dict[str, list[list[float]]]] = {L: {} for L in PRIMARY_C21_LAYERS}
    mat24_scaled_by_L_arm: dict[int, dict[str, list[list[float]]]] = {L: {} for L in PRIMARY_C21_LAYERS}

    for L in PRIMARY_C21_LAYERS:
        dose_l = float(receipt["block1_C2_1"][str(L)]["dose"])
        for a in C21_ARMS:
            M_raw = [[0.0] * 24 for _ in range(24)]
            M_scaled = [[0.0] * 24 for _ in range(24)]
            for i in range(24):
                rows = receipt["block1_C2_1"][str(L)]["arms"][a][t_strs[i]]
                d_vals = [float(r["all_24_dlogits"][i]) for r in rows]
                diag_iL = mean(d_vals)
                diag_by_arm_i_L[a][i][L] = diag_iL
                denom = max(abs(diag_iL), 0.25)
                if abs(diag_iL) <= 0.25 and i in prim_idxs:
                    guarded_sources_by_arm[a].add(i)

                for j in range(24):
                    M_raw[i][j] = mean([float(r["all_24_dlogits"][j]) for r in rows])
                    M_scaled[i][j] = mean([float(r["all_24_dlogits"][j]) / max(dose_l / float(r["rms_clean"]), 1e-12) for r in rows])

                same_abs = mean([abs(float(r["all_24_dlogits"][j])) for r in rows for j in range(24) if j != i and strata[j] == strata[i]])
                cross_abs = mean([abs(float(r["all_24_dlogits"][j])) for r in rows for j in range(24) if strata[j] != strata[i]])
                same_by_arm_i_L[a][i][L] = same_abs / denom
                cross_all_by_arm_i_L[a][i][L] = cross_abs / denom

                if i in prim_idxs:
                    ho_abs = mean([abs(float(r["all_24_dlogits"][j])) for r in rows for j in ho_idxs if strata[j] != strata[i]])
                    ho_cross_by_arm_i_L[a][i][L] = ho_abs / denom
                    # Per-prompt breakdown (F6)
                    for pi, r in enumerate(rows):
                        p_ho = mean([abs(float(r["all_24_dlogits"][j])) for j in ho_idxs if strata[j] != strata[i]]) / max(abs(float(r["all_24_dlogits"][i])), 0.25)
                        per_prompt_ho_cross[a][pi].setdefault(i, []).append(p_ho)  # type: ignore[arg-type]

            mat24_by_L_arm[L][a] = M_raw
            mat24_scaled_by_L_arm[L][a] = M_scaled

    degenerate_diag = any(len(s) > 2 for s in guarded_sources_by_arm.values())

    # Equal-weight layer averages
    diag_arm: dict[str, list[float]] = {
        a: [mean([diag_by_arm_i_L[a][i][L] for L in PRIMARY_C21_LAYERS]) for i in range(24)] for a in C21_ARMS
    }
    ho_cross_arm: dict[str, list[float]] = {
        a: [max(mean([ho_cross_by_arm_i_L[a][i][L] for L in PRIMARY_C21_LAYERS]), 1e-6) for i in prim_idxs]
        for a in C21_ARMS
    }
    same_i_arm: dict[str, list[float]] = {
        a: [mean([same_by_arm_i_L[a][i][L] for L in PRIMARY_C21_LAYERS]) for i in range(24)] for a in C21_ARMS
    }
    cross_i_arm: dict[str, list[float]] = {
        a: [max(mean([cross_all_by_arm_i_L[a][i][L] for L in PRIMARY_C21_LAYERS]), 1e-6) for i in range(24)] for a in C21_ARMS
    }

    # Holm Tests 1-4
    x_ctr_raw = [math.log(ho_cross_arm["R_ridge_raw"][k] / ho_cross_arm["R_ridge_centered"][k]) for k in range(16)]
    x_shk_ctr = [math.log(ho_cross_arm["R_ridge_centered"][k] / ho_cross_arm["R_ridge_shrink"][k]) for k in range(16)]
    x_shk_ll = [math.log(ho_cross_arm["logit_lens"][k] / ho_cross_arm["R_ridge_shrink"][k]) for k in range(16)]
    x_gain_ret = [
        math.log(max(diag_arm["R_ridge_shrink"][i], 1e-6) / max(diag_arm["R_ridge_raw"][i], 1e-6)) - math.log(0.75)
        for i in range(24)
    ]

    p_c21_ctr_raw = exact_sign_flip_p_one_sided(x_ctr_raw)
    p_c21_shk_ctr = exact_sign_flip_p_one_sided(x_shk_ctr)
    p_c21_shk_ll = exact_sign_flip_p_one_sided(x_shk_ll)
    p_c21_gain_ret = mc_sign_flip_p_one_sided(x_gain_ret, n_flips=n_flips_24, seed=PERM_SEED)

    ci_ctr_raw = bca_cluster_ci(16, lambda idx: math.exp(mean([x_ctr_raw[k] for k in idx])), n_boot=n_boot, seed=BCA_SEED)
    ci_shk_ctr = bca_cluster_ci(16, lambda idx: math.exp(mean([x_shk_ctr[k] for k in idx])), n_boot=n_boot, seed=BCA_SEED + 1)
    ci_shk_ll = bca_cluster_ci(16, lambda idx: math.exp(mean([x_shk_ll[k] for k in idx])), n_boot=n_boot, seed=BCA_SEED + 2)
    ci_gain_ret = bca_cluster_ci(
        24,
        lambda idx: math.exp(mean([math.log(max(diag_arm["R_ridge_shrink"][i], 1e-6) / max(diag_arm["R_ridge_raw"][i], 1e-6)) for i in idx])),
        n_boot=n_boot, seed=BCA_SEED + 3,
    )
    ci_sc_ratio = bca_cluster_ci(
        24,
        lambda idx: (
            (mean([same_i_arm["R_ridge_shrink"][i] for i in idx]) / max(mean([cross_i_arm["R_ridge_shrink"][i] for i in idx]), 1e-9))
            / max(mean([same_i_arm["R_ridge_centered"][i] for i in idx]) / max(mean([cross_i_arm["R_ridge_centered"][i] for i in idx]), 1e-9), 1e-9)
        ),
        n_boot=n_boot, seed=BCA_SEED + 4,
    )

    # Gate CAL-C2.1 (F2): Absolute frozen-lens transfer + mu_U SVD mechanism
    fref = receipt.get("frozen_cal_reference", {})
    u_mu_n = [float(v) for v in fref.get("u_mu_proj_n", [0.0] * 24)]
    mu_cos_by_arm: dict[str, dict[str, float]] = {}
    rank1_by_arm: dict[str, dict[str, float]] = {}
    for a in ("R_mf", "R_ridge_raw", "R_ridge_centered", "R_ridge_shrink", "logit_lens"):
        c_list, r1_list = [], []
        for L in PRIMARY_C21_LAYERS:
            M = [row[:] for row in mat24_by_L_arm[L][a]]
            for i in range(24):
                M[i][i] = 0.0
            v1, r1 = top_right_singular_vector_and_rank1_share(M)
            c_mu = abs(math.fsum(x * y for x, y in zip(v1, u_mu_n)))
            c_list.append(c_mu)
            r1_list.append(r1)
        mu_cos_by_arm[a] = {"mean_3L": round(mean(c_list), 4), "per_L": [round(v, 4) for v in c_list]}
        rank1_by_arm[a] = {"mean_3L": round(mean(r1_list), 4), "per_L": [round(v, 4) for v in r1_list]}

    # Held-out 16x8x3L regression of scaled on-model shifts on frozen u_j . (R_out w_i) for R_ridge_raw
    ho_f_preds = fref.get("frozen_heldout_preds_16x8", {})
    reg_x, reg_y = [], []
    for L in PRIMARY_C21_LAYERS:
        fmat = ho_f_preds.get(str(L), {}).get("R_ridge_raw")
        if fmat is not None:
            for pi_k, i in enumerate(prim_idxs):
                for hj_k, j in enumerate(ho_idxs):
                    reg_x.append(float(fmat[pi_k][hj_k]))
                    reg_y.append(float(mat24_scaled_by_L_arm[L]["R_ridge_raw"][i][j]))
    cal_slope, cal_r2 = ols_slope_and_r2(reg_x, reg_y) if reg_x else (0.0, 0.0)
    ho_cross_raw_mean = mean(ho_cross_arm["R_ridge_raw"])
    frozen_diag_shk_3L = mean([float(c2_constants["layer_geometry"][str(L)]["frozen_lens_transfer_preds_24"]["R_ridge_shrink"]["diag_unit_gain_mean"]) for L in PRIMARY_C21_LAYERS])
    scaled_diag_shk_3L = mean([mat24_scaled_by_L_arm[L]["R_ridge_shrink"][i][i] for L in PRIMARY_C21_LAYERS for i in range(24)])

    cal_c21_pass = (
        (0.04 <= ho_cross_raw_mean <= 0.25)
        and (cal_r2 >= 0.40)
        and (0.45 <= cal_slope <= 1.60)
        and (scaled_diag_shk_3L >= 0.50 * frozen_diag_shk_3L)
        and (mu_cos_by_arm["R_ridge_raw"]["mean_3L"] >= 0.70)
        and (mu_cos_by_arm["R_ridge_shrink"]["mean_3L"] <= 0.45)
    )

    # Secondary 5-arm ordering & per-prompt breakdown (F6, F10)
    arm_cross_all_means = {a: round(mean(cross_i_arm[a]), 4) for a in C21_ARMS}
    five_arm_order_ok = (
        arm_cross_all_means["R_ridge_shrink"]
        < arm_cross_all_means["R_ridge_centered"]
        < arm_cross_all_means["logit_lens"]
        < arm_cross_all_means["R_ridge_raw"]
        < arm_cross_all_means["R_mf"]
    )
    per_prompt_ratios_f6 = []
    for pi in range(8):
        p_raw = mean([mean(per_prompt_ho_cross["R_ridge_raw"][pi][i]) for i in prim_idxs])
        p_ctr = mean([mean(per_prompt_ho_cross["R_ridge_centered"][pi][i]) for i in prim_idxs])
        p_shk = mean([mean(per_prompt_ho_cross["R_ridge_shrink"][pi][i]) for i in prim_idxs])
        per_prompt_ratios_f6.append({
            "prompt_id": prompts_8[pi]["id"],
            "raw_over_ctr": round(p_raw / max(p_ctr, 1e-6), 3),
            "ctr_over_shk": round(p_ctr / max(p_shk, 1e-6), 3),
        })

    # ------------------------------------------------------------------
    # PILLAR 2: C2.2 Matched-Length Pure ChatML-Template Causal Patching
    # ------------------------------------------------------------------
    c22_tr = receipt["block2_C2_2"]["traces"]
    pr_map = c2_constants["c22_l18_frozen_pred_ret"]
    pred_gap_mean_12 = mean([float(pr_map[ts]["pred_gap_at_direct3"]) for ts in c22_t_strs])
    pred_gap_mean_open = mean([float(pr_map[ts]["pred_gap_at_direct3"]) for ts in c22_open_strs])
    pred_gap_mean_func = mean([float(pr_map[ts]["pred_gap_at_direct3"]) for ts in c22_func_strs])

    def _pair_M(cname: str, pidx: int, subset_strs: list[str], pred_denom: float) -> float:
        gaps = []
        for ts in subset_strs:
            r_rmf = c22_tr[cname][str(pidx)][ts]["R_mf"]
            r_dm0 = c22_tr[cname][str(pidx)][ts]["dm0"]
            if r_rmf.get("capped", False) or r_dm0.get("capped", False):
                raise ValueError(f"Capped trace in C2.2 {cname} pair {pidx} target {ts}")
            gaps.append(float(r_rmf["exact_dlogit"]) - float(r_dm0["exact_dlogit"]))
        return mean(gaps) / pred_denom

    M_cond: dict[str, list[float]] = {
        cname: [_pair_M(cname, pidx, c22_t_strs, pred_gap_mean_12) for pidx in range(12)]
        for cname in c2_constants["planned_budget"]["c22_conditions_list"]
    }
    gap_pairs = [M_cond["unpatched_ChatML"][i] - M_cond["unpatched_LongBare"][i] for i in range(12)]
    len_pairs = [M_cond["unpatched_LongBare"][i] - M_cond["unpatched_ShortBare"][i] for i in range(12)]
    ci_gap = bca_cluster_ci(12, lambda idx: mean([gap_pairs[i] for i in idx]), n_boot=n_boot, seed=BCA_SEED + 10)
    ci_len = bca_cluster_ci(12, lambda idx: mean([len_pairs[i] for i in idx]), n_boot=n_boot, seed=BCA_SEED + 11)
    g_gap = ci_gap["point"]
    s_sign = 1.0 if g_gap >= 0.0 else -1.0
    gap_gate_pass = (abs(g_gap) >= 0.30) and ((ci_gap["q_0.05"] > 0.0) or (ci_gap["q_0.95"] < 0.0))
    len_gap_present = (abs(ci_len["point"]) >= 0.30) and ((ci_len["q_0.05"] > 0.0) or (ci_len["q_0.95"] < 0.0))

    # Exact denominator-free paired contrasts d_i(0.50) & pooled closure BCa CIs
    def _closure_stat(deltas: list[float], gaps: list[float], idx: list[int]) -> float:
        g_sub = mean([gaps[i] for i in idx])
        if abs(g_sub) < 1e-9:
            return 0.0
        return mean([deltas[i] for i in idx]) / g_sub

    del_b2c = [M_cond["unpatched_ChatML"][i] - M_cond["patch_LongBare_to_ChatML_MLP_L19_31"][i] for i in range(12)]
    del_c2b = [M_cond["patch_ChatML_to_LongBare_MLP_L19_31"][i] - M_cond["unpatched_LongBare"][i] for i in range(12)]
    d50_b2c = [s_sign * (del_b2c[i] - 0.50 * gap_pairs[i]) for i in range(12)]
    d50_c2b = [s_sign * (del_c2b[i] - 0.50 * gap_pairs[i]) for i in range(12)]

    p_c22_b2c = exact_sign_flip_p_one_sided(d50_b2c)
    p_c22_c2b = exact_sign_flip_p_one_sided(d50_c2b)
    ci_b2c = bca_cluster_ci(12, lambda idx: _closure_stat(del_b2c, gap_pairs, idx), n_boot=n_boot, seed=BCA_SEED + 12)
    ci_c2b = bca_cluster_ci(12, lambda idx: _closure_stat(del_c2b, gap_pairs, idx), n_boot=n_boot, seed=BCA_SEED + 13)

    # Gate EQ-C2.2: 4 negative controls + numeric open-vs-func shape & clean-KL bar
    ctrl_specs = [
        ("ctrl_L4_16_MLP_LongBare_to_ChatML", 0.25, 0.40, 14),
        ("ctrl_rand_iso_same_site_L19_31", 0.25, 0.40, 15),
        ("ctrl_mismatch_prompt_MLP_L19_31", 0.35, 0.50, 16),
        ("ctrl_self_attenuated_L19_31", 0.25, 0.40, 17),
    ]
    eq_c22_ctrls: dict[str, Any] = {}
    eq_c22_all_ctrls_ok = True
    for cname, pt_max, ci_max, seed_off in ctrl_specs:
        del_c = [M_cond["unpatched_ChatML"][i] - M_cond[cname][i] for i in range(12)]
        ci_c = bca_cluster_ci(12, lambda idx, dc=del_c: _closure_stat(dc, gap_pairs, idx), n_boot=n_boot, seed=BCA_SEED + seed_off)
        ok_c = (abs(ci_c["point"]) <= pt_max) and (ci_c["q_0.95"] <= ci_max)
        eq_c22_all_ctrls_ok = eq_c22_all_ctrls_ok and ok_c
        eq_c22_ctrls[cname] = {"point": round(ci_c["point"], 4), "q05": round(ci_c["q_0.05"], 4), "q95": round(ci_c["q_0.95"], 4), "pass": ok_c}

    # Shape & clean-KL clause (5)
    g_open = mean([_pair_M("unpatched_ChatML", i, c22_open_strs, pred_gap_mean_open) - _pair_M("unpatched_LongBare", i, c22_open_strs, pred_gap_mean_open) for i in range(12)])
    g_func = mean([_pair_M("unpatched_ChatML", i, c22_func_strs, pred_gap_mean_func) - _pair_M("unpatched_LongBare", i, c22_func_strs, pred_gap_mean_func) for i in range(12)])
    cl_open = mean([_pair_M("unpatched_ChatML", i, c22_open_strs, pred_gap_mean_open) - _pair_M("patch_LongBare_to_ChatML_MLP_L19_31", i, c22_open_strs, pred_gap_mean_open) for i in range(12)]) / (g_open if abs(g_open) > 1e-9 else 1.0)
    cl_func = mean([_pair_M("unpatched_ChatML", i, c22_func_strs, pred_gap_mean_func) - _pair_M("patch_LongBare_to_ChatML_MLP_L19_31", i, c22_func_strs, pred_gap_mean_func) for i in range(12)]) / (g_func if abs(g_func) > 1e-9 else 1.0)
    unsteered_kls = [float(r["kl"]) for rows in receipt["block2_C2_2"]["unsteered_patched_kl"].values() for r in rows]
    med_unsteered_kl = median(unsteered_kls)
    shape_kl_ok = (cl_open >= 0.65) and (abs(cl_open - cl_func) <= 0.35) and (med_unsteered_kl <= 0.25)
    eq_c22_pass = eq_c22_all_ctrls_ok and shape_kl_ok

    # ------------------------------------------------------------------
    # PILLAR 3: C2.3 5-Layer Post-Erasure Sweep
    # ------------------------------------------------------------------
    c23_per_layer: dict[str, Any] = {}
    post_erasure_flags: list[tuple[int, bool]] = []
    for L in SWEEP_LAYERS:
        blk = receipt["block2b_C2_3"][str(L)]
        open_mlp = [float(r["sum_mlp"]) for i, ts in enumerate(t_strs) if strata[i] != "function" for r in blk["logit_lens"][ts]]
        func_mlp = [float(r["sum_mlp"]) for i, ts in enumerate(t_strs) if strata[i] == "function" for r in blk["logit_lens"][ts]]
        open_ret = [float(r["retention"]) for i, ts in enumerate(t_strs) if strata[i] != "function" for r in blk["logit_lens"][ts]]
        rand_ret = [float(r["retention"]) for r in blk["rand_iso"]]
        eo, ef, ro, rr = median(open_mlp), median(func_mlp), median(open_ret), median(rand_ret)
        pe = (eo >= -1.20) and (abs(eo - ef) <= 0.60) and (ro > rr)
        post_erasure_flags.append((L, pe))
        c23_per_layer[str(L)] = {
            "E_open": round(eo, 4), "E_func": round(ef, 4), "gap": round(eo - ef, 4),
            "ret_open": round(ro, 4), "ret_rand": round(rr, 4), "post_erasure": pe,
        }
    selected_band: list[int] | None = None
    for idx in range(len(post_erasure_flags) - 1):
        if post_erasure_flags[idx][1] and post_erasure_flags[idx + 1][1]:
            run = [post_erasure_flags[idx][0], post_erasure_flags[idx + 1][0]]
            k = idx + 2
            while k < len(post_erasure_flags) and post_erasure_flags[k][1]:
                run.append(post_erasure_flags[k][0])
                k += 1
            selected_band = run
            break
    c23_status = "LOCALIZED" if selected_band is not None else "C2.3_UNLOCALIZED"
    c23_c3_band = selected_band if selected_band is not None else [24, 27]

    # ------------------------------------------------------------------
    # PILLAR 4: C2.4 Masking (Holm 7), Gate EQ-C2.4c (TOST), & Secondary Readouts
    # ------------------------------------------------------------------
    d_blk = receipt["block3_C2_4"]["doses"]

    def _c24_target_mean(dose_str: str, arm_key: str, field: str = "exact_dlogit") -> list[float]:
        return [mean([float(r[field]) for r in d_blk[dose_str][arm_key][ts]]) for ts in t_strs]

    # Holm 7: C2.4a masked vs unmasked (pooled across 0.50 & 1.25 per target, exact matched pre-norm norm)
    dl_mask_pool = [0.5 * (a + b) for a, b in zip(_c24_target_mean("0.5", "L24.prenorm_ff_masked"), _c24_target_mean("1.25", "L24.prenorm_ff_masked"))]
    dl_unm_pool = [0.5 * (a + b) for a, b in zip(_c24_target_mean("0.5", "L24.prenorm_ff_unmasked"), _c24_target_mean("1.25", "L24.prenorm_ff_unmasked"))]
    x_c24a = [math.log(max(dl_mask_pool[i], 1e-6) / max(dl_unm_pool[i], 1e-6)) for i in range(24)]
    p_c24a = mc_sign_flip_p_one_sided(x_c24a, n_flips=n_flips_24, seed=PERM_SEED + 1)
    ci_c24a = bca_cluster_ci(24, lambda idx: math.exp(mean([x_c24a[i] for i in idx])), n_boot=n_boot, seed=BCA_SEED + 20)

    # Secondary C2.4a unmasked_postmatched diagnostic (F9: capped_rate == 0 precondition!)
    pm_capped = sum(1 for m in ("0.5", "1.25") for ts in t_strs for r in d_blk[m]["L24.prenorm_ff_unmasked_postmatched"][ts] if r.get("capped", False))
    if pm_capped == 0:
        dl_pm_pool = [0.5 * (a + b) for a, b in zip(_c24_target_mean("0.5", "L24.prenorm_ff_unmasked_postmatched"), _c24_target_mean("1.25", "L24.prenorm_ff_unmasked_postmatched"))]
        x_pm = [math.log(max(dl_mask_pool[i], 1e-6) / max(dl_pm_pool[i], 1e-6)) for i in range(24)]
        c24a_postmatched_diag = {"capped_count": 0, "geomean_masked_over_postmatched": round(math.exp(mean(x_pm)), 4), "precondition_ok": True}
    else:
        c24a_postmatched_diag = {"capped_count": pm_capped, "geomean_masked_over_postmatched": None, "precondition_ok": False}

    # Secondary descriptive C2.4b (F3 & F7): high-dose 1.25x & 0.50x estimation + packing contrast (both L2 norm & |dlogit|)
    c24b_desc: dict[str, Any] = {}
    for m in ("0.5", "1.25"):
        dl_m70 = _c24_target_mean(m, "L24.m70a30")
        dl_ff = _c24_target_mean(m, "L24.prenorm_ff_masked")
        x_m70 = [math.log(max(dl_m70[i], 1e-6) / max(dl_ff[i], 1e-6)) for i in range(24)]
        ci_m70 = bca_cluster_ci(24, lambda idx, xm=x_m70: math.exp(mean([xm[i] for i in idx])), n_boot=n_boot, seed=BCA_SEED + (21 if m == "0.5" else 22))
        l2_m70_r = _c24_target_mean(m, "L24.m70a30_rand_masked", "dlogit_l2_norm")
        l2_ff_r = _c24_target_mean(m, "L24.prenorm_ff_rand_masked", "dlogit_l2_norm")
        r_pack_l2 = math.exp(mean([math.log(max(l2_m70_r[i], 1e-6) / max(l2_ff_r[i], 1e-6)) for i in range(24)]))
        abs_m70_r = _c24_target_mean(m, "L24.m70a30_rand_masked", "abs_target_dlogit")
        abs_ff_r = _c24_target_mean(m, "L24.prenorm_ff_rand_masked", "abs_target_dlogit")
        r_pack_abs = math.exp(mean([math.log(max(abs_m70_r[i], 1e-6) / max(abs_ff_r[i], 1e-6)) for i in range(24)]))
        c24b_desc[f"dose_{m}x"] = {
            "R_target": round(ci_m70["point"], 4), "q05": round(ci_m70["q_0.05"], 4), "q95": round(ci_m70["q_0.95"], 4),
            "R_pack_rand_l2_norm": round(r_pack_l2, 4), "R_target_over_pack_l2": round(ci_m70["point"] / max(r_pack_l2, 1e-9), 4),
            "R_pack_rand_abs_target": round(r_pack_abs, 4), "R_target_over_pack_abs": round(ci_m70["point"] / max(r_pack_abs, 1e-9), 4),
        }

    # Gate EQ-C2.4c: K=3 Median Additivity TOST at 0.50x (plus 1.25x calibration)
    def _k3_rel_int(dose_str: str) -> tuple[list[float], int]:
        d21 = _c24_target_mean(dose_str, "L21.prenorm_ff_masked")
        d24 = _c24_target_mean(dose_str, "L24.prenorm_ff_masked")
        d27 = _c24_target_mean(dose_str, "L27.prenorm_ff_masked")
        dk3 = _c24_target_mean(dose_str, "K3_combined")
        rels, n_fl = [], 0
        for i in range(24):
            s = d21[i] + d24[i] + d27[i]
            if abs(s) < 0.1:
                n_fl += 1
            rels.append((dk3[i] - s) / max(abs(s), 0.1))
        return rels, n_fl

    rel_05, n_fl_05 = _k3_rel_int("0.5")
    rel_125, n_fl_125 = _k3_rel_int("1.25")
    ci_k3_05 = bca_cluster_ci(24, lambda idx: median([rel_05[i] for i in idx]), n_boot=n_boot, seed=BCA_SEED + 23)
    if n_fl_05 <= 2 and abs(ci_k3_05["point"]) <= 0.20 and ci_k3_05["q_0.05"] >= -0.30 and ci_k3_05["q_0.95"] <= 0.30:
        eq_c24c_verdict = "ADDITIVE"
        c3_k3_action = "LICENSE_K3_SIMULTANEOUS"
    elif ci_k3_05["q_0.05"] > 0.30 or ci_k3_05["q_0.95"] < -0.30:
        eq_c24c_verdict = "NON_ADDITIVE"
        c3_k3_action = "RESTRICT_TO_SINGLE_LAYER_L24"
    else:
        eq_c24c_verdict = "INCONCLUSIVE"
        c3_k3_action = "REQUIRE_1_DOSE_C3_PILOT"

    # ------------------------------------------------------------------
    # UNIFIED 7-TEST HOLM-BONFERRONI FAMILY & FINAL VERDICTS
    # ------------------------------------------------------------------
    holm_pvals = {
        "p_C2.1_ctr_vs_raw": p_c21_ctr_raw,
        "p_C2.1_shk_vs_ctr": p_c21_shk_ctr,
        "p_C2.1_shk_vs_ll": p_c21_shk_ll,
        "p_C2.1_gain_ret": p_c21_gain_ret,
        "p_C2.2_bare_into_chat": p_c22_b2c,
        "p_C2.2_chat_into_bare": p_c22_c2b,
        "p_C2.4a_mask_vs_unmask": p_c24a,
    }
    holm = holm_bonferroni_stepdown(holm_pvals, alpha=0.05)

    # Bar checks (evaluated cleanly; gated on C2.0)
    bar_1 = holm["p_C2.1_ctr_vs_raw"]["rejected"] and (ci_ctr_raw["point"] >= 2.0) and (ci_ctr_raw["q_0.05"] > 1.50)
    bar_2 = holm["p_C2.1_shk_vs_ctr"]["rejected"] and (ci_shk_ctr["point"] >= 1.15) and (ci_sc_ratio["point"] >= 1.15) and (ci_sc_ratio["q_0.05"] >= 1.05)
    bar_3 = holm["p_C2.1_shk_vs_ll"]["rejected"] and (ci_shk_ll["point"] >= 1.35) and (ci_shk_ll["q_0.05"] > 1.15)
    bar_4 = holm["p_C2.1_gain_ret"]["rejected"] and (ci_gain_ret["point"] >= 0.80) and (ci_gain_ret["q_0.05"] >= 0.75)
    bar_5 = gap_gate_pass and holm["p_C2.2_bare_into_chat"]["rejected"] and (0.75 <= ci_b2c["point"] <= 1.30) and (ci_b2c["q_0.05"] >= 0.55) and (ci_b2c["q_0.95"] <= 1.50)
    bar_6 = gap_gate_pass and holm["p_C2.2_chat_into_bare"]["rejected"] and (0.75 <= ci_c2b["point"] <= 1.30) and (ci_c2b["q_0.05"] >= 0.55) and (ci_c2b["q_0.95"] <= 1.50)
    bar_7 = holm["p_C2.4a_mask_vs_unmask"]["rejected"] and (ci_c24a["point"] >= 1.40) and (ci_c24a["q_0.05"] > 1.20)

    if not c2_0_pass:
        v_c21 = v_c22 = v_c23 = v_c24a = v_c24c = "BLOCKED_BY_C2.0"
    else:
        if degenerate_diag:
            v_c21 = "DEGENERATE_DIAG"
        elif bar_1 and bar_2 and bar_3 and bar_4:
            v_c21 = "PASS_CALIBRATED" if cal_c21_pass else "RELATIVE_ARM_WIN_ONLY"
        elif bar_1 and bar_3 and bar_4:
            v_c21 = "PARTIAL_CENTERING_AND_LL_WIN"
        else:
            v_c21 = "FAIL"

        if not gap_gate_pass:
            v_c22 = "LENGTH_DRIVEN_NOT_TEMPLATE" if len_gap_present else "NO_FORMAT_GAP_AT_MATCHED_LENGTH"
        elif bar_5 and bar_6 and eq_c22_pass:
            v_c22 = "PASS"
        elif (bar_5 or bar_6) and eq_c22_pass:
            v_c22 = "PARTIAL_UNIDIRECTIONAL"
        else:
            v_c22 = "FAIL"

        v_c23 = c23_status
        v_c24a = "PASS" if bar_7 else "FAIL"
        v_c24c = eq_c24c_verdict

    return {
        "schema_version": "olmo3-stage-c2-scorecard-v4",
        "confirmatory_verdicts": {
            "C2.0": "PASS" if c2_0_pass else "FAIL",
            "C2.1": v_c21,
            "C2.2": v_c22,
            "C2.3": v_c23,
            "C2.4a": v_c24a,
            "EQ-C2.4c": v_c24c,
        },
        "gate_C2_0": {
            "pass": c2_0_pass,
            "C2.0.1_shams_and_liveness": {
                "pass": c2_0_1_pass, "shams_present_ok": shams_present_ok,
                "max_write_sham": max_sham_w, "max_patch_sham": max_sham_p, "liveness_ok": liveness_ok,
            },
            "C2.0.2_finiteness_and_telescoping": {
                "pass": c2_0_2_pass, "nonfinite_count": len(nonfinite_paths),
                "tele_p95": round(tele_p95, 6), "tele_p999": round(tele_p999, 6),
            },
            "C2.0.3_linearity": {
                "pass": c2_0_3_pass,
                "regime_N": {"N": reg_n_n, "capped": reg_n_cap, "rmsn_gt_20": reg_n_20, "rmsn_gt_12": reg_n_12},
                "regime_D": {
                    "N": reg_d_n, "capped": reg_d_cap, "rmsn_gt_20": reg_d_20, "rmsn_gt_12": reg_d_12,
                    "fde_p95": round(fde_p95, 6), "fde_gt_045": reg_d_fde45, "fde_gt_075_diagnostic": reg_d_fde75,
                },
            },
            "C2.0.4_provenance": {"pass": c2_0_4_pass, "strict": strict_provenance, "receipt_hashes": ph},
        },
        "c0_anchor_colon_prediction_nongating": c0_anchor_pred,
        "holm_7_family": holm,
        "pillar_C2_1": {
            "degenerate_diag": degenerate_diag,
            "guarded_sources_count": {a: len(s) for a, s in guarded_sources_by_arm.items()},
            "ctr_vs_raw": {k: round(v, 4) for k, v in ci_ctr_raw.items()} | {"bar_pass": bar_1},
            "shk_vs_ctr": {k: round(v, 4) for k, v in ci_shk_ctr.items()} | {"same_over_cross_ratio": {k: round(v, 4) for k, v in ci_sc_ratio.items()}, "bar_pass": bar_2},
            "shk_vs_ll": {k: round(v, 4) for k, v in ci_shk_ll.items()} | {"bar_pass": bar_3},
            "gain_ret": {k: round(v, 4) for k, v in ci_gain_ret.items()} | {"bar_pass": bar_4},
            "gate_CAL_C2_1": {
                "pass": cal_c21_pass,
                "ho_cross_raw_mean": round(ho_cross_raw_mean, 4),
                "ols_slope_raw": round(cal_slope, 4),
                "ols_r2_raw": round(cal_r2, 4),
                "scaled_diag_shk_3L": round(scaled_diag_shk_3L, 4),
                "frozen_diag_shk_3L": round(frozen_diag_shk_3L, 4),
                "mu_u_cos": mu_cos_by_arm,
                "rank1_svd_share": rank1_by_arm,
            },
            "secondary_five_arm_ordering": {"means": arm_cross_all_means, "strict_order_ok": five_arm_order_ok},
            "secondary_per_prompt_breakdown_F6": per_prompt_ratios_f6,
        },
        "pillar_C2_2": {
            "gap_gate_pass": gap_gate_pass,
            "G_gap_chat_minus_long": {k: round(v, 4) for k, v in ci_gap.items()},
            "G_len_long_minus_short": {k: round(v, 4) for k, v in ci_len.items()},
            "bare_into_chat": {k: round(v, 4) for k, v in ci_b2c.items()} | {"bar_pass": bar_5},
            "chat_into_bare": {k: round(v, 4) for k, v in ci_c2b.items()} | {"bar_pass": bar_6},
            "gate_EQ_C2_2": {
                "pass": eq_c22_pass,
                "controls": eq_c22_ctrls,
                "closure_open": round(cl_open, 4),
                "closure_func": round(cl_func, 4),
                "median_unsteered_patched_kl": round(med_unsteered_kl, 4),
                "shape_and_kl_pass": shape_kl_ok,
            },
        },
        "pillar_C2_3": {"status": c23_status, "selected_c3_band": c23_c3_band, "per_layer": c23_per_layer},
        "pillar_C2_4": {
            "C2.4a_mask_vs_unmask": {k: round(v, 4) for k, v in ci_c24a.items()} | {"bar_pass": bar_7},
            "secondary_C2.4a_postmatched_diag": c24a_postmatched_diag,
            "secondary_C2.4b_m70a30_descriptive": c24b_desc,
            "gate_EQ_C2_4c": {
                "verdict": eq_c24c_verdict,
                "c3_action": c3_k3_action,
                "dose_0.5x": {k: round(v, 4) for k, v in ci_k3_05.items()} | {"n_floored": n_fl_05},
                "secondary_dose_1.25x_calibration": {"median_rel_int": round(median(rel_125), 4), "n_floored": n_fl_125},
            },
        },
    }


def render_scorecard_markdown(sc: dict[str, Any]) -> str:
    v = sc["confirmatory_verdicts"]
    lines = [
        "# OLMo 3 7B Stage C2 Confirmatory Scorecard (v4)",
        "",
        "## Confirmatory Verdicts",
        f"- **Global Gate `C2.0`:** `{v['C2.0']}`",
        f"- **`C2.1` (Whitening Arm Comparison & Calibration):** `{v['C2.1']}`",
        f"- **`C2.2` (Pure ChatML-Template `L19–L31` MLP Patching):** `{v['C2.2']}`",
        f"- **`C2.3` (`5`-Layer Post-Erasure Band):** `{v['C2.3']}` (band `{sc['pillar_C2_3']['selected_c3_band']}`)",
        f"- **`C2.4a` (`L24` Low-`γ` `pre_norm` Masking):** `{v['C2.4a']}`",
        f"- **`Gate EQ-C2.4c` (`K=3` Median Additivity TOST):** `{v['EQ-C2.4c']}` → `{sc['pillar_C2_4']['gate_EQ_C2_4c']['c3_action']}`",
        "",
        "## Unified `m = 7` Holm-Bonferroni Family",
        "| Rank | Test ID | `p` | `α/(8-k)` | Rejected |",
        "|---:|---|---:|---:|:---:|",
    ]
    for tid, row in sorted(sc["holm_7_family"].items(), key=lambda kv: kv[1]["rank"]):
        lines.append(f"| {row['rank']} | `{tid}` | `{row['p']:.6f}` | `{row['alpha_thresh']:.6f}` | `{row['rejected']}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="100% stdlib deterministic Stage C2 v4 scorer.")
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--c2-constants", type=Path, required=True)
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--no-strict-provenance", action="store_true")
    args = ap.parse_args()

    receipt = load_json_maybe_gz(args.receipt)
    c2_const = json.loads(args.c2_constants.read_text(encoding="utf-8"))
    const_sha = sha256_file(args.c2_constants)
    design_sha = sha256_file(args.design) if (args.design and args.design.is_file()) else None
    strict = (not args.no_strict_provenance) and (design_sha is not None)

    sc = score_stage_c2(receipt, c2_const, design_sha256=design_sha, constants_sha256=const_sha, strict_provenance=strict)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(sc, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_scorecard_markdown(sc), encoding="utf-8")
    print(f"Wrote {args.out_json} (sha256={sha256_file(args.out_json)}) and {args.out_md}")
    print("Verdicts:", json.dumps(sc["confirmatory_verdicts"]))


if __name__ == "__main__":
    main()
