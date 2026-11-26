#!/usr/bin/env python3
"""100% Python 3.10+ stdlib-only deterministic scorer for OLMo 3 7B Stage C2
(`OLMO3-TRANSPORT-C2-DESIGN.md` Registered v4, public `main` @ `ecd49b28`).

Requires ZERO third-party packages (no numpy, scipy, or torch) and uses `math.fsum`
for all float reductions so any clean Python 3.10+ machine reproduces `c2_scorecard.json`
and `C2-SCORECARD.md` byte-for-byte via `/usr/bin/python3 -I` + `cmp`.

Fail-closed guarantees:
  - Hard-refuses smoke/partial/schema-mismatched receipts (ValueError).
  - Evaluates Global Gate `C2.0` FIRST and short-circuits cleanly with the registered
    `BLOCKED_BY_C2.0` scorecard on ANY `C2.0` failure (NaNs, capped C2.2 traces, wrong shams,
    missing/junk active-arm names, truncated block row counts, bad doses, or provenance mismatch).
  - Pins all 61 expected active-arm names by exact set equality in `C2.0.1`.
  - Pins row counts across ALL blocks (`2,880`, `864 + 1,728 + 72`, `960 + 320`, `1,728`, `16`)
    against `planned_budget` and checks live `post_mlp_norms` dose scaling in `C2.0.3`.
  - Enforces the registered BCa containment invariant (`CI_low <= point_estimate <= CI_high`) fail-closed.
  - Notation erratum (GLM #7 / Claude #121): design line 146 wrote `KL(p_patched || p_ref)`, while
    C0's inherited formula and this code compute `KL(p_ref || p_patched) = sum p_c * (log p_c - log p_i)`.
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
C24_ARM_KEYS = (
    "L24.prenorm_ff_unmasked",
    "L24.prenorm_ff_unmasked_postmatched",
    "L24.prenorm_ff_masked",
    "L24.m70a30",
    "L24.prenorm_ff_rand_masked",
    "L24.m70a30_rand_masked",
    "L21.prenorm_ff_masked",
    "L27.prenorm_ff_masked",
    "K3_combined",
)

PERM_SEED = 2026112501
PERM_N_FLIPS_24 = 100_000
BCA_SEED = 2026112502
BCA_N_RESAMPLES = 10_000

# Single source of truth for all registered Stage C2 v4 numeric bars (verified against design in stdlib tests!)
REGISTERED_THRESHOLDS: dict[str, float] = {
    "C2.0.2_tele_p95_max": 0.08,
    "C2.0.2_tele_p999_max": 0.15,
    "C2.0.3_rmsn_gt_20": 0.20,
    "C2.0.3_rmsn_gt_12": 0.12,
    "C2.0.3_fde_gt_045": 0.45,
    "C2.0.3_fde_gt_075": 0.75,
    "C2.1_diag_eps": 0.25,
    "C2.1_bar1_pt": 2.00,
    "C2.1_bar1_q05": 1.50,
    "C2.1_bar2_pt": 1.15,
    "C2.1_bar2_sc_pt": 1.15,
    "C2.1_bar2_sc_q05": 1.05,
    "C2.1_bar3_pt": 1.35,
    "C2.1_bar3_q05": 1.15,
    "C2.1_bar4_pt": 0.80,
    "C2.1_bar4_q05": 0.75,
    "CAL_C2.1_ho_raw_lo": 0.04,
    "CAL_C2.1_ho_raw_hi": 0.25,
    "CAL_C2.1_r2_min": 0.40,
    "CAL_C2.1_slope_lo": 0.45,
    "CAL_C2.1_slope_hi": 1.60,
    "CAL_C2.1_diag_shk_frac": 0.50,
    "CAL_C2.1_mu_raw_min": 0.70,
    "CAL_C2.1_mu_shk_max": 0.45,
    "C2.2_gap_gate_min": 0.30,
    "C2.2_closure_pt_lo": 0.75,
    "C2.2_closure_pt_hi": 1.30,
    "C2.2_closure_ci_lo": 0.55,
    "C2.2_closure_ci_hi": 1.50,
    "EQ_C2.2_ctrl_std_pt": 0.25,
    "EQ_C2.2_ctrl_std_q95": 0.40,
    "EQ_C2.2_ctrl_mis_pt": 0.35,
    "EQ_C2.2_ctrl_mis_q95": 0.50,
    "EQ_C2.2_shape_open_min": 0.65,
    "EQ_C2.2_shape_diff_max": 0.35,
    "EQ_C2.2_kl_max": 0.25,
    "C2.3_e_open_min": -1.20,
    "C2.3_gap_max": 0.60,
    "C2.4a_bar7_pt": 1.40,
    "C2.4a_bar7_q05": 1.20,
    "EQ_C2.4c_med_max": 0.20,
    "EQ_C2.4c_ci_bound": 0.30,
    "EQ_C2.4c_floor": 0.10,
}


def expected_active_arm_names(c2_constants: dict[str, Any]) -> set[str]:
    """Exact set of 61 active-arm liveness keys emitted by run_olmo3_c2.py (GLM #4)."""
    keys: set[str] = set()
    for L in PRIMARY_C21_LAYERS:
        for a in C21_ARMS:
            keys.add(f"C2.1_L{L}_{a}")
    for cname in c2_constants["planned_budget"]["c22_conditions_list"]:
        for an in ("R_mf", "dm0"):
            keys.add(f"C2.2_{cname}_{an}")
    for L in SWEEP_LAYERS:
        keys.add(f"C2.3_L{L}_logit_lens")
        keys.add(f"C2.3_L{L}_rand_iso")
    for m in ("0.5", "1.25"):
        for ak in C24_ARM_KEYS:
            keys.add(f"C2.4_{m}_{ak}")
    return keys


# ---------------------------------------------------------------------------
# Pure, table-testable gate evaluators (single source of truth for T[...] uses)
# ---------------------------------------------------------------------------
def eval_c20_telescoping(
    nonfinite_count: int,
    tele_p95: float | None,
    tele_p999: float | None,
    T: dict[str, float] = REGISTERED_THRESHOLDS,
) -> bool:
    return (
        (nonfinite_count == 0)
        and (tele_p95 is not None)
        and (tele_p999 is not None)
        and (tele_p95 <= T["C2.0.2_tele_p95_max"])
        and (tele_p999 <= T["C2.0.2_tele_p999_max"])
    )


def classify_rmsn_shrink(shrink: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> tuple[bool, bool]:
    return (shrink > T["C2.0.3_rmsn_gt_20"], shrink > T["C2.0.3_rmsn_gt_12"])


def classify_fde_drift(fe: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> tuple[bool, bool]:
    return (fe > T["C2.0.3_fde_gt_045"], fe > T["C2.0.3_fde_gt_075"])


def c21_diag_denom(diag_val: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> float:
    return max(abs(diag_val), T["C2.1_diag_eps"])


def is_c21_diag_guarded(diag_val: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> bool:
    return abs(diag_val) <= T["C2.1_diag_eps"]


def eval_cal_c21(
    ho_cross_raw_mean: float,
    cal_r2: float,
    cal_slope: float,
    scaled_diag_shk_3L: float,
    frozen_diag_shk_3L: float,
    mu_raw: float,
    mu_shk: float,
    T: dict[str, float] = REGISTERED_THRESHOLDS,
) -> bool:
    return (
        (T["CAL_C2.1_ho_raw_lo"] <= ho_cross_raw_mean <= T["CAL_C2.1_ho_raw_hi"])
        and (cal_r2 >= T["CAL_C2.1_r2_min"])
        and (T["CAL_C2.1_slope_lo"] <= cal_slope <= T["CAL_C2.1_slope_hi"])
        and (scaled_diag_shk_3L >= T["CAL_C2.1_diag_shk_frac"] * frozen_diag_shk_3L)
        and (mu_raw >= T["CAL_C2.1_mu_raw_min"])
        and (mu_shk <= T["CAL_C2.1_mu_shk_max"])
    )


def eval_c22_gap_gate(pt: float, q05: float, q95: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> bool:
    return (abs(pt) >= T["C2.2_gap_gate_min"]) and ((q05 > 0.0) or (q95 < 0.0))


def eq_c22_ctrl_bounds(cname: str, T: dict[str, float] = REGISTERED_THRESHOLDS) -> tuple[float, float]:
    if cname == "ctrl_mismatch_prompt_MLP_L19_31":
        return (T["EQ_C2.2_ctrl_mis_pt"], T["EQ_C2.2_ctrl_mis_q95"])
    if cname in ("ctrl_L4_16_MLP_LongBare_to_ChatML", "ctrl_rand_iso_same_site_L19_31", "ctrl_self_attenuated_L19_31"):
        return (T["EQ_C2.2_ctrl_std_pt"], T["EQ_C2.2_ctrl_std_q95"])
    raise ValueError(f"Unknown EQ-C2.2 control condition: {cname!r}")


def eval_eq_c22_ctrl(cname: str, pt: float, q95: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> bool:
    pt_max, ci_max = eq_c22_ctrl_bounds(cname, T)
    return (abs(pt) <= pt_max) and (q95 <= ci_max)


def eval_eq_c22_shape_kl(cl_open: float, cl_func: float, med_kl: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> bool:
    return (
        (cl_open >= T["EQ_C2.2_shape_open_min"])
        and (abs(cl_open - cl_func) <= T["EQ_C2.2_shape_diff_max"])
        and (med_kl <= T["EQ_C2.2_kl_max"])
    )


def eval_c23_layer(eo: float, ef: float, ro: float, rr: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> bool:
    return (eo >= T["C2.3_e_open_min"]) and (abs(eo - ef) <= T["C2.3_gap_max"]) and (ro > rr)


def is_c24c_floored(d_sum: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> bool:
    return abs(d_sum) < T["EQ_C2.4c_floor"]


def c24c_rel_int(d_k3: float, d_sum: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> float:
    return (d_k3 - d_sum) / max(abs(d_sum), T["EQ_C2.4c_floor"])


def eval_eq_c24c_verdict(
    n_fl_05: int,
    pt: float,
    q05: float,
    q95: float,
    T: dict[str, float] = REGISTERED_THRESHOLDS,
) -> tuple[str, str]:
    eq_b = T["EQ_C2.4c_ci_bound"]
    if n_fl_05 <= 2 and abs(pt) <= T["EQ_C2.4c_med_max"] and q05 >= -eq_b and q95 <= eq_b:
        return ("ADDITIVE", "LICENSE_K3_SIMULTANEOUS")
    if q05 > eq_b or q95 < -eq_b:
        return ("NON_ADDITIVE", "RESTRICT_TO_SINGLE_LAYER_L24")
    return ("INCONCLUSIVE", "REQUIRE_1_DOSE_C3_PILOT")


def eval_bar_1(rejected: bool, pt: float, q05: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> bool:
    return rejected and (pt >= T["C2.1_bar1_pt"]) and (q05 > T["C2.1_bar1_q05"])


def eval_bar_2(rejected: bool, pt: float, sc_pt: float, sc_q05: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> bool:
    return (
        rejected
        and (pt >= T["C2.1_bar2_pt"])
        and (sc_pt >= T["C2.1_bar2_sc_pt"])
        and (sc_q05 >= T["C2.1_bar2_sc_q05"])
    )


def eval_bar_3(rejected: bool, pt: float, q05: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> bool:
    return rejected and (pt >= T["C2.1_bar3_pt"]) and (q05 > T["C2.1_bar3_q05"])


def eval_bar_4(rejected: bool, pt: float, q05: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> bool:
    return rejected and (pt >= T["C2.1_bar4_pt"]) and (q05 >= T["C2.1_bar4_q05"])


def eval_bar_5_or_6(
    gap_gate_pass: bool,
    rejected: bool,
    pt: float,
    q05: float,
    q95: float,
    T: dict[str, float] = REGISTERED_THRESHOLDS,
) -> bool:
    return (
        gap_gate_pass
        and rejected
        and (T["C2.2_closure_pt_lo"] <= pt <= T["C2.2_closure_pt_hi"])
        and (q05 >= T["C2.2_closure_ci_lo"])
        and (q95 <= T["C2.2_closure_ci_hi"])
    )


def eval_bar_7(rejected: bool, pt: float, q05: float, T: dict[str, float] = REGISTERED_THRESHOLDS) -> bool:
    return rejected and (pt >= T["C2.4a_bar7_pt"]) and (q05 > T["C2.4a_bar7_q05"])


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
    """Exact 2^N one-sided sign-flip enumeration (N=12 -> 4,096; N=16 -> 65,536)."""
    _require_nonempty_finite(diffs, "exact_sign_flip_p_one_sided")
    n = len(diffs)
    if n > 18:
        raise ValueError(f"exact_sign_flip_p_one_sided N={n} too large for exact enumeration")
    obs_sum = math.fsum(diffs)
    total = 1 << n
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
    """Pure-stdlib power iteration on A = M^T M returning (v1_right, s1^2 / ||M||_F^2)."""
    m_rows = len(mat)
    n_cols = len(mat[0])
    frob_sq = math.fsum(v * v for row in mat for v in row)
    if frob_sq <= 1e-18:
        return [0.0] * n_cols, 0.0
    A = [
        [math.fsum(mat[r][i] * mat[r][j] for r in range(m_rows)) for j in range(n_cols)]
        for i in range(n_cols)
    ]
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
# Main Scorer (Decides Gate C2.0 FIRST and short-circuits cleanly on failure!)
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
    if receipt.get("partial_checkpoint", False):
        raise ValueError("FAIL-CLOSED: Refusing to score a partial_checkpoint file.")
    if receipt.get("schema_version") != "olmo3-stage-c2-receipt-v4":
        raise ValueError(f"Unsupported receipt schema_version: {receipt.get('schema_version')!r}")
    meta = receipt.get("meta", {})
    if meta.get("smoke", False):
        raise ValueError("FAIL-CLOSED: Refusing to score a smoke/truncated Stage C2 receipt.")

    T = REGISTERED_THRESHOLDS
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
    c22_conds = list(c2_constants["planned_budget"]["c22_conditions_list"])

    # ------------------------------------------------------------------
    # GLOBAL GATE C2.0 (Decided FIRST; short-circuits cleanly if FAIL!)
    # ------------------------------------------------------------------
    def _blocked_scorecard(gate_doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "olmo3-stage-c2-scorecard-v4",
            "confirmatory_verdicts": {
                "C2.0": "FAIL",
                "C2.1": "BLOCKED_BY_C2.0",
                "C2.2": "BLOCKED_BY_C2.0",
                "C2.3": "BLOCKED_BY_C2.0",
                "C2.4a": "BLOCKED_BY_C2.0",
                "EQ-C2.4c": "BLOCKED_BY_C2.0",
            },
            "gate_C2_0": gate_doc,
            "blocked_by_c2_0": True,
            "holm_7_family": {},
        }

    try:
        # C2.0.1: 13 write shams + 6 patch shams == 0.0 exact; EXACT 61 active-arm names > 0.0 (GLM #4)
        shams_obj = receipt["shams"]
        write_shams = shams_obj["write_13"]
        patch_shams = shams_obj["patch_6"]
        req_w = set(c2_constants["planned_budget"]["write_shams_list"])
        req_p = set(c2_constants["planned_budget"]["patch_shams_list"])
        shams_present_ok = (set(write_shams.keys()) == req_w) and (set(patch_shams.keys()) == req_p)
        max_sham_w = max(abs(float(v)) for v in write_shams.values()) if write_shams else float("inf")
        max_sham_p = max(abs(float(v)) for v in patch_shams.values()) if patch_shams else float("inf")
        active_mins = receipt["active_arm_min_max_abs_dlogit"]
        exp_active_keys = expected_active_arm_names(c2_constants)
        active_keys_ok = set(active_mins.keys()) == exp_active_keys
        liveness_ok = active_keys_ok and all(float(v) > 0.0 for v in active_mins.values())
        c2_0_1_pass = shams_present_ok and (max_sham_w == 0.0) and (max_sham_p == 0.0) and liveness_ok

        # C2.0.2: Finiteness FIRST (never pass NaN to quantile/mean, GLM #3!) + 4-term telescoping
        nonfinite_paths = audit_finiteness(receipt)
        tele_errs: list[float] = []
        if not nonfinite_paths:
            for L in PRIMARY_C21_LAYERS:
                for aname in C21_ARMS:
                    for ts in t_strs:
                        for rec in receipt["block1_C2_1"][str(L)]["arms"][aname][ts]:
                            tele_errs.append(abs(float(rec["telescoping_err"])))
            for cname in c22_conds:
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
            tele_p95: float | None = quantile(tele_errs, 0.95) if tele_errs else None
            tele_p999: float | None = quantile(tele_errs, 0.999) if tele_errs else None
            c2_0_2_pass = eval_c20_telescoping(len(nonfinite_paths), tele_p95, tele_p999, T)
        else:
            tele_p95 = None
            tele_p999 = None
            c2_0_2_pass = False

        # C2.0.3: Per-cell row counts (DS R2 F4) + full live pmn dose checks (DS R2 F1) + Regime-N/D ceilings
        g203 = c2_constants["c2_0_3_named_gate_populations"]
        pb = c2_constants["planned_budget"]
        pmn_map = receipt.get("post_mlp_norms")
        pmn_complete_ok = (
            isinstance(pmn_map, dict)
            and all(str(L) in pmn_map and math.isfinite(float(pmn_map[str(L)])) and float(pmn_map[str(L)]) > 0.0 for L in range(32))
        )
        unmasked_eff_l24 = float(c2_constants["prenorm_gamma_mask_geometry"]["24"]["ff"]["unmasked_deliverable_frac"])
        pr_map_c20 = c2_constants["c22_l18_frozen_pred_ret"]

        reg_n_n = reg_n_cap = reg_n_20 = reg_n_12 = 0
        dose_checks_ok = pmn_complete_ok
        per_cell_counts_ok = True
        if not nonfinite_paths:
            for L in PRIMARY_C21_LAYERS:
                if pmn_complete_ok:
                    exp_d = 0.75 * float(pmn_map[str(L)])
                    if abs(float(receipt["block1_C2_1"][str(L)]["dose"]) - exp_d) > 1e-3:
                        dose_checks_ok = False
                for aname in C21_ARMS:
                    for ts in t_strs:
                        rows = receipt["block1_C2_1"][str(L)]["arms"][aname][ts]
                        if len(rows) != 8:
                            per_cell_counts_ok = False
                        for rec in rows:
                            reg_n_n += 1
                            if rec.get("capped", False):
                                reg_n_cap += 1
                                continue
                            rc, ri = float(rec["rms_clean"]), float(rec["rms_int"])
                            shrink = abs(ri - rc) / max(rc, 1e-12)
                            gt20, gt12 = classify_rmsn_shrink(shrink, T)
                            if gt20:
                                reg_n_20 += 1
                            if gt12:
                                reg_n_12 += 1

        reg_d_n = reg_d_cap = reg_d_20 = reg_d_12 = reg_d_fde45 = reg_d_fde75 = 0
        c22_patched_n = c22_patched_cap = 0
        fde_errs: list[float] = []
        if not nonfinite_paths:
            b2_obj = receipt["block2_C2_2"]
            if pmn_complete_ok:
                exp_cap_l18 = 3.0 * float(pmn_map["18"])
                if "dose_cap" not in b2_obj or abs(float(b2_obj["dose_cap"]) - exp_cap_l18) > 1e-3:
                    dose_checks_ok = False
            b2_traces = b2_obj["traces"]
            if set(b2_traces.keys()) != set(c22_conds):
                per_cell_counts_ok = False
            for cname in c22_conds:
                is_unp = cname in ("unpatched_LongBare", "unpatched_ChatML", "unpatched_ShortBare")
                for pidx in range(12):
                    for ts in c22_t_strs:
                        u_dot_w = float(pr_map_c20[ts]["u_dot_w_Rmf"])
                        for aname in ("R_mf", "dm0"):
                            rec = b2_traces[cname][str(pidx)][ts][aname]
                            if is_unp:
                                reg_d_n += 1
                                if rec.get("capped", False):
                                    reg_d_cap += 1
                                    continue
                                rc, ri = float(rec["rms_clean"]), float(rec["rms_int"])
                                if pmn_complete_ok:
                                    if "dose" not in rec:
                                        dose_checks_ok = False
                                    else:
                                        d_obs = float(rec["dose"])
                                        if d_obs > 3.0 * float(pmn_map["18"]) + 1e-3 or abs(d_obs - 3.0 * rc / u_dot_w) > 1e-2:
                                            dose_checks_ok = False
                                shrink = abs(ri - rc) / max(rc, 1e-12)
                                gt20, gt12 = classify_rmsn_shrink(shrink, T)
                                if gt20:
                                    reg_d_20 += 1
                                if gt12:
                                    reg_d_12 += 1
                                fe = abs(float(rec["frozen_direct_effect"]) - 3.0)
                                fde_errs.append(fe)
                                gt45, gt75 = classify_fde_drift(fe, T)
                                if gt45:
                                    reg_d_fde45 += 1
                                if gt75:
                                    reg_d_fde75 += 1
                            else:
                                c22_patched_n += 1
                                if rec.get("capped", False):
                                    c22_patched_cap += 1
                                elif pmn_complete_ok:
                                    if "dose" not in rec:
                                        dose_checks_ok = False
                                    else:
                                        rc = float(rec["rms_clean"])
                                        d_obs = float(rec["dose"])
                                        if d_obs > 3.0 * float(pmn_map["18"]) + 1e-3 or abs(d_obs - 3.0 * rc / u_dot_w) > 1e-2:
                                            dose_checks_ok = False

        patched_conds = [c for c in c22_conds if not c.startswith("unpatched_")]
        b2_unsteered_map = receipt["block2_C2_2"]["unsteered_patched_kl"]
        if set(b2_unsteered_map.keys()) != set(patched_conds):
            per_cell_counts_ok = False
        b2_unsteered_n = 0
        for cname in patched_conds:
            u_rows = b2_unsteered_map[cname]
            if len(u_rows) != 12:
                per_cell_counts_ok = False
            b2_unsteered_n += len(u_rows)

        b2b_ll_n = 0
        b2b_ri_n = 0
        for L in SWEEP_LAYERS:
            blk = receipt["block2b_C2_3"][str(L)]
            if pmn_complete_ok and not nonfinite_paths:
                if abs(float(blk["dose"]) - 0.75 * float(pmn_map[str(L)])) > 1e-3:
                    dose_checks_ok = False
            for ts in t_strs:
                ll_rows = blk["logit_lens"][ts]
                if len(ll_rows) != 8:
                    per_cell_counts_ok = False
                b2b_ll_n += len(ll_rows)
            ri_rows = blk["rand_iso"]
            if len(ri_rows) != 64:
                per_cell_counts_ok = False
            b2b_ri_n += len(ri_rows)

        b3_n = 0
        for m in ("0.5", "1.25"):
            mult = float(m)
            for ak in C24_ARM_KEYS:
                exp_b3_d: float | None = None
                if pmn_complete_ok:
                    pmn21, pmn24, pmn27 = float(pmn_map["21"]), float(pmn_map["24"]), float(pmn_map["27"])
                    if ak == "L21.prenorm_ff_masked":
                        exp_b3_d = mult * pmn21
                    elif ak == "L27.prenorm_ff_masked":
                        exp_b3_d = mult * pmn27
                    elif ak == "L24.prenorm_ff_unmasked_postmatched":
                        exp_b3_d = (mult * pmn24) / unmasked_eff_l24
                    elif ak == "K3_combined":
                        exp_b3_d = mult * (pmn21 + pmn24 + pmn27)
                    else:
                        exp_b3_d = mult * pmn24
                for ts in t_strs:
                    b3_rows = receipt["block3_C2_4"]["doses"][m][ak][ts]
                    if len(b3_rows) != 4:
                        per_cell_counts_ok = False
                    b3_n += len(b3_rows)
                    if exp_b3_d is not None and not nonfinite_paths:
                        for row in b3_rows:
                            if row.get("capped", False):
                                continue
                            if "dose" not in row or abs(float(row["dose"]) - exp_b3_d) > 1e-3:
                                dose_checks_ok = False

        exp_anc_ids = {p["id"] for p in c2_constants["c0_anchor_colon_prediction"]["prompts_16"]}
        anc_map = receipt["c0_anchor_colon"]
        if set(anc_map.keys()) != exp_anc_ids:
            per_cell_counts_ok = False
        anc_n = len(anc_map)

        all_counts_ok = (
            per_cell_counts_ok
            and (reg_n_n == pb["block1_C2_1_3L_5arms_24x8"])
            and (reg_d_n == g203["regime_D_C22_unpatched_population_N"])
            and (reg_d_n + c22_patched_n == pb["block2_C2_2_steered_9cond_12t_12p_2arms"])
            and (b2_unsteered_n == pb["block2_C2_2_unsteered_patched_kl_6cond_12p"])
            and (b2b_ll_n + b2b_ri_n == pb["block2b_C2_3_5L_32v_8p"])
            and (b2b_ll_n == 5 * 24 * 8)
            and (b2b_ri_n == 5 * 8 * 8)
            and (b3_n == pb["block3_C2_4_9arms_24t_4p_2doses"])
            and (anc_n == pb["c0_colon_anchor_pairs_16"])
        )
        fde_p95: float | None = quantile(fde_errs, 0.95) if fde_errs else None
        c2_0_3_pass = (
            (not nonfinite_paths)
            and all_counts_ok
            and dose_checks_ok
            and (reg_n_cap <= g203["regime_N_max_capped"])
            and (reg_n_20 <= g203["regime_N_max_rmsn_gt_20pct"])
            and (reg_n_12 <= g203["regime_N_max_rmsn_gt_12pct"])
            and (reg_d_cap <= g203["regime_D_max_capped"])
            and (c22_patched_cap == 0)
            and (reg_d_20 <= g203["regime_D_max_rmsn_gt_20pct"])
            and (reg_d_12 <= g203["regime_D_max_rmsn_gt_12pct"])
            and (fde_p95 is not None)
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

        gate_c2_0_doc = {
            "pass": c2_0_pass,
            "C2.0.1_shams_and_liveness": {
                "pass": c2_0_1_pass, "shams_present_ok": shams_present_ok, "active_keys_exact_ok": active_keys_ok,
                "max_write_sham": max_sham_w, "max_patch_sham": max_sham_p, "liveness_ok": liveness_ok,
            },
            "C2.0.2_finiteness_and_telescoping": {
                "pass": c2_0_2_pass, "nonfinite_count": len(nonfinite_paths),
                "nonfinite_sample": nonfinite_paths[:5],
                "tele_p95": round(tele_p95, 6) if tele_p95 is not None else None,
                "tele_p999": round(tele_p999, 6) if tele_p999 is not None else None,
            },
            "C2.0.3_linearity_and_completeness": {
                "pass": c2_0_3_pass,
                "all_block_counts_ok": all_counts_ok,
                "per_cell_counts_ok": per_cell_counts_ok,
                "pmn_complete_ok": pmn_complete_ok,
                "dose_checks_ok": dose_checks_ok,
                "block_row_counts": {
                    "block1_C2_1": reg_n_n, "block2_unpatched": reg_d_n, "block2_patched": c22_patched_n,
                    "block2_unsteered_kl": b2_unsteered_n, "block2b_logit_lens": b2b_ll_n,
                    "block2b_rand_iso": b2b_ri_n, "block3_C2_4": b3_n, "c0_anchor_colon": anc_n,
                },
                "regime_N": {"N": reg_n_n, "capped": reg_n_cap, "rmsn_gt_20": reg_n_20, "rmsn_gt_12": reg_n_12},
                "regime_D": {
                    "N": reg_d_n, "capped": reg_d_cap, "patched_capped": c22_patched_cap,
                    "rmsn_gt_20": reg_d_20, "rmsn_gt_12": reg_d_12,
                    "fde_p95": round(fde_p95, 6) if fde_p95 is not None else None,
                    "fde_gt_045": reg_d_fde45, "fde_gt_075_diagnostic": reg_d_fde75,
                },
            },
            "C2.0.4_provenance": {"pass": c2_0_4_pass, "strict": strict_provenance, "receipt_hashes": ph},
        }
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        return _blocked_scorecard({
            "pass": False,
            "schema_error": f"{type(exc).__name__}: {exc}",
        })

    # Short-circuit cleanly on ANY C2.0 failure (GLM #3, DS F3: never crash on missing cells, NaNs, or capped traces!)
    if not c2_0_pass:
        return _blocked_scorecard(gate_c2_0_doc)

    # Non-gating C0-colon anchor prediction
    anc_raw = receipt["c0_anchor_colon"]
    anc_pairs: dict[str, dict[str, Any]] = {}
    for _, row in anc_raw.items():
        anc_pairs.setdefault(row["pair_id"], {})[row["ending"]] = row
    r24_all, r27_all, r24_bare, r27_bare = [], [], [], []
    for _, ends in sorted(anc_pairs.items()):
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
    diag_by_arm_i_L: dict[str, dict[int, dict[int, float]]] = {a: {i: {} for i in range(24)} for a in C21_ARMS}
    ho_cross_by_arm_i_L: dict[str, dict[int, dict[int, float]]] = {a: {i: {} for i in prim_idxs} for a in C21_ARMS}
    same_by_arm_i_L: dict[str, dict[int, dict[int, float]]] = {a: {i: {} for i in range(24)} for a in C21_ARMS}
    cross_all_by_arm_i_L: dict[str, dict[int, dict[int, float]]] = {a: {i: {} for i in range(24)} for a in C21_ARMS}
    guarded_sources_by_arm: dict[str, set[int]] = {a: set() for a in C21_ARMS}
    per_prompt_ho_cross: dict[str, dict[int, dict[int, list[float]]]] = {a: {pi: {} for pi in range(8)} for a in C21_ARMS}
    mat24_by_L_arm: dict[int, dict[str, list[list[float]]]] = {L: {} for L in PRIMARY_C21_LAYERS}
    mat24_scaled_by_L_arm: dict[int, dict[str, list[list[float]]]] = {L: {} for L in PRIMARY_C21_LAYERS}

    for L in PRIMARY_C21_LAYERS:
        dose_l = float(receipt["block1_C2_1"][str(L)]["dose"])
        for a in C21_ARMS:
            M_raw = [[0.0] * 24 for _ in range(24)]
            M_scaled = [[0.0] * 24 for _ in range(24)]
            for i in range(24):
                rows = receipt["block1_C2_1"][str(L)]["arms"][a][t_strs[i]]
                diag_iL = mean([float(r["all_24_dlogits"][i]) for r in rows])
                diag_by_arm_i_L[a][i][L] = diag_iL
                denom = c21_diag_denom(diag_iL, T)
                if is_c21_diag_guarded(diag_iL, T) and i in prim_idxs:
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
                    for pi, r in enumerate(rows):
                        p_ho = mean([abs(float(r["all_24_dlogits"][j])) for j in ho_idxs if strata[j] != strata[i]]) / c21_diag_denom(float(r["all_24_dlogits"][i]), T)
                        per_prompt_ho_cross[a][pi].setdefault(i, []).append(p_ho)

            mat24_by_L_arm[L][a] = M_raw
            mat24_scaled_by_L_arm[L][a] = M_scaled

    degenerate_diag = any(len(s) > 2 for s in guarded_sources_by_arm.values())

    diag_arm: dict[str, list[float]] = {a: [mean([diag_by_arm_i_L[a][i][L] for L in PRIMARY_C21_LAYERS]) for i in range(24)] for a in C21_ARMS}
    ho_cross_arm: dict[str, list[float]] = {a: [max(mean([ho_cross_by_arm_i_L[a][i][L] for L in PRIMARY_C21_LAYERS]), 1e-6) for i in prim_idxs] for a in C21_ARMS}
    same_i_arm: dict[str, list[float]] = {a: [mean([same_by_arm_i_L[a][i][L] for L in PRIMARY_C21_LAYERS]) for i in range(24)] for a in C21_ARMS}
    cross_i_arm: dict[str, list[float]] = {a: [max(mean([cross_all_by_arm_i_L[a][i][L] for L in PRIMARY_C21_LAYERS]), 1e-6) for i in range(24)] for a in C21_ARMS}

    x_ctr_raw = [math.log(ho_cross_arm["R_ridge_raw"][k] / ho_cross_arm["R_ridge_centered"][k]) for k in range(16)]
    x_shk_ctr = [math.log(ho_cross_arm["R_ridge_centered"][k] / ho_cross_arm["R_ridge_shrink"][k]) for k in range(16)]
    x_shk_ll = [math.log(ho_cross_arm["logit_lens"][k] / ho_cross_arm["R_ridge_shrink"][k]) for k in range(16)]
    x_gain_ret = [math.log(max(diag_arm["R_ridge_shrink"][i], 1e-6) / max(diag_arm["R_ridge_raw"][i], 1e-6)) - math.log(T["C2.1_bar4_q05"]) for i in range(24)]

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

    fref = receipt.get("frozen_cal_reference", {})
    u_mu_n = [float(v) for v in fref.get("u_mu_proj_n", [0.0] * 24)]
    mu_cos_by_arm: dict[str, dict[str, Any]] = {}
    rank1_by_arm: dict[str, dict[str, Any]] = {}
    for a in C21_ARMS:
        c_list, r1_list = [], []
        for L in PRIMARY_C21_LAYERS:
            M = [row[:] for row in mat24_by_L_arm[L][a]]
            for i in range(24):
                M[i][i] = 0.0
            v1, r1 = top_right_singular_vector_and_rank1_share(M)
            c_list.append(abs(math.fsum(x * y for x, y in zip(v1, u_mu_n))))
            r1_list.append(r1)
        mu_cos_by_arm[a] = {"mean_3L": round(mean(c_list), 4), "per_L": [round(v, 4) for v in c_list]}
        rank1_by_arm[a] = {"mean_3L": round(mean(r1_list), 4), "per_L": [round(v, 4) for v in r1_list]}

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

    cal_c21_pass = eval_cal_c21(
        ho_cross_raw_mean,
        cal_r2,
        cal_slope,
        scaled_diag_shk_3L,
        frozen_diag_shk_3L,
        mu_cos_by_arm["R_ridge_raw"]["mean_3L"],
        mu_cos_by_arm["R_ridge_shrink"]["mean_3L"],
        T,
    )

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
        gaps = [
            float(c22_tr[cname][str(pidx)][ts]["R_mf"]["exact_dlogit"])
            - float(c22_tr[cname][str(pidx)][ts]["dm0"]["exact_dlogit"])
            for ts in subset_strs
        ]
        return mean(gaps) / pred_denom

    M_cond: dict[str, list[float]] = {
        cname: [_pair_M(cname, pidx, c22_t_strs, pred_gap_mean_12) for pidx in range(12)]
        for cname in c22_conds
    }
    gap_pairs = [M_cond["unpatched_ChatML"][i] - M_cond["unpatched_LongBare"][i] for i in range(12)]
    len_pairs = [M_cond["unpatched_LongBare"][i] - M_cond["unpatched_ShortBare"][i] for i in range(12)]
    ci_gap = bca_cluster_ci(12, lambda idx: mean([gap_pairs[i] for i in idx]), n_boot=n_boot, seed=BCA_SEED + 10)
    ci_len = bca_cluster_ci(12, lambda idx: mean([len_pairs[i] for i in idx]), n_boot=n_boot, seed=BCA_SEED + 11)
    g_gap = ci_gap["point"]
    s_sign = 1.0 if g_gap >= 0.0 else -1.0
    gap_gate_pass = eval_c22_gap_gate(g_gap, ci_gap["q_0.05"], ci_gap["q_0.95"], T)
    len_gap_present = eval_c22_gap_gate(ci_len["point"], ci_len["q_0.05"], ci_len["q_0.95"], T)

    def _closure_stat(deltas: list[float], gaps: list[float], idx: list[int]) -> float:
        g_sub = mean([gaps[i] for i in idx])
        return 0.0 if abs(g_sub) < 1e-9 else mean([deltas[i] for i in idx]) / g_sub

    del_b2c = [M_cond["unpatched_ChatML"][i] - M_cond["patch_LongBare_to_ChatML_MLP_L19_31"][i] for i in range(12)]
    del_c2b = [M_cond["patch_ChatML_to_LongBare_MLP_L19_31"][i] - M_cond["unpatched_LongBare"][i] for i in range(12)]
    d50_b2c = [s_sign * (del_b2c[i] - 0.50 * gap_pairs[i]) for i in range(12)]
    d50_c2b = [s_sign * (del_c2b[i] - 0.50 * gap_pairs[i]) for i in range(12)]

    p_c22_b2c = exact_sign_flip_p_one_sided(d50_b2c)
    p_c22_c2b = exact_sign_flip_p_one_sided(d50_c2b)
    ci_b2c = bca_cluster_ci(12, lambda idx: _closure_stat(del_b2c, gap_pairs, idx), n_boot=n_boot, seed=BCA_SEED + 12)
    ci_c2b = bca_cluster_ci(12, lambda idx: _closure_stat(del_c2b, gap_pairs, idx), n_boot=n_boot, seed=BCA_SEED + 13)

    ctrl_specs = [
        ("ctrl_L4_16_MLP_LongBare_to_ChatML", 14),
        ("ctrl_rand_iso_same_site_L19_31", 15),
        ("ctrl_mismatch_prompt_MLP_L19_31", 16),
        ("ctrl_self_attenuated_L19_31", 17),
    ]
    eq_c22_ctrls: dict[str, Any] = {}
    eq_c22_all_ctrls_ok = True
    for cname, seed_off in ctrl_specs:
        del_c = [M_cond["unpatched_ChatML"][i] - M_cond[cname][i] for i in range(12)]
        ci_c = bca_cluster_ci(12, lambda idx, dc=del_c: _closure_stat(dc, gap_pairs, idx), n_boot=n_boot, seed=BCA_SEED + seed_off)
        ok_c = eval_eq_c22_ctrl(cname, ci_c["point"], ci_c["q_0.95"], T)
        eq_c22_all_ctrls_ok = eq_c22_all_ctrls_ok and ok_c
        eq_c22_ctrls[cname] = {"point": round(ci_c["point"], 4), "q05": round(ci_c["q_0.05"], 4), "q95": round(ci_c["q_0.95"], 4), "pass": ok_c}

    g_open = mean([_pair_M("unpatched_ChatML", i, c22_open_strs, pred_gap_mean_open) - _pair_M("unpatched_LongBare", i, c22_open_strs, pred_gap_mean_open) for i in range(12)])
    g_func = mean([_pair_M("unpatched_ChatML", i, c22_func_strs, pred_gap_mean_func) - _pair_M("unpatched_LongBare", i, c22_func_strs, pred_gap_mean_func) for i in range(12)])
    cl_open = mean([_pair_M("unpatched_ChatML", i, c22_open_strs, pred_gap_mean_open) - _pair_M("patch_LongBare_to_ChatML_MLP_L19_31", i, c22_open_strs, pred_gap_mean_open) for i in range(12)]) / (g_open if abs(g_open) > 1e-9 else 1.0)
    cl_func = mean([_pair_M("unpatched_ChatML", i, c22_func_strs, pred_gap_mean_func) - _pair_M("patch_LongBare_to_ChatML_MLP_L19_31", i, c22_func_strs, pred_gap_mean_func) for i in range(12)]) / (g_func if abs(g_func) > 1e-9 else 1.0)
    unsteered_kls = [float(r["kl"]) for rows in receipt["block2_C2_2"]["unsteered_patched_kl"].values() for r in rows]
    med_unsteered_kl = median(unsteered_kls)
    shape_kl_ok = eval_eq_c22_shape_kl(cl_open, cl_func, med_unsteered_kl, T)
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
        pe = eval_c23_layer(eo, ef, ro, rr, T)
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

    dl_mask_pool = [0.5 * (a + b) for a, b in zip(_c24_target_mean("0.5", "L24.prenorm_ff_masked"), _c24_target_mean("1.25", "L24.prenorm_ff_masked"))]
    dl_unm_pool = [0.5 * (a + b) for a, b in zip(_c24_target_mean("0.5", "L24.prenorm_ff_unmasked"), _c24_target_mean("1.25", "L24.prenorm_ff_unmasked"))]
    x_c24a = [math.log(max(dl_mask_pool[i], 1e-6) / max(dl_unm_pool[i], 1e-6)) for i in range(24)]
    p_c24a = mc_sign_flip_p_one_sided(x_c24a, n_flips=n_flips_24, seed=PERM_SEED + 1)
    ci_c24a = bca_cluster_ci(24, lambda idx: math.exp(mean([x_c24a[i] for i in idx])), n_boot=n_boot, seed=BCA_SEED + 20)

    pm_capped = sum(1 for m in ("0.5", "1.25") for ts in t_strs for r in d_blk[m]["L24.prenorm_ff_unmasked_postmatched"][ts] if r.get("capped", False))
    if pm_capped == 0:
        dl_pm_pool = [0.5 * (a + b) for a, b in zip(_c24_target_mean("0.5", "L24.prenorm_ff_unmasked_postmatched"), _c24_target_mean("1.25", "L24.prenorm_ff_unmasked_postmatched"))]
        x_pm = [math.log(max(dl_mask_pool[i], 1e-6) / max(dl_pm_pool[i], 1e-6)) for i in range(24)]
        c24a_postmatched_diag = {"capped_count": 0, "geomean_masked_over_postmatched": round(math.exp(mean(x_pm)), 4), "precondition_ok": True}
    else:
        c24a_postmatched_diag = {"capped_count": pm_capped, "geomean_masked_over_postmatched": None, "precondition_ok": False}

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

    def _k3_rel_int(dose_str: str) -> tuple[list[float], int]:
        d21 = _c24_target_mean(dose_str, "L21.prenorm_ff_masked")
        d24 = _c24_target_mean(dose_str, "L24.prenorm_ff_masked")
        d27 = _c24_target_mean(dose_str, "L27.prenorm_ff_masked")
        dk3 = _c24_target_mean(dose_str, "K3_combined")
        rels, n_fl = [], 0
        for i in range(24):
            s = d21[i] + d24[i] + d27[i]
            if is_c24c_floored(s, T):
                n_fl += 1
            rels.append(c24c_rel_int(dk3[i], s, T))
        return rels, n_fl

    rel_05, n_fl_05 = _k3_rel_int("0.5")
    rel_125, n_fl_125 = _k3_rel_int("1.25")
    ci_k3_05 = bca_cluster_ci(24, lambda idx: median([rel_05[i] for i in idx]), n_boot=n_boot, seed=BCA_SEED + 23)
    eq_c24c_verdict, c3_k3_action = eval_eq_c24c_verdict(n_fl_05, ci_k3_05["point"], ci_k3_05["q_0.05"], ci_k3_05["q_0.95"], T)

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

    bar_1 = eval_bar_1(holm["p_C2.1_ctr_vs_raw"]["rejected"], ci_ctr_raw["point"], ci_ctr_raw["q_0.05"], T)
    bar_2 = eval_bar_2(holm["p_C2.1_shk_vs_ctr"]["rejected"], ci_shk_ctr["point"], ci_sc_ratio["point"], ci_sc_ratio["q_0.05"], T)
    bar_3 = eval_bar_3(holm["p_C2.1_shk_vs_ll"]["rejected"], ci_shk_ll["point"], ci_shk_ll["q_0.05"], T)
    bar_4 = eval_bar_4(holm["p_C2.1_gain_ret"]["rejected"], ci_gain_ret["point"], ci_gain_ret["q_0.05"], T)
    bar_5 = eval_bar_5_or_6(gap_gate_pass, holm["p_C2.2_bare_into_chat"]["rejected"], ci_b2c["point"], ci_b2c["q_0.05"], ci_b2c["q_0.95"], T)
    bar_6 = eval_bar_5_or_6(gap_gate_pass, holm["p_C2.2_chat_into_bare"]["rejected"], ci_c2b["point"], ci_c2b["q_0.05"], ci_c2b["q_0.95"], T)
    bar_7 = eval_bar_7(holm["p_C2.4a_mask_vs_unmask"]["rejected"], ci_c24a["point"], ci_c24a["q_0.05"], T)

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
            "C2.0": "PASS",
            "C2.1": v_c21,
            "C2.2": v_c22,
            "C2.3": v_c23,
            "C2.4a": v_c24a,
            "EQ-C2.4c": v_c24c,
        },
        "gate_C2_0": gate_c2_0_doc,
        "blocked_by_c2_0": False,
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
        f"- **`C2.3` (`5`-Layer Post-Erasure Band):** `{v['C2.3']}`",
        f"- **`C2.4a` (`L24` Low-`γ` `pre_norm` Masking):** `{v['C2.4a']}`",
        f"- **`Gate EQ-C2.4c` (`K=3` Median Additivity TOST):** `{v['EQ-C2.4c']}`",
    ]
    if not sc.get("blocked_by_c2_0", False):
        lines.extend([
            "",
            f"- **Selected C3 Band:** `{sc['pillar_C2_3']['selected_c3_band']}` · **`EQ-C2.4c` Action:** `{sc['pillar_C2_4']['gate_EQ_C2_4c']['c3_action']}`",
            "",
            "## Unified `m = 7` Holm-Bonferroni Family",
            "| Rank | Test ID | `p` | `α/(8-k)` | Rejected |",
            "|---:|---|---:|---:|:---:|",
        ])
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
