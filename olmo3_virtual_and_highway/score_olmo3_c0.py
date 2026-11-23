#!/usr/bin/env python3
"""100% Python 3.10+ stdlib-only deterministic scorer for Stage C0 (`OLMO3-TRANSPORT-C0-DESIGN.md` Registered v1).

Requires ZERO third-party packages (no numpy, scipy, or torch) so any clean machine can reproduce
`scorecard.json` and `SCORECARD.md` byte-for-byte via `python3 -I` + `cmp`.
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

EXPECTED_R_LENS_SHA256 = "ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955"
EXPECTED_CONSTANTS_SHA256 = "3be64ef80f12d34fc59d7e561d5617dd96ecb605649d5f64e1a98bd974acbe98"
EXPECTED_DESIGN_SHA256 = "59b9da1a585dbecba26954ae41a394f9a1a385bcc831a234b859d9d5e52a78bf"

L18_PLUS = [18, 21, 24, 27]
L12_TO_27 = [12, 15, 18, 21, 24, 27]
L_ANALOGUE = [12, 18, 24]
L_EARLY = [6, 12]
NON_NULL_ARMS = ["R_mf", "R_ridge", "logit_lens", "direct_matched"]

BOOT_RESAMPLES = 10_000
PERM_RESAMPLES = 10_000
SEED_BOOT = 20261122
SEED_PERM = 20261123


# ---------------------------------------------------------------------------
# Pure-stdlib I/O & hashing
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_uncompressed(path: Path) -> str:
    h = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_maybe_gz(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Pure-stdlib statistics (Registered v1 §2 conventions)
# ---------------------------------------------------------------------------
def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def quantile(xs: list[float], q: float) -> float:
    """Linear interpolation at index q * (n - 1), matching numpy.quantile(..., method='linear')."""
    if not xs:
        return 0.0
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
    """Acklam's rational approximation to Phi^{-1}(p), max relative error < 1.15e-9."""
    if p <= 0.0:
        return -38.0
    if p >= 1.0:
        return 38.0
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def avg_ranks(xs: list[float]) -> list[float]:
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[order[j]] == xs[order[i]]:
            j += 1
        avg_r = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[order[k]] = avg_r
        i = j
    return ranks


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    rx = avg_ranks(xs)
    ry = avg_ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return max(-1.0, min(1.0, num / (dx * dy)))


def fisher_z_mean(rhos: list[float]) -> float:
    if not rhos:
        return 0.0
    zs = [math.atanh(max(-0.999999, min(0.999999, r))) for r in rhos]
    return math.tanh(sum(zs) / len(zs))


def exact_sign_flip_p_one_sided(vals: list[float]) -> float:
    """Exact one-sided paired sign-flip p-value P(mean(s_i * x_i) >= mean(x_i)) over all 2^n flips.

    Uses Gray-code traversal (O(2^n) with 1 addition per step — ~15 ms for n=16 in pure Python).
    """
    n = len(vals)
    if n == 0:
        return 1.0
    obs = sum(vals)
    tol = 1e-12 * max(1.0, sum(abs(x) for x in vals))
    # Start with all signs +1 (code = 0)
    cur = obs
    signs = [1] * n
    ge = 1  # code = 0 has sum == obs
    total = 1 << n
    prev_gray = 0
    for step in range(1, total):
        gray = step ^ (step >> 1)
        diff = gray ^ prev_gray
        bit = diff.bit_length() - 1
        prev_gray = gray
        if signs[bit] == 1:
            signs[bit] = -1
            cur -= 2.0 * vals[bit]
        else:
            signs[bit] = 1
            cur += 2.0 * vals[bit]
        if cur >= obs - tol:
            ge += 1
    return ge / total


def bca_cluster_ci(
    n_clusters: int,
    stat_fn,
    alphas: tuple[float, ...],
    n_boot: int = BOOT_RESAMPLES,
    seed: int = SEED_BOOT,
) -> dict:
    """10,000-resample BCa cluster bootstrap over indices `0..n_clusters-1`.

    `stat_fn(indices: list[int]) -> float` evaluates the statistic on a list of cluster indices.
    Returns `point` and BCa quantiles at each alpha in `alphas`.
    """
    full_idx = list(range(n_clusters))
    theta_hat = float(stat_fn(full_idx))

    rng = random.Random(seed)
    boots = [0.0] * n_boot
    for b in range(n_boot):
        idx = [rng.randrange(n_clusters) for _ in range(n_clusters)]
        boots[b] = float(stat_fn(idx))
    boots.sort()

    # Bias-correction z0
    less = sum(1 for v in boots if v < theta_hat)
    eq = sum(1 for v in boots if v == theta_hat)
    prop = (less + 0.5 * eq) / n_boot
    prop = max(0.5 / n_boot, min(1.0 - 0.5 / n_boot, prop))
    z0 = norm_ppf(prop)

    # Jackknife acceleration a_hat
    jacks = [float(stat_fn([j for j in range(n_clusters) if j != i])) for i in range(n_clusters)]
    j_mean = sum(jacks) / n_clusters
    diffs = [j_mean - jv for jv in jacks]
    denom = sum(d * d for d in diffs)
    if denom <= 1e-18:
        a_hat = 0.0
    else:
        a_hat = sum(d * d * d for d in diffs) / (6.0 * (denom ** 1.5))

    out = {"point": round(theta_hat, 6)}
    for al in alphas:
        z_a = norm_ppf(al)
        adj_denom = 1.0 - a_hat * (z0 + z_a)
        if abs(adj_denom) < 1e-12:
            q_adj = al
        else:
            q_adj = norm_cdf(z0 + (z0 + z_a) / adj_denom)
        q_adj = max(0.0, min(1.0, q_adj))
        out[f"q_{al:g}"] = round(quantile(boots, q_adj), 6)
    return out


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Step-down Holm-Bonferroni at familywise `alpha`."""
    m = len(pvals)
    ordered = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    out: dict[str, dict] = {}
    running_max_adj = 0.0
    rejected_so_far = True
    for rank, (name, p) in enumerate(ordered):
        mult = m - rank
        p_adj = min(1.0, max(running_max_adj, p * mult))
        running_max_adj = p_adj
        thr = alpha / mult
        rej = rejected_so_far and (p <= thr)
        rejected_so_far = rej
        out[name] = {
            "p_raw": round(p, 6),
            "p_holm": round(p_adj, 6),
            "threshold": round(thr, 6),
            "rejected": rej,
        }
    return out


def kruskal_wallis_h(groups: list[list[float]]) -> dict:
    """Pure-stdlib Kruskal-Wallis H statistic with tie correction."""
    all_vals = []
    for g_idx, g in enumerate(groups):
        for v in g:
            all_vals.append((v, g_idx))
    n = len(all_vals)
    if n < 2 or len(groups) < 2:
        return {"H": 0.0, "df": max(0, len(groups) - 1)}
    vals_only = [x[0] for x in all_vals]
    ranks = avg_ranks(vals_only)
    group_rsums = [0.0] * len(groups)
    group_ns = [len(g) for g in groups]
    for r, (_, g_idx) in zip(ranks, all_vals):
        group_rsums[g_idx] += r
    h_raw = (12.0 / (n * (n + 1))) * sum((rs * rs) / ng for rs, ng in zip(group_rsums, group_ns) if ng > 0) - 3.0 * (n + 1)
    # Tie correction
    s_vals = sorted(vals_only)
    tie_sum = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and s_vals[j] == s_vals[i]:
            j += 1
        t = j - i
        if t > 1:
            tie_sum += t * t * t - t
        i = j
    denom = 1.0 - tie_sum / (n * n * n - n) if n > 1 else 1.0
    h_val = h_raw / denom if denom > 1e-12 else h_raw
    return {"H": round(h_val, 4), "df": len(groups) - 1}


def summarize_cancr(xs: list[float]) -> dict:
    if not xs:
        return {}
    return {
        "n": len(xs),
        "median": round(median(xs), 4),
        "iqr": round(quantile(xs, 0.75) - quantile(xs, 0.25), 4),
        "p90": round(quantile(xs, 0.90), 4),
        "max": round(max(xs), 4),
        "mean": round(mean(xs), 4),
    }


# ---------------------------------------------------------------------------
# Trace aggregation helpers
# ---------------------------------------------------------------------------
def _prompt_subset_indices(prompts: list[dict], key: str | None = None, val: str | None = None) -> list[int]:
    if key is None:
        return list(range(len(prompts)))
    return [i for i, p in enumerate(prompts) if p[key] == val]


def _avg_arm_cell_regime_n(
    rn_layer: dict,
    arm: str,
    target_str: str,
    prompt_idxs: list[int],
) -> dict:
    """Average Regime-N trace metrics over `prompt_idxs` (and K=4 draws for `direct_matched`)."""
    if arm == "direct_matched":
        draws = rn_layer[arm][target_str]
        rows = [draws[k][pi] for k in range(len(draws)) for pi in prompt_idxs]
    else:
        rows = [rn_layer[arm][target_str][pi] for pi in prompt_idxs]
    return {
        "exact_dlogit": mean([r["exact_dlogit"] for r in rows]),
        "pred_dlogit": mean([r["pred_dlogit"] for r in rows]),
        "direct_linear": mean([r["direct_linear"] for r in rows]),
        "retention": mean([r["retention"] for r in rows]),
    }


def _avg_arm_cell_regime_d(
    rd_layer: dict,
    arm: str,
    target_str: str,
    prompt_idxs: list[int],
) -> dict | None:
    """Average retained (uncapped) Regime-D trace metrics over `prompt_idxs` (and K=4 draws for dm)."""
    if arm == "direct_matched":
        draws = rd_layer[arm][target_str]
        rows = [draws[k][pi] for k in range(len(draws)) for pi in prompt_idxs if not draws[k][pi].get("capped", False)]
    else:
        rows = [rd_layer[arm][target_str][pi] for pi in prompt_idxs if not rd_layer[arm][target_str][pi].get("capped", False)]
    if not rows:
        return None
    effs = [max(r["dlogit_z"], 0.0) / math.sqrt(max(r["kl"], 1e-6)) for r in rows]
    return {
        "exact_dlogit": mean([r["exact_dlogit"] for r in rows]),
        "retention": mean([r["retention"] for r in rows]),
        "eff": mean(effs),
        "dlogit_z": mean([r["dlogit_z"] for r in rows]),
        "kl": mean([r["kl"] for r in rows]),
    }


# ---------------------------------------------------------------------------
# Main Stage C0 scoring function
# ---------------------------------------------------------------------------
def score_c0(
    receipt_path: Path,
    constants_path: Path,
    design_path: Path | None = None,
    n_boot: int = BOOT_RESAMPLES,
    n_perm: int = PERM_RESAMPLES,
    strict_provenance: bool = True,
) -> dict:
    receipt = load_json_maybe_gz(receipt_path)
    const = json.loads(constants_path.read_text(encoding="utf-8"))

    const_sha = sha256_file(constants_path)
    design_sha = sha256_file(design_path) if (design_path and design_path.exists()) else receipt["meta"].get("design_sha256", "")
    receipt_sha = sha256_uncompressed(receipt_path)

    targets = receipt["targets_16"]
    t_strs = [t["string"] for t in targets]
    t_domains = [t["domain"] for t in targets]
    n_t = len(targets)
    prompts = receipt["prompts_8"]
    all_p_idxs = list(range(len(prompts)))
    layers = [int(L) for L in receipt["layers"]]
    rn = receipt["regime_N"]
    rd = receipt["regime_D"]
    lg = const["layer_geometry"]

    # Map frozen pred_gap_D3 and tr_R_over_d per (L, target)
    pred_gap_map: dict[tuple[int, str], float] = {}
    tr_r_over_d_map: dict[int, float] = {}
    for L in layers:
        l_str = str(L)
        tr_r_over_d_map[L] = float(lg[l_str]["trace_R_over_d"])
        for pt in lg[l_str]["per_target"]:
            pred_gap_map[(L, pt["target"])] = float(pt["pred_gap_D3"])

    # =========================================================================
    # GATE C0.0 (All 4 clauses must pass before reading C0.1-C0.4)
    # =========================================================================
    # C0.0.1: Zero-dose sham
    sham_vals = {str(L): float(receipt["shams"][str(L)]) for L in layers}
    c0_0_1_pass = all(v == 0.0 for v in sham_vals.values())

    # C0.0.2: Quantile telescoping error across ALL bf16 traces
    all_tel: list[float] = []
    for L in layers:
        l_str = str(L)
        for reg in (rn[l_str], rd[l_str]):
            for arm in NON_NULL_ARMS:
                for t_str in t_strs:
                    item = reg[arm][t_str]
                    rows = [r for draw in item for r in draw] if arm == "direct_matched" else item
                    for r in rows:
                        if not r.get("capped", False):
                            all_tel.append(abs(float(r["telescoping_err"])))
    tel_p95 = quantile(all_tel, 0.95)
    tel_p999 = quantile(all_tel, 0.999)
    tel_max = max(all_tel) if all_tel else 0.0
    c0_0_2_pass = (tel_p95 <= 0.08) and (tel_p999 <= 0.15)

    # C0.0.3: Regime-D cap_hit_rate <= 2% per (arm, L) on L12-L27 AND |frozen_direct_effect - 3.0|
    cap_rates: dict[str, dict[str, float]] = {}
    fde_errs: list[float] = []
    fde_med_by_al: dict[str, dict[str, float]] = {}
    max_cap_rate = 0.0
    gated_d_layers = [L for L in L12_TO_27 if L in layers]
    for L in gated_d_layers:
        l_str = str(L)
        cap_rates[l_str] = {}
        fde_med_by_al[l_str] = {}
        for arm in NON_NULL_ARMS:
            tot = 0
            capped = 0
            al_errs = []
            for t_str in t_strs:
                item = rd[l_str][arm][t_str]
                rows = [r for draw in item for r in draw] if arm == "direct_matched" else item
                for r in rows:
                    tot += 1
                    if r.get("capped", False):
                        capped += 1
                    else:
                        err = abs(float(r["frozen_direct_effect"]) - 3.0)
                        fde_errs.append(err)
                        al_errs.append(err)
            rate = capped / tot if tot > 0 else 1.0
            cap_rates[l_str][arm] = round(rate, 6)
            fde_med_by_al[l_str][arm] = round(median(al_errs), 6) if al_errs else 0.0
            max_cap_rate = max(max_cap_rate, rate)
    fde_p95 = quantile(fde_errs, 0.95)
    fde_p999 = quantile(fde_errs, 0.999)
    fde_max = max(fde_errs) if fde_errs else 0.0
    c0_0_3_pass = (max_cap_rate <= 0.02) and (fde_p95 <= 0.30) and (fde_p999 <= 0.45)

    # C0.0.4: Provenance
    meta_r_sha = receipt["meta"].get("r_lens_sha256", "")
    meta_c_sha = receipt["meta"].get("constants_sha256", "")
    meta_d_sha = receipt["meta"].get("design_sha256", "")
    if strict_provenance:
        c0_0_4_pass = (
            meta_r_sha == EXPECTED_R_LENS_SHA256
            and meta_c_sha == EXPECTED_CONSTANTS_SHA256
            and const_sha == EXPECTED_CONSTANTS_SHA256
            and (meta_d_sha == EXPECTED_DESIGN_SHA256)
            and (design_sha == EXPECTED_DESIGN_SHA256)
        )
    else:
        c0_0_4_pass = (meta_c_sha == const_sha) and bool(meta_r_sha)

    c0_0_passed = c0_0_1_pass and c0_0_2_pass and c0_0_3_pass and c0_0_4_pass
    gate_c0_0 = {
        "passed": c0_0_passed,
        "C0_0_1_sham": {"passed": c0_0_1_pass, "max_abs_dlogit_by_layer": sham_vals},
        "C0_0_2_telescoping": {
            "passed": c0_0_2_pass,
            "n_traces": len(all_tel),
            "median": round(median(all_tel), 6),
            "p95": round(tel_p95, 6),
            "p99_9": round(tel_p999, 6),
            "max": round(tel_max, 6),
        },
        "C0_0_3_regime_d_cap_and_denom": {
            "passed": c0_0_3_pass,
            "max_cap_hit_rate": round(max_cap_rate, 6),
            "cap_hit_rate_by_layer_arm": cap_rates,
            "n_retained_traces": len(fde_errs),
            "fde_err_p95": round(fde_p95, 6),
            "fde_err_p99_9": round(fde_p999, 6),
            "fde_err_max": round(fde_max, 6),
            "fde_err_median_by_layer_arm": fde_med_by_al,
        },
        "C0_0_4_provenance": {
            "passed": c0_0_4_pass,
            "receipt_uncompressed_sha256": receipt_sha,
            "r_lens_sha256": meta_r_sha,
            "constants_sha256": const_sha,
            "design_sha256": design_sha,
        },
    }

    # =========================================================================
    # Precompute Regime-N prompt-averaged cells for all (L, arm, target)
    # =========================================================================
    def build_rn_grid(p_idxs: list[int]) -> dict[tuple[int, str], list[dict]]:
        grid: dict[tuple[int, str], list[dict]] = {}
        for L in layers:
            l_str = str(L)
            for arm in NON_NULL_ARMS:
                grid[(L, arm)] = [_avg_arm_cell_regime_n(rn[l_str], arm, t_str, p_idxs) for t_str in t_strs]
        return grid

    rn_grid = build_rn_grid(all_p_idxs)

    # =========================================================================
    # C0.1: Within-arm target ranking & transport-arm calibration
    # =========================================================================
    l18_avail = [L for L in L18_PLUS if L in layers]
    early_avail = [L for L in L_EARLY if L in layers]

    def _rho_within_from_grid(grid: dict, l_list: list[int], arms: list[str], t_indices: list[int], use_direct: bool = False) -> float:
        rhos = []
        key_x = "direct_linear" if use_direct else "pred_dlogit"
        for L in l_list:
            for arm in arms:
                cells = grid[(L, arm)]
                xs = [cells[i][key_x] for i in t_indices]
                ys = [cells[i]["exact_dlogit"] for i in t_indices]
                rhos.append(spearman_rho(xs, ys))
        return fisher_z_mean(rhos)

    full_t_idx = list(range(n_t))
    rho_within_l18 = _rho_within_from_grid(rn_grid, l18_avail, NON_NULL_ARMS, full_t_idx, False)
    rho_direct_l18 = _rho_within_from_grid(rn_grid, l18_avail, NON_NULL_ARMS, full_t_idx, True)
    delta_rho_l18 = rho_within_l18 - rho_direct_l18
    rho_within_rmf = _rho_within_from_grid(rn_grid, l18_avail, ["R_mf"], full_t_idx, False)
    rho_within_early = _rho_within_from_grid(rn_grid, early_avail, NON_NULL_ARMS, full_t_idx, False)

    # Per-(arm, L) Spearman table
    rho_by_layer_arm: dict[str, dict[str, float]] = {}
    for L in layers:
        rho_by_layer_arm[str(L)] = {
            arm: round(
                spearman_rho(
                    [c["pred_dlogit"] for c in rn_grid[(L, arm)]],
                    [c["exact_dlogit"] for c in rn_grid[(L, arm)]],
                ),
                6,
            )
            for arm in NON_NULL_ARMS
        }

    # Joint 16-target label permutation for p_1 (16 L18+ cells) and p_perm(R_mf) (4 L18+ layers)
    rng_p1 = random.Random(SEED_PERM)
    ge_p1 = 0
    ge_rmf = 0
    perm_idx = list(range(n_t))
    for _ in range(n_perm):
        rng_p1.shuffle(perm_idx)
        rhos_all = []
        rhos_rmf = []
        for L in l18_avail:
            for arm in NON_NULL_ARMS:
                cells = rn_grid[(L, arm)]
                xs = [cells[i]["pred_dlogit"] for i in range(n_t)]
                ys = [cells[perm_idx[i]]["exact_dlogit"] for i in range(n_t)]
                r = spearman_rho(xs, ys)
                rhos_all.append(r)
                if arm == "R_mf":
                    rhos_rmf.append(r)
        if fisher_z_mean(rhos_all) >= rho_within_l18 - 1e-12:
            ge_p1 += 1
        if fisher_z_mean(rhos_rmf) >= rho_within_rmf - 1e-12:
            ge_rmf += 1
    p_1 = (ge_p1 + 1) / (n_perm + 1)
    p_perm_rmf = (ge_rmf + 1) / (n_perm + 1)

    # BCa CIs for C0.1a: Delta_rho, rho_within(R_mf), and descriptive early-vs-late
    delta_rho_ci = bca_cluster_ci(
        n_t,
        lambda idx: _rho_within_from_grid(rn_grid, l18_avail, NON_NULL_ARMS, idx, False)
        - _rho_within_from_grid(rn_grid, l18_avail, NON_NULL_ARMS, idx, True),
        alphas=(0.05, 0.025, 0.975),
        n_boot=n_boot,
        seed=SEED_BOOT + 1,
    )
    rho_rmf_ci = bca_cluster_ci(
        n_t,
        lambda idx: _rho_within_from_grid(rn_grid, l18_avail, ["R_mf"], idx, False),
        alphas=(0.05, 0.025, 0.975),
        n_boot=n_boot,
        seed=SEED_BOOT + 2,
    )
    early_vs_late_ci = bca_cluster_ci(
        n_t,
        lambda idx: _rho_within_from_grid(rn_grid, l18_avail, NON_NULL_ARMS, idx, False)
        - _rho_within_from_grid(rn_grid, early_avail, NON_NULL_ARMS, idx, False),
        alphas=(0.025, 0.975),
        n_boot=n_boot,
        seed=SEED_BOOT + 3,
    )

    # C0.1b: Per-arm direct-floored median fractional error on L18+
    c0_1b_arm_meds: dict[str, float] = {}
    for arm in NON_NULL_ARMS:
        ferrs = []
        for L in l18_avail:
            for c in rn_grid[(L, arm)]:
                denom = max(abs(c["pred_dlogit"]), 0.5 * abs(c["direct_linear"]), 1e-6)
                ferrs.append(abs(c["exact_dlogit"] - c["pred_dlogit"]) / denom)
        c0_1b_arm_meds[arm] = round(median(ferrs), 6)

    # Descriptive cross-arm within-layer Spearman
    cross_arm_rho = {}
    for L in layers:
        xs, ys = [], []
        for arm in NON_NULL_ARMS:
            for c in rn_grid[(L, arm)]:
                xs.append(c["pred_dlogit"])
                ys.append(c["exact_dlogit"])
        cross_arm_rho[str(L)] = round(spearman_rho(xs, ys), 6)

    # =========================================================================
    # C0.2: Pooled & per-layer lens-calibrated gap at equal direct gain (Regime D)
    # =========================================================================
    def build_rd_grid(p_idxs: list[int]) -> dict[tuple[int, str], list[dict | None]]:
        grid: dict[tuple[int, str], list[dict | None]] = {}
        for L in layers:
            l_str = str(L)
            for arm in NON_NULL_ARMS:
                grid[(L, arm)] = [_avg_arm_cell_regime_d(rd[l_str], arm, t_str, p_idxs) for t_str in t_strs]
        return grid

    rd_grid = build_rd_grid(all_p_idxs)

    def _gap_ratios_for_indices(grid: dict, l_list: list[int], t_indices: list[int]) -> list[float]:
        ratios = []
        for L in l_list:
            for i in t_indices:
                c_mf = grid[(L, "R_mf")][i]
                c_dm = grid[(L, "direct_matched")][i]
                if c_mf is not None and c_dm is not None:
                    gap = c_mf["exact_dlogit"] - c_dm["exact_dlogit"]
                    ratios.append(gap / pred_gap_map[(L, t_strs[i])])
        return ratios

    c0_2_pooled_ratio = median(_gap_ratios_for_indices(rd_grid, gated_d_layers, full_t_idx))
    c0_2_pooled_ci = bca_cluster_ci(
        n_t,
        lambda idx: median(_gap_ratios_for_indices(rd_grid, gated_d_layers, idx)),
        alphas=(0.05, 0.025, 0.975),
        n_boot=n_boot,
        seed=SEED_BOOT + 4,
    )
    c0_2_per_layer_ratio = {
        str(L): round(median(_gap_ratios_for_indices(rd_grid, [L], full_t_idx)), 6) for L in layers
    }

    # Exact 2^16 paired sign-flip p_2 on 16 target-mean L12-L27 gaps
    target_mean_gaps: list[float] = []
    for i in range(n_t):
        gs = []
        for L in gated_d_layers:
            c_mf = rd_grid[(L, "R_mf")][i]
            c_dm = rd_grid[(L, "direct_matched")][i]
            if c_mf is not None and c_dm is not None:
                gs.append(c_mf["exact_dlogit"] - c_dm["exact_dlogit"])
        target_mean_gaps.append(mean(gs))
    p_2 = exact_sign_flip_p_one_sided(target_mean_gaps)

    # C0.2(b): logit_lens contraction bounds on L18+ (R_mf descriptive)
    ll_rets_l18_pooled: list[float] = []
    ll_rets_by_layer: dict[str, float] = {}
    rmf_rets_l18_pooled: list[float] = []
    dm_minus_ll_l18: list[float] = []
    for L in l18_avail:
        l_ll = [c["retention"] for c in rd_grid[(L, "logit_lens")] if c is not None]
        l_rmf = [c["retention"] for c in rd_grid[(L, "R_mf")] if c is not None]
        ll_rets_by_layer[str(L)] = round(median(l_ll), 6)
        ll_rets_l18_pooled.extend(l_ll)
        rmf_rets_l18_pooled.extend(l_rmf)
        for i in range(n_t):
            c_dm = rd_grid[(L, "direct_matched")][i]
            c_ll = rd_grid[(L, "logit_lens")][i]
            if c_dm is not None and c_ll is not None:
                dm_minus_ll_l18.append(c_dm["exact_dlogit"] - c_ll["exact_dlogit"])
    ll_ret_l18_med = median(ll_rets_l18_pooled)
    rmf_ret_l18_med = median(rmf_rets_l18_pooled)
    m4_diag_val = abs(median(dm_minus_ll_l18)) if dm_minus_ll_l18 else 0.0

    analogue_layers_avail = [L for L in L_ANALOGUE if L in layers]

    # =========================================================================
    # C0.3: u_t-specific contraction vs. 32-draw isotropic Jacobian diagonal
    # =========================================================================
    vjv_by_layer: dict[int, float] = {}
    cancr_self_by_layer: dict[str, dict] = {}
    for L in layers:
        l_str = str(L)
        v_vals = [r["vJv"] for draw_rows in rn[l_str]["rand_iso"] for r in draw_rows]
        cs_vals = [r["canc_r_self"] for draw_rows in rn[l_str]["rand_iso"] for r in draw_rows]
        vjv_by_layer[L] = mean(v_vals)
        cancr_self_by_layer[l_str] = summarize_cancr(cs_vals)

    d_t_vals: list[float] = []
    d_t_tr_sens: list[float] = []
    sign_counts_by_layer: dict[str, dict] = {}
    for L in layers:
        diffs_l = [vjv_by_layer[L] - rn_grid[(L, "logit_lens")][i]["retention"] for i in range(n_t)]
        sign_counts_by_layer[str(L)] = {
            "vJv": round(vjv_by_layer[L], 6),
            "tr_R_over_d": round(tr_r_over_d_map[L], 6),
            "vJv_minus_tr": round(vjv_by_layer[L] - tr_r_over_d_map[L], 6),
            "mean_d_L": round(mean(diffs_l), 6),
            "pos_count": sum(1 for x in diffs_l if x > 0.0),
            "n_targets": n_t,
        }
    for i in range(n_t):
        d_t_vals.append(mean([vjv_by_layer[L] - rn_grid[(L, "logit_lens")][i]["retention"] for L in l18_avail]))
        d_t_tr_sens.append(mean([tr_r_over_d_map[L] - rn_grid[(L, "logit_lens")][i]["retention"] for L in l18_avail]))

    p_3 = exact_sign_flip_p_one_sided(d_t_vals)
    d_t_ci = bca_cluster_ci(
        n_t,
        lambda idx: mean([d_t_vals[i] for i in idx]),
        alphas=(0.05, 0.025, 0.975),
        n_boot=n_boot,
        seed=SEED_BOOT + 5,
    )
    p_3_tr_sens = exact_sign_flip_p_one_sided(d_t_tr_sens)
    d_t_tr_sens_ci = bca_cluster_ci(
        n_t,
        lambda idx: mean([d_t_tr_sens[i] for i in idx]),
        alphas=(0.05, 0.025, 0.975),
        n_boot=n_boot,
        seed=SEED_BOOT + 6,
    )
    baseline_shift_l18 = mean([vjv_by_layer[L] - tr_r_over_d_map[L] for L in l18_avail])

    # C0.3 descriptives: within-arm Retention IQR/median + Kruskal-Wallis across 16 targets
    ret_het_desc = {}
    for arm in NON_NULL_ARMS:
        t_groups = [[rn_grid[(L, arm)][i]["retention"] for L in l18_avail] for i in range(n_t)]
        flat = [v for g in t_groups for v in g]
        med_v = median(flat)
        iqr_v = quantile(flat, 0.75) - quantile(flat, 0.25)
        ret_het_desc[arm] = {
            "median": round(med_v, 6),
            "iqr_over_median": round(iqr_v / max(abs(med_v), 1e-6), 4),
            "kruskal_wallis": kruskal_wallis_h(t_groups),
        }

    # =========================================================================
    # C0.4: Equal-direct specificity efficiency (Regime D) & all-cross-domain null (Regime N)
    # =========================================================================
    eff_rmf_t: list[float] = []
    eff_rr_t: list[float] = []
    l_t_vals: list[float] = []
    l_t_excess_125: list[float] = []
    dz_rmf_t, dz_rr_t, kl_rmf_t, kl_rr_t = [], [], [], []
    for i in range(n_t):
        t_str = t_strs[i]
        e_mf_traces, e_rr_traces = [], []
        z_mf_tr, z_rr_tr, k_mf_tr, k_rr_tr = [], [], [], []
        for L in l18_avail:
            l_str = str(L)
            for r in rd[l_str]["R_mf"][t_str]:
                if not r.get("capped", False):
                    e_mf_traces.append(max(r["dlogit_z"], 0.0) / math.sqrt(max(r["kl"], 1e-6)))
                    z_mf_tr.append(r["dlogit_z"])
                    k_mf_tr.append(r["kl"])
            for r in rd[l_str]["R_ridge"][t_str]:
                if not r.get("capped", False):
                    e_rr_traces.append(max(r["dlogit_z"], 0.0) / math.sqrt(max(r["kl"], 1e-6)))
                    z_rr_tr.append(r["dlogit_z"])
                    k_rr_tr.append(r["kl"])
        em = mean(e_mf_traces)
        er = mean(e_rr_traces)
        eff_rmf_t.append(em)
        eff_rr_t.append(er)
        lt = math.log(max(er, 1e-6) / max(em, 1e-6))
        l_t_vals.append(lt)
        l_t_excess_125.append(lt - math.log(1.25))
        dz_rmf_t.append(mean(z_mf_tr))
        dz_rr_t.append(mean(z_rr_tr))
        kl_rmf_t.append(mean(k_mf_tr))
        kl_rr_t.append(mean(k_rr_tr))

    p_4 = exact_sign_flip_p_one_sided(l_t_vals)
    p_4_125 = exact_sign_flip_p_one_sided(l_t_excess_125)
    geo_mean_eff_ratio = math.exp(mean(l_t_vals))
    count_rr_gt_rmf = sum(1 for a, b in zip(eff_rr_t, eff_rmf_t) if a > b)
    count_rr_ge_125 = sum(1 for a, b in zip(eff_rr_t, eff_rmf_t) if a >= 1.25 * b)
    eff_ratio_ci = bca_cluster_ci(
        n_t,
        lambda idx: math.exp(mean([l_t_vals[i] for i in idx])),
        alphas=(0.05, 0.025, 0.975),
        n_boot=n_boot,
        seed=SEED_BOOT + 7,
    )

    # All-cross-domain null in Regime N (L18+): 192 cross pairs vs 512 rand_iso pairs
    # For each readout target ti and write target tj (domain[tj] != domain[ti]):
    #   prompt-and-L18+-averaged |dlogit(ti; R_mf(tj))|
    cross_pair_vals: list[float] = []
    cross_per_target_mean: list[float] = []
    cross_per_trace_vals: list[float] = []
    same_domain_pair_vals: list[float] = []
    for ti in range(n_t):
        ti_pairs = []
        for tj in range(n_t):
            if ti == tj:
                continue
            tj_str = t_strs[tj]
            # Average |dlogit| over all prompts and L18+ layers for this (ti, tj) pair
            vals_tr = [
                abs(float(r["all_16_dlogits"][ti]))
                for L in l18_avail
                for r in rn[str(L)]["R_mf"][tj_str]
            ]
            pair_avg = mean(vals_tr)
            if t_domains[tj] != t_domains[ti]:
                cross_pair_vals.append(pair_avg)
                ti_pairs.append(pair_avg)
                cross_per_trace_vals.extend(vals_tr)
            else:
                same_domain_pair_vals.append(pair_avg)
        cross_per_target_mean.append(mean(ti_pairs))

    # 512 (m, ti) rand_iso pairs on L18+ (averaged over prompts & L18+ per (m, ti))
    m_rand_actual = len(rn[str(l18_avail[0])]["rand_iso"]) if l18_avail else 0
    rand_pair_vals: list[float] = []
    rand_per_target_mean: list[float] = []
    rand_per_trace_vals: list[float] = []
    for ti in range(n_t):
        ti_r = []
        for m in range(m_rand_actual):
            vals_tr = [
                abs(float(r["all_16_dlogits"][ti]))
                for L in l18_avail
                for r in rn[str(L)]["rand_iso"][m]
            ]
            pair_avg = mean(vals_tr)
            rand_pair_vals.append(pair_avg)
            ti_r.append(pair_avg)
            rand_per_trace_vals.extend(vals_tr)
        rand_per_target_mean.append(mean(ti_r))

    f_mean = mean(rand_pair_vals)
    f_p95 = quantile(rand_pair_vals, 0.95)
    cross_mean = mean(cross_pair_vals)
    cross_p95 = quantile(cross_pair_vals, 0.95)
    cross_max = max(cross_pair_vals) if cross_pair_vals else 0.0
    cross_cnt_gt_2f = sum(1 for v in cross_pair_vals if v > 2.0 * f_mean)

    paired_cross_minus_rand = [c - r for c, r in zip(cross_per_target_mean, rand_per_target_mean)]
    cross_diff_ci90 = bca_cluster_ci(
        n_t,
        lambda idx: mean([paired_cross_minus_rand[i] for i in idx]),
        alphas=(0.90, 0.95),
        n_boot=n_boot,
        seed=SEED_BOOT + 8,
    )
    cross_agg_pass = (
        (cross_mean <= f_mean + 0.10)
        and (cross_diff_ci90["q_0.9"] <= 0.15)
        and (cross_p95 <= f_p95 + 0.20)
    )
    cross_pair_pass = (cross_max <= 1.75 * f_p95) and (cross_cnt_gt_2f <= 5)

    # =========================================================================
    # 4-Test Holm-Bonferroni Family (alpha = 0.05) & Prediction Verdicts
    # =========================================================================
    holm = holm_bonferroni({"C0.1": p_1, "C0.2": p_2, "C0.3": p_3, "C0.4": p_4}, alpha=0.05)

    c0_1a_pass = (
        (rho_within_l18 >= 0.30)
        and (rho_within_rmf >= 0.25)
        and (p_perm_rmf < 0.05)
        and holm["C0.1"]["rejected"]
        and (delta_rho_ci["q_0.05"] > 0.0)
    )
    c0_1b_pass = (c0_1b_arm_meds["R_mf"] <= 0.30) and (c0_1b_arm_meds["R_ridge"] <= 0.30)
    c0_1_pass = c0_1a_pass and c0_1b_pass

    c0_2a_pass = (
        (c0_2_pooled_ratio >= 0.70)
        and (c0_2_pooled_ci["q_0.05"] > 0.50)
        and all(c0_2_per_layer_ratio[str(L)] >= 0.50 for L in analogue_layers_avail)
    )
    c0_2b_pass = (ll_ret_l18_med <= 0.75) and all(ll_rets_by_layer[str(L)] <= 0.85 for L in l18_avail)
    c0_2_pass = c0_2a_pass and c0_2b_pass and holm["C0.2"]["rejected"]

    c0_3_pass = holm["C0.3"]["rejected"] and (d_t_ci["q_0.05"] > 0.0)

    c0_4_eff_pass = holm["C0.4"]["rejected"] and (geo_mean_eff_ratio >= 1.25) and (count_rr_gt_rmf >= 12)
    c0_4_null_pass = cross_agg_pass and cross_pair_pass
    c0_4_pass = c0_4_eff_pass and c0_4_null_pass

    # Stage C1 Gate: C0.0 AND C0.1 AND C0.2 AND C0.3
    stage_c1_gate_pass = c0_0_passed and c0_1_pass and c0_2_pass and c0_3_pass

    # =========================================================================
    # Registered 2 x 2 Format x Style Moderation Diagnostics (non-gating)
    # =========================================================================
    moderation = {}
    for factor, lv_a, lv_b in [("format", "bare", "chatml"), ("style", "narrative", "analytical")]:
        idx_a = _prompt_subset_indices(prompts, factor, lv_a)
        idx_b = _prompt_subset_indices(prompts, factor, lv_b)
        if not idx_a or not idx_b:
            continue
        rn_a, rn_b = build_rn_grid(idx_a), build_rn_grid(idx_b)
        rd_a, rd_b = build_rd_grid(idx_a), build_rd_grid(idx_b)
        rho_a = _rho_within_from_grid(rn_a, l18_avail, NON_NULL_ARMS, full_t_idx, False)
        rho_b = _rho_within_from_grid(rn_b, l18_avail, NON_NULL_ARMS, full_t_idx, False)
        rho_diff_ci = bca_cluster_ci(
            n_t,
            lambda idx: _rho_within_from_grid(rn_a, l18_avail, NON_NULL_ARMS, idx, False)
            - _rho_within_from_grid(rn_b, l18_avail, NON_NULL_ARMS, idx, False),
            alphas=(0.025, 0.975),
            n_boot=n_boot,
            seed=SEED_BOOT + (10 if factor == "format" else 12),
        )
        gap_a = median(_gap_ratios_for_indices(rd_a, gated_d_layers, full_t_idx))
        gap_b = median(_gap_ratios_for_indices(rd_b, gated_d_layers, full_t_idx))
        gap_diff_ci = bca_cluster_ci(
            n_t,
            lambda idx: median(_gap_ratios_for_indices(rd_a, gated_d_layers, idx))
            - median(_gap_ratios_for_indices(rd_b, gated_d_layers, idx)),
            alphas=(0.025, 0.975),
            n_boot=n_boot,
            seed=SEED_BOOT + (11 if factor == "format" else 13),
        )
        moderation[f"{lv_a}_vs_{lv_b}"] = {
            "C0_1a_rho_within": {lv_a: round(rho_a, 6), lv_b: round(rho_b, 6), "diff_ci95": rho_diff_ci},
            "C0_2a_gap_ratio": {lv_a: round(gap_a, 6), lv_b: round(gap_b, 6), "diff_ci95": gap_diff_ci},
        }

    scorecard = {
        "schema_version": "stage_c0_scorecard_v1",
        "meta": receipt["meta"],
        "stage_c1_gate_passed": stage_c1_gate_pass,
        "verdicts": {
            "C0.0": "PASS" if c0_0_passed else "FAIL",
            "C0.1": ("PASS" if c0_1_pass else "FAIL") if c0_0_passed else "BLOCKED_BY_C0.0",
            "C0.2": ("PASS" if c0_2_pass else "FAIL") if c0_0_passed else "BLOCKED_BY_C0.0",
            "C0.3": ("PASS" if c0_3_pass else "FAIL") if c0_0_passed else "BLOCKED_BY_C0.0",
            "C0.4": ("PASS" if c0_4_pass else "FAIL") if c0_0_passed else "BLOCKED_BY_C0.0",
        },
        "holm_family": holm,
        "gate_C0_0": gate_c0_0,
        "C0_1": {
            "passed": c0_1_pass,
            "C0_1a_passed": c0_1a_pass,
            "C0_1b_passed": c0_1b_pass,
            "rho_within_L18_plus": round(rho_within_l18, 6),
            "rho_direct_L18_plus": round(rho_direct_l18, 6),
            "delta_rho_L18_plus": round(delta_rho_l18, 6),
            "delta_rho_bca_ci": delta_rho_ci,
            "rho_within_R_mf_L18_plus": round(rho_within_rmf, 6),
            "rho_within_R_mf_bca_ci": rho_rmf_ci,
            "p_1_joint_perm": round(p_1, 6),
            "p_perm_R_mf": round(p_perm_rmf, 6),
            "C0_1b_median_frac_err_L18_plus": c0_1b_arm_meds,
            "descriptive_rho_within_early_L6_L12": round(rho_within_early, 6),
            "descriptive_late_minus_early_bca_ci": early_vs_late_ci,
            "descriptive_rho_by_layer_arm": rho_by_layer_arm,
            "descriptive_cross_arm_rho_by_layer": cross_arm_rho,
        },
        "C0_2": {
            "passed": c0_2_pass,
            "C0_2a_passed": c0_2a_pass,
            "C0_2b_passed": c0_2b_pass,
            "pooled_L12_27_gap_ratio_median": round(c0_2_pooled_ratio, 6),
            "pooled_L12_27_gap_ratio_bca_ci": c0_2_pooled_ci,
            "per_layer_gap_ratio_median": c0_2_per_layer_ratio,
            "p_2_sign_flip": round(p_2, 6),
            "L18_plus_Retention_logit_lens_pooled_median": round(ll_ret_l18_med, 6),
            "L18_plus_Retention_logit_lens_by_layer": ll_rets_by_layer,
            "descriptive_L18_plus_Retention_R_mf_pooled_median": round(rmf_ret_l18_med, 6),
            "descriptive_m4_abs_median_dm_minus_ll": round(m4_diag_val, 6),
            "descriptive_m4_flagged_gt_0_25": m4_diag_val > 0.25,
        },
        "C0_3": {
            "passed": c0_3_pass,
            "mean_d_t": round(mean(d_t_vals), 6),
            "d_t_bca_ci": d_t_ci,
            "p_3_sign_flip": round(p_3, 6),
            "per_target_d_t": {t_strs[i]: round(d_t_vals[i], 6) for i in range(n_t)},
            "sensitivity_vs_tr_R_over_d": {
                "mean_d_t_tr": round(mean(d_t_tr_sens), 6),
                "bca_ci": d_t_tr_sens_ci,
                "p_sign_flip": round(p_3_tr_sens, 6),
                "mean_L18_plus_vJv_minus_tr": round(baseline_shift_l18, 6),
                "baseline_shift_flagged_lt_neg0_03": baseline_shift_l18 < -0.03,
            },
            "by_layer_sign_counts": sign_counts_by_layer,
            "descriptive_within_arm_retention_heterogeneity": ret_het_desc,
            "descriptive_cancr_self_by_layer": cancr_self_by_layer,
        },
        "C0_4": {
            "passed": c0_4_pass,
            "efficiency_passed": c0_4_eff_pass,
            "cross_domain_null_passed": c0_4_null_pass,
            "geo_mean_eff_ratio_Rridge_over_Rmf": round(geo_mean_eff_ratio, 6),
            "geo_mean_eff_ratio_bca_ci": eff_ratio_ci,
            "count_Rridge_gt_Rmf": count_rr_gt_rmf,
            "count_Rridge_ge_1_25x_Rmf": count_rr_ge_125,
            "p_4_sign_flip": round(p_4, 6),
            "descriptive_p_4_excess_1_25x": round(p_4_125, 6),
            "descriptive_mean_dlogit_z": {"R_mf": round(mean(dz_rmf_t), 4), "R_ridge": round(mean(dz_rr_t), 4)},
            "descriptive_mean_kl": {"R_mf": round(mean(kl_rmf_t), 5), "R_ridge": round(mean(kl_rr_t), 5)},
            "cross_domain_null": {
                "aggregate_passed": cross_agg_pass,
                "per_pair_passed": cross_pair_pass,
                "n_cross_pairs": len(cross_pair_vals),
                "n_rand_pairs": len(rand_pair_vals),
                "F_mean": round(f_mean, 6),
                "F_p95": round(f_p95, 6),
                "cross_mean": round(cross_mean, 6),
                "cross_p95": round(cross_p95, 6),
                "cross_max": round(cross_max, 6),
                "cross_count_gt_2F_mean": cross_cnt_gt_2f,
                "paired_diff_bca_ci": cross_diff_ci90,
                "descriptive_per_trace_p95": {
                    "cross": round(quantile(cross_per_trace_vals, 0.95), 6),
                    "rand_iso": round(quantile(rand_per_trace_vals, 0.95), 6),
                },
                "descriptive_same_domain_wrong_targets": {
                    "n_pairs": len(same_domain_pair_vals),
                    "mean": round(mean(same_domain_pair_vals), 6),
                    "max": round(max(same_domain_pair_vals), 6) if same_domain_pair_vals else 0.0,
                },
            },
        },
        "moderation_2x2": moderation,
    }
    return scorecard


def render_scorecard_md(sc: dict) -> str:
    v = sc["verdicts"]
    g = sc["gate_C0_0"]
    c1, c2, c3, c4 = sc["C0_1"], sc["C0_2"], sc["C0_3"], sc["C0_4"]
    h = sc["holm_family"]
    lines = [
        "# Stage C0 Scorecard (OLMo 3 7B Registered v1)",
        "",
        f"- **Stage C1 Gate (`C0.0` + `C0.1` + `C0.2` + `C0.3`):** **{'PASS' if sc['stage_c1_gate_passed'] else 'FAIL'}**",
        f"- **Receipt SHA-256 (uncompressed):** `{g['C0_0_4_provenance']['receipt_uncompressed_sha256']}`",
        f"- **Constants SHA-256:** `{g['C0_0_4_provenance']['constants_sha256']}`",
        f"- **Design SHA-256:** `{g['C0_0_4_provenance']['design_sha256']}`",
        "",
        "## Gate & Prediction Verdicts",
        "",
        "| ID | Verdict | Key Metric | Holm `p_raw` (`p_holm`) |",
        "|---|---|---|---|",
        f"| **C0.0** | **{v['C0.0']}** | sham `{max(g['C0_0_1_sham']['max_abs_dlogit_by_layer'].values())}`, tel p95 `{g['C0_0_2_telescoping']['p95']:.4f}`, cap max `{g['C0_0_3_regime_d_cap_and_denom']['max_cap_hit_rate']:.4f}`, fde p95 `{g['C0_0_3_regime_d_cap_and_denom']['fde_err_p95']:.4f}` | — |",
        f"| **C0.1** | **{v['C0.1']}** | `ρ_within={c1['rho_within_L18_plus']:.4f}`, `ρ(R_mf)={c1['rho_within_R_mf_L18_plus']:.4f}` (`p={c1['p_perm_R_mf']:.4f}`), `Δρ={c1['delta_rho_L18_plus']:.4f}` (`q05={c1['delta_rho_bca_ci']['q_0.05']:.4f}`), `frac_err(R_mf/R_ridge)={c1['C0_1b_median_frac_err_L18_plus']['R_mf']:.3f}/{c1['C0_1b_median_frac_err_L18_plus']['R_ridge']:.3f}` | `{h['C0.1']['p_raw']:.6f}` (`{h['C0.1']['p_holm']:.6f}`) |",
        f"| **C0.2** | **{v['C0.2']}** | pooled gap ratio `{c2['pooled_L12_27_gap_ratio_median']:.4f}` (`q05={c2['pooled_L12_27_gap_ratio_bca_ci']['q_0.05']:.4f}`), `Ret(ll)={c2['L18_plus_Retention_logit_lens_pooled_median']:.4f}` | `{h['C0.2']['p_raw']:.6f}` (`{h['C0.2']['p_holm']:.6f}`) |",
        f"| **C0.3** | **{v['C0.3']}** | `mean(d_t)={c3['mean_d_t']:+.4f}` (`q05={c3['d_t_bca_ci']['q_0.05']:+.4f}`) | `{h['C0.3']['p_raw']:.6f}` (`{h['C0.3']['p_holm']:.6f}`) |",
        f"| **C0.4** | **{v['C0.4']}** | `geo_ratio={c4['geo_mean_eff_ratio_Rridge_over_Rmf']:.3f}x` (`{c4['count_Rridge_gt_Rmf']}/16`), cross mean `{c4['cross_domain_null']['cross_mean']:.3f}` vs F `{c4['cross_domain_null']['F_mean']:.3f}`, max `{c4['cross_domain_null']['cross_max']:.3f}` | `{h['C0.4']['p_raw']:.6f}` (`{h['C0.4']['p_holm']:.6f}`) |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--constants", type=Path, default=repo_root / "olmo3_virtual_and_highway/c0_prereg_constants.json")
    ap.add_argument("--design", type=Path, default=repo_root / "olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0-DESIGN.md")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=BOOT_RESAMPLES)
    ap.add_argument("--n-perm", type=int, default=PERM_RESAMPLES)
    ap.add_argument("--no-strict-provenance", action="store_true")
    args = ap.parse_args()

    sc = score_c0(
        receipt_path=args.receipt,
        constants_path=args.constants,
        design_path=args.design,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        strict_provenance=not args.no_strict_provenance,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sc_json = args.out_dir / "scorecard.json"
    sc_md = args.out_dir / "SCORECARD.md"
    sc_json.write_text(json.dumps(sc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    sc_md.write_text(render_scorecard_md(sc), encoding="utf-8")
    print(f"Wrote {sc_json} (sha256={sha256_file(sc_json)})")
    print(f"Wrote {sc_md} (sha256={sha256_file(sc_md)})")
    print(f"Verdicts: {sc['verdicts']} | Stage C1 gate: {sc['stage_c1_gate_passed']}")


if __name__ == "__main__":
    main()
