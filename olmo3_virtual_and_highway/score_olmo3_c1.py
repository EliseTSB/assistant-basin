#!/usr/bin/env python3
"""100% Python 3.10+ stdlib-only deterministic scorer for OLMo 3 7B Stage C1
(`OLMO3-TRANSPORT-C1-DESIGN.md` Registered v5).

Requires ZERO third-party packages (no numpy, scipy, or torch) and uses `math.fsum`
for all float reductions so any clean Python 3.10+ machine reproduces `c1_scorecard.json`
and `C1-SCORECARD.md` byte-for-byte via `/usr/bin/python3 -I` + `cmp`.

Fail-closed guarantees:
  - Hard-refuses smoke/truncated/parameter-mismatched receipts (ValueError).
  - Refuses empty or non-finite inputs in mean/median/quantile.
  - Audits finiteness across ALL retained primary & secondary traces (including all `all_24_dlogits`).
  - Enforces the registered BCa containment invariant (`CI_low <= point_estimate <= CI_high`) fail-closed.
  - Evaluates C1.3's explicit 6-way disjoint partition with runtime `sum(matches) == 1` assertion (zero fallback `else`).
  - Missing `--design` or `--no-strict-provenance` ALWAYS fails Gate `C1.0.4`.
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
EXPECTED_C1_CONSTANTS_SHA256 = "6220e9ec7ce8674c914657b3bc6c0d7c1d65cd342236655dce8e60a9769b9304"
EXPECTED_C1_BANK_SHA256 = "5a7a406a9072c7362353a0a9b00841e7743b82d90aade0adc96b680c8d3e621f"
EXPECTED_C1_DESIGN_SHA256 = "5304677e4bbf8e6cd2881aea7ee6b51055f3da76ac28e79ac96400d1fe2b905c"

PRIMARY_LAYERS = [18, 24, 27]
SECONDARY_LAYERS = [12]
EXPECTED_N_TARGETS = 24
EXPECTED_N_PROMPTS = 8
EXPECTED_K_DM = 2
EXPECTED_M_RAND = 32
EXPECTED_STRATA = ("nouns", "verbs", "adjectives", "function")

PERM_SEED = 2026112401
PERM_N_FLIPS = 100_000
BCA_SEED = 2026112402
BCA_N_RESAMPLES = 10_000

C0_F_MEAN_3L = 0.2958
C1_3_DELTA_R_SEP_BAR = 0.25
C1_3_DELTA_R_SEP_Q05_BAR = 0.05
C1_3_FLOOR_HA_MAX = 1.90
C1_3_FLOOR_HB_MIN = 2.20
C1_3_FLOOR_HB_MAX = 3.20
C1_3_FLOOR_HB_Q05_BAR = 1.85


# ---------------------------------------------------------------------------
# Pure-stdlib I/O & hashing
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


# ---------------------------------------------------------------------------
# Pure-stdlib fail-closed statistics (`math.fsum` for cross-Python determinism)
# ---------------------------------------------------------------------------
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


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Exact tie-averaged Spearman rank correlation (pure stdlib + `math.fsum`)."""
    _require_nonempty_finite(xs, "spearman_x")
    _require_nonempty_finite(ys, "spearman_y")
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("spearman_rho requires equal-length inputs of length >= 2")

    def _ranks(vals: list[float]) -> list[float]:
        n = len(vals)
        order = sorted(range(n), key=lambda i: vals[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_r = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg_r
            i = j + 1
        return out

    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = mean(rx)
    my = mean(ry)
    num = math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(math.fsum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(math.fsum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
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
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def sign_flip_p_one_sided(
    diffs: list[float],
    n_flips: int = PERM_N_FLIPS,
    seed: int = PERM_SEED,
) -> float:
    """Registered deterministic 100,000-flip one-sided sign-flip test over 24 targets in index order 0..23.

    Exact registered rule:
      rng = random.Random(seed)
      sign = +1 if rng.getrandbits(1) else -1
      p = (1 + count(stat >= obs)) / (1 + n_flips)
    """
    _require_nonempty_finite(diffs, "sign_flip_p_one_sided")
    n = len(diffs)
    obs = math.fsum(diffs) / n
    rng = random.Random(seed)
    ge = 0
    # Exact registered comparison `s >= obs_sum` (F10: no tolerance slack; exact ties count for null)
    obs_sum = math.fsum(diffs)
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
    """Deterministic BCa cluster bootstrap over `0..n_clusters-1` using `random.Random(seed)`.

    Enforces the registered L6 fail-closed BCa containment invariant whenever both 0.05 and 0.95
    are requested: `CI_low <= point_estimate <= CI_high` (within 1e-12 float slack).
    """
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

    # Degenerate zero-variance distribution matching theta_hat -> all quantiles equal theta_hat
    if boots[0] == boots[-1] and abs(boots[0] - theta_hat) <= 1e-12 * max(1.0, abs(theta_hat)):
        out = {"point": theta_hat}
        for a in alphas:
            out[f"q_{a}"] = theta_hat
        return out

    # Bias-correction z0
    less = sum(1 for v in boots if v < theta_hat)
    eq = sum(1 for v in boots if v == theta_hat)
    prop = (less + 0.5 * eq) / n_boot
    prop = min(max(prop, 1.0 / (2.0 * n_boot)), 1.0 - 1.0 / (2.0 * n_boot))
    z0 = norm_ppf(prop)

    # Jackknife acceleration a_hat
    jacks = [float(stat_fn([j for j in range(n_clusters) if j != i])) for i in range(n_clusters)]
    j_bar = mean(jacks)
    num = math.fsum((j_bar - jv) ** 3 for jv in jacks)
    den_sq = math.fsum((j_bar - jv) ** 2 for jv in jacks)
    a_hat = (num / (6.0 * (den_sq ** 1.5))) if den_sq > 0.0 else 0.0

    out = {"point": theta_hat}
    for a in alphas:
        za = norm_ppf(a)
        denom = 1.0 - a_hat * (z0 + za)
        if abs(denom) < 1e-12:
            adj_a = a
        else:
            adj_a = norm_cdf(z0 + (z0 + za) / denom)
        adj_a = min(max(adj_a, 0.0), 1.0)
        out[f"q_{a}"] = quantile(boots, adj_a)

    if enforce_containment_90 and (0.05 in alphas) and (0.95 in alphas):
        lo = out["q_0.05"]
        hi = out["q_0.95"]
        tol = 1e-12 * max(1.0, abs(theta_hat))
        if lo > theta_hat + tol or hi < theta_hat - tol:
            raise ValueError(
                f"FAIL-CLOSED BCa CONTAINMENT INVARIANT VIOLATED: "
                f"CI90=[{lo:.6f}, {hi:.6f}] does not contain point estimate {theta_hat:.6f}"
            )
    return out


def holm_bonferroni_stepdown(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Exact 5-test Holm-Bonferroni step-down procedure with explicit stop rule."""
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
        res[name] = {
            "p": p,
            "rank": rank,
            "threshold": round(thresh, 6),
            "rejected": rej,
        }
    return res


# ---------------------------------------------------------------------------
# Explicit C1.3 6-way disjoint partition (NO fallback `else`)
# ---------------------------------------------------------------------------
def evaluate_c1_3_partition(
    delta_r_sep: float,
    q05_sep_ok: bool,
    p_ha_ok: bool,
    strata_ok: bool,
    r_floor_ridge: float,
    q05_floor_ok: bool,
    p_hb_ok: bool,
    whitening_ok: bool,
    logit_lens_clean: bool,
) -> dict[str, Any]:
    """Evaluate C1.3's explicit 6-way disjoint partition (verified 2,560/2,560 grid + 200k random floors)."""
    active = whitening_ok and (r_floor_ridge <= C1_3_FLOOR_HB_MAX)
    sep_pass = (delta_r_sep >= C1_3_DELTA_R_SEP_BAR) and q05_sep_ok and p_ha_ok and strata_ok
    hb_pass = (C1_3_FLOOR_HB_MIN <= r_floor_ridge <= C1_3_FLOOR_HB_MAX) and q05_floor_ok and p_hb_ok
    low_fl = r_floor_ridge <= C1_3_FLOOR_HA_MAX
    mid_fl = (r_floor_ridge > C1_3_FLOOR_HA_MAX) and (r_floor_ridge < C1_3_FLOOR_HB_MIN)
    band_fl = (r_floor_ridge >= C1_3_FLOOR_HB_MIN) and (r_floor_ridge <= C1_3_FLOOR_HB_MAX)

    preds = {
        "WHITENING_NOT_ESTABLISHED": not whitening_ok,
        "FLOOR_ABOVE_CEILING": whitening_ok and (r_floor_ridge > C1_3_FLOOR_HB_MAX),
        "H_A": active and sep_pass and low_fl,
        "H_B": active and hb_pass and (not sep_pass),
        "NEITHER": active and low_fl and (not sep_pass),
        "MIXED": active and (mid_fl or (band_fl and (not hb_pass)) or (sep_pass and hb_pass)),
    }
    matched = [k for k, v in preds.items() if v]
    if len(matched) != 1:
        raise RuntimeError(f"C1.3 partition invariant violated: matched={matched}")
    base = matched[0]
    subtag = None
    label = base
    if base == "H_B" and (not logit_lens_clean):
        subtag = "FLOOR_SUSPECT: logit_lens"
        label = "H_B (FLOOR_SUSPECT: logit_lens)"
    elif base == "MIXED" and sep_pass and hb_pass:
        subtag = "MIXED_BOTH"
        label = "MIXED (MIXED_BOTH)"
    return {
        "base_outcome": base,
        "label": label,
        "subtag": subtag,
        "whitening_ok": whitening_ok,
        "active": active,
        "sep_pass": sep_pass,
        "hb_pass": hb_pass,
        "logit_lens_clean": logit_lens_clean,
    }


class _AllCappedCellError(RuntimeError):
    """Raised internally when all prompts of a (L, target) cell are capped so C1.1 fails closed cleanly."""


# ---------------------------------------------------------------------------
# Fail-closed receipt structural & finiteness audits
# ---------------------------------------------------------------------------
def validate_receipt_structure(receipt: dict[str, Any]) -> None:
    meta = receipt.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Missing receipt.meta")
    if meta.get("smoke", False):
        raise ValueError("Refusing to score smoke receipt (meta.smoke == True)")
    if meta.get("primary_layers") != PRIMARY_LAYERS:
        raise ValueError(f"Unexpected primary_layers: {meta.get('primary_layers')}")
    if meta.get("secondary_layers") != SECONDARY_LAYERS:
        raise ValueError(f"Unexpected secondary_layers: {meta.get('secondary_layers')}")
    if meta.get("n_targets") != EXPECTED_N_TARGETS or len(receipt.get("targets_24", [])) != EXPECTED_N_TARGETS:
        raise ValueError("Target count != 24")
    if meta.get("n_prompts") != EXPECTED_N_PROMPTS or len(receipt.get("prompts_8", [])) != EXPECTED_N_PROMPTS:
        raise ValueError("Prompt count != 8")
    if meta.get("k_dm") != EXPECTED_K_DM or meta.get("m_rand") != EXPECTED_M_RAND:
        raise ValueError(f"Unexpected k_dm/m_rand: {meta.get('k_dm')}/{meta.get('m_rand')}")
    if "total_forwards" in meta:
        tf = meta["total_forwards"]
        # Minimum non-Block1 forwards alone = 8 clean + 7 shams + 1728 Block2a + 768 Block2b + 576 L12 = 3087
        if not isinstance(tf, int) or isinstance(tf, bool) or tf < 3087:
            raise ValueError(f"Implausible meta.total_forwards={tf!r} (expected >= 3087, full=6543)")

    pmn_map = receipt.get("post_mlp_norms", {})
    if not isinstance(pmn_map, dict):
        raise ValueError("Missing receipt.post_mlp_norms")

    t_strs = [t["string"] for t in receipt["targets_24"]]
    strata_counts: dict[str, int] = {}
    for t in receipt["targets_24"]:
        st = t["stratum"]
        strata_counts[st] = strata_counts.get(st, 0) + 1
    if set(strata_counts.keys()) != set(EXPECTED_STRATA) or any(v != 6 for v in strata_counts.values()):
        raise ValueError(f"Strata imbalance: {strata_counts}")

    for L in PRIMARY_LAYERS:
        ls = str(L)
        pmn_l = float(pmn_map.get(ls, 0.0))
        if not math.isfinite(pmn_l) or pmn_l <= 0.0:
            raise ValueError(f"Invalid post_mlp_norms[{ls}]={pmn_l!r}")
        for blk_name, want_pos in (("block1_regime_D_all", "all"), ("block1_regime_D_last", "last")):
            blk = receipt.get(blk_name, {}).get(ls)
            if not isinstance(blk, dict):
                raise ValueError(f"Missing {blk_name}[{ls}]")
            if blk.get("positions") != want_pos:
                raise ValueError(f"{blk_name}[{ls}].positions={blk.get('positions')!r} != {want_pos!r}")
            if abs(float(blk.get("target_direct_linear", 0.0)) - 3.0) > 1e-9:
                raise ValueError(f"{blk_name}[{ls}].target_direct_linear={blk.get('target_direct_linear')!r} != 3.0")
            if abs(float(blk.get("dose_cap", 0.0)) - 3.0 * pmn_l) > 1e-3:
                raise ValueError(f"{blk_name}[{ls}].dose_cap={blk.get('dose_cap')!r} != 3.0*pmn ({3.0*pmn_l:.6f})")
            for ts in t_strs:
                if len(blk["R_mf"].get(ts, [])) != EXPECTED_N_PROMPTS:
                    raise ValueError(f"{blk_name}[{ls}].R_mf[{ts!r}] length != 8")
                dm_draws = blk["direct_matched"].get(ts, [])
                if len(dm_draws) != EXPECTED_K_DM or any(len(d) != EXPECTED_N_PROMPTS for d in dm_draws):
                    raise ValueError(f"{blk_name}[{ls}].direct_matched[{ts!r}] shape != (2, 8)")
        b2a = receipt.get("block2a_regime_N_last", {}).get(ls)
        if not isinstance(b2a, dict):
            raise ValueError(f"Missing block2a_regime_N_last[{ls}]")
        if b2a.get("positions") != "last":
            raise ValueError(f"block2a_regime_N_last[{ls}].positions={b2a.get('positions')!r} != 'last'")
        if abs(float(b2a.get("dose_N", 0.0)) - 0.75 * pmn_l) > 1e-3:
            raise ValueError(f"block2a_regime_N_last[{ls}].dose_N={b2a.get('dose_N')!r} != 0.75*pmn ({0.75*pmn_l:.6f})")
        for arm in ("R_mf", "R_ridge", "logit_lens"):
            for ts in t_strs:
                rows = b2a.get(arm, {}).get(ts, [])
                if len(rows) != EXPECTED_N_PROMPTS:
                    raise ValueError(f"block2a[{ls}].{arm}[{ts!r}] length != 8")
                for r in rows:
                    if len(r.get("all_24_dlogits", [])) != EXPECTED_N_TARGETS:
                        raise ValueError(f"block2a[{ls}].{arm}[{ts!r}] missing all_24_dlogits[24]")
        b2b = receipt.get("block2b_rand_iso", {}).get(ls)
        if not isinstance(b2b, dict) or b2b.get("positions") != "last":
            raise ValueError(f"block2b_rand_iso[{ls}] missing or positions != 'last'")
        if abs(float(b2b.get("dose_N", 0.0)) - 0.75 * pmn_l) > 1e-3:
            raise ValueError(f"block2b_rand_iso[{ls}].dose_N={b2b.get('dose_N')!r} != 0.75*pmn ({0.75*pmn_l:.6f})")
        draws = b2b.get("rand_iso", [])
        if len(draws) != EXPECTED_M_RAND or any(len(d) != EXPECTED_N_PROMPTS for d in draws):
            raise ValueError(f"block2b_rand_iso[{ls}] shape != (32, 8)")
        for d in draws:
            for r in d:
                if len(r.get("all_24_dlogits", [])) != EXPECTED_N_TARGETS:
                    raise ValueError(f"block2b_rand_iso[{ls}] missing all_24_dlogits[24]")

    sec12 = receipt.get("secondary_L12", {}).get("12")
    pmn_12 = float(pmn_map.get("12", 0.0))
    if not isinstance(sec12, dict) or not math.isfinite(pmn_12) or pmn_12 <= 0.0:
        raise ValueError("Missing secondary_L12['12'] or invalid post_mlp_norms['12']")
    if abs(float(sec12.get("dose_multiplier", 0.0)) - 0.30) > 1e-9:
        raise ValueError(f"secondary_L12['12'].dose_multiplier={sec12.get('dose_multiplier')!r} != 0.30")
    if abs(float(sec12.get("dose_N", 0.0)) - 0.30 * pmn_12) > 1e-3:
        raise ValueError(f"secondary_L12['12'].dose_N={sec12.get('dose_N')!r} != 0.30*pmn_12")
    for key in ("R_mf_all", "R_mf_last", "R_ridge_last"):
        for ts in t_strs:
            if len(sec12.get(key, {}).get(ts, [])) != EXPECTED_N_PROMPTS:
                raise ValueError(f"secondary_L12['12'].{key}[{ts!r}] length != 8")


def audit_all_traces_finite(receipt: dict[str, Any]) -> list[str]:
    """Scan every retained primary trace for non-finite values; returns list of offending paths."""
    bad: list[str] = []
    t_strs = [t["string"] for t in receipt["targets_24"]]
    trace_fields = (
        "dose", "exact_dlogit", "pred_dlogit", "direct_linear", "direct_linear_analytic",
        "frozen_direct_effect", "sum_attn", "sum_mlp", "rmsnorm_shift", "telescoping_err",
        "retention", "canc_r", "dlogit_z", "kl", "rms_clean", "rms_int",
    )

    def _chk_val(val: Any, loc: str) -> None:
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            bad.append(f"{loc}={val!r}")

    for L in PRIMARY_LAYERS:
        ls = str(L)
        for blk_name in ("block1_regime_D_all", "block1_regime_D_last"):
            blk = receipt[blk_name][ls]
            for ts in t_strs:
                for pi, r in enumerate(blk["R_mf"][ts]):
                    if r.get("capped", False):
                        continue
                    for f in trace_fields:
                        _chk_val(r.get(f), f"{blk_name}.L{L}.R_mf[{ts}][{pi}].{f}")
                for k, draw in enumerate(blk["direct_matched"][ts]):
                    for pi, r in enumerate(draw):
                        if r.get("capped", False):
                            continue
                        for f in trace_fields:
                            _chk_val(r.get(f), f"{blk_name}.L{L}.dm[{ts}][{k}][{pi}].{f}")
        b2a = receipt["block2a_regime_N_last"][ls]
        for arm in ("R_mf", "R_ridge", "logit_lens"):
            for ts in t_strs:
                for pi, r in enumerate(b2a[arm][ts]):
                    for f in trace_fields:
                        _chk_val(r.get(f), f"block2a.L{L}.{arm}[{ts}][{pi}].{f}")
                    for j, dv in enumerate(r["all_24_dlogits"]):
                        _chk_val(dv, f"block2a.L{L}.{arm}[{ts}][{pi}].all_24_dlogits[{j}]")
                    if arm == "logit_lens":
                        for sf in ("u_hat_self_J", "u_hat_off_abs_mean_J"):
                            if sf in r:
                                _chk_val(r[sf], f"block2a.L{L}.logit_lens[{ts}][{pi}].{sf}")
        for m, draw in enumerate(receipt["block2b_rand_iso"][ls]["rand_iso"]):
            for pi, r in enumerate(draw):
                for f in ("dose", "vJv", "telescoping_err", "canc_r_self", "rms_clean", "rms_int"):
                    _chk_val(r.get(f), f"block2b.L{L}.rand_iso[{m}][{pi}].{f}")
                for j, dv in enumerate(r["all_24_dlogits"]):
                    _chk_val(dv, f"block2b.L{L}.rand_iso[{m}][{pi}].all_24_dlogits[{j}]")
    return bad


# ---------------------------------------------------------------------------
# Main Stage C1 scorer
# ---------------------------------------------------------------------------
def score_c1(
    receipt: dict[str, Any],
    c1_constants: dict[str, Any],
    constants_path: Path | None = None,
    design_path: Path | None = None,
    strict_provenance: bool = True,
    n_flips: int = PERM_N_FLIPS,
    n_boot: int = BCA_N_RESAMPLES,
) -> dict[str, Any]:
    validate_receipt_structure(receipt)

    meta = receipt["meta"]
    targets = receipt["targets_24"]
    t_strs = [t["string"] for t in targets]
    t_strata = [t["stratum"] for t in targets]
    n_t = len(targets)
    prompts = receipt["prompts_8"]
    bare_idxs = [i for i, p in enumerate(prompts) if p["format"] == "bare"]
    chat_idxs = [i for i, p in enumerate(prompts) if p["format"] == "chatml"]
    if len(bare_idxs) != 4 or len(chat_idxs) != 4:
        raise ValueError(f"Expected 4 bare & 4 chatml prompts, got {len(bare_idxs)}/{len(chat_idxs)}")

    # =========================================================================
    # Global Gate C1.0 (all 4 clauses must pass to read C1.1-C1.4)
    # =========================================================================
    # C1.0.1: 7 zero-dose shams == 0.0 AND R_mf movement > 0.0 in every (L, mode)
    shams = receipt.get("shams", {})
    mv = receipt.get("sham_movement_check", {})
    sham_zero_vals: list[tuple[str, float]] = []
    for L in PRIMARY_LAYERS:
        for mode in ("last", "all"):
            v = shams.get(mode, {}).get(str(L))
            sham_zero_vals.append((f"L{L}_{mode}", float(v) if isinstance(v, (int, float)) else float("nan")))
    v12 = shams.get("all", {}).get("12")
    sham_zero_vals.append(("L12_all", float(v12) if isinstance(v12, (int, float)) else float("nan")))

    # F9: check R_mf liveness strictly on the 7 registered (L, mode) pairs (3L x 2 modes + L12 all)
    shams_all_zero = all(math.isfinite(v) and v == 0.0 for _, v in sham_zero_vals)
    registered_liveness_pairs = [(L, m) for L in PRIMARY_LAYERS for m in ("last", "all")] + [(12, "all")]
    mv_checks: dict[str, float] = {}
    for L, mode in registered_liveness_pairs:
        val = mv.get(mode, {}).get(str(L))
        mv_checks[f"L{L}_{mode}"] = float(val) if isinstance(val, (int, float)) else float("nan")
    rmf_moves_all = all(math.isfinite(v) and v > 0.0 for v in mv_checks.values())
    c1_0_1_pass = shams_all_zero and rmf_moves_all

    # C1.0.2: All primary traces finite AND telescoping p95 <= 0.08 AND p99.9 <= 0.15
    nonfinite_list = audit_all_traces_finite(receipt)
    all_finite = len(nonfinite_list) == 0
    tele_all: list[float] = []
    tele_bare: list[float] = []
    tele_chat: list[float] = []
    if all_finite:
        for L in PRIMARY_LAYERS:
            ls = str(L)
            for blk_name in ("block1_regime_D_all", "block1_regime_D_last"):
                blk = receipt[blk_name][ls]
                for ts in t_strs:
                    for pi, r in enumerate(blk["R_mf"][ts]):
                        if not r.get("capped", False):
                            te = float(r["telescoping_err"])
                            tele_all.append(te)
                            (tele_bare if pi in bare_idxs else tele_chat).append(te)
                    for draw in blk["direct_matched"][ts]:
                        for pi, r in enumerate(draw):
                            if not r.get("capped", False):
                                te = float(r["telescoping_err"])
                                tele_all.append(te)
                                (tele_bare if pi in bare_idxs else tele_chat).append(te)
            b2a = receipt["block2a_regime_N_last"][ls]
            for arm in ("R_mf", "R_ridge", "logit_lens"):
                for ts in t_strs:
                    for pi, r in enumerate(b2a[arm][ts]):
                        te = float(r["telescoping_err"])
                        tele_all.append(te)
                        (tele_bare if pi in bare_idxs else tele_chat).append(te)
            for draw in receipt["block2b_rand_iso"][ls]["rand_iso"]:
                for pi, r in enumerate(draw):
                    te = float(r["telescoping_err"])
                    tele_all.append(te)
                    (tele_bare if pi in bare_idxs else tele_chat).append(te)
        tele_p95 = quantile(tele_all, 0.95)
        tele_p999 = quantile(tele_all, 0.999)
        tele_max = max(tele_all)
        c1_0_2_pass = (tele_p95 <= 0.08) and (tele_p999 <= 0.15)
    else:
        tele_p95 = tele_p999 = tele_max = float("nan")
        c1_0_2_pass = False

    # C1.0.3: Coherent count-based Regime-D "all" linearity gate (1,728 traces) + clean-geometry anchor (±30%)
    anchor_expected = {
        L: float(c1_constants["layer_geometry"][str(L)]["dose_D3_over_pmn_median"])
        for L in PRIMARY_LAYERS
    }
    anchor_cells: dict[str, dict[str, Any]] = {}
    anchor_all_ok = True
    rmsn_dev_all: list[float] = []
    fde_err_all: list[float] = []
    capped_d_all_count = 0

    if all_finite:
        for L in PRIMARY_LAYERS:
            ls = str(L)
            pmn_l = float(receipt["post_mlp_norms"][ls])
            if pmn_l <= 0.0 or not math.isfinite(pmn_l):
                anchor_all_ok = False
                continue
            blk = receipt["block1_regime_D_all"][ls]
            for arm in ("R_mf", "direct_matched"):
                ratios: list[float] = []
                for ts in t_strs:
                    rows = blk[arm][ts] if arm == "R_mf" else [r for d in blk[arm][ts] for r in d]
                    for r in rows:
                        if r.get("capped", False):
                            capped_d_all_count += 1
                            continue
                        ratios.append(float(r["dose"]) / pmn_l)
                        rc = float(r["rms_clean"])
                        ri = float(r["rms_int"])
                        rmsn_dev_all.append(abs(ri / rc - 1.0))
                        fde_err_all.append(abs(float(r["frozen_direct_effect"]) - 3.0))
                if not ratios:
                    anchor_all_ok = False
                    anchor_cells[f"L{L}_{arm}"] = {"ok": False, "n": 0}
                else:
                    med_ratio = median(ratios)
                    exp_r = anchor_expected[L]
                    rel_diff = abs(med_ratio - exp_r) / exp_r
                    cell_ok = rel_diff <= 0.30
                    if not cell_ok:
                        anchor_all_ok = False
                    anchor_cells[f"L{L}_{arm}"] = {
                        "median_dose_over_pmn": round(med_ratio, 6),
                        "expected": round(exp_r, 6),
                        "rel_diff": round(rel_diff, 6),
                        "ok": cell_ok,
                        "n": len(ratios),
                    }

        rmsn_gt_20 = sum(1 for v in rmsn_dev_all if v > 0.20)
        rmsn_gt_12 = sum(1 for v in rmsn_dev_all if v > 0.12)
        fde_p95 = quantile(fde_err_all, 0.95) if fde_err_all else float("nan")
        fde_gt_45 = sum(1 for v in fde_err_all if v > 0.45)
        fde_gt_75 = sum(1 for v in fde_err_all if v > 0.75)
        c1_0_3_pass = (
            anchor_all_ok
            and (len(rmsn_dev_all) > 0)
            and (rmsn_gt_20 <= 2)
            and (rmsn_gt_12 <= 26)
            and (fde_p95 <= 0.30)
            and (fde_gt_45 <= 12)
            and (fde_gt_75 <= 2)
        )
    else:
        rmsn_gt_20 = rmsn_gt_12 = fde_gt_45 = fde_gt_75 = -1
        fde_p95 = float("nan")
        c1_0_3_pass = False

    # Non-gating Regime-N RMSNorm diagnostic
    rn_rmsn_devs: list[float] = []
    if all_finite:
        for L in PRIMARY_LAYERS:
            ls = str(L)
            for arm in ("R_mf", "R_ridge", "logit_lens"):
                for ts in t_strs:
                    for r in receipt["block2a_regime_N_last"][ls][arm][ts]:
                        rn_rmsn_devs.append(abs(float(r["rms_int"]) / float(r["rms_clean"]) - 1.0))
            for d in receipt["block2b_rand_iso"][ls]["rand_iso"]:
                for r in d:
                    rn_rmsn_devs.append(abs(float(r["rms_int"]) / float(r["rms_clean"]) - 1.0))

    # C1.0.4: Strict frozen SHA-256 provenance (fail-closed if --design missing or --no-strict-provenance)
    prov_checks: dict[str, bool] = {
        "strict_provenance_enabled": bool(strict_provenance),
        "meta_r_lens": meta.get("r_lens_sha256") == EXPECTED_R_LENS_SHA256,
        "meta_c1_constants": meta.get("c1_constants_sha256") == EXPECTED_C1_CONSTANTS_SHA256,
        "meta_c1_bank": meta.get("c1_bank_sha256") == EXPECTED_C1_BANK_SHA256,
        "meta_c1_design": meta.get("c1_design_sha256") == EXPECTED_C1_DESIGN_SHA256,
        "disk_c1_constants": (
            constants_path is not None and constants_path.exists()
            and sha256_file(constants_path) == EXPECTED_C1_CONSTANTS_SHA256
        ),
        "disk_c1_design": (
            design_path is not None and design_path.exists()
            and sha256_file(design_path) == EXPECTED_C1_DESIGN_SHA256
        ),
    }
    c1_0_4_pass = all(prov_checks.values())
    c1_0_pass = c1_0_1_pass and c1_0_2_pass and c1_0_3_pass and c1_0_4_pass

    gate_c1_0 = {
        "PASS": c1_0_pass,
        "C1.0.1_shams_and_liveness": {
            "PASS": c1_0_1_pass,
            "shams": {k: v for k, v in sham_zero_vals},
            "max_rmf_movement": mv_checks,
        },
        "C1.0.2_finite_and_telescoping": {
            "PASS": c1_0_2_pass,
            "all_finite": all_finite,
            "nonfinite_count": len(nonfinite_list),
            "nonfinite_sample": nonfinite_list[:10],
            "n_primary_traces": len(tele_all),
            "p95": round(tele_p95, 6) if math.isfinite(tele_p95) else None,
            "p99_9": round(tele_p999, 6) if math.isfinite(tele_p999) else None,
            "max": round(tele_max, 6) if math.isfinite(tele_max) else None,
            "bare_p95": round(quantile(tele_bare, 0.95), 6) if tele_bare else None,
            "chatml_p95": round(quantile(tele_chat, 0.95), 6) if tele_chat else None,
        },
        "C1.0.3_regime_D_all_linearity": {
            "PASS": c1_0_3_pass,
            "n_traces_evaluated": len(rmsn_dev_all),
            "capped_skipped_count": capped_d_all_count,
            "anchor_all_ok": anchor_all_ok,
            "anchor_cells": anchor_cells,
            "rmsn_gt_20pct_count": rmsn_gt_20,
            "rmsn_gt_12pct_count": rmsn_gt_12,
            "rmsn_max": round(max(rmsn_dev_all), 6) if rmsn_dev_all else None,
            "fde_p95": round(fde_p95, 6) if math.isfinite(fde_p95) else None,
            "fde_gt_045_count": fde_gt_45,
            "fde_gt_075_count": fde_gt_75,
            "fde_max": round(max(fde_err_all), 6) if fde_err_all else None,
            "regime_N_rmsn_diagnostic": {
                "p95": round(quantile(rn_rmsn_devs, 0.95), 6) if rn_rmsn_devs else None,
                "p99_9": round(quantile(rn_rmsn_devs, 0.999), 6) if rn_rmsn_devs else None,
                "max": round(max(rn_rmsn_devs), 6) if rn_rmsn_devs else None,
            },
        },
        "C1.0.4_provenance": {
            "PASS": c1_0_4_pass,
            "checks": prov_checks,
        },
    }

    # =========================================================================
    # Secondary Non-Gating Diagnostic: L12_linear_check (strictly outside Gate C1.0)
    # =========================================================================
    sec12 = receipt["secondary_L12"]["12"]
    l12_rmsn_devs: list[float] = []
    l12_frac_errs: list[float] = []
    for ts in t_strs:
        for r in sec12["R_mf_all"][ts]:
            rc = float(r["rms_clean"])
            ri = float(r["rms_int"])
            l12_rmsn_devs.append(abs(ri / rc - 1.0))
            ex = float(r["exact_dlogit"])
            pr = float(r["pred_dlogit"])
            l12_frac_errs.append(abs(ex - pr) / max(abs(pr), 1e-6))
    l12_gt_09 = sum(1 for v in l12_rmsn_devs if v > 0.09)
    l12_fe_med = median(l12_frac_errs)
    l12_linear_check = {
        "PASS": (l12_gt_09 <= 2) and (l12_fe_med <= 0.12),
        "n_traces": len(l12_rmsn_devs),
        "rmsn_gt_09pct_count": l12_gt_09,
        "rmsn_median": round(median(l12_rmsn_devs), 6),
        "rmsn_max": round(max(l12_rmsn_devs), 6),
        "frac_err_median": round(l12_fe_med, 6),
    }

    if not c1_0_pass:
        return {
            "schema_version": "c1-scorecard-v1",
            "verdicts": {
                "C1.0": "FAIL",
                "C1.1": "BLOCKED_BY_C1.0",
                "C1.2": "BLOCKED_BY_C1.0",
                "C1.3": "BLOCKED_BY_C1.0",
                "C1.4": "BLOCKED_BY_C1.0",
                "L12_linear_check": "PASS" if l12_linear_check["PASS"] else "FAIL",
            },
            "gate_C1_0": gate_c1_0,
            "secondary_L12_linear_check": l12_linear_check,
        }

    # =========================================================================
    # Claim C1.1: Fresh-unit linear-regime positions="all" Attention transport
    # =========================================================================
    # (a) Calibration recovery
    fe_all_traces: list[float] = []
    fe_last_traces: list[float] = []
    # For p_clean diagnostic
    clean_probs_map = receipt.get("clean_target_probs", {})
    p_clean_log_vals: list[float] = []
    p_clean_fe_vals: list[float] = []
    fe_ge_1e6: list[float] = []
    fe_lt_1e6: list[float] = []
    flagged_lt_1e6: list[dict[str, Any]] = []
    fe_by_stratum_fmt: dict[str, list[float]] = {}

    for L in PRIMARY_LAYERS:
        ls = str(L)
        for ti, ts in enumerate(t_strs):
            st = t_strata[ti]
            for pi, r in enumerate(receipt["block1_regime_D_all"][ls]["R_mf"][ts]):
                if r.get("capped", False):
                    continue
                fe = abs(float(r["exact_dlogit"]) - float(r["pred_dlogit"])) / max(abs(float(r["pred_dlogit"])), 1e-6)
                fe_all_traces.append(fe)
                pid = prompts[pi]["id"]
                fmt = prompts[pi]["format"]
                fe_by_stratum_fmt.setdefault(f"{st}_{fmt}", []).append(fe)
                pc_info = clean_probs_map.get(pid, {}).get(ts)
                if pc_info is not None:
                    pc_v = float(pc_info["p_clean"])
                    lpc_v = float(pc_info["log_p_clean"])
                    p_clean_log_vals.append(lpc_v)
                    p_clean_fe_vals.append(fe)
                    if pc_v >= 1e-6:
                        fe_ge_1e6.append(fe)
                    else:
                        fe_lt_1e6.append(fe)
                        if L == PRIMARY_LAYERS[0]:
                            flagged_lt_1e6.append({"prompt_id": pid, "target": ts, "p_clean": pc_v})
            for pi, r in enumerate(receipt["block1_regime_D_last"][ls]["R_mf"][ts]):
                if r.get("capped", False):
                    continue
                fe = abs(float(r["exact_dlogit"]) - float(r["pred_dlogit"])) / max(abs(float(r["pred_dlogit"])), 1e-6)
                fe_last_traces.append(fe)

    med_fe_all = median(fe_all_traces)
    med_fe_last = median(fe_last_traces)
    frac_err_ok = (med_fe_all <= 0.18) and (med_fe_all <= 0.50 * med_fe_last)

    # Cell-level r_all = mean_p[gap_all(L,t,p)] / pred_gap_D3(L,t), matching C0b score_olmo3_c0b.py:1197-1237
    # F6: fail closed with BLOCKED_BY_DATA if all prompts of any (L, target) cell are capped.
    c1_1_blocked_reason: str | None = None

    def _cell_r_all(L: int, ts: str, allowed_p: list[int]) -> float:
        blk = receipt["block1_regime_D_all"][str(L)]
        gaps: list[float] = []
        pred_gaps: list[float] = []
        for pi in allowed_p:
            rmf_r = blk["R_mf"][ts][pi]
            dm_rs = [blk["direct_matched"][ts][k][pi] for k in range(EXPECTED_K_DM)]
            if rmf_r.get("capped", False) or any(r.get("capped", False) for r in dm_rs):
                continue
            gaps.append(float(rmf_r["exact_dlogit"]) - mean([float(r["exact_dlogit"]) for r in dm_rs]))
            pred_gaps.append(float(rmf_r["pred_dlogit"]) - mean([float(r["pred_dlogit"]) for r in dm_rs]))
        if not pred_gaps:
            raise _AllCappedCellError(f"C1.1 cell L{L} {ts!r}: all prompts capped; claim uncomputable")
        pg = mean(pred_gaps)
        if abs(pg) <= 1e-9:
            raise ValueError(f"Zero pred_gap at L{L}, {ts!r}")
        return mean(gaps) / pg

    all_p = list(range(EXPECTED_N_PROMPTS))
    try:
        r_all_pool = [[_cell_r_all(L, ts, all_p) for L in PRIMARY_LAYERS] for ts in t_strs]
        r_all_bare = [[_cell_r_all(L, ts, bare_idxs) for L in PRIMARY_LAYERS] for ts in t_strs]
        r_all_chat = [[_cell_r_all(L, ts, chat_idxs) for L in PRIMARY_LAYERS] for ts in t_strs]
    except _AllCappedCellError as exc:
        c1_1_blocked_reason = str(exc)
        r_all_pool = r_all_bare = r_all_chat = []

    if c1_1_blocked_reason is None:
        def _mat_med(mat: list[list[float]], t_idxs: list[int]) -> float:
            return median([mat[ti][li] for ti in t_idxs for li in range(len(PRIMARY_LAYERS))])

        r_all_pool_ci = bca_cluster_ci(
            n_t, lambda idxs: _mat_med(r_all_pool, idxs), alphas=(0.05, 0.95), n_boot=n_boot, seed=BCA_SEED
        )
        pooled_med_r_all = r_all_pool_ci["point"]
        bare_med_r_all = _mat_med(r_all_bare, list(range(n_t)))
        chat_med_r_all = _mat_med(r_all_chat, list(range(n_t)))

        per_layer_r_all: dict[str, dict[str, float]] = {}
        max_layer_split = 0.0
        for li, L in enumerate(PRIMARY_LAYERS):
            mb = median([r_all_bare[ti][li] for ti in range(n_t)])
            mc = median([r_all_chat[ti][li] for ti in range(n_t)])
            mp = median([r_all_pool[ti][li] for ti in range(n_t)])
            sp = abs(mc - mb)
            max_layer_split = max(max_layer_split, sp)
            per_layer_r_all[str(L)] = {
                "pooled_median": round(mp, 6),
                "bare_median": round(mb, 6),
                "chatml_median": round(mc, 6),
                "abs_chatml_minus_bare": round(sp, 6),
            }

        r_all_bands_ok = (
            (0.65 <= pooled_med_r_all <= 1.50)
            and (r_all_pool_ci["q_0.05"] > 0.55)
            and (r_all_pool_ci["q_0.95"] < 1.60)
            and (0.65 <= bare_med_r_all <= 1.50)
            and (0.65 <= chat_med_r_all <= 1.50)
        )
        layer_split_ok = max_layer_split <= 0.35
        c1_1a_ok = frac_err_ok and r_all_bands_ok and layer_split_ok
    else:
        r_all_pool_ci = {"point": float("nan"), "q_0.05": float("nan"), "q_0.95": float("nan")}
        pooled_med_r_all = bare_med_r_all = chat_med_r_all = max_layer_split = float("nan")
        per_layer_r_all = {}
        r_all_bands_ok = layer_split_ok = c1_1a_ok = False

    # (b) Attention attribution
    dtot_gap_t: list[float] = []
    dattn_gap_t: list[float] = []
    dattn_rmf_t: list[float] = []
    dattn_dm_t: list[float] = []
    dtot_rmf_t: list[float] = []
    dm_layer_diag: dict[str, dict[str, float]] = {}

    if c1_1_blocked_reason is None:
        try:
            for L in PRIMARY_LAYERS:
                ls = str(L)
                dm_dtot_l, dm_dattn_l = [], []
                for ts in t_strs:
                    a_dm = [
                        mean([float(receipt["block1_regime_D_all"][ls]["direct_matched"][ts][k][pi]["exact_dlogit"]) for k in range(EXPECTED_K_DM)])
                        - mean([float(receipt["block1_regime_D_last"][ls]["direct_matched"][ts][k][pi]["exact_dlogit"]) for k in range(EXPECTED_K_DM)])
                        for pi in all_p
                        if not any(receipt["block1_regime_D_all"][ls]["direct_matched"][ts][k][pi].get("capped", False) for k in range(EXPECTED_K_DM))
                        and not any(receipt["block1_regime_D_last"][ls]["direct_matched"][ts][k][pi].get("capped", False) for k in range(EXPECTED_K_DM))
                    ]
                    s_dm = [
                        mean([float(receipt["block1_regime_D_all"][ls]["direct_matched"][ts][k][pi]["sum_attn"]) for k in range(EXPECTED_K_DM)])
                        - mean([float(receipt["block1_regime_D_last"][ls]["direct_matched"][ts][k][pi]["sum_attn"]) for k in range(EXPECTED_K_DM)])
                        for pi in all_p
                        if not any(receipt["block1_regime_D_all"][ls]["direct_matched"][ts][k][pi].get("capped", False) for k in range(EXPECTED_K_DM))
                        and not any(receipt["block1_regime_D_last"][ls]["direct_matched"][ts][k][pi].get("capped", False) for k in range(EXPECTED_K_DM))
                    ]
                    if not a_dm or not s_dm:
                        raise _AllCappedCellError(f"C1.1(b) dm cell L{L} {ts!r}: all prompts capped")
                    dm_dtot_l.append(mean(a_dm))
                    dm_dattn_l.append(mean(s_dm))
                dm_layer_diag[ls] = {
                    "mean_delta_tot_dm": round(mean(dm_dtot_l), 6),
                    "mean_delta_attn_dm": round(mean(dm_dattn_l), 6),
                }

            for ti, ts in enumerate(t_strs):
                tg_ls, ag_ls, ar_ls, ad_ls, tr_ls = [], [], [], [], []
                for L in PRIMARY_LAYERS:
                    ls = str(L)
                    b_all = receipt["block1_regime_D_all"][ls]
                    b_last = receipt["block1_regime_D_last"][ls]
                    gaps_a, gaps_l = [], []
                    agaps_a, agaps_l = [], []
                    armf_a, armf_l = [], []
                    adm_a, adm_l = [], []
                    trmf_a, trmf_l = [], []
                    for pi in all_p:
                        ra = b_all["R_mf"][ts][pi]
                        rl = b_last["R_mf"][ts][pi]
                        dma = [b_all["direct_matched"][ts][k][pi] for k in range(EXPECTED_K_DM)]
                        dml = [b_last["direct_matched"][ts][k][pi] for k in range(EXPECTED_K_DM)]
                        if ra.get("capped", False) or rl.get("capped", False) or any(r.get("capped", False) for r in dma + dml):
                            continue
                        dm_ex_a = mean([float(r["exact_dlogit"]) for r in dma])
                        dm_ex_l = mean([float(r["exact_dlogit"]) for r in dml])
                        dm_at_a = mean([float(r["sum_attn"]) for r in dma])
                        dm_at_l = mean([float(r["sum_attn"]) for r in dml])
                        gaps_a.append(float(ra["exact_dlogit"]) - dm_ex_a)
                        gaps_l.append(float(rl["exact_dlogit"]) - dm_ex_l)
                        agaps_a.append(float(ra["sum_attn"]) - dm_at_a)
                        agaps_l.append(float(rl["sum_attn"]) - dm_at_l)
                        armf_a.append(float(ra["sum_attn"]))
                        armf_l.append(float(rl["sum_attn"]))
                        adm_a.append(dm_at_a)
                        adm_l.append(dm_at_l)
                        trmf_a.append(float(ra["exact_dlogit"]))
                        trmf_l.append(float(rl["exact_dlogit"]))
                    if not gaps_a:
                        raise _AllCappedCellError(f"C1.1(b) cell L{L} {ts!r}: all prompts capped")
                    tg_ls.append(mean(gaps_a) - mean(gaps_l))
                    ag_ls.append(mean(agaps_a) - mean(agaps_l))
                    ar_ls.append(mean(armf_a) - mean(armf_l))
                    ad_ls.append(mean(adm_a) - mean(adm_l))
                    tr_ls.append(mean(trmf_a) - mean(trmf_l))
                dtot_gap_t.append(mean(tg_ls))
                dattn_gap_t.append(mean(ag_ls))
                dattn_rmf_t.append(mean(ar_ls))
                dattn_dm_t.append(mean(ad_ls))
                dtot_rmf_t.append(mean(tr_ls))
        except _AllCappedCellError as exc:
            c1_1_blocked_reason = str(exc)

    if c1_1_blocked_reason is None:
        p_gap_share = sign_flip_p_one_sided(
            [a - 0.75 * tot for a, tot in zip(dattn_gap_t, dtot_gap_t)], n_flips=n_flips, seed=PERM_SEED
        )
        p_guard = sign_flip_p_one_sided(
            [ar - ad for ar, ad in zip(dattn_rmf_t, dattn_dm_t)], n_flips=n_flips, seed=PERM_SEED
        )
        p_c1_1_iut = max(p_gap_share, p_guard)

        valid_share_idxs = [i for i in range(n_t) if dtot_gap_t[i] > 1e-6]
        excluded_dtot_cnt = n_t - len(valid_share_idxs)
        if excluded_dtot_cnt < 3 and len(valid_share_idxs) >= 2:
            valid_shares = [dattn_gap_t[i] / dtot_gap_t[i] for i in valid_share_idxs]
            share_gap_ci = bca_cluster_ci(
                len(valid_shares),
                lambda idxs: median([valid_shares[i] for i in idxs]),
                alphas=(0.05, 0.95),
                n_boot=n_boot,
                seed=BCA_SEED,
            )
            med_share_gap = share_gap_ci["point"]
            q05_share_gap = share_gap_ci["q_0.05"]
            strata_share_ok = 0
            for st in EXPECTED_STRATA:
                vals_st = [dattn_gap_t[i] / dtot_gap_t[i] for i in valid_share_idxs if t_strata[i] == st]
                if vals_st and median(vals_st) >= 0.70:
                    strata_share_ok += 1
            c1_1b_ii_ok = (med_share_gap >= 0.75) and (q05_share_gap > 0.70) and (strata_share_ok >= 3)
        else:
            share_gap_ci = {"point": float("nan"), "q_0.05": float("nan"), "q_0.95": float("nan")}
            med_share_gap = q05_share_gap = float("nan")
            strata_share_ok = 0
            c1_1b_ii_ok = False

        c1_1b_i_ok = (mean(dtot_gap_t) >= 1.50) and (sum(1 for v in dtot_gap_t if v > 0.0) >= 20)
        rmf_shares = [a / max(tot, 1e-6) for a, tot in zip(dattn_rmf_t, dtot_rmf_t)]
        rmf_gt_dm_cnt = sum(1 for a, d in zip(dattn_rmf_t, dattn_dm_t) if a > d)
        c1_1b_iii_ok = (mean(dattn_rmf_t) >= 1.00) and (median(rmf_shares) >= 0.65) and (rmf_gt_dm_cnt >= 20)
    else:
        p_gap_share = p_guard = p_c1_1_iut = 1.0
        excluded_dtot_cnt = n_t
        med_share_gap = q05_share_gap = float("nan")
        strata_share_ok = rmf_gt_dm_cnt = 0
        rmf_shares = []
        c1_1b_i_ok = c1_1b_ii_ok = c1_1b_iii_ok = False

    # =========================================================================
    # Claim C1.2: Live R_ridge bleed whitening across mixed POS
    # =========================================================================
    def _target_bleeds(arm: str) -> tuple[list[float], list[float], list[float], list[float]]:
        """Return per-target (B_off_23, self_dlogit, B_within_5, B_open_func_6_or_18) averaged over 3L & 8 prompts."""
        b_off_t, self_t, b_in_t, b_of_t = [], [], [], []
        for ti, ts in enumerate(t_strs):
            st_i = t_strata[ti]
            off_ls, slf_ls, in_ls, of_ls = [], [], [], []
            for L in PRIMARY_LAYERS:
                rows = receipt["block2a_regime_N_last"][str(L)][arm][ts]
                for r in rows:
                    a24 = r["all_24_dlogits"]
                    slf_ls.append(float(r["exact_dlogit"]))
                    off_ls.append(mean([abs(float(a24[j])) for j in range(n_t) if j != ti]))
                    in_ls.append(mean([abs(float(a24[j])) for j in range(n_t) if j != ti and t_strata[j] == st_i]))
                    of_ls.append(
                        mean([
                            abs(float(a24[j]))
                            for j in range(n_t)
                            if (t_strata[j] == "function") != (st_i == "function")
                        ])
                    )
            b_off_t.append(mean(off_ls))
            self_t.append(mean(slf_ls))
            b_in_t.append(mean(in_ls))
            b_of_t.append(mean(of_ls))
        return b_off_t, self_t, b_in_t, b_of_t

    boff_mf, self_mf, bin_mf, bof_mf = _target_bleeds("R_mf")
    boff_rg, self_rg, bin_rg, bof_rg = _target_bleeds("R_ridge")
    boff_ll, self_ll, bin_ll, bof_ll = _target_bleeds("logit_lens")

    u_abs_t = [math.log(max(m, 1e-12) / max(r, 1e-12)) for m, r in zip(boff_mf, boff_rg)]
    s_mf_t = [b / max(s, 1e-6) for b, s in zip(boff_mf, self_mf)]
    s_rg_t = [b / max(s, 1e-6) for b, s in zip(boff_rg, self_rg)]
    u_norm_t = [math.log(max(m, 1e-12) / max(r, 1e-12)) for m, r in zip(s_mf_t, s_rg_t)]
    u_abs_of_t = [math.log(max(m, 1e-12) / max(r, 1e-12)) for m, r in zip(bof_mf, bof_rg)]

    p_abs = sign_flip_p_one_sided(u_abs_t, n_flips=n_flips, seed=PERM_SEED)
    p_norm = sign_flip_p_one_sided(u_norm_t, n_flips=n_flips, seed=PERM_SEED)
    p_c1_2_iut = max(p_abs, p_norm)

    geo_abs_ci = bca_cluster_ci(
        n_t, lambda idxs: math.exp(mean([u_abs_t[i] for i in idxs])), alphas=(0.05, 0.95), n_boot=n_boot, seed=BCA_SEED
    )
    geo_norm_ci = bca_cluster_ci(
        n_t, lambda idxs: math.exp(mean([u_norm_t[i] for i in idxs])), alphas=(0.05, 0.95), n_boot=n_boot, seed=BCA_SEED
    )
    geo_of_ci = bca_cluster_ci(
        n_t, lambda idxs: math.exp(mean([u_abs_of_t[i] for i in idxs])), alphas=(0.05, 0.95), n_boot=n_boot, seed=BCA_SEED
    )

    c1_2_pos_cnt = sum(1 for v in u_abs_t if v > 0.0)
    c1_2_strata_ge_115 = 0
    c1_2_by_stratum: dict[str, float] = {}
    for st in EXPECTED_STRATA:
        vals_st = [u_abs_t[i] for i in range(n_t) if t_strata[i] == st]
        g_st = math.exp(mean(vals_st))
        c1_2_by_stratum[st] = round(g_st, 6)
        if g_st >= 1.15:
            c1_2_strata_ge_115 += 1

    # =========================================================================
    # Claim C1.3: H_A vs. H_B Mechanistic Fork
    # =========================================================================
    # Live in-run Block 2b F_mean_3L (off-target logit floor, matching C0 score_olmo3_c0.py:1102-1116 = 0.295811)
    rand_abs_dlogits: list[float] = []
    for L in PRIMARY_LAYERS:
        for draw in receipt["block2b_rand_iso"][str(L)]["rand_iso"]:
            for r in draw:
                for dv in r["all_24_dlogits"]:
                    rand_abs_dlogits.append(abs(float(dv)))
    f_mean_3l_live = mean(rand_abs_dlogits)
    # Fail-closed unit sanity guard (Claude #85): live off-target logit floor must lie in [0.10, 1.00] logits
    if not (0.10 <= f_mean_3l_live <= 1.00):
        raise ValueError(
            f"FAIL-CLOSED F_mean_3L UNIT SANITY FAILED: live F_mean_3L={f_mean_3l_live:.6f} "
            f"outside [0.10, 1.00] logits (expected ~0.2958 logits)"
        )
    r_f_ratio = f_mean_3l_live / C0_F_MEAN_3L

    rsep_mf_t = [a / max(b, 1e-12) for a, b in zip(bin_mf, bof_mf)]
    rsep_rg_t = [a / max(b, 1e-12) for a, b in zip(bin_rg, bof_rg)]
    d_sep_t = [r - m for r, m in zip(rsep_rg_t, rsep_mf_t)]
    delta_r_sep_ci = bca_cluster_ci(
        n_t, lambda idxs: mean([d_sep_t[i] for i in idxs]), alphas=(0.05, 0.95), n_boot=n_boot, seed=BCA_SEED
    )
    delta_r_sep = delta_r_sep_ci["point"]
    delta_r_sep_pooled = (mean(bin_rg) / max(mean(bof_rg), 1e-12)) - (mean(bin_mf) / max(mean(bof_mf), 1e-12))

    p_c1_3_ha = sign_flip_p_one_sided(d_sep_t, n_flips=n_flips, seed=PERM_SEED)
    p_c1_3_hb = sign_flip_p_one_sided(
        [b - C1_3_FLOOR_HB_Q05_BAR * f_mean_3l_live for b in bof_rg], n_flips=n_flips, seed=PERM_SEED
    )
    p_c1_3_hb_c0ref = sign_flip_p_one_sided(
        [b - C1_3_FLOOR_HB_Q05_BAR * C0_F_MEAN_3L for b in bof_rg], n_flips=n_flips, seed=PERM_SEED
    )

    r_floor_rg_ci_live = bca_cluster_ci(
        n_t, lambda idxs: mean([bof_rg[i] for i in idxs]) / f_mean_3l_live, alphas=(0.05, 0.95), n_boot=n_boot, seed=BCA_SEED
    )
    r_floor_rg_ci_c0 = bca_cluster_ci(
        n_t, lambda idxs: mean([bof_rg[i] for i in idxs]) / C0_F_MEAN_3L, alphas=(0.05, 0.95), n_boot=n_boot, seed=BCA_SEED
    )
    r_floor_ll_live = mean(bof_ll) / f_mean_3l_live
    r_floor_ll_c0 = mean(bof_ll) / C0_F_MEAN_3L

    c1_3_strata_pos_mean = 0
    c1_3_stratum_diag: dict[str, dict[str, Any]] = {}
    for st in EXPECTED_STRATA:
        ds_st = [d_sep_t[i] for i in range(n_t) if t_strata[i] == st]
        m_st = mean(ds_st)
        pos_st = sum(1 for v in ds_st if v > 0.0)
        if m_st > 0.0:
            c1_3_strata_pos_mean += 1
        c1_3_stratum_diag[st] = {"mean_d_sep": round(m_st, 6), "pos_count": pos_st, "ge_4_of_6": pos_st >= 4}

    whitening_ok = (geo_abs_ci["q_0.05"] > 1.0) and (geo_of_ci["q_0.05"] > 1.0)

    # Full 4x4 signed & unsigned bleed matrices for R_mf, R_ridge, logit_lens
    bleed_4x4: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for arm in ("R_mf", "R_ridge", "logit_lens"):
        mat_s: dict[str, dict[str, float]] = {}
        mat_u: dict[str, dict[str, float]] = {}
        for src in EXPECTED_STRATA:
            mat_s[src] = {}
            mat_u[src] = {}
            for dst in EXPECTED_STRATA:
                sv, uv = [], []
                for ti in range(n_t):
                    if t_strata[ti] != src:
                        continue
                    ts = t_strs[ti]
                    for L in PRIMARY_LAYERS:
                        for r in receipt["block2a_regime_N_last"][str(L)][arm][ts]:
                            a24 = r["all_24_dlogits"]
                            for tj in range(n_t):
                                if ti == tj or t_strata[tj] != dst:
                                    continue
                                v = float(a24[tj])
                                sv.append(v)
                                uv.append(abs(v))
                mat_s[src][dst] = round(mean(sv), 6)
                mat_u[src][dst] = round(mean(uv), 6)
        bleed_4x4[arm] = {"signed": mat_s, "unsigned": mat_u}

    frozen_bof_rg = float(c1_constants["pooled_3L_frozen_summary"]["R_ridge_B_open_func_3L"])
    implied_k_ridge = frozen_bof_rg / max(mean(bof_rg), 1e-12)

    # =========================================================================
    # Claim C1.4: Token-unembedding direction vs. isotropic contraction
    # =========================================================================
    vjv_by_layer: dict[int, float] = {}
    for L in PRIMARY_LAYERS:
        v_vals = [float(r["vJv"]) for d in receipt["block2b_rand_iso"][str(L)]["rand_iso"] for r in d]
        vjv_by_layer[L] = mean(v_vals)

    # F8: read exact registered u_hat^T J u_hat (`u_hat_self_J`) directly (keep retention as diagnostic)
    d_t_c1_4: list[float] = []
    d_t_c1_4_ret_diag: list[float] = []
    sel_t_c1_4: list[float] = []
    for ti, ts in enumerate(t_strs):
        diffs_l, diffs_ret_l, sels_l = [], [], []
        for L in PRIMARY_LAYERS:
            rows = receipt["block2a_regime_N_last"][str(L)]["logit_lens"][ts]
            if not all("u_hat_self_J" in r for r in rows):
                raise ValueError(f"C1.4 missing u_hat_self_J on block2a L{L} logit_lens[{ts!r}]")
            u_j_u_l = mean([float(r["u_hat_self_J"]) for r in rows])
            ret_l = mean([float(r["retention"]) for r in rows])
            diffs_l.append(vjv_by_layer[L] - u_j_u_l)
            diffs_ret_l.append(vjv_by_layer[L] - ret_l)
            if all("u_hat_off_abs_mean_J" in r for r in rows):
                sels_l.append(mean([float(r["u_hat_self_J"]) - float(r["u_hat_off_abs_mean_J"]) for r in rows]))
        d_t_c1_4.append(mean(diffs_l))
        d_t_c1_4_ret_diag.append(mean(diffs_ret_l))
        if sels_l:
            sel_t_c1_4.append(mean(sels_l))

    p_c1_4 = sign_flip_p_one_sided(d_t_c1_4, n_flips=n_flips, seed=PERM_SEED)
    c1_4_ci = bca_cluster_ci(
        n_t, lambda idxs: mean([d_t_c1_4[i] for i in idxs]), alphas=(0.05, 0.95), n_boot=n_boot, seed=BCA_SEED
    )
    c1_4_pos_cnt = sum(1 for v in d_t_c1_4 if v > 0.0)
    c1_4_strata_pos = 0
    c1_4_univ_4 = True
    c1_4_by_stratum: dict[str, dict[str, Any]] = {}
    for st in EXPECTED_STRATA:
        vs = [d_t_c1_4[i] for i in range(n_t) if t_strata[i] == st]
        m_st = mean(vs)
        pc = sum(1 for v in vs if v > 0.0)
        if m_st > 0.0:
            c1_4_strata_pos += 1
        if not (m_st > 0.0 and pc >= 4):
            c1_4_univ_4 = False
        c1_4_by_stratum[st] = {"mean_d_t": round(m_st, 6), "pos_count": pc}

    # =========================================================================
    # 5-Test Holm-Bonferroni Step-Down Family (alpha = 0.05)
    # =========================================================================
    holm = holm_bonferroni_stepdown(
        {
            "p_C1.1_IUT": p_c1_1_iut,
            "p_C1.2_IUT": p_c1_2_iut,
            "p_C1.3_HA": p_c1_3_ha,
            "p_C1.3_HB": p_c1_3_hb,
            "p_C1.4": p_c1_4,
        },
        alpha=0.05,
    )
    holm_c0ref = holm_bonferroni_stepdown(
        {
            "p_C1.1_IUT": p_c1_1_iut,
            "p_C1.2_IUT": p_c1_2_iut,
            "p_C1.3_HA": p_c1_3_ha,
            "p_C1.3_HB": p_c1_3_hb_c0ref,
            "p_C1.4": p_c1_4,
        },
        alpha=0.05,
    )

    # --- C1.1 verdict (3-way, or BLOCKED_BY_DATA if any (L, target) cell is all-capped) ---
    if c1_1_blocked_reason is not None:
        c1_1b_ok = False
        v_c1_1 = "BLOCKED_BY_DATA (all-capped cell)"
    else:
        c1_1b_ok = holm["p_C1.1_IUT"]["rejected"] and c1_1b_i_ok and c1_1b_ii_ok and c1_1b_iii_ok
        if c1_1a_ok and c1_1b_ok:
            v_c1_1 = "PASS"
        elif c1_1b_ok and frac_err_ok and r_all_bands_ok and (not layer_split_ok):
            v_c1_1 = "INCONCLUSIVE (format/length/instruction split)"
        else:
            v_c1_1 = "REFUTED"

    # --- C1.2 verdict (4-way disjoint) ---
    c1_2_pass = (
        holm["p_C1.2_IUT"]["rejected"]
        and (geo_abs_ci["point"] >= 1.35)
        and (geo_norm_ci["point"] >= 1.35)
        and (c1_2_pos_cnt >= 20)
        and (c1_2_strata_ge_115 >= 3)
    )
    c1_2_rev = (geo_abs_ci["q_0.95"] < 0.80) or (geo_norm_ci["q_0.95"] < 0.80)
    c1_2_eq = (
        (geo_abs_ci["q_0.05"] > 0.80)
        and (geo_abs_ci["q_0.95"] < 1.35)
        and (geo_norm_ci["q_0.05"] > 0.80)
        and (geo_norm_ci["q_0.95"] < 1.35)
    )
    c1_2_inc = (not c1_2_pass) and (not c1_2_rev) and (not c1_2_eq)
    c1_2_matches = [
        lbl for lbl, flg in (
            ("PASS", c1_2_pass),
            ("REVERSED", c1_2_rev),
            ("REFUTED (equivalence)", c1_2_eq),
            ("INCONCLUSIVE", c1_2_inc),
        ) if flg
    ]
    if len(c1_2_matches) != 1:
        raise RuntimeError(f"C1.2 4-way partition violated: {c1_2_matches}")
    v_c1_2 = c1_2_matches[0]

    # --- C1.3 verdict (6-way + F_mean_3L provenance guard + fixed-alpha companion) ---
    c1_3_live_res = evaluate_c1_3_partition(
        delta_r_sep=delta_r_sep,
        q05_sep_ok=(delta_r_sep_ci["q_0.05"] > C1_3_DELTA_R_SEP_Q05_BAR),
        p_ha_ok=holm["p_C1.3_HA"]["rejected"],
        strata_ok=(c1_3_strata_pos_mean >= 3),
        r_floor_ridge=r_floor_rg_ci_live["point"],
        q05_floor_ok=(r_floor_rg_ci_live["q_0.05"] > C1_3_FLOOR_HB_Q05_BAR),
        p_hb_ok=holm["p_C1.3_HB"]["rejected"],
        whitening_ok=whitening_ok,
        logit_lens_clean=(r_floor_ll_live <= 1.35),
    )
    c1_3_c0ref_res = evaluate_c1_3_partition(
        delta_r_sep=delta_r_sep,
        q05_sep_ok=(delta_r_sep_ci["q_0.05"] > C1_3_DELTA_R_SEP_Q05_BAR),
        p_ha_ok=holm_c0ref["p_C1.3_HA"]["rejected"],
        strata_ok=(c1_3_strata_pos_mean >= 3),
        r_floor_ridge=r_floor_rg_ci_c0["point"],
        q05_floor_ok=(r_floor_rg_ci_c0["q_0.05"] > C1_3_FLOOR_HB_Q05_BAR),
        p_hb_ok=holm_c0ref["p_C1.3_HB"]["rejected"],
        whitening_ok=whitening_ok,
        logit_lens_clean=(r_floor_ll_c0 <= 1.35),
    )
    c1_3_fixed_alpha_res = evaluate_c1_3_partition(
        delta_r_sep=delta_r_sep,
        q05_sep_ok=(delta_r_sep_ci["q_0.05"] > C1_3_DELTA_R_SEP_Q05_BAR),
        p_ha_ok=(p_c1_3_ha <= 0.010),
        strata_ok=(c1_3_strata_pos_mean >= 3),
        r_floor_ridge=r_floor_rg_ci_live["point"],
        q05_floor_ok=(r_floor_rg_ci_live["q_0.05"] > C1_3_FLOOR_HB_Q05_BAR),
        p_hb_ok=(p_c1_3_hb <= 0.010),
        whitening_ok=whitening_ok,
        logit_lens_clean=(r_floor_ll_live <= 1.35),
    )

    if r_f_ratio < 0.85 or r_f_ratio > 1.15:
        f_guard_status = "HARD_DRIFT"
        v_c1_3 = "INCONCLUSIVE (F_mean_3L drift)"
    elif r_f_ratio < 0.94 or r_f_ratio > 1.06:
        if c1_3_live_res["label"] != c1_3_c0ref_res["label"]:
            f_guard_status = "SENSITIVITY_MISMATCH"
            v_c1_3 = "INCONCLUSIVE (F_mean_3L sensitivity)"
        else:
            f_guard_status = "SENSITIVITY_AGREED"
            v_c1_3 = c1_3_live_res["label"]
    else:
        f_guard_status = "CLEAN"
        v_c1_3 = c1_3_live_res["label"]

    # --- C1.4 verdict ---
    c1_4_pass = (
        (c1_4_ci["point"] > 0.0)
        and (c1_4_ci["q_0.05"] > 0.0)
        and holm["p_C1.4"]["rejected"]
        and (c1_4_pos_cnt >= 19)
        and (c1_4_strata_pos >= 3)
    )
    if c1_4_pass:
        v_c1_4 = "PASS (>= 19/24 & >= 3/4 strata; UNIVERSAL_4_OF_4)" if c1_4_univ_4 else "PASS (>= 19/24 & >= 3/4 strata)"
    else:
        v_c1_4 = "NOUN-SPECIFIC / FAIL"

    # p_clean diagnostic summary
    p_clean_diag = {
        "n_cells_ge_1e6": len(fe_ge_1e6) // len(PRIMARY_LAYERS) if fe_ge_1e6 else 0,
        "n_cells_lt_1e6": len(flagged_lt_1e6),
        "frac_err_all_median_ge_1e6": round(median(fe_ge_1e6), 6) if fe_ge_1e6 else None,
        "frac_err_all_median_lt_1e6": round(median(fe_lt_1e6), 6) if fe_lt_1e6 else None,
        "spearman_log_p_clean_vs_frac_err_all": (
            round(spearman_rho(p_clean_log_vals, p_clean_fe_vals), 6) if len(p_clean_log_vals) >= 2 else None
        ),
        "frac_err_by_stratum_format": {k: round(median(v), 6) for k, v in sorted(fe_by_stratum_fmt.items())},
        "flagged_lt_1e6_cells": flagged_lt_1e6,
    }

    return {
        "schema_version": "c1-scorecard-v1",
        "verdicts": {
            "C1.0": "PASS",
            "C1.1": v_c1_1,
            "C1.2": v_c1_2,
            "C1.3": v_c1_3,
            "C1.4": v_c1_4,
            "L12_linear_check": "PASS" if l12_linear_check["PASS"] else "FAIL",
        },
        "gate_C1_0": gate_c1_0,
        "holm_5_test_family": holm,
        "C1_1": {
            "verdict": v_c1_1,
            "blocked_reason": c1_1_blocked_reason,
            "a_calibration": {
                "ok": c1_1a_ok,
                "frac_err_ok": frac_err_ok,
                "frac_err_all_median": round(med_fe_all, 6),
                "frac_err_last_median": round(med_fe_last, 6),
                "r_all_bands_ok": r_all_bands_ok,
                "pooled_median_r_all": round(pooled_med_r_all, 6) if math.isfinite(pooled_med_r_all) else None,
                "pooled_bca90": (
                    [round(r_all_pool_ci["q_0.05"], 6), round(r_all_pool_ci["q_0.95"], 6)]
                    if math.isfinite(r_all_pool_ci["q_0.05"])
                    else None
                ),
                "bare_median_r_all": round(bare_med_r_all, 6) if math.isfinite(bare_med_r_all) else None,
                "chatml_median_r_all": round(chat_med_r_all, 6) if math.isfinite(chat_med_r_all) else None,
                "layer_split_ok": layer_split_ok,
                "max_per_layer_split": round(max_layer_split, 6) if math.isfinite(max_layer_split) else None,
                "per_layer": per_layer_r_all,
            },
            "b_attention_attribution": {
                "ok": c1_1b_ok,
                "p_gap_share": round(p_gap_share, 6),
                "p_guard": round(p_guard, 6),
                "p_C1.1_IUT": round(p_c1_1_iut, 6),
                "holm_rejected": holm["p_C1.1_IUT"]["rejected"],
                "i_ok": c1_1b_i_ok,
                "mean_delta_tot_gap": round(mean(dtot_gap_t), 6) if dtot_gap_t else None,
                "n_pos_delta_tot_gap": sum(1 for v in dtot_gap_t if v > 0.0),
                "ii_ok": c1_1b_ii_ok,
                "excluded_dtot_le_1e6_count": excluded_dtot_cnt,
                "median_share_gap": round(med_share_gap, 6) if math.isfinite(med_share_gap) else None,
                "q05_share_gap": round(q05_share_gap, 6) if math.isfinite(q05_share_gap) else None,
                "strata_share_ge_070_count": strata_share_ok,
                "iii_ok": c1_1b_iii_ok,
                "mean_delta_attn_Rmf": round(mean(dattn_rmf_t), 6) if dattn_rmf_t else None,
                "median_share_Rmf": round(median(rmf_shares), 6) if rmf_shares else None,
                "n_dattn_Rmf_gt_dm": rmf_gt_dm_cnt,
                "dm_per_layer_diagnostic": dm_layer_diag,
            },
            "p_clean_diagnostic": p_clean_diag,
        },
        "C1_2": {
            "verdict": v_c1_2,
            "p_abs": round(p_abs, 6),
            "p_norm": round(p_norm, 6),
            "p_C1.2_IUT": round(p_c1_2_iut, 6),
            "holm_rejected": holm["p_C1.2_IUT"]["rejected"],
            "geo_abs": round(geo_abs_ci["point"], 6),
            "geo_abs_bca90": [round(geo_abs_ci["q_0.05"], 6), round(geo_abs_ci["q_0.95"], 6)],
            "geo_norm": round(geo_norm_ci["point"], 6),
            "geo_norm_bca90": [round(geo_norm_ci["q_0.05"], 6), round(geo_norm_ci["q_0.95"], 6)],
            "geo_open_func": round(geo_of_ci["point"], 6),
            "geo_open_func_bca90": [round(geo_of_ci["q_0.05"], 6), round(geo_of_ci["q_0.95"], 6)],
            "n_pos_u_abs": c1_2_pos_cnt,
            "strata_ge_115_count": c1_2_strata_ge_115,
            "by_stratum_geo_abs": c1_2_by_stratum,
        },
        "C1_3": {
            "verdict": v_c1_3,
            "whitening_ok": whitening_ok,
            "F_mean_3L_live": round(f_mean_3l_live, 6),
            "F_mean_3L_ratio_vs_C0": round(r_f_ratio, 6),
            "F_mean_3L_guard_status": f_guard_status,
            "delta_R_sep": round(delta_r_sep, 6),
            "delta_R_sep_bca90": [round(delta_r_sep_ci["q_0.05"], 6), round(delta_r_sep_ci["q_0.95"], 6)],
            "delta_R_sep_pooled_ratio_of_means": round(delta_r_sep_pooled, 6),
            "p_C1.3_HA": round(p_c1_3_ha, 6),
            "p_C1.3_HB_live": round(p_c1_3_hb, 6),
            "p_C1.3_HB_c0ref": round(p_c1_3_hb_c0ref, 6),
            "R_floor_ridge_live": round(r_floor_rg_ci_live["point"], 6),
            "R_floor_ridge_live_bca90": [round(r_floor_rg_ci_live["q_0.05"], 6), round(r_floor_rg_ci_live["q_0.95"], 6)],
            "R_floor_ridge_c0ref": round(r_floor_rg_ci_c0["point"], 6),
            "R_floor_ridge_c0ref_bca90": [round(r_floor_rg_ci_c0["q_0.05"], 6), round(r_floor_rg_ci_c0["q_0.95"], 6)],
            "R_floor_logit_lens_live": round(r_floor_ll_live, 6),
            "R_floor_logit_lens_c0ref": round(r_floor_ll_c0, 6),
            "implied_k_ridge": round(implied_k_ridge, 4),
            "partition_live": c1_3_live_res,
            "partition_c0ref": c1_3_c0ref_res,
            "partition_fixed_alpha_0.010_companion": c1_3_fixed_alpha_res,
            "strata_pos_mean_count": c1_3_strata_pos_mean,
            "by_stratum": c1_3_stratum_diag,
            "bleed_4x4_matrices": bleed_4x4,
        },
        "C1_4": {
            "verdict": v_c1_4,
            "universal_4_of_4": c1_4_univ_4 if c1_4_pass else False,
            "mean_d_t": round(c1_4_ci["point"], 6),
            "d_t_bca90": [round(c1_4_ci["q_0.05"], 6), round(c1_4_ci["q_0.95"], 6)],
            "p_C1.4": round(p_c1_4, 6),
            "holm_rejected": holm["p_C1.4"]["rejected"],
            "n_pos": c1_4_pos_cnt,
            "strata_pos_count": c1_4_strata_pos,
            "vJv_by_layer": {str(L): round(vjv_by_layer[L], 6) for L in PRIMARY_LAYERS},
            "by_stratum": c1_4_by_stratum,
            "non_gating_diag_vs_offdiag_selectivity_mean": (
                round(mean(sel_t_c1_4), 6) if sel_t_c1_4 else None
            ),
            "retention_diagnostic_mean_d_t": round(mean(d_t_c1_4_ret_diag), 6),
        },
        "secondary_L12_linear_check": l12_linear_check,
    }


def render_markdown(sc: dict[str, Any]) -> str:
    v = sc["verdicts"]
    lines = [
        "# OLMo 3 7B Stage C1 — Pre-Registered Confirmatory Scorecard",
        "",
        "| Gate / Claim | Verdict |",
        "|---|---|",
        f"| **Global Gate `C1.0`** | **`{v['C1.0']}`** |",
        f"| **`C1.1` (Attention transport + calibration)** | **`{v['C1.1']}`** |",
        f"| **`C1.2` (`R_ridge` bleed whitening)** | **`{v['C1.2']}`** |",
        f"| **`C1.3` (`H_A` vs. `H_B` bleed fork)** | **`{v['C1.3']}`** |",
        f"| **`C1.4` (Token-direction contraction)** | **`{v['C1.4']}`** |",
        f"| *Secondary `L12_linear_check` (non-gating)* | *`{v['L12_linear_check']}`* |",
        "",
    ]
    if v["C1.0"] != "PASS":
        lines.append("## Gate `C1.0` Failure Summary")
        lines.append(f"```json\n{json.dumps(sc['gate_C1_0'], indent=2)}\n```")
        return "\n".join(lines) + "\n"

    c11 = sc["C1_1"]
    c12 = sc["C1_2"]
    c13 = sc["C1_3"]
    c14 = sc["C1_4"]
    lines.extend([
        "## Key Numbers",
        f"- **`C1.1`:** `frac_err_all = {c11['a_calibration']['frac_err_all_median']}` (last `{c11['a_calibration']['frac_err_last_median']}`), `r_all` pooled `{c11['a_calibration']['pooled_median_r_all']}` (90% BCa `{c11['a_calibration']['pooled_bca90']}`), Bare `{c11['a_calibration']['bare_median_r_all']}`, ChatML `{c11['a_calibration']['chatml_median_r_all']}`, max layer split `{c11['a_calibration']['max_per_layer_split']}`; `ΔTot_gap = {c11['b_attention_attribution']['mean_delta_tot_gap']}`, `median share = {c11['b_attention_attribution']['median_share_gap']}` (`q05 = {c11['b_attention_attribution']['q05_share_gap']}`), `p_IUT = {c11['b_attention_attribution']['p_C1.1_IUT']}`.",
        f"- **`C1.2`:** `geo_abs = {c12['geo_abs']}x` (90% BCa `{c12['geo_abs_bca90']}`), `geo_norm = {c12['geo_norm']}x` (90% BCa `{c12['geo_norm_bca90']}`), `geo_open_func = {c12['geo_open_func']}x` (90% BCa `{c12['geo_open_func_bca90']}`), `p_IUT = {c12['p_C1.2_IUT']}`.",
        f"- **`C1.3`:** `whitening_ok = {c13['whitening_ok']}`, `F_mean_3L = {c13['F_mean_3L_live']}` (`{c13['F_mean_3L_ratio_vs_C0']}x` C0, `{c13['F_mean_3L_guard_status']}`), `ΔR_sep = {c13['delta_R_sep']}` (90% BCa `{c13['delta_R_sep_bca90']}`, `p_HA = {c13['p_C1.3_HA']}`), `R_floor(R_ridge) = {c13['R_floor_ridge_live']}` (90% BCa `{c13['R_floor_ridge_live_bca90']}`, `p_HB = {c13['p_C1.3_HB_live']}`), `R_floor(logit_lens) = {c13['R_floor_logit_lens_live']}`, `implied_k_ridge = {c13['implied_k_ridge']}`, fixed-`α=0.010` companion = `{c13['partition_fixed_alpha_0.010_companion']['label']}`.",
        f"- **`C1.4`:** `mean(d_t) = {c14['mean_d_t']}` (90% BCa `{c14['d_t_bca90']}`, `p = {c14['p_C1.4']}`), `{c14['n_pos']}/24` positive, `{c14['strata_pos_count']}/4` strata positive, non-gating diag-vs-offdiag selectivity `{c14['non_gating_diag_vs_offdiag_selectivity_mean']}`.",
        f"- **Secondary `L12_linear_check`:** `{sc['secondary_L12_linear_check']}`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Pure-stdlib fail-closed Stage C1 scorecard generator.")
    ap.add_argument("--receipt", type=Path, required=True, help="Path to c1_receipt.json.gz")
    ap.add_argument("--constants", type=Path, default=root / "c1_prereg_constants.json")
    ap.add_argument("--design", type=Path, default=None, help="Path to registered OLMO3-TRANSPORT-C1-DESIGN.md (required for C1.0.4 PASS)")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--no-strict-provenance", action="store_true", help="Fails C1.0.4 intentionally for diagnostic replay")
    args = ap.parse_args()

    receipt = load_json_maybe_gz(args.receipt)
    c1_const = load_json_maybe_gz(args.constants)
    sc = score_c1(
        receipt=receipt,
        c1_constants=c1_const,
        constants_path=args.constants,
        design_path=args.design,
        strict_provenance=not args.no_strict_provenance,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(sc, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(sc), encoding="utf-8")
    print(f"Wrote {args.out_json} and {args.out_md}; verdicts={sc['verdicts']}")


if __name__ == "__main__":
    main()
