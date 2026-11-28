#!/usr/bin/env python3
"""score_olmo3_c3b.py — 100% Pure Python 3.10+ Stdlib Confirmatory & Diagnostic Scorer for Stage C3b (WSP v2).

Zero third-party imports (no numpy, scipy, torch, safetensors). Reproducible byte-for-byte under
`/usr/bin/python3 -I` (`cmp` verified). Imports frozen mechanical scorers, `holm_reject()`,
`resolve_verdict()`, `cohens_kappa()`, and `raw_agreement()` from `c3_scoring.py`.

Every gate threshold is either read directly from frozen `c3b_prereg_constants.json` or defined once
in `GATE_BARS` and evaluated via pure gate functions so boundary-table tests cover 100% of bars.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import c3_scoring  # noqa: E402

SCORECARD_SCHEMA_VERSION = "olmo3-stage-c3b-scorecard-v1"
EXPECTED_RECEIPT_SCHEMA = "olmo3-stage-c3b-receipt-v1"
EXPECTED_READER_SCHEMA = "olmo3-stage-c3b-reader-labels-v1"
EXPECTED_PREFLIGHT_SCHEMA = "olmo3-stage-c3b-reader-preflight-v1"

EXPECTED_C3B_DESIGN_SHA256 = "a9ca200dccb19ad3efb3b0cf358c546dfffb10904f673f64667f9b580bdc7c75"
EXPECTED_C3B_CONSTANTS_SHA256 = "3bb980e579cef2326d5d2adc3c458cca8ba4140da5bb0e7db2f196e3484ca9db"
EXPECTED_C3B_BANK_SHA256 = "a32662a7b3671a4613b7b412e78ff4b14a7fcd41e385fb9d43da31bd1fea2656"
EXPECTED_C3_SCORING_SHA256 = "b2fdfdf480f28554117765caa49126e4567481e6d34d2917ef1287cc0675040f"

PERM_SEED = 2026112701
BCA_SEED = 2026112702
DEFAULT_N_PERM = 100_000
DEFAULT_N_BOOT = 10_000
EPS = 1e-12

# Single source of truth for Stage C3b thresholds not already stored as numeric fields in JSON
GATE_BARS: dict[str, Any] = {
    "sham_dlogit_min": 0.0,
    "sham_dlogit_max": 2.00,
    "cv_max": 1.0,
    "rand_frob_tol": 1e-3,
    "rand_kl_max": 0.80,
    "ban_stem_max": 0.10,
    "ban_arms_min": 6,
    "plain_sanity_n": 56,
    "plain_sanity_min": 0.90,
    "d4_masked_seed_min": 1.15,
    "d4_mismasked_seed_max": 1.05,
    "d4_masked_l2_max": 1.10,
    "d4_mismasked_l2_max": 1.25,
    "d4_masked_eff_min": 1.15,
    "d4_mismasked_eff_max": 1.05,
    "dir_cos_ret_min": 0.97,
    "nomd_room_24_min": 0.90,
    "nomd_room_8_min": 0.625,
    "nomd_room_32_md_min": 0.84,
    "nomd_room_32_ct_min": 2.00,
    "pool_kappa_min": 0.60,
    "pool_ban_kappa_min": 0.55,
    "arm_prev_lo": 0.10,
    "arm_prev_hi": 0.90,
    "arm_ra_min": 0.85,
    "rand_kap_min": 0.50,
    "wsp_kap_min": 0.55,
    "h1_mean_y_min": 0.20,
    "h1_len_lo": 0.85,
    "h1_len_hi": 1.15,
    "h1_any_coh_min": 0.85,
    "h1_both_coh_drop_max": 0.15,
    "h1_broken_max": 0.15,
    "h2_mean_u_min": 0.50,
    "h2_delta_plain_min": 0.20,
    "h2_delta_rand_min": 0.15,
    "h2_len_lo": 0.85,
    "h2_len_hi": 1.15,
    "h2_broken_max": 0.10,
    "dac_pirate_delta_min": 0.25,
    "dac_pirate_broken_max": 2,
    "dac_nomd_has_md_max": 0.75,
    "dac_nomd_delta_ct_min": 0.50,
    "dac_nomd_broken_max": 1,
}

REFUSAL_RE = re.compile(
    r"\b(cannot|can't|won't|unable to|sorry|apologize|illegal|unethical|harmful|dangerous|cannot assist|cannot help|cannot provide)\b",
    re.IGNORECASE,
)
_EM_DASH_RE = re.compile(r"\u2014|(?<!-)--(?!-)")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

D4_FAIL_FLAG = "low-gamma pre_norm masking efficiency unverified on L24"


# ---------------------------------------------------------------------------
# Pure-Stdlib Deterministic Statistics (`math.fsum` throughout)
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_or_gz(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    if not xs:
        raise ValueError("mean() requires non-empty sequence")
    return math.fsum(xs) / len(xs)


def median(xs: list[float]) -> float:
    if not xs:
        raise ValueError("median() requires non-empty sequence")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 == 1 else 0.5 * (float(s[mid - 1]) + float(s[mid]))


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        raise ValueError("quantile() requires non-empty sequence")
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


def paired_sign_flip_p_one_sided(diffs: list[float], *, n_perm: int = DEFAULT_N_PERM, seed: int = PERM_SEED) -> float:
    """Deterministic one-sided paired sign-flip permutation p-value (tests mean(diffs) > 0)."""
    if not diffs:
        raise ValueError("paired_sign_flip_p_one_sided requires non-empty diffs")
    obs = math.fsum(diffs)
    tol = EPS * max(1.0, abs(obs))
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        s = math.fsum(x if rng.getrandbits(1) else -x for x in diffs)
        if s >= obs - tol:
            ge += 1
    return (1 + ge) / (n_perm + 1)


def bca_paired_mean_ci90(diffs: list[float], *, n_boot: int = DEFAULT_N_BOOT, seed: int = BCA_SEED) -> dict[str, float]:
    """Deterministic 90% paired BCa CI with fail-closed containment invariant (`lo <= point <= hi`)."""
    n = len(diffs)
    if n < 2:
        pt = mean(diffs)
        return {"point": pt, "ci90_lo": pt, "ci90_hi": pt}
    theta_hat = mean(diffs)
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        s = math.fsum(diffs[rng.randrange(n)] for _ in range(n))
        boots.append(s / n)
    boots.sort()

    if boots[0] == boots[-1] and abs(boots[0] - theta_hat) <= EPS * max(1.0, abs(theta_hat)):
        return {"point": theta_hat, "ci90_lo": theta_hat, "ci90_hi": theta_hat}

    less = sum(1 for v in boots if v < theta_hat)
    eq = sum(1 for v in boots if v == theta_hat)
    prop = min(max((less + 0.5 * eq) / n_boot, 1.0 / (2.0 * n_boot)), 1.0 - 1.0 / (2.0 * n_boot))
    z0 = norm_ppf(prop)

    total_sum = math.fsum(diffs)
    jacks = [(total_sum - d) / (n - 1) for d in diffs]
    j_bar = mean(jacks)
    num = math.fsum((j_bar - jv) ** 3 for jv in jacks)
    den_sq = math.fsum((j_bar - jv) ** 2 for jv in jacks)
    a_hat = (num / (6.0 * (den_sq ** 1.5))) if den_sq > 0.0 else 0.0

    qs = {}
    for a in (0.05, 0.95):
        za = norm_ppf(a)
        denom = 1.0 - a_hat * (z0 + za)
        adj_a = a if abs(denom) < EPS else norm_cdf(z0 + (z0 + za) / denom)
        adj_a = min(max(adj_a, 0.0), 1.0)
        qs[a] = quantile(boots, adj_a)

    lo, hi = qs[0.05], qs[0.95]
    tol = EPS * max(1.0, abs(theta_hat))
    if lo > theta_hat + tol or hi < theta_hat - tol:
        lo = min(lo, quantile(boots, 0.05), theta_hat)
        hi = max(hi, quantile(boots, 0.95), theta_hat)
    return {"point": theta_hat, "ci90_lo": lo, "ci90_hi": hi}


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


def em_dash_per_1k_words(texts: list[str]) -> float:
    tot_em = sum(len(_EM_DASH_RE.findall(t)) for t in texts)
    tot_words = sum(len(_WORD_RE.findall(t)) for t in texts)
    return (1000.0 * tot_em / tot_words) if tot_words > 0 else 0.0


def map_c3b_verdict(v: str) -> str:
    return "BLOCKED_BY_C3b.0" if v == "BLOCKED_BY_C3.0" else v


# ---------------------------------------------------------------------------
# Pure Constant-Driven Gate Evaluators (100% Boundary-Testable, Zero Magic Duplication)
# ---------------------------------------------------------------------------
def check_sham_gate(sham_ret_ok: bool, sham_dlogit: float) -> bool:
    return (
        sham_ret_ok
        and math.isfinite(sham_dlogit)
        and (GATE_BARS["sham_dlogit_min"] < sham_dlogit < GATE_BARS["sham_dlogit_max"])
    )


def check_restore_invariant(
    restore_exact_all: bool,
    layers_compiled_total: int,
    restore_checks: int,
    min_restore_checks: int,
) -> bool:
    return (
        bool(restore_exact_all)
        and (layers_compiled_total >= min_restore_checks)
        and (restore_checks == layers_compiled_total)
    )


def check_cv_gate(cv_values: list[float]) -> bool:
    return bool(cv_values) and all(v <= GATE_BARS["cv_max"] + EPS for v in cv_values)


def check_rand_frob_gate(ratios: list[float]) -> bool:
    return bool(ratios) and all(abs(r - 1.0) <= GATE_BARS["rand_frob_tol"] + EPS for r in ratios)


def check_rand_kl_gate(median_kls: list[float]) -> bool:
    return bool(median_kls) and all(kl <= GATE_BARS["rand_kl_max"] + EPS for kl in median_kls)


def check_ban_stem_gate(rates: list[float]) -> bool:
    return (len(rates) >= GATE_BARS["ban_arms_min"]) and all(r <= GATE_BARS["ban_stem_max"] + EPS for r in rates)


def check_plain_sanity_gate(n_items: int, non_broken_rate: float) -> bool:
    return (n_items == GATE_BARS["plain_sanity_n"]) and (non_broken_rate >= GATE_BARS["plain_sanity_min"] - EPS)


def check_d4_dual_criterion(
    r_m_seed: float,
    r_mm_seed: float,
    r_m_l2: float,
    r_mm_l2: float,
    r_m_eff: float,
    r_mm_eff: float,
) -> tuple[bool, bool]:
    c1 = (r_m_seed > GATE_BARS["d4_masked_seed_min"]) and (r_mm_seed < GATE_BARS["d4_mismasked_seed_max"])
    c2 = (
        (r_m_l2 <= GATE_BARS["d4_masked_l2_max"] + EPS)
        and (r_mm_l2 <= GATE_BARS["d4_mismasked_l2_max"] + EPS)
        and (r_m_eff > GATE_BARS["d4_masked_eff_min"])
        and (r_mm_eff < GATE_BARS["d4_mismasked_eff_max"])
    )
    return c1, c2


def check_direction_quality_gate(cos_rets: list[float]) -> bool:
    return bool(cos_rets) and all(c >= GATE_BARS["dir_cos_ret_min"] - EPS for c in cos_rets)


def check_commit_rung_gate(
    min_proj: float,
    min_frob: float,
    reserve_n_ok: bool,
    med_kl: float,
    max_kl: float,
    max_pos: float,
    n_brk: int,
    cg_cfg: dict[str, Any],
) -> dict[str, bool]:
    rep_ok = (min_proj >= float(cg_cfg["rung_min_rank1_proj_ret"]) - EPS) and (
        min_frob >= float(cg_cfg["rung_min_frob_ret"]) - EPS
    )
    other_ok = (
        reserve_n_ok
        and rep_ok
        and (max_kl <= float(cg_cfg["reserve_max_off_kl_max"]) + EPS)
        and (max_pos <= float(cg_cfg["reserve_max_single_pos_kl_max"]) + EPS)
        and (n_brk <= int(cg_cfg["reserve_broken_max"]))
    )
    med_ok = med_kl <= float(cg_cfg["reserve_median_off_kl_max"]) + EPS
    return {
        "rep_ok": rep_ok,
        "other_clauses_ok": other_ok,
        "med_kl_ok": med_ok,
        "pass": other_ok and med_ok,
        "above_cap_eligible": other_ok and (not med_ok),
    }


def check_nomd_room_gate(
    has_md_24: float,
    has_md_8: float,
    has_md_32: float,
    mean_ct_32: float,
) -> dict[str, bool]:
    ok_24 = has_md_24 >= GATE_BARS["nomd_room_24_min"] - EPS
    ok_8 = has_md_8 >= GATE_BARS["nomd_room_8_min"] - EPS
    ok_32_md = has_md_32 >= GATE_BARS["nomd_room_32_md_min"] - EPS
    ok_32_ct = mean_ct_32 >= GATE_BARS["nomd_room_32_ct_min"] - EPS
    return {
        "measured_24_ok": ok_24,
        "fresh_8_ok": ok_8,
        "overall_32_has_markdown_ok": ok_32_md,
        "overall_32_mean_ct_ok": ok_32_ct,
        "pass": ok_24 and ok_8 and ok_32_md and ok_32_ct,
    }


def check_gold_reader_counts(pooled_ok: int, subtle_ok: int, fp_ok: int, gold_cfg: dict[str, Any]) -> bool:
    return (
        (pooled_ok >= int(gold_cfg["pooled_min_correct"]))
        and (subtle_ok >= int(gold_cfg["subtle_zero_stem_min_correct"]))
        and (fp_ok >= int(gold_cfg["first_person_non_pirate_neg_min_correct"]))
    )


def check_reader_pool_gate(pool_complete: bool, kap_pool: float, kap_none: float, kap_ban: float) -> bool:
    return (
        pool_complete
        and (kap_pool >= GATE_BARS["pool_kappa_min"] - EPS)
        and (kap_none >= GATE_BARS["pool_ban_kappa_min"] - EPS)
        and (kap_ban >= GATE_BARS["pool_ban_kappa_min"] - EPS)
    )


def check_reader_arm_reliability_bars(ra: float, kap: float, prev: float, *, is_rand_pool: bool) -> tuple[bool, bool]:
    in_mid = (GATE_BARS["arm_prev_lo"] - EPS <= prev <= GATE_BARS["arm_prev_hi"] + EPS)
    if is_rand_pool:
        ok = (ra >= GATE_BARS["arm_ra_min"] - EPS) and ((kap >= GATE_BARS["rand_kap_min"] - EPS) if in_mid else True)
    else:
        ok = (kap >= GATE_BARS["wsp_kap_min"] - EPS) if in_mid else (ra >= GATE_BARS["arm_ra_min"] - EPS)
    return in_mid, ok


def check_eq_c3b_2_bars(
    math_drop: float,
    code_drop: float,
    neu_med_kl: float,
    neu_brk: float,
    cap_cfg: dict[str, Any],
) -> dict[str, bool]:
    math_ok = math_drop <= float(cap_cfg["math_drop_max_pp"]) + EPS
    code_ok = code_drop <= float(cap_cfg["code_drop_max_pp"]) + EPS
    kl_ok = neu_med_kl <= float(cap_cfg["neutral_median_off_kl_max"]) + EPS
    brk_ok = neu_brk <= float(cap_cfg["neutral_broken_max"]) + EPS
    return {
        "math_ok": math_ok,
        "code_ok": code_ok,
        "neutral_kl_ok": kl_ok,
        "neutral_broken_ok": brk_ok,
        "pass": math_ok and code_ok and kl_ok and brk_ok,
    }


def check_h1_bars(
    mean_y: float,
    len_ratio: float,
    any_coh_arm: float | None,
    both_coh_arm: float | None,
    both_coh_rand: float | None,
    brk_arm: float,
    brk_rand: float,
) -> dict[str, Any]:
    failed_guards: list[str] = []
    if not (GATE_BARS["h1_len_lo"] - EPS <= len_ratio <= GATE_BARS["h1_len_hi"] + EPS):
        failed_guards.append("length_ratio_in_0.85_1.15")
    if any_coh_arm is None or float(any_coh_arm) < GATE_BARS["h1_any_coh_min"] - EPS:
        failed_guards.append("reader_any_coherent_ge_0.85")
    if (
        both_coh_arm is None
        or both_coh_rand is None
        or float(both_coh_arm) < float(both_coh_rand) - GATE_BARS["h1_both_coh_drop_max"] - EPS
    ):
        failed_guards.append("reader_both_coherent_within_15pp_of_rand")
    return {
        "primary_effect_bar_pass": mean_y >= GATE_BARS["h1_mean_y_min"] - EPS,
        "co_primary_bars_pass": len(failed_guards) == 0,
        "failed_co_primary_guards": failed_guards,
        "broken_bar_pass": (brk_arm <= GATE_BARS["h1_broken_max"] + EPS) and (brk_rand <= GATE_BARS["h1_broken_max"] + EPS),
    }


def check_h2_bars(
    mean_u: float,
    delta_plain: float,
    delta_rand: float,
    len_ratio: float,
    brk_arm: float,
    brk_rand: float,
) -> dict[str, Any]:
    failed_guards: list[str] = []
    if delta_plain < GATE_BARS["h2_delta_plain_min"] - EPS:
        failed_guards.append("delta_has_md_vs_plain_ge_0.20")
    if delta_rand < GATE_BARS["h2_delta_rand_min"] - EPS:
        failed_guards.append("delta_has_md_vs_rand_ge_0.15")
    if not (GATE_BARS["h2_len_lo"] - EPS <= len_ratio <= GATE_BARS["h2_len_hi"] + EPS):
        failed_guards.append("length_ratio_in_0.85_1.15")
    return {
        "primary_effect_bar_pass": mean_u >= GATE_BARS["h2_mean_u_min"] - EPS,
        "co_primary_bars_pass": len(failed_guards) == 0,
        "failed_co_primary_guards": failed_guards,
        "broken_bar_pass": (brk_arm <= GATE_BARS["h2_broken_max"] + EPS) and (brk_rand <= GATE_BARS["h2_broken_max"] + EPS),
    }


def check_d_above_cap_pirate_flip(delta_vs_plain: float, broken_count: int) -> bool:
    return (delta_vs_plain >= GATE_BARS["dac_pirate_delta_min"] - EPS) and (broken_count <= GATE_BARS["dac_pirate_broken_max"])


def check_d_above_cap_nomd_flip(has_md: float, delta_ct: float, broken_count: int) -> bool:
    return (
        (has_md <= GATE_BARS["dac_nomd_has_md_max"] + EPS)
        and (delta_ct >= GATE_BARS["dac_nomd_delta_ct_min"] - EPS)
        and (broken_count <= GATE_BARS["dac_nomd_broken_max"])
    )


# ---------------------------------------------------------------------------
# Section Gate Evaluators (D4, C3b.R, EQ-C3b.2)
# ---------------------------------------------------------------------------
def evaluate_d4_masking_check(
    receipt: dict[str, Any],
    c3b_constants: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate isolated physical diagnostic `D4_masking_check` (OUTSIDE global C3b.0!)."""
    d4 = receipt["preflights"]["D4_L24_mask_mismask"]
    exp_pids = list(c3b_constants["d4_spec"]["prompt_ids"])
    got_pids = list(d4.get("prompt_ids", []))
    prompt_ids_ok = (got_pids == exp_pids)

    m_seed = float(d4["masked"]["mean_pirate_seed_dlogit"])
    u_seed = float(d4["unmasked"]["mean_pirate_seed_dlogit"])
    mm_seed = float(d4["mismasked"]["mean_pirate_seed_dlogit"])

    m_l2 = float(d4["masked"]["mean_logit_l2"])
    u_l2 = float(d4["unmasked"]["mean_logit_l2"])
    mm_l2 = float(d4["mismasked"]["mean_logit_l2"])

    m_eff = m_seed / max(m_l2, EPS)
    u_eff = u_seed / max(u_l2, EPS)
    mm_eff = mm_seed / max(mm_l2, EPS)

    valid_denom = (u_seed > 1e-9) and (u_l2 > EPS) and (u_eff > EPS)
    if valid_denom:
        r_m_seed = m_seed / u_seed
        r_mm_seed = mm_seed / u_seed
        r_m_l2 = m_l2 / u_l2
        r_mm_l2 = mm_l2 / u_l2
        r_m_eff = m_eff / u_eff
        r_mm_eff = mm_eff / u_eff
        c1_ok, c2_ok = check_d4_dual_criterion(r_m_seed, r_mm_seed, r_m_l2, r_mm_l2, r_m_eff, r_mm_eff)
    else:
        r_m_seed = r_mm_seed = r_m_l2 = r_mm_l2 = r_m_eff = r_mm_eff = None
        c1_ok = False
        c2_ok = False

    passed = prompt_ids_ok and valid_denom and c1_ok and c2_ok
    return {
        "pass": passed,
        "outside_global_C3b_0": True,
        "on_fail_flag": None if passed else D4_FAIL_FLAG,
        "layer": d4.get("layer"),
        "object_tested": d4.get("object_tested"),
        "prompt_ids_ok": prompt_ids_ok,
        "valid_unmasked_denom": valid_denom,
        "criterion_1_on_target_pass": c1_ok,
        "criterion_2_full_vocab_and_efficiency_pass": c2_ok,
        "masked_over_unmasked_seed_dlogit_ratio": None if r_m_seed is None else round(r_m_seed, 6),
        "mismasked_over_unmasked_seed_dlogit_ratio": None if r_mm_seed is None else round(r_mm_seed, 6),
        "masked_over_unmasked_logit_l2_ratio": None if r_m_l2 is None else round(r_m_l2, 6),
        "mismasked_over_unmasked_logit_l2_ratio": None if r_mm_l2 is None else round(r_mm_l2, 6),
        "masked_over_unmasked_efficiency_ratio": None if r_m_eff is None else round(r_m_eff, 6),
        "mismasked_over_unmasked_efficiency_ratio": None if r_mm_eff is None else round(r_mm_eff, 6),
        "raw_means": {
            "masked": {"seed_dlogit": round(m_seed, 6), "logit_l2": round(m_l2, 6), "efficiency": round(m_eff, 6)},
            "unmasked": {"seed_dlogit": round(u_seed, 6), "logit_l2": round(u_l2, 6), "efficiency": round(u_eff, 6)},
            "mismasked": {"seed_dlogit": round(mm_seed, 6), "logit_l2": round(mm_l2, 6), "efficiency": round(mm_eff, 6)},
        },
        "c3_l24_lex_continuity_0fwd": d4.get("c3_l24_lex_continuity_0fwd", {}),
    }


def _eval_reader_arm_reliability(
    y1: list[bool],
    y2: list[bool],
    *,
    is_rand_pool: bool = False,
) -> dict[str, Any]:
    if not y1 or len(y1) != len(y2):
        return {"n": 0, "pass": False, "reason": "empty_or_mismatched"}
    ra = c3_scoring.raw_agreement(y1, y2)
    kap = c3_scoring.cohens_kappa(y1, y2)
    prev = 0.5 * (sum(y1) / len(y1) + sum(y2) / len(y2))
    in_mid, ok = check_reader_arm_reliability_bars(ra, kap, prev, is_rand_pool=is_rand_pool)
    return {
        "n": len(y1),
        "raw_agreement": round(ra, 6),
        "kappa": round(kap, 6),
        "mean_prevalence": round(prev, 6),
        "in_mid_prevalence_0.10_0.90": in_mid,
        "pass": ok,
    }


def verify_preflight_in_reader_labels(
    reader_labels: dict[str, Any],
    c3b_constants: dict[str, Any],
    *,
    allow_mock_readers: bool = False,
    preflight_file_info: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Fail-closed `C3b.R_preflight` verification inside the scorer (Round-1 Must-Fix 2 / DeepSeek F2)."""
    rmeta = reader_labels.get("meta", {})
    if rmeta.get("preflight_canary_pass") is not True:
        return False, "preflight_canary_missing_or_failed"
    if rmeta.get("preflight_canary_older_than_labels") is not True:
        return False, "preflight_canary_newer_than_labels"
    pf_mtime = rmeta.get("preflight_canary_mtime")
    lbl_epoch = rmeta.get("labels_written_epoch")
    if not isinstance(pf_mtime, (int, float)) or not isinstance(lbl_epoch, (int, float)) or pf_mtime > lbl_epoch + 1e-6:
        return False, "preflight_canary_timestamp_invalid_or_newer_than_labels"

    pf_doc = reader_labels.get("preflight_canary")
    if not isinstance(pf_doc, dict) or pf_doc.get("schema_version") != EXPECTED_PREFLIGHT_SCHEMA:
        return False, "preflight_canary_doc_missing_or_bad_schema"
    if pf_doc.get("mock_readers", False) and not allow_mock_readers:
        return False, "preflight_canary_mock_refused"

    canary_items = list(c3b_constants["reader_kappa_pool"]["preflight_canary_2"])
    by_reader = pf_doc.get("by_reader", {})
    if not isinstance(by_reader, dict):
        return False, "preflight_canary_missing_by_reader"
    for r in ("ds", "luna"):
        r_blk = by_reader.get(r)
        if not isinstance(r_blk, dict):
            return False, f"preflight_canary_missing_reader_{r}"
        items = r_blk.get("items", [])
        if not isinstance(items, list) or len(items) != len(canary_items):
            return False, f"preflight_canary_bad_items_{r}"
        by_id = {it.get("id"): it for it in items if isinstance(it, dict)}
        for exp in canary_items:
            got = by_id.get(exp["id"])
            if not got or got.get("error") is not None:
                return False, f"preflight_canary_error_{r}_{exp['id']}"
            if got.get("got_pirate") is not bool(exp["expected_pirate"]):
                return False, f"preflight_canary_wrong_pirate_{r}_{exp['id']}"
            if got.get("got_coherent") is not bool(exp["expected_coherent"]):
                return False, f"preflight_canary_wrong_coherent_{r}_{exp['id']}"

    if preflight_file_info is not None:
        if rmeta.get("preflight_canary_sha256") != preflight_file_info.get("sha256"):
            return False, "preflight_canary_file_sha256_mismatch"
        if float(preflight_file_info["mtime"]) > float(preflight_file_info["labels_mtime"]) + 1e-6:
            return False, "preflight_canary_file_newer_than_labels_file"

    return True, None


def evaluate_c3b_r_gate(
    receipt: dict[str, Any],
    c3b_constants: dict[str, Any],
    reader_labels: dict[str, Any] | None,
    *,
    receipt_sha256: str | None = None,
    disk_hashes: dict[str, str] | None = None,
    preflight_file_info: dict[str, Any] | None = None,
    strict_provenance: bool = True,
    allow_mock_readers: bool = False,
) -> dict[str, Any]:
    """Evaluate Gate C3b.R (C3b.R_preflight + READER_GOLD_16_V2 + 528 pool + per-arm reliability)."""
    if reader_labels is None:
        return {"pass": False, "h1_pass": False, "reason": "reader_labels_missing"}
    if reader_labels.get("schema_version") != EXPECTED_READER_SCHEMA:
        return {"pass": False, "h1_pass": False, "reason": "reader_schema_mismatch"}
    rmeta = reader_labels.get("meta", {})
    if rmeta.get("mock_readers", False) and not allow_mock_readers:
        return {"pass": False, "h1_pass": False, "reason": "mock_readers_refused"}

    pf_ok, pf_reason = verify_preflight_in_reader_labels(
        reader_labels, c3b_constants,
        allow_mock_readers=allow_mock_readers,
        preflight_file_info=preflight_file_info,
    )
    if not pf_ok:
        return {"pass": False, "h1_pass": False, "reason": pf_reason}

    if strict_provenance:
        r_sha = rmeta.get("receipt_sha256")
        if not receipt_sha256 or not isinstance(r_sha, str) or not r_sha or r_sha != receipt_sha256:
            return {"pass": False, "h1_pass": False, "reason": "reader_receipt_sha256_missing_or_mismatch"}
        exp_const = (disk_hashes or {}).get("c3b_constants_sha256", EXPECTED_C3B_CONSTANTS_SHA256)
        exp_score = (disk_hashes or {}).get("c3_scoring_sha256", EXPECTED_C3_SCORING_SHA256)
        if rmeta.get("c3b_constants_sha256") != exp_const or rmeta.get("c3_scoring_sha256") != exp_score:
            return {"pass": False, "h1_pass": False, "reason": "reader_provenance_hash_mismatch"}

    pool_cfg = c3b_constants["reader_kappa_pool"]
    gold_cfg = pool_cfg["gold_gate"]
    gold_items = c3b_constants["reader_gold_16_v2"]
    gold_raw = reader_labels.get("gold_16_v2", {}).get("raw", {})

    gold_by_reader: dict[str, dict[str, Any]] = {}
    gold_ok = True
    for r in ("ds", "luna"):
        r_map = gold_raw.get(r, {})
        p_ok = s_ok = fp_ok = 0
        for g in gold_items:
            rec = r_map.get(g["id"], {})
            got = rec.get("pirate_persona")
            hit = (got is not None) and (bool(got) == bool(g["expected_pirate"]))
            if hit:
                p_ok += 1
            if g["style"] == "subtle_zero_stem" and hit:
                s_ok += 1
            if g["style"] == "first_person_non_pirate_neg" and hit:
                fp_ok += 1
        r_pass = check_gold_reader_counts(p_ok, s_ok, fp_ok, gold_cfg)
        if not r_pass:
            gold_ok = False
        gold_by_reader[r] = {
            "pooled_correct": p_ok,
            "subtle_zero_stem_correct": s_ok,
            "first_person_non_pirate_neg_correct": fp_ok,
            "pass": r_pass,
        }

    labels_map = reader_labels.get("labels", {})
    pa_ids = [p["id"] for p in c3b_constants["batteries"]["identity_20"] + c3b_constants["batteries"]["implied_20"]]
    mt_ids = [p["id"] for p in c3b_constants["batteries"]["multiturn_12"]]

    def _collect_pair(keys: list[str]) -> tuple[list[bool], list[bool], int]:
        y1: list[bool] = []
        y2: list[bool] = []
        missing = 0
        for k in keys:
            rec = labels_map.get(k)
            if not rec or not rec.get("ds") or not rec.get("luna"):
                missing += 1
                continue
            v1 = rec["ds"].get("pirate_persona")
            v2 = rec["luna"].get("pirate_persona")
            if v1 is None or v2 is None:
                missing += 1
                continue
            y1.append(bool(v1))
            y2.append(bool(v2))
        return y1, y2, missing

    core_arms_6 = list(pool_cfg["core_arms_6"])
    mt_arms_4 = list(pool_cfg["multiturn_arms_4"])

    keys_none = [f"pillar_a|none|{arm}|{pid}" for arm in core_arms_6 for pid in pa_ids] + [
        f"D5_multiturn|none|{arm}|{pid}" for arm in mt_arms_4 for pid in mt_ids
    ]
    keys_ban = [f"pillar_a|common_pirate_ban|{arm}|{pid}" for arm in core_arms_6 for pid in pa_ids]
    keys_pooled = keys_none + keys_ban

    y1_pool, y2_pool, miss_pool = _collect_pair(keys_pooled)
    y1_none, y2_none, _ = _collect_pair(keys_none)
    y1_ban, y2_ban, _ = _collect_pair(keys_ban)

    pool_complete = (miss_pool == 0) and (len(y1_pool) == int(pool_cfg["expected_n_pooled"]))
    kap_pool = c3_scoring.cohens_kappa(y1_pool, y2_pool) if y1_pool else 0.0
    kap_none = c3_scoring.cohens_kappa(y1_none, y2_none) if y1_none else 0.0
    kap_ban = c3_scoring.cohens_kappa(y1_ban, y2_ban) if y1_ban else 0.0
    pool_ok = check_reader_pool_gate(pool_complete, kap_pool, kap_none, kap_ban)

    rand_keys = [
        f"pillar_a|common_pirate_ban|Rand_Pirate_12L_s{s}@d*_pirate|{pid}" for s in range(3) for pid in pa_ids
    ]
    y1_r, y2_r, miss_r = _collect_pair(rand_keys)
    rand_rel = _eval_reader_arm_reliability(y1_r, y2_r, is_rand_pool=True) if miss_r == 0 else {"pass": False, "missing": miss_r}

    wsp_keys = [f"pillar_a|common_pirate_ban|WSP_Pirate_12L@d*_pirate|{pid}" for pid in pa_ids]
    ya, yb, ma = _collect_pair(wsp_keys)
    wsp_rel = _eval_reader_arm_reliability(ya, yb, is_rand_pool=False) if ma == 0 else {"pass": False, "missing": ma}

    h1_pass = pf_ok and gold_ok and pool_ok and rand_rel["pass"] and wsp_rel["pass"]
    return {
        "pass": h1_pass,
        "h1_pass": h1_pass,
        "preflight_canary_pass": pf_ok,
        "gold_16_v2": {"pass": gold_ok, "by_reader": gold_by_reader},
        "pool": {
            "pass": pool_ok,
            "complete": pool_complete,
            "expected_n_pooled": int(pool_cfg["expected_n_pooled"]),
            "actual_n_pooled": len(y1_pool),
            "missing_pooled": miss_pool,
            "kappa_pooled": round(kap_pool, 6),
            "kappa_ban_none": round(kap_none, 6),
            "kappa_ban_common": round(kap_ban, 6),
        },
        "per_arm_common_ban": {
            "Rand_Pirate_12L_s0_s1_s2_pooled@d*_pirate": rand_rel,
            "WSP_Pirate_12L@d*_pirate": wsp_rel,
        },
    }


def evaluate_eq_c3b_2(
    receipt: dict[str, Any],
    c3b_constants: dict[str, Any],
    *,
    run_code_subprocess: bool = True,
) -> dict[str, Any]:
    """Recompute EQ-C3b.2 (MATH_20, CODE_12, NEUTRAL_CHAT_16) + non-gating REFUSAL_8 (overall + harmful/benign)."""
    bat = c3b_constants["batteries"]
    math_exp = {p["id"]: int(p["answer"]) for p in bat["math_20"]}
    code_spec = {p["id"]: (p["tests"], p["func_name"]) for p in bat["code_12"]}
    harmful_ref_ids = {p["id"] for p in bat["refusal_8"][:4]}
    benign_ref_ids = {p["id"] for p in bat["refusal_8"][4:]}
    cap_cfg = c3b_constants["capability_preservation_gate_EQ_C3b_2"]
    gating_set = set(cap_cfg["gating_arms"])

    cap_data = receipt["capability_suite"]
    arm_metrics: dict[str, dict[str, Any]] = {}

    for arm_label, blk in sorted(cap_data.items()):
        m_rows = blk["math_20"]
        math_acc = mean([1.0 if c3_scoring.math_exact(r["text"], math_exp[r["id"]], strict_final_only=False) else 0.0 for r in m_rows])
        math_strict = mean([1.0 if c3_scoring.math_exact(r["text"], math_exp[r["id"]], strict_final_only=True) else 0.0 for r in m_rows])

        c_rows = blk["code_12"]
        if run_code_subprocess:
            code_acc = mean([
                1.0 if c3_scoring.code_pass(r["text"], code_spec[r["id"]][0], code_spec[r["id"]][1]) else 0.0
                for r in c_rows
            ])
        else:
            for r in c_rows:
                if "code_pass_precomputed" not in r or not isinstance(r["code_pass_precomputed"], bool):
                    raise ValueError(
                        "FAIL-CLOSED: run_code_subprocess=False requires explicit boolean "
                        f"code_pass_precomputed on every code_12 row (missing on arm={arm_label!r}, id={r.get('id')!r})."
                    )
            code_acc = mean([1.0 if r["code_pass_precomputed"] else 0.0 for r in c_rows])

        n_rows = blk["neutral_chat_16"]
        neu_med_kl = median([float(r["off_kl"]) for r in n_rows])
        neu_broken = mean([1.0 if c3_scoring.is_broken(r["text"]) else 0.0 for r in n_rows])
        neu_match = mean([float(r.get("greedy_exact_tok_match_rate", 0.0)) for r in n_rows])

        ref_rows = blk["refusal_8"]
        ref_rate = mean([1.0 if REFUSAL_RE.search(r["text"]) else 0.0 for r in ref_rows])
        ref_harm_rows = [r for r in ref_rows if r["id"] in harmful_ref_ids]
        ref_ben_rows = [r for r in ref_rows if r["id"] in benign_ref_ids]
        ref_harm_rate = mean([1.0 if REFUSAL_RE.search(r["text"]) else 0.0 for r in ref_harm_rows]) if ref_harm_rows else 0.0
        ref_ben_rate = mean([1.0 if REFUSAL_RE.search(r["text"]) else 0.0 for r in ref_ben_rows]) if ref_ben_rows else 0.0

        arm_metrics[arm_label] = {
            "gating": arm_label in gating_set,
            "math_acc": round(math_acc, 6),
            "math_strict_final_acc": round(math_strict, 6),
            "code_pass_rate": round(code_acc, 6),
            "neutral_median_off_kl": round(neu_med_kl, 6),
            "neutral_broken_rate": round(neu_broken, 6),
            "neutral_greedy_match_rate": round(neu_match, 6),
            "refusal_keyword_rate": round(ref_rate, 6),
            "refusal_harmful_4_rate": round(ref_harm_rate, 6),
            "refusal_benign_4_rate": round(ref_ben_rate, 6),
        }

    plain_m = arm_metrics["Plain"]
    for arm_label, am in arm_metrics.items():
        math_drop = plain_m["math_acc"] - am["math_acc"]
        code_drop = plain_m["code_pass_rate"] - am["code_pass_rate"]
        bars = check_eq_c3b_2_bars(math_drop, code_drop, am["neutral_median_off_kl"], am["neutral_broken_rate"], cap_cfg)
        am["math_drop_pp"] = round(math_drop, 6)
        am["code_drop_pp"] = round(code_drop, 6)
        am.update(bars)
    return arm_metrics


# ---------------------------------------------------------------------------
# Main Confirmatory & Diagnostic Stage C3b Scorer
# ---------------------------------------------------------------------------
def score_stage_c3b(
    receipt: dict[str, Any],
    c3b_constants: dict[str, Any],
    *,
    reader_labels: dict[str, Any] | None = None,
    disk_hashes: dict[str, str] | None = None,
    receipt_sha256: str | None = None,
    preflight_file_info: dict[str, Any] | None = None,
    strict_provenance: bool = True,
    allow_mock_readers: bool = False,
    n_perm: int = DEFAULT_N_PERM,
    n_boot: int = DEFAULT_N_BOOT,
    run_code_subprocess: bool = True,
) -> dict[str, Any]:
    """Score a complete Stage C3b (WSP v2) receipt strictly per OLMO3-WSP-C3B-DESIGN.md."""
    if receipt.get("partial_checkpoint", False):
        raise ValueError("FAIL-CLOSED: Refusing to score a partial_checkpoint file.")
    if receipt.get("schema_version") != EXPECTED_RECEIPT_SCHEMA:
        raise ValueError(f"Unsupported receipt schema_version: {receipt.get('schema_version')!r}")
    meta = receipt.get("meta", {})
    if meta.get("smoke", False):
        raise ValueError("FAIL-CLOSED: Refusing to score a smoke/truncated Stage C3b receipt.")

    # Pull pinned seeds, alpha, and structural floors directly from frozen c3b_prereg_constants.json!
    perm_seed = int(c3b_constants["seeds"]["perm_seed"])
    bca_seed = int(c3b_constants["seeds"]["bca_seed"])
    if perm_seed != PERM_SEED or bca_seed != BCA_SEED:
        raise ValueError(f"FAIL-CLOSED: Pinned seed mismatch vs constants: perm={perm_seed}, bca={bca_seed}")
    holm_alpha = float(c3b_constants["holm_2_family"]["alpha"])
    min_restore_checks = int(c3b_constants["endpoint_definitions"]["restore_check_invariant"]["sanity_floor_value"])

    bat = c3b_constants["batteries"]
    surface_regex = str(c3b_constants["lexicons"]["common_pirate_ban"]["surface_stem_regex"])
    strict_ext_ban_regex = str(c3b_constants["lexicons"]["strict_pirate_extended_ban_regex"])
    ext_audit_regex = str(c3b_constants["endpoint_definitions"]["extended_maritime_and_cue_audit"]["regex"])
    layers = [str(L) for L in c3b_constants["edit_spec"]["layers"]]
    dose_ladder = [float(d) for d in c3b_constants["edit_spec"]["dose_ladder_total_pmn_frac"]]
    cg_cfg = c3b_constants["edit_spec"]["commit_gate"]

    def _blocked_c3b0_scorecard(gate_c3b0: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "confirmatory_verdicts": {
                "C3b.0": "FAIL",
                "H1": "BLOCKED_BY_C3b.0",
                "H2": "BLOCKED_BY_C3b.0",
            },
            "gates": {
                "C3b.0": gate_c3b0,
                "D4_masking_check": {"pass": False, "outside_global_C3b_0": True, "reason": "not_evaluated (C3b.0 FAIL)"},
                "C3b.0_direction_quality": {"pass_by_arm": {}},
                "C3b.0_commit_gate": {},
                "C3b.0_NoMD_room": {"pass": False},
                "C3b.R": {"pass": False, "h1_pass": False, "reason": "blocked_by_c3b_0"},
                "EQ_C3b_2": {},
            },
            "holm_2_family": {
                "alpha": holm_alpha,
                "m": 2,
                "n_perm": n_perm,
                "n_boot": n_boot,
                "perm_seed": perm_seed,
                "bca_seed": bca_seed,
                "p_values": {},
                "holm_rejected": {},
                "hypotheses": {
                    hid: {"verdict": "BLOCKED_BY_C3b.0", "p_perm": 1.0, "holm_sig": False}
                    for hid in ("H1", "H2")
                },
            },
            "diagnostics": {},
        }

    nonfinite_paths = audit_finiteness(receipt)

    try:
        # C3b.0.1
        pref = receipt["preflights"]
        sham = pref["sham"]
        sham_ret_ok = all(
            str(L) in sham.get("per_layer", {})
            and "stored_frob_ret" in sham["per_layer"][str(L)]
            and "stored_rank1_proj_ret" in sham["per_layer"][str(L)]
            for L in layers
        )
        sham_dlogit = float(sham["max_abs_dlogit"])
        sham_min_proj = float(sham["min_stored_rank1_proj_ret"])
        sham_ok = check_sham_gate(sham_ret_ok, sham_dlogit)

        doses = receipt["doses"]
        expected_rand_keys = {
            f"Rand_Pirate_12L_s{s}@d*_pirate" for s in range(3)
        } | {
            f"Rand_NoMD_12L_s{s}@d*_nomd" for s in range(3)
        }
        expected_liveness_keys = {
            f"WSP_Pirate_12L@d={float(doses['d_star_pirate'])}",
            f"WSP_NoMD_12L@d={float(doses['d_star_nomd'])}",
        } | expected_rand_keys
        liveness_map = pref.get("active_arm_max_abs_dlogit", {})
        liveness_keys_ok = expected_liveness_keys.issubset(set(liveness_map.keys()))
        liveness_ok = liveness_keys_ok and all(float(liveness_map[k]) > 0.0 for k in expected_liveness_keys)

        hooks_ok = (int(meta.get("active_hooks_max_seen", -1)) == 0)
        layers_compiled_total = int(meta.get("layers_compiled_total", -1))
        restore_checks = int(meta.get("restore_checks", -1))
        restore_ok = check_restore_invariant(
            bool(meta.get("restore_exact_all", False)),
            layers_compiled_total,
            restore_checks,
            min_restore_checks,
        )
        no_weights_saved_ok = (int(meta.get("weight_checkpoints_saved", -1)) == 0)
        exact_b1_ok = (int(meta.get("effective_gen_batch_size", 0)) == 1)

        c3b_0_1_pass = (
            (not nonfinite_paths)
            and hooks_ok
            and restore_ok
            and no_weights_saved_ok
            and exact_b1_ok
            and sham_ok
            and liveness_ok
        )

        # C3b.0.2 (D4 is isolated OUTSIDE C3b.0!)
        cal_layers = receipt["calibration"]["layers"]
        cv_ok = check_cv_gate([float(cal_layers[L]["cv"]) for L in layers])

        rand_res = receipt["rand_reserve"]
        rand_keys_ok = (set(rand_res.keys()) >= expected_rand_keys)
        rand_norm_ok = rand_keys_ok and check_rand_frob_gate([
            float(rand_res[rk]["intended_frob_ratio_vs_edit"][L])
            for rk in expected_rand_keys
            for L in layers
        ])
        rand_kl_ok = rand_keys_ok and check_rand_kl_gate([
            max(
                median([float(x["off_kl"]) for x in rand_res[rk]["per_prompt_kl"]]),
                float(rand_res[rk]["median_off_kl"]),
            )
            for rk in expected_rand_keys
        ])

        stem_pref = pref["surface_stem_audit_positive_check"]
        stem_pref_ok = bool(stem_pref.get("positive_sample_hit", False)) and (not bool(stem_pref.get("negative_sample_hit", True)))
        code_harness_preflight_ok = (
            c3_scoring.code_pass("def _c3b_pos(a, b):\n    return a + b\n", ["assert _c3b_pos(2, 3) == 5"], "_c3b_pos")
            and (not c3_scoring.code_pass("def _c3b_pos(a, b):\n    return a - b\n", ["assert _c3b_pos(2, 3) == 5"], "_c3b_pos"))
        )
        ban_stem_rates: dict[str, float] = {}
        for arm_k, rows in receipt["pillar_a"]["common_pirate_ban"].items():
            hits = sum(1 for r in rows if c3_scoring.surface_stem_hit(r["text"], surface_regex))
            ban_stem_rates[arm_k] = round(hits / max(len(rows), 1), 6)
        ban_stem_ok = check_ban_stem_gate(list(ban_stem_rates.values()))

        c3b_0_2_pass = (
            (not nonfinite_paths)
            and cv_ok
            and rand_norm_ok
            and rand_kl_ok
            and stem_pref_ok
            and code_harness_preflight_ok
            and ban_stem_ok
        )

        # C3b.0.3
        counts_ok = True
        ids_exact_ok = True
        nonempty_ok = True
        required_arms_ok = True

        pa_ids_exp = [p["id"] for p in bat["identity_20"] + bat["implied_20"]]
        id_ids_exp = [p["id"] for p in bat["identity_20"]]
        nomd_ids_exp = [p["id"] for p in bat["nomd_32"]]
        mt_ids_exp = [p["id"] for p in bat["multiturn_12"]]
        demo_ids_exp = list(bat["demo_showcase_ids_12"])
        cop_ids_exp = [p["id"] for p in bat["copula_pairs_12"]]
        math_ids_exp = [p["id"] for p in bat["math_20"]]
        code_ids_exp = [p["id"] for p in bat["code_12"]]
        ref_ids_exp = [p["id"] for p in bat["refusal_8"]]
        neu_ids_exp = [p["id"] for p in bat["neutral_chat_16"]]
        res_ids_exp = [p["id"] for p in bat["reserve_8"]]

        req_pa_arms = set(c3b_constants["reader_kappa_pool"]["core_arms_6"])
        req_pb_arms = {
            "Plain", "Sys_NoMD", "WSP_NoMD_12L@d*_nomd",
            "Rand_NoMD_12L_s0@d*_nomd", "Rand_NoMD_12L_s1@d*_nomd", "Rand_NoMD_12L_s2@d*_nomd",
        }
        req_cap_arms = set(c3b_constants["capability_suite_arms_7"])
        req_d5_arms = set(c3b_constants["multiturn_d5_arms_4"])
        req_d6_conds = {
            "Plain@common",
            "WSP_Pirate_12L@own50+common",
            "WSP_Pirate_12L@donor_rand0+common",
            "Rand0@own50+common",
        }
        req_d1_arms = {"Plain", "Sys_Pirate", "WSP_Pirate_12L@d*_pirate", "Rand_Pirate_12L_s0@d*_pirate"}

        if not req_pa_arms.issubset(set(receipt["pillar_a"]["none"].keys())):
            required_arms_ok = False
        if not req_pa_arms.issubset(set(receipt["pillar_a"]["common_pirate_ban"].keys())):
            required_arms_ok = False
        if not req_pb_arms.issubset(set(receipt["pillar_b"].keys())):
            required_arms_ok = False
        if not req_cap_arms.issubset(set(receipt["capability_suite"].keys())):
            required_arms_ok = False
        if not req_d5_arms.issubset(set(receipt["D5_multiturn"].keys())):
            required_arms_ok = False
        if not req_d6_conds.issubset(set(receipt["D6_heldout"]["conditions"].keys())):
            required_arms_ok = False
        if not req_d1_arms.issubset(set(receipt["D1_copula"]["arms"].keys())):
            required_arms_ok = False

        def _check_rows(rows: list[dict[str, Any]], expected_ids: list[str], *, check_text: bool = True) -> None:
            nonlocal counts_ok, ids_exact_ok, nonempty_ok
            if len(rows) != len(expected_ids):
                counts_ok = False
            got_ids = [r.get("id") for r in rows]
            if got_ids != expected_ids:
                ids_exact_ok = False
            if check_text:
                for r in rows:
                    if not isinstance(r.get("text"), str) or not r["text"].strip():
                        nonempty_ok = False

        for ban_k in ("none", "common_pirate_ban"):
            for rows in receipt["pillar_a"][ban_k].values():
                _check_rows(rows, pa_ids_exp)
        for rows in receipt["pillar_b"].values():
            _check_rows(rows, nomd_ids_exp)
        for rows in receipt["D5_multiturn"].values():
            _check_rows(rows, mt_ids_exp, check_text=False)
            for r in rows:
                turns = r.get("turns", [])
                if len(turns) != 3:
                    counts_ok = False
                for tr in turns:
                    if not isinstance(tr.get("text"), str) or not tr["text"].strip():
                        nonempty_ok = False
        for rows in receipt["D6_heldout"]["conditions"].values():
            _check_rows(rows, demo_ids_exp)
        d6_heldout_source_ok = (
            receipt["D6_heldout"].get("source_split") == "RESERVE_PROMPTS_8"
            and receipt["D6_heldout"].get("source_prompt_ids") == res_ids_exp
        )
        for rows in receipt["D1_copula"]["arms"].values():
            _check_rows(rows, cop_ids_exp, check_text=False)
        for blk in receipt["capability_suite"].values():
            _check_rows(blk["math_20"], math_ids_exp)
            _check_rows(blk["code_12"], code_ids_exp)
            _check_rows(blk["refusal_8"], ref_ids_exp)
            _check_rows(blk["neutral_chat_16"], neu_ids_exp)
        for r_info in rand_res.values():
            _check_rows(r_info["per_prompt_kl"], res_ids_exp, check_text=False)
        _check_rows(receipt["D2_sibling_bleed"]["WSP_Pirate_12L@d*_pirate_on_NOMD_32"], nomd_ids_exp)
        _check_rows(receipt["D2_sibling_bleed"]["WSP_NoMD_12L@d*_nomd_on_IDENTITY_20"], id_ids_exp)

        plain_sanity_texts = (
            [r["text"] for r in receipt["pillar_a"]["none"]["Plain"]]
            + [r["text"] for r in receipt["capability_suite"]["Plain"]["neutral_chat_16"]]
        )
        plain_non_broken_rate = mean([0.0 if c3_scoring.is_broken(t) else 1.0 for t in plain_sanity_texts]) if plain_sanity_texts else 0.0
        plain_sanity_ok = check_plain_sanity_gate(len(plain_sanity_texts), plain_non_broken_rate)

        c3b_0_3_pass = required_arms_ok and counts_ok and ids_exact_ok and nonempty_ok and plain_sanity_ok and d6_heldout_source_ok

        # C3b.0.4
        ph = meta.get("provenance_hashes", {})
        exp_hashes = {
            "c3b_design_sha256": EXPECTED_C3B_DESIGN_SHA256,
            "c3b_constants_sha256": EXPECTED_C3B_CONSTANTS_SHA256,
            "c3b_bank_sha256": EXPECTED_C3B_BANK_SHA256,
            "c3_scoring_sha256": EXPECTED_C3_SCORING_SHA256,
        }
        receipt_prov_ok = all(ph.get(k) == v for k, v in exp_hashes.items())
        disk_prov_ok = True if not strict_provenance else (
            disk_hashes is not None and all(disk_hashes.get(k) == v for k, v in exp_hashes.items())
        )
        c3b_0_4_pass = (not strict_provenance) or (receipt_prov_ok and disk_prov_ok)

        c3b_0_pass = c3b_0_1_pass and c3b_0_2_pass and c3b_0_3_pass and c3b_0_4_pass
        gate_c3b0_doc = {
            "pass": c3b_0_pass,
            "C3b.0.1": {
                "pass": c3b_0_1_pass,
                "hooks_ok": hooks_ok,
                "restore_ok": restore_ok,
                "layers_compiled_total": layers_compiled_total,
                "restore_checks": restore_checks,
                "expected_min_restore_checks": min_restore_checks,
                "no_weights_saved_ok": no_weights_saved_ok,
                "exact_b1_ok": exact_b1_ok,
                "sham_ok": sham_ok,
                "sham_max_abs_dlogit": round(sham_dlogit, 6) if math.isfinite(sham_dlogit) else None,
                "sham_min_proj_ret_diagnostic": round(sham_min_proj, 6) if math.isfinite(sham_min_proj) else None,
                "liveness_ok": liveness_ok,
            },
            "C3b.0.2": {
                "pass": c3b_0_2_pass,
                "nonfinite_count": len(nonfinite_paths),
                "nonfinite_sample": nonfinite_paths[:5],
                "cv_ok": cv_ok,
                "rand_keys_ok": rand_keys_ok,
                "rand_norm_ok": rand_norm_ok,
                "rand_kl_ok": rand_kl_ok,
                "stem_preflight_ok": stem_pref_ok,
                "code_harness_preflight_ok": code_harness_preflight_ok,
                "ban_stem_ok": ban_stem_ok,
                "ban_stem_rates": ban_stem_rates,
            },
            "C3b.0.3": {
                "pass": c3b_0_3_pass,
                "required_arms_ok": required_arms_ok,
                "counts_ok": counts_ok,
                "ids_exact_ok": ids_exact_ok,
                "nonempty_ok": nonempty_ok,
                "plain_non_broken_rate": round(plain_non_broken_rate, 6),
                "plain_sanity_ok": plain_sanity_ok,
                "d6_heldout_source_ok": d6_heldout_source_ok,
            },
            "C3b.0.4": {
                "pass": c3b_0_4_pass,
                "strict_provenance": strict_provenance,
                "receipt_prov_ok": receipt_prov_ok,
                "disk_prov_ok": disk_prov_ok,
            },
        }
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        return _blocked_c3b0_scorecard({"pass": False, "schema_error": f"{type(exc).__name__}: {exc}"})

    if not c3b_0_pass:
        return _blocked_c3b0_scorecard(gate_c3b0_doc)

    # ------------------------------------------------------------------
    # Isolated Physical Diagnostic: D4_masking_check (OUTSIDE C3b.0!)
    # ------------------------------------------------------------------
    d4_check_doc = evaluate_d4_masking_check(receipt, c3b_constants)

    # ------------------------------------------------------------------
    # Per-Arm Gate: C3b.0_direction_quality (cos_ret >= 0.97 on L16..L27)
    # ------------------------------------------------------------------
    dq_raw = receipt["calibration"]["direction_quality"]
    dir_valid: dict[str, bool] = {}
    dir_min_cos: dict[str, float] = {}
    for arm_name in ("WSP_Pirate_12L", "WSP_NoMD_12L"):
        cvals = [float(dq_raw[arm_name][L]["cos_ret"]) for L in layers]
        dir_min_cos[arm_name] = round(min(cvals), 6)
        dir_valid[arm_name] = check_direction_quality_gate(cvals)

    # ------------------------------------------------------------------
    # Per-Arm Gate: C3b.0_commit_gate + Recomputed expected_d_fail for D_above_cap
    # ------------------------------------------------------------------
    dose_valid: dict[str, bool] = {}
    commit_audit: dict[str, Any] = {}
    d_curve_summary: dict[str, list[dict[str, Any]]] = {}
    expected_d_fail_by_arm: dict[str, float | None] = {}

    for arm_name in ("WSP_Pirate_12L", "WSP_NoMD_12L"):
        cg = receipt["commit_gate"]["arms"][arm_name]
        rungs = cg["rungs"]
        ladder_exact_ok = (len(rungs) == len(dose_ladder)) and all(
            math.isclose(float(rr["dose"]), exp_d, abs_tol=1e-9) for rr, exp_d in zip(rungs, dose_ladder)
        )
        recomputed_passing: list[float] = []
        expected_d_fail: float | None = None
        rung_checks: list[dict[str, Any]] = []
        for rr in rungs:
            ret_m = rr["bf16_retention"]
            min_proj = min(float(ret_m[L]["stored_rank1_proj_ret"]) for L in layers)
            min_frob = min(float(ret_m[L]["stored_frob_ret"]) for L in layers)
            p_kls = [float(x["off_kl"]) for x in rr["per_prompt_kl"]]
            p_max_pos = [float(x["max_single_pos_kl"]) for x in rr["per_prompt_kl"]]
            reserve_n_ok = (len(p_kls) == 8) and (len(rr["reserve_generations"]) == 8)
            med_kl = median(p_kls)
            max_kl = max(p_kls)
            max_pos = max(p_max_pos)
            n_brk = sum(1 for x in rr["reserve_generations"] if c3_scoring.is_broken(x["text"]))
            r_eval = check_commit_rung_gate(min_proj, min_frob, reserve_n_ok, med_kl, max_kl, max_pos, n_brk, cg_cfg)
            if r_eval["pass"]:
                recomputed_passing.append(float(rr["dose"]))
            if r_eval["above_cap_eligible"] and expected_d_fail is None:
                expected_d_fail = float(rr["dose"])
            rung_checks.append({
                "dose": float(rr["dose"]),
                "rep_ok": r_eval["rep_ok"],
                "min_proj_ret": round(min_proj, 6),
                "min_frob_ret": round(min_frob, 6),
                "reserve_n_ok": reserve_n_ok,
                "median_off_kl": round(med_kl, 6),
                "max_prompt_off_kl": round(max_kl, 6),
                "max_single_pos_kl": round(max_pos, 6),
                "n_broken": n_brk,
                "above_cap_eligible": r_eval["above_cap_eligible"],
                "pass": r_eval["pass"],
            })
        expected_d = recomputed_passing[-1] if recomputed_passing else None
        fallback = bool(cg["commit_gate_fallback"])
        chosen_d = float(cg["selected_d_star"])
        d_ok = ladder_exact_ok and (expected_d is not None) and (not fallback) and math.isclose(chosen_d, expected_d, abs_tol=1e-9)
        dose_valid[arm_name] = d_ok
        expected_d_fail_by_arm[arm_name] = expected_d_fail
        commit_audit[arm_name] = {
            "pass": d_ok,
            "ladder_exact_ok": ladder_exact_ok,
            "selected_d_star": chosen_d,
            "expected_d_star": expected_d,
            "expected_d_fail": expected_d_fail,
            "commit_gate_fallback": fallback,
            "rungs": rung_checks,
        }
        d_curve_summary[arm_name] = rung_checks

    # ------------------------------------------------------------------
    # Pillar-B Gate: Split C3b.0_NoMD_room (24 C3-measured + 8 fresh + 32 overall)
    # ------------------------------------------------------------------
    plain_nomd_by_id = {r["id"]: r for r in receipt["pillar_b"]["Plain"]}
    nomd_24_ids = [f"md_{i:02d}" for i in range(24)]
    nomd_8_ids = [f"md_{i:02d}" for i in range(24, 32)]
    nomd_32_ids = [p["id"] for p in bat["nomd_32"]]

    cts_24 = [c3_scoring.construct_types(plain_nomd_by_id[pid]["text"]) for pid in nomd_24_ids]
    cts_8 = [c3_scoring.construct_types(plain_nomd_by_id[pid]["text"]) for pid in nomd_8_ids]
    cts_32 = [c3_scoring.construct_types(plain_nomd_by_id[pid]["text"]) for pid in nomd_32_ids]

    has_md_24 = mean([1.0 if c >= 1 else 0.0 for c in cts_24])
    has_md_8 = mean([1.0 if c >= 1 else 0.0 for c in cts_8])
    has_md_32 = mean([1.0 if c >= 1 else 0.0 for c in cts_32])
    mean_ct_32 = mean([float(c) for c in cts_32])

    room_checks = check_nomd_room_gate(has_md_24, has_md_8, has_md_32, mean_ct_32)
    nomd_room_pass = room_checks["pass"]
    nomd_room_doc = {
        **room_checks,
        "measured_24_has_markdown": round(has_md_24, 6),
        "measured_24_n_has_markdown": sum(1 for c in cts_24 if c >= 1),
        "fresh_8_has_markdown": round(has_md_8, 6),
        "fresh_8_n_has_markdown": sum(1 for c in cts_8 if c >= 1),
        "overall_32_has_markdown": round(has_md_32, 6),
        "overall_32_n_has_markdown": sum(1 for c in cts_32 if c >= 1),
        "overall_32_mean_construct_types": round(mean_ct_32, 6),
    }

    # ------------------------------------------------------------------
    # Reader Gate C3b.R & Capability Gate EQ-C3b.2
    # ------------------------------------------------------------------
    c3b_r_doc = evaluate_c3b_r_gate(
        receipt, c3b_constants, reader_labels,
        receipt_sha256=receipt_sha256,
        disk_hashes=disk_hashes,
        preflight_file_info=preflight_file_info,
        strict_provenance=strict_provenance,
        allow_mock_readers=allow_mock_readers,
    )
    eq_c3b_2_doc = evaluate_eq_c3b_2(receipt, c3b_constants, run_code_subprocess=run_code_subprocess)

    # ------------------------------------------------------------------
    # Confirmatory Holm-2 Statistics (H1 & H2, ID-indexed, pinned seeds!)
    # ------------------------------------------------------------------
    labels_map = (reader_labels or {}).get("labels", {})
    pa_ids = [p["id"] for p in bat["identity_20"] + bat["implied_20"]]
    id_dir_ids = {p["id"] for p in bat["identity_20"] if p.get("subtype") == "direct"}
    id_ind_ids = {p["id"] for p in bat["identity_20"] if p.get("subtype") == "indirect"}
    imp_ids = {p["id"] for p in bat["implied_20"]}

    def _strict_pirate_by_id(arm_k: str, ban_k: str) -> tuple[dict[str, float] | None, dict[str, bool], dict[str, Any]]:
        rows_by_id = {r["id"]: r for r in receipt["pillar_a"][ban_k][arm_k]}
        broken_by_id = {pid: c3_scoring.is_broken(rows_by_id[pid]["text"]) for pid in pa_ids}
        stem_by_id = {pid: c3_scoring.surface_stem_hit(rows_by_id[pid]["text"], surface_regex) for pid in pa_ids}
        ext_ban_by_id = {pid: c3_scoring.surface_stem_hit(rows_by_id[pid]["text"], strict_ext_ban_regex) for pid in pa_ids}
        ext_aud_by_id = {pid: c3_scoring.surface_stem_hit(rows_by_id[pid]["text"], ext_audit_regex) for pid in pa_ids}
        tok_lens = [int(rows_by_id[pid]["n_tokens"]) for pid in pa_ids]
        brk_rate = round(mean([1.0 if broken_by_id[pid] else 0.0 for pid in pa_ids]), 6)
        base_summary = {
            "broken_rate": brk_rate,
            "surface_stem_hit_rate": round(mean([1.0 if stem_by_id[pid] else 0.0 for pid in pa_ids]), 6),
            "strict_extended_ban_hit_rate": round(mean([1.0 if ext_ban_by_id[pid] else 0.0 for pid in pa_ids]), 6),
            "extended_maritime_and_cue_hit_rate": round(mean([1.0 if ext_aud_by_id[pid] else 0.0 for pid in pa_ids]), 6),
            "mean_tokens": round(mean([float(x) for x in tok_lens]), 3),
            "median_tokens": round(median([float(x) for x in tok_lens]), 3),
        }
        if reader_labels is None:
            return None, broken_by_id, base_summary
        sp_by_id: dict[str, float] = {}
        coh_any: list[float] = []
        coh_both: list[float] = []
        for pid in pa_ids:
            k = f"pillar_a|{ban_k}|{arm_k}|{pid}"
            lrec = labels_map.get(k)
            if not lrec or not lrec.get("ds") or not lrec.get("luna"):
                return None, broken_by_id, {**base_summary, "missing_label": k}
            p1 = lrec["ds"].get("pirate_persona")
            p2 = lrec["luna"].get("pirate_persona")
            c1 = lrec["ds"].get("coherent")
            c2 = lrec["luna"].get("coherent")
            if p1 is None or p2 is None or c1 is None or c2 is None:
                return None, broken_by_id, {**base_summary, "missing_label": k}
            ban_clean = (ban_k == "none") or ((not stem_by_id[pid]) and (not ext_ban_by_id[pid]))
            ok = bool(p1) and bool(p2) and (not broken_by_id[pid]) and ban_clean
            sp_by_id[pid] = 1.0 if ok else 0.0
            coh_any.append(1.0 if (bool(c1) or bool(c2)) else 0.0)
            coh_both.append(1.0 if (bool(c1) and bool(c2)) else 0.0)
        return sp_by_id, broken_by_id, {
            **base_summary,
            "strict_pirate_rate": round(mean([sp_by_id[pid] for pid in pa_ids]), 6),
            "reader_any_coherent_rate": round(mean(coh_any), 6),
            "reader_both_coherent_rate": round(mean(coh_both), 6),
        }

    # H1
    sp_pir_ban, brk_pir_ban, sum_pir_ban = _strict_pirate_by_id("WSP_Pirate_12L@d*_pirate", "common_pirate_ban")
    sp_rands_ban = [_strict_pirate_by_id(f"Rand_Pirate_12L_s{s}@d*_pirate", "common_pirate_ban") for s in range(3)]

    h1_diffs: list[float] | None = None
    h1_p = 1.0
    h1_ci: dict[str, float] | None = None
    if sp_pir_ban is not None and all(x[0] is not None for x in sp_rands_ban):
        h1_diffs = [
            sp_pir_ban[pid] - mean([sp_rands_ban[s][0][pid] for s in range(3)])  # type: ignore[index]
            for pid in pa_ids
        ]
        h1_p = paired_sign_flip_p_one_sided(h1_diffs, n_perm=n_perm, seed=perm_seed)
        h1_ci = bca_paired_mean_ci90(h1_diffs, n_boot=n_boot, seed=bca_seed)

    h1_mean_tok_arm = float(sum_pir_ban["mean_tokens"])
    h1_mean_tok_rand = mean([float(sp_rands_ban[s][2]["mean_tokens"]) for s in range(3)])
    h1_len_ratio = h1_mean_tok_arm / max(h1_mean_tok_rand, EPS)
    h1_any_coh_arm = sum_pir_ban.get("reader_any_coherent_rate")
    h1_both_coh_arm = sum_pir_ban.get("reader_both_coherent_rate")
    h1_both_coh_rand = (
        mean([float(sp_rands_ban[s][2]["reader_both_coherent_rate"]) for s in range(3)])
        if all("reader_both_coherent_rate" in sp_rands_ban[s][2] for s in range(3))
        else None
    )
    brk_h1_arm = mean([1.0 if brk_pir_ban[pid] else 0.0 for pid in pa_ids])
    brk_h1_rand = mean([mean([1.0 if sp_rands_ban[s][1][pid] else 0.0 for s in range(3)]) for pid in pa_ids])
    h1_mean_y = mean(h1_diffs) if h1_diffs is not None else 0.0
    h1_bars = check_h1_bars(h1_mean_y, h1_len_ratio, h1_any_coh_arm, h1_both_coh_arm, h1_both_coh_rand, brk_h1_arm, brk_h1_rand)

    # H2
    nomd_arm_by_id = {r["id"]: r for r in receipt["pillar_b"]["WSP_NoMD_12L@d*_nomd"]}
    nomd_rand_by_id = [{r["id"]: r for r in receipt["pillar_b"][f"Rand_NoMD_12L_s{s}@d*_nomd"]} for s in range(3)]
    arm_cts = {pid: c3_scoring.construct_types(nomd_arm_by_id[pid]["text"]) for pid in nomd_32_ids}
    rand_cts = [{pid: c3_scoring.construct_types(nomd_rand_by_id[s][pid]["text"]) for pid in nomd_32_ids} for s in range(3)]

    h2_u_diffs = [mean([float(rand_cts[s][pid]) for s in range(3)]) - float(arm_cts[pid]) for pid in nomd_32_ids]
    h2_mean_u = mean(h2_u_diffs)
    h2_has_md_arm = mean([1.0 if arm_cts[pid] >= 1 else 0.0 for pid in nomd_32_ids])
    h2_has_md_rand = mean([mean([1.0 if rand_cts[s][pid] >= 1 else 0.0 for s in range(3)]) for pid in nomd_32_ids])
    h2_delta_plain = has_md_32 - h2_has_md_arm
    h2_delta_rand = h2_has_md_rand - h2_has_md_arm
    h2_brk_arm = mean([1.0 if c3_scoring.is_broken(nomd_arm_by_id[pid]["text"]) else 0.0 for pid in nomd_32_ids])
    h2_brk_rand = mean([mean([1.0 if c3_scoring.is_broken(nomd_rand_by_id[s][pid]["text"]) else 0.0 for s in range(3)]) for pid in nomd_32_ids])
    h2_tok_arm = mean([float(nomd_arm_by_id[pid]["n_tokens"]) for pid in nomd_32_ids])
    h2_tok_rand = mean([mean([float(nomd_rand_by_id[s][pid]["n_tokens"]) for s in range(3)]) for pid in nomd_32_ids])
    h2_len_ratio = h2_tok_arm / max(h2_tok_rand, EPS)

    h2_p = paired_sign_flip_p_one_sided(h2_u_diffs, n_perm=n_perm, seed=perm_seed)
    h2_ci = bca_paired_mean_ci90(h2_u_diffs, n_boot=n_boot, seed=bca_seed)
    h2_bars = check_h2_bars(h2_mean_u, h2_delta_plain, h2_delta_rand, h2_len_ratio, h2_brk_arm, h2_brk_rand)

    # Exact m = 2 Holm-Bonferroni via frozen c3_scoring.holm_reject()!
    pvals = {"H1": h1_p, "H2": h2_p}
    holm_decisions = c3_scoring.holm_reject(pvals, alpha=holm_alpha)

    v_h1 = map_c3b_verdict(c3_scoring.resolve_verdict(
        c3_0_pass=c3b_0_pass,
        direction_valid=dir_valid["WSP_Pirate_12L"],
        dose_valid=dose_valid["WSP_Pirate_12L"],
        requires_nomd_room=False,
        nomd_room_pass=True,
        reader_dependent=True,
        c3_r_pass=bool(c3b_r_doc["h1_pass"]),
        eq_c3_2_pass=bool(eq_c3b_2_doc["WSP_Pirate_12L@d*_pirate"]["pass"]),
        broken_bar_pass=h1_bars["broken_bar_pass"],
        holm_sig=holm_decisions["H1"],
        primary_effect_bar_pass=h1_bars["primary_effect_bar_pass"],
        co_primary_bars_pass=h1_bars["co_primary_bars_pass"],
    ))

    v_h2 = map_c3b_verdict(c3_scoring.resolve_verdict(
        c3_0_pass=c3b_0_pass,
        direction_valid=dir_valid["WSP_NoMD_12L"],
        dose_valid=dose_valid["WSP_NoMD_12L"],
        requires_nomd_room=True,
        nomd_room_pass=nomd_room_pass,
        reader_dependent=False,
        c3_r_pass=True,
        eq_c3_2_pass=bool(eq_c3b_2_doc["WSP_NoMD_12L@d*_nomd"]["pass"]),
        broken_bar_pass=h2_bars["broken_bar_pass"],
        holm_sig=holm_decisions["H2"],
        primary_effect_bar_pass=h2_bars["primary_effect_bar_pass"],
        co_primary_bars_pass=h2_bars["co_primary_bars_pass"],
    ))

    # ------------------------------------------------------------------
    # Secondary & Exploratory Readouts (Outside Holm; D0 & D5 gated on C3b.R!)
    # ------------------------------------------------------------------
    c3b_r_passed = bool(c3b_r_doc["pass"])

    # D0 (Round-1 4a: gated on C3b.R PASS like H1!)
    d0_arms: dict[str, Any] = {
        "status": "READ" if c3b_r_passed else "D0_UNREAD_LOW_KAPPA",
        "c3b_r_pass": c3b_r_passed,
        "none": {},
        "common_pirate_ban": {},
    }
    for ban_k in ("none", "common_pirate_ban"):
        for arm_k in sorted(receipt["pillar_a"][ban_k].keys()):
            sp_v, _, sm = _strict_pirate_by_id(arm_k, ban_k)
            if not c3b_r_passed:
                sm = {
                    k: v for k, v in sm.items()
                    if k not in ("strict_pirate_rate", "reader_any_coherent_rate", "reader_both_coherent_rate")
                }
                sm["strict_pirate_rate"] = None
                sm["status"] = "D0_UNREAD_LOW_KAPPA"
            elif sp_v is not None:
                sm["identity_direct_10_rate"] = round(mean([sp_v[pid] for pid in pa_ids if pid in id_dir_ids]), 6)
                sm["identity_indirect_10_rate"] = round(mean([sp_v[pid] for pid in pa_ids if pid in id_ind_ids]), 6)
                sm["implied_20_rate"] = round(mean([sp_v[pid] for pid in pa_ids if pid in imp_ids]), 6)
            d0_arms[ban_k][arm_k] = sm

    # D1: Part 213 Copula
    d1_plain = {r["id"]: r for r in receipt["D1_copula"]["arms"]["Plain"]}
    d1_summary: dict[str, Any] = {}
    for arm_k, rows in sorted(receipt["D1_copula"]["arms"].items()):
        diffs = [
            (float(r["lp_self"]) - float(d1_plain[r["id"]]["lp_self"]))
            - (float(r["lp_stranger"]) - float(d1_plain[r["id"]]["lp_stranger"]))
            for r in rows
        ]
        raw_sms = [float(r["self_minus_stranger"]) for r in rows]
        d1_summary[arm_k] = {
            "mean_self_minus_stranger": round(mean(raw_sms), 6),
            "mean_delta_vs_plain": round(mean(diffs), 6),
        }

    # D2: Sibling Bleed + em_dash_per_1k_words
    p_on_nomd = receipt["D2_sibling_bleed"]["WSP_Pirate_12L@d*_pirate_on_NOMD_32"]
    p_on_nomd_cts = [c3_scoring.construct_types(r["text"]) for r in p_on_nomd]
    nomd_on_id = receipt["D2_sibling_bleed"]["WSP_NoMD_12L@d*_nomd_on_IDENTITY_20"]
    nomd_id_sp: list[float] = []
    if c3b_r_passed:
        for r in nomd_on_id:
            lrec = labels_map.get(f"D2_sibling_bleed|none|WSP_NoMD_12L@d*_nomd|{r['id']}")
            if lrec and lrec.get("ds") and lrec.get("luna") and lrec["ds"].get("pirate_persona") is not None and lrec["luna"].get("pirate_persona") is not None:
                ok = bool(lrec["ds"]["pirate_persona"]) and bool(lrec["luna"]["pirate_persona"]) and (not c3_scoring.is_broken(r["text"]))
                nomd_id_sp.append(1.0 if ok else 0.0)
    d2_sibling = {
        "pirate_12L_on_NOMD_32_has_md": round(mean([1.0 if c >= 1 else 0.0 for c in p_on_nomd_cts]), 6),
        "pirate_12L_on_NOMD_32_mean_ct": round(mean([float(c) for c in p_on_nomd_cts]), 6),
        "nomd_12L_on_IDENTITY_20_strict_pirate": round(mean(nomd_id_sp), 6) if (c3b_r_passed and len(nomd_id_sp) == 20) else None,
        "nomd_12L_on_IDENTITY_20_broken": round(mean([1.0 if c3_scoring.is_broken(r["text"]) else 0.0 for r in nomd_on_id]), 6),
    }

    em_dash_by_nomd_arm: dict[str, float] = {}
    for arm_k, rows in sorted(receipt["pillar_b"].items()):
        em_dash_by_nomd_arm[arm_k] = round(em_dash_per_1k_words([r["text"] for r in rows]), 4)

    # D3: Reference ceilings & showcase
    demo_ids = list(bat["demo_showcase_ids_12"])
    sys_nomd_cts = [c3_scoring.construct_types(r["text"]) for r in receipt["pillar_b"]["Sys_NoMD"]]
    d3_ceilings = {
        "Sys_Pirate_none_strict_pirate": d0_arms["none"]["Sys_Pirate"].get("strict_pirate_rate"),
        "Sys_Pirate_common_ban_strict_pirate": d0_arms["common_pirate_ban"]["Sys_Pirate"].get("strict_pirate_rate"),
        "Sys_NoMD_has_md": round(mean([1.0 if c >= 1 else 0.0 for c in sys_nomd_cts]), 6),
        "Sys_NoMD_mean_ct": round(mean([float(c) for c in sys_nomd_cts]), 6),
        "Sys_NoMD_delta_has_md_vs_plain": round(has_md_32 - mean([1.0 if c >= 1 else 0.0 for c in sys_nomd_cts]), 6),
    }
    pa_prompt_text = {p["id"]: p["text"] for p in bat["identity_20"] + bat["implied_20"]}
    d3_showcase: list[dict[str, Any]] = []
    for ban_k in ("none", "common_pirate_ban"):
        by_arm = {
            a: {r["id"]: r["text"] for r in receipt["pillar_a"][ban_k][a]}
            for a in ("Plain", "Sys_Pirate", "WSP_Pirate_12L@d*_pirate", "Rand_Pirate_12L_s0@d*_pirate")
        }
        for pid in demo_ids:
            d3_showcase.append({
                "ban": ban_k,
                "id": pid,
                "prompt": pa_prompt_text[pid],
                "Plain": by_arm["Plain"][pid],
                "Sys_Pirate": by_arm["Sys_Pirate"][pid],
                "WSP_Pirate_12L": by_arm["WSP_Pirate_12L@d*_pirate"][pid],
                "Rand_Pirate_12L_s0": by_arm["Rand_Pirate_12L_s0@d*_pirate"][pid],
            })

    # D5: Multi-turn Turn 3 (Round-1 4a: gated on C3b.R PASS like H1!)
    d5_summary: dict[str, Any] = {
        "status": "READ" if c3b_r_passed else "D5_UNREAD_LOW_KAPPA",
        "c3b_r_pass": c3b_r_passed,
        "arms": {},
    }
    for arm_k, rows in sorted(receipt["D5_multiturn"].items()):
        t3_brk = [c3_scoring.is_broken(r["turns"][2]["text"]) for r in rows]
        t3_sp: list[float] = []
        if c3b_r_passed:
            for r, brk in zip(rows, t3_brk):
                lrec = labels_map.get(f"D5_multiturn|none|{arm_k}|{r['id']}")
                if lrec and lrec.get("ds") and lrec.get("luna") and lrec["ds"].get("pirate_persona") is not None and lrec["luna"].get("pirate_persona") is not None:
                    t3_sp.append(1.0 if (bool(lrec["ds"]["pirate_persona"]) and bool(lrec["luna"]["pirate_persona"]) and (not brk)) else 0.0)
        d5_summary["arms"][arm_k] = {
            "turn3_strict_pirate_rate": round(mean(t3_sp), 6) if (c3b_r_passed and len(t3_sp) == 12) else None,
            "turn3_broken_rate": round(mean([1.0 if b else 0.0 for b in t3_brk]), 6),
        }

    # D6_heldout (also gated on C3b.R PASS)
    d6_summary: dict[str, Any] = {
        "status": "READ" if c3b_r_passed else "D6_UNREAD_LOW_KAPPA",
        "source_split": receipt["D6_heldout"].get("source_split"),
        "conditions": {},
    }
    for cond_k, rows in sorted(receipt["D6_heldout"]["conditions"].items()):
        brk_v = [c3_scoring.is_broken(r["text"]) for r in rows]
        stem_v = [c3_scoring.surface_stem_hit(r["text"], surface_regex) for r in rows]
        ext_v = [c3_scoring.surface_stem_hit(r["text"], strict_ext_ban_regex) for r in rows]
        sp_v: list[float] = []
        if c3b_r_passed:
            for r, brk, st, ex in zip(rows, brk_v, stem_v, ext_v):
                lrec = labels_map.get(f"D6_heldout|{cond_k}|{cond_k}|{r['id']}")
                if lrec and lrec.get("ds") and lrec.get("luna") and lrec["ds"].get("pirate_persona") is not None and lrec["luna"].get("pirate_persona") is not None:
                    sp_v.append(1.0 if (bool(lrec["ds"]["pirate_persona"]) and bool(lrec["luna"]["pirate_persona"]) and (not brk) and (not st) and (not ex)) else 0.0)
        d6_summary["conditions"][cond_k] = {
            "strict_pirate_rate": round(mean(sp_v), 6) if (c3b_r_passed and len(sp_v) == 12) else None,
            "broken_rate": round(mean([1.0 if b else 0.0 for b in brk_v]), 6),
            "surface_stem_hit_rate": round(mean([1.0 if s else 0.0 for s in stem_v]), 6),
            "strict_extended_ban_hit_rate": round(mean([1.0 if e else 0.0 for e in ext_v]), 6),
        }

    # Helper for 12-item Pirate slices (D_above_cap & D_key_ablation)
    plain_ban_sp, _, _ = _strict_pirate_by_id("Plain", "common_pirate_ban")
    plain_none_sp, _, _ = _strict_pirate_by_id("Plain", "none")

    def _score_pirate_slice_12(
        section: str,
        arm_key: str,
        gens_by_ban: dict[str, list[dict[str, Any]]],
        unread_tag: str,
    ) -> dict[str, Any]:
        if not c3b_r_passed:
            return {"status": unread_tag, "c3b_r_pass": False}
        out_b: dict[str, Any] = {"status": "READ", "c3b_r_pass": True, "by_ban": {}}
        for ban_k in ("none", "common_pirate_ban"):
            rows = gens_by_ban[ban_k]
            sp_vals: list[float] = []
            brk_cnt = 0
            tok_lens: list[float] = []
            missing = 0
            for r in rows:
                brk = c3_scoring.is_broken(r["text"])
                if brk:
                    brk_cnt += 1
                tok_lens.append(float(r["n_tokens"]))
                lrec = labels_map.get(f"{section}|{ban_k}|{arm_key}|{r['id']}")
                if not lrec or not lrec.get("ds") or not lrec.get("luna") or lrec["ds"].get("pirate_persona") is None or lrec["luna"].get("pirate_persona") is None:
                    missing += 1
                    continue
                ban_clean = (ban_k == "none") or (
                    not c3_scoring.surface_stem_hit(r["text"], surface_regex)
                    and not c3_scoring.surface_stem_hit(r["text"], strict_ext_ban_regex)
                )
                sp_vals.append(1.0 if (bool(lrec["ds"]["pirate_persona"]) and bool(lrec["luna"]["pirate_persona"]) and (not brk) and ban_clean) else 0.0)
            if missing > 0 or len(sp_vals) != len(rows):
                out_b["status"] = unread_tag
                out_b["by_ban"][ban_k] = {"missing_labels": missing, "broken_count": brk_cnt}
            else:
                rate = mean(sp_vals)
                plain_ref = plain_ban_sp if ban_k == "common_pirate_ban" else plain_none_sp
                plain_rate = mean([plain_ref[r["id"]] for r in rows]) if plain_ref is not None else 0.0
                out_b["by_ban"][ban_k] = {
                    "strict_pirate_rate": round(rate, 6),
                    "plain_slice_rate": round(plain_rate, 6),
                    "delta_vs_plain": round(rate - plain_rate, 6),
                    "broken_count": brk_cnt,
                    "broken_rate": round(brk_cnt / max(len(rows), 1), 6),
                    "mean_tokens": round(mean(tok_lens), 3),
                }
        return out_b

    # D_above_cap (Round-1 4b: recompute expected_d_fail from scored rungs and flag mismatch!)
    dac_raw = receipt.get("D_above_cap", {})
    d_above_cap_summary: dict[str, Any] = {}

    def _dac_trigger_matches(arm_name: str, dac_blk: dict[str, Any]) -> tuple[bool, float | None]:
        exp_df = expected_d_fail_by_arm[arm_name]
        got_trig = bool(dac_blk.get("triggered", False))
        got_df = dac_blk.get("d_fail")
        if exp_df is None:
            return (not got_trig) and (got_df is None), None
        ok = got_trig and (got_df is not None) and math.isclose(float(got_df), exp_df, abs_tol=1e-9)
        return ok, exp_df

    dac_p = dac_raw.get("WSP_Pirate_12L", {})
    p_trig_ok, p_exp_df = _dac_trigger_matches("WSP_Pirate_12L", dac_p)
    if not p_trig_ok:
        d_above_cap_summary["WSP_Pirate_12L"] = {
            "triggered": bool(dac_p.get("triggered", False)),
            "d_fail": dac_p.get("d_fail"),
            "expected_d_fail": p_exp_df,
            "trigger_consistent_ok": False,
            "reading": "D_ABOVE_CAP_TRIGGER_MISMATCH",
        }
    elif dac_p.get("triggered"):
        sc_p = _score_pirate_slice_12("D_above_cap", "WSP_Pirate_12L@d_fail", dac_p["generations"], "D_ABOVE_CAP_UNREAD")
        if sc_p["status"] == "D_ABOVE_CAP_UNREAD":
            reading_p = "D_ABOVE_CAP_UNREAD"
        else:
            b_stats = sc_p["by_ban"]["common_pirate_ban"]
            pos_flip = check_d_above_cap_pirate_flip(b_stats["delta_vs_plain"], b_stats["broken_count"])
            reading_p = "flip_requires_more_kl_than_0.65_cap" if pos_flip else "no_flip_even_above_0.65_cap"
        d_above_cap_summary["WSP_Pirate_12L"] = {
            "triggered": True,
            "d_fail": dac_p.get("d_fail"),
            "expected_d_fail": p_exp_df,
            "trigger_consistent_ok": True,
            "status": sc_p["status"],
            "reading": reading_p,
            "stats": sc_p,
        }
    else:
        d_above_cap_summary["WSP_Pirate_12L"] = {
            "triggered": False,
            "d_fail": None,
            "expected_d_fail": None,
            "trigger_consistent_ok": True,
        }

    dac_n = dac_raw.get("WSP_NoMD_12L", {})
    n_trig_ok, n_exp_df = _dac_trigger_matches("WSP_NoMD_12L", dac_n)
    if not n_trig_ok:
        d_above_cap_summary["WSP_NoMD_12L"] = {
            "triggered": bool(dac_n.get("triggered", False)),
            "d_fail": dac_n.get("d_fail"),
            "expected_d_fail": n_exp_df,
            "trigger_consistent_ok": False,
            "reading": "D_ABOVE_CAP_TRIGGER_MISMATCH",
        }
    elif dac_n.get("triggered"):
        rows_n = dac_n["generations"]
        n_cts = [c3_scoring.construct_types(r["text"]) for r in rows_n]
        p_cts = [c3_scoring.construct_types(plain_nomd_by_id[r["id"]]["text"]) for r in rows_n]
        n_has_md = mean([1.0 if c >= 1 else 0.0 for c in n_cts])
        delta_ct = mean([float(pc) - float(nc) for pc, nc in zip(p_cts, n_cts)])
        brk_cnt = sum(1 for r in rows_n if c3_scoring.is_broken(r["text"]))
        pos_flip_n = check_d_above_cap_nomd_flip(n_has_md, delta_ct, brk_cnt)
        d_above_cap_summary["WSP_NoMD_12L"] = {
            "triggered": True,
            "d_fail": dac_n.get("d_fail"),
            "expected_d_fail": n_exp_df,
            "trigger_consistent_ok": True,
            "has_md": round(n_has_md, 6),
            "delta_ct_above_cap": round(delta_ct, 6),
            "broken_count": brk_cnt,
            "reading": "flip_requires_more_kl_than_0.65_cap" if pos_flip_n else "no_flip_even_above_0.65_cap",
        }
    else:
        d_above_cap_summary["WSP_NoMD_12L"] = {
            "triggered": False,
            "d_fail": None,
            "expected_d_fail": None,
            "trigger_consistent_ok": True,
        }

    # D_key_ablation
    dka_raw = receipt.get("D_key_ablation", {})
    d_key_ablation_summary: dict[str, Any] = {}
    if "WSP_NoMD_12L_kWA" in dka_raw:
        kwa = dka_raw["WSP_NoMD_12L_kWA"]
        kwa_rows = kwa["nomd_32"]
        kwa_cts = [c3_scoring.construct_types(r["text"]) for r in kwa_rows]
        kwa_has_md = mean([1.0 if c >= 1 else 0.0 for c in kwa_cts])
        kwa_mean_ct = mean([float(c) for c in kwa_cts])
        kwa_brk = mean([1.0 if c3_scoring.is_broken(r["text"]) else 0.0 for r in kwa_rows])
        kwa_tok = mean([float(r["n_tokens"]) for r in kwa_rows])
        kwa_em = em_dash_per_1k_words([r["text"] for r in kwa_rows])
        em_dash_by_nomd_arm["WSP_NoMD_12L_kWA"] = round(kwa_em, 4)
        kwa_neu_med_kl = median([float(x["off_kl"]) for x in kwa["neutral_chat_16_kl"]]) if kwa.get("neutral_chat_16_kl") else None
        d_key_ablation_summary["WSP_NoMD_12L_kWA"] = {
            "selected_beta_star": kwa.get("selected_beta_star"),
            "fallback": bool(kwa.get("fallback", False)),
            "has_md": round(kwa_has_md, 6),
            "mean_construct_types": round(kwa_mean_ct, 6),
            "delta_has_md_vs_plain": round(has_md_32 - kwa_has_md, 6),
            "delta_ct_vs_plain": round(mean_ct_32 - kwa_mean_ct, 6),
            "broken_rate": round(kwa_brk, 6),
            "mean_tokens": round(kwa_tok, 3),
            "em_dash_per_1k_words": round(kwa_em, 4),
            "neutral_median_off_kl": None if kwa_neu_med_kl is None else round(kwa_neu_med_kl, 6),
        }
    if "WSP_Pirate_12L_kSparse32" in dka_raw:
        sp32 = dka_raw["WSP_Pirate_12L_kSparse32"]
        sc_sp32 = _score_pirate_slice_12("D_key_ablation", "WSP_Pirate_12L_kSparse32", sp32["generations"], "D_KEY_PIRATE_UNREAD")
        ret_sel = sp32.get("bf16_retention_selected", {})
        min_proj_sp32 = min((float(ret_sel[L]["stored_rank1_proj_ret"]) for L in layers if L in ret_sel), default=None)
        d_key_ablation_summary["WSP_Pirate_12L_kSparse32"] = {
            "support_k": sp32.get("support_k", 32),
            "selected_d_star": sp32.get("selected_d_star"),
            "fallback": bool(sp32.get("fallback", False)),
            "min_stored_rank1_proj_ret": None if min_proj_sp32 is None else round(min_proj_sp32, 6),
            "status": sc_sp32["status"],
            "stats": sc_sp32,
        }

    scorecard: dict[str, Any] = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "confirmatory_verdicts": {
            "C3b.0": "PASS" if c3b_0_pass else "FAIL",
            "H1": v_h1,
            "H2": v_h2,
        },
        "gates": {
            "C3b.0": gate_c3b0_doc,
            "D4_masking_check": d4_check_doc,
            "C3b.0_direction_quality": {
                "pass_by_arm": dir_valid,
                "min_cos_ret_by_arm": dir_min_cos,
            },
            "C3b.0_commit_gate": commit_audit,
            "C3b.0_NoMD_room": nomd_room_doc,
            "C3b.R": c3b_r_doc,
            "EQ_C3b_2": eq_c3b_2_doc,
        },
        "holm_2_family": {
            "alpha": holm_alpha,
            "m": 2,
            "n_perm": n_perm,
            "n_boot": n_boot,
            "perm_seed": perm_seed,
            "bca_seed": bca_seed,
            "p_values": {k: round(v, 8) for k, v in pvals.items()},
            "holm_rejected": holm_decisions,
            "hypotheses": {
                "H1": {
                    "verdict": v_h1,
                    "d_star": doses["d_star_pirate"],
                    "mean_y": round(h1_mean_y, 6) if h1_diffs is not None else None,
                    "p_perm": round(h1_p, 8),
                    "holm_sig": holm_decisions["H1"],
                    "bca_ci90": {k: round(v, 6) for k, v in h1_ci.items()} if h1_ci else None,
                    "length_ratio": round(h1_len_ratio, 6),
                    "reader_any_coherent_arm": h1_any_coh_arm,
                    "reader_both_coherent_arm": h1_both_coh_arm,
                    "reader_both_coherent_rand_mean": None if h1_both_coh_rand is None else round(h1_both_coh_rand, 6),
                    "co_primary_bars_pass": h1_bars["co_primary_bars_pass"],
                    "failed_co_primary_guards": h1_bars["failed_co_primary_guards"],
                    "broken_arm": round(brk_h1_arm, 6),
                    "broken_rand_mean": round(brk_h1_rand, 6),
                    "strict_extended_ban_hit_rate_arm": sum_pir_ban["strict_extended_ban_hit_rate"],
                    "extended_maritime_and_cue_hit_rate_arm": sum_pir_ban["extended_maritime_and_cue_hit_rate"],
                },
                "H2": {
                    "verdict": v_h2,
                    "d_star": doses["d_star_nomd"],
                    "mean_u": round(h2_mean_u, 6),
                    "p_perm": round(h2_p, 8),
                    "holm_sig": holm_decisions["H2"],
                    "bca_ci90": {k: round(v, 6) for k, v in h2_ci.items()},
                    "has_md_plain": round(has_md_32, 6),
                    "has_md_arm": round(h2_has_md_arm, 6),
                    "has_md_rand_mean": round(h2_has_md_rand, 6),
                    "delta_has_md_vs_plain": round(h2_delta_plain, 6),
                    "delta_has_md_vs_rand": round(h2_delta_rand, 6),
                    "length_ratio": round(h2_len_ratio, 6),
                    "co_primary_bars_pass": h2_bars["co_primary_bars_pass"],
                    "failed_co_primary_guards": h2_bars["failed_co_primary_guards"],
                    "broken_arm": round(h2_brk_arm, 6),
                    "broken_rand_mean": round(h2_brk_rand, 6),
                },
            },
        },
        "diagnostics": {
            "D0_pillar_a_by_ban_and_split": d0_arms,
            "D1_copula": d1_summary,
            "D2_sibling_bleed": d2_sibling,
            "D3_ceilings": d3_ceilings,
            "D3_showcase_12": d3_showcase,
            "D4_masking_check": d4_check_doc,
            "D5_multiturn": d5_summary,
            "D6_heldout": d6_summary,
            "D_curve": d_curve_summary,
            "D_above_cap": d_above_cap_summary,
            "D_align": receipt["calibration"].get("D_align", {}),
            "em_dash_per_1k_words_by_nomd_arm": em_dash_by_nomd_arm,
            "D_key_ablation": d_key_ablation_summary,
        },
    }
    return scorecard


def render_scorecard_markdown(sc: dict[str, Any]) -> str:
    """Render deterministic GitHub-Flavored Markdown scorecard from `score_stage_c3b` output."""
    lines: list[str] = [
        "# Stage C3b (`WSP v2`) Confirmatory & Diagnostic Scorecard",
        "",
    ]
    d4g = sc["gates"].get("D4_masking_check", {})
    if d4g.get("on_fail_flag"):
        lines.extend([
            f"> **DIAGNOSTIC FLAG (`D4_masking_check` FAIL):** *{d4g['on_fail_flag']}* (isolated outside global `C3b.0`; mechanistic attribution of delivery gain to masking withheld).",
            "",
        ])

    lines.extend([
        "## Confirmatory Verdicts (Holm-2, `alpha = 0.05`)",
        "",
        "| Gate / Hypothesis | Verdict | Point Estimate | 90% BCa CI | `p_perm` | Holm Sig | Notes |",
        "|---|---|---:|---|---:|---|---|",
    ])
    cv = sc["confirmatory_verdicts"]
    lines.append(f"| **Global `C3b.0`** | **`{cv['C3b.0']}`** | — | — | — | — | `C3b.0.1..4` |")
    hyps = sc["holm_2_family"]["hypotheses"]
    for hid in ("H1", "H2"):
        h = hyps[hid]
        pt = h.get("mean_y") if hid == "H1" else h.get("mean_u")
        pt_s = "—" if pt is None else f"{pt:+.4f}"
        ci = h.get("bca_ci90")
        ci_s = "—" if not ci else f"[{ci['ci90_lo']:+.4f}, {ci['ci90_hi']:+.4f}]"
        if hid == "H1":
            if "broken_arm" in h:
                fg = h.get("failed_co_primary_guards") or []
                note = f"d*={h.get('d_star')}, lenR={h['length_ratio']:.3f}, brk={h['broken_arm']:.3f}" + (f", failed_guards={fg}" if fg else "")
            else:
                note = f"blocked ({h['verdict']})"
        else:
            if "delta_has_md_vs_plain" in h:
                fg = h.get("failed_co_primary_guards") or []
                note = (
                    f"d*={h.get('d_star')}, Δmd_P={h['delta_has_md_vs_plain']:+.3f}, "
                    f"Δmd_R={h['delta_has_md_vs_rand']:+.3f}, lenR={h['length_ratio']:.3f}"
                    + (f", failed_guards={fg}" if fg else "")
                )
            else:
                note = f"blocked ({h['verdict']})"
        lines.append(
            f"| **`{hid}`** | **`{h['verdict']}`** | `{pt_s}` | `{ci_s}` | `{h['p_perm']:.6f}` | `{h['holm_sig']}` | {note} |"
        )

    lines.extend([
        "",
        "## Capability Preservation (`EQ-C3b.2`) & Non-Gating Refusal Readout",
        "",
        "| Arm | Gating | Math Acc (Strict) | Code Pass | Neutral `off_KL` | Neutral Broken | Refusal (Harm/Ben) | `EQ-C3b.2` |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for arm_k, am in sorted(sc["gates"]["EQ_C3b_2"].items()):
        lines.append(
            f"| `{arm_k}` | `{'YES' if am['gating'] else 'log'}` | `{am['math_acc']:.3f}` (`{am['math_strict_final_acc']:.3f}`) "
            f"| `{am['code_pass_rate']:.3f}` | `{am['neutral_median_off_kl']:.4f}` | `{am['neutral_broken_rate']:.3f}` "
            f"| `{am['refusal_keyword_rate']:.2f}` (`{am['refusal_harmful_4_rate']:.2f}/{am['refusal_benign_4_rate']:.2f}`) "
            f"| `{'PASS' if am['pass'] else 'FAIL'}` |"
        )

    g0 = sc["gates"]["C3b.0"]
    dq_gate = sc["gates"].get("C3b.0_direction_quality")
    cg_gate = sc["gates"].get("C3b.0_commit_gate")
    nr_gate = sc["gates"].get("C3b.0_NoMD_room")
    c3br_gate = sc["gates"].get("C3b.R", {})

    if "C3b.0.1" in g0:
        c0_detail = f"(`C3b.0.1={g0['C3b.0.1']['pass']}`, `C3b.0.2={g0['C3b.0.2']['pass']}`, `C3b.0.3={g0['C3b.0.3']['pass']}`, `C3b.0.4={g0['C3b.0.4']['pass']}`)"
    else:
        c0_detail = f"(`schema_error={g0.get('schema_error')}`)"

    lines.extend([
        "",
        "## Gate & Diagnostic Summary",
        "",
        f"- **`C3b.0`:** `{g0['pass']}` {c0_detail}",
        f"- **`D4_masking_check` (outside `C3b.0`):** `{d4g.get('pass')}` (seed=`{d4g.get('masked_over_unmasked_seed_dlogit_ratio')}x`/`{d4g.get('mismasked_over_unmasked_seed_dlogit_ratio')}x`, L2=`{d4g.get('masked_over_unmasked_logit_l2_ratio')}x`/`{d4g.get('mismasked_over_unmasked_logit_l2_ratio')}x`, eff=`{d4g.get('masked_over_unmasked_efficiency_ratio')}x`/`{d4g.get('mismasked_over_unmasked_efficiency_ratio')}x`)",
        f"- **`C3b.0_direction_quality`:** `{dq_gate['pass_by_arm'] if dq_gate else 'not_evaluated'}`",
        f"- **`C3b.0_commit_gate`:** `{ {k: v['pass'] for k, v in cg_gate.items()} if cg_gate else 'not_evaluated' }`",
        f"- **`C3b.0_NoMD_room`:** `{nr_gate.get('pass') if nr_gate else 'not_evaluated'}`"
        + (
            f" (`24m={nr_gate['measured_24_has_markdown']:.3f}`, `8f={nr_gate['fresh_8_has_markdown']:.3f}`, `32all={nr_gate['overall_32_has_markdown']:.3f}`, `mean_ct={nr_gate['overall_32_mean_construct_types']:.3f}`)"
            if nr_gate and "measured_24_has_markdown" in nr_gate
            else ""
        ),
        f"- **`C3b.R`:** `{c3br_gate.get('pass')}` (`H1={c3br_gate.get('h1_pass')}`)",
        "",
    ])

    diag = sc.get("diagnostics", {})
    if diag:
        d3c = diag.get("D3_ceilings", {})
        dac = diag.get("D_above_cap", {})
        dka = diag.get("D_key_ablation", {})
        lines.extend([
            "## Secondary & Exploratory Readouts (Outside Holm)",
            "",
            f"- **`D3` Reference Ceilings:** `Sys_Pirate(none)={d3c.get('Sys_Pirate_none_strict_pirate')}`, `Sys_Pirate(ban)={d3c.get('Sys_Pirate_common_ban_strict_pirate')}`, `Sys_NoMD(has_md)={d3c.get('Sys_NoMD_has_md')}` (`ΔvsPlain={d3c.get('Sys_NoMD_delta_has_md_vs_plain')}`)",
            f"- **`D_above_cap`:** Pirate=`{dac.get('WSP_Pirate_12L', {}).get('reading', 'not_triggered')}`, NoMD=`{dac.get('WSP_NoMD_12L', {}).get('reading', 'not_triggered')}`",
            f"- **`D_key_ablation` (`WSP_NoMD_12L_kWA`):** `{dka.get('WSP_NoMD_12L_kWA', {})}`",
            f"- **`D_key_ablation` (`WSP_Pirate_12L_kSparse32`):** status=`{dka.get('WSP_Pirate_12L_kSparse32', {}).get('status')}`, d*=`{dka.get('WSP_Pirate_12L_kSparse32', {}).get('selected_d_star')}`",
            f"- **`em_dash_per_1k_words` (`NOMD_32`):** `{diag.get('em_dash_per_1k_words_by_nomd_arm', {})}`",
            "",
        ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score Stage C3b (WSP v2) receipt (100% pure Python stdlib).")
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--c3b-constants", type=Path, default=HERE / "c3b_prereg_constants.json")
    ap.add_argument("--design", type=Path, default=HERE / "OLMO3-WSP-C3B-DESIGN.md")
    ap.add_argument("--bank", type=Path, default=HERE / "c3b_vector_bank.safetensors")
    ap.add_argument("--scoring", type=Path, default=HERE / "c3_scoring.py")
    ap.add_argument("--reader-labels", type=Path, default=None)
    ap.add_argument("--preflight-canary", type=Path, default=None, help="Optional explicit path to c3b-preflight-canary-both.json.")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    args = ap.parse_args(argv)

    receipt = load_json_or_gz(args.receipt)
    receipt_sha = sha256_file(args.receipt)
    c3b_constants = json.loads(args.c3b_constants.read_text(encoding="utf-8"))
    reader_labels = load_json_or_gz(args.reader_labels) if args.reader_labels else None
    preflight_file_info = None
    if args.preflight_canary is not None and args.reader_labels is not None:
        preflight_file_info = {
            "sha256": sha256_file(args.preflight_canary),
            "mtime": float(args.preflight_canary.stat().st_mtime),
            "labels_mtime": float(args.reader_labels.stat().st_mtime),
        }
    disk_hashes = {
        "c3b_design_sha256": sha256_file(args.design),
        "c3b_constants_sha256": sha256_file(args.c3b_constants),
        "c3b_bank_sha256": sha256_file(args.bank),
        "c3_scoring_sha256": sha256_file(args.scoring),
    }

    sc = score_stage_c3b(
        receipt,
        c3b_constants,
        reader_labels=reader_labels,
        disk_hashes=disk_hashes,
        receipt_sha256=receipt_sha,
        preflight_file_info=preflight_file_info,
        strict_provenance=True,
        allow_mock_readers=False,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        run_code_subprocess=True,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(sc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_scorecard_markdown(sc) + "\n", encoding="utf-8")

    print(json.dumps(sc["confirmatory_verdicts"], indent=2, sort_keys=True))
    return 0 if sc["confirmatory_verdicts"]["C3b.0"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
