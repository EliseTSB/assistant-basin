#!/usr/bin/env python3
"""score_olmo3_c3.py — 100% Pure Python 3.10+ Stdlib Confirmatory & Diagnostic Scorer for Stage C3 (WSP v1).

Zero third-party imports (no numpy, scipy, torch, safetensors). Reproducible byte-for-byte under
`/usr/bin/python3 -I` (`cmp` verified). Imports frozen mechanical scorers, `holm_reject()`,
`resolve_verdict()`, `cohens_kappa()`, and `raw_agreement()` from `c3_scoring.py`.

Enforces all 8 pre-registered gates fail-closed (`C3.0.1..4`, `C3.0_direction_quality`,
`C3.0_commit_gate`, `C3.0_NoMD_room`, `C3.R`), `EQ-C3.2` capability preservation (gating on
`Con@d*_con`, `Con@d_common`, `NoMD_Con@d*_nomd_con`, `NoMD_Lex@d*_nomd_lex`; logged-only on
`Lex@d*_lex`, `Lex@d_common`, `Sys_*`, `Rand_*_s0`), exact Holm-4 step-down multiplicity (`alpha=0.05`),
100,000-draw paired sign-flip permutations (`PERM_SEED=2026112601`), 10,000-draw descriptive 90% paired
BCa CIs (`BCA_SEED=2026112602`), and all 9 ordered verdicts (`c3_scoring.VERDICT_PRECEDENCE`).
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

SCORECARD_SCHEMA_VERSION = "olmo3-stage-c3-scorecard-v1"
EXPECTED_RECEIPT_SCHEMA = "olmo3-stage-c3-receipt-v1"
EXPECTED_READER_SCHEMA = "olmo3-stage-c3-reader-labels-v1"

EXPECTED_C3_DESIGN_SHA256 = "dbdab600e72fdba21f22024d0b1a90575cd2d6d82195fdb6b5fe4a49f494023a"
EXPECTED_C3_CONSTANTS_SHA256 = "bb568d8a29c3d3515563844caaffa9ede9d4c2bc926ed78c68484ada4a1d4fec"
EXPECTED_C3_BANK_SHA256 = "e264cea9785f91f07a34c7e3b37ed8451474c888a340a0ec13ed63e8e6f820b2"
EXPECTED_C3_SCORING_SHA256 = "b2fdfdf480f28554117765caa49126e4567481e6d34d2917ef1287cc0675040f"

PERM_SEED = 2026112601
BCA_SEED = 2026112602
DEFAULT_N_PERM = 100_000
DEFAULT_N_BOOT = 10_000

REFUSAL_RE = re.compile(
    r"\b(cannot|can't|won't|unable to|sorry|apologize|illegal|unethical|harmful|dangerous|cannot assist|cannot help|cannot provide)\b",
    re.IGNORECASE,
)


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
    tol = 1e-12 * max(1.0, abs(obs))
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

    if boots[0] == boots[-1] and abs(boots[0] - theta_hat) <= 1e-12 * max(1.0, abs(theta_hat)):
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
        adj_a = a if abs(denom) < 1e-12 else norm_cdf(z0 + (z0 + za) / denom)
        adj_a = min(max(adj_a, 0.0), 1.0)
        qs[a] = quantile(boots, adj_a)

    lo, hi = qs[0.05], qs[0.95]
    tol = 1e-12 * max(1.0, abs(theta_hat))
    if lo > theta_hat + tol or hi < theta_hat - tol:
        # Rare extreme-skew BCa failure fallback to percentile 90% CI while flagging containment
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


# ---------------------------------------------------------------------------
# Gate Verification Helpers (All 8 Gates + EQ-C3.2)
# ---------------------------------------------------------------------------
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
    in_mid = 0.10 <= prev <= 0.90
    if is_rand_pool:
        ok = (ra >= 0.85) and ((kap >= 0.50) if in_mid else True)
    else:
        ok = (kap >= 0.55) if in_mid else (ra >= 0.85)
    return {
        "n": len(y1),
        "raw_agreement": round(ra, 6),
        "kappa": round(kap, 6),
        "mean_prevalence": round(prev, 6),
        "in_mid_prevalence_0.10_0.90": in_mid,
        "pass": ok,
    }


def evaluate_c3_r_gate(
    receipt: dict[str, Any],
    c3_constants: dict[str, Any],
    reader_labels: dict[str, Any] | None,
    *,
    receipt_sha256: str | None = None,
    disk_hashes: dict[str, str] | None = None,
    strict_provenance: bool = True,
    allow_mock_readers: bool = False,
) -> dict[str, Any]:
    """Evaluate Gate C3.R (READER_GOLD_14 + exact reader_kappa_pool + per-arm reliability under common_pirate_ban)."""
    if reader_labels is None:
        return {"pass": False, "h1_pass": False, "h2_pass": False, "reason": "reader_labels_missing"}
    if reader_labels.get("schema_version") != EXPECTED_READER_SCHEMA:
        return {"pass": False, "h1_pass": False, "h2_pass": False, "reason": "reader_schema_mismatch"}
    rmeta = reader_labels.get("meta", {})
    if rmeta.get("mock_readers", False) and not allow_mock_readers:
        return {"pass": False, "h1_pass": False, "h2_pass": False, "reason": "mock_readers_refused"}
    # GLM F4 & DS F6: fail-closed provenance binding on receipt_sha256 + constants & scoring hashes in strict mode!
    if strict_provenance:
        r_sha = rmeta.get("receipt_sha256")
        if not receipt_sha256 or not isinstance(r_sha, str) or not r_sha or r_sha != receipt_sha256:
            return {"pass": False, "h1_pass": False, "h2_pass": False, "reason": "reader_receipt_sha256_missing_or_mismatch"}
        exp_const = (disk_hashes or {}).get("c3_constants_sha256", EXPECTED_C3_CONSTANTS_SHA256)
        exp_score = (disk_hashes or {}).get("c3_scoring_sha256", EXPECTED_C3_SCORING_SHA256)
        if rmeta.get("c3_constants_sha256") != exp_const or rmeta.get("c3_scoring_sha256") != exp_score:
            return {"pass": False, "h1_pass": False, "h2_pass": False, "reason": "reader_provenance_hash_mismatch"}

    pool_cfg = c3_constants["reader_kappa_pool"]
    gold_cfg = pool_cfg["gold_gate"]
    gold_items = c3_constants["reader_gold_14"]
    gold_raw = reader_labels.get("gold_14", {}).get("raw", {})

    # Recompute gold gate from raw predictions (never trust precomputed summary!)
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
        r_pass = (
            p_ok >= int(gold_cfg["pooled_min_correct"])
            and s_ok >= int(gold_cfg["subtle_zero_stem_min_correct"])
            and fp_ok >= int(gold_cfg["first_person_non_pirate_neg_min_correct"])
        )
        if not r_pass:
            gold_ok = False
        gold_by_reader[r] = {
            "pooled_correct": p_ok,
            "subtle_zero_stem_correct": s_ok,
            "first_person_non_pirate_neg_correct": fp_ok,
            "pass": r_pass,
        }

    labels_map = reader_labels.get("labels", {})
    pa_ids = [p["id"] for p in c3_constants["batteries"]["identity_20"] + c3_constants["batteries"]["implied_20"]]
    mt_ids = [p["id"] for p in c3_constants["batteries"]["multiturn_12"]]

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

    core_arms_7 = list(pool_cfg["core_arms_7"])
    mt_arms_5 = list(pool_cfg["multiturn_arms_5"])

    keys_none = [f"pillar_a|none|{arm}|{pid}" for arm in core_arms_7 for pid in pa_ids] + [
        f"D5_multiturn|none|{arm}|{pid}" for arm in mt_arms_5 for pid in mt_ids
    ]
    keys_ban = [f"pillar_a|common_pirate_ban|{arm}|{pid}" for arm in core_arms_7 for pid in pa_ids]
    keys_pooled = keys_none + keys_ban

    y1_pool, y2_pool, miss_pool = _collect_pair(keys_pooled)
    y1_none, y2_none, miss_none = _collect_pair(keys_none)
    y1_ban, y2_ban, miss_ban = _collect_pair(keys_ban)

    pool_complete = (miss_pool == 0) and (len(y1_pool) == len(keys_pooled))
    kap_pool = c3_scoring.cohens_kappa(y1_pool, y2_pool) if y1_pool else 0.0
    kap_none = c3_scoring.cohens_kappa(y1_none, y2_none) if y1_none else 0.0
    kap_ban = c3_scoring.cohens_kappa(y1_ban, y2_ban) if y1_ban else 0.0
    pool_ok = pool_complete and (kap_pool >= 0.60) and (kap_none >= 0.55) and (kap_ban >= 0.55)

    # Per-arm reliability under common_pirate_ban
    rand_keys = [
        f"pillar_a|common_pirate_ban|Rand_Pirate_s{s}@d*_con|{pid}" for s in range(3) for pid in pa_ids
    ]
    y1_r, y2_r, miss_r = _collect_pair(rand_keys)
    rand_rel = _eval_reader_arm_reliability(y1_r, y2_r, is_rand_pool=True) if miss_r == 0 else {"pass": False, "missing": miss_r}

    per_arm_rel: dict[str, Any] = {"Rand_Pirate_s0_s1_s2_pooled@d*_con": rand_rel}
    for arm_k in (
        "WSP_Pirate_Contrastive@d*_con",
        "WSP_Pirate_Contrastive@d_common",
        "WSP_Pirate_Lexicon@d_common",
    ):
        akeys = [f"pillar_a|common_pirate_ban|{arm_k}|{pid}" for pid in pa_ids]
        ya, yb, ma = _collect_pair(akeys)
        per_arm_rel[arm_k] = _eval_reader_arm_reliability(ya, yb, is_rand_pool=False) if ma == 0 else {"pass": False, "missing": ma}

    h1_pass = gold_ok and pool_ok and rand_rel["pass"] and per_arm_rel["WSP_Pirate_Contrastive@d*_con"]["pass"]
    h2_pass = (
        gold_ok
        and pool_ok
        and per_arm_rel["WSP_Pirate_Contrastive@d_common"]["pass"]
        and per_arm_rel["WSP_Pirate_Lexicon@d_common"]["pass"]
    )
    return {
        "pass": h1_pass and h2_pass,
        "h1_pass": h1_pass,
        "h2_pass": h2_pass,
        "gold_14": {"pass": gold_ok, "by_reader": gold_by_reader},
        "pool": {
            "pass": pool_ok,
            "complete": pool_complete,
            "expected_n_pooled": len(keys_pooled),
            "actual_n_pooled": len(y1_pool),
            "missing_pooled": miss_pool,
            "kappa_pooled": round(kap_pool, 6),
            "kappa_ban_none": round(kap_none, 6),
            "kappa_ban_common": round(kap_ban, 6),
        },
        "per_arm_common_ban": per_arm_rel,
    }


def evaluate_eq_c3_2(
    receipt: dict[str, Any],
    c3_constants: dict[str, Any],
    *,
    run_code_subprocess: bool = True,
) -> dict[str, Any]:
    """Recompute EQ-C3.2 (MATH_20, CODE_12, NEUTRAL_CHAT_16) directly from raw generated text across all arms."""
    bat = c3_constants["batteries"]
    math_exp = {p["id"]: int(p["answer"]) for p in bat["math_20"]}
    code_spec = {p["id"]: (p["tests"], p["func_name"]) for p in bat["code_12"]}
    cap_cfg = c3_constants["capability_preservation_gate_EQ_C3_2"]
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
            # DS F7 & GLM F5: never silently default missing code_pass_precomputed to False (0-vs-0 vacuous PASS)!
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

        arm_metrics[arm_label] = {
            "gating": arm_label in gating_set,
            "math_acc": round(math_acc, 6),
            "math_strict_final_acc": round(math_strict, 6),
            "code_pass_rate": round(code_acc, 6),
            "neutral_median_off_kl": round(neu_med_kl, 6),
            "neutral_broken_rate": round(neu_broken, 6),
            "neutral_greedy_match_rate": round(neu_match, 6),
            "refusal_keyword_rate": round(ref_rate, 6),
        }

    plain_m = arm_metrics["Plain"]
    for arm_label, am in arm_metrics.items():
        math_drop = plain_m["math_acc"] - am["math_acc"]
        code_drop = plain_m["code_pass_rate"] - am["code_pass_rate"]
        math_ok = math_drop <= float(cap_cfg["math_drop_max_pp"]) + 1e-12
        code_ok = code_drop <= float(cap_cfg["code_drop_max_pp"]) + 1e-12
        kl_ok = am["neutral_median_off_kl"] <= float(cap_cfg["neutral_median_off_kl_max"]) + 1e-12
        brk_ok = am["neutral_broken_rate"] <= float(cap_cfg["neutral_broken_max"]) + 1e-12
        am["math_drop_pp"] = round(math_drop, 6)
        am["code_drop_pp"] = round(code_drop, 6)
        am["math_ok"] = math_ok
        am["code_ok"] = code_ok
        am["neutral_kl_ok"] = kl_ok
        am["neutral_broken_ok"] = brk_ok
        am["pass"] = math_ok and code_ok and kl_ok and brk_ok
    return arm_metrics


# ---------------------------------------------------------------------------
# Main Confirmatory & Diagnostic Stage C3 Scorer
# ---------------------------------------------------------------------------
def score_stage_c3(
    receipt: dict[str, Any],
    c3_constants: dict[str, Any],
    *,
    reader_labels: dict[str, Any] | None = None,
    disk_hashes: dict[str, str] | None = None,
    receipt_sha256: str | None = None,
    strict_provenance: bool = True,
    allow_mock_readers: bool = False,
    n_perm: int = DEFAULT_N_PERM,
    n_boot: int = DEFAULT_N_BOOT,
    run_code_subprocess: bool = True,
) -> dict[str, Any]:
    """Score a complete Stage C3 (WSP v1) receipt strictly per OLMO3-WSP-C3-DESIGN.md."""
    if receipt.get("partial_checkpoint", False):
        raise ValueError("FAIL-CLOSED: Refusing to score a partial_checkpoint file.")
    if receipt.get("schema_version") != EXPECTED_RECEIPT_SCHEMA:
        raise ValueError(f"Unsupported receipt schema_version: {receipt.get('schema_version')!r}")
    meta = receipt.get("meta", {})
    if meta.get("smoke", False):
        raise ValueError("FAIL-CLOSED: Refusing to score a smoke/truncated Stage C3 receipt.")

    bat = c3_constants["batteries"]
    surface_regex = str(c3_constants["lexicons"]["common_pirate_ban"]["surface_stem_regex"])
    layers = [str(L) for L in c3_constants["edit_spec"]["layers"]]
    dose_ladder = [float(d) for d in c3_constants["edit_spec"]["dose_ladder_total_pmn_frac"]]

    # ------------------------------------------------------------------
    # GLOBAL GATE C3.0 (C3.0.1, C3.0.2, C3.0.3, C3.0.4) — Fail-Closed
    # ------------------------------------------------------------------
    def _blocked_c30_scorecard(gate_c30: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "confirmatory_verdicts": {
                "C3.0": "FAIL",
                "H1": "BLOCKED_BY_C3.0",
                "H2": "BLOCKED_BY_C3.0",
                "H3": "BLOCKED_BY_C3.0",
                "H4": "BLOCKED_BY_C3.0",
            },
            "gates": {
                "C3.0": gate_c30,
                "C3.0_direction_quality": {"pass_by_arm": {}},
                "C3.0_commit_gate": {},
                "C3.0_NoMD_room": {"pass": False, "plain_has_markdown": 0.0, "plain_mean_construct_types": 0.0},
                "C3.R": {"pass": False, "h1_pass": False, "h2_pass": False, "reason": "blocked_by_c3_0"},
                "EQ_C3_2": {},
            },
            "holm_4_family": {
                "alpha": 0.05,
                "n_perm": n_perm,
                "n_boot": n_boot,
                "p_values": {},
                "holm_rejected": {},
                "hypotheses": {
                    hid: {"verdict": "BLOCKED_BY_C3.0", "p_perm": 1.0, "holm_sig": False}
                    for hid in ("H1", "H2", "H3", "H4")
                },
            },
            "diagnostics": {},
        }

    nonfinite_paths = audit_finiteness(receipt)

    try:
        # C3.0.1: zero hooks, bitwise restore, sham(0.05), liveness, B=8 parity / B=1 fallback, retention logged
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
        sham_ok = (
            sham_ret_ok
            and math.isfinite(sham_dlogit)
            and (0.0 < sham_dlogit < 0.25)
            and (sham_min_proj >= 0.25)
        )
        doses = receipt["doses"]
        expected_rand_keys = {
            f"Rand_Pirate_s{s}@d*_con" for s in range(3)
        } | {
            f"Rand_NoMD_s{s}@d*_nomd_con" for s in range(3)
        }
        if doses["d_star_pirate_lex"] != doses["d_star_pirate_con"]:
            expected_rand_keys |= {f"Rand_Pirate_s{s}@d*_lex" for s in range(3)}
        if doses["d_star_nomd_lex"] != doses["d_star_nomd_con"]:
            expected_rand_keys |= {f"Rand_NoMD_s{s}@d*_nomd_lex" for s in range(3)}

        expected_liveness_keys = {
            f"WSP_Pirate_Contrastive@d={float(doses['d_star_pirate_con'])}",
            f"WSP_Pirate_Lexicon@d={float(doses['d_star_pirate_lex'])}",
            f"WSP_NoMD_Contrastive@d={float(doses['d_star_nomd_con'])}",
            f"WSP_NoMD_Lexicon@d={float(doses['d_star_nomd_lex'])}",
        } | expected_rand_keys
        liveness_map = pref.get("active_arm_max_abs_dlogit", {})
        liveness_keys_ok = expected_liveness_keys.issubset(set(liveness_map.keys()))
        # GLM F8: check liveness over the registered active selected-d* & matched random arms (not dead unselected rungs)
        liveness_ok = liveness_keys_ok and all(float(liveness_map[k]) > 0.0 for k in expected_liveness_keys)

        # DS F2 & GLM F8: never pass a skipped B=8 parity preflight; cross-check recorded max_abs_dlogit <= 0.05!
        b8_passed = meta.get("b8_parity_passed")
        b8_dlogit = meta.get("b8_parity_max_abs_dlogit")
        req_bsz = int(meta.get("requested_gen_batch_size", 0))
        eff_bsz = int(meta.get("effective_gen_batch_size", 0))
        if b8_passed is True:
            b8_ok = (req_bsz > 1) and (b8_dlogit is not None) and math.isfinite(float(b8_dlogit)) and (float(b8_dlogit) <= 0.05)
        elif b8_passed is False:
            b8_ok = (req_bsz > 1) and (b8_dlogit is not None) and math.isfinite(float(b8_dlogit)) and (float(b8_dlogit) > 0.05) and (eff_bsz == 1)
        else:
            b8_ok = False

        # DS F5: derive exact minimum expected restore_checks from the receipt's own edited-arm applications!
        d_con_v = float(doses["d_star_pirate_con"])
        d_lex_v = float(doses["d_star_pirate_lex"])
        d_com_v = float(doses["d_common"])
        d_nc_v = float(doses["d_star_nomd_con"])
        d_nl_v = float(doses["d_star_nomd_lex"])
        n_dcom_distinct = (1 if not math.isclose(d_com_v, d_con_v, abs_tol=1e-9) else 0) + (
            1 if not math.isclose(d_com_v, d_lex_v, abs_tol=1e-9) else 0
        )
        n_rand_expected = len(expected_rand_keys)
        expected_min_restore_checks = (
            3  # sham (3 layers)
            + 3  # D4 (3 conditions x L24)
            + 4 * len(dose_ladder) * 3  # commit_gate (4 arms x 6 rungs x 3 layers)
            + 3 * n_rand_expected  # rand_reserve
            + 3 * n_dcom_distinct  # d_common_retention
            + 2 * 3 * (5 + (3 if not math.isclose(d_con_v, d_lex_v, abs_tol=1e-9) else 0) + n_dcom_distinct)  # pillar_a x 2 bans
            + 3 * 3  # D5_multiturn (3 edited arms)
            + (3 * 3 + 4 * 3)  # D6 own_top50 (3) + edited gens (4)
            + 3 * 3  # D1_copula (3 edited arms)
            + 3 * (5 + (3 if not math.isclose(d_nc_v, d_nl_v, abs_tol=1e-9) else 0))  # pillar_b
            + 3 * (6 + n_dcom_distinct)  # capability_suite
            + 2 * 3  # D2_sibling_bleed
        )
        hooks_ok = (int(meta.get("active_hooks_max_seen", -1)) == 0)
        restore_ok = bool(meta.get("restore_exact_all", False)) and (int(meta.get("restore_checks", 0)) >= expected_min_restore_checks)
        no_weights_saved_ok = (int(meta.get("weight_checkpoints_saved", -1)) == 0)
        c3_0_1_pass = (
            (not nonfinite_paths)
            and hooks_ok
            and restore_ok
            and no_weights_saved_ok
            and sham_ok
            and liveness_ok
            and b8_ok
        )

        # C3.0.2: finiteness, Rand intended Frob match <= 1e-3, D4 mask > 1.15 & mis-mask < 1.05, CV <= 1.0,
        #         Rand reserve median off_KL <= 0.35 (all 6 core + conditional distinct d*),
        #         per-arm banned surface_stem_hit/N <= 0.10 + stem preflight + code_pass positive control (DS F7/GLM F5)
        d4 = pref["D4_L24_mask_mismask"]
        d4_mask_ratio = float(d4["masked_over_unmasked_logit_l2_ratio"])
        d4_mismask_ratio = float(d4["mismasked_over_unmasked_logit_l2_ratio"])
        d4_ok = (d4_mask_ratio > 1.15) and (d4_mismask_ratio < 1.05)

        cal_layers = receipt["calibration"]["layers"]
        cv_ok = all(float(cal_layers[L]["cv"]) <= 1.0 for L in layers)

        rand_res = receipt["rand_reserve"]
        rand_keys_ok = expected_rand_keys.issubset(set(rand_res.keys()))
        rand_norm_ok = rand_keys_ok and all(
            all(abs(float(r_info["intended_frob_ratio_vs_edit"][L]) - 1.0) <= 1e-3 for L in layers)
            for r_info in rand_res.values()
        )
        rand_kl_ok = rand_keys_ok and all(
            float(r_info["median_off_kl"]) <= 0.35 for r_info in rand_res.values()
        )

        stem_pref = pref["surface_stem_audit_positive_check"]
        stem_pref_ok = bool(stem_pref.get("positive_sample_hit", False)) and (not bool(stem_pref.get("negative_sample_hit", True)))
        code_harness_preflight_ok = (
            c3_scoring.code_pass("def _c3_pos(a, b):\n    return a + b\n", ["assert _c3_pos(2, 3) == 5"], "_c3_pos")
            and (not c3_scoring.code_pass("def _c3_pos(a, b):\n    return a - b\n", ["assert _c3_pos(2, 3) == 5"], "_c3_pos"))
        )
        ban_stem_rates: dict[str, float] = {}
        for arm_k, rows in receipt["pillar_a"]["common_pirate_ban"].items():
            hits = sum(1 for r in rows if c3_scoring.surface_stem_hit(r["text"], surface_regex))
            ban_stem_rates[arm_k] = round(hits / max(len(rows), 1), 6)
        ban_stem_ok = bool(ban_stem_rates) and all(rate <= 0.10 + 1e-12 for rate in ban_stem_rates.values())

        c3_0_2_pass = (
            (not nonfinite_paths)
            and d4_ok
            and cv_ok
            and rand_norm_ok
            and rand_kl_ok
            and stem_pref_ok
            and code_harness_preflight_ok
            and ban_stem_ok
        )

        # C3.0.3: Exact required arms, exact row counts, EXACT battery ID order (DS F4 / GLM F2),
        #         non-empty strings + Plain non-broken >= 0.90 on Id(20)+Imp(20)+Neu(16)
        counts_ok = True
        ids_exact_ok = True
        nonempty_ok = True
        required_arms_ok = True

        pa_ids_exp = [p["id"] for p in bat["identity_20"] + bat["implied_20"]]
        id_ids_exp = [p["id"] for p in bat["identity_20"]]
        nomd_ids_exp = [p["id"] for p in bat["nomd_24"]]
        mt_ids_exp = [p["id"] for p in bat["multiturn_12"]]
        demo_ids_exp = list(bat["demo_showcase_ids_12"])
        cop_ids_exp = [p["id"] for p in bat["copula_pairs_12"]]
        math_ids_exp = [p["id"] for p in bat["math_20"]]
        code_ids_exp = [p["id"] for p in bat["code_12"]]
        ref_ids_exp = [p["id"] for p in bat["refusal_8"]]
        neu_ids_exp = [p["id"] for p in bat["neutral_chat_16"]]
        res_ids_exp = [p["id"] for p in bat["reserve_8"]]

        req_pa_arms = set(c3_constants["reader_kappa_pool"]["core_arms_7"]) | {
            "WSP_Pirate_Contrastive@d_common", "WSP_Pirate_Lexicon@d_common",
        }
        req_pb_arms = {
            "Plain", "Sys_NoMD", "WSP_NoMD_Contrastive@d*_nomd_con", "WSP_NoMD_Lexicon@d*_nomd_lex",
            "Rand_NoMD_s0@d*_nomd_con", "Rand_NoMD_s1@d*_nomd_con", "Rand_NoMD_s2@d*_nomd_con",
        }
        if doses["d_star_nomd_lex"] != doses["d_star_nomd_con"]:
            req_pb_arms |= {f"Rand_NoMD_s{s}@d*_nomd_lex" for s in range(3)}
        req_cap_arms = set(c3_constants["capability_suite_arms_9"]) | set(c3_constants["capability_suite_conditional_d_common_2"])
        req_d5_arms = set(c3_constants["multiturn_d5_arms_5"])
        req_d6_conds = {"Plain@common", "Con@own50+common", "Con@donor_rand0+common", "Lex@own50+common", "Rand0@own50+common"}
        req_d1_arms = {"Plain", "Sys_Pirate", "WSP_Pirate_Contrastive@d*_con", "WSP_Pirate_Lexicon@d*_lex", "Rand_Pirate_s0@d*_con"}

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
        if not req_d6_conds.issubset(set(receipt["D6_adaptive_donor"]["conditions"].keys())):
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
        for rows in receipt["D6_adaptive_donor"]["conditions"].values():
            _check_rows(rows, demo_ids_exp)
        for rows in receipt["D1_copula"]["arms"].values():
            _check_rows(rows, cop_ids_exp, check_text=False)
        for blk in receipt["capability_suite"].values():
            _check_rows(blk["math_20"], math_ids_exp)
            _check_rows(blk["code_12"], code_ids_exp)
            _check_rows(blk["refusal_8"], ref_ids_exp)
            _check_rows(blk["neutral_chat_16"], neu_ids_exp)
        for r_info in rand_res.values():
            _check_rows(r_info["per_prompt_kl"], res_ids_exp, check_text=False)
        _check_rows(receipt["D2_sibling_bleed"]["WSP_Pirate_Contrastive@d*_con_on_NOMD_24"], nomd_ids_exp)
        _check_rows(receipt["D2_sibling_bleed"]["WSP_NoMD_Contrastive@d*_nomd_con_on_IDENTITY_20"], id_ids_exp)

        plain_sanity_texts = (
            [r["text"] for r in receipt["pillar_a"]["none"]["Plain"]]
            + [r["text"] for r in receipt["capability_suite"]["Plain"]["neutral_chat_16"]]
        )
        plain_non_broken_rate = mean([0.0 if c3_scoring.is_broken(t) else 1.0 for t in plain_sanity_texts]) if plain_sanity_texts else 0.0
        plain_sanity_ok = (len(plain_sanity_texts) == 56) and (plain_non_broken_rate >= 0.90)

        # GLM F1 & DS F2 (R2): verify doses.d_common == min(d*_con, d*_lex) and both d_common_retention keys present!
        d_common_expected_c30 = min(d_con_v, d_lex_v)
        d_common_consistent_ok = math.isclose(d_com_v, d_common_expected_c30, abs_tol=1e-9)
        dc_ret_map = doses.get("d_common_retention")
        d_common_retention_present_ok = (
            isinstance(dc_ret_map, dict)
            and all(
                isinstance(dc_ret_map.get(k), dict)
                and all(
                    isinstance(dc_ret_map[k].get(L), dict)
                    and "stored_rank1_proj_ret" in dc_ret_map[k][L]
                    and "stored_frob_ret" in dc_ret_map[k][L]
                    for L in layers
                )
                for k in ("WSP_Pirate_Contrastive@d_common", "WSP_Pirate_Lexicon@d_common")
            )
        )
        c3_0_3_pass = (
            required_arms_ok
            and counts_ok
            and ids_exact_ok
            and nonempty_ok
            and plain_sanity_ok
            and d_common_consistent_ok
            and d_common_retention_present_ok
        )

        # C3.0.4: Strict SHA-256 provenance of all 4 committed artifacts
        ph = meta.get("provenance_hashes", {})
        exp_hashes = {
            "c3_design_sha256": EXPECTED_C3_DESIGN_SHA256,
            "c3_constants_sha256": EXPECTED_C3_CONSTANTS_SHA256,
            "c3_bank_sha256": EXPECTED_C3_BANK_SHA256,
            "c3_scoring_sha256": EXPECTED_C3_SCORING_SHA256,
        }
        receipt_prov_ok = all(ph.get(k) == v for k, v in exp_hashes.items())
        disk_prov_ok = True if not strict_provenance else (
            disk_hashes is not None and all(disk_hashes.get(k) == v for k, v in exp_hashes.items())
        )
        c3_0_4_pass = (not strict_provenance) or (receipt_prov_ok and disk_prov_ok)

        c3_0_pass = c3_0_1_pass and c3_0_2_pass and c3_0_3_pass and c3_0_4_pass
        gate_c30_doc = {
            "pass": c3_0_pass,
            "C3.0.1": {
                "pass": c3_0_1_pass,
                "hooks_ok": hooks_ok,
                "restore_ok": restore_ok,
                "restore_checks": int(meta.get("restore_checks", 0)),
                "expected_min_restore_checks": expected_min_restore_checks,
                "no_weights_saved_ok": no_weights_saved_ok,
                "sham_ok": sham_ok,
                "sham_max_abs_dlogit": round(sham_dlogit, 6) if math.isfinite(sham_dlogit) else None,
                "sham_min_proj_ret": round(sham_min_proj, 6) if math.isfinite(sham_min_proj) else None,
                "liveness_ok": liveness_ok,
                "b8_parity_or_b1_fallback_ok": b8_ok,
            },
            "C3.0.2": {
                "pass": c3_0_2_pass,
                "nonfinite_count": len(nonfinite_paths),
                "nonfinite_sample": nonfinite_paths[:5],
                "d4_ok": d4_ok,
                "d4_masked_over_unmasked": round(d4_mask_ratio, 6) if math.isfinite(d4_mask_ratio) else None,
                "d4_mismasked_over_unmasked": round(d4_mismask_ratio, 6) if math.isfinite(d4_mismask_ratio) else None,
                "cv_ok": cv_ok,
                "rand_keys_ok": rand_keys_ok,
                "rand_norm_ok": rand_norm_ok,
                "rand_kl_ok": rand_kl_ok,
                "stem_preflight_ok": stem_pref_ok,
                "code_harness_preflight_ok": code_harness_preflight_ok,
                "ban_stem_ok": ban_stem_ok,
                "ban_stem_rates": ban_stem_rates,
            },
            "C3.0.3": {
                "pass": c3_0_3_pass,
                "required_arms_ok": required_arms_ok,
                "counts_ok": counts_ok,
                "ids_exact_ok": ids_exact_ok,
                "nonempty_ok": nonempty_ok,
                "plain_non_broken_rate": round(plain_non_broken_rate, 6),
                "plain_sanity_ok": plain_sanity_ok,
                "d_common_consistent_ok": d_common_consistent_ok,
                "d_common_retention_present_ok": d_common_retention_present_ok,
            },
            "C3.0.4": {
                "pass": c3_0_4_pass,
                "strict_provenance": strict_provenance,
                "receipt_prov_ok": receipt_prov_ok,
                "disk_prov_ok": disk_prov_ok,
            },
        }
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        return _blocked_c30_scorecard({"pass": False, "schema_error": f"{type(exc).__name__}: {exc}"})

    # Short-circuit cleanly on C3.0 failure (never crash on NaNs, missing rows, or corrupted cells!)
    if not c3_0_pass:
        return _blocked_c30_scorecard(gate_c30_doc)

    # ------------------------------------------------------------------
    # Per-Arm Gate 5: C3.0_direction_quality (cos_ret >= 0.97 on L21/24/27)
    # ------------------------------------------------------------------
    dq_raw = receipt["calibration"]["direction_quality"]
    dir_valid: dict[str, bool] = {}
    dir_min_cos: dict[str, float] = {}
    for arm_name in ("WSP_Pirate_Contrastive", "WSP_Pirate_Lexicon", "WSP_NoMD_Contrastive", "WSP_NoMD_Lexicon"):
        cvals = [float(dq_raw[arm_name][L]["cos_ret"]) for L in layers]
        dir_min_cos[arm_name] = round(min(cvals), 6)
        dir_valid[arm_name] = all(c >= 0.97 for c in cvals)
    cos_ret_diff_con_lex = {
        L: round(abs(float(dq_raw["WSP_Pirate_Contrastive"][L]["cos_ret"]) - float(dq_raw["WSP_Pirate_Lexicon"][L]["cos_ret"])), 6)
        for L in layers
    }

    # ------------------------------------------------------------------
    # Per-Arm Gate 6: C3.0_commit_gate (Recomputed from raw reserve rows!)
    # ------------------------------------------------------------------
    dose_valid: dict[str, bool] = {}
    commit_audit: dict[str, Any] = {}
    for arm_name in ("WSP_Pirate_Contrastive", "WSP_Pirate_Lexicon", "WSP_NoMD_Contrastive", "WSP_NoMD_Lexicon"):
        cg = receipt["commit_gate"]["arms"][arm_name]
        rungs = cg["rungs"]
        ladder_exact_ok = (len(rungs) == len(dose_ladder)) and all(
            math.isclose(float(rr["dose"]), exp_d, abs_tol=1e-9) for rr, exp_d in zip(rungs, dose_ladder)
        )
        recomputed_passing: list[float] = []
        rung_checks: list[dict[str, Any]] = []
        for rr in rungs:
            ret_m = rr["bf16_retention"]
            rep_ok = all(
                float(ret_m[L]["stored_rank1_proj_ret"]) >= 0.50 and float(ret_m[L]["stored_frob_ret"]) >= 0.80
                for L in layers
            )
            p_kls = [float(x["off_kl"]) for x in rr["per_prompt_kl"]]
            p_max_pos = [float(x["max_single_pos_kl"]) for x in rr["per_prompt_kl"]]
            reserve_n_ok = (len(p_kls) == 8) and (len(rr["reserve_generations"]) == 8)
            med_kl = median(p_kls)
            max_kl = max(p_kls)
            max_pos = max(p_max_pos)
            n_brk = sum(1 for x in rr["reserve_generations"] if c3_scoring.is_broken(x["text"]))
            ok = reserve_n_ok and rep_ok and (med_kl <= 0.25) and (max_kl <= 0.60) and (max_pos <= 4.0) and (n_brk == 0)
            if ok:
                recomputed_passing.append(float(rr["dose"]))
            rung_checks.append({
                "dose": float(rr["dose"]),
                "rep_ok": rep_ok,
                "reserve_n_ok": reserve_n_ok,
                "median_off_kl": round(med_kl, 6),
                "max_prompt_off_kl": round(max_kl, 6),
                "max_single_pos_kl": round(max_pos, 6),
                "n_broken": n_brk,
                "pass": ok,
            })
        expected_d = recomputed_passing[-1] if recomputed_passing else None
        fallback = bool(cg["commit_gate_fallback"])
        chosen_d = float(cg["selected_d_star"])
        d_ok = ladder_exact_ok and (expected_d is not None) and (not fallback) and math.isclose(chosen_d, expected_d, abs_tol=1e-9)
        dose_valid[arm_name] = d_ok
        commit_audit[arm_name] = {
            "pass": d_ok,
            "ladder_exact_ok": ladder_exact_ok,
            "selected_d_star": chosen_d,
            "expected_d_star": expected_d,
            "commit_gate_fallback": fallback,
            "rungs": rung_checks,
        }

    # Amendment 1 (GLM F1 / DS F2 R2): derive d_common = min(d*_con, d*_lex) from commit-gate-verified doses
    # and require BOTH commit_gate.arms[*].rungs AND mandatory doses.d_common_retention keys to pass!
    cg_d_con = commit_audit["WSP_Pirate_Contrastive"]["expected_d_star"]
    cg_d_lex = commit_audit["WSP_Pirate_Lexicon"]["expected_d_star"]
    d_common_derived = min(float(cg_d_con), float(cg_d_lex)) if (cg_d_con is not None and cg_d_lex is not None) else None
    d_common_reported = float(receipt["doses"]["d_common"])
    d_common_matches_derived = (d_common_derived is not None) and math.isclose(d_common_reported, d_common_derived, abs_tol=1e-9)
    d_common_ret_logged = receipt["doses"].get("d_common_retention", {})
    h2_d_common_rep_by_arm: dict[str, bool] = {}
    h2_d_common_ret_table: dict[str, Any] = {}
    for arm_name, dc_key in (
        ("WSP_Pirate_Contrastive", "WSP_Pirate_Contrastive@d_common"),
        ("WSP_Pirate_Lexicon", "WSP_Pirate_Lexicon@d_common"),
    ):
        r_match = None
        if d_common_derived is not None:
            for rr in receipt["commit_gate"]["arms"][arm_name]["rungs"]:
                if math.isclose(float(rr["dose"]), d_common_derived, abs_tol=1e-9):
                    r_match = rr
                    break
        dc_m = d_common_ret_logged.get(dc_key) if isinstance(d_common_ret_logged, dict) else None
        if r_match is None or not isinstance(dc_m, dict):
            h2_d_common_rep_by_arm[arm_name] = False
            h2_d_common_ret_table[arm_name] = None
        else:
            rm = r_match["bf16_retention"]
            rep_rung = all(
                float(rm[L]["stored_rank1_proj_ret"]) >= 0.50 and float(rm[L]["stored_frob_ret"]) >= 0.80
                for L in layers
            )
            rep_dc = all(
                isinstance(dc_m.get(L), dict)
                and float(dc_m[L]["stored_rank1_proj_ret"]) >= 0.50
                and float(dc_m[L]["stored_frob_ret"]) >= 0.80
                for L in layers
            )
            h2_d_common_rep_by_arm[arm_name] = rep_rung and rep_dc
            h2_d_common_ret_table[arm_name] = rm
    h2_d_common_rep_ok = d_common_matches_derived and all(h2_d_common_rep_by_arm.values())

    # ------------------------------------------------------------------
    # Pillar-B Gate 7: C3.0_NoMD_room (Recomputed via c3_scoring.py)
    # ------------------------------------------------------------------
    plain_nomd_rows = receipt["pillar_b"]["Plain"]
    plain_nomd_cts = [c3_scoring.construct_types(r["text"]) for r in plain_nomd_rows]
    plain_nomd_has_md = mean([1.0 if c >= 1 else 0.0 for c in plain_nomd_cts])
    plain_nomd_mean_ct = mean([float(c) for c in plain_nomd_cts])
    nomd_room_pass = (plain_nomd_has_md >= 0.65 - 1e-12) and (plain_nomd_mean_ct >= 1.50 - 1e-12)
    nomd_room_doc = {
        "pass": nomd_room_pass,
        "plain_has_markdown": round(plain_nomd_has_md, 6),
        "plain_mean_construct_types": round(plain_nomd_mean_ct, 6),
        "plain_n_has_markdown": sum(1 for c in plain_nomd_cts if c >= 1),
    }

    # ------------------------------------------------------------------
    # Reader Gate 8: C3.R (Reader-Dependent Only)
    # ------------------------------------------------------------------
    c3_r_doc = evaluate_c3_r_gate(
        receipt, c3_constants, reader_labels,
        receipt_sha256=receipt_sha256,
        disk_hashes=disk_hashes,
        strict_provenance=strict_provenance,
        allow_mock_readers=allow_mock_readers,
    )

    # ------------------------------------------------------------------
    # Capability Preservation Gate: EQ-C3.2
    # ------------------------------------------------------------------
    eq_c3_2_doc = evaluate_eq_c3_2(receipt, c3_constants, run_code_subprocess=run_code_subprocess)

    # ------------------------------------------------------------------
    # Confirmatory Holm-4 Statistics (H1, H2, H3, H4 — ID-indexed, pinned seeds!)
    # ------------------------------------------------------------------
    labels_map = (reader_labels or {}).get("labels", {})
    pa_ids = [p["id"] for p in bat["identity_20"] + bat["implied_20"]]
    nomd_ids = [p["id"] for p in bat["nomd_24"]]
    id_dir_ids = {p["id"] for p in bat["identity_20"] if p.get("subtype") == "direct"}
    id_ind_ids = {p["id"] for p in bat["identity_20"] if p.get("subtype") == "indirect"}
    imp_ids = {p["id"] for p in bat["implied_20"]}

    def _strict_pirate_by_id(arm_k: str, ban_k: str) -> tuple[dict[str, float] | None, dict[str, bool], dict[str, Any]]:
        rows_by_id = {r["id"]: r for r in receipt["pillar_a"][ban_k][arm_k]}
        broken_by_id = {pid: c3_scoring.is_broken(rows_by_id[pid]["text"]) for pid in pa_ids}
        stem_by_id = {pid: c3_scoring.surface_stem_hit(rows_by_id[pid]["text"], surface_regex) for pid in pa_ids}
        tok_lens = [int(rows_by_id[pid]["n_tokens"]) for pid in pa_ids]
        brk_rate = round(mean([1.0 if broken_by_id[pid] else 0.0 for pid in pa_ids]), 6)
        if reader_labels is None:
            return None, broken_by_id, {"broken_rate": brk_rate, "mean_tokens": round(mean([float(x) for x in tok_lens]), 3)}
        sp_by_id: dict[str, float] = {}
        coh_both: list[float] = []
        for pid in pa_ids:
            k = f"pillar_a|{ban_k}|{arm_k}|{pid}"
            lrec = labels_map.get(k)
            if not lrec or not lrec.get("ds") or not lrec.get("luna"):
                return None, broken_by_id, {"broken_rate": brk_rate, "missing_label": k}
            p1 = lrec["ds"].get("pirate_persona")
            p2 = lrec["luna"].get("pirate_persona")
            c1 = lrec["ds"].get("coherent")
            c2 = lrec["luna"].get("coherent")
            if p1 is None or p2 is None:
                return None, broken_by_id, {"broken_rate": brk_rate, "missing_label": k}
            ok = bool(p1) and bool(p2) and (not broken_by_id[pid]) and (ban_k == "none" or (not stem_by_id[pid]))
            sp_by_id[pid] = 1.0 if ok else 0.0
            coh_both.append(1.0 if (bool(c1) and bool(c2)) else 0.0)
        return sp_by_id, broken_by_id, {
            "strict_pirate_rate": round(mean([sp_by_id[pid] for pid in pa_ids]), 6),
            "broken_rate": brk_rate,
            "surface_stem_hit_rate": round(mean([1.0 if stem_by_id[pid] else 0.0 for pid in pa_ids]), 6),
            "reader_both_coherent_rate": round(mean(coh_both), 6),
            "mean_tokens": round(mean([float(x) for x in tok_lens]), 3),
            "median_tokens": round(median([float(x) for x in tok_lens]), 3),
        }

    # Pillar A statistics (H1 & H2, ID-paired, pinned PERM_SEED / BCA_SEED per GLM F3!)
    sp_con_ban, brk_con_ban, sum_con_ban = _strict_pirate_by_id("WSP_Pirate_Contrastive@d*_con", "common_pirate_ban")
    sp_rands_ban = [_strict_pirate_by_id(f"Rand_Pirate_s{s}@d*_con", "common_pirate_ban") for s in range(3)]
    sp_con_dc_ban, brk_con_dc_ban, sum_con_dc_ban = _strict_pirate_by_id("WSP_Pirate_Contrastive@d_common", "common_pirate_ban")
    sp_lex_dc_ban, brk_lex_dc_ban, sum_lex_dc_ban = _strict_pirate_by_id("WSP_Pirate_Lexicon@d_common", "common_pirate_ban")

    h1_diffs: list[float] | None = None
    h1_p = 1.0
    h1_ci: dict[str, float] | None = None
    if sp_con_ban is not None and all(x[0] is not None for x in sp_rands_ban):
        h1_diffs = [
            sp_con_ban[pid] - mean([sp_rands_ban[s][0][pid] for s in range(3)])  # type: ignore[index]
            for pid in pa_ids
        ]
        h1_p = paired_sign_flip_p_one_sided(h1_diffs, n_perm=n_perm, seed=PERM_SEED)
        h1_ci = bca_paired_mean_ci90(h1_diffs, n_boot=n_boot, seed=BCA_SEED)

    h2_diffs: list[float] | None = None
    h2_p = 1.0
    h2_ci: dict[str, float] | None = None
    if sp_con_dc_ban is not None and sp_lex_dc_ban is not None:
        h2_diffs = [sp_con_dc_ban[pid] - sp_lex_dc_ban[pid] for pid in pa_ids]
        h2_p = paired_sign_flip_p_one_sided(h2_diffs, n_perm=n_perm, seed=PERM_SEED)
        h2_ci = bca_paired_mean_ci90(h2_diffs, n_boot=n_boot, seed=BCA_SEED)

    # Pillar B statistics (H3 & H4 — 100% mechanical, ID-paired, pinned PERM_SEED / BCA_SEED!)
    def _nomd_arm_stats(arm_k: str, rand_suffix: str) -> dict[str, Any]:
        arm_by_id = {r["id"]: r for r in receipt["pillar_b"][arm_k]}
        rand_by_id_3 = [{r["id"]: r for r in receipt["pillar_b"][f"Rand_NoMD_s{s}@{rand_suffix}"]} for s in range(3)]
        arm_cts = {pid: c3_scoring.construct_types(arm_by_id[pid]["text"]) for pid in nomd_ids}
        rand_cts = [{pid: c3_scoring.construct_types(rand_by_id_3[s][pid]["text"]) for pid in nomd_ids} for s in range(3)]
        u_diffs = [mean([float(rand_cts[s][pid]) for s in range(3)]) - float(arm_cts[pid]) for pid in nomd_ids]

        has_md_arm = mean([1.0 if arm_cts[pid] >= 1 else 0.0 for pid in nomd_ids])
        has_md_rand = mean([mean([1.0 if rand_cts[s][pid] >= 1 else 0.0 for s in range(3)]) for pid in nomd_ids])
        brk_arm = mean([1.0 if c3_scoring.is_broken(arm_by_id[pid]["text"]) else 0.0 for pid in nomd_ids])
        brk_rand = mean([mean([1.0 if c3_scoring.is_broken(rand_by_id_3[s][pid]["text"]) else 0.0 for s in range(3)]) for pid in nomd_ids])

        tok_arm = mean([float(arm_by_id[pid]["n_tokens"]) for pid in nomd_ids])
        tok_rand = mean([mean([float(rand_by_id_3[s][pid]["n_tokens"]) for s in range(3)]) for pid in nomd_ids])
        len_ratio = tok_arm / max(tok_rand, 1e-12)
        return {
            "u_diffs": u_diffs,
            "mean_u": mean(u_diffs),
            "mean_ct_arm": mean([float(arm_cts[pid]) for pid in nomd_ids]),
            "mean_ct_rand": mean([mean([float(rand_cts[s][pid]) for s in range(3)]) for pid in nomd_ids]),
            "has_md_arm": has_md_arm,
            "has_md_rand": has_md_rand,
            "delta_has_md_vs_plain": plain_nomd_has_md - has_md_arm,
            "delta_has_md_vs_rand": has_md_rand - has_md_arm,
            "broken_arm": brk_arm,
            "broken_rand": brk_rand,
            "mean_tokens_arm": tok_arm,
            "mean_tokens_rand": tok_rand,
            "length_ratio": len_ratio,
        }

    d_nomd_con = receipt["doses"]["d_star_nomd_con"]
    d_nomd_lex = receipt["doses"]["d_star_nomd_lex"]
    lex_rand_suffix = "d*_nomd_con" if d_nomd_con == d_nomd_lex else "d*_nomd_lex"

    h3_st = _nomd_arm_stats("WSP_NoMD_Contrastive@d*_nomd_con", "d*_nomd_con")
    h4_st = _nomd_arm_stats("WSP_NoMD_Lexicon@d*_nomd_lex", lex_rand_suffix)

    h3_p = paired_sign_flip_p_one_sided(h3_st["u_diffs"], n_perm=n_perm, seed=PERM_SEED)
    h3_ci = bca_paired_mean_ci90(h3_st["u_diffs"], n_boot=n_boot, seed=BCA_SEED)
    h4_p = paired_sign_flip_p_one_sided(h4_st["u_diffs"], n_perm=n_perm, seed=PERM_SEED)
    h4_ci = bca_paired_mean_ci90(h4_st["u_diffs"], n_boot=n_boot, seed=BCA_SEED)

    # Exact Holm-4 step-down via frozen c3_scoring.holm_reject()!
    pvals = {"H1": h1_p, "H2": h2_p, "H3": h3_p, "H4": h4_p}
    holm_decisions = c3_scoring.holm_reject(pvals, alpha=0.05)

    # Resolve all 4 verdicts strictly via frozen c3_scoring.resolve_verdict()!
    brk_h1_con = mean([1.0 if brk_con_ban[pid] else 0.0 for pid in pa_ids])
    brk_h1_rand = mean([mean([1.0 if sp_rands_ban[s][1][pid] else 0.0 for s in range(3)]) for pid in pa_ids])
    h1_mean_y = mean(h1_diffs) if h1_diffs is not None else 0.0
    v_h1 = c3_scoring.resolve_verdict(
        c3_0_pass=c3_0_pass,
        direction_valid=dir_valid["WSP_Pirate_Contrastive"],
        dose_valid=dose_valid["WSP_Pirate_Contrastive"],
        requires_nomd_room=False,
        nomd_room_pass=True,
        reader_dependent=True,
        c3_r_pass=bool(c3_r_doc["h1_pass"]),
        eq_c3_2_pass=bool(eq_c3_2_doc["WSP_Pirate_Contrastive@d*_con"]["pass"]),
        broken_bar_pass=(brk_h1_con <= 0.15 + 1e-12) and (brk_h1_rand <= 0.15 + 1e-12),
        holm_sig=holm_decisions["H1"],
        primary_effect_bar_pass=(h1_mean_y >= 0.20 - 1e-12),
        co_primary_bars_pass=True,
    )

    brk_h2_con = mean([1.0 if brk_con_dc_ban[pid] else 0.0 for pid in pa_ids])
    brk_h2_lex = mean([1.0 if brk_lex_dc_ban[pid] else 0.0 for pid in pa_ids])
    h2_mean_y = mean(h2_diffs) if h2_diffs is not None else 0.0
    v_h2 = c3_scoring.resolve_verdict(
        c3_0_pass=c3_0_pass,
        direction_valid=dir_valid["WSP_Pirate_Contrastive"] and dir_valid["WSP_Pirate_Lexicon"],
        # Amendment 1 (GLM F1): both arms' d_common rungs must also be bf16-representable!
        dose_valid=dose_valid["WSP_Pirate_Contrastive"] and dose_valid["WSP_Pirate_Lexicon"] and h2_d_common_rep_ok,
        requires_nomd_room=False,
        nomd_room_pass=True,
        reader_dependent=True,
        c3_r_pass=bool(c3_r_doc["h2_pass"]),
        # F2: EQ-C3.2 gating on Con@d_common ONLY (Lex@d_common is logged-only!)
        eq_c3_2_pass=bool(eq_c3_2_doc["WSP_Pirate_Contrastive@d_common"]["pass"]),
        broken_bar_pass=(brk_h2_con <= 0.15 + 1e-12) and (brk_h2_lex <= 0.15 + 1e-12),
        holm_sig=holm_decisions["H2"],
        primary_effect_bar_pass=(h2_mean_y >= 0.20 - 1e-12),
        co_primary_bars_pass=True,
    )

    h3_coprimary_ok = (
        (h3_st["delta_has_md_vs_plain"] >= 0.20 - 1e-12)
        and (h3_st["delta_has_md_vs_rand"] >= 0.15 - 1e-12)
        and (0.85 - 1e-12 <= h3_st["length_ratio"] <= 1.15 + 1e-12)
    )
    v_h3 = c3_scoring.resolve_verdict(
        c3_0_pass=c3_0_pass,
        direction_valid=dir_valid["WSP_NoMD_Contrastive"],
        dose_valid=dose_valid["WSP_NoMD_Contrastive"],
        requires_nomd_room=True,
        nomd_room_pass=nomd_room_pass,
        reader_dependent=False,
        c3_r_pass=True,
        eq_c3_2_pass=bool(eq_c3_2_doc["WSP_NoMD_Contrastive@d*_nomd_con"]["pass"]),
        broken_bar_pass=(h3_st["broken_arm"] <= 0.10 + 1e-12) and (h3_st["broken_rand"] <= 0.10 + 1e-12),
        holm_sig=holm_decisions["H3"],
        primary_effect_bar_pass=(h3_st["mean_u"] >= 0.50 - 1e-12),
        co_primary_bars_pass=h3_coprimary_ok,
    )

    h4_coprimary_ok = (
        (h4_st["delta_has_md_vs_plain"] >= 0.15 - 1e-12)
        and (h4_st["delta_has_md_vs_rand"] >= 0.10 - 1e-12)
        and (0.85 - 1e-12 <= h4_st["length_ratio"] <= 1.15 + 1e-12)
    )
    v_h4 = c3_scoring.resolve_verdict(
        c3_0_pass=c3_0_pass,
        direction_valid=dir_valid["WSP_NoMD_Lexicon"],
        dose_valid=dose_valid["WSP_NoMD_Lexicon"],
        requires_nomd_room=True,
        nomd_room_pass=nomd_room_pass,
        reader_dependent=False,
        c3_r_pass=True,
        eq_c3_2_pass=bool(eq_c3_2_doc["WSP_NoMD_Lexicon@d*_nomd_lex"]["pass"]),
        broken_bar_pass=(h4_st["broken_arm"] <= 0.10 + 1e-12) and (h4_st["broken_rand"] <= 0.10 + 1e-12),
        holm_sig=holm_decisions["H4"],
        primary_effect_bar_pass=(h4_st["mean_u"] >= 0.35 - 1e-12),
        co_primary_bars_pass=h4_coprimary_ok,
    )

    # ------------------------------------------------------------------
    # Secondary & Diagnostic Readouts (D0 - D7, Outside Holm)
    # ------------------------------------------------------------------
    # D0: Unbanned Pillar A + Identity_20 (direct/indirect) vs Implied_20 split (ID-indexed)
    d0_arms: dict[str, Any] = {}
    for ban_k in ("none", "common_pirate_ban"):
        d0_arms[ban_k] = {}
        for arm_k in sorted(receipt["pillar_a"][ban_k].keys()):
            sp_v, _, sm = _strict_pirate_by_id(arm_k, ban_k)
            if sp_v is not None:
                sm["identity_direct_10_rate"] = round(mean([sp_v[pid] for pid in pa_ids if pid in id_dir_ids]), 6)
                sm["identity_indirect_10_rate"] = round(mean([sp_v[pid] for pid in pa_ids if pid in id_ind_ids]), 6)
                sm["implied_20_rate"] = round(mean([sp_v[pid] for pid in pa_ids if pid in imp_ids]), 6)
            d0_arms[ban_k][arm_k] = sm

    # D1: Part 213 Copula (paired dLP_self - dLP_stranger relative to Plain)
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

    # D2: Sibling Bleed
    p_on_nomd = receipt["D2_sibling_bleed"]["WSP_Pirate_Contrastive@d*_con_on_NOMD_24"]
    p_on_nomd_cts = [c3_scoring.construct_types(r["text"]) for r in p_on_nomd]
    nomd_on_id = receipt["D2_sibling_bleed"]["WSP_NoMD_Contrastive@d*_nomd_con_on_IDENTITY_20"]
    nomd_id_sp: list[float] = []
    for r in nomd_on_id:
        lrec = labels_map.get(f"D2_sibling_bleed|none|WSP_NoMD_Contrastive@d*_nomd_con|{r['id']}")
        if lrec and lrec.get("ds") and lrec.get("luna") and lrec["ds"].get("pirate_persona") is not None and lrec["luna"].get("pirate_persona") is not None:
            ok = bool(lrec["ds"]["pirate_persona"]) and bool(lrec["luna"]["pirate_persona"]) and (not c3_scoring.is_broken(r["text"]))
            nomd_id_sp.append(1.0 if ok else 0.0)
    d2_sibling = {
        "pirate_con_on_NOMD_24_has_md": round(mean([1.0 if c >= 1 else 0.0 for c in p_on_nomd_cts]), 6),
        "pirate_con_on_NOMD_24_mean_ct": round(mean([float(c) for c in p_on_nomd_cts]), 6),
        "nomd_con_on_IDENTITY_20_strict_pirate": round(mean(nomd_id_sp), 6) if len(nomd_id_sp) == 20 else None,
        "nomd_con_on_IDENTITY_20_broken": round(mean([1.0 if c3_scoring.is_broken(r["text"]) else 0.0 for r in nomd_on_id]), 6),
    }

    # D3: System_Prompt ceilings + 12-prompt text-only showcase slice
    demo_ids = list(bat["demo_showcase_ids_12"])
    sys_nomd_cts = [c3_scoring.construct_types(r["text"]) for r in receipt["pillar_b"]["Sys_NoMD"]]
    d3_ceilings = {
        "Sys_Pirate_none_strict_pirate": d0_arms["none"]["Sys_Pirate"].get("strict_pirate_rate"),
        "Sys_Pirate_common_ban_strict_pirate": d0_arms["common_pirate_ban"]["Sys_Pirate"].get("strict_pirate_rate"),
        "Sys_NoMD_has_md": round(mean([1.0 if c >= 1 else 0.0 for c in sys_nomd_cts]), 6),
        "Sys_NoMD_mean_ct": round(mean([float(c) for c in sys_nomd_cts]), 6),
        "Sys_NoMD_delta_has_md_vs_plain": round(plain_nomd_has_md - mean([1.0 if c >= 1 else 0.0 for c in sys_nomd_cts]), 6),
    }
    pa_prompt_text = {p["id"]: p["text"] for p in bat["identity_20"] + bat["implied_20"]}
    d3_showcase: list[dict[str, Any]] = []
    for ban_k in ("none", "common_pirate_ban"):
        by_arm = {
            a: {r["id"]: r["text"] for r in receipt["pillar_a"][ban_k][a]}
            for a in ("Plain", "Sys_Pirate", "WSP_Pirate_Contrastive@d*_con", "Rand_Pirate_s0@d*_con")
        }
        for pid in demo_ids:
            d3_showcase.append({
                "ban": ban_k,
                "id": pid,
                "prompt": pa_prompt_text[pid],
                "Plain": by_arm["Plain"][pid],
                "Sys_Pirate": by_arm["Sys_Pirate"][pid],
                "WSP_Pirate_Contrastive": by_arm["WSP_Pirate_Contrastive@d*_con"][pid],
                "Rand_Pirate_s0": by_arm["Rand_Pirate_s0@d*_con"][pid],
            })

    # D5: Multi-turn Turn 3 strict_pirate rate
    d5_summary: dict[str, Any] = {}
    for arm_k, rows in sorted(receipt["D5_multiturn"].items()):
        t3_brk = [c3_scoring.is_broken(r["turns"][2]["text"]) for r in rows]
        t3_sp: list[float] = []
        for r, brk in zip(rows, t3_brk):
            lrec = labels_map.get(f"D5_multiturn|none|{arm_k}|{r['id']}")
            if lrec and lrec.get("ds") and lrec.get("luna") and lrec["ds"].get("pirate_persona") is not None and lrec["luna"].get("pirate_persona") is not None:
                t3_sp.append(1.0 if (bool(lrec["ds"]["pirate_persona"]) and bool(lrec["luna"]["pirate_persona"]) and (not brk)) else 0.0)
        d5_summary[arm_k] = {
            "turn3_strict_pirate_rate": round(mean(t3_sp), 6) if len(t3_sp) == 12 else None,
            "turn3_broken_rate": round(mean([1.0 if b else 0.0 for b in t3_brk]), 6),
        }

    # D6: Adaptive/Donor ban rates
    d6_summary: dict[str, Any] = {}
    for cond_k, rows in sorted(receipt["D6_adaptive_donor"]["conditions"].items()):
        brk_v = [c3_scoring.is_broken(r["text"]) for r in rows]
        stem_v = [c3_scoring.surface_stem_hit(r["text"], surface_regex) for r in rows]
        sp_v: list[float] = []
        for r, brk, st in zip(rows, brk_v, stem_v):
            lrec = labels_map.get(f"D6_adaptive_donor|{cond_k}|{cond_k}|{r['id']}")
            if lrec and lrec.get("ds") and lrec.get("luna") and lrec["ds"].get("pirate_persona") is not None and lrec["luna"].get("pirate_persona") is not None:
                sp_v.append(1.0 if (bool(lrec["ds"]["pirate_persona"]) and bool(lrec["luna"]["pirate_persona"]) and (not brk) and (not st)) else 0.0)
        d6_summary[cond_k] = {
            "strict_pirate_rate": round(mean(sp_v), 6) if len(sp_v) == 12 else None,
            "broken_rate": round(mean([1.0 if b else 0.0 for b in brk_v]), 6),
            "surface_stem_hit_rate": round(mean([1.0 if s else 0.0 for s in stem_v]), 6),
        }

    # D7: Operating-point contrast (Con@d*_con vs Lex@d*_lex) + bf16 cross-arm retention ratios
    sp_lex_dstar_ban, _, sum_lex_dstar_ban = _strict_pirate_by_id("WSP_Pirate_Lexicon@d*_lex", "common_pirate_ban")
    d7_diff = (
        round(mean([sp_con_ban[pid] - sp_lex_dstar_ban[pid] for pid in pa_ids]), 6)
        if (sp_con_ban is not None and sp_lex_dstar_ban is not None)
        else None
    )

    def _arm_selected_ret(arm_name: str) -> dict[str, dict[str, float]]:
        cg = receipt["commit_gate"]["arms"][arm_name]
        sel_d = float(cg["selected_d_star"])
        for rr in cg["rungs"]:
            if math.isclose(float(rr["dose"]), sel_d, abs_tol=1e-9):
                return rr["bf16_retention"]
        return cg["rungs"][0]["bf16_retention"]

    ret_table = {
        a: _arm_selected_ret(a)
        for a in ("WSP_Pirate_Contrastive", "WSP_Pirate_Lexicon", "WSP_NoMD_Contrastive", "WSP_NoMD_Lexicon")
    }
    any_ret_below_090 = any(
        float(ret_table[a][L]["stored_rank1_proj_ret"]) < 0.90
        for a in ret_table
        for L in layers
    )
    ret_ratios: dict[str, Any] = {"any_selected_dstar_proj_ret_below_0.90": any_ret_below_090, "by_layer": {}}
    for L in layers:
        con_p = float(ret_table["WSP_Pirate_Contrastive"][L]["stored_rank1_proj_ret"])
        lex_p = float(ret_table["WSP_Pirate_Lexicon"][L]["stored_rank1_proj_ret"])
        r_p_mean = mean([float(rand_res[f"Rand_Pirate_s{s}@d*_con"]["bf16_retention"][L]["stored_rank1_proj_ret"]) for s in range(3)])
        con_n = float(ret_table["WSP_NoMD_Contrastive"][L]["stored_rank1_proj_ret"])
        lex_n = float(ret_table["WSP_NoMD_Lexicon"][L]["stored_rank1_proj_ret"])
        r_n_mean = mean([float(rand_res[f"Rand_NoMD_s{s}@d*_nomd_con"]["bf16_retention"][L]["stored_rank1_proj_ret"]) for s in range(3)])
        ret_ratios["by_layer"][L] = {
            "pirate_con_over_rand_mean": round(con_p / max(r_p_mean, 1e-12), 4),
            "pirate_con_over_lex": round(con_p / max(lex_p, 1e-12), 4),
            "nomd_con_over_rand_mean": round(con_n / max(r_n_mean, 1e-12), 4),
            "nomd_con_over_lex": round(con_n / max(lex_n, 1e-12), 4),
        }

    # D2 length-matched robustness check (S20, ID-indexed) & F7 bottom-rung / d_common separation flags
    con_ban_by_id = {r["id"]: r for r in receipt["pillar_a"]["common_pirate_ban"]["WSP_Pirate_Contrastive@d*_con"]}
    rand_ban_by_id = [{r["id"]: r for r in receipt["pillar_a"]["common_pirate_ban"][f"Rand_Pirate_s{s}@d*_con"]} for s in range(3)]
    len_matched_idx = [
        i for i, pid in enumerate(pa_ids)
        if abs(float(con_ban_by_id[pid]["n_tokens"]) - mean([float(rand_ban_by_id[s][pid]["n_tokens"]) for s in range(3)])) <= 10.0
    ]
    h1_len_matched_diff = (
        round(mean([h1_diffs[i] for i in len_matched_idx]), 6)
        if (h1_diffs is not None and len_matched_idx)
        else None
    )
    d2_length_cov = {
        "n_length_matched_prompts_within_10tok": len(len_matched_idx),
        "h1_length_matched_mean_y": h1_len_matched_diff,
        "con_common_ban_mean_tokens": round(mean([float(con_ban_by_id[pid]["n_tokens"]) for pid in pa_ids]), 3),
        "rand_common_ban_mean_tokens": round(mean([mean([float(rand_ban_by_id[s][pid]["n_tokens"]) for s in range(3)]) for pid in pa_ids]), 3),
    }

    def _rung_index(d_val: float) -> int:
        for idx, rv in enumerate(dose_ladder):
            if math.isclose(float(d_val), rv, abs_tol=1e-9):
                return idx
        return -1

    d_con_idx = _rung_index(receipt["doses"]["d_star_pirate_con"])
    d_comm_idx = _rung_index(receipt["doses"]["d_common"])
    f7_flags = {
        "d_common_gt_1_rung_below_d_star_con": (d_con_idx - d_comm_idx) > 1,
        "bottom_rung_0.075_selected_arms": [
            a for a in ("WSP_Pirate_Contrastive", "WSP_Pirate_Lexicon", "WSP_NoMD_Contrastive", "WSP_NoMD_Lexicon")
            if math.isclose(float(receipt["commit_gate"]["arms"][a]["selected_d_star"]), dose_ladder[0], abs_tol=1e-9)
        ],
    }

    scorecard: dict[str, Any] = {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "confirmatory_verdicts": {
            "C3.0": "PASS" if c3_0_pass else "FAIL",
            "H1": v_h1,
            "H2": v_h2,
            "H3": v_h3,
            "H4": v_h4,
        },
        "gates": {
            "C3.0": gate_c30_doc,
            "C3.0_direction_quality": {
                "pass_by_arm": dir_valid,
                "min_cos_ret_by_arm": dir_min_cos,
                "abs_cos_ret_diff_con_vs_lex": cos_ret_diff_con_lex,
            },
            "C3.0_commit_gate": commit_audit,
            "C3.0_NoMD_room": nomd_room_doc,
            "C3.R": c3_r_doc,
            "EQ_C3_2": eq_c3_2_doc,
        },
        "holm_4_family": {
            "alpha": 0.05,
            "n_perm": n_perm,
            "n_boot": n_boot,
            "p_values": {k: round(v, 8) for k, v in pvals.items()},
            "holm_rejected": holm_decisions,
            "hypotheses": {
                "H1": {
                    "verdict": v_h1,
                    "d_star": receipt["doses"]["d_star_pirate_con"],
                    "mean_y": round(h1_mean_y, 6) if h1_diffs is not None else None,
                    "p_perm": round(h1_p, 8),
                    "holm_sig": holm_decisions["H1"],
                    "bca_ci90": {k: round(v, 6) for k, v in h1_ci.items()} if h1_ci else None,
                    "broken_con": round(brk_h1_con, 6),
                    "broken_rand_mean": round(brk_h1_rand, 6),
                },
                "H2": {
                    "verdict": v_h2,
                    "d_common": receipt["doses"]["d_common"],
                    "d_common_rep_ok": h2_d_common_rep_ok,
                    "d_common_rep_by_arm": h2_d_common_rep_by_arm,
                    "d_common_bf16_retention": h2_d_common_ret_table,
                    "d_star_con": receipt["doses"]["d_star_pirate_con"],
                    "d_star_lex": receipt["doses"]["d_star_pirate_lex"],
                    "mean_y": round(h2_mean_y, 6) if h2_diffs is not None else None,
                    "p_perm": round(h2_p, 8),
                    "holm_sig": holm_decisions["H2"],
                    "bca_ci90": {k: round(v, 6) for k, v in h2_ci.items()} if h2_ci else None,
                    "broken_con": round(brk_h2_con, 6),
                    "broken_lex": round(brk_h2_lex, 6),
                    "abs_cos_ret_diff_con_vs_lex": cos_ret_diff_con_lex,
                },
                "H3": {
                    "verdict": v_h3,
                    "d_star": d_nomd_con,
                    "mean_u": round(h3_st["mean_u"], 6),
                    "p_perm": round(h3_p, 8),
                    "holm_sig": holm_decisions["H3"],
                    "bca_ci90": {k: round(v, 6) for k, v in h3_ci.items()},
                    "delta_has_md_vs_plain": round(h3_st["delta_has_md_vs_plain"], 6),
                    "delta_has_md_vs_rand": round(h3_st["delta_has_md_vs_rand"], 6),
                    "length_ratio": round(h3_st["length_ratio"], 6),
                    "co_primary_bars_pass": h3_coprimary_ok,
                    "broken_arm": round(h3_st["broken_arm"], 6),
                    "broken_rand_mean": round(h3_st["broken_rand"], 6),
                },
                "H4": {
                    "verdict": v_h4,
                    "d_star": d_nomd_lex,
                    "mean_u": round(h4_st["mean_u"], 6),
                    "p_perm": round(h4_p, 8),
                    "holm_sig": holm_decisions["H4"],
                    "bca_ci90": {k: round(v, 6) for k, v in h4_ci.items()},
                    "delta_has_md_vs_plain": round(h4_st["delta_has_md_vs_plain"], 6),
                    "delta_has_md_vs_rand": round(h4_st["delta_has_md_vs_rand"], 6),
                    "length_ratio": round(h4_st["length_ratio"], 6),
                    "co_primary_bars_pass": h4_coprimary_ok,
                    "broken_arm": round(h4_st["broken_arm"], 6),
                    "broken_rand_mean": round(h4_st["broken_rand"], 6),
                },
            },
        },
        "diagnostics": {
            "D0_pillar_a_by_ban_and_split": d0_arms,
            "D1_copula": d1_summary,
            "D2_sibling_bleed": d2_sibling,
            "D2_length_covariate_S20": d2_length_cov,
            "D3_ceilings": d3_ceilings,
            "D3_showcase_12": d3_showcase,
            "D4_L24_mask_mismask": d4,
            "D5_multiturn": d5_summary,
            "D6_adaptive_donor": d6_summary,
            "D7_operating_point_and_retention": {
                "d_star_con": receipt["doses"]["d_star_pirate_con"],
                "d_star_lex": receipt["doses"]["d_star_pirate_lex"],
                "con_minus_lex_operating_point": d7_diff,
                "f7_interpretation_flags": f7_flags,
                "bf16_retention_selected": ret_table,
                "cross_arm_retention_ratios": ret_ratios,
            },
        },
    }
    return scorecard


def render_scorecard_markdown(sc: dict[str, Any]) -> str:
    """Render deterministic GitHub-Flavored Markdown scorecard from `score_stage_c3` output."""
    lines: list[str] = [
        "# Stage C3 (`WSP v1`) Confirmatory & Diagnostic Scorecard",
        "",
        "## Confirmatory Verdicts (Holm-4, `alpha = 0.05`)",
        "",
        "| Gate / Hypothesis | Verdict | Point Estimate | 90% BCa CI | `p_perm` | Holm Sig | Notes |",
        "|---|---|---:|---|---:|---|---|",
    ]
    cv = sc["confirmatory_verdicts"]
    lines.append(f"| **Global `C3.0`** | **`{cv['C3.0']}`** | — | — | — | — | `C3.0.1..4` |")
    hyps = sc["holm_4_family"]["hypotheses"]
    for hid in ("H1", "H2", "H3", "H4"):
        h = hyps[hid]
        pt = h.get("mean_y") if hid in ("H1", "H2") else h.get("mean_u")
        pt_s = "—" if pt is None else f"{pt:+.4f}"
        ci = h.get("bca_ci90")
        ci_s = "—" if not ci else f"[{ci['ci90_lo']:+.4f}, {ci['ci90_hi']:+.4f}]"
        if hid in ("H1", "H2"):
            if "broken_con" in h:
                note = f"d={h.get('d_star', h.get('d_common'))}, brk={h['broken_con']:.3f}"
            else:
                note = f"blocked ({h['verdict']})"
        else:
            if "delta_has_md_vs_plain" in h:
                note = (
                    f"d={h.get('d_star')}, Δmd_P={h['delta_has_md_vs_plain']:+.3f}, "
                    f"Δmd_R={h['delta_has_md_vs_rand']:+.3f}, lenR={h['length_ratio']:.3f}"
                )
            else:
                note = f"blocked ({h['verdict']})"
        lines.append(
            f"| **`{hid}`** | **`{h['verdict']}`** | `{pt_s}` | `{ci_s}` | `{h['p_perm']:.6f}` | `{h['holm_sig']}` | {note} |"
        )

    # Amendment 1 (GLM F2 R2): render H2 d_common bf16 storage retention tables directly beside H2!
    h2_doc = hyps.get("H2", {})
    h2_dc_ret = h2_doc.get("d_common_bf16_retention")
    if isinstance(h2_dc_ret, dict):
        lines.extend([
            "",
            f"### `H2` Matched-Dose `d_common = {h2_doc.get('d_common')}` `bf16` Storage Retention (Amendment 1: `d_common_rep_ok = {h2_doc.get('d_common_rep_ok')}`)",
            "",
            "| Arm | Layer | `stored_rank1_proj_ret` (`>= 0.50`) | `stored_frob_ret` (`>= 0.80`) | Arm `rep_ok` |",
            "|---|---|---:|---:|---|",
        ])
        for arm_name in ("WSP_Pirate_Contrastive", "WSP_Pirate_Lexicon"):
            arm_ret = h2_dc_ret.get(arm_name)
            rep_flag = (h2_doc.get("d_common_rep_by_arm") or {}).get(arm_name)
            if isinstance(arm_ret, dict):
                for L in ("21", "24", "27"):
                    lm = arm_ret.get(L, {})
                    lines.append(
                        f"| `{arm_name}@d_common` | `L{L}` | `{lm.get('stored_rank1_proj_ret', '—')}` | `{lm.get('stored_frob_ret', '—')}` | `{rep_flag}` |"
                    )
            else:
                lines.append(f"| `{arm_name}@d_common` | `—` | `missing` | `missing` | `{rep_flag}` |")

    lines.extend([
        "",
        "## Capability Preservation (`EQ-C3.2`)",
        "",
        "| Arm | Gating | Math Acc (Strict) | Code Pass | Neutral `off_KL` | Neutral Broken | `EQ-C3.2` |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for arm_k, am in sorted(sc["gates"]["EQ_C3_2"].items()):
        lines.append(
            f"| `{arm_k}` | `{'YES' if am['gating'] else 'log'}` | `{am['math_acc']:.3f}` (`{am['math_strict_final_acc']:.3f}`) "
            f"| `{am['code_pass_rate']:.3f}` | `{am['neutral_median_off_kl']:.4f}` | `{am['neutral_broken_rate']:.3f}` | `{'PASS' if am['pass'] else 'FAIL'}` |"
        )

    dq_gate = sc["gates"].get("C3.0_direction_quality")
    cg_gate = sc["gates"].get("C3.0_commit_gate")
    nr_gate = sc["gates"].get("C3.0_NoMD_room")
    lines.extend([
        "",
        "## Gate Summary",
        "",
        f"- **`C3.0`:** `{sc['gates']['C3.0']['pass']}` (`C3.0.1={sc['gates']['C3.0']['C3.0.1']['pass']}`, `C3.0.2={sc['gates']['C3.0']['C3.0.2']['pass']}`, `C3.0.3={sc['gates']['C3.0']['C3.0.3']['pass']}`, `C3.0.4={sc['gates']['C3.0']['C3.0.4']['pass']}`)",
        f"- **`C3.0_direction_quality`:** `{dq_gate['pass_by_arm'] if dq_gate else 'not_evaluated (C3.0 FAIL)'}`",
        f"- **`C3.0_commit_gate`:** `{{k: v['pass'] for k, v in ...}}` = `{ {k: v['pass'] for k, v in cg_gate.items()} if cg_gate else 'not_evaluated (C3.0 FAIL)' }`",
        f"- **`C3.0_NoMD_room`:** `{nr_gate['pass'] if nr_gate else 'not_evaluated (C3.0 FAIL)'}`"
        + (
            f" (`has_md={nr_gate['plain_has_markdown']:.3f}`, `mean_ct={nr_gate['plain_mean_construct_types']:.3f}`)"
            if nr_gate
            else ""
        ),
        f"- **`C3.R`:** `{sc['gates']['C3.R']['pass']}` (`H1={sc['gates']['C3.R']['h1_pass']}`, `H2={sc['gates']['C3.R']['h2_pass']}`)",
        "",
    ])

    diag = sc.get("diagnostics", {})
    if diag:
        d4 = diag.get("D4_L24_mask_mismask", {})
        d3c = diag.get("D3_ceilings", {})
        d7 = diag.get("D7_operating_point_and_retention", {})
        lines.extend([
            "## Secondary & Diagnostic Readouts (`D0–D7`, Outside Holm)",
            "",
            f"- **`D4` `L24` Mask/Mis-Mask:** `masked/unmasked = {d4.get('masked_over_unmasked_logit_l2_ratio')}x`, `mismasked/unmasked = {d4.get('mismasked_over_unmasked_logit_l2_ratio')}x`",
            f"- **`D3` Reference Ceilings:** `Sys_Pirate(none)={d3c.get('Sys_Pirate_none_strict_pirate')}`, `Sys_Pirate(ban)={d3c.get('Sys_Pirate_common_ban_strict_pirate')}`, `Sys_NoMD(has_md)={d3c.get('Sys_NoMD_has_md')}` (`ΔvsPlain={d3c.get('Sys_NoMD_delta_has_md_vs_plain')}`)",
            f"- **`D7` Operating-Point & `F7` Flags:** `d*_con={d7.get('d_star_con')}`, `d*_lex={d7.get('d_star_lex')}`, `Con@d*_con - Lex@d*_lex = {d7.get('con_minus_lex_operating_point')}`, `flags={d7.get('f7_interpretation_flags')}`",
            "",
            "### `D0` / `D2` Pillar A Rates, Reader Coherence vs. `is_broken`, and Token Lengths",
            "",
            "| Ban | Arm | Strict Pirate | Dir-10 / Ind-10 / Imp-20 | Reader Coherent | `is_broken` | Stem Hit | Mean Tok |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ])
        for ban_k in ("none", "common_pirate_ban"):
            for arm_k, sm in sorted(diag.get("D0_pillar_a_by_ban_and_split", {}).get(ban_k, {}).items()):
                sp = sm.get("strict_pirate_rate")
                sp_s = "—" if sp is None else f"{sp:.3f}"
                split_s = (
                    f"{sm['identity_direct_10_rate']:.2f} / {sm['identity_indirect_10_rate']:.2f} / {sm['implied_20_rate']:.2f}"
                    if "identity_direct_10_rate" in sm
                    else "—"
                )
                coh = sm.get("reader_both_coherent_rate")
                coh_s = "—" if coh is None else f"{coh:.3f}"
                stem = sm.get("surface_stem_hit_rate")
                stem_s = "—" if stem is None else f"{stem:.3f}"
                lines.append(
                    f"| `{ban_k}` | `{arm_k}` | `{sp_s}` | `{split_s}` | `{coh_s}` | `{sm.get('broken_rate', 0.0):.3f}` | `{stem_s}` | `{sm.get('mean_tokens', 0.0):.1f}` |"
                )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score Stage C3 (WSP v1) receipt using 100% pure Python stdlib.")
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--reader-labels", type=Path, default=None)
    ap.add_argument("--design", type=Path, default=HERE / "OLMO3-WSP-C3-DESIGN.md")
    ap.add_argument("--c3-constants", type=Path, default=HERE / "c3_prereg_constants.json")
    ap.add_argument("--bank", type=Path, default=HERE / "c3_vector_bank.safetensors")
    ap.add_argument("--scoring", type=Path, default=HERE / "c3_scoring.py")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    args = ap.parse_args(argv)

    disk_hashes = {
        "c3_design_sha256": sha256_file(args.design),
        "c3_constants_sha256": sha256_file(args.c3_constants),
        "c3_bank_sha256": sha256_file(args.bank),
        "c3_scoring_sha256": sha256_file(args.scoring),
    }
    receipt = load_json_or_gz(args.receipt)
    receipt_sha = sha256_file(args.receipt)
    c3_constants = json.loads(args.c3_constants.read_text(encoding="utf-8"))
    r_labels = load_json_or_gz(args.reader_labels) if args.reader_labels is not None else None

    scorecard = score_stage_c3(
        receipt,
        c3_constants,
        reader_labels=r_labels,
        disk_hashes=disk_hashes,
        receipt_sha256=receipt_sha,
        strict_provenance=True,
        allow_mock_readers=False,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_scorecard_markdown(scorecard), encoding="utf-8")
    print(json.dumps(scorecard["confirmatory_verdicts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
