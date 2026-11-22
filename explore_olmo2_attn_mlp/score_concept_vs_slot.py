#!/usr/bin/env python3
"""OLMo 2 1B Instruct — Concept vs. Slot Head-Gating & Late-Layer Heads Offline Scorer.

Implements every pre-registered gate and endpoint (`CS.0`–`CS.4`) of
`explore_olmo2_attn_mlp/CONCEPT-VS-SLOT-DESIGN.md` v3.1 (Design PASS Frozen;
SHA-256: 88986b904f499893ec5aba4039af9142846e45307552409cdaf0237f21b90bbd).

Contracts:
- Single-read external 6-pin (`design`, `manifest`, `runner`, `scorer`,
  `targets213`, `m7_transport`) verification.
- `--verify-smoke-only`: checks ALL 6 `CS.0` clauses + full 12-pair `C1..C6`
  `CS.1` unpatched transport gate for BOTH classes + preliminary head-null
  sanity check (`|med| <= 0.30`) on Smoke's 156 blocked rows, emitting ZERO
  head-gating contrasts or scientific verdicts (`CS.2`/`CS.3`/`CS.4`).
- Full scoring mode: hard-refuses smoke receipts (`rc != 0`, no scorecard written),
  enforces `CS.0` strict precedence, `CS.1` per-class transport & head-null gates,
  `CS.2` studentized pair-cluster bootstrap (10,000 deterministic draws) + exact
  paired `t_11` + 5-control interaction DiD + `C_SN` concordance + peer-median
  collapse guard + within-kind (`5` same-kind pairs + `n=23` 7-param HC3 OLS) guard
  + transport (`Δ_F_ratio` + Weaver within-class regression & ANCOVA) guard +
  3-readout agreement + mandatory Branch 3/4 wording + all descriptive sensitivities,
  `CS.3` exact 7-test Bonferroni (`99.29%` unit-cluster CI) with named DLA-matched
  primary controls & frozen `(C_SR, C_SN)` signs, and `CS.4` `L7H5` 2-channel
  directional antisymmetry diagnostic (exact Part B `S_odd` sign).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import run_concept_vs_slot as rcvs


CS0_FAIL_SCI = "uninformative (cs0_failed)"
MANDATORY_NULL_WORDING = "any class difference is smaller than 0.35 logits (about 68% of Part B's L8H2 C_SR)"

# Pre-registered thresholds (v3.1 §5).
CS1_TRANSPORT_FLOOR = 0.50
CS1_HEAD_NULL_FULL_CAP = 0.15
CS1_HEAD_NULL_SMOKE_CAP = 0.30

CS2_EPS = 0.35
CS2_CTRL_INTERACTION_MARGIN = 0.15
CS2_LEVEL_FLOOR = 0.25
CS2_SHARED_CSN_MARGIN = 0.10
CS2_MIN_PAIR_AGREE = 9
CS2_PEER_MED_SHARE_CAP = 0.50

CS3_N_TESTS = 7
CS3_ALPHA = 0.05 / CS3_N_TESTS  # 0.007142857... -> 99.29% two-sided CI
CS3_ABS_FLOOR = 0.25
CS3_CTRL_MARGIN = 0.15
CS3_PEER_MED_SHARE_CAP = 0.50

CS4_MEAN_FLOOR = 0.20
CS4_CTRL_MARGIN = 0.15
CS4_UNIT_FLOOR = 0.30
CS4_MIN_UNITS = 4

BOOT_DRAWS = 10000
BOOT_SEED_BASE = 2026112299

L7_CTRL_HEADS: tuple[str, ...] = ("L7H13", "L7H4", "L7H9")
L8_CTRL_HEADS: tuple[str, ...] = ("L8H3", "L8H13", "L8H7", "L8H9", "L8H0")
CS3_CANDIDATES: tuple[str, ...] = ("L11H9", "L11H2", "L11H3", "L13H1", "L13H4", "L14H10", "L14H3")
LAYER_NEUTRAL_CTRLS: dict[int, tuple[str, ...]] = {
    11: ("L11H1", "L11H6", "L11H4", "L11H10"),
    13: ("L13H5", "L13H13"),
    14: ("L14H12", "L14H8"),
}

STORIES_6: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "C6")
CO_PRIMARY_CHANNELS: tuple[str, ...] = ("raw", "spec", "logp")
CHANNEL_DISPLAY: dict[str, str] = {
    "raw": "RawLogit",
    "spec": "SpecLogit",
    "logp": "TargetLogProb",
    "phrase_mean": "PhraseLogProbMean",
    "peer_med": "PeerMed",
}

SAME_KIND_PAIR_IDS: tuple[str, ...] = ("P02", "P03", "P05", "P08", "P11")
INTACT_CLEAN_PAIR_IDS: tuple[str, ...] = ("P01", "P03", "P05", "P06", "P08")
PART_B_BRIDGE_UNITS: tuple[str, ...] = ("p92-02", "p92-05", "p92-13", "p92-46", "p92-09")
READER_AGREED_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("P01", "p92-10", "p92-29"),
    ("P04", "p92-13", "p92-39"),
    ("P06", "p92-35", "p92-48"),
    ("P07", "p92-36", "p92-08"),
    ("P08", "p92-47", "p92-45"),
    ("P09", "p92-33", "p92-15"),
    ("L6_ra", "p92-09", "p92-43"),
)


def _is_finite_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _mean(vals: list[float]) -> float:
    if not vals:
        raise ValueError("Cannot take mean of empty list")
    return float(sum(vals) / len(vals))


def _det_seed(tag: str) -> int:
    import hashlib
    return int(hashlib.sha256(f"{BOOT_SEED_BASE}|{tag}".encode("utf-8")).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Statistical Primitives (Studentized Cluster Bootstrap, Exact t, HC3 OLS)
# ---------------------------------------------------------------------------

def studentized_cluster_bootstrap(
    mat: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = BOOT_DRAWS,
    seed: int = 12345,
) -> dict[str, Any]:
    """Nested studentized cluster bootstrap over K clusters x S stories + exact t_{K-1} CI.

    mat: shape (K, S) float64 array where row k is cluster k across S stories.
    """
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D cluster matrix, got shape {arr.shape}")
    K, S = arr.shape
    if K < 2 or S < 1:
        raise ValueError(f"Need K>=2, S>=1, got {(K, S)}")

    cluster_means = arr.mean(axis=1)
    theta_hat = float(cluster_means.mean())
    sd_k = float(cluster_means.std(ddof=1))
    se_hat = sd_k / math.sqrt(K)
    t_crit = float(sp_stats.t.ppf(1.0 - alpha / 2.0, df=K - 1))
    t_lo = theta_hat - t_crit * se_hat
    t_hi = theta_hat + t_crit * se_hat

    if se_hat <= 1e-12:
        return {
            "mean": theta_hat,
            "se": 0.0,
            "sd_clusters": 0.0,
            "n_clusters": K,
            "n_stories": S,
            "alpha": alpha,
            "ci_level": 1.0 - alpha,
            "boot_ci": [theta_hat, theta_hat],
            "exact_t_ci": [theta_hat, theta_hat],
            "cluster_means": [float(x) for x in cluster_means],
        }

    rng = np.random.default_rng(seed)
    # Vectorized in chunks of 2000 for crisp memory & speed.
    t_stats = np.empty(n_boot, dtype=np.float64)
    chunk = 2000
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx_k = rng.integers(0, K, size=(m, K))  # (m, K)
        idx_s = rng.integers(0, S, size=(m, K, S))  # (m, K, S)
        # Gather arr[idx_k[:, :, None], idx_s] -> (m, K, S)
        sampled = arr[idx_k[:, :, None], idx_s]
        c_means = sampled.mean(axis=2)  # (m, K)
        th_b = c_means.mean(axis=1)
        se_b = c_means.std(axis=1, ddof=1) / math.sqrt(K)
        diff = th_b - theta_hat
        # Handle ultra-rare degenerate resample se_b == 0 safely.
        t_chunk = np.where(
            se_b > 1e-12,
            diff / np.where(se_b > 1e-12, se_b, 1.0),
            np.where(np.abs(diff) <= 1e-12, 0.0, np.sign(diff) * 1e6),
        )
        t_stats[done : done + m] = t_chunk
        done += m

    q_lo = float(np.quantile(t_stats, alpha / 2.0))
    q_hi = float(np.quantile(t_stats, 1.0 - alpha / 2.0))
    boot_lo = theta_hat - q_hi * se_hat
    boot_hi = theta_hat - q_lo * se_hat

    return {
        "mean": theta_hat,
        "se": se_hat,
        "sd_clusters": sd_k,
        "n_clusters": K,
        "n_stories": S,
        "alpha": alpha,
        "ci_level": 1.0 - alpha,
        "boot_ci": [float(boot_lo), float(boot_hi)],
        "exact_t_ci": [float(t_lo), float(t_hi)],
        "cluster_means": [float(x) for x in cluster_means],
    }


def ols_hc3(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Ordinary least squares with HC3 heteroskedasticity-consistent standard errors."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    if n <= p:
        raise ValueError(f"OLS underdetermined: n={n}, p={p}")
    rank = int(np.linalg.matrix_rank(X))
    if rank < p:
        raise ValueError(f"OLS design matrix rank-deficient: rank={rank} < p={p}")
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    H_diag = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    denom = np.clip(1.0 - H_diag, 1e-6, None)
    w = (resid / denom) ** 2
    meat = X.T @ (w[:, None] * X)
    cov_hc3 = xtx_inv @ meat @ xtx_inv
    se_hc3 = np.sqrt(np.clip(np.diag(cov_hc3), 0.0, None))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {
        "n": int(n),
        "p": int(p),
        "df_resid": int(n - p),
        "beta": [float(b) for b in beta],
        "se_hc3": [float(s) for s in se_hc3],
        "r2": float(r2),
    }


def welch_two_sample_t(x_concept: list[float], x_slot: list[float], alpha: float = 0.05) -> dict[str, Any]:
    a = np.asarray(x_concept, dtype=np.float64)
    b = np.asarray(x_slot, dtype=np.float64)
    na, nb = len(a), len(b)
    va = float(a.var(ddof=1))
    vb = float(b.var(ddof=1))
    diff = float(a.mean() - b.mean())
    se2 = va / na + vb / nb
    se = math.sqrt(max(se2, 0.0))
    if se <= 1e-12:
        return {"diff_concept_minus_slot": diff, "se": 0.0, "df": float(na + nb - 2), "ci": [diff, diff], "n_concept": na, "n_slot": nb}
    df = (se2 ** 2) / (((va / na) ** 2) / (na - 1) + ((vb / nb) ** 2) / (nb - 1))
    t_crit = float(sp_stats.t.ppf(1.0 - alpha / 2.0, df=df))
    return {
        "diff_concept_minus_slot": diff,
        "mean_concept": float(a.mean()),
        "mean_slot": float(b.mean()),
        "se": se,
        "df": float(df),
        "ci": [diff - t_crit * se, diff + t_crit * se],
        "n_concept": na,
        "n_slot": nb,
    }


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    a = np.asarray(xs, dtype=np.float64)
    b = np.asarray(ys, dtype=np.float64)
    if len(a) < 2 or float(a.std(ddof=0)) <= 1e-12 or float(b.std(ddof=0)) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# CS.0 — Technical Validation & Complete Manifest Join
# ---------------------------------------------------------------------------

def _validate_blocked_channel(obj: Any, label: str) -> float:
    if not isinstance(obj, dict):
        raise ValueError(f"{label}: missing channel dict")
    for k in ("d0", "d_pos", "d_neg"):
        if not _is_finite_num(obj.get(k)):
            raise ValueError(f"{label}: non-finite/missing {k}")
    for arr_k in ("wd_pos", "wd_neg"):
        arr = obj.get(arr_k)
        if not isinstance(arr, list) or len(arr) != rcvs.N_NULL_DRAWS or not all(_is_finite_num(v) for v in arr):
            raise ValueError(f"{label}: invalid {arr_k}")
    return abs(float(obj["d0"]))


def validate_cs0_and_join(
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    pins: dict[str, str],
    expect_full: bool,
    allow_fixture: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify exact schema, 6-pin binding, provenance, complete manifest join, and ALL 6 CS.0 clauses."""
    meta = receipt.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Receipt missing meta object")
    if meta.get("schema_version") != rcvs.SCHEMA_VERSION:
        raise ValueError(f"Schema version mismatch: {meta.get('schema_version')!r}")
    exp_art = "concept_vs_slot_full_receipt" if expect_full else "concept_vs_slot_smoke_receipt"
    if meta.get("artifact_type") != exp_art:
        raise ValueError(f"Artifact type mismatch: got {meta.get('artifact_type')!r}, expected {exp_art!r}")
    exp_mode = "full" if expect_full else "smoke"
    if meta.get("run_mode") != exp_mode:
        raise ValueError(f"Run mode mismatch: got {meta.get('run_mode')!r}, expected {exp_mode!r}")
    # F3: enforce explicit meta["complete"] flag on full (True) and smoke (False) receipts.
    exp_complete = True if expect_full else False
    if meta.get("complete") is not exp_complete:
        raise ValueError(f"Receipt meta.complete mismatch: got {meta.get('complete')!r}, expected {exp_complete!r}")

    prov = meta.get("execution_provenance")
    dev_name = meta.get("device_name")
    is_cuda_dev = meta.get("is_cuda_device")
    if not allow_fixture:
        if prov != "measured_gpu_run":
            raise ValueError(f"Refused non-measured execution_provenance={prov!r}")
        if is_cuda_dev is not True:
            raise ValueError(f"Refused measured_gpu_run without is_cuda_device=True (got {is_cuda_dev!r})")
        dev_norm = dev_name.strip().lower() if isinstance(dev_name, str) else ""
        if not dev_norm or dev_norm in ("cpu", "mps", "meta", "none", "unknown") or dev_norm.startswith("cpu"):
            raise ValueError(f"Refused measured_gpu_run with non-GPU device_name={dev_name!r}")

    rp = meta.get("pins", {})
    for k, v in pins.items():
        if rp.get(k) != v:
            raise ValueError(f"Receipt meta.pins[{k}] mismatch: got {rp.get(k)!r}, expected {v!r}")
    if meta.get("operating_norm_name") != "OLD_NORM" or meta.get("operating_norm_multipliers") != rcvs.OLD_NORM:
        raise ValueError("Receipt OLD_NORM multipliers mismatch")

    sched = manifest["schedule_full" if expect_full else "schedule_smoke"]
    den = manifest["meta"]["denominators"]["full" if expect_full else "smoke"]
    den_full = manifest["meta"]["denominators"]["full"]

    rcvs.validate_timing_breakdown(
        meta.get("timing"),
        den["n_b11_forwards"],
        den["n_b1_forwards"],
        full_b11=den_full["n_b11_forwards"],
        full_b1=den_full["n_b1_forwards"],
    )
    if int(meta.get("n_b11_forwards", -1)) != den["n_b11_forwards"]:
        raise ValueError("Receipt n_b11_forwards mismatch")
    if int(meta.get("n_b1_forwards", -1)) != den["n_b1_forwards"]:
        raise ValueError("Receipt n_b1_forwards mismatch")

    unp_rows = receipt.get("unpatched_rows", [])
    blk_rows = receipt.get("blocked_rows", [])
    shm_rows = receipt.get("sham_rows", [])
    act_rows = receipt.get("actuation_rows", [])
    ce_rows = receipt.get("causal_equiv_rows", [])

    if len(unp_rows) != den["n_unpatched_rows"]:
        raise ValueError(f"unpatched_rows len={len(unp_rows)} != {den['n_unpatched_rows']}")
    if len(blk_rows) != den["n_blocked_rows"]:
        raise ValueError(f"blocked_rows len={len(blk_rows)} != {den['n_blocked_rows']}")
    if len(shm_rows) != den["n_sham_rows"]:
        raise ValueError(f"sham_rows len={len(shm_rows)} != {den['n_sham_rows']}")
    if len(act_rows) != den["n_actuation_rows"]:
        raise ValueError(f"actuation_rows len={len(act_rows)} != {den['n_actuation_rows']}")
    if len(ce_rows) != den["n_causal_equiv_rows"]:
        raise ValueError(f"causal_equiv_rows len={len(ce_rows)} != {den['n_causal_equiv_rows']}")

    exp_unp_keys = {(e["unit"], e["prompt_id"]) for e in sched["unpatched"]}
    exp_blk_keys = {(e["unit"], e["prompt_id"], e["head_name"]) for e in sched["blocked"]}
    exp_shm_keys = {e["sham_id"] for e in sched["sham"]}
    exp_act_keys = {e["actuation_id"] for e in sched["actuation"]}
    exp_ce_keys = {e["check_id"] for e in sched["causal_equiv"]}

    unp_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    blk_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    shm_by_key: dict[str, dict[str, Any]] = {}
    act_by_key: dict[str, dict[str, Any]] = {}
    ce_by_key: dict[str, dict[str, Any]] = {}

    failures: list[str] = []
    max_sham = {ch: 0.0 for ch in rcvs.CHANNELS_5}
    max_decomp = 0.0
    max_sib_sci = 0.0
    max_sib_act = 0.0
    min_act_delta = float("inf")
    min_act_readout_delta = float("inf")
    min_blk_pos_drop = float("inf")
    max_orth_cos = 0.0
    max_ce_err = 0.0
    unpatched_has_sham = False

    # 1. Unpatched rows (Clause 2 & Clause 5).
    for r in unp_rows:
        k = (r["unit"], r["prompt_id"])
        if k in unp_by_key:
            raise ValueError(f"Duplicate unpatched key: {k}")
        unp_by_key[k] = r
        if any("sham" in fk.lower() for fk in r.keys()):
            unpatched_has_sham = True
            failures.append(f"unpatched_has_sham_key:{k}")
        tr = r.get("transport", {})
        if any("sham" in fk.lower() for fk in tr.keys()):
            unpatched_has_sham = True
            failures.append(f"unpatched_transport_has_sham_key:{k}")
        if not _is_finite_num(r.get("max_abs_wdown_orth_cos")):
            failures.append(f"unpatched_missing_orth_cos:{k}")
        else:
            max_orth_cos = max(max_orth_cos, float(r["max_abs_wdown_orth_cos"]))
        for ch in rcvs.CHANNELS_5:
            for fld in (f"base_{ch}", f"t_pos_{ch}", f"t_neg_{ch}"):
                if not _is_finite_num(tr.get(fld)):
                    failures.append(f"unpatched_nonfinite:{k}:{fld}")
            for arr_fld in (f"wdown_pos_{ch}", f"wdown_neg_{ch}"):
                arr = tr.get(arr_fld)
                if not isinstance(arr, list) or len(arr) != rcvs.N_NULL_DRAWS or not all(_is_finite_num(v) for v in arr):
                    failures.append(f"unpatched_bad_null_arr:{k}:{arr_fld}")
        if not _is_finite_num(tr.get("base_phrase_logp")):
            failures.append(f"unpatched_nonfinite:{k}:base_phrase_logp")
        errs = [
            abs(float(tr["base_spec"]) - (float(tr["base_raw"]) - float(tr["base_peer_med"]))),
            abs(float(tr["t_pos_spec"]) - (float(tr["t_pos_raw"]) - float(tr["t_pos_peer_med"]))),
            abs(float(tr["t_neg_spec"]) - (float(tr["t_neg_raw"]) - float(tr["t_neg_peer_med"]))),
            *[abs(float(tr["wdown_pos_spec"][j]) - (float(tr["wdown_pos_raw"][j]) - float(tr["wdown_pos_peer_med"][j]))) for j in range(rcvs.N_NULL_DRAWS)],
            *[abs(float(tr["wdown_neg_spec"][j]) - (float(tr["wdown_neg_raw"][j]) - float(tr["wdown_neg_peer_med"][j]))) for j in range(rcvs.N_NULL_DRAWS)],
        ]
        max_decomp = max(max_decomp, max(errs))

    if set(unp_by_key.keys()) != exp_unp_keys:
        raise ValueError("Unpatched row keys do not match manifest schedule")

    # 2. Blocked rows (Clause 2 & Clause 3).
    for r in blk_rows:
        k = (r["unit"], r["prompt_id"], r["head_name"])
        if k in blk_by_key:
            raise ValueError(f"Duplicate blocked key: {k}")
        blk_by_key[k] = r
        for ch in rcvs.CHANNELS_5:
            d0_abs = _validate_blocked_channel(r.get(ch), f"blk:{k}:{ch}")
            max_sham[ch] = max(max_sham[ch], d0_abs)
        sp, rw, pm = r["spec"], r["raw"], r["peer_med"]
        d_errs = [
            abs(float(sp["d0"]) - (float(rw["d0"]) - float(pm["d0"]))),
            abs(float(sp["d_pos"]) - (float(rw["d_pos"]) - float(pm["d_pos"]))),
            abs(float(sp["d_neg"]) - (float(rw["d_neg"]) - float(pm["d_neg"]))),
            *[abs(float(sp["wd_pos"][j]) - (float(rw["wd_pos"][j]) - float(pm["wd_pos"][j]))) for j in range(rcvs.N_NULL_DRAWS)],
            *[abs(float(sp["wd_neg"][j]) - (float(rw["wd_neg"][j]) - float(pm["wd_neg"][j]))) for j in range(rcvs.N_NULL_DRAWS)],
        ]
        max_decomp = max(max_decomp, max(d_errs))
        sib = r.get("sibling_max_abs_leakage")
        if not _is_finite_num(sib):
            failures.append(f"blk_missing_sibling:{k}")
        else:
            max_sib_sci = max(max_sib_sci, float(sib))
        # F1: recompute blocking positive control directly from blocked row d_pos across channels.
        row_max_pos_drop = max(abs(float(r[ch]["d_pos"])) for ch in rcvs.CHANNELS_5)
        min_blk_pos_drop = min(min_blk_pos_drop, row_max_pos_drop)
        if row_max_pos_drop <= rcvs.TOL_BLOCK_ACTUATION:
            failures.append(f"blk_inert_zero_drop:{k}:max_pos_drop={row_max_pos_drop}")

    if set(blk_by_key.keys()) != exp_blk_keys:
        raise ValueError("Blocked row keys do not match manifest schedule")

    # 3. Dedicated full-B11 sham rows (Clause 2).
    for r in shm_rows:
        sid = r["sham_id"]
        if sid in shm_by_key:
            raise ValueError(f"Duplicate sham_id: {sid}")
        shm_by_key[sid] = r
        for ch in rcvs.CHANNELS_5:
            v = r.get(f"max_abs_sham_{ch}")
            if not _is_finite_num(v):
                failures.append(f"sham_nonfinite:{sid}:{ch}")
            else:
                max_sham[ch] = max(max_sham[ch], float(v))
    if set(shm_by_key.keys()) != exp_shm_keys:
        raise ValueError("Sham row keys do not match manifest schedule")

    # 4. Synthetic head-actuation controls (Clause 3 & Clause 4).
    act_passed = 0
    for r in act_rows:
        aid = r["actuation_id"]
        if aid in act_by_key:
            raise ValueError(f"Duplicate actuation_id: {aid}")
        act_by_key[aid] = r
        td = r.get("target_max_abs_delta")
        rd = r.get("readout_max_abs_delta")
        sl = r.get("sibling_max_abs_leakage")
        if not (_is_finite_num(td) and _is_finite_num(rd) and _is_finite_num(sl)):
            failures.append(f"actuation_nonfinite:{aid}")
            continue
        min_act_delta = min(min_act_delta, float(td))
        min_act_readout_delta = min(min_act_readout_delta, float(rd))
        max_sib_act = max(max_sib_act, float(sl))
        if (
            float(td) > rcvs.MIN_ACTUATION_DELTA
            and float(rd) > rcvs.MIN_ACTUATION_READOUT_DELTA
            and float(sl) <= rcvs.TOL_SIBLING
            and bool(r.get("passed", False))
        ):
            act_passed += 1
        else:
            failures.append(f"actuation_failed:{aid}:td={td}:rd={rd}:sl={sl}")
    if set(act_by_key.keys()) != exp_act_keys:
        raise ValueError("Actuation row keys do not match manifest schedule")

    # 5. Single-pass causal equivalence checks (Clause 6).
    ce_passed = 0
    for r in ce_rows:
        cid = r["check_id"]
        if cid in ce_by_key:
            raise ValueError(f"Duplicate check_id: {cid}")
        ce_by_key[cid] = r
        err = r.get("max_abs_full_vocab_read_pos_diff")
        if not _is_finite_num(err):
            failures.append(f"causal_equiv_nonfinite:{cid}")
            continue
        max_ce_err = max(max_ce_err, float(err))
        if float(err) <= rcvs.TOL_CAUSAL_EQUIV and bool(r.get("passed", False)):
            ce_passed += 1
        else:
            failures.append(f"causal_equiv_failed:{cid}:err={err}")
    if set(ce_by_key.keys()) != exp_ce_keys:
        raise ValueError("Causal equivalence row keys do not match manifest schedule")

    # Evaluate exact CS.0 Clauses 1-6.
    clause_1_ok = True
    clause_2_ok = (
        max_sham["spec"] <= rcvs.TOL_SHAM
        and max_sham["raw"] <= rcvs.TOL_SHAM
        and max_sham["logp"] <= rcvs.TOL_SHAM
        and max_sham["phrase_mean"] <= rcvs.TOL_SHAM
        and max_decomp <= rcvs.TOL_DECOMP
    )
    clause_3_ok = max_sib_sci <= rcvs.TOL_SIBLING and max_sib_act <= rcvs.TOL_SIBLING
    clause_4_ok = (
        act_passed == den["n_actuation_rows"]
        and min_act_delta > rcvs.MIN_ACTUATION_DELTA
        and min_act_readout_delta > rcvs.MIN_ACTUATION_READOUT_DELTA
        and min_blk_pos_drop > rcvs.TOL_BLOCK_ACTUATION
    )
    clause_5_ok = (not unpatched_has_sham) and (max_orth_cos <= rcvs.TOL_ORTH)
    clause_6_ok = ce_passed == den["n_causal_equiv_rows"] and max_ce_err <= rcvs.TOL_CAUSAL_EQUIV

    if not clause_2_ok:
        failures.append(f"clause_2_sham_decomp_failed:sham={max_sham},decomp={max_decomp}")
    if not clause_3_ok:
        failures.append(f"clause_3_sibling_leakage_failed:sci={max_sib_sci},act={max_sib_act}")
    if not clause_4_ok:
        failures.append(
            f"clause_4_actuation_or_block_pos_ctrl_failed:{act_passed}/{den['n_actuation_rows']},"
            f"min_td={min_act_delta},min_rd={min_act_readout_delta},min_blk={min_blk_pos_drop}"
        )
    if not clause_5_ok:
        failures.append(f"clause_5_unpatched_or_orth_failed:unp_sham={unpatched_has_sham},orth={max_orth_cos}")
    if not clause_6_ok:
        failures.append(f"clause_6_causal_equiv_failed:{ce_passed}/{den['n_causal_equiv_rows']},err={max_ce_err}")

    cs0_passed = len(failures) == 0
    cs0_report = {
        "status": "passed" if cs0_passed else "failed",
        "passed": cs0_passed,
        "clause_1_pins_provenance_join": clause_1_ok,
        "clause_2_sham_decomp_finiteness": clause_2_ok,
        "clause_3_sibling_zero_leakage": clause_3_ok,
        "clause_4_synthetic_actuation_26_and_block_pos_ctrl": clause_4_ok,
        "clause_5_unpatched_no_sham_and_orth": clause_5_ok,
        "clause_6_causal_phrase_equivalence": clause_6_ok,
        "actuation_passed_count": act_passed,
        "actuation_expected_count": den["n_actuation_rows"],
        "causal_equiv_passed_count": ce_passed,
        "causal_equiv_expected_count": den["n_causal_equiv_rows"],
        "max_abs_sham_raw": max_sham["raw"],
        "max_abs_sham_spec": max_sham["spec"],
        "max_abs_sham_logp": max_sham["logp"],
        "max_abs_sham_phrase": max_sham["phrase_mean"],
        "max_abs_decomp_err": max_decomp,
        "max_sibling_leakage_science": max_sib_sci,
        "max_sibling_leakage_actuation": max_sib_act,
        "min_actuation_target_delta": min_act_delta,
        "min_actuation_readout_delta": min_act_readout_delta,
        "min_blocked_pos_drop": min_blk_pos_drop,
        "max_abs_wdown_orth_cos": max_orth_cos,
        "max_causal_equiv_err": max_ce_err,
        "failures": failures,
    }
    return cs0_report, {
        "unp_by_key": unp_by_key,
        "blk_by_key": blk_by_key,
        "shm_by_key": shm_by_key,
        "act_by_key": act_by_key,
        "ce_by_key": ce_by_key,
    }


# ---------------------------------------------------------------------------
# CS.1 — Per-Class Informative Transport & Head-Null Gate
# ---------------------------------------------------------------------------

def evaluate_cs1_transport(
    unp_by_key: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate CS.1 per-class transport on C1..C6 (72 rows per class x cell; mean-then-max over k=4 nulls)."""
    classes = {
        "slot_12": rcvs.PRIMARY_SLOT_UNITS,
        "concept_12": rcvs.PRIMARY_CONCEPT_UNITS,
    }
    by_class_cell: dict[str, dict[str, Any]] = {}
    for cls_name, uids in classes.items():
        by_class_cell[cls_name] = {}
        for cell in rcvs.BANK_CELLS:
            rows = [unp_by_key[(u, f"{s}_{cell}")] for u in uids for s in STORIES_6]
            mean_raw = _mean([float(r["transport"]["t_pos_raw"]) for r in rows])
            mean_spec = _mean([float(r["transport"]["t_pos_spec"]) for r in rows])
            mean_logp = _mean([float(r["transport"]["t_pos_logp"]) for r in rows])
            mean_phrase = _mean([float(r["transport"]["t_pos_phrase_mean"]) for r in rows])
            # Explicitly mean across 72 rows FIRST, then max over the k=4 null draws (L6 / Weaver #10).
            null_draw_means = [
                _mean([float(r["transport"]["wdown_pos_spec"][j]) for r in rows])
                for j in range(rcvs.N_NULL_DRAWS)
            ]
            max_null_draw_mean = max(null_draw_means)
            # Unit-level transport SD (useful pre-full spread check in Smoke without peeking at blocking!).
            unit_raw_means = [_mean([float(unp_by_key[(u, f"{s}_{cell}")]["transport"]["t_pos_raw"]) for s in STORIES_6]) for u in uids]
            unit_spec_means = [_mean([float(unp_by_key[(u, f"{s}_{cell}")]["transport"]["t_pos_spec"]) for s in STORIES_6]) for u in uids]
            sd_u_raw = float(np.std(unit_raw_means, ddof=1))
            sd_u_spec = float(np.std(unit_spec_means, ddof=1))

            peer_only_flip = bool(mean_spec >= CS1_TRANSPORT_FLOOR and (mean_raw < CS1_TRANSPORT_FLOOR or mean_logp <= 0.0))
            passed = bool(
                mean_spec >= CS1_TRANSPORT_FLOOR
                and mean_raw >= CS1_TRANSPORT_FLOOR
                and mean_logp > 0.0
                and mean_spec > max_null_draw_mean
            )
            by_class_cell[cls_name][cell] = {
                "n_rows": len(rows),
                "mean_raw": mean_raw,
                "mean_spec": mean_spec,
                "mean_logp_delta": mean_logp,
                "mean_phrase_mean_delta": mean_phrase,
                "unit_sd_raw": sd_u_raw,
                "unit_sd_spec": sd_u_spec,
                "wdown_null_draw_means_spec": null_draw_means,
                "max_wdown_null_draw_mean_spec": max_null_draw_mean,
                "peer_median_only_flip": peer_only_flip,
                "passed": passed,
            }

    # Determine contrast-level transport failures cleanly in deterministic order.
    def _first_fail(cells: tuple[str, ...]) -> str | None:
        for cls_name in ("slot_12", "concept_12"):
            for cell in cells:
                if not by_class_cell[cls_name][cell]["passed"]:
                    return f"uninformative (transport_gate_failed:{cls_name}|{cell})"
        return None

    csr_fail = _first_fail(("SELF", "ROLE_BARE"))
    csn_fail = _first_fail(("SELF", "NAME_BARE"))
    role_slot_fail = None if by_class_cell["slot_12"]["ROLE_BARE"]["passed"] else "uninformative (transport_gate_failed:slot_12|ROLE_BARE)"
    role_concept_fail = None if by_class_cell["concept_12"]["ROLE_BARE"]["passed"] else "uninformative (transport_gate_failed:concept_12|ROLE_BARE)"

    ratios = {
        cell: {
            "raw": by_class_cell["concept_12"][cell]["mean_raw"] / max(abs(by_class_cell["slot_12"][cell]["mean_raw"]), 1e-9),
            "spec": by_class_cell["concept_12"][cell]["mean_spec"] / max(abs(by_class_cell["slot_12"][cell]["mean_spec"]), 1e-9),
            "logp": by_class_cell["concept_12"][cell]["mean_logp_delta"] / max(abs(by_class_cell["slot_12"][cell]["mean_logp_delta"]), 1e-9),
        }
        for cell in rcvs.BANK_CELLS
    }
    all_6_passed = all(by_class_cell[c][cell]["passed"] for c in ("slot_12", "concept_12") for cell in rcvs.BANK_CELLS)
    return {
        "all_6_cells_passed": all_6_passed,
        "c_sr_transport_passed": csr_fail is None,
        "c_sr_transport_failure": csr_fail,
        "c_sn_transport_passed": csn_fail is None,
        "c_sn_transport_failure": csn_fail,
        "role_slot_failure": role_slot_fail,
        "role_concept_failure": role_concept_fail,
        "concept_to_slot_ratios": ratios,
        "by_class_cell": by_class_cell,
    }


def _row_wdown_even_odd_meds_spec(blk_row: dict[str, Any]) -> tuple[float, float]:
    sp = blk_row["spec"]
    d0 = float(sp["d0"])
    ev = [0.5 * (float(a) + float(b)) - d0 for a, b in zip(sp["wd_pos"], sp["wd_neg"])]
    od = [0.5 * (float(a) - float(b)) for a, b in zip(sp["wd_pos"], sp["wd_neg"])]
    return float(statistics.median(ev)), float(statistics.median(od))


def evaluate_cs1_head_null_full(
    blk_by_key: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Per-class head-null gate across all 26 heads x 2 classes x 3 cells (cap <= 0.15)."""
    classes = {"slot_12": rcvs.PRIMARY_SLOT_UNITS, "concept_12": rcvs.PRIMARY_CONCEPT_UNITS}
    per_head: dict[str, dict[str, Any]] = {}
    worst_abs = 0.0
    any_failed = False

    for hn in rcvs.HEAD_NAMES_26:
        per_head[hn] = {}
        for cls_name, uids in classes.items():
            per_head[hn][cls_name] = {}
            for cell in rcvs.BANK_CELLS:
                ev_meds, od_meds = [], []
                for u in uids:
                    for s in STORIES_6:
                        e_m, o_m = _row_wdown_even_odd_meds_spec(blk_by_key[(u, f"{s}_{cell}", hn)])
                        ev_meds.append(e_m)
                        od_meds.append(o_m)
                m_ev = _mean(ev_meds)
                m_od = _mean(od_meds)
                worst_abs = max(worst_abs, abs(m_ev), abs(m_od))
                ok = abs(m_ev) <= CS1_HEAD_NULL_FULL_CAP and abs(m_od) <= CS1_HEAD_NULL_FULL_CAP
                if not ok:
                    any_failed = True
                per_head[hn][cls_name][cell] = {
                    "mean_even_med_spec": m_ev,
                    "mean_odd_med_spec": m_od,
                    "passed": ok,
                }

    return {
        "cap": CS1_HEAD_NULL_FULL_CAP,
        "all_passed": not any_failed,
        "worst_abs_mean_med_spec": worst_abs,
        "by_head": per_head,
    }


def _head_null_failure_for_cells(
    head_null_report: dict[str, Any],
    head_name: str,
    classes: tuple[str, ...],
    cells: tuple[str, ...],
) -> str | None:
    bh = head_null_report["by_head"][head_name]
    for cls_name in classes:
        for cell in cells:
            if not bh[cls_name][cell]["passed"]:
                return f"uninformative (head_null_failed:{head_name}|{cls_name}|{cell})"
    return None


# ---------------------------------------------------------------------------
# Smoke-Only Verifier (CS.0 + Full 12-Pair CS.1 Transport + Prelim Head-Null;
#                      ZERO Scientific Verdicts!)
# ---------------------------------------------------------------------------

def verify_smoke_receipt_only(
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    pins: dict[str, str],
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Smoke verifier invoked by `--verify-smoke-only` and pre-load by `--mode full`.

    Checks CS.0 + full 12-pair C1..C6 CS.1 unpatched transport + preliminary head-null
    (`|med| <= 0.30`) on Smoke's 156 blocked rows. Emits ZERO head-gating contrasts or
    CS.2/CS.3/CS.4 scientific verdicts.
    """
    cs0_report, bound = validate_cs0_and_join(
        receipt=receipt,
        manifest=manifest,
        pins=pins,
        expect_full=False,
        allow_fixture=allow_fixture,
    )
    cs1_transport = evaluate_cs1_transport(bound["unp_by_key"])

    # Preliminary head-null check on Smoke's 156 blocked rows (P08 x C1 x 26 heads; cap <= 0.30).
    worst_prelim_even = 0.0
    worst_prelim_odd = 0.0
    prelim_breaches: list[str] = []
    for k, row in bound["blk_by_key"].items():
        e_m, o_m = _row_wdown_even_odd_meds_spec(row)
        worst_prelim_even = max(worst_prelim_even, abs(e_m))
        worst_prelim_odd = max(worst_prelim_odd, abs(o_m))
        if abs(e_m) > CS1_HEAD_NULL_SMOKE_CAP or abs(o_m) > CS1_HEAD_NULL_SMOKE_CAP:
            prelim_breaches.append(f"{k}:even={e_m:.4f},odd={o_m:.4f}")
    prelim_ok = len(prelim_breaches) == 0

    # Amendment 1 (2026-11-22, Claude #20): prelim 156-row single-row check is a reported, non-gating
    # early warning matching §5 line 184; smoke_gate_passed gates strictly on CS.0 + CS.1 12-pair transport
    # (Full's registered 0.15 72-row per-class CS.1 head-null gate remains 100% untouched).
    smoke_gate_passed = bool(cs0_report["passed"] and cs1_transport["all_6_cells_passed"])
    meta = receipt["meta"]
    return {
        "artifact_type": "concept_vs_slot_smoke_verification",
        "execution_provenance": meta["execution_provenance"],
        "smoke_gate_passed": smoke_gate_passed,
        "smoke_technical_check_passed": bool(cs0_report["passed"]),
        "smoke_transport_gate_passed": bool(cs1_transport["all_6_cells_passed"]),
        "smoke_prelim_head_null_early_warning_clear": prelim_ok,
        "counts": {
            "n_unpatched_rows": meta["n_unpatched_rows"],
            "n_blocked_rows": meta["n_blocked_rows"],
            "n_sham_rows": meta["n_sham_rows"],
            "n_actuation_rows": meta["n_actuation_rows"],
            "n_causal_equiv_rows": meta["n_causal_equiv_rows"],
            "n_b11_forwards": meta["n_b11_forwards"],
            "n_b1_forwards": meta["n_b1_forwards"],
        },
        "CS0_technical_check": cs0_report,
        "CS1_unpatched_transport_12_pairs": cs1_transport,
        "CS1_prelim_head_null_156_rows": {
            "cap": CS1_HEAD_NULL_SMOKE_CAP,
            "n_rows_checked": len(bound["blk_by_key"]),
            "worst_abs_even_med_spec": worst_prelim_even,
            "worst_abs_odd_med_spec": worst_prelim_odd,
            "passed": prelim_ok,
            "breaches": prelim_breaches[:20],
        },
        "timing": meta["timing"],
    }


# ---------------------------------------------------------------------------
# Contrast Accessors (Exact Sign Conventions from v3.1 §3.2)
# ---------------------------------------------------------------------------

def _drop_pos(blk_by_key: dict[tuple[str, str, str], dict[str, Any]], u: str, s: str, cell: str, h: str, ch: str) -> float:
    return float(blk_by_key[(u, f"{s}_{cell}", h)][ch]["d_pos"])


def _drop_neg(blk_by_key: dict[tuple[str, str, str], dict[str, Any]], u: str, s: str, cell: str, h: str, ch: str) -> float:
    return float(blk_by_key[(u, f"{s}_{cell}", h)][ch]["d_neg"])


def _c_sr(blk_by_key: dict[tuple[str, str, str], dict[str, Any]], u: str, s: str, h: str, ch: str) -> float:
    return _drop_pos(blk_by_key, u, s, "SELF", h, ch) - _drop_pos(blk_by_key, u, s, "ROLE_BARE", h, ch)


def _c_sn(blk_by_key: dict[tuple[str, str, str], dict[str, Any]], u: str, s: str, h: str, ch: str) -> float:
    return _drop_pos(blk_by_key, u, s, "SELF", h, ch) - _drop_pos(blk_by_key, u, s, "NAME_BARE", h, ch)


def _s_odd_role(blk_by_key: dict[tuple[str, str, str], dict[str, Any]], u: str, s: str, h: str, ch: str) -> float:
    # Exact Part B score_causal_attn.py line 701 convention: 0.5 * (drop_-0.8 - drop_+0.8).
    return 0.5 * (_drop_neg(blk_by_key, u, s, "ROLE_BARE", h, ch) - _drop_pos(blk_by_key, u, s, "ROLE_BARE", h, ch))


def _pair_delta_matrix(
    blk_by_key: dict[tuple[str, str, str], dict[str, Any]],
    pairs: list[dict[str, Any]],
    h: str,
    ch: str,
    contrast: str = "C_SR",
) -> np.ndarray:
    fn = _c_sr if contrast == "C_SR" else _c_sn
    mat = np.zeros((len(pairs), len(STORIES_6)), dtype=np.float64)
    for pi, p in enumerate(pairs):
        for si, s in enumerate(STORIES_6):
            mat[pi, si] = fn(blk_by_key, p["concept_unit"], s, h, ch) - fn(blk_by_key, p["slot_unit"], s, h, ch)
    return mat


def _unit_contrast_matrix(
    blk_by_key: dict[tuple[str, str, str], dict[str, Any]],
    units: tuple[str, ...] | list[str],
    h: str,
    ch: str,
    contrast: str = "C_SR",
) -> np.ndarray:
    fn = _c_sr if contrast == "C_SR" else _c_sn
    mat = np.zeros((len(units), len(STORIES_6)), dtype=np.float64)
    for ui, u in enumerate(units):
        for si, s in enumerate(STORIES_6):
            mat[ui, si] = fn(blk_by_key, u, s, h, ch)
    return mat


# ---------------------------------------------------------------------------
# CS.2 — Primary Concept-vs-Slot Head-Gating Interaction (L8H2, L8H15)
# ---------------------------------------------------------------------------

def _score_cs2_head_channel(
    head_name: str,
    ch: str,
    manifest: dict[str, Any],
    unp_by_key: dict[tuple[str, str], dict[str, Any]],
    blk_by_key: dict[tuple[str, str, str], dict[str, Any]],
    csn_gate_failure: str | None,
) -> dict[str, Any]:
    pairs = manifest["meta"]["primary_pairs"]
    units_meta = manifest["units"]

    # 1. Primary 12-pair Delta_class(C_SR) + studentized pair-cluster bootstrap & exact t_11.
    mat_delta = _pair_delta_matrix(blk_by_key, pairs, head_name, ch, "C_SR")
    boot_delta = studentized_cluster_bootstrap(mat_delta, alpha=0.05, n_boot=BOOT_DRAWS, seed=_det_seed(f"cs2|{head_name}|{ch}|delta"))
    delta_bar = boot_delta["mean"]
    lo, hi = boot_delta["boot_ci"]
    pair_means = {pairs[i]["pair_id"]: boot_delta["cluster_means"][i] for i in range(len(pairs))}
    n_pairs_neg = sum(1 for v in boot_delta["cluster_means"] if v < 0.0)
    n_pairs_pos = sum(1 for v in boot_delta["cluster_means"] if v > 0.0)

    # Per-class C_SR & C_SN matrices.
    mat_slot_csr = _unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_SLOT_UNITS, head_name, ch, "C_SR")
    mat_conc_csr = _unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_CONCEPT_UNITS, head_name, ch, "C_SR")
    mean_slot_csr = float(mat_slot_csr.mean())
    mean_conc_csr = float(mat_conc_csr.mean())

    mat_slot_csn = _unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_SLOT_UNITS, head_name, ch, "C_SN")
    mat_conc_csn = _unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_CONCEPT_UNITS, head_name, ch, "C_SN")
    mean_slot_csn = float(mat_slot_csn.mean())
    mean_conc_csn = float(mat_conc_csn.mean())
    delta_bar_csn = mean_conc_csn - mean_slot_csn

    # 2. Interaction-level control gate (DiD against all 5 L8 controls, fixes H1).
    ctrl_deltas: dict[str, float] = {}
    ctrl_slot_csr: dict[str, float] = {}
    ctrl_conc_csr: dict[str, float] = {}
    ctrl_slot_csn: dict[str, float] = {}
    ctrl_conc_csn: dict[str, float] = {}
    for c in L8_CTRL_HEADS:
        ctrl_deltas[c] = float(_pair_delta_matrix(blk_by_key, pairs, c, ch, "C_SR").mean())
        ctrl_slot_csr[c] = float(_unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_SLOT_UNITS, c, ch, "C_SR").mean())
        ctrl_conc_csr[c] = float(_unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_CONCEPT_UNITS, c, ch, "C_SR").mean())
        ctrl_slot_csn[c] = float(_unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_SLOT_UNITS, c, ch, "C_SN").mean())
        ctrl_conc_csn[c] = float(_unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_CONCEPT_UNITS, c, ch, "C_SN").mean())

    ctrl_min_delta = min(ctrl_deltas.values())
    ctrl_max_delta = max(ctrl_deltas.values())
    ctrl_max_slot_head = max(L8_CTRL_HEADS, key=lambda c: ctrl_slot_csr[c])
    ctrl_max_conc_head = max(L8_CTRL_HEADS, key=lambda c: ctrl_conc_csr[c])
    ctrl_max_slot_csr = ctrl_slot_csr[ctrl_max_slot_head]
    ctrl_max_conc_csr = ctrl_conc_csr[ctrl_max_conc_head]
    ctrl_max_slot_csn = max(ctrl_slot_csn.values())
    ctrl_max_conc_csn = max(ctrl_conc_csn.values())

    beats_ctrl_slot_interaction = bool(
        delta_bar <= ctrl_min_delta - CS2_CTRL_INTERACTION_MARGIN and mean_slot_csr >= CS2_LEVEL_FLOOR
    )
    beats_ctrl_concept_interaction = bool(
        delta_bar >= ctrl_max_delta + CS2_CTRL_INTERACTION_MARGIN and mean_conc_csr >= CS2_LEVEL_FLOOR
    )
    beats_ctrl_shared_level = bool(
        mean_slot_csr >= CS2_LEVEL_FLOOR
        and mean_conc_csr >= CS2_LEVEL_FLOOR
        and mean_slot_csr > ctrl_max_slot_csr
        and mean_conc_csr > ctrl_max_conc_csr
        and (mean_slot_csn - ctrl_max_slot_csn) >= CS2_SHARED_CSN_MARGIN
        and (mean_conc_csn - ctrl_max_conc_csn) >= CS2_SHARED_CSN_MARGIN
    )

    # Branch 4 per-class level-minus-control 95% cluster CI upper bounds.
    mat_slot_excess = mat_slot_csr - _unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_SLOT_UNITS, ctrl_max_slot_head, ch, "C_SR")
    mat_conc_excess = mat_conc_csr - _unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_CONCEPT_UNITS, ctrl_max_conc_head, ch, "C_SR")
    boot_slot_ex = studentized_cluster_bootstrap(mat_slot_excess, alpha=0.05, n_boot=BOOT_DRAWS, seed=_det_seed(f"cs2|{head_name}|{ch}|ex_slot"))
    boot_conc_ex = studentized_cluster_bootstrap(mat_conc_excess, alpha=0.05, n_boot=BOOT_DRAWS, seed=_det_seed(f"cs2|{head_name}|{ch}|ex_conc"))

    # 3. Length-matched C_SN concordance checks.
    csn_ok = csn_gate_failure is None
    csn_conc_slot = bool(
        csn_ok
        and delta_bar_csn < 0.0
        and mean_slot_csn > 0.0
        and mean_slot_csn > ctrl_max_slot_csn
    )
    csn_conc_concept = bool(
        csn_ok
        and delta_bar_csn > 0.0
        and mean_conc_csn > 0.0
        and mean_conc_csn > ctrl_max_conc_csn
    )
    csn_conc_shared = bool(
        csn_ok
        and mean_slot_csn > 0.0
        and mean_conc_csn > 0.0
        and mean_slot_csn > ctrl_max_slot_csn
        and mean_conc_csn > ctrl_max_conc_csn
    )

    # 4. Active persona_kind confound guard (H4): 5 same-kind pairs + n=23 7-param HC3 OLS.
    delta_same_kind = _mean([pair_means[pid] for pid in SAME_KIND_PAIR_IDS])
    # Shared 5 persona_kind strata (excluding singleton 'another AI' unit p92-38).
    shared_pks = ("animal", "deity or spirit", "fictional character", "object or place")  # 'human' = reference
    ols_rows_x, ols_rows_y, ols_units = [], [], []
    for u in rcvs.PRIMARY_UNITS_24:
        pk = units_meta[u]["persona_kind"]
        if pk == "another AI":
            continue
        is_c = 1.0 if units_meta[u]["kind_haiku"] == "concept" else 0.0
        y_u = _mean([_c_sr(blk_by_key, u, s, head_name, ch) for s in STORIES_6])
        base_raw_u = _mean([float(unp_by_key[(u, f"{s}_SELF")]["transport"]["base_raw"]) for s in STORIES_6])
        row_x = [1.0, is_c] + [1.0 if pk == spk else 0.0 for spk in shared_pks] + [base_raw_u]
        ols_rows_x.append(row_x)
        ols_rows_y.append(y_u)
        ols_units.append(u)
    within_kind_ols = ols_hc3(np.asarray(ols_rows_x), np.asarray(ols_rows_y))
    beta_within_kind = within_kind_ols["beta"][1]
    se_within_kind = within_kind_ols["se_hc3"][1]

    kind_guard_slot = bool(delta_same_kind < 0.0 and beta_within_kind < 0.0)
    kind_guard_concept = bool(delta_same_kind > 0.0 and beta_within_kind > 0.0)

    # 5. Active transport-scale guard & Weaver's within-class regression (M3 & Weaver #10).
    t_key = f"t_pos_{ch}"
    mean_tr_slot_self = _mean([float(unp_by_key[(u, f"{s}_SELF")]["transport"][t_key]) for u in rcvs.PRIMARY_SLOT_UNITS for s in STORIES_6])
    mean_tr_slot_role = _mean([float(unp_by_key[(u, f"{s}_ROLE_BARE")]["transport"][t_key]) for u in rcvs.PRIMARY_SLOT_UNITS for s in STORIES_6])
    mean_tr_conc_self = _mean([float(unp_by_key[(u, f"{s}_SELF")]["transport"][t_key]) for u in rcvs.PRIMARY_CONCEPT_UNITS for s in STORIES_6])
    mean_tr_conc_role = _mean([float(unp_by_key[(u, f"{s}_ROLE_BARE")]["transport"][t_key]) for u in rcvs.PRIMARY_CONCEPT_UNITS for s in STORIES_6])

    denom_slot = 0.5 * (mean_tr_slot_self + mean_tr_slot_role)
    denom_conc = 0.5 * (mean_tr_conc_self + mean_tr_conc_role)
    f_sr_ratio_slot = mean_slot_csr / denom_slot if abs(denom_slot) > 1e-9 else float("nan")
    f_sr_ratio_conc = mean_conc_csr / denom_conc if abs(denom_conc) > 1e-9 else float("nan")
    delta_f_ratio = f_sr_ratio_conc - f_sr_ratio_slot

    # Unit-story fractional gating F_SR (descriptive, on cells with T_SELF >= 0.50 and T_ROLE >= 0.50).
    def _unit_story_f_sr(uids: tuple[str, ...]) -> tuple[float | None, int]:
        vals = []
        for u in uids:
            for s in STORIES_6:
                ts = float(unp_by_key[(u, f"{s}_SELF")]["transport"][t_key])
                tr = float(unp_by_key[(u, f"{s}_ROLE_BARE")]["transport"][t_key])
                if ts >= CS1_TRANSPORT_FLOOR and tr >= CS1_TRANSPORT_FLOOR:
                    vals.append(_drop_pos(blk_by_key, u, s, "SELF", head_name, ch) / ts - _drop_pos(blk_by_key, u, s, "ROLE_BARE", head_name, ch) / tr)
        return (_mean(vals) if vals else None), len(vals)

    f_sr_cell_slot, n_f_slot = _unit_story_f_sr(rcvs.PRIMARY_SLOT_UNITS)
    f_sr_cell_conc, n_f_conc = _unit_story_f_sr(rcvs.PRIMARY_CONCEPT_UNITS)

    # Weaver's within-class transport regression & joint ANCOVA.
    def _unit_mean_tr_sr(u: str) -> float:
        return 0.5 * (
            _mean([float(unp_by_key[(u, f"{s}_SELF")]["transport"][t_key]) for s in STORIES_6])
            + _mean([float(unp_by_key[(u, f"{s}_ROLE_BARE")]["transport"][t_key]) for s in STORIES_6])
        )

    slot_u_y = [_mean([_c_sr(blk_by_key, u, s, head_name, ch) for s in STORIES_6]) for u in rcvs.PRIMARY_SLOT_UNITS]
    slot_u_tr = [_unit_mean_tr_sr(u) for u in rcvs.PRIMARY_SLOT_UNITS]
    conc_u_y = [_mean([_c_sr(blk_by_key, u, s, head_name, ch) for s in STORIES_6]) for u in rcvs.PRIMARY_CONCEPT_UNITS]
    conc_u_tr = [_unit_mean_tr_sr(u) for u in rcvs.PRIMARY_CONCEPT_UNITS]

    def _safe_ols(X_m: np.ndarray, y_v: np.ndarray) -> dict[str, Any]:
        try:
            return ols_hc3(X_m, y_v)
        except ValueError as exc:
            return {"error": str(exc), "beta": [float("nan")] * X_m.shape[1], "se_hc3": [float("nan")] * X_m.shape[1], "r2": float("nan")}

    ols_tr_slot = _safe_ols(np.column_stack([np.ones(12), slot_u_tr]), np.asarray(slot_u_y))
    ols_tr_conc = _safe_ols(np.column_stack([np.ones(12), conc_u_tr]), np.asarray(conc_u_y))
    ancova_X = np.column_stack([
        np.ones(24),
        np.array([0.0] * 12 + [1.0] * 12),
        np.array(slot_u_tr + conc_u_tr),
    ])
    ancova_tr = _safe_ols(ancova_X, np.asarray(slot_u_y + conc_u_y))
    beta_class_adj = ancova_tr["beta"][1]
    se_class_adj = ancova_tr["se_hc3"][1]

    transport_guard_slot = bool(
        math.isfinite(delta_f_ratio) and math.isfinite(beta_class_adj) and delta_f_ratio < 0.0 and beta_class_adj < 0.0
    )
    transport_guard_concept = bool(
        math.isfinite(delta_f_ratio) and math.isfinite(beta_class_adj) and delta_f_ratio > 0.0 and beta_class_adj > 0.0
    )

    # 6. Strict ordered disjoint per-channel verdict tree (1 -> 2 -> 3 -> 4 -> 5).
    b1_core = bool(hi < -CS2_EPS and n_pairs_neg >= CS2_MIN_PAIR_AGREE and beats_ctrl_slot_interaction)
    b2_core = bool(lo > CS2_EPS and n_pairs_pos >= CS2_MIN_PAIR_AGREE and beats_ctrl_concept_interaction)
    b3_core = bool(lo >= -CS2_EPS and hi <= CS2_EPS and beats_ctrl_shared_level)

    verdict: str
    downgrade_reason: str | None = None
    if b1_core:
        if csn_gate_failure is not None:
            # F4: preserve exact registered CS.1 consequence label when NAME_BARE transport/head-null fails.
            verdict = csn_gate_failure
            downgrade_reason = "branch1_cs1_csn_gate_failed"
        elif not csn_conc_slot:
            verdict = "length_confound_unconfirmed"
            downgrade_reason = "branch1_failed_csn_concordance"
        elif not kind_guard_slot:
            verdict = "inconclusive_kind_confounded"
            downgrade_reason = "branch1_failed_within_kind_guard"
        elif not transport_guard_slot:
            verdict = "transport_scale_confounded"
            downgrade_reason = "branch1_failed_transport_scale_guard"
        else:
            verdict = "slot_stronger_gating"
    elif b2_core:
        if csn_gate_failure is not None:
            verdict = csn_gate_failure
            downgrade_reason = "branch2_cs1_csn_gate_failed"
        elif not csn_conc_concept:
            verdict = "length_confound_unconfirmed"
            downgrade_reason = "branch2_failed_csn_concordance"
        elif not kind_guard_concept:
            verdict = "inconclusive_kind_confounded"
            downgrade_reason = "branch2_failed_within_kind_guard"
        elif not transport_guard_concept:
            verdict = "transport_scale_confounded"
            downgrade_reason = "branch2_failed_transport_scale_guard"
        else:
            verdict = "concept_stronger_gating"
    elif b3_core:
        if csn_gate_failure is not None:
            verdict = csn_gate_failure
            downgrade_reason = "branch3_cs1_csn_gate_failed"
        elif not csn_conc_shared:
            verdict = "length_confound_unconfirmed"
            downgrade_reason = "branch3_failed_csn_concordance"
        else:
            verdict = "practically_equivalent_subject_gating"
    elif (
        lo >= -CS2_EPS
        and hi <= CS2_EPS
        and boot_slot_ex["boot_ci"][1] <= CS2_EPS
        and boot_conc_ex["boot_ci"][1] <= CS2_EPS
    ):
        verdict = "both_weak_or_null"
    else:
        verdict = "inconclusive_wide_interval"

    out: dict[str, Any] = {
        "channel": CHANNEL_DISPLAY[ch],
        "verdict": verdict,
        "downgrade_reason": downgrade_reason,
        "delta_class_C_SR": boot_delta,
        "pair_means_C_SR": pair_means,
        "n_pairs_negative": n_pairs_neg,
        "n_pairs_positive": n_pairs_pos,
        "class_means_C_SR": {"slot_12": mean_slot_csr, "concept_12": mean_conc_csr},
        "class_means_C_SN": {"slot_12": mean_slot_csn, "concept_12": mean_conc_csn, "delta_concept_minus_slot": delta_bar_csn},
        "control_gate": {
            "ctrl_deltas_C_SR": ctrl_deltas,
            "ctrl_min_delta": ctrl_min_delta,
            "ctrl_max_delta": ctrl_max_delta,
            "ctrl_slot_C_SR": ctrl_slot_csr,
            "ctrl_concept_C_SR": ctrl_conc_csr,
            "ctrl_max_slot_head": ctrl_max_slot_head,
            "ctrl_max_slot_C_SR": ctrl_max_slot_csr,
            "ctrl_max_concept_head": ctrl_max_conc_head,
            "ctrl_max_concept_C_SR": ctrl_max_conc_csr,
            "level_excess_slot": mean_slot_csr - ctrl_max_slot_csr,
            "level_excess_concept": mean_conc_csr - ctrl_max_conc_csr,
            "level_excess_slot_boot_ci": boot_slot_ex["boot_ci"],
            "level_excess_concept_boot_ci": boot_conc_ex["boot_ci"],
            "ctrl_max_slot_C_SN": ctrl_max_slot_csn,
            "ctrl_max_concept_C_SN": ctrl_max_conc_csn,
            "beats_ctrl_slot_interaction": beats_ctrl_slot_interaction,
            "beats_ctrl_concept_interaction": beats_ctrl_concept_interaction,
            "beats_ctrl_shared_level": beats_ctrl_shared_level,
        },
        "guards": {
            "csn_concordant_slot": csn_conc_slot,
            "csn_concordant_concept": csn_conc_concept,
            "csn_concordant_shared": csn_conc_shared,
            "delta_same_kind_5pairs": delta_same_kind,
            "within_kind_ols_n23": {
                "n": within_kind_ols["n"],
                "p": within_kind_ols["p"],
                "df_resid": within_kind_ols["df_resid"],
                "beta_within_kind": beta_within_kind,
                "se_hc3": se_within_kind,
                "r2": within_kind_ols["r2"],
            },
            "kind_guard_slot": kind_guard_slot,
            "kind_guard_concept": kind_guard_concept,
            "transport_scale": {
                "F_SR_ratio_slot": f_sr_ratio_slot,
                "F_SR_ratio_concept": f_sr_ratio_conc,
                "delta_F_ratio": delta_f_ratio,
                "F_SR_cell_mean_slot": f_sr_cell_slot,
                "F_SR_cell_n_slot": n_f_slot,
                "F_SR_cell_mean_concept": f_sr_cell_conc,
                "F_SR_cell_n_concept": n_f_conc,
                "weaver_slope_slot": ols_tr_slot["beta"][1],
                "weaver_se_slot": ols_tr_slot["se_hc3"][1],
                "weaver_r2_slot": ols_tr_slot["r2"],
                "weaver_slope_concept": ols_tr_conc["beta"][1],
                "weaver_se_concept": ols_tr_conc["se_hc3"][1],
                "weaver_r2_concept": ols_tr_conc["r2"],
                "weaver_ancova_beta_class_adj": beta_class_adj,
                "weaver_ancova_se_hc3": se_class_adj,
                "weaver_ancova_r2": ancova_tr["r2"],
            },
            "transport_guard_slot": transport_guard_slot,
            "transport_guard_concept": transport_guard_concept,
        },
    }
    if verdict in ("practically_equivalent_subject_gating", "both_weak_or_null"):
        out["mandatory_report_wording"] = MANDATORY_NULL_WORDING
    return out


def _evaluate_cs2_sensitivities(
    head_name: str,
    manifest: dict[str, Any],
    unp_by_key: dict[tuple[str, str], dict[str, Any]],
    blk_by_key: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    pairs = manifest["meta"]["primary_pairs"]
    units_meta = manifest["units"]

    # (a) Full-phrase readout: PhraseLogProbMean across all 12 pairs.
    mat_phrase = _pair_delta_matrix(blk_by_key, pairs, head_name, "phrase_mean", "C_SR")
    boot_phrase = studentized_cluster_bootstrap(mat_phrase, alpha=0.05, n_boot=BOOT_DRAWS, seed=_det_seed(f"cs2|{head_name}|phrase_mean"))

    # (b) Frozen clean-token exclusion: (i) 5 intact pairs exact paired t_4, (ii) 15 surviving units Welch t.
    intact_pairs = [p for p in pairs if p["pair_id"] in INTACT_CLEAN_PAIR_IDS]
    intact_clean_by_ch: dict[str, Any] = {}
    surviving_slot = [u for u in rcvs.PRIMARY_SLOT_UNITS if not units_meta[u]["clean_token_excluded"]]
    surviving_conc = [u for u in rcvs.PRIMARY_CONCEPT_UNITS if not units_meta[u]["clean_token_excluded"]]
    assert len(intact_pairs) == 5 and len(surviving_slot) == 8 and len(surviving_conc) == 7

    for ch in ("raw", "spec", "logp", "phrase_mean"):
        p_vec = [
            _mean([_c_sr(blk_by_key, p["concept_unit"], s, head_name, ch) - _c_sr(blk_by_key, p["slot_unit"], s, head_name, ch) for s in STORIES_6])
            for p in intact_pairs
        ]
        m5 = float(np.mean(p_vec))
        se5 = float(np.std(p_vec, ddof=1) / math.sqrt(5))
        t4 = float(sp_stats.t.ppf(0.975, df=4))
        s_vals = [_mean([_c_sr(blk_by_key, u, s, head_name, ch) for s in STORIES_6]) for u in surviving_slot]
        c_vals = [_mean([_c_sr(blk_by_key, u, s, head_name, ch) for s in STORIES_6]) for u in surviving_conc]
        intact_clean_by_ch[CHANNEL_DISPLAY[ch]] = {
            "intact_5_pairs": {
                "pair_ids": [p["pair_id"] for p in intact_pairs],
                "pair_means": {p["pair_id"]: v for p, v in zip(intact_pairs, p_vec)},
                "mean": m5,
                "se": se5,
                "exact_t4_ci": [m5 - t4 * se5, m5 + t4 * se5],
            },
            "unpaired_15_units_welch": welch_two_sample_t(c_vals, s_vals),
        }

    # (c) Baseline-logit diagnostic: corr(C_SR, base_raw) pooled & within-class + OLS C_SR ~ is_concept + base_raw.
    u_base_slot = [_mean([float(unp_by_key[(u, f"{s}_SELF")]["transport"]["base_raw"]) for s in STORIES_6]) for u in rcvs.PRIMARY_SLOT_UNITS]
    u_base_conc = [_mean([float(unp_by_key[(u, f"{s}_SELF")]["transport"]["base_raw"]) for s in STORIES_6]) for u in rcvs.PRIMARY_CONCEPT_UNITS]
    u_csr_slot = [_mean([_c_sr(blk_by_key, u, s, head_name, "raw") for s in STORIES_6]) for u in rcvs.PRIMARY_SLOT_UNITS]
    u_csr_conc = [_mean([_c_sr(blk_by_key, u, s, head_name, "raw") for s in STORIES_6]) for u in rcvs.PRIMARY_CONCEPT_UNITS]
    X_b = np.column_stack([np.ones(24), np.array([0.0] * 12 + [1.0] * 12), np.array(u_base_slot + u_base_conc)])
    try:
        base_ols = ols_hc3(X_b, np.asarray(u_csr_slot + u_csr_conc))
    except ValueError as exc:
        base_ols = {"error": str(exc)}
    baseline_diag = {
        "corr_pooled_n24": _pearson_r(u_base_slot + u_base_conc, u_csr_slot + u_csr_conc),
        "corr_slot_n12": _pearson_r(u_base_slot, u_csr_slot),
        "corr_concept_n12": _pearson_r(u_base_conc, u_csr_conc),
        "ols_c_sr_on_concept_and_base_raw": base_ols,
    }

    # (d) Reader-agreed 7-pair stratum, Part B 5-core-slot bridge, exact-token cross-layer witness.
    ra_by_ch: dict[str, Any] = {}
    for ch in ("raw", "spec", "logp", "phrase_mean"):
        pv = {
            pid: _mean([_c_sr(blk_by_key, cu, s, head_name, ch) - _c_sr(blk_by_key, su, s, head_name, ch) for s in STORIES_6])
            for pid, su, cu in READER_AGREED_PAIRS
        }
        vals = list(pv.values())
        m7 = float(np.mean(vals))
        se7 = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
        t6 = float(sp_stats.t.ppf(0.975, df=len(vals) - 1))
        ra_by_ch[CHANNEL_DISPLAY[ch]] = {"mean": m7, "se": se7, "exact_t6_ci": [m7 - t6 * se7, m7 + t6 * se7], "pair_means": pv}

    part_b_bridge = {
        CHANNEL_DISPLAY[ch]: {
            "mean_C_SR": _mean([_c_sr(blk_by_key, u, s, head_name, ch) for u in PART_B_BRIDGE_UNITS for s in STORIES_6]),
            "mean_C_SN": _mean([_c_sn(blk_by_key, u, s, head_name, ch) for u in PART_B_BRIDGE_UNITS for s in STORIES_6]),
            "per_unit_C_SR": {u: _mean([_c_sr(blk_by_key, u, s, head_name, ch) for s in STORIES_6]) for u in PART_B_BRIDGE_UNITS},
        }
        for ch in ("raw", "spec", "logp")
    }

    exact_dog_witness = {
        CHANNEL_DISPLAY[ch]: {
            "p92-08_L4_concept_dog_C_SR": _mean([_c_sr(blk_by_key, "p92-08", s, head_name, ch) for s in STORIES_6]),
            "p92-09_L6_slot_dog_C_SR": _mean([_c_sr(blk_by_key, "p92-09", s, head_name, ch) for s in STORIES_6]),
            "diff_concept_minus_slot": _mean([_c_sr(blk_by_key, "p92-08", s, head_name, ch) - _c_sr(blk_by_key, "p92-09", s, head_name, ch) for s in STORIES_6]),
        }
        for ch in ("raw", "spec", "logp", "phrase_mean")
    }

    return {
        "phrase_log_prob_mean_12_pairs": boot_phrase,
        "clean_token_sensitivity": intact_clean_by_ch,
        "baseline_logit_diagnostic": baseline_diag,
        "reader_agreed_7_pairs": ra_by_ch,
        "part_b_5_core_slot_bridge": part_b_bridge,
        "exact_token_dog_witness": exact_dog_witness,
    }


def evaluate_cs2(
    manifest: dict[str, Any],
    unp_by_key: dict[tuple[str, str], dict[str, Any]],
    blk_by_key: dict[tuple[str, str, str], dict[str, Any]],
    cs0_passed: bool,
    cs1_transport: dict[str, Any],
    cs1_head_null: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for hname in ("L8H2", "L8H15"):
        if not cs0_passed:
            out[hname] = {"overall_verdict": CS0_FAIL_SCI, "status": "gated_by_cs0"}
            continue
        if cs1_transport["c_sr_transport_failure"] is not None:
            out[hname] = {"overall_verdict": cs1_transport["c_sr_transport_failure"], "status": "gated_by_cs1_transport"}
            continue
        hn_fail = _head_null_failure_for_cells(cs1_head_null, hname, ("slot_12", "concept_12"), ("SELF", "ROLE_BARE"))
        if hn_fail is not None:
            out[hname] = {"overall_verdict": hn_fail, "status": "gated_by_cs1_head_null"}
            continue

        csn_gate_failure = (
            cs1_transport["c_sn_transport_failure"]
            or _head_null_failure_for_cells(cs1_head_null, hname, ("slot_12", "concept_12"), ("SELF", "NAME_BARE"))
        )
        per_ch = {
            CHANNEL_DISPLAY[ch]: _score_cs2_head_channel(hname, ch, manifest, unp_by_key, blk_by_key, csn_gate_failure)
            for ch in CO_PRIMARY_CHANNELS
        }

        # Peer-median collapse guard diagnostic (F5: strictly directional positive Branches 1 and 2).
        pm_slot = _mean([_c_sr(blk_by_key, u, s, hname, "peer_med") for u in rcvs.PRIMARY_SLOT_UNITS for s in STORIES_6])
        pm_conc = _mean([_c_sr(blk_by_key, u, s, hname, "peer_med") for u in rcvs.PRIMARY_CONCEPT_UNITS for s in STORIES_6])
        sp_slot = per_ch["SpecLogit"]["class_means_C_SR"]["slot_12"]
        sp_conc = per_ch["SpecLogit"]["class_means_C_SR"]["concept_12"]
        rw_slot = per_ch["RawLogit"]["class_means_C_SR"]["slot_12"]
        rw_conc = per_ch["RawLogit"]["class_means_C_SR"]["concept_12"]
        pm_share_slot = abs(pm_slot) / max(abs(sp_slot), 1e-9)
        pm_share_conc = abs(pm_conc) / max(abs(sp_conc), 1e-9)
        spec_v = per_ch["SpecLogit"]["verdict"]
        peer_median_only_gating = bool(
            (spec_v == "slot_stronger_gating" and rw_slot < CS2_LEVEL_FLOOR and pm_share_slot > CS2_PEER_MED_SHARE_CAP)
            or (spec_v == "concept_stronger_gating" and rw_conc < CS2_LEVEL_FLOOR and pm_share_conc > CS2_PEER_MED_SHARE_CAP)
        )

        v_raw = per_ch["RawLogit"]["verdict"]
        v_spec = per_ch["SpecLogit"]["verdict"]
        v_logp = per_ch["TargetLogProb"]["verdict"]
        overall = v_raw if (v_raw == v_spec == v_logp) else "inconclusive_readout_disagreement"

        head_out: dict[str, Any] = {
            "overall_verdict": overall,
            "co_primary_verdicts": {"RawLogit": v_raw, "SpecLogit": v_spec, "TargetLogProb": v_logp},
            "peer_median_guard": {
                "peer_med_share_slot": pm_share_slot,
                "peer_med_share_concept": pm_share_conc,
                "peer_median_only_gating": peer_median_only_gating,
            },
            "per_channel": per_ch,
            "sensitivities": _evaluate_cs2_sensitivities(hname, manifest, unp_by_key, blk_by_key),
        }
        if overall in ("practically_equivalent_subject_gating", "both_weak_or_null"):
            head_out["mandatory_report_wording"] = MANDATORY_NULL_WORDING
        out[hname] = head_out
    return out


# ---------------------------------------------------------------------------
# CS.3 — Held-Out Confirmation of 7 Late Lens-Screened Heads (L11, L13, L14)
# ---------------------------------------------------------------------------

def evaluate_cs3(
    manifest: dict[str, Any],
    blk_by_key: dict[tuple[str, str, str], dict[str, Any]],
    cs0_passed: bool,
    cs1_transport: dict[str, Any],
    cs1_head_null: dict[str, Any],
) -> dict[str, Any]:
    head_meta_by_name = {h["name"]: h for h in manifest["heads"]}
    pairs = manifest["meta"]["primary_pairs"]
    results: dict[str, Any] = {}
    n_confirmed = 0
    n_length_confounded = 0

    for hname in CS3_CANDIDATES:
        hm = head_meta_by_name[hname]
        if not cs0_passed:
            results[hname] = {"verdict": CS0_FAIL_SCI, "status": "gated_by_cs0"}
            continue
        if cs1_transport["c_sr_transport_failure"] is not None:
            results[hname] = {"verdict": cs1_transport["c_sr_transport_failure"], "status": "gated_by_cs1_transport"}
            continue
        hn_fail = _head_null_failure_for_cells(cs1_head_null, hname, ("slot_12", "concept_12"), ("SELF", "ROLE_BARE"))
        if hn_fail is not None:
            results[hname] = {"verdict": hn_fail, "status": "gated_by_cs1_head_null"}
            continue

        pred_sr = int(hm["pred_c_sr_sign"])
        pred_sn = int(hm["pred_c_sn_sign"])
        prim_ctrls = list(hm["primary_controls"])
        sec_ctrls = list(hm["secondary_controls"])
        layer_ctrls = list(LAYER_NEUTRAL_CTRLS[int(hm["layer"])])

        # Pooled 24 primary units x 6 stories across all channels.
        boots: dict[str, Any] = {}
        for ch in ("raw", "spec", "logp", "phrase_mean"):
            mat24 = _unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_UNITS_24, hname, ch, "C_SR")
            boots[CHANNEL_DISPLAY[ch]] = studentized_cluster_bootstrap(
                mat24, alpha=CS3_ALPHA, n_boot=BOOT_DRAWS, seed=_det_seed(f"cs3|{hname}|{ch}")
            )
        m_raw = boots["RawLogit"]["mean"]
        m_spec = boots["SpecLogit"]["mean"]
        m_logp = boots["TargetLogProb"]["mean"]
        pm_pooled = float(_unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_UNITS_24, hname, "peer_med", "C_SR").mean())
        pm_share = abs(pm_pooled) / max(abs(m_spec), 1e-9)

        csn_raw = float(_unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_UNITS_24, hname, "raw", "C_SN").mean())
        csn_spec = float(_unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_UNITS_24, hname, "spec", "C_SN").mean())
        csn_logp = float(_unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_UNITS_24, hname, "logp", "C_SN").mean())

        def _ctrl_abs_map(ctrl_list: list[str], ch: str) -> dict[str, float]:
            return {
                c: abs(float(_unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_UNITS_24, c, ch, "C_SR").mean()))
                for c in ctrl_list
            }

        prim_abs_raw = _ctrl_abs_map(prim_ctrls, "raw")
        prim_abs_spec = _ctrl_abs_map(prim_ctrls, "spec")
        layer_abs_raw = _ctrl_abs_map(layer_ctrls, "raw")
        layer_abs_spec = _ctrl_abs_map(layer_ctrls, "spec")
        sec_abs_raw = _ctrl_abs_map(sec_ctrls, "raw") if sec_ctrls else {}
        sec_abs_spec = _ctrl_abs_map(sec_ctrls, "spec") if sec_ctrls else {}

        max_prim_raw = max(prim_abs_raw.values())
        max_prim_spec = max(prim_abs_spec.values())

        # Exact 5 confirmation criteria (v3.1 §5 CS.3).
        c1_sign = (m_raw * pred_sr > 0.0) and (m_spec * pred_sr > 0.0) and (m_logp * pred_sr > 0.0)
        c2_mag_and_pm = (abs(m_raw) >= CS3_ABS_FLOOR) and (abs(m_spec) >= CS3_ABS_FLOOR) and (pm_share <= CS3_PEER_MED_SHARE_CAP)
        c3_ctrl = (abs(m_raw) >= max_prim_raw + CS3_CTRL_MARGIN) and (abs(m_spec) >= max_prim_spec + CS3_CTRL_MARGIN)
        lo_r, hi_r = boots["RawLogit"]["boot_ci"]
        lo_s, hi_s = boots["SpecLogit"]["boot_ci"]
        if pred_sr > 0:
            c4_ci = (lo_r > 0.0) and (lo_s > 0.0)
        else:
            c4_ci = (hi_r < 0.0) and (hi_s < 0.0)
        csn_gate_failure = (
            cs1_transport["c_sn_transport_failure"]
            or _head_null_failure_for_cells(cs1_head_null, hname, ("slot_12", "concept_12"), ("SELF", "NAME_BARE"))
        )
        c5_csn = bool((csn_gate_failure is None) and (csn_raw * pred_sn > 0.0) and (csn_spec * pred_sn > 0.0))

        if c1_sign and c2_mag_and_pm and c3_ctrl and c4_ci:
            if csn_gate_failure is not None:
                verdict = csn_gate_failure
            elif c5_csn:
                verdict = "confirmed_late_gate"
                n_confirmed += 1
            else:
                verdict = "confirmed_length_confounded"
                n_length_confounded += 1
        else:
            verdict = "not_confirmed"

        # Secondary/descriptive per-class breakdown.
        sec_class = {}
        for ch in ("raw", "spec", "logp"):
            ms = float(_unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_SLOT_UNITS, hname, ch, "C_SR").mean())
            mc = float(_unit_contrast_matrix(blk_by_key, rcvs.PRIMARY_CONCEPT_UNITS, hname, ch, "C_SR").mean())
            sec_class[CHANNEL_DISPLAY[ch]] = {"slot_12": ms, "concept_12": mc, "delta_concept_minus_slot": mc - ms}

        dla_ratios = {
            c: float(head_meta_by_name[c]["old_rms_dla"]) / max(float(hm["old_rms_dla"]), 1e-12)
            for c in prim_ctrls + sec_ctrls
        }

        results[hname] = {
            "verdict": verdict,
            "pred_c_sr_sign": "+" if pred_sr > 0 else "-",
            "pred_c_sn_sign": "+" if pred_sn > 0 else "-",
            "copy_r_saturated": bool(hm["copy_r_saturated"]),
            "lens_sr_z": hm["lens_sr_z"],
            "lens_copy_r": hm["lens_copy_r"],
            "old_rms_dla": hm["old_rms_dla"],
            "primary_controls": prim_ctrls,
            "secondary_controls": sec_ctrls,
            "ctrl_rms_dla_ratios": dla_ratios,
            "pooled_C_SR": boots,
            "pooled_C_SN": {"RawLogit": csn_raw, "SpecLogit": csn_spec, "TargetLogProb": csn_logp},
            "peer_med_share": pm_share,
            "controls_abs_C_SR": {
                "primary_raw": prim_abs_raw,
                "primary_spec": prim_abs_spec,
                "primary_max_raw": max_prim_raw,
                "primary_max_spec": max_prim_spec,
                "layer_neutral_max_raw": max(layer_abs_raw.values()),
                "layer_neutral_max_spec": max(layer_abs_spec.values()),
                "secondary_raw": sec_abs_raw,
                "secondary_spec": sec_abs_spec,
            },
            "criteria": {
                "c1_sign_3ch": c1_sign,
                "c2_mag_and_peer_med": c2_mag_and_pm,
                "c3_beats_primary_ctrl": c3_ctrl,
                "c4_bonferroni_ci_excludes_0": c4_ci,
                "c5_csn_sign_match": c5_csn,
            },
            "secondary_per_class": sec_class,
        }

    return {
        "n_tests": CS3_N_TESTS,
        "alpha_bonferroni": CS3_ALPHA,
        "ci_level": 1.0 - CS3_ALPHA,
        "n_confirmed_late_gate": n_confirmed,
        "n_confirmed_length_confounded": n_length_confounded,
        "heads": results,
    }


# ---------------------------------------------------------------------------
# CS.4 — L7H5 Signed Antisymmetry Diagnostic (ROLE_BARE)
# ---------------------------------------------------------------------------

def evaluate_cs4(
    blk_by_key: dict[tuple[str, str, str], dict[str, Any]],
    cs0_passed: bool,
    cs1_transport: dict[str, Any],
    cs1_head_null: dict[str, Any],
) -> dict[str, Any]:
    classes = {
        "slot_12": (rcvs.PRIMARY_SLOT_UNITS, cs1_transport["role_slot_failure"]),
        "concept_12": (rcvs.PRIMARY_CONCEPT_UNITS, cs1_transport["role_concept_failure"]),
    }
    by_class: dict[str, Any] = {}
    for cls_name, (uids, tr_fail) in classes.items():
        if not cs0_passed:
            by_class[cls_name] = {"overall_verdict": CS0_FAIL_SCI, "status": "gated_by_cs0"}
            continue
        if tr_fail is not None:
            by_class[cls_name] = {"overall_verdict": tr_fail, "status": "gated_by_cs1_transport"}
            continue
        hn_fail = _head_null_failure_for_cells(cs1_head_null, "L7H5", (cls_name,), ("ROLE_BARE",))
        if hn_fail is not None:
            by_class[cls_name] = {"overall_verdict": hn_fail, "status": "gated_by_cs1_head_null"}
            continue

        per_ch: dict[str, Any] = {}
        for ch in ("raw", "spec"):
            u_means = {u: _mean([_s_odd_role(blk_by_key, u, s, "L7H5", ch) for s in STORIES_6]) for u in uids}
            cls_mean = _mean(list(u_means.values()))
            ctrl_means = {
                c: _mean([_s_odd_role(blk_by_key, u, s, c, ch) for u in uids for s in STORIES_6])
                for c in L7_CTRL_HEADS
            }
            ctrl_max = max(ctrl_means.values())
            n_ge_030 = sum(1 for v in u_means.values() if v >= CS4_UNIT_FLOOR)
            verdict_ch = (
                "class_antisymmetry_present"
                if (cls_mean >= CS4_MEAN_FLOOR and (cls_mean - ctrl_max) >= CS4_CTRL_MARGIN and n_ge_030 >= CS4_MIN_UNITS)
                else "not_class_wide"
            )
            per_ch[CHANNEL_DISPLAY[ch]] = {
                "verdict": verdict_ch,
                "mean_S_odd": cls_mean,
                "ctrl_means_S_odd": ctrl_means,
                "ctrl_max_S_odd": ctrl_max,
                "margin_over_ctrl_max": cls_mean - ctrl_max,
                "n_units_ge_030": n_ge_030,
                "per_unit_S_odd": u_means,
            }
        v_r = per_ch["RawLogit"]["verdict"]
        v_s = per_ch["SpecLogit"]["verdict"]
        overall = v_r if v_r == v_s else "inconclusive_readout_disagreement"
        by_class[cls_name] = {
            "overall_verdict": overall,
            "co_primary_verdicts": {"RawLogit": v_r, "SpecLogit": v_s},
            "per_channel": per_ch,
        }

    # Unit diagnostics: L6 units (p92-09, p92-40, p92-43) & exact-token witness (p92-08 vs p92-09).
    unit_diag: dict[str, Any] = {}
    if cs0_passed:
        for u in ("p92-09", "p92-40", "p92-43", "p92-08"):
            unit_diag[u] = {
                CHANNEL_DISPLAY[ch]: _mean([_s_odd_role(blk_by_key, u, s, "L7H5", ch) for s in STORIES_6])
                for ch in ("raw", "spec")
            }
    return {"by_class": by_class, "unit_diagnostics": unit_diag}


# ---------------------------------------------------------------------------
# Full Offline Scorer Entry Point
# ---------------------------------------------------------------------------

def score_full_receipt(
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    pins: dict[str, str],
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Score a complete Full receipt (hard-refuses smoke receipts!)."""
    cs0_report, bound = validate_cs0_and_join(
        receipt=receipt,
        manifest=manifest,
        pins=pins,
        expect_full=True,
        allow_fixture=allow_fixture,
    )
    cs0_ok = bool(cs0_report["passed"])
    cs1_transport = evaluate_cs1_transport(bound["unp_by_key"])
    cs1_head_null = evaluate_cs1_head_null_full(bound["blk_by_key"])

    cs2 = evaluate_cs2(manifest, bound["unp_by_key"], bound["blk_by_key"], cs0_ok, cs1_transport, cs1_head_null)
    cs3 = evaluate_cs3(manifest, bound["blk_by_key"], cs0_ok, cs1_transport, cs1_head_null)
    cs4 = evaluate_cs4(bound["blk_by_key"], cs0_ok, cs1_transport, cs1_head_null)

    meta = receipt["meta"]
    return {
        "meta": {
            "schema_version": rcvs.SCHEMA_VERSION,
            "artifact_type": "concept_vs_slot_scorecard",
            "execution_provenance": meta["execution_provenance"],
            "device_name": meta["device_name"],
            "is_cuda_device": meta["is_cuda_device"],
            "external_pins_verified": True,
            "complete_join_verified": True,
            "pins": dict(pins),
            "operating_norm_name": meta["operating_norm_name"],
            "n_b11_forwards": meta["n_b11_forwards"],
            "n_b1_forwards": meta["n_b1_forwards"],
            "timing": meta["timing"],
        },
        "CS0_technical_gate": cs0_report,
        "CS1_transport_and_head_null": {
            "status": "passed" if (cs0_ok and cs1_transport["all_6_cells_passed"] and cs1_head_null["all_passed"]) else ("gated_by_cs0" if not cs0_ok else "partial_or_failed"),
            "transport": cs1_transport,
            "head_null": cs1_head_null,
        },
        "CS2_concept_vs_slot_gating": cs2,
        "CS3_late_lens_screened_heads": cs3,
        "CS4_L7H5_antisymmetry_diagnostic": cs4,
    }


# ---------------------------------------------------------------------------
# Exhaustive Adversarial & Branch-Logic Self-Test Suite
# ---------------------------------------------------------------------------

def _make_synthetic_full_receipt(manifest: dict[str, Any], pins: dict[str, str]) -> dict[str, Any]:
    """Build a clean passing synthetic Full receipt fixture for deterministic scorer unit tests."""
    sched = manifest["schedule_full"]
    unp_rows = []
    for e in sched["unpatched"]:
        u = e["unit"]
        is_c = manifest["units"][u]["kind_haiku"] == "concept"
        tr_val = 4.5 if is_c else 3.0
        unp_rows.append({
            "unit": u,
            "unit_layer": manifest["units"][u]["layer"],
            "prompt_id": e["prompt_id"],
            "story": e["story"],
            "cell": e["cell"],
            "n_content_tokens": manifest["units"][u]["n_content_tokens"],
            "dose_mag": 0.8,
            "norm_multiplier": float(rcvs.OLD_NORM[str(manifest["units"][u]["layer"])]),
            "injected_norm": 0.8 * float(rcvs.OLD_NORM[str(manifest["units"][u]["layer"])]),
            "read_pos_resid_norm": 10.0,
            "norm_ratio": 0.7,
            "null_seed": 1,
            "max_abs_wdown_orth_cos": 1e-7,
            "max_abs_decomp_err": 0.0,
            "transport": {
                "base_raw": 4.0 + (0.5 if is_c else 0.0),
                "base_peer_med": 1.0,
                "base_spec": 3.0 + (0.5 if is_c else 0.0),
                "base_logp": -5.0,
                "base_phrase_mean": -5.0,
                "base_phrase_logp": -5.0 * manifest["units"][u]["n_content_tokens"],
                "t_pos_raw": tr_val,
                "t_pos_peer_med": 0.0,
                "t_pos_spec": tr_val,
                "t_pos_logp": tr_val,
                "t_pos_phrase_mean": tr_val * 0.8,
                "t_neg_raw": -tr_val,
                "t_neg_peer_med": 0.0,
                "t_neg_spec": -tr_val,
                "t_neg_logp": -tr_val,
                "t_neg_phrase_mean": -tr_val * 0.8,
                "wdown_pos_raw": [0.05, -0.05, 0.02, -0.02],
                "wdown_pos_peer_med": [0.0, 0.0, 0.0, 0.0],
                "wdown_pos_spec": [0.05, -0.05, 0.02, -0.02],
                "wdown_pos_logp": [0.02, -0.02, 0.01, -0.01],
                "wdown_pos_phrase_mean": [0.01, -0.01, 0.01, -0.01],
                "wdown_neg_raw": [-0.05, 0.05, -0.02, 0.02],
                "wdown_neg_peer_med": [0.0, 0.0, 0.0, 0.0],
                "wdown_neg_spec": [-0.05, 0.05, -0.02, 0.02],
                "wdown_neg_logp": [-0.02, 0.02, -0.01, 0.01],
                "wdown_neg_phrase_mean": [-0.01, 0.01, -0.01, 0.01],
            },
        })

    blk_rows = []
    for e in sched["blocked"]:
        u = e["unit"]
        # Realistic non-zero active blocking baseline (d_pos=1e-4 identically across cells -> C_SR=C_SN=S_odd=0.0,
        # passing F1 blocking positive control while representing a genuine null contrast!).
        act_null_ch = {"d0": 0.0, "d_pos": 1e-4, "d_neg": 1e-4, "wd_pos": [0.0, 0.0, 0.0, 0.0], "wd_neg": [0.0, 0.0, 0.0, 0.0]}
        zero_pm = {"d0": 0.0, "d_pos": 0.0, "d_neg": 0.0, "wd_pos": [0.0, 0.0, 0.0, 0.0], "wd_neg": [0.0, 0.0, 0.0, 0.0]}
        hm = rcvs.HEAD_BY_NAME[e["head_name"]]
        blk_rows.append({
            "unit": u,
            "unit_layer": manifest["units"][u]["layer"],
            "prompt_id": e["prompt_id"],
            "story": e["story"],
            "cell": e["cell"],
            "head_name": e["head_name"],
            "head_layer": hm["layer"],
            "head_idx": hm["head"],
            "n_content_tokens": manifest["units"][u]["n_content_tokens"],
            "sibling_max_abs_leakage": 0.0,
            "max_abs_decomp_err": 0.0,
            "raw": copy.deepcopy(act_null_ch),
            "peer_med": copy.deepcopy(zero_pm),
            "spec": copy.deepcopy(act_null_ch),
            "logp": copy.deepcopy(act_null_ch),
            "phrase_mean": copy.deepcopy(act_null_ch),
        })

    shm_rows = [
        {
            "sham_id": s["sham_id"], "unit": s["unit"], "prompt_id": s["prompt_id"],
            "head_name": s["head_name"], "head_layer": rcvs.HEAD_BY_NAME[s["head_name"]]["layer"],
            "head_idx": rcvs.HEAD_BY_NAME[s["head_name"]]["head"],
            "max_abs_sham_raw": 0.0, "max_abs_sham_peer_med": 0.0, "max_abs_sham_spec": 0.0,
            "max_abs_sham_logp": 0.0, "max_abs_sham_phrase_mean": 0.0, "passed": True,
        }
        for s in sched["sham"]
    ]
    act_rows = [
        {
            "actuation_id": a["actuation_id"], "unit": a["unit"], "prompt_id": a["prompt_id"],
            "head_name": a["head_name"], "head_layer": rcvs.HEAD_BY_NAME[a["head_name"]]["layer"],
            "head_idx": rcvs.HEAD_BY_NAME[a["head_name"]]["head"], "seed": a["seed"],
            "target_max_abs_delta": 0.5, "readout_max_abs_delta": 0.25, "sibling_max_abs_leakage": 0.0, "passed": True,
        }
        for a in sched["actuation"]
    ]
    ce_rows = [
        {
            "check_id": c["check_id"], "unit": c["unit"], "prompt_id": c["prompt_id"],
            "n_content_tokens": c["n_content_tokens"], "prefix_len": 33, "full_len": 33 + c["n_content_tokens"] - 1,
            "batch_size": rcvs.BATCH_SIZE,
            "max_abs_full_vocab_read_pos_diff": 0.0, "passed": True,
        }
        for c in sched["causal_equiv"]
    ]
    den = manifest["meta"]["denominators"]["full"]
    ms11, ms1 = 80.0, 20.0
    b11_sec = ms11 * den["n_b11_forwards"] / 1000.0
    b1_sec = ms1 * den["n_b1_forwards"] / 1000.0
    return {
        "meta": {
            "schema_version": rcvs.SCHEMA_VERSION,
            "artifact_type": "concept_vs_slot_full_receipt",
            "run_mode": "full",
            "execution_provenance": "measured_gpu_run",
            "device_name": "NVIDIA L4",
            "is_cuda_device": True,
            "complete": True,
            "model_repo": rcvs.MODEL_REPO,
            "model_commit": rcvs.MODEL_COMMIT,
            "dtype": rcvs.DTYPE_STR,
            "attn_implementation": rcvs.ATTN_IMPL,
            "operating_norm_name": "OLD_NORM",
            "operating_norm_multipliers": dict(rcvs.OLD_NORM),
            "pins": dict(pins),
            "n_unpatched_rows": den["n_unpatched_rows"],
            "n_blocked_rows": den["n_blocked_rows"],
            "n_sham_rows": den["n_sham_rows"],
            "n_actuation_rows": den["n_actuation_rows"],
            "n_causal_equiv_rows": den["n_causal_equiv_rows"],
            "n_b11_forwards": den["n_b11_forwards"],
            "n_b1_forwards": den["n_b1_forwards"],
            "n_total_forwards": den["n_total_forwards"],
            "timing": {
                "model_load_sec": 5.0,
                "b11_elapsed_sec": b11_sec,
                "b1_elapsed_sec": b1_sec,
                "non_fwd_runner_sec": 10.0,
                "total_runner_sec": b11_sec + b1_sec + 10.0,
                "n_b11_done": den["n_b11_forwards"],
                "n_b1_done": den["n_b1_forwards"],
                "ms_per_b11": ms11,
                "ms_per_b1": ms1,
                "projected_full_b11_sec": b11_sec,
                "projected_full_b1_sec": b1_sec,
                "projected_full_fwd_sec": b11_sec + b1_sec,
            },
        },
        "unit_vectors": {},
        "unpatched_rows": unp_rows,
        "blocked_rows": blk_rows,
        "sham_rows": shm_rows,
        "actuation_rows": act_rows,
        "causal_equiv_rows": ce_rows,
    }


def _set_blk_signal(rec: dict[str, Any], head: str, fn) -> None:
    """Apply fn(row, is_concept, cell) -> dict[ch, (d_pos, d_neg)] across blocked_rows for `head`."""
    slot_set = set(rcvs.PRIMARY_SLOT_UNITS)
    for r in rec["blocked_rows"]:
        if r["head_name"] != head:
            continue
        is_c = r["unit"] not in slot_set
        vals = fn(r, is_c, r["cell"])
        if vals is None:
            continue
        for ch, (dp, dn) in vals.items():
            r[ch]["d_pos"] = float(dp)
            r[ch]["d_neg"] = float(dn)
        # Keep Spec == Raw - PeerMed exact.
        r["peer_med"]["d_pos"] = r["raw"]["d_pos"] - r["spec"]["d_pos"]
        r["peer_med"]["d_neg"] = r["raw"]["d_neg"] - r["spec"]["d_neg"]


def run_scorer_self_tests(
    design_path: Path,
    manifest_path: Path,
    runner_path: Path,
    scorer_path: Path,
    targets213_path: Path,
    m7_transport_path: Path,
) -> dict[str, Any]:
    checks: list[str] = []
    d_sha = rcvs.sha256_file(design_path)
    m_sha = rcvs.sha256_file(manifest_path)
    r_sha = rcvs.sha256_file(runner_path)
    s_sha = rcvs.sha256_file(scorer_path)
    t_sha = rcvs.sha256_file(targets213_path)
    m7_sha = rcvs.sha256_file(m7_transport_path)

    manifest, pins, _bytes = rcvs._admit_frozen_package_with_bytes(
        design_path, manifest_path, runner_path, scorer_path, targets213_path, m7_transport_path,
        d_sha, m_sha, r_sha, s_sha, t_sha, m7_sha,
    )
    base_rec = _make_synthetic_full_receipt(manifest, pins)
    # Give every unit distinct non-degenerate base_raw + transport so OLS is full-rank.
    for idx, ur in enumerate(base_rec["unpatched_rows"]):
        u_num = int(ur["unit"].split("-")[1])
        s_num = int(ur["story"][1])
        jitter = 0.05 * ((u_num * 7 + s_num * 3) % 11)
        is_c = manifest["units"][ur["unit"]]["kind_haiku"] == "concept"
        tr_v = (4.5 if is_c else 3.0) + jitter
        ur["transport"]["base_raw"] = 2.0 + 0.13 * u_num + 0.02 * s_num
        ur["transport"]["base_spec"] = ur["transport"]["base_raw"] - 1.0
        for ch in ("raw", "spec", "logp", "phrase_mean"):
            ur["transport"][f"t_pos_{ch}"] = tr_v
            ur["transport"][f"t_neg_{ch}"] = -tr_v

    # 1. Baseline all-zero blocking -> Branch 4 (`both_weak_or_null`) with mandatory report wording!
    sc0 = score_full_receipt(base_rec, manifest, pins, allow_fixture=False)
    assert sc0["CS0_technical_gate"]["passed"] is True
    assert sc0["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == "both_weak_or_null"
    assert sc0["CS2_concept_vs_slot_gating"]["L8H2"]["mandatory_report_wording"] == MANDATORY_NULL_WORDING
    assert sc0["CS3_late_lens_screened_heads"]["n_confirmed_late_gate"] == 0
    assert sc0["CS4_L7H5_antisymmetry_diagnostic"]["by_class"]["slot_12"]["overall_verdict"] == "not_class_wide"
    checks.append("cs2_branch4_both_weak_or_null_with_mandatory_wording_ok")

    # 2. Full scorer hard-refuses a smoke receipt AND refuses meta["complete"] == False (F3)!
    smoke_like = copy.deepcopy(base_rec)
    smoke_like["meta"]["artifact_type"] = "concept_vs_slot_smoke_receipt"
    smoke_like["meta"]["run_mode"] = "smoke"
    refused_smoke = False
    try:
        score_full_receipt(smoke_like, manifest, pins, allow_fixture=False)
    except ValueError:
        refused_smoke = True
    assert refused_smoke

    incomp_rec = copy.deepcopy(base_rec)
    incomp_rec["meta"]["complete"] = False
    refused_incomp = False
    try:
        score_full_receipt(incomp_rec, manifest, pins, allow_fixture=False)
    except ValueError:
        refused_incomp = True
    assert refused_incomp, "F3: Full scorer MUST refuse meta['complete']=False"
    checks.append("full_scorer_refuses_smoke_and_incomplete_receipt_ok")

    # 3. CS.0 strict precedence: causal_equiv breach, F1 inert all-zero block, F1 zero actuation readout,
    #    and F2/M3 sibling leakage 5e-4 (> 1e-6, < 1e-2)!
    r_cs0 = copy.deepcopy(base_rec)
    r_cs0["causal_equiv_rows"][0]["max_abs_full_vocab_read_pos_diff"] = 0.05
    r_cs0["causal_equiv_rows"][0]["passed"] = False
    sc_cs0 = score_full_receipt(r_cs0, manifest, pins, allow_fixture=False)
    assert sc_cs0["CS0_technical_gate"]["passed"] is False
    assert sc_cs0["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == CS0_FAIL_SCI
    assert sc_cs0["CS3_late_lens_screened_heads"]["heads"]["L13H1"]["verdict"] == CS0_FAIL_SCI
    assert sc_cs0["CS4_L7H5_antisymmetry_diagnostic"]["by_class"]["slot_12"]["overall_verdict"] == CS0_FAIL_SCI

    # F1: all-zero inert blocking receipt MUST fail CS.0 (NEVER score as both_weak_or_null!).
    r_inert = copy.deepcopy(base_rec)
    for br in r_inert["blocked_rows"]:
        for ch in rcvs.CHANNELS_5:
            br[ch]["d_pos"] = 0.0
            br[ch]["d_neg"] = 0.0
    sc_inert = score_full_receipt(r_inert, manifest, pins, allow_fixture=False)
    assert sc_inert["CS0_technical_gate"]["passed"] is False
    assert sc_inert["CS0_technical_gate"]["clause_4_synthetic_actuation_26_and_block_pos_ctrl"] is False
    assert sc_inert["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == CS0_FAIL_SCI

    # F1: actuation readout_max_abs_delta == 0.0 MUST fail CS.0!
    r_act_rd = copy.deepcopy(base_rec)
    r_act_rd["actuation_rows"][0]["readout_max_abs_delta"] = 0.0
    sc_act_rd = score_full_receipt(r_act_rd, manifest, pins, allow_fixture=False)
    assert sc_act_rd["CS0_technical_gate"]["passed"] is False

    # F2/M3 mutant kill: sibling leakage = 5e-4 (> 1e-6, < 1e-2) MUST fail CS.0!
    r_sib = copy.deepcopy(base_rec)
    r_sib["blocked_rows"][0]["sibling_max_abs_leakage"] = 5e-4
    sc_sib = score_full_receipt(r_sib, manifest, pins, allow_fixture=False)
    assert sc_sib["CS0_technical_gate"]["passed"] is False
    assert sc_sib["CS0_technical_gate"]["clause_3_sibling_zero_leakage"] is False
    assert sc_sib["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == CS0_FAIL_SCI
    checks.append("cs0_strict_precedence_causal_equiv_f1_inert_act_readout_m3_sibling_ok")

    # 4. CS.1 transport failure & peer_median_only_flip + head-null consequence (M2).
    r_tr = copy.deepcopy(base_rec)
    for ur in r_tr["unpatched_rows"]:
        if manifest["units"][ur["unit"]]["kind_haiku"] == "concept" and ur["cell"] == "ROLE_BARE":
            ur["transport"]["t_pos_raw"] = 0.20
            ur["transport"]["t_pos_spec"] = 0.80
            ur["transport"]["t_pos_peer_med"] = -0.60
    sc_tr = score_full_receipt(r_tr, manifest, pins, allow_fixture=False)
    assert sc_tr["CS1_transport_and_head_null"]["transport"]["by_class_cell"]["concept_12"]["ROLE_BARE"]["peer_median_only_flip"] is True
    assert sc_tr["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == "uninformative (transport_gate_failed:concept_12|ROLE_BARE)"

    r_hn = copy.deepcopy(base_rec)
    for br in r_hn["blocked_rows"]:
        if br["head_name"] == "L8H2" and br["cell"] == "SELF" and manifest["units"][br["unit"]]["kind_haiku"] == "slot":
            br["spec"]["wd_pos"] = [0.40, 0.40, 0.40, 0.40]
            br["spec"]["wd_neg"] = [0.40, 0.40, 0.40, 0.40]
            br["raw"]["wd_pos"] = [0.40, 0.40, 0.40, 0.40]
            br["raw"]["wd_neg"] = [0.40, 0.40, 0.40, 0.40]
    sc_hn = score_full_receipt(r_hn, manifest, pins, allow_fixture=False)
    assert sc_hn["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == "uninformative (head_null_failed:L8H2|slot_12|SELF)"
    assert sc_hn["CS2_concept_vs_slot_gating"]["L8H15"]["overall_verdict"] == "both_weak_or_null"  # isolated to failing head!
    checks.append("cs1_transport_peer_flip_and_head_null_consequence_ok")

    # 5. CS.2 Branch 1 (`slot_stronger_gating`) + every downgrade & guard!
    r_b1 = copy.deepcopy(base_rec)
    def _sig_b1(row, is_c, cell):
        u_j = 0.02 * (int(row["unit"].split("-")[1]) % 5)
        if cell == "SELF":
            v = (0.10 if is_c else 0.90) + u_j
        elif cell == "NAME_BARE":
            v = (0.10 if is_c else 0.20) + u_j
        else:
            v = (0.10 if is_c else 0.05) + u_j
        return {ch: (v, 0.0) for ch in ("raw", "spec", "logp", "phrase_mean")}
    _set_blk_signal(r_b1, "L8H2", _sig_b1)
    sc_b1 = score_full_receipt(r_b1, manifest, pins, allow_fixture=False)
    assert sc_b1["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == "slot_stronger_gating", sc_b1["CS2_concept_vs_slot_gating"]["L8H2"]["per_channel"]["RawLogit"]
    # F2/M4 mutant kill: mandatory_report_wording MUST BE ABSENT on Branch 1 (overall AND per-channel)!
    assert "mandatory_report_wording" not in sc_b1["CS2_concept_vs_slot_gating"]["L8H2"]
    for _ch_nm, _ch_d in sc_b1["CS2_concept_vs_slot_gating"]["L8H2"]["per_channel"].items():
        assert "mandatory_report_wording" not in _ch_d, f"M4: wording leaked onto Branch 1 {_ch_nm}"

    # 5a0. Clean Branch 2 (`concept_stronger_gating`) + F2/M4 wording ABSENT check!
    r_b2 = copy.deepcopy(base_rec)
    def _sig_b2(row, is_c, cell):
        u_j = 0.02 * (int(row["unit"].split("-")[1]) % 5)
        if cell == "SELF":
            v = (1.60 if is_c else 0.15) + u_j
        elif cell == "NAME_BARE":
            v = (0.30 if is_c else 0.10) + u_j
        else:
            v = 0.05 + u_j
        return {ch: (v, 0.0) for ch in ("raw", "spec", "logp", "phrase_mean")}
    _set_blk_signal(r_b2, "L8H2", _sig_b2)
    sc_b2 = score_full_receipt(r_b2, manifest, pins, allow_fixture=False)
    assert sc_b2["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == "concept_stronger_gating", sc_b2["CS2_concept_vs_slot_gating"]["L8H2"]["per_channel"]["RawLogit"]
    assert "mandatory_report_wording" not in sc_b2["CS2_concept_vs_slot_gating"]["L8H2"]
    for _ch_nm, _ch_d in sc_b2["CS2_concept_vs_slot_gating"]["L8H2"]["per_channel"].items():
        assert "mandatory_report_wording" not in _ch_d, f"M4: wording leaked onto Branch 2 {_ch_nm}"

    # 5a1. F4: NAME_BARE transport failure on Branch 1 MUST emit exact registered CS.1 label!
    r_f4 = copy.deepcopy(r_b1)
    for ur in r_f4["unpatched_rows"]:
        if manifest["units"][ur["unit"]]["kind_haiku"] == "concept" and ur["cell"] == "NAME_BARE":
            ur["transport"]["t_pos_raw"] = 0.10
            ur["transport"]["t_pos_spec"] = 0.10
    sc_f4 = score_full_receipt(r_f4, manifest, pins, allow_fixture=False)
    assert sc_f4["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == "uninformative (transport_gate_failed:concept_12|NAME_BARE)"

    # 5a. H1 DiD control gate blocks Branch 1 when L8H3 has the exact same interaction!
    r_h1 = copy.deepcopy(r_b1)
    _set_blk_signal(r_h1, "L8H3", _sig_b1)
    sc_h1 = score_full_receipt(r_h1, manifest, pins, allow_fixture=False)
    assert sc_h1["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == "inconclusive_wide_interval"
    assert "mandatory_report_wording" not in sc_h1["CS2_concept_vs_slot_gating"]["L8H2"]

    # 5b. H2 Branch 4 MUST NOT trigger when Delta = -0.50 (outside [-0.35, +0.35])!
    assert sc_h1["CS2_concept_vs_slot_gating"]["L8H2"]["per_channel"]["RawLogit"]["verdict"] != "both_weak_or_null"

    # 5c. C_SN length-confound downgrade (`length_confound_unconfirmed`).
    r_csn = copy.deepcopy(r_b1)
    def _sig_csn_fail(row, is_c, cell):
        d = _sig_b1(row, is_c, cell)
        if cell == "NAME_BARE":
            # Make SELF == NAME_BARE so C_SN == 0!
            d = _sig_b1(row, is_c, "SELF")
        return d
    _set_blk_signal(r_csn, "L8H2", _sig_csn_fail)
    sc_csn = score_full_receipt(r_csn, manifest, pins, allow_fixture=False)
    assert sc_csn["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == "length_confound_unconfirmed"

    # 5d. H4 Kind-confound downgrade (`inconclusive_kind_confounded`) when 5 same-kind pairs flip sign while 12-pair Delta < -0.35!
    r_kind = copy.deepcopy(r_b1)
    same_kind_units = {p["slot_unit"] for p in rcvs.PRIMARY_PAIRS if p["same_persona_kind"]} | {p["concept_unit"] for p in rcvs.PRIMARY_PAIRS if p["same_persona_kind"]}
    # P02, P03 mildly positive (+0.20); P05, P08, P11 mildly negative (-0.05) -> mean(5 same-kind) = +0.05 > 0,
    # while 7 cross-kind pairs are strongly negative (-2.50) -> 10/12 negative & boot_ci hi << -0.35!
    big_pos_sk = {"p92-04", "p92-00", "p92-02", "p92-11"}
    def _sig_kind_flip2(row, is_c, cell):
        u_j = 0.01 * (int(row["unit"].split("-")[1]) % 3)
        if row["unit"] in big_pos_sk:
            v = (0.25 if is_c else 0.05) if cell == "SELF" else 0.05
        elif row["unit"] in same_kind_units:
            v = (0.05 if is_c else 0.10) if cell == "SELF" else 0.05
        else:
            v = (0.05 if is_c else 2.55) if cell == "SELF" else (0.10 if cell == "NAME_BARE" else 0.05)
        return {ch: (v + u_j, 0.0) for ch in ("raw", "spec", "logp", "phrase_mean")}
    _set_blk_signal(r_kind, "L8H2", _sig_kind_flip2)
    sc_kind = score_full_receipt(r_kind, manifest, pins, allow_fixture=False)
    assert sc_kind["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == "inconclusive_kind_confounded", sc_kind["CS2_concept_vs_slot_gating"]["L8H2"]["per_channel"]["RawLogit"]

    # 5e. M3 Transport-scale confound downgrade (`transport_scale_confounded`):
    # Concept has larger raw C_SR (+1.00 vs +0.35 -> Branch 2 raw Delta = +0.65 > 0.35),
    # BUT Concept transport is 4x Slot transport (12.0 vs 3.0), so F_SR_ratio(concept)=0.083 < F_SR_ratio(slot)=0.117!
    r_tr_conf = copy.deepcopy(base_rec)
    for ur in r_tr_conf["unpatched_rows"]:
        u_num = int(ur["unit"].split("-")[1])
        s_num = int(ur["story"][1])
        jitter = 0.05 * ((u_num * 5 + s_num * 3) % 7)
        is_c = manifest["units"][ur["unit"]]["kind_haiku"] == "concept"
        tr_v = (12.0 if is_c else 3.0) + jitter
        for ch in ("raw", "spec", "logp", "phrase_mean"):
            ur["transport"][f"t_pos_{ch}"] = tr_v
    def _sig_b2_tr_conf(row, is_c, cell):
        u_j = 0.01 * (int(row["unit"].split("-")[1]) % 3)
        if cell == "SELF":
            v = (1.05 if is_c else 0.40) + u_j
        elif cell == "NAME_BARE":
            v = (0.20 if is_c else 0.15) + u_j
        else:
            v = 0.05 + u_j
        return {ch: (v, 0.0) for ch in ("raw", "spec", "logp", "phrase_mean")}
    _set_blk_signal(r_tr_conf, "L8H2", _sig_b2_tr_conf)
    sc_trc = score_full_receipt(r_tr_conf, manifest, pins, allow_fixture=False)
    assert sc_trc["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == "transport_scale_confounded", sc_trc["CS2_concept_vs_slot_gating"]["L8H2"]["per_channel"]["RawLogit"]

    # 5f. 3-readout disagreement (`inconclusive_readout_disagreement`).
    r_dis = copy.deepcopy(r_b1)
    for br in r_dis["blocked_rows"]:
        if br["head_name"] == "L8H2":
            br["logp"]["d_pos"] = 0.0
    sc_dis = score_full_receipt(r_dis, manifest, pins, allow_fixture=False)
    assert sc_dis["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == "inconclusive_readout_disagreement"

    # 5g. Branch 3 (`practically_equivalent_subject_gating`) + mandatory wording.
    r_b3 = copy.deepcopy(base_rec)
    def _sig_b3(row, is_c, cell):
        u_j = 0.01 * (int(row["unit"].split("-")[1]) % 3)
        v = (0.60 if cell == "SELF" else (0.10 if cell == "NAME_BARE" else 0.05)) + u_j
        return {ch: (v, 0.0) for ch in ("raw", "spec", "logp", "phrase_mean")}
    _set_blk_signal(r_b3, "L8H2", _sig_b3)
    sc_b3 = score_full_receipt(r_b3, manifest, pins, allow_fixture=False)
    assert sc_b3["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"] == "practically_equivalent_subject_gating"
    assert sc_b3["CS2_concept_vs_slot_gating"]["L8H2"]["mandatory_report_wording"] == MANDATORY_NULL_WORDING
    checks.append("cs2_all_5_branches_h1_h2_h4_m3_csn_3ch_disagreement_ok")

    # 6. CS.3: confirmed_late_gate (+ and -), confirmed_length_confounded, primary-control margin block,
    #    AND F2/M5 Bonferroni CI mutant kill (passes c1,c2,c3,c5 but fails c4 99.29% CI)!
    r_cs3 = copy.deepcopy(base_rec)
    def _sig_pos_late(row, _is_c, cell):
        u_j = 0.02 * (int(row["unit"].split("-")[1]) % 5)
        v = (0.70 if cell == "SELF" else (0.20 if cell == "NAME_BARE" else 0.05)) + u_j
        return {ch: (v, 0.0) for ch in ("raw", "spec", "logp", "phrase_mean")}
    def _sig_neg_late_lenconf(row, _is_c, cell):
        u_j = 0.02 * (int(row["unit"].split("-")[1]) % 5)
        # C_SR = -0.65 (matches pred '-'), BUT C_SN = +0.10 (wrong sign -> confirmed_length_confounded!)
        v = (0.20 if cell == "SELF" else (0.10 if cell == "NAME_BARE" else 0.85)) + u_j
        return {ch: (v, 0.0) for ch in ("raw", "spec", "logp", "phrase_mean")}
    def _sig_wide_ci_late(row, is_c, cell):
        # F2/M5: 12 slot units at C_SR=+1.80, 12 concept units at C_SR=-1.00 -> pooled mean = +0.40 (>= 0.25, beats ctrl),
        # C_SN = +0.40 (> 0), BUT SE ~ 0.29 so 99.29% CI crosses 0 -> MUST be not_confirmed!
        if cell == "SELF":
            v = -0.95 if is_c else 1.85
        elif cell == "NAME_BARE":
            v = -0.95 if is_c else 0.05
        else:
            v = 0.05
        return {ch: (v, 0.0) for ch in ("raw", "spec", "logp", "phrase_mean")}
    _set_blk_signal(r_cs3, "L13H1", _sig_pos_late)
    _set_blk_signal(r_cs3, "L14H3", _sig_neg_late_lenconf)
    _set_blk_signal(r_cs3, "L14H10", _sig_wide_ci_late)
    # Give L11H9 strong signal AND give its named primary control L11H1 equal signal -> MUST NOT confirm!
    _set_blk_signal(r_cs3, "L11H9", _sig_pos_late)
    _set_blk_signal(r_cs3, "L11H1", _sig_pos_late)
    sc_cs3 = score_full_receipt(r_cs3, manifest, pins, allow_fixture=False)
    assert sc_cs3["CS3_late_lens_screened_heads"]["heads"]["L13H1"]["verdict"] == "confirmed_late_gate"
    assert sc_cs3["CS3_late_lens_screened_heads"]["heads"]["L14H3"]["verdict"] == "confirmed_length_confounded"
    assert sc_cs3["CS3_late_lens_screened_heads"]["heads"]["L11H9"]["verdict"] == "not_confirmed"
    l14h10 = sc_cs3["CS3_late_lens_screened_heads"]["heads"]["L14H10"]
    assert l14h10["criteria"]["c1_sign_3ch"] is True
    assert l14h10["criteria"]["c2_mag_and_peer_med"] is True
    assert l14h10["criteria"]["c3_beats_primary_ctrl"] is True
    assert l14h10["criteria"]["c5_csn_sign_match"] is True
    assert l14h10["criteria"]["c4_bonferroni_ci_excludes_0"] is False
    assert l14h10["verdict"] == "not_confirmed", "F2/M5: wide CI must NOT confirm!"
    assert sc_cs3["CS3_late_lens_screened_heads"]["heads"]["L11H2"]["copy_r_saturated"] is True
    checks.append("cs3_bonferroni_m5_ci_kill_confirmed_length_confounded_and_ctrl_margin_ok")

    # 7. CS.4: exact Part B S_odd sign, 2-channel agreement, AND F2/M2 min_units=4 mutant kill!
    r_cs4 = copy.deepcopy(base_rec)
    def _sig_l7h5(row, is_c, cell):
        if cell != "ROLE_BARE":
            return None
        if not is_c:
            # Slot: d_pos = -0.45, d_neg = +0.45 -> S_odd = +0.45 on both Raw & Spec -> class_antisymmetry_present!
            return {"raw": (-0.45, 0.45), "spec": (-0.45, 0.45), "logp": (-0.45, 0.45), "phrase_mean": (-0.45, 0.45)}
        # Concept: Raw S_odd = +0.45, Spec S_odd = +0.05 -> inconclusive_readout_disagreement!
        return {"raw": (-0.45, 0.45), "spec": (-0.05, 0.05), "logp": (-0.45, 0.45), "phrase_mean": (-0.45, 0.45)}
    _set_blk_signal(r_cs4, "L7H5", _sig_l7h5)
    sc_cs4 = score_full_receipt(r_cs4, manifest, pins, allow_fixture=False)
    assert sc_cs4["CS4_L7H5_antisymmetry_diagnostic"]["by_class"]["slot_12"]["overall_verdict"] == "class_antisymmetry_present"
    assert sc_cs4["CS4_L7H5_antisymmetry_diagnostic"]["by_class"]["concept_12"]["overall_verdict"] == "inconclusive_readout_disagreement"

    # F2/M2 mutant kill: only 3/12 slot units >= +0.30 (at +1.00) -> mean=+0.25 >= 0.20, margin=+0.25 >= 0.15,
    # BUT n_units_ge_030 = 3 < 4 -> MUST be not_class_wide!
    r_cs4_m2 = copy.deepcopy(base_rec)
    three_slot = set(rcvs.PRIMARY_SLOT_UNITS[:3])
    def _sig_l7h5_3units(row, _is_c, cell):
        if cell != "ROLE_BARE":
            return None
        if row["unit"] in three_slot:
            return {"raw": (-1.00, 1.00), "spec": (-1.00, 1.00), "logp": (-1.00, 1.00), "phrase_mean": (-1.00, 1.00)}
        return {"raw": (1e-4, 1e-4), "spec": (1e-4, 1e-4), "logp": (1e-4, 1e-4), "phrase_mean": (1e-4, 1e-4)}
    _set_blk_signal(r_cs4_m2, "L7H5", _sig_l7h5_3units)
    sc_cs4_m2 = score_full_receipt(r_cs4_m2, manifest, pins, allow_fixture=False)
    s_m2 = sc_cs4_m2["CS4_L7H5_antisymmetry_diagnostic"]["by_class"]["slot_12"]
    assert s_m2["per_channel"]["RawLogit"]["mean_S_odd"] >= 0.20
    assert s_m2["per_channel"]["RawLogit"]["n_units_ge_030"] == 3
    assert s_m2["overall_verdict"] == "not_class_wide", "F2/M2: 3/12 units must be not_class_wide!"
    checks.append("cs4_exact_part_b_s_odd_sign_2ch_agreement_and_m2_min_units_kill_ok")

    # 8. Verify verify_smoke_receipt_only emits ZERO head-gating or CS.2/CS.3/CS.4 scientific verdicts.
    smoke_rec = copy.deepcopy(base_rec)
    smoke_den = manifest["meta"]["denominators"]["smoke"]
    smoke_blk_keys = {(e["unit"], e["prompt_id"], e["head_name"]) for e in manifest["schedule_smoke"]["blocked"]}
    smoke_rec["blocked_rows"] = [r for r in smoke_rec["blocked_rows"] if (r["unit"], r["prompt_id"], r["head_name"]) in smoke_blk_keys]
    smoke_rec["meta"]["artifact_type"] = "concept_vs_slot_smoke_receipt"
    smoke_rec["meta"]["run_mode"] = "smoke"
    smoke_rec["meta"]["complete"] = False
    smoke_rec["meta"]["n_blocked_rows"] = smoke_den["n_blocked_rows"]
    smoke_rec["meta"]["n_b11_forwards"] = smoke_den["n_b11_forwards"]
    smoke_rec["meta"]["n_total_forwards"] = smoke_den["n_total_forwards"]
    b11_s = 80.0 * smoke_den["n_b11_forwards"] / 1000.0
    b1_s = 20.0 * smoke_den["n_b1_forwards"] / 1000.0
    smoke_rec["meta"]["timing"].update({
        "b11_elapsed_sec": b11_s,
        "b1_elapsed_sec": b1_s,
        "total_runner_sec": b11_s + b1_s + 5.0,
        "n_b11_done": smoke_den["n_b11_forwards"],
        "n_b1_done": smoke_den["n_b1_forwards"],
    })
    sv = verify_smoke_receipt_only(smoke_rec, manifest, pins, allow_fixture=False)
    assert sv["smoke_gate_passed"] is True
    sv_blob = json.dumps(sv)
    for forbidden in ("CS2", "CS3", "CS4", "C_SR", "C_SN", "S_odd", "slot_stronger"):
        assert forbidden not in sv_blob, f"Smoke verifier leaked forbidden scientific token {forbidden!r}"
    checks.append("smoke_verifier_zero_scientific_verdict_leakage_ok")

    return {
        "scorer_self_tests_passed": True,
        "n_checks": len(checks),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Concept vs. Slot Head-Gating & Late-Layer Heads Offline Scorer")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--verify-smoke-only", action="store_true")
    ap.add_argument("--receipt", type=Path, default=None)
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--runner", type=Path, default=None)
    ap.add_argument("--scorer", type=Path, default=None)
    ap.add_argument("--targets213", type=Path, default=None)
    ap.add_argument("--m7-transport", type=Path, default=None)
    ap.add_argument("--design-sha256", type=str, default="")
    ap.add_argument("--manifest-sha256", type=str, default="")
    ap.add_argument("--runner-sha256", type=str, default="")
    ap.add_argument("--scorer-sha256", type=str, default="")
    ap.add_argument("--targets213-sha256", type=str, default=rcvs.TARGETS213_SHA256)
    ap.add_argument("--m7-transport-sha256", type=str, default=rcvs.M7_TRANSPORT_SHA256)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    repo_root = args.repo_root.resolve()
    base_dir = repo_root / "explore_olmo2_attn_mlp"
    design_p = args.design or (base_dir / "CONCEPT-VS-SLOT-DESIGN.md")
    manifest_p = args.manifest or (base_dir / "concept_vs_slot_manifest.json")
    runner_p = args.runner or (base_dir / "run_concept_vs_slot.py")
    scorer_p = args.scorer or Path(__file__).resolve()
    t213_p = args.targets213 or (repo_root / rcvs.TARGETS213_REL_PATH)
    m7_p = args.m7_transport or (repo_root / rcvs.M7_TRANSPORT_REL_PATH)

    if args.self_test_only:
        res = run_scorer_self_tests(
            design_path=design_p,
            manifest_path=manifest_p,
            runner_path=runner_p,
            scorer_path=scorer_p,
            targets213_path=t213_p,
            m7_transport_path=m7_p,
        )
        print(json.dumps(res, indent=2))
        return 0

    if args.receipt is None or not args.receipt.is_file():
        raise FileNotFoundError(f"Missing --receipt file: {args.receipt}")

    manifest, pins, _bytes = rcvs._admit_frozen_package_with_bytes(
        design_path=design_p,
        manifest_path=manifest_p,
        runner_path=runner_p,
        scorer_path=scorer_p,
        targets213_path=t213_p,
        m7_transport_path=m7_p,
        expected_design_sha=args.design_sha256,
        expected_manifest_sha=args.manifest_sha256,
        expected_runner_sha=args.runner_sha256,
        expected_scorer_sha=args.scorer_sha256,
        expected_targets213_sha=args.targets213_sha256,
        expected_m7_transport_sha=args.m7_transport_sha256,
    )
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))

    if args.verify_smoke_only:
        verdict = verify_smoke_receipt_only(receipt=receipt, manifest=manifest, pins=pins, allow_fixture=False)
        if args.out is not None:
            rcvs.atomic_write_text(args.out, json.dumps(verdict, indent=2))
        print(json.dumps(verdict, indent=2))
        return 0 if verdict["smoke_gate_passed"] else 2

    scorecard = score_full_receipt(receipt=receipt, manifest=manifest, pins=pins, allow_fixture=False)
    if args.out is not None:
        rcvs.atomic_write_text(args.out, json.dumps(scorecard, indent=2))
    print(json.dumps({
        "artifact_type": scorecard["meta"]["artifact_type"],
        "out": str(args.out) if args.out else None,
        "CS0_passed": scorecard["CS0_technical_gate"]["passed"],
        "CS1_status": scorecard["CS1_transport_and_head_null"]["status"],
        "CS2_L8H2_verdict": scorecard["CS2_concept_vs_slot_gating"]["L8H2"]["overall_verdict"],
        "CS2_L8H15_verdict": scorecard["CS2_concept_vs_slot_gating"]["L8H15"]["overall_verdict"],
        "CS3_n_confirmed": scorecard["CS3_late_lens_screened_heads"]["n_confirmed_late_gate"],
        "CS4_slot_verdict": scorecard["CS4_L7H5_antisymmetry_diagnostic"]["by_class"]["slot_12"]["overall_verdict"],
        "CS4_concept_verdict": scorecard["CS4_L7H5_antisymmetry_diagnostic"]["by_class"]["concept_12"]["overall_verdict"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
