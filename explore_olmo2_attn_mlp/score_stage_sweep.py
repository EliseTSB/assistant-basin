#!/usr/bin/env python3
"""Pinned Offline Scorer for OLMo 2 1B Stage-Sweep Design v2.1 (`STAGE-SWEEP-DESIGN.md`).

Enforces:
  - All 4 external 64-hex SHA-256 CLI pins (`--expected-manifest-sha256`, `--expected-design-sha256`,
    `--expected-runner-sha256`, `--expected-scorer-sha256`) checked against BOTH actual disk files
    AND every stage receipt's `meta` + `meta.pins` (refuses unchecked CLI invocations).
  - Exact 4-stage manifest-to-result joins (`midtrain_end`, `sft`, `dpo`, `instruct`), hard-refusing
    smoke/partial/missing/duplicate/extra/non-causal/non-finite/peer-mismatch/wrong-commit inputs.
  - Independent re-computation of Arm B `select_calibrated_dose(...)` from raw `CAL` rows, verifying
    exact agreement with each receipt's `cal_summary` (unbracketed monotone curves strictly `None`).
  - Total deterministic decision maps for `SS.1`–`SS.7` (Alden #58 & #62 exact contracts), keeping
    Arm A (common dose `+0.8`) and Arm B (pre-rebound calibrated dose) strictly separate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any

import run_stage_sweep as rss


HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
SCHEMA_VERSION = rss.SCHEMA_VERSION
STAGE_ORDER = rss.STAGE_ORDER
CORE_UNITS = rss.CORE_UNITS
L3_EARLY_UNITS = rss.L3_EARLY_UNITS
L5L6_LATE_UNITS = rss.L5L6_LATE_UNITS
RAND_L8_HEADS = rss.RAND_L8_HEADS
EVAL_STORIES = tuple(es["story"] for es in rss.EVAL_STORY_SPECS)
SENS_STORIES = ("E1", "E2")


def _require_hex64(label: str, val: str) -> str:
    if not isinstance(val, str) or not HEX64_RE.match(val):
        raise ValueError(f"Invalid 64-hex SHA-256 pin for {label}: {val!r}")
    return val


def _finite_summary(vals: list[float]) -> dict[str, Any]:
    if not vals:
        raise ValueError("Cannot summarize empty values")
    for v in vals:
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"Non-finite value encountered: {v!r}")
    s = sorted(float(v) for v in vals)
    n = len(s)
    q25 = s[(n - 1) // 4]
    q75 = s[(3 * (n - 1)) // 4]
    return {
        "n": n,
        "mean": float(statistics.mean(s)),
        "median": float(statistics.median(s)),
        "iqr": float(q75 - q25),
        "min": float(s[0]),
        "max": float(s[-1]),
    }


def _check_finite_recursive(obj: Any, ctx: str) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite number at {ctx}: {obj!r}")
        return
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            _check_finite_recursive(item, f"{ctx}[{i}]")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite_recursive(v, f"{ctx}.{k}")


# ---------------------------------------------------------------------------
# Pure deterministic decision-map helpers (directly unit-tested against #58 & #62)
# ---------------------------------------------------------------------------

def decide_ss2a_bare(g_bare: float, r_bare: float) -> dict[str, Any]:
    if g_bare <= -0.25 and g_bare <= r_bare - 0.15:
        status = "holds"
    elif g_bare > -0.15 or g_bare > r_bare:
        status = "fails"
    else:
        status = "inconclusive"
    return {"status": status, "G_bare": g_bare, "R_bare": r_bare, "margin_vs_rand": g_bare - r_bare}


def decide_ss2b_alias(g_alias: float, r_alias: float, g_bare: float) -> dict[str, Any]:
    """Exact SS.2(b) rule + separate declared `G_alias - G_bare` attenuation rule (Alden #62 Fix 1)."""
    fail_reason = None
    if g_alias <= -0.25 and g_alias <= r_alias - 0.15:
        status = "holds"
    elif g_alias > -0.15 or g_alias > r_alias:
        status = "fails"
        weak = g_alias > -0.15
        rand_bad = g_alias > r_alias
        if weak and rand_bad:
            fail_reason = "alias_effect_weak_and_random_head_failed"
        elif weak:
            fail_reason = "alias_effect_weak"
        else:
            fail_reason = "random_head_comparison_failed"
    else:
        status = "inconclusive"

    delta_alias_minus_bare = g_alias - g_bare
    if g_bare > -0.25:
        attenuation_status = "not_applicable_bare_below_threshold"
    elif delta_alias_minus_bare >= 0.15:
        attenuation_status = "attenuated_under_same_referent_manipulation"
    else:
        attenuation_status = "preserved_or_stronger_under_same_referent_manipulation"

    return {
        "status": status,
        "fail_reason": fail_reason,
        "G_alias": g_alias,
        "R_alias": r_alias,
        "margin_vs_rand": g_alias - r_alias,
        "G_bare": g_bare,
        "delta_alias_minus_bare": delta_alias_minus_bare,
        "attenuation_status": attenuation_status,
    }


def classify_ss2c_mechanism(
    delta_raw: float,
    delta_peer_med: float,
    frac_peer_tokens_pos: float,
    frac_pairs_peer_med_pos: float,
) -> dict[str, Any]:
    if abs(delta_peer_med) >= 2.0 * abs(delta_raw) and delta_peer_med >= 0.20 and frac_peer_tokens_pos >= 0.75:
        cls = "peer_dominated"
    elif abs(delta_raw) >= 2.0 * abs(delta_peer_med) and delta_raw <= -0.20:
        cls = "target_dominated"
    elif abs(delta_raw) < 0.20 and abs(delta_peer_med) < 0.20:
        cls = "weak_or_null"
    else:
        cls = "mixed_or_narrow_peer"
    return {
        "classification": cls,
        "delta_raw": delta_raw,
        "delta_peer_med": delta_peer_med,
        "spec_reconstructed": delta_raw - delta_peer_med,
        "frac_peer_tokens_pos": frac_peer_tokens_pos,
        "frac_pairs_peer_med_pos": frac_pairs_peer_med_pos,
    }


def decide_ss3_onset(
    mid_gate: str,
    sft_gate: str,
    c_mid: float | None,
    c_sft: float | None,
    r_sft: float | None,
) -> dict[str, Any]:
    """Total deterministic decision map for SS.3 (`midtrain_end -> sft`, Alden #58 Fix 3)."""
    if mid_gate.startswith("invalidated") or sft_gate.startswith("invalidated"):
        return {"status": "invalidated", "mid_gate": mid_gate, "sft_gate": sft_gate}
    if sft_gate.startswith("unreadable") or sft_gate.startswith("uninformative"):
        return {"status": sft_gate, "mid_gate": mid_gate, "sft_gate": sft_gate}
    if mid_gate.startswith("unreadable"):
        return {"status": "unreadable", "mid_gate": mid_gate, "sft_gate": sft_gate}

    assert c_sft is not None and r_sft is not None
    margin_sft = c_sft - r_sft

    if mid_gate.startswith("uninformative"):
        if c_sft >= 0.25 and margin_sft >= 0.15:
            fb = "holds (first_measurable_at_sft)"
        elif c_sft < 0.25 or margin_sft < 0.0:
            fb = "fails (not_measurable_at_sft)"
        else:
            fb = "inconclusive (sft_random_margin_intermediate)"
        return {
            "status": "uninformative (midtrain_gate_or_cal_failed)",
            "first_measurable_at_sft_status": fb,
            "mid_gate": mid_gate,
            "sft_gate": sft_gate,
            "C_SR_sft": c_sft,
            "R_SR_sft": r_sft,
            "margin_sft_vs_rand": margin_sft,
        }

    assert c_mid is not None
    i_onset = c_sft - c_mid
    if c_sft >= 0.25 and margin_sft >= 0.15 and i_onset >= 0.25 and abs(c_mid) < 0.20:
        status = "holds"
    elif i_onset <= 0.0 or c_sft < 0.25 or margin_sft < 0.0:
        status = "fails"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "mid_gate": mid_gate,
        "sft_gate": sft_gate,
        "C_SR_midtrain": c_mid,
        "C_SR_sft": c_sft,
        "R_SR_sft": r_sft,
        "margin_sft_vs_rand": margin_sft,
        "I_onset": i_onset,
    }


def decide_ss4_later_change(
    gates: dict[str, str],
    c_sr: dict[str, float | None],
) -> dict[str, Any]:
    """Total deterministic decision map for SS.4 (`sft -> dpo -> instruct`, Alden #54 & #58 Fix 3)."""
    st_list = ("sft", "dpo", "instruct")
    g_vals = [gates[s] for s in st_list]
    if any(g.startswith("invalidated") for g in g_vals):
        return {"status": "invalidated", "gates": {s: gates[s] for s in st_list}}
    if any(g.startswith("unreadable") for g in g_vals):
        return {"status": "unreadable", "gates": {s: gates[s] for s in st_list}}
    if any(g.startswith("uninformative") for g in g_vals):
        return {"status": "uninformative", "gates": {s: gates[s] for s in st_list}}

    cs, cd, ci = float(c_sr["sft"]), float(c_sr["dpo"]), float(c_sr["instruct"])
    i_dpo = cd - cs
    i_rlvr = ci - cd
    amp = (i_dpo >= 0.25) or (i_rlvr >= 0.25)
    att_or_loss = (i_dpo <= -0.25) or (i_rlvr <= -0.25) or (cs < 0.25) or (cd < 0.25) or (ci < 0.25)
    if (not amp) and (not att_or_loss):
        status = "holds (post_sft_stability)"
    elif amp and not ((i_dpo <= -0.25) or (i_rlvr <= -0.25)):
        status = "fails_amplification"
    elif att_or_loss and not amp:
        status = "fails_attenuation_or_loss"
    else:
        status = "fails_non_monotone_mixed"
    return {
        "status": status,
        "C_SR_sft": cs,
        "C_SR_dpo": cd,
        "C_SR_instruct": ci,
        "I_dpo": i_dpo,
        "I_rlvr": i_rlvr,
    }


def decide_ss5_finite_bank(
    delta_sn_by_us: dict[tuple[str, str], float],
    c_sr_val: float,
) -> dict[str, Any]:
    """Two-factor finite-bank (`5` units x `4` stories) consistency rule (Alden #58 Fix 3)."""
    all_vals = [delta_sn_by_us[(u, e)] for u in CORE_UNITS for e in EVAL_STORIES]
    grand_m = float(statistics.mean(all_vals))
    unit_means = {u: float(statistics.mean(delta_sn_by_us[(u, e)] for e in EVAL_STORIES)) for u in CORE_UNITS}
    story_means = {e: float(statistics.mean(delta_sn_by_us[(u, e)] for u in CORE_UNITS)) for e in EVAL_STORIES}
    max_story_abs = max(abs(v) for v in story_means.values())
    n_units_within = sum(1 for v in unit_means.values() if abs(v) <= 0.25)

    if abs(grand_m) <= 0.15 and max_story_abs <= 0.25 and n_units_within >= 4 and c_sr_val >= 0.25:
        status = "holds (finite_bank_close_and_separated_from_role)"
    elif abs(grand_m) > 0.25 or c_sr_val < 0.25:
        status = "fails"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "population_equivalence": "not_claimed_finite_5x4_bank",
        "grand_mean": grand_m,
        "C_SR": c_sr_val,
        "unit_means": unit_means,
        "story_means": story_means,
        "max_story_abs": max_story_abs,
        "n_units_within_025": n_units_within,
    }


def decide_ss6_l7h5_instruct(
    dneg_pooled: float,
    dneg_l5l6: float,
    dneg_l3: float,
    ratio_l5l6: float,
    max_null_draw_mean: float,
) -> dict[str, Any]:
    if (
        dneg_pooled >= 0.10
        and dneg_l5l6 >= 0.25
        and (dneg_l5l6 - dneg_l3) >= 0.25
        and ratio_l5l6 > 1.0
        and dneg_pooled > max_null_draw_mean
    ):
        status = "holds"
    elif dneg_pooled <= 0.0 or dneg_l5l6 <= 0.0 or dneg_l5l6 <= dneg_l3:
        status = "fails"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "dneg_pooled": dneg_pooled,
        "dneg_l5l6": dneg_l5l6,
        "dneg_l3": dneg_l3,
        "l5l6_minus_l3": dneg_l5l6 - dneg_l3,
        "ratio_l5l6": ratio_l5l6,
        "max_null_draw_mean": max_null_draw_mean,
    }


def decide_ss7_plain_vs_chat(
    chat_gate: str,
    plain_gate: str,
    c_chat_e1e2: float | None,
    c_plain: float | None,
) -> dict[str, Any]:
    """Total 16-state gate + 4-way valid partition for SS.7 on matched E1/E2 probes (Alden #62 Fix 2)."""
    valid_states = ("invalidated", "uninformative", "unreadable", "valid")
    if chat_gate not in valid_states or plain_gate not in valid_states:
        raise ValueError(f"Invalid SS.7 gate state(s): chat={chat_gate!r}, plain={plain_gate!r}")

    base = {
        "chat_gate_e1e2": chat_gate,
        "plain_gate_e1e2": plain_gate,
        "C_SR_chat_E1E2": c_chat_e1e2,
        "C_SR_plain": c_plain,
    }
    if chat_gate == "invalidated" or plain_gate == "invalidated":
        return {**base, "status": "invalidated"}
    if chat_gate == "uninformative" and plain_gate == "valid":
        return {**base, "status": "transport_diff_plain_only"}
    if chat_gate == "valid" and plain_gate == "uninformative":
        return {**base, "status": "transport_diff_chat_only"}
    if chat_gate == "unreadable" or plain_gate == "unreadable":
        return {**base, "status": "unreadable"}
    if chat_gate == "uninformative" and plain_gate == "uninformative":
        return {**base, "status": "uninformative (both_transport_failed)"}

    assert c_chat_e1e2 is not None and c_plain is not None
    if c_chat_e1e2 >= 0.25 and c_plain >= 0.25:
        status = "chat_independent (preserved_across_wrappers)"
    elif c_chat_e1e2 >= 0.25 and c_plain < 0.15:
        status = "chat_dependent"
    elif c_chat_e1e2 < 0.25 and c_plain >= 0.25:
        status = "plain_positive_only"
    else:
        status = "neither_or_intermediate"
    return {**base, "status": status}


# ---------------------------------------------------------------------------
# Full 4-Stage Receipt Join, Verification & Scoring
# ---------------------------------------------------------------------------

def _verify_exact_row_keys(
    rows: list[dict[str, Any]],
    expected_keys: set[tuple],
    key_fn,
    label: str,
) -> dict[tuple, dict[str, Any]]:
    seen: dict[tuple, dict[str, Any]] = {}
    for idx, r in enumerate(rows):
        _check_finite_recursive(r, f"{label}[{idx}]")
        k = key_fn(r)
        if k in seen:
            raise ValueError(f"Duplicate row in {label}: {k}")
        if k not in expected_keys:
            raise ValueError(f"Unexpected/extra row in {label}: {k}")
        seen[k] = r
    missing = expected_keys - set(seen.keys())
    if missing:
        raise ValueError(f"Missing {len(missing)} expected row(s) in {label}, e.g. {next(iter(missing))}")
    return seen


def _evaluate_matched_e1e2_gate(
    unp_rows: list[dict[str, Any]],
    head_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    max_sh = max(max(r["spec"]["max_abs_sham"], r["raw"]["max_abs_sham"]) for r in head_rows)
    tr_self = rss.evaluate_cell_transport([r for r in unp_rows if r["cell"] == "SELF"])
    tr_role = rss.evaluate_cell_transport([r for r in unp_rows if r["cell"] == "ROLE_BARE"])
    l8_self = [r for r in head_rows if r["head_name"] == "L8H15" and r["cell"] == "SELF"]
    l8_role = [r for r in head_rows if r["head_name"] == "L8H15" and r["cell"] == "ROLE_BARE"]
    null_checks = {
        "SELF_even": float(statistics.mean(r["spec"]["wdown_orth"]["even_med"] for r in l8_self)),
        "SELF_odd": float(statistics.mean(r["spec"]["wdown_orth"]["odd_med"] for r in l8_self)),
        "ROLE_BARE_even": float(statistics.mean(r["spec"]["wdown_orth"]["even_med"] for r in l8_role)),
        "ROLE_BARE_odd": float(statistics.mean(r["spec"]["wdown_orth"]["odd_med"] for r in l8_role)),
    }
    null_ok = all(abs(v) <= rss.NULL_TOL for v in null_checks.values())
    if max_sh > rss.SHAM_TOL:
        gate = "invalidated"
    elif not (tr_self["passed"] and tr_role["passed"]):
        gate = "uninformative"
    elif not null_ok:
        gate = "unreadable"
    else:
        gate = "valid"
    return gate, {
        "gate": gate,
        "max_abs_sham": max_sh,
        "transport_self": tr_self,
        "transport_role_bare": tr_role,
        "l8h15_null": null_checks,
    }


def _evaluate_eval_arm(
    stage: str,
    arm_name: str,
    dose_mag: float | None,
    cal_summary: dict[str, Any],
    head_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if dose_mag is None:
        return {
            "arm": arm_name,
            "dose": None,
            "gate": f"uninformative ({cal_summary['cal_status']})",
            "metrics": None,
        }
    d_key = f"{dose_mag:.1f}"
    cal_d = cal_summary["per_dose"][d_key]

    # Gate 1: Sham across 100% of EVAL rows at this arm + CAL rows at this dose
    eval_max_sham = max(max(r["spec"]["max_abs_sham"], r["raw"]["max_abs_sham"]) for r in head_rows)
    cal_max_sham = cal_d["max_abs_sham"]
    max_sham = max(eval_max_sham, cal_max_sham)

    # Gate 2: Cell-wise CAL transport at this exact arm dose
    transport_ok = cal_d["transport_cal_self"]["passed"] and cal_d["transport_cal_third"]["passed"]

    # Gate 3: EVAL null gate on L8H15 & L7H5 on contrasted cells (SELF, ROLE_BARE, NAME_BARE, ALIAS_TO_ROLE, ALIAS_TO_NAME)
    l8_null_by_cell: dict[str, dict[str, float]] = {}
    l7_null_by_cell: dict[str, dict[str, float]] = {}
    for cell in ("SELF", "ROLE_BARE", "NAME_BARE", "ALIAS_TO_ROLE", "ALIAS_TO_NAME"):
        r8 = [r for r in head_rows if r["head_name"] == "L8H15" and r["cell"] == cell]
        r7 = [r for r in head_rows if r["head_name"] == "L7H5" and r["cell"] == cell]
        l8_null_by_cell[cell] = {
            "even": float(statistics.mean(r["spec"]["wdown_orth"]["even_med"] for r in r8)),
            "odd": float(statistics.mean(r["spec"]["wdown_orth"]["odd_med"] for r in r8)),
        }
        l7_null_by_cell[cell] = {
            "even": float(statistics.mean(r["spec"]["wdown_orth"]["even_med"] for r in r7)),
            "odd": float(statistics.mean(r["spec"]["wdown_orth"]["odd_med"] for r in r7)),
        }
    l8_null_ok = all(
        abs(v["even"]) <= rss.NULL_TOL and abs(v["odd"]) <= rss.NULL_TOL
        for v in l8_null_by_cell.values()
    )
    l7_null_ok = all(
        abs(v["even"]) <= rss.NULL_TOL and abs(v["odd"]) <= rss.NULL_TOL
        for v in l7_null_by_cell.values()
    )

    if max_sham > rss.SHAM_TOL:
        gate_l8 = "invalidated (sham_gate_failed)"
        gate_l7 = "invalidated (sham_gate_failed)"
    elif not transport_ok:
        gate_l8 = "uninformative (transport_gate_failed)"
        gate_l7 = "uninformative (transport_gate_failed)"
    else:
        gate_l8 = "valid" if l8_null_ok else "unreadable (null_gate_failed)"
        gate_l7 = "valid" if l7_null_ok else "unreadable (null_gate_failed)"

    # Index head rows by (head_name, cell, unit, story)
    idx: dict[tuple[str, str, str, str], dict[str, Any]] = {
        (r["head_name"], r["cell"], r["unit"], r["story"]): r for r in head_rows
    }

    def pair_diff(head: str, cell_a: str, cell_b: str, field: str = "spec", sign_key: str = "d_pos") -> dict[tuple[str, str], float]:
        return {
            (u, e): idx[(head, cell_a, u, e)][field][sign_key] - idx[(head, cell_b, u, e)][field][sign_key]
            for u in CORE_UNITS
            for e in EVAL_STORIES
        }

    c_sr_pairs = pair_diff("L8H15", "SELF", "ROLE_BARE", "spec", "d_pos")
    c_sr_raw_pairs = pair_diff("L8H15", "SELF", "ROLE_BARE", "raw", "d_pos")
    c_sr_pmed_pairs = pair_diff("L8H15", "SELF", "ROLE_BARE", "peer_med", "d_pos")
    c_sa_pairs = pair_diff("L8H15", "SELF", "ALIAS_TO_ROLE", "spec", "d_pos")
    d_sn_bare_pairs = pair_diff("L8H15", "SELF", "NAME_BARE", "spec", "d_pos")
    d_sn_alias_pairs = pair_diff("L8H15", "SELF", "ALIAS_TO_NAME", "spec", "d_pos")
    g_bare_pairs = pair_diff("L8H15", "ROLE_BARE", "NAME_BARE", "spec", "d_pos")
    g_alias_pairs = pair_diff("L8H15", "ALIAS_TO_ROLE", "ALIAS_TO_NAME", "spec", "d_pos")
    g_alias_raw_pairs = pair_diff("L8H15", "ALIAS_TO_ROLE", "ALIAS_TO_NAME", "raw", "d_pos")
    g_alias_pmed_pairs = pair_diff("L8H15", "ALIAS_TO_ROLE", "ALIAS_TO_NAME", "peer_med", "d_pos")

    rand_l8_c_sr = {
        h: float(statistics.mean(pair_diff(h, "SELF", "ROLE_BARE", "spec", "d_pos").values()))
        for h in RAND_L8_HEADS
    }
    rand_l8_g_bare = {
        h: float(statistics.mean(pair_diff(h, "ROLE_BARE", "NAME_BARE", "spec", "d_pos").values()))
        for h in RAND_L8_HEADS
    }
    rand_l8_g_alias = {
        h: float(statistics.mean(pair_diff(h, "ALIAS_TO_ROLE", "ALIAS_TO_NAME", "spec", "d_pos").values()))
        for h in RAND_L8_HEADS
    }

    # L7H5 summaries on ROLE_BARE and ALIAS_TO_ROLE
    def l7_block(cell: str) -> dict[str, Any]:
        def sub(units: tuple[str, ...]) -> dict[str, Any]:
            rows_sub = [idx[("L7H5", cell, u, e)] for u in units for e in EVAL_STORIES]
            dp = float(statistics.mean(r["spec"]["d_pos"] for r in rows_sub))
            dn = float(statistics.mean(r["spec"]["d_neg"] for r in rows_sub))
            ev = float(statistics.mean(r["spec"]["even"] for r in rows_sub))
            od = float(statistics.mean(r["spec"]["odd"] for r in rows_sub))
            ratio = abs(od) / (abs(ev) + 0.05)
            null_draw_means = [
                float(statistics.mean(r["spec"]["wdown_orth"]["d_neg_draws"][k] for r in rows_sub))
                for k in range(rss.N_NULL_DRAWS)
            ]
            return {
                "n": len(rows_sub),
                "d_pos_spec": dp,
                "d_neg_spec": dn,
                "d_pos_raw": float(statistics.mean(r["raw"]["d_pos"] for r in rows_sub)),
                "d_neg_raw": float(statistics.mean(r["raw"]["d_neg"] for r in rows_sub)),
                "even_spec": ev,
                "odd_spec": od,
                "odd_even_ratio_spec": ratio,
                "null_dneg_draw_means_spec": null_draw_means,
                "max_null_dneg_draw_mean_spec": max(null_draw_means),
            }
        return {
            "pooled_5": sub(CORE_UNITS),
            "l5l6_late_write": sub(L5L6_LATE_UNITS),
            "l3_early_write": sub(L3_EARLY_UNITS),
            "l6_only_p92_09_descriptive": sub(("p92-09",)),
        }

    return {
        "arm": arm_name,
        "dose": dose_mag,
        "gate": gate_l8,
        "gate_l7h5": gate_l7,
        "max_abs_sham": max_sham,
        "transport_ok": transport_ok,
        "l8h15_null_by_cell": l8_null_by_cell,
        "l7h5_null_by_cell": l7_null_by_cell,
        "C_SR_spec": _finite_summary(list(c_sr_pairs.values())),
        "C_SR_raw": _finite_summary(list(c_sr_raw_pairs.values())),
        "C_SR_peer_med": _finite_summary(list(c_sr_pmed_pairs.values())),
        "C_SR_by_unit": {u: float(statistics.mean(c_sr_pairs[(u, e)] for e in EVAL_STORIES)) for u in CORE_UNITS},
        "C_SR_by_story": {e: float(statistics.mean(c_sr_pairs[(u, e)] for u in CORE_UNITS)) for e in EVAL_STORIES},
        "C_SA_spec": _finite_summary(list(c_sa_pairs.values())),
        "rand_L8_C_SR": rand_l8_c_sr,
        "R_SR_max_rand_L8": max(rand_l8_c_sr.values()),
        "G_bare_spec": _finite_summary(list(g_bare_pairs.values())),
        "rand_L8_G_bare": rand_l8_g_bare,
        "R_bare_min_rand_L8": min(rand_l8_g_bare.values()),
        "G_alias_spec": _finite_summary(list(g_alias_pairs.values())),
        "G_alias_raw": _finite_summary(list(g_alias_raw_pairs.values())),
        "G_alias_peer_med": _finite_summary(list(g_alias_pmed_pairs.values())),
        "rand_L8_G_alias": rand_l8_g_alias,
        "R_alias_min_rand_L8": min(rand_l8_g_alias.values()),
        "delta_SN_bare_pairs": d_sn_bare_pairs,
        "delta_SN_alias_pairs": d_sn_alias_pairs,
        "c_sr_pairs": c_sr_pairs,
        "g_alias_pmed_pairs": g_alias_pmed_pairs,
        "l7h5_role_bare": l7_block("ROLE_BARE"),
        "l7h5_alias_to_role": l7_block("ALIAS_TO_ROLE"),
    }


def score_stage_sweep(
    manifest_path: Path,
    design_path: Path,
    stage_receipt_paths: dict[str, Path],
    expected_manifest_sha256: str,
    expected_design_sha256: str,
    expected_runner_sha256: str,
    expected_scorer_sha256: str,
) -> dict[str, Any]:
    # 1. Verify all 4 external 64-hex pins against actual disk files
    exp_m = _require_hex64("expected_manifest_sha256", expected_manifest_sha256)
    exp_d = _require_hex64("expected_design_sha256", expected_design_sha256)
    exp_r = _require_hex64("expected_runner_sha256", expected_runner_sha256)
    exp_s = _require_hex64("expected_scorer_sha256", expected_scorer_sha256)

    runner_path = Path(__file__).resolve().with_name("run_stage_sweep.py")
    scorer_path = Path(__file__).resolve()

    disk_m = rss.sha256_file(manifest_path)
    disk_d = rss.sha256_file(design_path)
    disk_r = rss.sha256_file(runner_path)
    disk_s = rss.sha256_file(scorer_path)

    for lbl, got, exp in (
        ("manifest", disk_m, exp_m),
        ("design", disk_d, exp_d),
        ("runner", disk_r, exp_r),
        ("scorer", disk_s, exp_s),
    ):
        if got != exp:
            raise ValueError(f"Disk {lbl} SHA-256 mismatch: disk={got} != expected={exp}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("meta", {}).get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Manifest schema_version mismatch")
    if manifest.get("meta", {}).get("design_sha256") != exp_d:
        raise ValueError("Manifest internal design_sha256 != expected_design_sha256")
    if not manifest.get("meta", {}).get("four_tokenizer_parity_verified"):
        raise ValueError("Manifest missing four_tokenizer_parity_verified=true")

    # Expected manifest join sets
    cal_prompts = [p for p in manifest["prompts"] if p["bank"] == "CAL"]
    cal_third_prompts = [p for p in cal_prompts if p["cell"] == "CAL_THIRD"]
    eval_prompts = [p for p in manifest["prompts"] if p["bank"] == "EVAL"]
    sens_prompts = [p for p in manifest["prompts"] if p["bank"] == "SENS_PLAIN"]
    pairs = rss.eligible_pairs()
    head_names = [hs["name"] for hs in rss.HEAD_SPECS]
    token_pool = manifest["meta"]["distinct_slot_token_pool"]

    exp_cal_unp_keys = {(u, p["prompt_id"], round(d, 4)) for u in CORE_UNITS for p in cal_prompts for d in rss.CAL_DOSE_GRID}
    exp_cal_head_keys = {(u, p["prompt_id"], h, round(d, 4)) for u in CORE_UNITS for p in cal_third_prompts for h in head_names for d in rss.CAL_DOSE_GRID}
    exp_eval_unp_keys_fn = lambda d: {(u, p["prompt_id"], round(d, 4)) for u in CORE_UNITS for p in eval_prompts}
    exp_eval_head_keys_fn = lambda d: {(u, p["prompt_id"], h, round(d, 4)) for u in CORE_UNITS for p in eval_prompts for h in head_names}
    exp_sens_unp_keys = {(u, p["prompt_id"], round(rss.COMMON_DOSE, 4)) for u in CORE_UNITS for p in sens_prompts}
    exp_sens_head_keys = {(u, p["prompt_id"], h, round(rss.COMMON_DOSE, 4)) for u in CORE_UNITS for p in sens_prompts for h in head_names}

    # 2. Load & strictly validate all 4 stage receipts
    receipts: dict[str, dict[str, Any]] = {}
    receipt_shas: dict[str, str] = {}
    cal_summaries: dict[str, dict[str, Any]] = {}
    arm_a_by_stage: dict[str, dict[str, Any]] = {}
    arm_b_by_stage: dict[str, dict[str, Any]] = {}
    ss7_by_stage: dict[str, dict[str, Any]] = {}

    for st in STAGE_ORDER:
        if st not in stage_receipt_paths:
            raise ValueError(f"Missing stage receipt path for {st}")
        rpath = stage_receipt_paths[st]
        receipt_shas[st] = rss.sha256_file(rpath)
        rec = json.loads(rpath.read_text(encoding="utf-8"))
        meta = rec.get("meta", {})

        if meta.get("artifact_type") != "stage_full_receipt" or meta.get("complete") is not True or meta.get("smoke_1unit") is not False:
            raise ValueError(f"Refused non-full receipt for {st}: artifact_type={meta.get('artifact_type')!r}, complete={meta.get('complete')!r}")
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Schema mismatch on {st}: {meta.get('schema_version')!r}")
        if meta.get("stage") != st or meta.get("model_commit") != rss.STAGE_SPECS[st]["commit"]:
            raise ValueError(f"Stage/commit mismatch on {st}: {meta.get('stage')!r} / {meta.get('model_commit')!r}")

        # Verify all 4 external pins against BOTH meta and meta.pins!
        pins = meta.get("pins", {})
        for k, exp in (
            ("manifest_sha256", exp_m),
            ("design_sha256", exp_d),
            ("runner_sha256", exp_r),
            ("scorer_sha256", exp_s),
        ):
            if meta.get(k) != exp or pins.get(k) != exp:
                raise ValueError(f"Receipt {st} pin mismatch on {k}: meta={meta.get(k)!r}, pins={pins.get(k)!r}, expected={exp!r}")

        # Exact row joins + causal + peer/algebraic assertions
        u_key = lambda r: (r["unit"], r["prompt_id"], round(float(r["dose_mag"]), 4))
        h_key = lambda r: (r["unit"], r["prompt_id"], r["head_name"], round(float(r["dose_mag"]), 4))

        _verify_exact_row_keys(rec["cal_unpatched_rows"], exp_cal_unp_keys, u_key, f"{st}.cal_unpatched")
        _verify_exact_row_keys(rec["cal_head_rows"], exp_cal_head_keys, h_key, f"{st}.cal_head")
        _verify_exact_row_keys(rec["eval_arm_a_unpatched_rows"], exp_eval_unp_keys_fn(rss.COMMON_DOSE), u_key, f"{st}.eval_a_unp")
        _verify_exact_row_keys(rec["eval_arm_a_head_rows"], exp_eval_head_keys_fn(rss.COMMON_DOSE), h_key, f"{st}.eval_a_head")
        _verify_exact_row_keys(rec["sens_plain_unpatched_rows"], exp_sens_unp_keys, u_key, f"{st}.sens_unp")
        _verify_exact_row_keys(rec["sens_plain_head_rows"], exp_sens_head_keys, h_key, f"{st}.sens_head")

        # Independently recompute Arm B calibration from raw CAL rows
        recomp_cal = rss.select_calibrated_dose(rec["cal_unpatched_rows"], rec["cal_head_rows"], dose_grid=rss.CAL_DOSE_GRID)
        if recomp_cal["cal_status"] != rec["cal_summary"]["cal_status"] or recomp_cal["d_cal"] != rec["cal_summary"]["d_cal"]:
            raise ValueError(f"Calibration mismatch on {st}: recomputed={recomp_cal['cal_status']}/{recomp_cal['d_cal']} vs receipt={rec['cal_summary']['cal_status']}/{rec['cal_summary']['d_cal']}")
        cal_summaries[st] = recomp_cal

        d_cal = recomp_cal["d_cal"]
        if recomp_cal["cal_status"] == "rebound_bracketed" and d_cal is not None:
            if abs(d_cal - rss.COMMON_DOSE) < 1e-12:
                if not meta.get("arm_b_reuses_arm_a"):
                    raise ValueError(f"Stage {st}: d_cal==0.8 but arm_b_reuses_arm_a is False")
                if rec["eval_arm_b_head_rows"] or rec["eval_arm_b_unpatched_rows"]:
                    raise ValueError(f"Stage {st}: unexpected duplicate Arm B rows when d_cal==0.8")
                arm_b_head_rows = rec["eval_arm_a_head_rows"]
            else:
                _verify_exact_row_keys(rec["eval_arm_b_unpatched_rows"], exp_eval_unp_keys_fn(d_cal), u_key, f"{st}.eval_b_unp")
                _verify_exact_row_keys(rec["eval_arm_b_head_rows"], exp_eval_head_keys_fn(d_cal), h_key, f"{st}.eval_b_head")
                arm_b_head_rows = rec["eval_arm_b_head_rows"]
        else:
            if rec["eval_arm_b_head_rows"] or rec["eval_arm_b_unpatched_rows"]:
                raise ValueError(f"Stage {st}: Arm B rows must be empty when cal_status={recomp_cal['cal_status']!r}")
            arm_b_head_rows = []

        # Verify strictly downstream, exact peer_deltas keys, and D_spec == D_raw - D_peer_med on all EVAL head rows
        for arm_lbl, hrows in (("arm_a", rec["eval_arm_a_head_rows"]), ("arm_b", rec["eval_arm_b_head_rows"])):
            for r in hrows:
                rss.assert_strictly_downstream(r["unit_layer"], r["head_layer"])
                if abs(r["spec"]["d_pos"] - (r["raw"]["d_pos"] - r["peer_med"]["d_pos"])) > 1e-5:
                    raise ValueError(f"Algebraic Spec!=Raw-PeerMed on {st}.{arm_lbl} {r['unit']}/{r['prompt_id']}/{r['head_name']}")
                exp_peers = {str(t) for t in manifest["bindings"][r["prompt_id"]][r["unit"]]["peers"]}
                if set(r.get("peer_deltas", {}).keys()) != exp_peers:
                    raise ValueError(f"Peer delta token keys mismatch on {st}.{arm_lbl} {r['unit']}/{r['prompt_id']}")

        arm_a_by_stage[st] = _evaluate_eval_arm(st, "arm_a_common_0.8", rss.COMMON_DOSE, recomp_cal, rec["eval_arm_a_head_rows"])
        arm_b_by_stage[st] = _evaluate_eval_arm(st, "arm_b_calibrated", d_cal, recomp_cal, arm_b_head_rows)

        # SS.7: matched E1/E2 chat vs plain gate parity
        chat_e1e2_unp = [r for r in rec["eval_arm_a_unpatched_rows"] if r["story"] in SENS_STORIES and r["cell"] in ("SELF", "ROLE_BARE")]
        chat_e1e2_head = [r for r in rec["eval_arm_a_head_rows"] if r["story"] in SENS_STORIES and r["cell"] in ("SELF", "ROLE_BARE")]
        chat_g, chat_g_det = _evaluate_matched_e1e2_gate(chat_e1e2_unp, chat_e1e2_head)
        plain_g, plain_g_det = _evaluate_matched_e1e2_gate(rec["sens_plain_unpatched_rows"], rec["sens_plain_head_rows"])

        def _c_sr_e1e2(hrows: list[dict[str, Any]]) -> float:
            m = {(r["cell"], r["unit"], r["story"]): r["spec"]["d_pos"] for r in hrows if r["head_name"] == "L8H15"}
            return float(statistics.mean(m[("SELF", u, e)] - m[("ROLE_BARE", u, e)] for u in CORE_UNITS for e in SENS_STORIES))

        c_chat_e1e2 = _c_sr_e1e2(chat_e1e2_head)
        c_plain_e1e2 = _c_sr_e1e2(rec["sens_plain_head_rows"])
        ss7_dec = decide_ss7_plain_vs_chat(chat_g, plain_g, c_chat_e1e2, c_plain_e1e2)
        ss7_dec["chat_gate_details"] = chat_g_det
        ss7_dec["plain_gate_details"] = plain_g_det
        ss7_by_stage[st] = ss7_dec
        receipts[st] = rec

    # Cross-stage unit vector cosine to instruct
    cos_to_instruct: dict[str, dict[str, float]] = {}
    inst_vecs = receipts["instruct"]["unit_vectors"]
    for st in STAGE_ORDER:
        cos_to_instruct[st] = {}
        for uid in CORE_UNITS:
            a = receipts[st]["unit_vectors"][uid]["hat_w"]
            b = inst_vecs[uid]["hat_w"]
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            cos_to_instruct[st][uid] = float(dot / max(na * nb, 1e-12))

    # 3. Evaluate SS.1 - SS.7 decisions
    ss1 = {
        "arm_a": {st: arm_a_by_stage[st]["gate"] for st in STAGE_ORDER},
        "arm_b": {st: arm_b_by_stage[st]["gate"] for st in STAGE_ORDER},
        "cal_status": {st: cal_summaries[st]["cal_status"] for st in STAGE_ORDER},
        "d_cal": {st: cal_summaries[st]["d_cal"] for st in STAGE_ORDER},
    }

    # SS.2 on instruct Arm A
    inst_a = arm_a_by_stage["instruct"]
    if inst_a["gate"] != "valid":
        ss2 = {"status": inst_a["gate"], "ss2a": {"status": inst_a["gate"]}, "ss2b": {"status": inst_a["gate"]}, "ss2c": {"status": inst_a["gate"]}}
    else:
        g_bare = inst_a["G_bare_spec"]["mean"]
        r_bare = inst_a["R_bare_min_rand_L8"]
        g_alias = inst_a["G_alias_spec"]["mean"]
        r_alias = inst_a["R_alias_min_rand_L8"]
        d_raw = inst_a["G_alias_raw"]["mean"]
        d_pmed = inst_a["G_alias_peer_med"]["mean"]

        # Exact 20-token peer sign reducer across all eligible (u, e) pairs
        inst_eval_idx = {
            (r["head_name"], r["cell"], r["unit"], r["story"]): r
            for r in receipts["instruct"]["eval_arm_a_head_rows"]
        }
        delta_t_map: dict[str, float] = {}
        for tid in token_pool:
            t_str = str(tid)
            diffs_t = []
            for u in CORE_UNITS:
                for e in EVAL_STORIES:
                    r_ar = inst_eval_idx[("L8H15", "ALIAS_TO_ROLE", u, e)]
                    r_an = inst_eval_idx[("L8H15", "ALIAS_TO_NAME", u, e)]
                    if t_str in r_ar["peer_deltas"]:
                        diffs_t.append(r_ar["peer_deltas"][t_str]["d_pos"] - r_an["peer_deltas"][t_str]["d_pos"])
            delta_t_map[t_str] = float(statistics.mean(diffs_t))
        frac_peer_tokens_pos = sum(1 for v in delta_t_map.values() if v > 0.0) / float(len(delta_t_map))
        frac_pairs_peer_med_pos = sum(1 for v in inst_a["g_alias_pmed_pairs"].values() if v > 0.0) / float(len(inst_a["g_alias_pmed_pairs"]))

        ss2a = decide_ss2a_bare(g_bare, r_bare)
        ss2b = decide_ss2b_alias(g_alias, r_alias, g_bare)
        ss2c = classify_ss2c_mechanism(d_raw, d_pmed, frac_peer_tokens_pos, frac_pairs_peer_med_pos)
        ss2c["delta_by_peer_token_id"] = delta_t_map
        ss2 = {"ss2a": ss2a, "ss2b": ss2b, "ss2c": ss2c}

    # SS.3, SS.4, SS.5 separately for Arm A and Arm B
    def score_arm_trajectory(arm_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
        gates = {st: arm_map[st]["gate"] for st in STAGE_ORDER}
        c_sr = {st: (arm_map[st]["C_SR_spec"]["mean"] if arm_map[st].get("C_SR_spec") is not None else None) for st in STAGE_ORDER}
        r_sft = arm_map["sft"].get("R_SR_max_rand_L8")
        ss3_res = decide_ss3_onset(gates["midtrain_end"], gates["sft"], c_sr["midtrain_end"], c_sr["sft"], r_sft)
        ss4_res = decide_ss4_later_change(gates, c_sr)

        ss5_by_stage: dict[str, Any] = {}
        for st in STAGE_ORDER:
            if gates[st] != "valid":
                ss5_by_stage[st] = {"status": gates[st], "population_equivalence": "not_claimed_finite_5x4_bank"}
            else:
                ss5_by_stage[st] = {
                    "bare": decide_ss5_finite_bank(arm_map[st]["delta_SN_bare_pairs"], c_sr[st]),
                    "alias": decide_ss5_finite_bank(arm_map[st]["delta_SN_alias_pairs"], c_sr[st]),
                }
        # Two-end role-to-name movement check (midtrain_end -> sft)
        if gates["midtrain_end"] == "valid" and gates["sft"] == "valid":
            mid_csr = c_sr["midtrain_end"]
            mid_u = arm_map["midtrain_end"]["C_SR_by_unit"]
            mid_e = arm_map["midtrain_end"]["C_SR_by_story"]
            mid_role_like = (
                abs(mid_csr) <= 0.15
                and max(abs(v) for v in mid_e.values()) <= 0.25
                and sum(1 for v in mid_u.values() if abs(v) <= 0.25) >= 4
                and arm_map["midtrain_end"]["G_bare_spec"]["mean"] <= -0.25
            )
            sft_name_like = ss5_by_stage["sft"]["bare"]["status"].startswith("holds")
            two_end_status = "holds (finite_bank_two_end_movement)" if (mid_role_like and sft_name_like) else "not_established"
        else:
            two_end_status = f"uninformative (mid={gates['midtrain_end']}, sft={gates['sft']})"
        return {
            "SS.3": ss3_res,
            "SS.4": ss4_res,
            "SS.5": {"by_stage": ss5_by_stage, "two_end_movement_midtrain_to_sft": two_end_status},
        }

    traj_a = score_arm_trajectory(arm_a_by_stage)
    traj_b = score_arm_trajectory(arm_b_by_stage)

    # SS.6 L7H5 instruct replication + 4-stage descriptive trajectory
    if inst_a["gate_l7h5"] != "valid":
        ss6_inst = {"status": inst_a["gate_l7h5"]}
    else:
        rb = inst_a["l7h5_role_bare"]
        ss6_inst = decide_ss6_l7h5_instruct(
            dneg_pooled=rb["pooled_5"]["d_neg_spec"],
            dneg_l5l6=rb["l5l6_late_write"]["d_neg_spec"],
            dneg_l3=rb["l3_early_write"]["d_neg_spec"],
            ratio_l5l6=rb["l5l6_late_write"]["odd_even_ratio_spec"],
            max_null_draw_mean=rb["pooled_5"]["max_null_dneg_draw_mean_spec"],
        )
    ss6 = {
        "instruct_arm_a_role_bare": ss6_inst,
        "by_stage_arm_a_role_bare": {st: arm_a_by_stage[st]["l7h5_role_bare"] for st in STAGE_ORDER},
        "by_stage_arm_a_alias_to_role": {st: arm_a_by_stage[st]["l7h5_alias_to_role"] for st in STAGE_ORDER},
    }

    # Strip large raw pair dicts from clean summary view while retaining all structured numbers
    def clean_arm_summary(arm_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
        out = {}
        for st, d in arm_map.items():
            out[st] = {
                k: v for k, v in d.items()
                if k not in ("delta_SN_bare_pairs", "delta_SN_alias_pairs", "c_sr_pairs", "g_alias_pmed_pairs")
            }
        return out

    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "verdict_type": "complete_4stage_scorecard",
            "complete_join_verified": True,
            "external_pins_verified": True,
            "manifest_sha256": exp_m,
            "design_sha256": exp_d,
            "runner_sha256": exp_r,
            "scorer_sha256": exp_s,
            "stage_receipt_sha256s": receipt_shas,
            "interpretive_caveats": [
                "Single training lineage (midtrain_end -> sft -> dpo -> RLVR/instruct), not independent replications.",
                "Same-coordinate tracking rule: null at L8H15/L7H5 means 'nothing at these coordinates', not 'absent'.",
                "Checkpoint-specific intervention result: stage-own write vectors and network parameters also vary.",
                "SS.5 is a finite-bank (5 units x 4 stories) consistency check; population equivalence is not claimed.",
            ],
        },
        "unit_vector_cos_to_instruct": cos_to_instruct,
        "cal_summaries": cal_summaries,
        "arm_a_common_0.8": clean_arm_summary(arm_a_by_stage),
        "arm_b_calibrated": clean_arm_summary(arm_b_by_stage),
        "decisions": {
            "SS.1": ss1,
            "SS.2": ss2,
            "SS.3": {"arm_a": traj_a["SS.3"], "arm_b": traj_b["SS.3"]},
            "SS.4": {"arm_a": traj_a["SS.4"], "arm_b": traj_b["SS.4"]},
            "SS.5": {"arm_a": traj_a["SS.5"], "arm_b": traj_b["SS.5"]},
            "SS.6": ss6,
            "SS.7": ss7_by_stage,
        },
    }


# ---------------------------------------------------------------------------
# Comprehensive CPU Self-Tests (All Alden #58 & #62 Witnesses + 4-Pin Refusals)
# ---------------------------------------------------------------------------

def run_self_tests() -> None:
    print("[self-test] Running score_stage_sweep.py witness, boundary & 4-pin verification suite...")

    # 1. Alden #62 Witness 1 — SS.2(b): G_bare=-0.50, R_bare=-0.25 (holds); G_alias=-0.60, R_alias=-0.75
    w2a = decide_ss2a_bare(-0.50, -0.25)
    assert w2a["status"] == "holds", w2a
    w2b = decide_ss2b_alias(g_alias=-0.60, r_alias=-0.75, g_bare=-0.50)
    assert w2b["status"] == "fails", w2b
    assert w2b["fail_reason"] == "random_head_comparison_failed", w2b
    assert abs(w2b["delta_alias_minus_bare"] - (-0.10)) < 1e-12, w2b
    assert w2b["attenuation_status"] == "preserved_or_stronger_under_same_referent_manipulation", w2b
    # Also check true attenuation case (G_bare=-0.50, G_alias=-0.10, R_alias=-0.05 -> alias_effect_weak + attenuated)
    w2b_att = decide_ss2b_alias(g_alias=-0.10, r_alias=-0.05, g_bare=-0.50)
    assert w2b_att["status"] == "fails" and w2b_att["fail_reason"] == "alias_effect_weak"
    assert w2b_att["attenuation_status"] == "attenuated_under_same_referent_manipulation"
    # Check inconclusive boundary
    w2b_inc = decide_ss2b_alias(g_alias=-0.20, r_alias=-0.10, g_bare=-0.50)
    assert w2b_inc["status"] == "inconclusive" and w2b_inc["fail_reason"] is None

    # 2. Alden #62 Witness 2 — SS.7: valid chat=-0.20, plain=+0.30 -> plain_positive_only
    w7 = decide_ss7_plain_vs_chat("valid", "valid", -0.20, +0.30)
    assert w7["status"] == "plain_positive_only", w7
    # Exhaustive check of ALL 16 (chat_gate, plain_gate) ordered pairs
    gate_states = ("invalidated", "uninformative", "unreadable", "valid")
    seen_16 = {}
    for cg in gate_states:
        for pg in gate_states:
            d = decide_ss7_plain_vs_chat(cg, pg, 0.30, 0.30)
            seen_16[(cg, pg)] = d["status"]
    assert len(seen_16) == 16
    assert seen_16[("uninformative", "valid")] == "transport_diff_plain_only"
    assert seen_16[("valid", "uninformative")] == "transport_diff_chat_only"
    assert seen_16[("uninformative", "uninformative")] == "uninformative (both_transport_failed)"
    assert seen_16[("uninformative", "unreadable")] == "unreadable"
    assert seen_16[("invalidated", "valid")] == "invalidated"
    # Exhaustive 49 boundary/control numeric pairs for valid SS.7 (7 x 7 grid across -0.20, 0.0, 0.149, 0.15, 0.249, 0.25, 0.30)
    grid7 = (-0.20, 0.0, 0.149, 0.15, 0.249, 0.25, 0.30)
    statuses_49 = set()
    for cv in grid7:
        for pv in grid7:
            st = decide_ss7_plain_vs_chat("valid", "valid", cv, pv)["status"]
            assert st in (
                "chat_independent (preserved_across_wrappers)",
                "chat_dependent",
                "plain_positive_only",
                "neither_or_intermediate",
            )
            statuses_49.add(st)
    assert len(statuses_49) == 4, statuses_49
    assert decide_ss7_plain_vs_chat("valid", "valid", 0.25, 0.25)["status"] == "chat_independent (preserved_across_wrappers)"
    assert decide_ss7_plain_vs_chat("valid", "valid", 0.25, 0.149)["status"] == "chat_dependent"
    assert decide_ss7_plain_vs_chat("valid", "valid", 0.25, 0.15)["status"] == "neither_or_intermediate"

    # 3. Alden #58 Witness 3 — SS.3 unassigned gap witness (SFT +0.35, midtrain +0.25, onset +0.10 -> inconclusive)
    w3_inc = decide_ss3_onset("valid", "valid", c_mid=0.25, c_sft=0.35, r_sft=0.05)
    assert w3_inc["status"] == "inconclusive", w3_inc
    w3_hold = decide_ss3_onset("valid", "valid", c_mid=0.05, c_sft=0.40, r_sft=0.10)
    assert w3_hold["status"] == "holds", w3_hold
    w3_fail = decide_ss3_onset("valid", "valid", c_mid=0.40, c_sft=0.35, r_sft=0.10)
    assert w3_fail["status"] == "fails", w3_fail
    w3_uninf = decide_ss3_onset("uninformative (transport_gate_failed)", "valid", c_mid=None, c_sft=0.40, r_sft=0.10)
    assert w3_uninf["status"] == "uninformative (midtrain_gate_or_cal_failed)"
    assert w3_uninf["first_measurable_at_sft_status"] == "holds (first_measurable_at_sft)"

    # 4. Alden #58 Witness 4 — SS.4 large negative post-SFT drop -> fails_attenuation_or_loss
    g_ok = {"sft": "valid", "dpo": "valid", "instruct": "valid"}
    assert decide_ss4_later_change(g_ok, {"sft": 0.60, "dpo": 0.20, "instruct": 0.20})["status"] == "fails_attenuation_or_loss"
    assert decide_ss4_later_change(g_ok, {"sft": 0.30, "dpo": 0.60, "instruct": 0.65})["status"] == "fails_amplification"
    assert decide_ss4_later_change(g_ok, {"sft": 0.30, "dpo": 0.60, "instruct": 0.30})["status"] == "fails_non_monotone_mixed"
    assert decide_ss4_later_change(g_ok, {"sft": 0.40, "dpo": 0.45, "instruct": 0.50})["status"] == "holds (post_sft_stability)"

    # 5. SS.5 two-factor finite-bank consistency (fails story consistency even when grand mean is 0!)
    bad_story = {(u, e): (0.40 if e in ("E1", "E2") else -0.40) for u in CORE_UNITS for e in EVAL_STORIES}
    assert decide_ss5_finite_bank(bad_story, 0.50)["status"] == "inconclusive"
    good_fb = {(u, e): 0.05 for u in CORE_UNITS for e in EVAL_STORIES}
    res_fb = decide_ss5_finite_bank(good_fb, 0.50)
    assert res_fb["status"] == "holds (finite_bank_close_and_separated_from_role)"
    assert res_fb["population_equivalence"] == "not_claimed_finite_5x4_bank"

    # 6. End-to-end synthetic 4-stage scoring + 4-pin external verification & refusals
    design_p = Path("explore_olmo2_attn_mlp/STAGE-SWEEP-DESIGN.md")
    runner_p = Path(__file__).resolve().with_name("run_stage_sweep.py")
    scorer_p = Path(__file__).resolve()
    d_sha = rss.sha256_file(design_p)
    r_sha = rss.sha256_file(runner_p)
    s_sha = rss.sha256_file(scorer_p)

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        shared_tok = rss._FakeTokenizer()
        manifest = rss.build_manifest(design_path=design_p, tokenizer_loader=lambda r, c: shared_tok)
        m_path = tdp / "manifest.json"
        m_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        m_sha = rss.sha256_file(m_path)

        cal_prompts = [p for p in manifest["prompts"] if p["bank"] == "CAL"]
        cal_third = [p for p in cal_prompts if p["cell"] == "CAL_THIRD"]
        eval_prompts = [p for p in manifest["prompts"] if p["bank"] == "EVAL"]
        sens_prompts = [p for p in manifest["prompts"] if p["bank"] == "SENS_PLAIN"]

        def make_stage_receipt(stage: str, c_sr_target: float, rebound_d: float | None) -> dict[str, Any]:
            cal_unp, cal_head = [], []
            for u in CORE_UNITS:
                ul = rss.SLOT_BY_ID[u]["layer"]
                for d in rss.CAL_DOSE_GRID:
                    for p in cal_prompts:
                        cal_unp.append({
                            "stage": stage, "unit": u, "unit_layer": ul,
                            "prompt_id": p["prompt_id"], "bank": "CAL", "cell": p["cell"],
                            "story": p["story"], "serialization": "chat", "dose_mag": d,
                            "injected_norm": d * 8.0, "read_pos_resid_norm": 8.0, "norm_ratio": d,
                            "transport": {
                                "n_write_spec": 1.2, "n_write_raw": 1.2, "n_write_peer_med": 0.0,
                                "n_write_neg_spec": -1.0, "n_write_neg_raw": -1.0,
                                "wdown_pos_draws_spec": [0.02] * 4, "wdown_pos_draws_raw": [0.02] * 4,
                                "iso_pos_draws_spec": [0.01] * 4, "iso_pos_draws_raw": [0.01] * 4,
                            },
                        })
                    for p in cal_third:
                        for hs in rss.HEAD_SPECS:
                            if rebound_d is not None and hs["name"] == "L8H15":
                                f_val = -0.60 if abs(d - rebound_d) < 1e-9 else (0.10 if d > rebound_d else -0.20)
                            else:
                                # Monotone unbracketed for midtrain_end!
                                f_val = -0.30 - 0.10 * d if hs["name"] == "L8H15" else -0.05
                            cal_head.append({
                                "stage": stage, "unit": u, "unit_layer": ul,
                                "prompt_id": p["prompt_id"], "bank": "CAL", "cell": "CAL_THIRD",
                                "story": p["story"], "serialization": "chat", "dose_mag": d,
                                "head_name": hs["name"], "head_layer": hs["layer"], "head_idx": hs["head"], "head_kind": hs["kind"],
                                "spec": {"d0": 0.0, "d_pos": f_val, "d_neg": -0.1, "even": 0.5 * (f_val - 0.1), "odd": 0.5 * (f_val + 0.1), "max_abs_sham": 0.0, "wdown_orth": {"d_pos_draws": [0.01]*4, "d_neg_draws": [0.01]*4, "even_draws": [0.01]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.01, "d_neg_med": 0.01, "even_med": 0.01, "odd_med": 0.0, "even_max_abs": 0.01, "odd_max_abs": 0.0}, "iso_resid": {"d_pos_draws": [0.0]*4, "d_neg_draws": [0.0]*4, "even_draws": [0.0]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.0, "d_neg_med": 0.0, "even_med": 0.0, "odd_med": 0.0, "even_max_abs": 0.0, "odd_max_abs": 0.0}},
                                "raw": {"d0": 0.0, "d_pos": 0.0, "d_neg": 0.0, "even": 0.0, "odd": 0.0, "max_abs_sham": 0.0, "wdown_orth": {"d_pos_draws": [0.0]*4, "d_neg_draws": [0.0]*4, "even_draws": [0.0]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.0, "d_neg_med": 0.0, "even_med": 0.0, "odd_med": 0.0, "even_max_abs": 0.0, "odd_max_abs": 0.0}, "iso_resid": {"d_pos_draws": [0.0]*4, "d_neg_draws": [0.0]*4, "even_draws": [0.0]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.0, "d_neg_med": 0.0, "even_med": 0.0, "odd_med": 0.0, "even_max_abs": 0.0, "odd_max_abs": 0.0}},
                                "peer_med": {"d0": 0.0, "d_pos": -f_val, "d_neg": 0.1, "even": -0.5 * (f_val - 0.1), "odd": -0.5 * (f_val + 0.1), "max_abs_sham": 0.0, "wdown_orth": {"d_pos_draws": [-0.01]*4, "d_neg_draws": [-0.01]*4, "even_draws": [-0.01]*4, "odd_draws": [0.0]*4, "d_pos_med": -0.01, "d_neg_med": -0.01, "even_med": -0.01, "odd_med": 0.0, "even_max_abs": 0.01, "odd_max_abs": 0.0}, "iso_resid": {"d_pos_draws": [0.0]*4, "d_neg_draws": [0.0]*4, "even_draws": [0.0]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.0, "d_neg_med": 0.0, "even_med": 0.0, "odd_med": 0.0, "even_max_abs": 0.0, "odd_max_abs": 0.0}},
                            })
            cal_sum = rss.select_calibrated_dose(cal_unp, cal_head)

            def make_eval_rows(d: float):
                eu, eh = [], []
                for u in CORE_UNITS:
                    ul = rss.SLOT_BY_ID[u]["layer"]
                    for p in eval_prompts:
                        eu.append({
                            "stage": stage, "unit": u, "unit_layer": ul,
                            "prompt_id": p["prompt_id"], "bank": "EVAL", "cell": p["cell"],
                            "story": p["story"], "serialization": "chat", "dose_mag": d,
                            "injected_norm": d * 8.0, "read_pos_resid_norm": 8.0, "norm_ratio": d,
                            "transport": {
                                "n_write_spec": 1.2, "n_write_raw": 1.2, "n_write_peer_med": 0.0,
                                "n_write_neg_spec": -1.0, "n_write_neg_raw": -1.0,
                                "wdown_pos_draws_spec": [0.02]*4, "wdown_pos_draws_raw": [0.02]*4,
                                "iso_pos_draws_spec": [0.01]*4, "iso_pos_draws_raw": [0.01]*4,
                            },
                        })
                        peers = manifest["bindings"][p["prompt_id"]][u]["peers"]
                        for hs in rss.HEAD_SPECS:
                            if hs["name"] == "L8H15":
                                if p["cell"] in ("ROLE_BARE", "ALIAS_TO_ROLE"):
                                    dp_raw, dp_pm = -0.05, c_sr_target - 0.05
                                    dp_spec = dp_raw - dp_pm
                                else:
                                    dp_raw, dp_pm, dp_spec = 0.0, 0.0, 0.0
                                dn_spec = -0.1
                            elif hs["name"] == "L7H5":
                                dp_spec = -0.50 if u in L5L6_LATE_UNITS else -0.10
                                dn_spec = +0.40 if u in L5L6_LATE_UNITS else +0.02
                                dp_raw, dp_pm = dp_spec, 0.0
                            else:
                                dp_raw, dp_pm, dp_spec, dn_spec = 0.0, 0.0, 0.0, 0.0
                            p_del = {str(tid): {"d_pos": dp_pm, "d_neg": 0.0} for tid in peers}
                            eh.append({
                                "stage": stage, "unit": u, "unit_layer": ul,
                                "prompt_id": p["prompt_id"], "bank": "EVAL", "cell": p["cell"],
                                "story": p["story"], "serialization": "chat", "dose_mag": d,
                                "head_name": hs["name"], "head_layer": hs["layer"], "head_idx": hs["head"], "head_kind": hs["kind"],
                                "spec": {"d0": 0.0, "d_pos": dp_spec, "d_neg": dn_spec, "even": 0.5*(dp_spec+dn_spec), "odd": 0.5*(dp_spec-dn_spec), "max_abs_sham": 0.0, "wdown_orth": {"d_pos_draws": [0.01]*4, "d_neg_draws": [0.01]*4, "even_draws": [0.01]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.01, "d_neg_med": 0.01, "even_med": 0.01, "odd_med": 0.0, "even_max_abs": 0.01, "odd_max_abs": 0.0}, "iso_resid": {"d_pos_draws": [0.0]*4, "d_neg_draws": [0.0]*4, "even_draws": [0.0]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.0, "d_neg_med": 0.0, "even_med": 0.0, "odd_med": 0.0, "even_max_abs": 0.0, "odd_max_abs": 0.0}},
                                "raw": {"d0": 0.0, "d_pos": dp_raw, "d_neg": 0.0, "even": 0.5*dp_raw, "odd": 0.5*dp_raw, "max_abs_sham": 0.0, "wdown_orth": {"d_pos_draws": [0.0]*4, "d_neg_draws": [0.0]*4, "even_draws": [0.0]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.0, "d_neg_med": 0.0, "even_med": 0.0, "odd_med": 0.0, "even_max_abs": 0.0, "odd_max_abs": 0.0}, "iso_resid": {"d_pos_draws": [0.0]*4, "d_neg_draws": [0.0]*4, "even_draws": [0.0]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.0, "d_neg_med": 0.0, "even_med": 0.0, "odd_med": 0.0, "even_max_abs": 0.0, "odd_max_abs": 0.0}},
                                "peer_med": {"d0": 0.0, "d_pos": dp_pm, "d_neg": -dn_spec, "even": 0.5*(dp_pm-dn_spec), "odd": 0.5*(dp_pm+dn_spec), "max_abs_sham": 0.0, "wdown_orth": {"d_pos_draws": [-0.01]*4, "d_neg_draws": [-0.01]*4, "even_draws": [-0.01]*4, "odd_draws": [0.0]*4, "d_pos_med": -0.01, "d_neg_med": -0.01, "even_med": -0.01, "odd_med": 0.0, "even_max_abs": 0.01, "odd_max_abs": 0.0}, "iso_resid": {"d_pos_draws": [0.0]*4, "d_neg_draws": [0.0]*4, "even_draws": [0.0]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.0, "d_neg_med": 0.0, "even_med": 0.0, "odd_med": 0.0, "even_max_abs": 0.0, "odd_max_abs": 0.0}},
                                "peer_deltas": p_del, "peer_mean_d_pos": dp_pm, "peer_iqr_d_pos": 0.0,
                            })
                return eu, eh

            ea_u, ea_h = make_eval_rows(0.8)
            d_cal = cal_sum["d_cal"]
            reuses = bool(cal_sum["cal_status"] == "rebound_bracketed" and d_cal is not None and abs(d_cal - 0.8) < 1e-12)
            if cal_sum["cal_status"] == "rebound_bracketed" and d_cal is not None and not reuses:
                eb_u, eb_h = make_eval_rows(d_cal)
            else:
                eb_u, eb_h = [], []

            su, sh = [], []
            for u in CORE_UNITS:
                ul = rss.SLOT_BY_ID[u]["layer"]
                for p in sens_prompts:
                    su.append({
                        "stage": stage, "unit": u, "unit_layer": ul,
                        "prompt_id": p["prompt_id"], "bank": "SENS_PLAIN", "cell": p["cell"],
                        "story": p["story"], "serialization": "plain", "dose_mag": 0.8,
                        "injected_norm": 6.4, "read_pos_resid_norm": 8.0, "norm_ratio": 0.8,
                        "transport": {
                            "n_write_spec": 1.1, "n_write_raw": 1.1, "n_write_peer_med": 0.0,
                            "n_write_neg_spec": -0.9, "n_write_neg_raw": -0.9,
                            "wdown_pos_draws_spec": [0.02]*4, "wdown_pos_draws_raw": [0.02]*4,
                            "iso_pos_draws_spec": [0.01]*4, "iso_pos_draws_raw": [0.01]*4,
                        },
                    })
                    for hs in rss.HEAD_SPECS:
                        dp_spec = -c_sr_target if (hs["name"] == "L8H15" and p["cell"] == "ROLE_BARE") else 0.0
                        sh.append({
                            "stage": stage, "unit": u, "unit_layer": ul,
                            "prompt_id": p["prompt_id"], "bank": "SENS_PLAIN", "cell": p["cell"],
                            "story": p["story"], "serialization": "plain", "dose_mag": 0.8,
                            "head_name": hs["name"], "head_layer": hs["layer"], "head_idx": hs["head"], "head_kind": hs["kind"],
                            "spec": {"d0": 0.0, "d_pos": dp_spec, "d_neg": 0.0, "even": 0.5*dp_spec, "odd": 0.5*dp_spec, "max_abs_sham": 0.0, "wdown_orth": {"d_pos_draws": [0.01]*4, "d_neg_draws": [0.01]*4, "even_draws": [0.01]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.01, "d_neg_med": 0.01, "even_med": 0.01, "odd_med": 0.0, "even_max_abs": 0.01, "odd_max_abs": 0.0}, "iso_resid": {"d_pos_draws": [0.0]*4, "d_neg_draws": [0.0]*4, "even_draws": [0.0]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.0, "d_neg_med": 0.0, "even_med": 0.0, "odd_med": 0.0, "even_max_abs": 0.0, "odd_max_abs": 0.0}},
                            "raw": {"d0": 0.0, "d_pos": 0.0, "d_neg": 0.0, "even": 0.0, "odd": 0.0, "max_abs_sham": 0.0, "wdown_orth": {"d_pos_draws": [0.0]*4, "d_neg_draws": [0.0]*4, "even_draws": [0.0]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.0, "d_neg_med": 0.0, "even_med": 0.0, "odd_med": 0.0, "even_max_abs": 0.0, "odd_max_abs": 0.0}, "iso_resid": {"d_pos_draws": [0.0]*4, "d_neg_draws": [0.0]*4, "even_draws": [0.0]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.0, "d_neg_med": 0.0, "even_med": 0.0, "odd_med": 0.0, "even_max_abs": 0.0, "odd_max_abs": 0.0}},
                            "peer_med": {"d0": 0.0, "d_pos": -dp_spec, "d_neg": 0.0, "even": -0.5*dp_spec, "odd": -0.5*dp_spec, "max_abs_sham": 0.0, "wdown_orth": {"d_pos_draws": [-0.01]*4, "d_neg_draws": [-0.01]*4, "even_draws": [-0.01]*4, "odd_draws": [0.0]*4, "d_pos_med": -0.01, "d_neg_med": -0.01, "even_med": -0.01, "odd_med": 0.0, "even_max_abs": 0.01, "odd_max_abs": 0.0}, "iso_resid": {"d_pos_draws": [0.0]*4, "d_neg_draws": [0.0]*4, "even_draws": [0.0]*4, "odd_draws": [0.0]*4, "d_pos_med": 0.0, "d_neg_med": 0.0, "even_med": 0.0, "odd_med": 0.0, "even_max_abs": 0.0, "odd_max_abs": 0.0}},
                        })

            pins = {"manifest_sha256": m_sha, "design_sha256": d_sha, "runner_sha256": r_sha, "scorer_sha256": s_sha}
            return {
                "meta": {
                    "schema_version": SCHEMA_VERSION, "stage": stage,
                    "model_repo": rss.STAGE_SPECS[stage]["repo"], "model_commit": rss.STAGE_SPECS[stage]["commit"],
                    "artifact_type": "stage_full_receipt", "run_mode": "full", "complete": True, "smoke_1unit": False,
                    "arm_b_reuses_arm_a": reuses,
                    **pins, "pins": pins,
                },
                "unit_vectors": {u: {"hat_w": [1.0, 0.0, 0.0, 0.0]} for u in CORE_UNITS},
                "cal_summary": cal_sum,
                "cal_unpatched_rows": cal_unp,
                "cal_head_rows": cal_head,
                "eval_arm_a_unpatched_rows": ea_u,
                "eval_arm_a_head_rows": ea_h,
                "eval_arm_b_unpatched_rows": eb_u,
                "eval_arm_b_head_rows": eb_h,
                "sens_plain_unpatched_rows": su,
                "sens_plain_head_rows": sh,
            }

        stage_paths: dict[str, Path] = {}
        # midtrain_end has monotone unbracketed CAL (Arm B = None!), sft rebounds at 0.6, dpo & instruct at 0.8
        configs = {
            "midtrain_end": (0.05, None),
            "sft": (0.50, 0.6),
            "dpo": (0.55, 0.8),
            "instruct": (0.60, 0.8),
        }
        for st, (csr_v, reb_d) in configs.items():
            rp = tdp / f"{st}.json"
            rp.write_text(json.dumps(make_stage_receipt(st, csr_v, reb_d)), encoding="utf-8")
            stage_paths[st] = rp

        sc = score_stage_sweep(m_path, design_p, stage_paths, m_sha, d_sha, r_sha, s_sha)
        assert sc["meta"]["external_pins_verified"] and sc["meta"]["complete_join_verified"]
        assert sc["decisions"]["SS.2"]["ss2a"]["status"] == "holds"
        assert sc["decisions"]["SS.2"]["ss2b"]["status"] == "holds"
        assert sc["decisions"]["SS.2"]["ss2c"]["classification"] == "peer_dominated"
        assert sc["decisions"]["SS.3"]["arm_a"]["status"] == "holds"
        # Arm B midtrain_end was unbracketed -> uninformative + first_measurable_at_sft holds!
        assert sc["decisions"]["SS.3"]["arm_b"]["status"] == "uninformative (midtrain_gate_or_cal_failed)"
        assert sc["decisions"]["SS.3"]["arm_b"]["first_measurable_at_sft_status"] == "holds (first_measurable_at_sft)"
        assert sc["decisions"]["SS.4"]["arm_a"]["status"] == "holds (post_sft_stability)"
        assert sc["decisions"]["SS.4"]["arm_b"]["status"] == "holds (post_sft_stability)"
        assert sc["decisions"]["SS.6"]["instruct_arm_a_role_bare"]["status"] == "holds"
        assert sc["decisions"]["SS.7"]["instruct"]["status"] == "chat_independent (preserved_across_wrappers)"

        # External 4-pin refusals (wrong runner/scorer/manifest/design pin, plus smoke receipt & missing row!)
        for bad_kwargs in (
            dict(expected_manifest_sha256="0" * 64, expected_design_sha256=d_sha, expected_runner_sha256=r_sha, expected_scorer_sha256=s_sha),
            dict(expected_manifest_sha256=m_sha, expected_design_sha256="1" * 64, expected_runner_sha256=r_sha, expected_scorer_sha256=s_sha),
            dict(expected_manifest_sha256=m_sha, expected_design_sha256=d_sha, expected_runner_sha256="2" * 64, expected_scorer_sha256=s_sha),
            dict(expected_manifest_sha256=m_sha, expected_design_sha256=d_sha, expected_runner_sha256=r_sha, expected_scorer_sha256="3" * 64),
        ):
            try:
                score_stage_sweep(m_path, design_p, stage_paths, **bad_kwargs)
                raise AssertionError(f"Should have refused wrong pin: {bad_kwargs}")
            except ValueError:
                pass

        # Receipt tampered pin (runner_sha256 = 0*64 inside receipt)
        bad_rec = json.loads(stage_paths["instruct"].read_text())
        bad_rec["meta"]["runner_sha256"] = "0" * 64
        bad_p = tdp / "bad_inst.json"
        bad_p.write_text(json.dumps(bad_rec), encoding="utf-8")
        try:
            score_stage_sweep(m_path, design_p, {**stage_paths, "instruct": bad_p}, m_sha, d_sha, r_sha, s_sha)
            raise AssertionError("Should have refused tampered receipt runner_sha256")
        except ValueError:
            pass

        # Smoke receipt refusal
        smoke_rec = json.loads(stage_paths["instruct"].read_text())
        smoke_rec["meta"]["artifact_type"] = "stage_smoke_receipt"
        smoke_rec["meta"]["smoke_1unit"] = True
        smoke_p = tdp / "smoke_inst.json"
        smoke_p.write_text(json.dumps(smoke_rec), encoding="utf-8")
        try:
            score_stage_sweep(m_path, design_p, {**stage_paths, "instruct": smoke_p}, m_sha, d_sha, r_sha, s_sha)
            raise AssertionError("Should have refused stage_smoke_receipt")
        except ValueError:
            pass

        # Missing row refusal
        miss_rec = json.loads(stage_paths["instruct"].read_text())
        miss_rec["eval_arm_a_head_rows"].pop()
        miss_p = tdp / "miss_inst.json"
        miss_p.write_text(json.dumps(miss_rec), encoding="utf-8")
        try:
            score_stage_sweep(m_path, design_p, {**stage_paths, "instruct": miss_p}, m_sha, d_sha, r_sha, s_sha)
            raise AssertionError("Should have refused missing EVAL row")
        except ValueError:
            pass

    print("[self-test] ALL score_stage_sweep.py tests PASSED (incl. #58 & #62 witnesses + 4-pin refusals)!")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Pinned Offline Scorer for OLMo 2 1B Stage-Sweep (v2.1)")
    p.add_argument("--self-test-only", action="store_true")
    p.add_argument("--manifest", type=Path, default=Path("explore_olmo2_attn_mlp/stage_sweep_manifest.json"))
    p.add_argument("--design", type=Path, default=Path("explore_olmo2_attn_mlp/STAGE-SWEEP-DESIGN.md"))
    p.add_argument("--midtrain", type=Path, default=None)
    p.add_argument("--sft", type=Path, default=None)
    p.add_argument("--dpo", type=Path, default=None)
    p.add_argument("--instruct", type=Path, default=None)
    p.add_argument("--expected-manifest-sha256", type=str, default=None)
    p.add_argument("--expected-design-sha256", type=str, default=None)
    p.add_argument("--expected-runner-sha256", type=str, default=None)
    p.add_argument("--expected-scorer-sha256", type=str, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.self_test_only:
        run_self_tests()
        return

    # Strictly require all 4 external pins AND all 4 stage receipts on CLI!
    missing = [
        name for name, val in (
            ("--midtrain", args.midtrain),
            ("--sft", args.sft),
            ("--dpo", args.dpo),
            ("--instruct", args.instruct),
            ("--expected-manifest-sha256", args.expected_manifest_sha256),
            ("--expected-design-sha256", args.expected_design_sha256),
            ("--expected-runner-sha256", args.expected_runner_sha256),
            ("--expected-scorer-sha256", args.expected_scorer_sha256),
        )
        if val is None
    ]
    if missing:
        raise SystemExit(f"Refused unchecked CLI invocation; missing required flags: {', '.join(missing)}")

    scorecard = score_stage_sweep(
        manifest_path=args.manifest,
        design_path=args.design,
        stage_receipt_paths={
            "midtrain_end": args.midtrain,
            "sft": args.sft,
            "dpo": args.dpo,
            "instruct": args.instruct,
        },
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_design_sha256=args.expected_design_sha256,
        expected_runner_sha256=args.expected_runner_sha256,
        expected_scorer_sha256=args.expected_scorer_sha256,
    )
    txt = json.dumps(scorecard, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
