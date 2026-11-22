#!/usr/bin/env python3
"""OLMo 2 1B Instruct — Part B (#4): Causal Attention-Reading & Span-Swap Offline Scorer.

Implements every pre-registered gate and total machine-readable endpoint (`CM.0`–`CM.4`)
of `explore_olmo2_attn_mlp/CAUSAL-ATTN-DESIGN.md` v3
(SHA-256: 9b41482fbd1fae267ece14a859274ddf886de3ecd94526e44a4c5697821030fa).

Contracts:
- Single-read external 4-pin (`design`, `manifest`, `runner`, `scorer`) + Part A scorecard
  pin (`9e23bba5...`) verification.
- `--verify-smoke-only`: strict technical verification of smoke receipt (`3 / 66 / 17 / 28`
  rows, `135` B19 + `77` B=1 forwards, full `CM.0` reconciliation, actuation, sibling
  isolation, sham, decomposition, timing) with ZERO scientific verdicts emitted.
- Full scoring mode: strictly REFUSES smoke receipts (`rc != 0`, no scorecard written),
  enforces `CM.0` strict precedence, `CM.1a` ALL-`L7H5` drop/stop rule (hard-gating both
  `p92-46` and `p92-09` on a primary `p92-46` drop), strictly disjoint `CM.2`/`CM.3`
  7-branch partitions + orthogonal `control_selectivity`, and 6-state `CM.4` decision map
  emitting signed `M_N2S` / `M_S2N` and all 6 unpooled control strata.
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

# Allow running directly as a script from any working directory.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import run_causal_attn as rca


CM0_FAIL_SCI = "uninformative (cm0_reconciliation_failed)"
L3_UNITS: tuple[str, ...] = ("p92-02", "p92-05", "p92-13")
L7_CTRL_HEADS: tuple[str, ...] = ("L7H4", "L7H9")
L8_CTRL_HEADS: tuple[str, ...] = ("L8H3", "L8H13")


def _mean(vals: list[float]) -> float:
    if not vals:
        raise ValueError("Cannot take mean of empty list")
    return float(sum(vals) / len(vals))


def _is_finite_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


# ---------------------------------------------------------------------------
# CM.0 — Technical Reconciliation, Actuation, Zero-Leakage & Join Validator
# ---------------------------------------------------------------------------

def _verify_b19_metric_summary(m: dict[str, Any], label: str) -> tuple[float, float]:
    """Verify finiteness, draw-based even/odd/median consistency, and return (max_abs_sham, max_deriv_err)."""
    for k in ("d0", "d_pos", "d_neg", "even", "odd", "max_abs_sham"):
        if not _is_finite_num(m.get(k)):
            raise ValueError(f"{label}: non-finite/missing {k}")
    d0, dp, dn = float(m["d0"]), float(m["d_pos"]), float(m["d_neg"])
    max_err = max(
        abs(float(m["even"]) - (0.5 * (dp + dn) - d0)),
        abs(float(m["odd"]) - (0.5 * (dp - dn))),
    )
    for grp in ("wdown_orth", "iso_resid"):
        g = m.get(grp)
        if not isinstance(g, dict):
            raise ValueError(f"{label}: missing {grp}")
        dp_d = g.get("d_pos_draws")
        dn_d = g.get("d_neg_draws")
        ev_d = g.get("even_draws")
        od_d = g.get("odd_draws")
        if not all(isinstance(x, list) and len(x) == rca.N_NULL_DRAWS for x in (dp_d, dn_d, ev_d, od_d)):
            raise ValueError(f"{label}.{grp}: bad draw array lengths")
        for arr in (dp_d, dn_d, ev_d, od_d):
            if not all(_is_finite_num(v) for v in arr):
                raise ValueError(f"{label}.{grp}: non-finite draw")
        for a, b, ev, od in zip(dp_d, dn_d, ev_d, od_d):
            max_err = max(max_err, abs(float(ev) - (0.5 * (float(a) + float(b)) - d0)))
            max_err = max(max_err, abs(float(od) - (0.5 * (float(a) - float(b)))))
        max_err = max(
            max_err,
            abs(float(g["d_pos_med"]) - float(statistics.median(dp_d))),
            abs(float(g["d_neg_med"]) - float(statistics.median(dn_d))),
            abs(float(g["even_med"]) - float(statistics.median(ev_d))),
            abs(float(g["odd_med"]) - float(statistics.median(od_d))),
            abs(float(g["even_max_abs"]) - max(abs(float(x)) for x in ev_d)),
            abs(float(g["odd_max_abs"]) - max(abs(float(x)) for x in od_d)),
        )
    return float(m["max_abs_sham"]), max_err


def validate_cm0_and_join(
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    pins: dict[str, str],
    part_a_scorecard: dict[str, Any],
    expect_full: bool,
    allow_fixture: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify exact schema, pins, Part A inheritance, complete join, and all 5 CM.0 clauses.

    Structural/provenance/pin mismatches raise ValueError (refusal).
    Scientific/technical CM.0 clause breaches (sham, decomp, reconciliation, sibling leakage,
    or actuation failure) are recorded cleanly in `cm0_report` so full scoring can gate
    CM.1-CM.4 to `uninformative (cm0_reconciliation_failed)` without silently passing!
    """
    meta = receipt.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Receipt missing meta object")
    if meta.get("schema_version") != rca.SCHEMA_VERSION:
        raise ValueError(f"Schema version mismatch: {meta.get('schema_version')!r}")
    exp_art = "causal_attn_full_receipt" if expect_full else "causal_attn_smoke_receipt"
    if meta.get("artifact_type") != exp_art:
        raise ValueError(f"Artifact type mismatch: got {meta.get('artifact_type')!r}, expected {exp_art!r}")
    exp_mode = "full" if expect_full else "smoke"
    if meta.get("run_mode") != exp_mode:
        raise ValueError(f"Run mode mismatch: got {meta.get('run_mode')!r}, expected {exp_mode!r}")
    prov = meta.get("execution_provenance")
    dev_name = meta.get("device_name")
    is_cuda_dev = meta.get("is_cuda_device")
    if not allow_fixture:
        if prov != "measured_gpu_run":
            raise ValueError(f"Refused non-measured execution_provenance={prov!r} (expected 'measured_gpu_run')")
        if is_cuda_dev is not True:
            raise ValueError(f"Refused measured_gpu_run without is_cuda_device=True (got {is_cuda_dev!r})")
        dev_norm = dev_name.strip().lower() if isinstance(dev_name, str) else ""
        if not dev_norm or dev_norm in ("cpu", "mps", "meta", "none", "unknown") or dev_norm.startswith("cpu"):
            raise ValueError(f"Refused measured_gpu_run with non-GPU device_name={dev_name!r}")

    # Pin & Part A handoff equality.
    rp = meta.get("pins", {})
    for k, v in pins.items():
        if rp.get(k) != v:
            raise ValueError(f"Receipt meta.pins[{k}] mismatch: got {rp.get(k)!r}, expected {v!r}")
    handoff = rca.resolve_part_a_handoff(part_a_scorecard)
    if meta.get("operating_norm_name") != handoff["operating_norm_name"]:
        raise ValueError("Receipt operating_norm_name mismatch vs Part A handoff")
    if meta.get("operating_norm_multipliers") != handoff["operating_norm_multipliers"]:
        raise ValueError("Receipt operating_norm_multipliers mismatch vs Part A handoff")
    if meta.get("part_a_part_b_norm_decision") != handoff["part_b_norm_decision"]:
        raise ValueError("Receipt part_a_part_b_norm_decision mismatch vs Part A scorecard")
    if meta.get("part_a_p92_09_part_b_track") != handoff["p92_09_part_b_track"]:
        raise ValueError("Receipt part_a_p92_09_part_b_track mismatch vs Part A scorecard")

    sched_key = "schedule_full" if expect_full else "schedule_smoke"
    sched = manifest[sched_key]
    den = manifest["meta"]["denominators"]["full" if expect_full else "smoke"]

    # Validate timing breakdown & forward counts (including projection consistency vs full denominators).
    den_full = manifest["meta"]["denominators"]["full"]
    rca.validate_timing_breakdown(
        meta.get("timing"),
        den["n_b19_forwards"],
        den["n_b1_forwards"],
        full_b19=den_full["n_b19_forwards"],
        full_b1=den_full["n_b1_forwards"],
    )
    if int(meta.get("n_b19_forwards", -1)) != den["n_b19_forwards"]:
        raise ValueError("Receipt n_b19_forwards count mismatch")
    if int(meta.get("n_b1_forwards", -1)) != den["n_b1_forwards"]:
        raise ValueError("Receipt n_b1_forwards count mismatch")

    unp_rows = receipt.get("unpatched_rows", [])
    hm_rows = receipt.get("head_mode_rows", [])
    act_rows = receipt.get("actuation_rows", [])
    sw_rows = receipt.get("swap_rows", [])

    if len(unp_rows) != den["n_unpatched_rows"]:
        raise ValueError(f"unpatched_rows len={len(unp_rows)} != {den['n_unpatched_rows']}")
    if len(hm_rows) != den["n_head_mode_rows"]:
        raise ValueError(f"head_mode_rows len={len(hm_rows)} != {den['n_head_mode_rows']}")
    if len(act_rows) != den["n_actuation_rows"]:
        raise ValueError(f"actuation_rows len={len(act_rows)} != {den['n_actuation_rows']}")
    if len(sw_rows) != den["n_swap_rows"]:
        raise ValueError(f"swap_rows len={len(sw_rows)} != {den['n_swap_rows']}")

    # Expected exact key sets.
    exp_unp_keys = {(u, p["prompt_id"]) for u in sched["units"] for p in sched["cm123_prompts"]}
    exp_hm_keys = {
        (u, p["prompt_id"], hm["head_name"], hm["mode"])
        for u in sched["units"]
        for p in sched["cm123_prompts"]
        for hm in sched["cm123_head_modes"]
    }
    exp_act_keys = {a["actuation_id"] for a in sched["actuation"]}
    exp_sw_keys = {
        (t["unit"], t["target_prompt_id"], t["donor_prompt_id"], t["swap_cond"], t["span_name"], h)
        for t in sched["cm4_tuples"]
        for h in sched["cm4_heads"]
    }

    unp_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    hm_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    act_by_key: dict[str, dict[str, Any]] = {}
    sw_by_key: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}

    failures: list[str] = []
    max_sham_spec = 0.0
    max_sham_raw = 0.0
    max_decomp = 0.0
    max_deriv = 0.0
    max_sib_science = 0.0
    max_sib_act = 0.0
    max_rec_pre_oproj = 0.0
    max_rec_full_vocab = 0.0
    max_rec_drop = 0.0
    min_act_target = float("inf")

    # 1. Verify unpatched rows (NO sham keys, finite, exact linear decomposition).
    for r in unp_rows:
        k = (r["unit"], r["prompt_id"])
        if k in unp_by_key:
            raise ValueError(f"Duplicate unpatched row key: {k}")
        unp_by_key[k] = r
        if any("sham" in fk.lower() for fk in r.keys()):
            failures.append(f"unpatched_row_has_sham_key:{k}")
        tr = r.get("transport", {})
        for fld in ("n_write_spec", "n_write_raw", "n_write_peer_med", "n_write_neg_spec", "n_write_neg_raw", "n_write_neg_peer_med"):
            if not _is_finite_num(tr.get(fld)):
                failures.append(f"unpatched_nonfinite:{k}:{fld}")
        errs = [
            abs(float(tr["n_write_spec"]) - (float(tr["n_write_raw"]) - float(tr["n_write_peer_med"]))),
            abs(float(tr["n_write_neg_spec"]) - (float(tr["n_write_neg_raw"]) - float(tr["n_write_neg_peer_med"]))),
        ]
        for grp in ("wdown_pos_draws", "iso_pos_draws"):
            sp_d = tr[f"{grp}_spec"]
            rw_d = tr[f"{grp}_raw"]
            pm_d = tr[f"{grp}_peer_med"]
            for a, b, c in zip(sp_d, rw_d, pm_d):
                errs.append(abs(float(a) - (float(b) - float(c))))
        max_decomp = max(max_decomp, max(errs))

    if set(unp_by_key.keys()) != exp_unp_keys:
        raise ValueError("Unpatched row keys do not match manifest schedule")

    # 2. Verify head-mode rows (sham, deriv consistency, linear decomp, sibling isolation).
    for r in hm_rows:
        k = (r["unit"], r["prompt_id"], r["head_name"], r["mode"])
        if k in hm_by_key:
            raise ValueError(f"Duplicate head_mode row key: {k}")
        hm_by_key[k] = r
        sh_s, dr_s = _verify_b19_metric_summary(r["spec"], f"hm:{k}:spec")
        sh_r, dr_r = _verify_b19_metric_summary(r["raw"], f"hm:{k}:raw")
        _sh_p, dr_p = _verify_b19_metric_summary(r["peer_med"], f"hm:{k}:pmed")
        max_sham_spec = max(max_sham_spec, sh_s)
        max_sham_raw = max(max_sham_raw, sh_r)
        max_deriv = max(max_deriv, dr_s, dr_r, dr_p)
        # Recompute linear decomposition independently.
        sp, rw, pm = r["spec"], r["raw"], r["peer_med"]
        d_errs = [abs(float(sp[fld]) - (float(rw[fld]) - float(pm[fld]))) for fld in ("d0", "d_pos", "d_neg", "even", "odd")]
        for grp in ("wdown_orth", "iso_resid"):
            for dk in ("d_pos_draws", "d_neg_draws", "even_draws", "odd_draws"):
                for a, b, c in zip(sp[grp][dk], rw[grp][dk], pm[grp][dk]):
                    d_errs.append(abs(float(a) - (float(b) - float(c))))
        max_decomp = max(max_decomp, max(d_errs))

        sib = r.get("sibling_max_abs_leakage")
        if not _is_finite_num(sib):
            failures.append(f"hm_missing_sibling_leakage:{k}")
        else:
            max_sib_science = max(max_sib_science, float(sib))

    if set(hm_by_key.keys()) != exp_hm_keys:
        raise ValueError("Head-mode row keys do not match manifest schedule")

    # 2b. Verify 3-part QKV_all == z_all reconciliation on EVERY (unit, prompt_id, head_name).
    n_qkv_reconciled = 0
    for k, r in hm_by_key.items():
        uid, pid, hname, mode = k
        if mode != "QKV_all":
            continue
        rec_obj = r.get("reconciliation_qkv_vs_z_all")
        if not isinstance(rec_obj, dict):
            failures.append(f"missing_qkv_reconciliation:{k}")
            continue
        for fld in ("pre_oproj_head_tensor_max_abs", "full_vocab_read_pos_logits_max_abs", "readout_drop_max_abs"):
            if not _is_finite_num(rec_obj.get(fld)):
                failures.append(f"nonfinite_qkv_reconciliation:{k}:{fld}")
        z_row = hm_by_key[(uid, pid, hname, "z_all")]
        recomputed_drop_err = max(
            abs(float(r["spec"]["d_pos"]) - float(z_row["spec"]["d_pos"])),
            abs(float(r["spec"]["d_neg"]) - float(z_row["spec"]["d_neg"])),
            abs(float(r["raw"]["d_pos"]) - float(z_row["raw"]["d_pos"])),
            abs(float(r["raw"]["d_neg"]) - float(z_row["raw"]["d_neg"])),
        )
        max_rec_pre_oproj = max(max_rec_pre_oproj, float(rec_obj["pre_oproj_head_tensor_max_abs"]))
        max_rec_full_vocab = max(max_rec_full_vocab, float(rec_obj["full_vocab_read_pos_logits_max_abs"]))
        max_rec_drop = max(max_rec_drop, float(rec_obj["readout_drop_max_abs"]), recomputed_drop_err)
        n_qkv_reconciled += 1

    # 3. Verify all 17 dedicated synthetic nonzero-donor actuation positive controls.
    act_passed_count = 0
    for r in act_rows:
        aid = r["actuation_id"]
        if aid in act_by_key:
            raise ValueError(f"Duplicate actuation_id: {aid}")
        act_by_key[aid] = r
        td = r.get("target_max_abs_delta")
        sl = r.get("sibling_max_abs_leakage")
        if not (_is_finite_num(td) and _is_finite_num(sl)):
            failures.append(f"actuation_nonfinite:{aid}")
            continue
        min_act_target = min(min_act_target, float(td))
        max_sib_act = max(max_sib_act, float(sl))
        row_ok = (
            float(td) > rca.MIN_ACTUATION_DELTA
            and float(sl) <= rca.TOL_SIBLING
            and float(r.get("unpatched_pos_max_abs_leakage", 0.0)) <= rca.TOL_SIBLING
            and bool(r.get("passed", False))
        )
        if row_ok:
            act_passed_count += 1
        else:
            failures.append(f"actuation_failed:{aid}:td={td}:sl={sl}")

    if set(act_by_key.keys()) != exp_act_keys:
        raise ValueError("Actuation row keys do not match manifest schedule")

    # 4. Verify CM.4 swap rows (exact token/position bindings, sham, decomp, sibling isolation).
    for r in sw_rows:
        k = (r["unit"], r["target_prompt_id"], r["donor_prompt_id"], r["swap_cond"], r["span_name"], r["head_name"])
        if k in sw_by_key:
            raise ValueError(f"Duplicate swap row key: {k}")
        sw_by_key[k] = r
        uid, t_pid, d_pid, _sc, sp_name, _hn = k
        t_bind = manifest["bindings"][t_pid][uid]
        d_bind = manifest["bindings"][d_pid][uid]
        if r.get("target_input_ids") != t_bind["input_ids"]:
            failures.append(f"swap_target_ids_mismatch:{k}")
        if r.get("donor_input_ids") != d_bind["input_ids"]:
            failures.append(f"swap_donor_ids_mismatch:{k}")
        if r.get("swap_indices") != rca.FROZEN_SPANS[sp_name]:
            failures.append(f"swap_indices_mismatch:{k}")
        if r.get("position_ids") != list(range(33)):
            failures.append(f"swap_position_ids_mismatch:{k}")
        for fld in ("delta_spec", "delta_raw", "delta_peer_med", "sham_spec", "sham_raw", "sham_peer_med", "sibling_max_abs_leakage"):
            if not _is_finite_num(r.get(fld)):
                failures.append(f"swap_nonfinite:{k}:{fld}")
        max_sham_spec = max(max_sham_spec, abs(float(r["sham_spec"])))
        max_sham_raw = max(max_sham_raw, abs(float(r["sham_raw"])))
        max_decomp = max(
            max_decomp,
            abs(float(r["delta_spec"]) - (float(r["delta_raw"]) - float(r["delta_peer_med"]))),
            abs(float(r["sham_spec"]) - (float(r["sham_raw"]) - float(r["sham_peer_med"]))),
        )
        max_sib_science = max(max_sib_science, float(r["sibling_max_abs_leakage"]))

    if set(sw_by_key.keys()) != exp_sw_keys:
        raise ValueError("Swap row keys do not match manifest schedule")

    # Evaluate exact CM.0 clauses 1-5.
    clause_2_ok = (
        max_sham_spec <= rca.TOL_SHAM
        and max_sham_raw <= rca.TOL_SHAM
        and max_decomp <= rca.TOL_DECOMP
        and max_deriv <= rca.TOL_DECOMP
    )
    clause_3_ok = (
        max_rec_pre_oproj <= rca.TOL_RECONCILE
        and max_rec_full_vocab <= rca.TOL_RECONCILE
        and max_rec_drop <= rca.TOL_RECONCILE
    )
    clause_4_ok = max_sib_science <= rca.TOL_SIBLING
    clause_5_ok = act_passed_count == den["n_actuation_rows"] and min_act_target > rca.MIN_ACTUATION_DELTA and max_sib_act <= rca.TOL_SIBLING

    if not clause_2_ok:
        failures.append(f"clause_2_sham_decomp_failed:sham_s={max_sham_spec},sham_r={max_sham_raw},decomp={max_decomp},deriv={max_deriv}")
    if not clause_3_ok:
        failures.append(f"clause_3_qkv_reconcile_failed:pre={max_rec_pre_oproj},vocab={max_rec_full_vocab},drop={max_rec_drop}")
    if not clause_4_ok:
        failures.append(f"clause_4_sibling_leakage_failed:max_sib={max_sib_science}")
    if not clause_5_ok:
        failures.append(f"clause_5_actuation_failed:passed={act_passed_count}/{den['n_actuation_rows']}")

    cm0_passed = len(failures) == 0
    cm0_report = {
        "status": "passed" if cm0_passed else "failed",
        "passed": cm0_passed,
        "clause_1_pins_and_part_a_handoff": True,
        "clause_2_sham_decomp_finiteness": clause_2_ok,
        "clause_3_qkv_all_vs_z_all_reconciliation": clause_3_ok,
        "clause_4_science_sibling_zero_leakage": clause_4_ok,
        "clause_5_synthetic_actuation_controls": clause_5_ok,
        "n_qkv_all_reconciled": n_qkv_reconciled,
        "actuation_passed_count": act_passed_count,
        "actuation_expected_count": den["n_actuation_rows"],
        "max_abs_sham_spec": max_sham_spec,
        "max_abs_sham_raw": max_sham_raw,
        "max_abs_decomp_err": max_decomp,
        "max_abs_deriv_err": max_deriv,
        "max_sibling_leakage_science": max_sib_science,
        "max_sibling_leakage_actuation": max_sib_act,
        "min_actuation_target_delta": min_act_target,
        "max_reconcile_pre_oproj": max_rec_pre_oproj,
        "max_reconcile_full_vocab": max_rec_full_vocab,
        "max_reconcile_drop": max_rec_drop,
        "failures": failures,
    }
    bound = {
        "unp_by_key": unp_by_key,
        "hm_by_key": hm_by_key,
        "act_by_key": act_by_key,
        "sw_by_key": sw_by_key,
        "handoff": handoff,
    }
    return cm0_report, bound


def verify_smoke_receipt_only(
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    pins: dict[str, str],
    part_a_scorecard: dict[str, Any],
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Technical-only smoke verifier (ZERO scientific verdicts emitted; invoked pre-load by --mode full)."""
    cm0_report, _bound = validate_cm0_and_join(
        receipt=receipt,
        manifest=manifest,
        pins=pins,
        part_a_scorecard=part_a_scorecard,
        expect_full=False,
        allow_fixture=allow_fixture,
    )
    meta = receipt["meta"]
    return {
        "artifact_type": "causal_attn_smoke_verification",
        "execution_provenance": meta["execution_provenance"],
        "smoke_technical_check_passed": bool(cm0_report["passed"]),
        "operating_norm_name": meta["operating_norm_name"],
        "counts": {
            "n_unpatched_rows": meta["n_unpatched_rows"],
            "n_head_mode_rows": meta["n_head_mode_rows"],
            "n_actuation_rows": meta["n_actuation_rows"],
            "n_swap_rows": meta["n_swap_rows"],
            "n_b19_forwards": meta["n_b19_forwards"],
            "n_b1_forwards": meta["n_b1_forwards"],
        },
        "CM0_technical_check": cm0_report,
        "timing": meta["timing"],
    }


# ---------------------------------------------------------------------------
# Pure Decision-Partition Classifiers (CM.2, CM.3, CM.4) — Exhaustively Unit-Tested
# ---------------------------------------------------------------------------

def classify_cm2_factorization(
    upstream_gate: str,
    d_tot: float,
    d_v: float,
    d_qk: float,
    ctrl_v_max: float | None = None,
    ctrl_qk_max: float | None = None,
    ctrl_margin_thresh: float = 0.15,
) -> dict[str, Any]:
    """Strictly disjoint, ordered 7-branch CM.2 within-head partition + orthogonal control_selectivity (v3 §3 CM.2)."""
    d_int = d_tot - d_v - d_qk
    f_v = (d_v / d_tot) if d_tot >= 0.25 else None
    f_qk = (d_qk / d_tot) if d_tot >= 0.25 else None
    f_int = (d_int / d_tot) if d_tot >= 0.25 else None

    if upstream_gate != "confirmed":
        verdict = upstream_gate
        winning_factor = None
    elif d_tot < 0.25:
        verdict = "effect_too_small_or_wrong_sign"
        winning_factor = None
    elif f_v >= 0.70 and abs(f_qk) < 0.25 and abs(f_int) < 0.25:
        verdict = "v_dominant_within_head"
        winning_factor = "V_only"
    elif f_qk >= 0.70 and abs(f_v) < 0.25 and abs(f_int) < 0.25:
        verdict = "qk_dominant_within_head"
        winning_factor = "QK_only"
    elif abs(f_int) >= 0.30:
        verdict = "interaction_required_within_head"
        winning_factor = None
    elif f_v >= 0.25 and f_qk >= 0.25 and abs(f_int) < 0.30:
        verdict = "additive_both_within_head"
        winning_factor = None
    else:
        verdict = "mixed_or_inconclusive_within_head"
        winning_factor = None

    ctrl_sel = "not_applicable"
    ctrl_margin = None
    if winning_factor == "V_only" and ctrl_v_max is not None:
        ctrl_margin = d_v - ctrl_v_max
        ctrl_sel = (
            "selective_over_matched_controls"
            if ctrl_margin >= ctrl_margin_thresh
            else "within_head_only_not_control_selective"
        )
    elif winning_factor == "QK_only" and ctrl_qk_max is not None:
        ctrl_margin = d_qk - ctrl_qk_max
        ctrl_sel = (
            "selective_over_matched_controls"
            if ctrl_margin >= ctrl_margin_thresh
            else "within_head_only_not_control_selective"
        )

    return {
        "verdict": verdict,
        "control_selectivity": ctrl_sel,
        "winning_factor": winning_factor,
        "control_margin": ctrl_margin,
        "control_margin_threshold": ctrl_margin_thresh,
        "D_tot": d_tot,
        "D_V": d_v,
        "D_QK": d_qk,
        "D_int": d_int,
        "f_V": f_v,
        "f_QK": f_qk,
        "f_int": f_int,
    }


def classify_cm3_position_scope(
    upstream_gate: str,
    d_all: float,
    d_rp: float,
    d_up: float,
    ctrl_rp_max: float | None = None,
    ctrl_up_max: float | None = None,
    ctrl_margin_thresh: float = 0.15,
) -> dict[str, Any]:
    """Strictly disjoint, ordered 7-branch CM.3 position-scope partition + orthogonal control_selectivity (v3 §3 CM.3)."""
    i_pos = d_all - (d_rp + d_up)
    f_rp = (d_rp / d_all) if d_all >= 0.25 else None
    f_up = (d_up / d_all) if d_all >= 0.25 else None
    f_int = (i_pos / d_all) if d_all >= 0.25 else None

    if upstream_gate != "confirmed":
        verdict = upstream_gate
        winning_scope = None
    elif d_all < 0.25:
        verdict = "effect_too_small_or_wrong_sign"
        winning_scope = None
    elif f_rp >= 0.70 and abs(f_up) < 0.25 and abs(f_int) < 0.25:
        verdict = "read_pos_dominant_within_head"
        winning_scope = "z_read_pos"
    elif f_up >= 0.70 and abs(f_rp) < 0.25 and abs(f_int) < 0.25:
        verdict = "upstream_dominant_within_head"
        winning_scope = "z_upstream"
    elif abs(f_int) >= 0.30:
        verdict = "interaction_required_within_head"
        winning_scope = None
    elif f_rp >= 0.25 and f_up >= 0.25 and abs(f_int) < 0.30:
        verdict = "additive_both_within_head"
        winning_scope = None
    else:
        verdict = "mixed_or_inconclusive_within_head"
        winning_scope = None

    ctrl_sel = "not_applicable"
    ctrl_margin = None
    if winning_scope == "z_read_pos" and ctrl_rp_max is not None:
        ctrl_margin = d_rp - ctrl_rp_max
        ctrl_sel = (
            "selective_over_matched_controls"
            if ctrl_margin >= ctrl_margin_thresh
            else "within_head_only_not_control_selective"
        )
    elif winning_scope == "z_upstream" and ctrl_up_max is not None:
        ctrl_margin = d_up - ctrl_up_max
        ctrl_sel = (
            "selective_over_matched_controls"
            if ctrl_margin >= ctrl_margin_thresh
            else "within_head_only_not_control_selective"
        )

    return {
        "verdict": verdict,
        "control_selectivity": ctrl_sel,
        "winning_scope": winning_scope,
        "control_margin": ctrl_margin,
        "control_margin_threshold": ctrl_margin_thresh,
        "D_all": d_all,
        "D_rp": d_rp,
        "D_up": d_up,
        "I_pos": i_pos,
        "f_rp": f_rp,
        "f_up": f_up,
        "f_pos_int": f_int,
    }


def classify_cm4_swap(
    upstream_gate: str,
    m_n2s: float,
    m_s2n: float,
    c_ctrl: float,
) -> dict[str, Any]:
    """Exhaustive, mutually exclusive 6-state CM.4 decision map (v3 §3 CM.4; Alden #121/#124).

    Always emits both signed `M_N2S` and `M_S2N` so any reverse component on an asymmetric
    outcome is transparently reported alongside the verdict.
    """
    pos_n2s = (m_n2s >= 0.15) and ((m_n2s - c_ctrl) >= 0.10)
    pos_s2n = (m_s2n >= 0.15) and ((m_s2n - c_ctrl) >= 0.10)

    if upstream_gate != "confirmed":
        verdict = upstream_gate
    elif m_n2s >= 0.15 and m_s2n >= 0.15 and (min(m_n2s, m_s2n) - c_ctrl) >= 0.10:
        verdict = "reciprocal_tested_coords_localized"
    elif pos_n2s ^ pos_s2n:
        verdict = "asymmetric_tested_coords_effect"
    elif (m_n2s <= -0.15 or m_s2n <= -0.15) and (max(-m_n2s, -m_s2n) - c_ctrl) >= 0.10:
        verdict = "opposite_direction_at_tested_coords"
    elif max(abs(m_n2s), abs(m_s2n)) < 0.10 and c_ctrl < 0.10:
        verdict = "small_effect_at_tested_coords"
    else:
        verdict = "inconclusive_or_control_confounded"

    return {
        "verdict": verdict,
        "M_N2S": float(m_n2s),
        "M_S2N": float(m_s2n),
        "mean_delta_N2S": float(m_n2s),
        "mean_delta_S2N": float(-m_s2n),
        "C_ctrl": float(c_ctrl),
        "min_pos_margin_over_ctrl": float(min(m_n2s, m_s2n) - c_ctrl),
        "max_neg_margin_over_ctrl": float(max(-m_n2s, -m_s2n) - c_ctrl),
    }


# ---------------------------------------------------------------------------
# Full Scientific Scoring (CM.0 -> CM.1 -> CM.2 -> CM.3 -> CM.4)
# ---------------------------------------------------------------------------

def _transport_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ms = _mean([float(r["transport"]["n_write_spec"]) for r in rows])
    mr = _mean([float(r["transport"]["n_write_raw"]) for r in rows])
    mp = _mean([float(r["transport"]["n_write_peer_med"]) for r in rows])
    wd_means = [_mean([float(r["transport"]["wdown_pos_draws_spec"][k]) for r in rows]) for k in range(rca.N_NULL_DRAWS)]
    max_wd_null = max(wd_means)
    passed = bool(ms >= 0.50 and mr >= 0.50 and ms > max_wd_null)
    return {
        "n_rows": len(rows),
        "mean_spec": ms,
        "mean_raw": mr,
        "mean_peer_med": mp,
        "max_wdown_null_spec": max_wd_null,
        "cell_transport_pass": passed,
    }


def _head_null_stats(hm_rows_cell: list[dict[str, Any]]) -> dict[str, Any]:
    ev = _mean([float(r["spec"]["wdown_orth"]["even_med"]) for r in hm_rows_cell])
    od = _mean([float(r["spec"]["wdown_orth"]["odd_med"]) for r in hm_rows_cell])
    passed = bool(abs(ev) <= 0.15 and abs(od) <= 0.15)
    return {
        "n_rows": len(hm_rows_cell),
        "mean_wdown_even_med_spec": ev,
        "mean_wdown_odd_med_spec": od,
        "head_null_pass": passed,
    }


def score_full_receipt(
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    pins: dict[str, str],
    part_a_scorecard: dict[str, Any],
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Score a full Part B receipt deterministically across CM.0-CM.4."""
    cm0, bound = validate_cm0_and_join(
        receipt=receipt,
        manifest=manifest,
        pins=pins,
        part_a_scorecard=part_a_scorecard,
        expect_full=True,
        allow_fixture=allow_fixture,
    )
    cm0_passed = bool(cm0["passed"])
    unp_by_key = bound["unp_by_key"]
    hm_by_key = bound["hm_by_key"]
    sw_by_key = bound["sw_by_key"]
    handoff = bound["handoff"]
    stories = [s["story"] for s in rca.BANK_STORIES]

    # Helpers to query CM.1-CM.3 rows cleanly.
    def _unp_cell(cell: str, units: tuple[str, ...] = rca.CORE_UNIT_IDS) -> list[dict[str, Any]]:
        return [unp_by_key[(u, f"{s}_{cell}")] for u in units for s in stories]

    def _hm_cell(head: str, cell: str, mode: str, units: tuple[str, ...] = rca.CORE_UNIT_IDS) -> list[dict[str, Any]]:
        return [hm_by_key[(u, f"{s}_{cell}", head, mode)] for u in units for s in stories]

    def _s_odd_unit_role(head: str, unit: str, mode: str, ch: str) -> dict[str, float]:
        rows = _hm_cell(head, "ROLE_BARE", mode, (unit,))
        dp = _mean([float(r[ch]["d_pos"]) for r in rows])
        dn = _mean([float(r[ch]["d_neg"]) for r in rows])
        return {"d_pos": dp, "d_neg": dn, "S_odd": 0.5 * (dn - dp)}

    def _l3_mean_s_odd_role(head: str, mode: str, ch: str) -> float:
        return _mean([_s_odd_unit_role(head, u, mode, ch)["S_odd"] for u in L3_UNITS])

    def _l8_metrics(head: str, mode: str, ch: str) -> dict[str, float]:
        d_self = _mean([float(r[ch]["d_pos"]) for r in _hm_cell(head, "SELF", mode)])
        d_name = _mean([float(r[ch]["d_pos"]) for r in _hm_cell(head, "NAME_BARE", mode)])
        d_role = _mean([float(r[ch]["d_pos"]) for r in _hm_cell(head, "ROLE_BARE", mode)])
        return {
            "d_pos_SELF": d_self,
            "d_pos_NAME_BARE": d_name,
            "d_pos_ROLE_BARE": d_role,
            "NAME_BARE_suppression": -d_name,
            "ROLE_BARE_suppression": -d_role,
            "C_SN": d_self - d_name,
            "C_SR": d_self - d_role,
        }

    # =========================================================================
    # CM.1a (L7H5 Confirmatory Gate & Pre-Bridge ALL-L7H5 Drop/Stop Rule)
    # =========================================================================
    tr_by_cell = {c: _transport_stats(_unp_cell(c)) for c in rca.BANK_CELLS}
    tr_p92_46_role = _transport_stats(_unp_cell("ROLE_BARE", ("p92-46",)))
    tr_p92_09_role = _transport_stats(_unp_cell("ROLE_BARE", ("p92-09",)))
    p92_46_unit_tr_ok = bool(tr_p92_46_role["mean_spec"] >= 0.50 and tr_p92_46_role["mean_raw"] >= 0.50)
    p92_09_unit_tr_ok = bool(tr_p92_09_role["mean_spec"] >= 0.50 and tr_p92_09_role["mean_raw"] >= 0.50)

    l7_nulls_role = {h: _head_null_stats(_hm_cell(h, "ROLE_BARE", "z_all")) for h in ("L7H5", "L7H4", "L7H9")}
    cm1a_valid = bool(
        cm0_passed
        and tr_by_cell["ROLE_BARE"]["cell_transport_pass"]
        and p92_46_unit_tr_ok
        and p92_09_unit_tr_ok
        and all(v["head_null_pass"] for v in l7_nulls_role.values())
    )

    l7_46_raw = _s_odd_unit_role("L7H5", "p92-46", "z_all", "raw")
    l7_46_spec = _s_odd_unit_role("L7H5", "p92-46", "z_all", "spec")
    l7_46_ctrl_raw = max(_s_odd_unit_role(h, "p92-46", "z_all", "raw")["S_odd"] for h in L7_CTRL_HEADS)
    l7_46_ctrl_spec = max(_s_odd_unit_role(h, "p92-46", "z_all", "spec")["S_odd"] for h in L7_CTRL_HEADS)
    l7_l3_raw = _l3_mean_s_odd_role("L7H5", "z_all", "raw")
    l7_l3_spec = _l3_mean_s_odd_role("L7H5", "z_all", "spec")

    l7h5_p92_46_confirmed = bool(
        cm1a_valid
        and l7_46_raw["d_pos"] <= -0.25
        and l7_46_raw["d_neg"] >= 0.25
        and l7_46_spec["d_pos"] <= -0.25
        and l7_46_spec["d_neg"] >= 0.25
        and (l7_46_raw["S_odd"] - l7_46_ctrl_raw) >= 0.15
        and (l7_46_spec["S_odd"] - l7_46_ctrl_spec) >= 0.15
        and (l7_46_raw["S_odd"] - l7_l3_raw) >= 0.20
    )

    if not cm0_passed:
        cm1a_verdict = CM0_FAIL_SCI
    elif not cm1a_valid:
        cm1a_verdict = "uninformative (cm1a_gate_failed)"
    elif not l7h5_p92_46_confirmed:
        cm1a_verdict = "dropped_confirmatory_failed"
    else:
        cm1a_verdict = "confirmed"

    # Separately gated p92-09 SpecLogit-only track (STRICTLY downstream of CM.1a == confirmed per #121!).
    l7_09_spec = _s_odd_unit_role("L7H5", "p92-09", "z_all", "spec")
    l7_09_raw = _s_odd_unit_role("L7H5", "p92-09", "z_all", "raw")
    l7_09_ctrl_spec = max(_s_odd_unit_role(h, "p92-09", "z_all", "spec")["S_odd"] for h in L7_CTRL_HEADS)
    p92_09_own_gate = bool(
        l7_09_spec["d_pos"] <= -0.50
        and l7_09_spec["d_neg"] >= 0.25
        and (l7_09_spec["S_odd"] - l7_09_ctrl_spec) >= 0.25
        and (l7_09_spec["S_odd"] - l7_l3_spec) >= 0.25
    )
    if cm1a_verdict != "confirmed":
        # Exact #121 drop/stop rule: primary drop hard-gates BOTH p92-46 and p92-09!
        cm1a_p92_09_verdict = cm1a_verdict
    elif handoff["p92_09_part_b_track"] != "include_spec_only_track" or not p92_09_own_gate:
        cm1a_p92_09_verdict = "dropped_or_not_included"
    else:
        cm1a_p92_09_verdict = "confirmed"

    # =========================================================================
    # CM.1b (L8H15 Confirmatory Gate)
    # =========================================================================
    l8_nulls = {
        h: {c: _head_null_stats(_hm_cell(h, c, "z_all")) for c in rca.BANK_CELLS}
        for h in ("L8H15", "L8H2", "L8H3", "L8H13")
    }
    cm1b_valid = bool(
        cm0_passed
        and all(tr_by_cell[c]["cell_transport_pass"] for c in rca.BANK_CELLS)
        and all(l8_nulls[h][c]["head_null_pass"] for h in l8_nulls for c in rca.BANK_CELLS)
    )
    l8_z_spec = {h: _l8_metrics(h, "z_all", "spec") for h in ("L8H15", "L8H2", "L8H3", "L8H13")}
    l8_z_raw = {h: _l8_metrics(h, "z_all", "raw") for h in ("L8H15", "L8H2", "L8H3", "L8H13")}

    ctrl_csn_spec = max(l8_z_spec[h]["C_SN"] for h in L8_CTRL_HEADS)
    ctrl_csr_spec = max(l8_z_spec[h]["C_SR"] for h in L8_CTRL_HEADS)
    l8h15_confirmed = bool(
        cm1b_valid
        and l8_z_spec["L8H15"]["C_SN"] >= 0.20
        and l8_z_raw["L8H15"]["C_SN"] >= 0.25
        and l8_z_spec["L8H15"]["C_SR"] >= 0.20
        and l8_z_raw["L8H15"]["C_SR"] >= 0.25
        and (l8_z_spec["L8H15"]["C_SN"] - ctrl_csn_spec) >= 0.15
        and (l8_z_spec["L8H15"]["C_SR"] - ctrl_csr_spec) >= 0.15
    )

    if not cm0_passed:
        cm1b_verdict = CM0_FAIL_SCI
    elif not cm1b_valid:
        cm1b_verdict = "uninformative (cm1b_gate_failed)"
    elif not l8h15_confirmed:
        cm1b_verdict = "dropped_confirmatory_failed"
    else:
        cm1b_verdict = "confirmed"

    cm1_section = {
        "transport_by_cell": tr_by_cell,
        "CM1a_L7H5": {
            "verdict": cm1a_verdict,
            "cm1a_valid": cm1a_valid,
            "l7h5_p92_46_confirmed": l7h5_p92_46_confirmed,
            "p92_46_unit_transport": tr_p92_46_role,
            "p92_09_unit_transport": tr_p92_09_role,
            "l7_head_nulls_ROLE_BARE": l7_nulls_role,
            "L7H5_p92_46_raw": l7_46_raw,
            "L7H5_p92_46_spec": l7_46_spec,
            "margins": {
                "p92_46_raw_over_ctrl": l7_46_raw["S_odd"] - l7_46_ctrl_raw,
                "p92_46_spec_over_ctrl": l7_46_spec["S_odd"] - l7_46_ctrl_spec,
                "p92_46_raw_over_L3": l7_46_raw["S_odd"] - l7_l3_raw,
                "L7H5_L3_mean_raw": l7_l3_raw,
                "L7H5_L3_mean_spec": l7_l3_spec,
            },
        },
        "CM1a_p92_09": {
            "verdict": cm1a_p92_09_verdict,
            "part_a_track": handoff["p92_09_part_b_track"],
            "p92_09_own_spec_gate_met": p92_09_own_gate,
            "L7H5_p92_09_spec": l7_09_spec,
            "L7H5_p92_09_raw_diagnostic": l7_09_raw,
            "margins": {
                "p92_09_spec_over_ctrl": l7_09_spec["S_odd"] - l7_09_ctrl_spec,
                "p92_09_spec_over_L3": l7_09_spec["S_odd"] - l7_l3_spec,
            },
        },
        "CM1b_L8H15": {
            "verdict": cm1b_verdict,
            "cm1b_valid": cm1b_valid,
            "l8h15_confirmed": l8h15_confirmed,
            "l8_head_nulls": l8_nulls,
            "heads_spec": l8_z_spec,
            "heads_raw": l8_z_raw,
            "margins": {
                "C_SN_spec_over_ctrl": l8_z_spec["L8H15"]["C_SN"] - ctrl_csn_spec,
                "C_SR_spec_over_ctrl": l8_z_spec["L8H15"]["C_SR"] - ctrl_csr_spec,
            },
        },
    }

    # =========================================================================
    # CM.2 (QK x V 2x2 Factorization) & CM.3 (Position-Scope Interaction)
    # =========================================================================
    def _eval_l7_cm2(unit: str, ch: str, gate: str, ctrl_thresh: float) -> dict[str, Any]:
        d_tot = _s_odd_unit_role("L7H5", unit, "QKV_all", ch)["S_odd"]
        d_v = _s_odd_unit_role("L7H5", unit, "V_only", ch)["S_odd"]
        d_qk = _s_odd_unit_role("L7H5", unit, "QK_only", ch)["S_odd"]
        c_v = max(_s_odd_unit_role(h, unit, "V_only", ch)["S_odd"] for h in L7_CTRL_HEADS)
        c_qk = max(_s_odd_unit_role(h, unit, "QK_only", ch)["S_odd"] for h in L7_CTRL_HEADS)
        res = classify_cm2_factorization(gate, d_tot, d_v, d_qk, c_v, c_qk, ctrl_thresh)
        res["diagnostics_Q_K_only"] = {
            "Q_only": _s_odd_unit_role("L7H5", unit, "Q_only", ch)["S_odd"],
            "K_only": _s_odd_unit_role("L7H5", unit, "K_only", ch)["S_odd"],
        }
        res["controls"] = {
            h: {m: _s_odd_unit_role(h, unit, m, ch)["S_odd"] for m in ("QKV_all", "QK_only", "V_only")}
            for h in L7_CTRL_HEADS
        }
        return res

    def _eval_l7_cm3(unit: str, ch: str, gate: str, ctrl_thresh: float) -> dict[str, Any]:
        d_all = _s_odd_unit_role("L7H5", unit, "z_all", ch)["S_odd"]
        d_rp = _s_odd_unit_role("L7H5", unit, "z_read_pos", ch)["S_odd"]
        d_up = _s_odd_unit_role("L7H5", unit, "z_upstream", ch)["S_odd"]
        c_rp = max(_s_odd_unit_role(h, unit, "z_read_pos", ch)["S_odd"] for h in L7_CTRL_HEADS)
        c_up = max(_s_odd_unit_role(h, unit, "z_upstream", ch)["S_odd"] for h in L7_CTRL_HEADS)
        res = classify_cm3_position_scope(gate, d_all, d_rp, d_up, c_rp, c_up, ctrl_thresh)
        res["controls"] = {
            h: {m: _s_odd_unit_role(h, unit, m, ch)["S_odd"] for m in ("z_all", "z_read_pos", "z_upstream")}
            for h in L7_CTRL_HEADS
        }
        return res

    l8_targets = ("NAME_BARE_suppression", "ROLE_BARE_suppression", "C_SN", "C_SR")

    def _eval_l8_cm2(head: str, target_key: str, ch: str, gate: str) -> dict[str, Any]:
        d_tot = _l8_metrics(head, "QKV_all", ch)[target_key]
        d_v = _l8_metrics(head, "V_only", ch)[target_key]
        d_qk = _l8_metrics(head, "QK_only", ch)[target_key]
        c_v = max(_l8_metrics(h, "V_only", ch)[target_key] for h in L8_CTRL_HEADS)
        c_qk = max(_l8_metrics(h, "QK_only", ch)[target_key] for h in L8_CTRL_HEADS)
        res = classify_cm2_factorization(gate, d_tot, d_v, d_qk, c_v, c_qk, 0.15)
        if head in rca.DIAG_HEADS_FULL:
            res["diagnostics_Q_K_only"] = {
                "Q_only": _l8_metrics(head, "Q_only", ch)[target_key],
                "K_only": _l8_metrics(head, "K_only", ch)[target_key],
            }
        return res

    def _eval_l8_cm3(head: str, target_key: str, ch: str, gate: str) -> dict[str, Any]:
        d_all = _l8_metrics(head, "z_all", ch)[target_key]
        d_rp = _l8_metrics(head, "z_read_pos", ch)[target_key]
        d_up = _l8_metrics(head, "z_upstream", ch)[target_key]
        c_rp = max(_l8_metrics(h, "z_read_pos", ch)[target_key] for h in L8_CTRL_HEADS)
        c_up = max(_l8_metrics(h, "z_upstream", ch)[target_key] for h in L8_CTRL_HEADS)
        return classify_cm3_position_scope(gate, d_all, d_rp, d_up, c_rp, c_up, 0.15)

    l8_comp_gate = "confirmed" if cm1b_valid else (CM0_FAIL_SCI if not cm0_passed else "uninformative (cm1b_gate_failed)")
    cm2_section = {
        "L7H5_p92_46_raw": _eval_l7_cm2("p92-46", "raw", cm1a_verdict, 0.15),
        "L7H5_p92_46_spec": _eval_l7_cm2("p92-46", "spec", cm1a_verdict, 0.15),
        "L7H5_p92_09_spec": _eval_l7_cm2("p92-09", "spec", cm1a_p92_09_verdict, 0.25),
        "L7H5_L3_descriptive": {
            ch: {m: _l3_mean_s_odd_role("L7H5", m, ch) for m in ("QKV_all", "QK_only", "V_only", "Q_only", "K_only")}
            for ch in ("raw", "spec")
        },
        "L8H15": {
            ch: {tk: _eval_l8_cm2("L8H15", tk, ch, cm1b_verdict) for tk in l8_targets}
            for ch in ("spec", "raw")
        },
        "L8H2_active_comparator": {
            ch: {tk: _eval_l8_cm2("L8H2", tk, ch, l8_comp_gate) for tk in l8_targets}
            for ch in ("spec", "raw")
        },
        "L8_controls_descriptive": {
            h: {ch: {m: _l8_metrics(h, m, ch) for m in ("QKV_all", "QK_only", "V_only")} for ch in ("spec", "raw")}
            for h in L8_CTRL_HEADS
        },
    }

    cm3_section = {
        "L7H5_p92_46_raw": _eval_l7_cm3("p92-46", "raw", cm1a_verdict, 0.15),
        "L7H5_p92_46_spec": _eval_l7_cm3("p92-46", "spec", cm1a_verdict, 0.15),
        "L7H5_p92_09_spec": _eval_l7_cm3("p92-09", "spec", cm1a_p92_09_verdict, 0.25),
        "L7H5_L3_descriptive": {
            ch: {m: _l3_mean_s_odd_role("L7H5", m, ch) for m in ("z_all", "z_read_pos", "z_upstream")}
            for ch in ("raw", "spec")
        },
        "L8H15": {
            ch: {tk: _eval_l8_cm3("L8H15", tk, ch, cm1b_verdict) for tk in l8_targets}
            for ch in ("spec", "raw")
        },
        "L8H2_active_comparator": {
            ch: {tk: _eval_l8_cm3("L8H2", tk, ch, l8_comp_gate) for tk in l8_targets}
            for ch in ("spec", "raw")
        },
        "L8_controls_descriptive": {
            h: {ch: {m: _l8_metrics(h, m, ch) for m in ("z_all", "z_read_pos", "z_upstream")} for ch in ("spec", "raw")}
            for h in L8_CTRL_HEADS
        },
    }

    # =========================================================================
    # CM.4 (Reciprocal Coordinate-Localized Contextual-State Swaps)
    # =========================================================================
    def _swap_stratum_mean(head: str, span_name: str, swap_cond: str, ch: str) -> float:
        fld = "delta_spec" if ch == "spec" else "delta_raw"
        vals = [
            float(r[fld])
            for r in sw_by_key.values()
            if r["head_name"] == head and r["span_name"] == span_name and r["swap_cond"] == swap_cond
        ]
        if len(vals) != 20:
            raise ValueError(f"Expected 20 rows for CM.4 stratum ({head}, {span_name}, {swap_cond}), got {len(vals)}")
        return _mean(vals)

    def _eval_cm4_pair(head: str, ref_span: str, nonref_span: str, ch: str, gate: str) -> dict[str, Any]:
        m_n2s = _swap_stratum_mean(head, ref_span, "N2S", ch)
        m_s2n = -_swap_stratum_mean(head, ref_span, "S2N", ch)
        # 6 strictly unpooled control strata (NEVER averaged across target conditions before abs!).
        ctrl_strata = {
            "cross_nonref_N2S": _swap_stratum_mean(head, nonref_span, "N2S", ch),
            "cross_nonref_S2N": _swap_stratum_mean(head, nonref_span, "S2N", ch),
            "same_cond_ref_S2S": _swap_stratum_mean(head, ref_span, "S2S", ch),
            "same_cond_ref_N2N": _swap_stratum_mean(head, ref_span, "N2N", ch),
            "same_cond_nonref_S2S": _swap_stratum_mean(head, nonref_span, "S2S", ch),
            "same_cond_nonref_N2N": _swap_stratum_mean(head, nonref_span, "N2N", ch),
        }
        c_ctrl = max(abs(v) for v in ctrl_strata.values())
        res = classify_cm4_swap(gate, m_n2s, m_s2n, c_ctrl)
        res["ref_span"] = ref_span
        res["nonref_span"] = nonref_span
        res["control_strata_means"] = ctrl_strata
        res["C_SN_context_only"] = (l8_z_spec[head]["C_SN"] if ch == "spec" else l8_z_raw[head]["C_SN"])
        return res

    def _eval_cm4c_asst(head: str, ch: str) -> dict[str, Any]:
        m_n2s = _swap_stratum_mean(head, "SPAN_REF_ASST", "N2S", ch)
        raw_s2n = _swap_stratum_mean(head, "SPAN_REF_ASST", "S2N", ch)
        return {
            "span": "SPAN_REF_ASST",
            "M_N2S": m_n2s,
            "M_S2N": -raw_s2n,
            "mean_delta_S2N": raw_s2n,
            "same_cond_asst_S2S": _swap_stratum_mean(head, "SPAN_REF_ASST", "S2S", ch),
            "same_cond_asst_N2N": _swap_stratum_mean(head, "SPAN_REF_ASST", "N2N", ch),
            "C_SN_context_only": (l8_z_spec[head]["C_SN"] if ch == "spec" else l8_z_raw[head]["C_SN"]),
        }

    cm4_heads_gate = {
        "L8H15": cm1b_verdict,
        "L8H2": l8_comp_gate,
        "L8H3": l8_comp_gate,
    }
    cm4_section = {
        "CM4a_full_4tok": {
            h: {ch: _eval_cm4_pair(h, "SPAN_REF", "SPAN_NONREF", ch, cm4_heads_gate[h]) for ch in ("spec", "raw")}
            for h in rca.CM4_HEADS_FULL
        },
        "CM4b_user_2tok": {
            h: {ch: _eval_cm4_pair(h, "SPAN_REF_USER", "SPAN_NONREF_USER", ch, cm4_heads_gate[h]) for ch in ("spec", "raw")}
            for h in rca.CM4_HEADS_FULL
        },
        "CM4c_asst_2tok_diagnostic": {
            h: {ch: _eval_cm4c_asst(h, ch) for ch in ("spec", "raw")}
            for h in rca.CM4_HEADS_FULL
        },
    }

    return {
        "meta": {
            "schema_version": rca.SCHEMA_VERSION,
            "artifact_type": "causal_attn_scorecard",
            "execution_provenance": receipt["meta"]["execution_provenance"],
            "operating_norm_name": handoff["operating_norm_name"],
            "operating_norm_multipliers": handoff["operating_norm_multipliers"],
            "part_a_part_b_norm_decision": handoff["part_b_norm_decision"],
            "part_a_p92_09_part_b_track": handoff["p92_09_part_b_track"],
            "pins": dict(pins),
            "external_pins_verified": True,
            "complete_join_verified": True,
        },
        "CM0_technical_gate": cm0,
        "CM1_confirmatory_gates": cm1_section,
        "CM2_qkv_factorization": cm2_section,
        "CM3_position_scope": cm3_section,
        "CM4_span_swaps": cm4_section,
    }


# ---------------------------------------------------------------------------
# Exhaustive Scorer CPU Self-Test Suite (Every Gate, Branch & Alden #121/#124 Fixture)
# ---------------------------------------------------------------------------

def _make_synthetic_full_receipt(
    manifest: dict[str, Any],
    pins: dict[str, str],
    *,
    p92_46_l7h5_pass: bool = True,
    p92_09_l7h5_huge: bool = True,
    l8h15_pass: bool = True,
    cm0_break_sibling: bool = False,
    cm0_break_reconcile: bool = False,
    cm0_break_actuation: bool = False,
    cm4_n2s_val: float = 0.45,
    cm4_s2n_raw_delta: float = -0.42,
    cm4_ctrl_val: float = 0.02,
    l7_v_frac: float = 0.90,
    l7_qk_frac: float = 0.05,
) -> dict[str, Any]:
    """Build a schema-exact synthetic full Part B receipt (60/2880/17/1200) for scorer fixture testing."""
    sched = manifest["schedule_full"]
    mmeta = manifest["meta"]

    def _mk_metric(dp: float, dn: float, d0: float = 0.0, null_val: float = 0.01) -> dict[str, Any]:
        ev = 0.5 * (dp + dn) - d0
        od = 0.5 * (dp - dn)
        nd = [null_val] * rca.N_NULL_DRAWS
        nn = [-null_val] * rca.N_NULL_DRAWS
        ev_n = [0.0] * rca.N_NULL_DRAWS
        od_n = [null_val] * rca.N_NULL_DRAWS
        nblk = {
            "d_pos_draws": nd,
            "d_neg_draws": nn,
            "even_draws": ev_n,
            "odd_draws": od_n,
            "d_pos_med": null_val,
            "d_neg_med": -null_val,
            "even_med": 0.0,
            "odd_med": null_val,
            "even_max_abs": 0.0,
            "odd_max_abs": abs(null_val),
        }
        return {
            "d0": d0,
            "d_pos": dp,
            "d_neg": dn,
            "even": ev,
            "odd": od,
            "max_abs_sham": 0.0,
            "wdown_orth": copy.deepcopy(nblk),
            "iso_resid": copy.deepcopy(nblk),
        }

    unp_rows = []
    for u in sched["units"]:
        ul = rca.UNIT_BY_ID[u]["layer"]
        m_L = float(mmeta["operating_norm_multipliers"][str(ul)])
        for p in sched["cm123_prompts"]:
            unp_rows.append({
                "unit": u,
                "unit_layer": ul,
                "prompt_id": p["prompt_id"],
                "story": p["story"],
                "cell": p["cell"],
                "dose_mag": 0.8,
                "operating_norm_name": mmeta["operating_norm_name"],
                "norm_multiplier": m_L,
                "injected_norm": 0.8 * m_L,
                "read_pos_resid_norm": 10.0,
                "norm_ratio": 0.8 * m_L / 10.0,
                "null_seed": 123,
                "max_abs_decomp_err": 0.0,
                "transport": {
                    "n_write_spec": 2.5,
                    "n_write_raw": 2.0,
                    "n_write_peer_med": -0.5,
                    "n_write_neg_spec": -2.5,
                    "n_write_neg_raw": -2.0,
                    "n_write_neg_peer_med": 0.5,
                    "wdown_pos_draws_spec": [0.1] * 4,
                    "wdown_pos_draws_raw": [0.05] * 4,
                    "wdown_pos_draws_peer_med": [-0.05] * 4,
                    "iso_pos_draws_spec": [0.05] * 4,
                    "iso_pos_draws_raw": [0.02] * 4,
                    "iso_pos_draws_peer_med": [-0.03] * 4,
                },
            })

    hm_rows = []
    for u in sched["units"]:
        ul = rca.UNIT_BY_ID[u]["layer"]
        for p in sched["cm123_prompts"]:
            cell = p["cell"]
            for hm in sched["cm123_head_modes"]:
                hname, mode = hm["head_name"], hm["mode"]
                hmeta = rca.HEAD_BY_NAME[hname]
                dp_s, dn_s, dp_r, dn_r = 0.0, 0.0, 0.0, 0.0

                # Program L7H5 late-write behavior on ROLE_BARE.
                if hname == "L7H5" and cell == "ROLE_BARE":
                    if u == "p92-46":
                        base_s = 0.60 if p92_46_l7h5_pass else 0.05
                        scale = {
                            "z_all": 1.0,
                            "QKV_all": 1.0,
                            "V_only": l7_v_frac,
                            "QK_only": l7_qk_frac,
                            "z_read_pos": 0.88,
                            "z_upstream": 0.08,
                            "Q_only": 0.02,
                            "K_only": 0.02,
                        }[mode]
                        dp_s = -base_s * scale
                        dn_s = base_s * scale
                        dp_r, dn_r = dp_s, dn_s
                    elif u == "p92-09":
                        base_s = 1.40 if p92_09_l7h5_huge else 0.05
                        scale = {
                            "z_all": 1.0,
                            "QKV_all": 1.0,
                            "V_only": 0.90,
                            "QK_only": 0.05,
                            "z_read_pos": 0.90,
                            "z_upstream": 0.05,
                            "Q_only": 0.02,
                            "K_only": 0.02,
                        }[mode]
                        dp_s = -base_s * scale
                        dn_s = 0.70 * scale if p92_09_l7h5_huge else 0.05
                        dp_r, dn_r = -1.0, -0.8

                # Program L8H15 third-person suppression behavior.
                if hname == "L8H15":
                    scale = {
                        "z_all": 1.0,
                        "QKV_all": 1.0,
                        "V_only": 0.10,
                        "QK_only": 0.85,
                        "z_read_pos": 0.10,
                        "z_upstream": 0.85,
                        "Q_only": 0.40,
                        "K_only": 0.40,
                    }[mode]
                    if l8h15_pass:
                        if cell == "SELF":
                            dp_s, dp_r = 0.10 * scale, 0.15 * scale
                        else:
                            dp_s, dp_r = -0.45 * scale, -0.40 * scale
                    else:
                        dp_s, dp_r = 0.0, 0.0

                spec_m = _mk_metric(dp_s, dn_s)
                raw_m = _mk_metric(dp_r, dn_r)
                pmed_m = _mk_metric(dp_r - dp_s, dn_r - dn_s, null_val=0.0)
                sib_val = 1e-3 if (cm0_break_sibling and hname == "L8H15" and mode == "QK_only" and u == "p92-46" and p["prompt_id"] == "B1_SELF") else 0.0

                row = {
                    "unit": u,
                    "unit_layer": ul,
                    "prompt_id": p["prompt_id"],
                    "story": p["story"],
                    "cell": cell,
                    "dose_mag": 0.8,
                    "head_name": hname,
                    "head_layer": hmeta["layer"],
                    "head_idx": hmeta["head"],
                    "mode": mode,
                    "sibling_max_abs_leakage": sib_val,
                    "max_abs_decomp_err": 0.0,
                    "spec": spec_m,
                    "raw": raw_m,
                    "peer_med": pmed_m,
                }
                if mode == "QKV_all":
                    rec_err = 1e-2 if (cm0_break_reconcile and hname == "L7H5" and u == "p92-46" and p["prompt_id"] == "B1_ROLE_BARE") else 0.0
                    row["reconciliation_qkv_vs_z_all"] = {
                        "pre_oproj_head_tensor_max_abs": rec_err,
                        "full_vocab_read_pos_logits_max_abs": rec_err,
                        "readout_drop_max_abs": 0.0,
                    }
                hm_rows.append(row)

    act_rows = []
    for idx, a in enumerate(sched["actuation"]):
        fail_this = cm0_break_actuation and idx == 0
        td = 1e-5 if fail_this else 0.85
        act_rows.append({
            "actuation_id": a["actuation_id"],
            "kind": a["kind"],
            "unit": a["unit"],
            "prompt_id": a["prompt_id"],
            "head_name": a["head_name"],
            "head_layer": rca.HEAD_BY_NAME[a["head_name"]]["layer"],
            "head_idx": rca.HEAD_BY_NAME[a["head_name"]]["head"],
            "mode": a["mode"],
            "swap_indices": a["swap_indices"],
            "seed": a["seed"],
            "target_max_abs_delta": td,
            "sibling_max_abs_leakage": 0.0,
            "passed": not fail_this,
        })

    sw_rows = []
    for tup in sched["cm4_tuples"]:
        uid = tup["unit"]
        t_pid = tup["target_prompt_id"]
        d_pid = tup["donor_prompt_id"]
        t_bind = manifest["bindings"][t_pid][uid]
        d_bind = manifest["bindings"][d_pid][uid]
        for hname in sched["cm4_heads"]:
            hmeta = rca.HEAD_BY_NAME[hname]
            if hname == "L8H15" and tup["span_name"] in ("SPAN_REF", "SPAN_REF_USER"):
                if tup["swap_cond"] == "N2S":
                    ds = cm4_n2s_val
                elif tup["swap_cond"] == "S2N":
                    ds = cm4_s2n_raw_delta
                else:
                    ds = cm4_ctrl_val
            else:
                ds = cm4_ctrl_val if hname == "L8H15" else 0.01
            sw_rows.append({
                "unit": uid,
                "target_story": tup["target_story"],
                "target_cell": tup["target_cell"],
                "target_prompt_id": t_pid,
                "donor_story": tup["donor_story"],
                "donor_cell": tup["donor_cell"],
                "donor_prompt_id": d_pid,
                "swap_cond": tup["swap_cond"],
                "span_name": tup["span_name"],
                "head_name": hname,
                "head_layer": hmeta["layer"],
                "head_idx": hmeta["head"],
                "target_input_ids": list(t_bind["input_ids"]),
                "donor_input_ids": list(d_bind["input_ids"]),
                "swap_indices": list(tup["swap_indices"]),
                "position_ids": list(range(33)),
                "delta_spec": ds,
                "delta_raw": ds,
                "delta_peer_med": 0.0,
                "sham_spec": 0.0,
                "sham_raw": 0.0,
                "sham_peer_med": 0.0,
                "sibling_max_abs_leakage": 0.0,
                "max_abs_decomp_err": 0.0,
            })

    return {
        "meta": {
            "schema_version": rca.SCHEMA_VERSION,
            "artifact_type": "causal_attn_full_receipt",
            "run_mode": "full",
            "execution_provenance": "synthetic_scorer_fixture",
            "device_name": "cpu",
            "complete": True,
            "smoke_1unit": False,
            "model_repo": rca.MODEL_REPO,
            "model_commit": rca.MODEL_COMMIT,
            "dtype": rca.DTYPE_STR,
            "attn_implementation": rca.ATTN_IMPL,
            "operating_norm_name": mmeta["operating_norm_name"],
            "operating_norm_multipliers": dict(mmeta["operating_norm_multipliers"]),
            "part_a_part_b_norm_decision": mmeta["part_a_part_b_norm_decision"],
            "part_a_p92_09_part_b_track": mmeta["part_a_p92_09_part_b_track"],
            "pins": dict(pins),
            "units_run": list(rca.CORE_UNIT_IDS),
            "n_unpatched_rows": 60,
            "n_head_mode_rows": 2880,
            "n_actuation_rows": 17,
            "n_swap_rows": 1200,
            "n_b19_forwards": 5820,
            "n_b1_forwards": 2457,
            "n_total_forwards": 8277,
            "max_abs_sham_spec": 0.0,
            "max_abs_sham_raw": 0.0,
            "max_abs_decomp_err": 0.0,
            "max_sibling_leakage": 0.0,
            "max_reconcile_pre_oproj": 0.0,
            "max_reconcile_full_vocab": 0.0,
            "max_reconcile_drop": 0.0,
            "timing": {
                "model_load_sec": 1.0,
                "b19_elapsed_sec": 58.2,
                "b1_elapsed_sec": 24.57,
                "non_fwd_runner_sec": 2.0,
                "total_runner_sec": 85.77,
                "n_b19_done": 5820,
                "n_b1_done": 2457,
                "ms_per_b19": 10.0,
                "ms_per_b1": 10.0,
                "projected_full_b19_sec": 58.2,
                "projected_full_b1_sec": 24.57,
                "projected_full_fwd_sec": 82.77,
            },
        },
        "unit_vectors": {},
        "unpatched_rows": unp_rows,
        "head_mode_rows": hm_rows,
        "actuation_rows": act_rows,
        "swap_rows": sw_rows,
    }


def run_scorer_self_tests(
    design_path: Path,
    manifest_path: Path,
    runner_path: Path,
    scorer_path: Path,
    part_a_scorecard_path: Path,
) -> dict[str, Any]:
    checks: list[str] = []

    d_sha = rca.sha256_file(design_path)
    m_sha = rca.sha256_file(manifest_path)
    r_sha = rca.sha256_file(runner_path)
    s_sha = rca.sha256_file(scorer_path)
    pa_sha = rca.sha256_file(part_a_scorecard_path)

    manifest, part_a_sc, pins, _ = rca._admit_frozen_package_with_bytes(
        design_path=design_path,
        manifest_path=manifest_path,
        runner_path=runner_path,
        scorer_path=scorer_path,
        part_a_scorecard_path=part_a_scorecard_path,
        expected_design_sha=d_sha,
        expected_manifest_sha=m_sha,
        expected_runner_sha=r_sha,
        expected_scorer_sha=s_sha,
        expected_part_a_scorecard_sha=pa_sha,
    )

    # 1. Exhaustive CM.2 7-branch + boundary (.75, .25, 0) + control_selectivity checks.
    assert classify_cm2_factorization("dropped_confirmatory_failed", 1.0, 0.9, 0.05)["verdict"] == "dropped_confirmatory_failed"
    assert classify_cm2_factorization("confirmed", 0.20, 0.18, 0.02)["verdict"] == "effect_too_small_or_wrong_sign"
    r_v = classify_cm2_factorization("confirmed", 1.0, 0.80, 0.10, ctrl_v_max=0.20, ctrl_qk_max=0.10, ctrl_margin_thresh=0.15)
    assert r_v["verdict"] == "v_dominant_within_head" and r_v["control_selectivity"] == "selective_over_matched_controls"
    r_v_nc = classify_cm2_factorization("confirmed", 1.0, 0.80, 0.10, ctrl_v_max=0.70, ctrl_qk_max=0.10, ctrl_margin_thresh=0.15)
    assert r_v_nc["verdict"] == "v_dominant_within_head" and r_v_nc["control_selectivity"] == "within_head_only_not_control_selective"
    r_qk = classify_cm2_factorization("confirmed", 1.0, 0.10, 0.80, ctrl_v_max=0.10, ctrl_qk_max=0.20, ctrl_margin_thresh=0.15)
    assert r_qk["verdict"] == "qk_dominant_within_head" and r_qk["control_selectivity"] == "selective_over_matched_controls"
    assert classify_cm2_factorization("confirmed", 1.0, 0.30, 0.30)["verdict"] == "interaction_required_within_head"
    # Exact disjointness boundary: (.75, .25, 0) MUST map uniquely to additive_both_within_head!
    assert classify_cm2_factorization("confirmed", 1.0, 0.75, 0.25)["verdict"] == "additive_both_within_head"
    assert classify_cm2_factorization("confirmed", 1.0, 0.60, 0.20)["verdict"] == "mixed_or_inconclusive_within_head"
    checks.append("cm2_all_7_branches_boundary_and_control_selectivity_ok")

    # 2. Exhaustive CM.3 7-branch + boundary (.75, .25, 0) + control_selectivity checks.
    assert classify_cm3_position_scope("uninformative (cm1a_gate_failed)", 1.0, 0.9, 0.05)["verdict"] == "uninformative (cm1a_gate_failed)"
    assert classify_cm3_position_scope("confirmed", 0.10, 0.09, 0.01)["verdict"] == "effect_too_small_or_wrong_sign"
    assert classify_cm3_position_scope("confirmed", 1.0, 0.85, 0.05, 0.20, 0.05)["verdict"] == "read_pos_dominant_within_head"
    assert classify_cm3_position_scope("confirmed", 1.0, 0.05, 0.85, 0.05, 0.20)["verdict"] == "upstream_dominant_within_head"
    assert classify_cm3_position_scope("confirmed", 1.0, 0.30, 0.30)["verdict"] == "interaction_required_within_head"
    assert classify_cm3_position_scope("confirmed", 1.0, 0.75, 0.25)["verdict"] == "additive_both_within_head"
    assert classify_cm3_position_scope("confirmed", 1.0, 0.60, 0.20)["verdict"] == "mixed_or_inconclusive_within_head"
    checks.append("cm3_all_7_branches_and_boundary_ok")

    # 3. Exhaustive CM.4 6-state checks INCLUDING Alden #124 one-positive / one-reverse fixture!
    assert classify_cm4_swap("dropped_confirmatory_failed", 0.5, 0.5, 0.01)["verdict"] == "dropped_confirmatory_failed"
    assert classify_cm4_swap("confirmed", 0.30, 0.25, 0.05)["verdict"] == "reciprocal_tested_coords_localized"
    assert classify_cm4_swap("confirmed", 0.30, 0.02, 0.05)["verdict"] == "asymmetric_tested_coords_effect"
    # Alden #124 explicit fixture: one positive (+0.40) + one reverse (-0.35) -> branch 3 wins AND both signed M_N2S/M_S2N emitted!
    r_asym_rev = classify_cm4_swap("confirmed", 0.40, -0.35, 0.02)
    assert r_asym_rev["verdict"] == "asymmetric_tested_coords_effect"
    assert r_asym_rev["M_N2S"] == 0.40 and r_asym_rev["M_S2N"] == -0.35 and r_asym_rev["mean_delta_S2N"] == 0.35
    # Pure reverse direction -> branch 4!
    r_rev = classify_cm4_swap("confirmed", -0.30, -0.05, 0.02)
    assert r_rev["verdict"] == "opposite_direction_at_tested_coords" and r_rev["M_N2S"] == -0.30
    # Magnitude-bounded small effect -> branch 5!
    assert classify_cm4_swap("confirmed", 0.04, -0.03, 0.05)["verdict"] == "small_effect_at_tested_coords"
    # Control-confounded / inconclusive -> branch 6!
    assert classify_cm4_swap("confirmed", 0.20, 0.20, 0.18)["verdict"] == "inconclusive_or_control_confounded"
    checks.append("cm4_all_6_branches_plus_alden_124_signed_asymmetric_reverse_ok")

    # 4. Full receipt fixture: happy path (all gates pass).
    rec_ok = _make_synthetic_full_receipt(manifest, pins)
    sc_ok = score_full_receipt(rec_ok, manifest, pins, part_a_sc, allow_fixture=True)
    assert sc_ok["CM0_technical_gate"]["passed"] is True
    assert sc_ok["CM1_confirmatory_gates"]["CM1a_L7H5"]["verdict"] == "confirmed"
    assert sc_ok["CM1_confirmatory_gates"]["CM1a_p92_09"]["verdict"] == "confirmed"
    assert sc_ok["CM1_confirmatory_gates"]["CM1b_L8H15"]["verdict"] == "confirmed"
    assert sc_ok["CM2_qkv_factorization"]["L7H5_p92_46_raw"]["verdict"] == "v_dominant_within_head"
    assert sc_ok["CM2_qkv_factorization"]["L7H5_p92_46_raw"]["control_selectivity"] == "selective_over_matched_controls"
    assert sc_ok["CM2_qkv_factorization"]["L8H15"]["spec"]["C_SN"]["verdict"] == "qk_dominant_within_head"
    assert sc_ok["CM3_position_scope"]["L7H5_p92_46_raw"]["verdict"] == "read_pos_dominant_within_head"
    assert sc_ok["CM3_position_scope"]["L8H15"]["spec"]["C_SN"]["verdict"] == "upstream_dominant_within_head"
    assert sc_ok["CM4_span_swaps"]["CM4a_full_4tok"]["L8H15"]["spec"]["verdict"] == "reciprocal_tested_coords_localized"
    assert sc_ok["CM4_span_swaps"]["CM4b_user_2tok"]["L8H15"]["spec"]["verdict"] == "reciprocal_tested_coords_localized"
    checks.append("full_fixture_happy_path_all_endpoints_ok")

    # 5. Alden #121 ALL-L7H5 drop/stop rule fixture: primary p92-46 fails EVEN THOUGH p92-09 spec is huge ->
    # BOTH p92-46 AND p92-09 CM.2/CM.3 MUST be hard-gated `dropped_confirmatory_failed`!
    rec_drop = _make_synthetic_full_receipt(manifest, pins, p92_46_l7h5_pass=False, p92_09_l7h5_huge=True)
    sc_drop = score_full_receipt(rec_drop, manifest, pins, part_a_sc, allow_fixture=True)
    assert sc_drop["CM1_confirmatory_gates"]["CM1a_L7H5"]["verdict"] == "dropped_confirmatory_failed"
    assert sc_drop["CM1_confirmatory_gates"]["CM1a_p92_09"]["verdict"] == "dropped_confirmatory_failed"
    assert sc_drop["CM2_qkv_factorization"]["L7H5_p92_46_raw"]["verdict"] == "dropped_confirmatory_failed"
    assert sc_drop["CM2_qkv_factorization"]["L7H5_p92_46_spec"]["verdict"] == "dropped_confirmatory_failed"
    assert sc_drop["CM2_qkv_factorization"]["L7H5_p92_09_spec"]["verdict"] == "dropped_confirmatory_failed"
    assert sc_drop["CM3_position_scope"]["L7H5_p92_09_spec"]["verdict"] == "dropped_confirmatory_failed"
    # L8H15 remains independent and confirmed!
    assert sc_drop["CM1_confirmatory_gates"]["CM1b_L8H15"]["verdict"] == "confirmed"
    checks.append("alden_121_primary_l7h5_drop_hard_gates_p92_09_ok")

    # 6. Full fixture with Alden #124 one-positive / one-reverse CM.4 values end-to-end!
    rec_asym = _make_synthetic_full_receipt(manifest, pins, cm4_n2s_val=0.40, cm4_s2n_raw_delta=0.35, cm4_ctrl_val=0.02)
    sc_asym = score_full_receipt(rec_asym, manifest, pins, part_a_sc, allow_fixture=True)
    cm4a = sc_asym["CM4_span_swaps"]["CM4a_full_4tok"]["L8H15"]["spec"]
    assert cm4a["verdict"] == "asymmetric_tested_coords_effect"
    assert abs(cm4a["M_N2S"] - 0.40) < 1e-9 and abs(cm4a["M_S2N"] - (-0.35)) < 1e-9
    checks.append("full_fixture_cm4_asymmetric_reverse_signed_emission_ok")

    # 6b. Explicit unpooled control-strata anti-cancellation fixture:
    # cross_nonref_N2S = +0.25 and cross_nonref_S2N = -0.25 (pooled mean = 0.0!) -> C_ctrl MUST be 0.25!
    rec_unpool = _make_synthetic_full_receipt(manifest, pins, cm4_n2s_val=0.30, cm4_s2n_raw_delta=-0.30, cm4_ctrl_val=0.01)
    for r in rec_unpool["swap_rows"]:
        if r["head_name"] == "L8H15" and r["span_name"] == "SPAN_NONREF":
            if r["swap_cond"] == "N2S":
                r["delta_spec"] = 0.25
                r["delta_raw"] = 0.25
            elif r["swap_cond"] == "S2N":
                r["delta_spec"] = -0.25
                r["delta_raw"] = -0.25
    sc_unpool = score_full_receipt(rec_unpool, manifest, pins, part_a_sc, allow_fixture=True)
    cm4a_u = sc_unpool["CM4_span_swaps"]["CM4a_full_4tok"]["L8H15"]["spec"]
    assert abs(cm4a_u["C_ctrl"] - 0.25) < 1e-9, f"Unpooled C_ctrl failed: {cm4a_u['C_ctrl']}"
    assert cm4a_u["verdict"] == "inconclusive_or_control_confounded", f"Expected confounded, got {cm4a_u['verdict']}"
    checks.append("cm4_unpooled_control_strata_anti_cancellation_ok")

    # 6c. F3 control-margin threshold coverage between 0.15 and 0.25:
    # Set control margin = 0.20 on V_only & z_read_pos for BOTH p92-46 (thresh 0.15) and p92-09 (thresh 0.25)
    # while keeping z_all confirmed (>= 0.25 on p92-09). Mutating 0.25 -> 0.15 at either CM.2 or CM.3 breaks this test!
    rec_f3 = _make_synthetic_full_receipt(manifest, pins)
    for r in rec_f3["head_mode_rows"]:
        if r["cell"] != "ROLE_BARE":
            continue
        # For p92-46: L7H5 V_only/z_read_pos = 0.54 (0.90*0.60, z_read_pos 0.88*0.60=0.528); put L7H4 at S_odd=0.33 -> margin ~0.20 (>= 0.15).
        # For p92-09: L7H5 z_all S_odd = 1.05, V_only & z_read_pos S_odd = 0.945; put L7H4 at z_all=0.75 (z_all margin 0.30 >= 0.25 -> confirmed!),
        # and V_only & z_read_pos at 0.745 -> margin = 0.20 (< 0.25 -> NOT control-selective!).
        if r["head_name"] == "L7H4":
            if r["unit"] == "p92-46" and r["mode"] in ("z_all", "QKV_all", "V_only", "z_read_pos"):
                val = 0.33
                for ch in ("spec", "raw"):
                    r[ch]["d_pos"] = -val
                    r[ch]["d_neg"] = val
                    r[ch]["even"] = 0.0
                    r[ch]["odd"] = -val
                r["peer_med"]["d_pos"] = 0.0
                r["peer_med"]["d_neg"] = 0.0
                r["peer_med"]["even"] = 0.0
                r["peer_med"]["odd"] = 0.0
            elif r["unit"] == "p92-09" and r["mode"] in ("z_all", "QKV_all", "V_only", "z_read_pos"):
                val = 0.745
                r["spec"]["d_pos"] = -val
                r["spec"]["d_neg"] = val
                r["spec"]["even"] = 0.0
                r["spec"]["odd"] = -val
                r["raw"]["d_pos"] = -val
                r["raw"]["d_neg"] = val
                r["raw"]["even"] = 0.0
                r["raw"]["odd"] = -val
                r["peer_med"]["d_pos"] = 0.0
                r["peer_med"]["d_neg"] = 0.0
                r["peer_med"]["even"] = 0.0
                r["peer_med"]["odd"] = 0.0
    sc_f3 = score_full_receipt(rec_f3, manifest, pins, part_a_sc, allow_fixture=True)
    assert sc_f3["CM1_confirmatory_gates"]["CM1a_L7H5"]["verdict"] == "confirmed"
    assert sc_f3["CM1_confirmatory_gates"]["CM1a_p92_09"]["verdict"] == "confirmed"
    # p92-46 margin ~0.20 >= 0.15 -> selective on both CM.2 and CM.3!
    assert sc_f3["CM2_qkv_factorization"]["L7H5_p92_46_raw"]["control_selectivity"] == "selective_over_matched_controls"
    assert sc_f3["CM2_qkv_factorization"]["L7H5_p92_46_spec"]["control_selectivity"] == "selective_over_matched_controls"
    assert sc_f3["CM3_position_scope"]["L7H5_p92_46_raw"]["control_selectivity"] == "selective_over_matched_controls"
    assert sc_f3["CM3_position_scope"]["L7H5_p92_46_spec"]["control_selectivity"] == "selective_over_matched_controls"
    # p92-09 margin 0.20 < 0.25 -> NOT control-selective on both CM.2 and CM.3!
    assert sc_f3["CM2_qkv_factorization"]["L7H5_p92_09_spec"]["verdict"] == "v_dominant_within_head"
    assert sc_f3["CM2_qkv_factorization"]["L7H5_p92_09_spec"]["control_selectivity"] == "within_head_only_not_control_selective"
    assert sc_f3["CM3_position_scope"]["L7H5_p92_09_spec"]["verdict"] == "read_pos_dominant_within_head"
    assert sc_f3["CM3_position_scope"]["L7H5_p92_09_spec"]["control_selectivity"] == "within_head_only_not_control_selective"
    checks.append("f3_p92_09_vs_p92_46_control_margin_0_20_threshold_separation_ok")

    # 7. CM.0 strict precedence fixtures (sibling leakage, QKV reconcile failure, actuation failure -> ALL CM.1-CM.4 uninformative!).
    for kw in ({"cm0_break_sibling": True}, {"cm0_break_reconcile": True}, {"cm0_break_actuation": True}):
        rec_bad = _make_synthetic_full_receipt(manifest, pins, **kw)
        sc_bad = score_full_receipt(rec_bad, manifest, pins, part_a_sc, allow_fixture=True)
        assert sc_bad["CM0_technical_gate"]["passed"] is False
        assert sc_bad["CM1_confirmatory_gates"]["CM1a_L7H5"]["verdict"] == CM0_FAIL_SCI
        assert sc_bad["CM1_confirmatory_gates"]["CM1a_p92_09"]["verdict"] == CM0_FAIL_SCI
        assert sc_bad["CM1_confirmatory_gates"]["CM1b_L8H15"]["verdict"] == CM0_FAIL_SCI
        assert sc_bad["CM2_qkv_factorization"]["L7H5_p92_46_raw"]["verdict"] == CM0_FAIL_SCI
        assert sc_bad["CM4_span_swaps"]["CM4a_full_4tok"]["L8H15"]["spec"]["verdict"] == CM0_FAIL_SCI
    checks.append("cm0_strict_precedence_all_3_failure_modes_ok")

    # 8. Smoke receipt refusal by full scorer (must raise ValueError and emit NO scorecard!).
    tiny_model = rca.TinyOlmo2ForCausalLM(seed=99, hidden_size=64, num_heads=16, num_layers=9, vocab_size=100278)
    tiny_model.eval()
    smoke_rec = rca.execute_causal_attn(
        manifest=manifest,
        pins=pins,
        smoke_mode=True,
        device="cpu",
        model_loader=lambda _r, _c, _d: tiny_model,
        execution_provenance="synthetic_scorer_fixture",
    )
    sm_ver = verify_smoke_receipt_only(smoke_rec, manifest, pins, part_a_sc, allow_fixture=True)
    assert sm_ver["smoke_technical_check_passed"] is True
    assert "CM1_confirmatory_gates" not in sm_ver
    refused_smoke_in_full = False
    try:
        score_full_receipt(smoke_rec, manifest, pins, part_a_sc, allow_fixture=True)
    except ValueError:
        refused_smoke_in_full = True
    assert refused_smoke_in_full, "Full scorer MUST refuse smoke receipt"
    checks.append("smoke_verifier_technical_only_and_full_scorer_refuses_smoke_ok")

    return {
        "scorer_cpu_self_tests_passed": True,
        "n_checks": len(checks),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Part B (#4) Causal Attention-Reading Offline Scorer")
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--verify-smoke-only", action="store_true")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--receipt", type=Path, default=None)
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--runner", type=Path, default=None)
    ap.add_argument("--part-a-scorecard", type=Path, default=None)
    ap.add_argument("--design-sha256", type=str, default="")
    ap.add_argument("--manifest-sha256", type=str, default="")
    ap.add_argument("--runner-sha256", type=str, default="")
    ap.add_argument("--scorer-sha256", type=str, default="")
    ap.add_argument("--part-a-scorecard-sha256", type=str, default=rca.PART_A_SCORECARD_SHA256)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    repo_root = args.repo_root.resolve()
    base_dir = repo_root / "explore_olmo2_attn_mlp"
    design_p = args.design or (base_dir / "CAUSAL-ATTN-DESIGN.md")
    manifest_p = args.manifest or (base_dir / "causal_attn_manifest.json")
    runner_p = args.runner or (base_dir / "run_causal_attn.py")
    scorer_p = Path(__file__).resolve()
    part_a_p = args.part_a_scorecard or Path(
        "/home/elisepc/green/Journal/notes/reports/vesper-bridge-full-2026-11-21/instruct_bridge_scorecard.json"
    )

    if args.self_test_only:
        res = run_scorer_self_tests(
            design_path=design_p,
            manifest_path=manifest_p,
            runner_path=runner_p,
            scorer_path=scorer_p,
            part_a_scorecard_path=part_a_p,
        )
        print(json.dumps(res, indent=2))
        return 0

    if args.receipt is None or not args.receipt.is_file():
        print(f"ERROR: missing --receipt ({args.receipt})", file=sys.stderr)
        return 2

    manifest, part_a_sc, pins, _ = rca._admit_frozen_package_with_bytes(
        design_path=design_p,
        manifest_path=manifest_p,
        runner_path=runner_p,
        scorer_path=scorer_p,
        part_a_scorecard_path=part_a_p,
        expected_design_sha=args.design_sha256,
        expected_manifest_sha=args.manifest_sha256,
        expected_runner_sha=args.runner_sha256,
        expected_scorer_sha=args.scorer_sha256,
        expected_part_a_scorecard_sha=args.part_a_scorecard_sha256,
    )
    receipt_bytes = args.receipt.read_bytes()
    receipt_sha256 = rca.sha256_bytes(receipt_bytes)
    receipt = json.loads(receipt_bytes.decode("utf-8"))

    if args.verify_smoke_only:
        ver = verify_smoke_receipt_only(receipt, manifest, pins, part_a_sc, allow_fixture=False)
        ver["receipt_sha256"] = receipt_sha256
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(ver, indent=2), encoding="utf-8")
        print(json.dumps(ver, indent=2))
        return 0 if ver["smoke_technical_check_passed"] else 1

    scorecard = score_full_receipt(receipt, manifest, pins, part_a_sc, allow_fixture=False)
    scorecard["meta"]["receipt_sha256"] = receipt_sha256
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    print(json.dumps({
        "CM0": scorecard["CM0_technical_gate"]["status"],
        "CM1a_L7H5": scorecard["CM1_confirmatory_gates"]["CM1a_L7H5"]["verdict"],
        "CM1a_p92_09": scorecard["CM1_confirmatory_gates"]["CM1a_p92_09"]["verdict"],
        "CM1b_L8H15": scorecard["CM1_confirmatory_gates"]["CM1b_L8H15"]["verdict"],
        "CM2_L7H5_p92_46_raw": scorecard["CM2_qkv_factorization"]["L7H5_p92_46_raw"]["verdict"],
        "CM3_L7H5_p92_46_raw": scorecard["CM3_position_scope"]["L7H5_p92_46_raw"]["verdict"],
        "CM4a_L8H15_spec": scorecard["CM4_span_swaps"]["CM4a_full_4tok"]["L8H15"]["spec"]["verdict"],
        "CM4b_L8H15_spec": scorecard["CM4_span_swaps"]["CM4b_user_2tok"]["L8H15"]["spec"]["verdict"],
        "out": str(args.out) if args.out else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
