#!/usr/bin/env python3
"""Pinned Offline Scorer for Frozen Instruct Signed-Dose, Role-Genericity & Turn-Boundary Study (v2.1).

Implements every registered decision (`SD.1`–`SD.6`) and refusal policy in
`explore_olmo2_attn_mlp/SIGNED-DOSE-DESIGN.md` (v2.1, folding Alden #36):
  - Hard-refuses smoke receipts, partial checkpoints, mismatched manifest/design digests,
    missing/duplicate/extra rows, non-causal `(unit, head)` pairs, Q3 `pos_indices` mismatches,
    non-finite values, or internally inconsistent derived metrics.
  - Performs an exact manifest-to-result key join on:
      * All `3,240` Q1+Q2 `(prompt_id, unit, head_name, dose_mag)` rows covering all
        `6,480` explicit signed `(prompt_id, unit, head_name, +/-dose_mag)` keys, and
      * All `740` Q3 `(prompt_id, unit, mask)` rows with exact mask `pos_indices` verification.
  - Enforces full gate propagation:
      * `SD.1(a)` sham failure -> Q1+Q2 `invalidated (sham gate failed)` (`SD.2`–`SD.5` not read, passed=False).
      * `SD.1(b)` null failure -> Q1+Q2 `unreadable (null gate failed)` (`SD.2`–`SD.5` unreadable, passed=False).
      * Independent Q3 null failure -> `SD.6` `unreadable (q3 null gate failed)` (passed=False).
      * Production call to `run_signed_dose.evaluate_sd6a_bos_share(..., floor=0.30)`.
  - `--self-test-only` exercises the REAL `score_results(...)` end-to-end on passing, refusing,
    gate-propagating, and negative-union-witness (`-0.05 / -0.20`) full receipts.
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_signed_dose import (  # noqa: E402
    CORE_UNITS,
    DOSE_MAGNITUDES,
    N_NULL_DRAWS,
    SCHEMA_VERSION,
    assert_causal_head,
    evaluate_sd6a_bos_share,
    sha256_file,
)


SHAM_THRESHOLD = 1e-4
Q12_NULL_THRESHOLD = 0.15
Q3_NULL_THRESHOLD = 0.15
SD6A_UNION_FLOOR = 0.30
SD1B_HEADS: tuple[str, ...] = ("L8H15", "L8H4", "L8H11", "L7H5")
BLOCK_A_THIRD_PROMPTS: tuple[str, ...] = ("third_F1", "third_F2")
Q3_STORIES: tuple[str, ...] = ("F1", "F2")


def _require_finite(val: Any, label: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"Non-numeric scalar for {label}: {val!r}")
    f = float(val)
    if not math.isfinite(f):
        raise ValueError(f"Non-finite scalar for {label}: {val!r}")
    return f


def _summarize(xs: list[float]) -> dict[str, Any]:
    if not xs:
        raise ValueError("Cannot summarize empty list")
    s = sorted(xs)
    n = len(s)
    med = float(statistics.median(s))
    q1 = float(statistics.median(s[: n // 2])) if n >= 2 else s[0]
    q3 = float(statistics.median(s[(n + 1) // 2 :])) if n >= 2 else s[0]
    return {
        "n": n,
        "mean": float(sum(s) / n),
        "median": med,
        "iqr": float(q3 - q1),
        "min": float(s[0]),
        "max": float(s[-1]),
    }


def _validate_q12_metric_block(blk: dict[str, Any], label: str) -> None:
    d0 = _require_finite(blk.get("d0"), f"{label}.d0")
    dp = _require_finite(blk.get("d_pos"), f"{label}.d_pos")
    dm = _require_finite(blk.get("d_neg"), f"{label}.d_neg")
    ev = _require_finite(blk.get("even"), f"{label}.even")
    od = _require_finite(blk.get("odd"), f"{label}.odd")
    sh = _require_finite(blk.get("max_abs_sham"), f"{label}.max_abs_sham")
    if sh < 0.0:
        raise ValueError(f"Negative max_abs_sham at {label}: {sh}")
    if abs(ev - (0.5 * (dp + dm) - d0)) > 1e-6:
        raise ValueError(f"Inconsistent even contrast at {label}: {ev} vs {0.5 * (dp + dm) - d0}")
    if abs(od - (0.5 * (dp - dm))) > 1e-6:
        raise ValueError(f"Inconsistent odd contrast at {label}: {od} vs {0.5 * (dp - dm)}")

    for nkey in ("wdown_orth", "iso_resid"):
        nb = blk.get(nkey)
        if not isinstance(nb, dict):
            raise ValueError(f"Missing null block {label}.{nkey}")
        for dfield, mfield in (
            ("d_pos_draws", "d_pos_med"),
            ("d_neg_draws", "d_neg_med"),
            ("even_draws", "even_med"),
            ("odd_draws", "odd_med"),
        ):
            draws = nb.get(dfield)
            if not isinstance(draws, list) or len(draws) != N_NULL_DRAWS:
                raise ValueError(f"Expected {N_NULL_DRAWS} draws at {label}.{nkey}.{dfield}")
            fdraws = [_require_finite(x, f"{label}.{nkey}.{dfield}[{i}]") for i, x in enumerate(draws)]
            med = _require_finite(nb.get(mfield), f"{label}.{nkey}.{mfield}")
            if abs(med - statistics.median(fdraws)) > 1e-6:
                raise ValueError(f"Median mismatch at {label}.{nkey}.{mfield}: {med} vs {statistics.median(fdraws)}")
        # Check draw-level even/odd consistency against d_pos_draws, d_neg_draws, d0
        ps = [float(x) for x in nb["d_pos_draws"]]
        ms = [float(x) for x in nb["d_neg_draws"]]
        es = [float(x) for x in nb["even_draws"]]
        os_ = [float(x) for x in nb["odd_draws"]]
        for k in range(N_NULL_DRAWS):
            if abs(es[k] - (0.5 * (ps[k] + ms[k]) - d0)) > 1e-6:
                raise ValueError(f"Inconsistent null even draw {k} at {label}.{nkey}")
            if abs(os_[k] - (0.5 * (ps[k] - ms[k]))) > 1e-6:
                raise ValueError(f"Inconsistent null odd draw {k} at {label}.{nkey}")


def _validate_q3_metric_block(blk: dict[str, Any], label: str) -> None:
    nz = _require_finite(blk.get("n_z"), f"{label}.n_z")
    for nkey, rfield, lfield in (
        ("wdown_draws", "r_z_wdown_med", "l_z_wdown"),
        ("iso_draws", "r_z_iso_med", "l_z_iso"),
    ):
        draws = blk.get(nkey)
        if not isinstance(draws, list) or len(draws) != N_NULL_DRAWS:
            raise ValueError(f"Expected {N_NULL_DRAWS} draws at {label}.{nkey}")
        fdraws = [_require_finite(x, f"{label}.{nkey}[{i}]") for i, x in enumerate(draws)]
        rz = _require_finite(blk.get(rfield), f"{label}.{rfield}")
        lz = _require_finite(blk.get(lfield), f"{label}.{lfield}")
        if abs(rz - statistics.median(fdraws)) > 1e-6:
            raise ValueError(f"Median mismatch at {label}.{rfield}: {rz} vs {statistics.median(fdraws)}")
        if abs(lz - (nz - rz)) > 1e-6:
            raise ValueError(f"Inconsistent {label}.{lfield}: {lz} vs {nz - rz}")


def score_results(
    results_path: Path,
    manifest_path: Path,
    design_path: Path,
    expected_manifest_sha256: str | None = None,
    expected_design_sha256: str | None = None,
) -> dict[str, Any]:
    if not results_path.is_file():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    if not design_path.is_file():
        raise FileNotFoundError(f"Design file not found: {design_path}")

    manifest_sha = sha256_file(manifest_path)
    design_sha = sha256_file(design_path)
    results_sha = sha256_file(results_path)
    scorer_sha = sha256_file(Path(__file__).resolve())

    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise ValueError(f"Manifest SHA mismatch: {manifest_sha} != {expected_manifest_sha256}")
    if expected_design_sha256 is not None and design_sha != expected_design_sha256:
        raise ValueError(f"Design SHA mismatch: {design_sha} != {expected_design_sha256}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))

    if manifest.get("meta", {}).get("design_sha256") != design_sha:
        raise ValueError(
            f"Manifest design_sha256 ({manifest.get('meta', {}).get('design_sha256')}) != design file ({design_sha})"
        )

    # 1. Receipt completeness & identity enforcement (refuse smoke / partial checkpoints / digest drift)
    rmeta = results.get("meta", {})
    if rmeta.get("artifact_type") != "full_receipt":
        raise ValueError(f"Refused non-full artifact_type={rmeta.get('artifact_type')!r} (expected 'full_receipt')")
    if rmeta.get("run_mode") != "full":
        raise ValueError(f"Refused non-full run_mode={rmeta.get('run_mode')!r} (expected 'full')")
    if rmeta.get("complete") is not True:
        raise ValueError(f"Refused incomplete receipt: complete={rmeta.get('complete')!r}")
    if rmeta.get("smoke_1unit") is True:
        raise ValueError("Refused smoke_1unit=True receipt")
    if rmeta.get("units_completed") != list(CORE_UNITS):
        raise ValueError(
            f"Refused incomplete units_completed={rmeta.get('units_completed')!r} (expected {list(CORE_UNITS)!r})"
        )
    if rmeta.get("manifest_sha256") != manifest_sha:
        raise ValueError(
            f"Refused results with manifest_sha256={rmeta.get('manifest_sha256')} != bound manifest {manifest_sha}"
        )
    if rmeta.get("design_sha256") != design_sha:
        raise ValueError(
            f"Refused results with design_sha256={rmeta.get('design_sha256')} != bound design {design_sha}"
        )

    # 2. Exact Q1+Q2 manifest-to-result key join (paired + explicit signed keys)
    m_doses = tuple(float(x) for x in manifest["meta"]["dose_magnitudes"])
    expected_q12: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    expected_signed_keys: set[tuple[str, str, str, float]] = set()
    for pr in manifest["q12_prompts"]:
        mags = m_doses if pr["block_a"] else (0.8,)
        for ep in manifest["eligible_pairs"]:
            assert_causal_head(int(ep["unit_layer"]), int(ep["head_layer"]), ep["unit"], ep["head_name"])
            for mag in mags:
                k = (pr["id"], ep["unit"], ep["head_name"], float(mag))
                if k in expected_q12:
                    raise ValueError(f"Duplicate expected Q1+Q2 key in manifest: {k}")
                expected_q12[k] = {
                    "cell": pr["cell"],
                    "frame": pr["frame"],
                    "block_a": bool(pr["block_a"]),
                    "unit_layer": int(ep["unit_layer"]),
                    "head_layer": int(ep["head_layer"]),
                    "head_idx": int(ep["head_idx"]),
                    "head_kind": ep["head_kind"],
                }
                expected_signed_keys.add((pr["id"], ep["unit"], ep["head_name"], +float(mag)))
                expected_signed_keys.add((pr["id"], ep["unit"], ep["head_name"], -float(mag)))

    q12_by_key: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    observed_signed_keys: set[tuple[str, str, str, float]] = set()
    raw_q12 = results.get("q12_rows")
    if not isinstance(raw_q12, list):
        raise ValueError("Missing or invalid q12_rows list")
    for idx, row in enumerate(raw_q12):
        mag = _require_finite(row.get("dose_mag"), f"q12_rows[{idx}].dose_mag")
        assert_causal_head(int(row["unit_layer"]), int(row["head_layer"]), str(row["unit"]), str(row["head_name"]))
        k = (str(row["prompt_id"]), str(row["unit"]), str(row["head_name"]), mag)
        if k in q12_by_key:
            raise ValueError(f"Duplicate Q1+Q2 row key: {k}")
        if k not in expected_q12:
            raise ValueError(f"Unexpected Q1+Q2 row key not in manifest: {k}")
        exp_m = expected_q12[k]
        for mf in ("cell", "frame", "block_a", "unit_layer", "head_layer", "head_idx", "head_kind"):
            if row.get(mf) != exp_m[mf]:
                raise ValueError(f"Metadata mismatch on Q1+Q2 key {k} field {mf}: {row.get(mf)!r} != {exp_m[mf]!r}")
        _validate_q12_metric_block(row["spec"], f"q12[{k}].spec")
        _validate_q12_metric_block(row["raw"], f"q12[{k}].raw")
        q12_by_key[k] = row
        observed_signed_keys.add((k[0], k[1], k[2], +mag))
        observed_signed_keys.add((k[0], k[1], k[2], -mag))

    missing_q12 = set(expected_q12.keys()) - set(q12_by_key.keys())
    if missing_q12:
        raise ValueError(f"Missing {len(missing_q12)} Q1+Q2 rows (first: {sorted(missing_q12)[0]})")
    if observed_signed_keys != expected_signed_keys:
        raise ValueError("Signed Q1+Q2 key set mismatch")

    # 3. Exact Q3 manifest-to-result key & mask join
    expected_q3: dict[tuple[str, str, str], dict[str, Any]] = {}
    for pr in manifest["q3_prompts"]:
        for uid in CORE_UNITS:
            pu = pr["per_unit"][uid]
            for mname, pidx in pu["masks"].items():
                k3 = (pr["id"], uid, mname)
                if k3 in expected_q3:
                    raise ValueError(f"Duplicate expected Q3 key in manifest: {k3}")
                expected_q3[k3] = {
                    "dialogue_frame": pr["dialogue_frame"],
                    "story": pr["story"],
                    "cell": pr["cell"],
                    "pos_indices": pidx,
                }

    q3_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    raw_q3 = results.get("q3_rows")
    if not isinstance(raw_q3, list):
        raise ValueError("Missing or invalid q3_rows list")
    for idx, row in enumerate(raw_q3):
        k3 = (str(row["prompt_id"]), str(row["unit"]), str(row["mask"]))
        if k3 in q3_by_key:
            raise ValueError(f"Duplicate Q3 row key: {k3}")
        if k3 not in expected_q3:
            raise ValueError(f"Unexpected Q3 row key not in manifest: {k3}")
        exp3 = expected_q3[k3]
        for mf in ("dialogue_frame", "story", "cell", "pos_indices"):
            if row.get(mf) != exp3[mf]:
                raise ValueError(f"Q3 mismatch on key {k3} field {mf}: {row.get(mf)!r} != {exp3[mf]!r}")
        _validate_q3_metric_block(row["spec"], f"q3[{k3}].spec")
        _validate_q3_metric_block(row["raw"], f"q3[{k3}].raw")
        q3_by_key[k3] = row

    missing_q3 = set(expected_q3.keys()) - set(q3_by_key.keys())
    if missing_q3:
        raise ValueError(f"Missing {len(missing_q3)} Q3 rows (first: {sorted(missing_q3)[0]})")

    # -----------------------------------------------------------------------------------------------------------------
    # 4. Evaluate Q1+Q2 Decisions (SD.1 -> SD.2, SD.3, SD.4, SD.5)
    # -----------------------------------------------------------------------------------------------------------------
    all_q12_rows = list(q12_by_key.values())
    max_sham_spec = max(float(r["spec"]["max_abs_sham"]) for r in all_q12_rows)
    max_sham_raw = max(float(r["raw"]["max_abs_sham"]) for r in all_q12_rows)
    max_sham_all = max(max_sham_spec, max_sham_raw)
    sd1a_pass = max_sham_all < SHAM_THRESHOLD

    def select_q12(
        head_name: str,
        prompt_ids: tuple[str, ...] | None = None,
        cell: str | None = None,
        dose_mag: float = 0.8,
    ) -> list[dict[str, Any]]:
        out = [
            r for r in all_q12_rows
            if r["head_name"] == head_name
            and abs(float(r["dose_mag"]) - dose_mag) < 1e-12
            and (prompt_ids is None or r["prompt_id"] in prompt_ids)
            and (cell is None or r["cell"] == cell)
        ]
        return out

    sd1b_by_head: dict[str, Any] = {}
    for h in SD1B_HEADS:
        rows_h = select_q12(h, prompt_ids=BLOCK_A_THIRD_PROMPTS, dose_mag=0.8)
        if len(rows_h) != 10:
            raise ValueError(f"Expected 10 SD.1(b) rows for {h}, got {len(rows_h)}")
        e_nulls = [float(r["spec"]["wdown_orth"]["even_med"]) for r in rows_h]
        o_nulls = [float(r["spec"]["wdown_orth"]["odd_med"]) for r in rows_h]
        mean_e = sum(e_nulls) / len(e_nulls)
        mean_o = sum(o_nulls) / len(o_nulls)
        h_pass = (abs(mean_e) <= Q12_NULL_THRESHOLD) and (abs(mean_o) <= Q12_NULL_THRESHOLD)
        sd1b_by_head[h] = {
            "n_tuples": len(rows_h),
            "mean_even_null_spec": float(mean_e),
            "mean_odd_null_spec": float(mean_o),
            "even_null_summary_spec": _summarize(e_nulls),
            "odd_null_summary_spec": _summarize(o_nulls),
            "passed": h_pass,
        }
    sd1b_pass = all(v["passed"] for v in sd1b_by_head.values())

    if not sd1a_pass:
        q12_validity = "invalidated (sham gate failed)"
        sd1_status = "failed (sham gate failed)"
    elif not sd1b_pass:
        q12_validity = "unreadable (null gate failed)"
        sd1_status = "failed (null gate failed)"
    else:
        q12_validity = "valid"
        sd1_status = "held"

    sd1_result = {
        "status": sd1_status,
        "passed": bool(sd1a_pass and sd1b_pass),
        "q12_validity": q12_validity,
        "clause_a_sham": {
            "passed": sd1a_pass,
            "threshold": SHAM_THRESHOLD,
            "max_abs_sham_spec": float(max_sham_spec),
            "max_abs_sham_raw": float(max_sham_raw),
            "max_abs_sham_all": float(max_sham_all),
            "n_patched_sham_tuples": len(all_q12_rows) * 2,
        },
        "clause_b_null": {
            "passed": sd1b_pass,
            "threshold": Q12_NULL_THRESHOLD,
            "by_head": sd1b_by_head,
        },
    }

    def gate_wrap(raw_pass: bool, details: dict[str, Any]) -> dict[str, Any]:
        if not sd1a_pass:
            return {"status": "not_read (invalidated: sham gate failed)", "passed": False, "details": details}
        if not sd1b_pass:
            return {"status": "unreadable (null gate failed)", "passed": False, "details": details}
        return {"status": "held" if raw_pass else "failed", "passed": bool(raw_pass), "details": details}

    # SD.2: Rectified suppression on L8H15 & L8H4 (Block A THIRD_GENERIC_NEW, 10 tuples/head)
    sd2_heads: dict[str, Any] = {}
    for h in ("L8H15", "L8H4"):
        by_mag_spec: dict[str, Any] = {}
        by_mag_raw: dict[str, Any] = {}
        dpos_means: dict[float, float] = {}
        for mag in DOSE_MAGNITUDES:
            rs = select_q12(h, prompt_ids=BLOCK_A_THIRD_PROMPTS, dose_mag=mag)
            if len(rs) != 10:
                raise ValueError(f"Expected 10 SD.2 rows for {h} @ {mag}, got {len(rs)}")
            sp_dpos = [float(r["spec"]["d_pos"]) for r in rs]
            sp_dneg = [float(r["spec"]["d_neg"]) for r in rs]
            sp_even = [float(r["spec"]["even"]) for r in rs]
            sp_odd = [float(r["spec"]["odd"]) for r in rs]
            by_mag_spec[str(mag)] = {
                "d_pos": _summarize(sp_dpos),
                "d_neg": _summarize(sp_dneg),
                "even": _summarize(sp_even),
                "odd": _summarize(sp_odd),
            }
            by_mag_raw[str(mag)] = {
                "d_pos": _summarize([float(r["raw"]["d_pos"]) for r in rs]),
                "d_neg": _summarize([float(r["raw"]["d_neg"]) for r in rs]),
                "even": _summarize([float(r["raw"]["even"]) for r in rs]),
                "odd": _summarize([float(r["raw"]["odd"]) for r in rs]),
            }
            dpos_means[mag] = by_mag_spec[str(mag)]["d_pos"]["mean"]

        monotonic = dpos_means[1.6] < dpos_means[0.8] < dpos_means[0.4] < dpos_means[0.2]
        m_e08 = by_mag_spec["0.8"]["even"]["mean"]
        m_e16 = by_mag_spec["1.6"]["even"]["mean"]
        m_o08 = by_mag_spec["0.8"]["odd"]["mean"]
        m_dneg08 = by_mag_spec["0.8"]["d_neg"]["mean"]
        cond_e08 = m_e08 <= -0.15
        cond_e16 = m_e16 <= -0.25
        cond_ratio = abs(m_e08) >= 0.40 * abs(m_o08)
        cond_no_flip = m_dneg08 < 0.25
        h_pass = monotonic and cond_e08 and cond_e16 and cond_ratio and cond_no_flip
        sd2_heads[h] = {
            "passed": h_pass,
            "checks": {
                "monotonic_negative_dpos": monotonic,
                "even_08_le_neg015": cond_e08,
                "even_16_le_neg025": cond_e16,
                "abs_even_ge_040_abs_odd_08": cond_ratio,
                "dneg_08_lt_pos025": cond_no_flip,
            },
            "spec_by_mag": by_mag_spec,
            "raw_by_mag": by_mag_raw,
        }
    sd2_result = gate_wrap(all(v["passed"] for v in sd2_heads.values()), {"by_head": sd2_heads})

    # SD.3: Odd-dominant / partial-flip contrast on L8H11 & L7H5 (Block A THIRD_GENERIC_NEW, 10 tuples)
    l8h15_08 = select_q12("L8H15", prompt_ids=BLOCK_A_THIRD_PROMPTS, dose_mag=0.8)
    l8h15_m_even = sum(float(r["spec"]["even"]) for r in l8h15_08) / len(l8h15_08)
    l8h15_m_odd = sum(float(r["spec"]["odd"]) for r in l8h15_08) / len(l8h15_08)
    l8h15_ratio = abs(l8h15_m_odd) / (abs(l8h15_m_even) + 0.05)

    sd3_heads: dict[str, Any] = {}
    for h in ("L8H11", "L7H5"):
        rs = select_q12(h, prompt_ids=BLOCK_A_THIRD_PROMPTS, dose_mag=0.8)
        if len(rs) != 10:
            raise ValueError(f"Expected 10 SD.3 rows for {h}, got {len(rs)}")
        dneg = [float(r["spec"]["d_neg"]) for r in rs]
        dneg_minus_null = [float(r["spec"]["d_neg"]) - float(r["spec"]["wdown_orth"]["d_neg_med"]) for r in rs]
        draw_means = [
            sum(float(r["spec"]["wdown_orth"]["d_neg_draws"][k]) for r in rs) / len(rs)
            for k in range(N_NULL_DRAWS)
        ]
        m_dneg = sum(dneg) / len(dneg)
        m_excess = sum(dneg_minus_null) / len(dneg_minus_null)
        max_draw_mean = max(draw_means)
        m_even = sum(float(r["spec"]["even"]) for r in rs) / len(rs)
        m_odd = sum(float(r["spec"]["odd"]) for r in rs) / len(rs)
        ratio_h = abs(m_odd) / (abs(m_even) + 0.05)

        c1 = m_dneg >= 0.10
        c2a = m_excess >= 0.10
        c2b = m_dneg > max_draw_mean
        c3 = ratio_h > l8h15_ratio
        sd3_heads[h] = {
            "passed": bool(c1 and c2a and c2b and c3),
            "checks": {
                "mean_dneg_ge_010": c1,
                "mean_excess_over_null_med_ge_010": c2a,
                "mean_dneg_gt_max_null_draw_mean": c2b,
                "odd_even_ratio_gt_L8H15": c3,
            },
            "mean_dneg_spec": float(m_dneg),
            "mean_excess_over_null_med_spec": float(m_excess),
            "null_draw_means_spec": [float(x) for x in draw_means],
            "max_null_draw_mean_spec": float(max_draw_mean),
            "mean_even_spec": float(m_even),
            "mean_odd_spec": float(m_odd),
            "odd_even_ratio_spec": float(ratio_h),
            "dneg_raw": _summarize([float(r["raw"]["d_neg"]) for r in rs]),
        }
    sd3_result = gate_wrap(
        all(v["passed"] for v in sd3_heads.values()),
        {"l8h15_reference_ratio": float(l8h15_ratio), "by_head": sd3_heads},
    )

    # SD.4: Frame-matched noun-form contrast on L8H15 (Block B frames F1–F3, 15 tuples/cell)
    role_rows = select_q12("L8H15", cell="ROLE_DEFINITE_NEW", dose_mag=0.8)
    name_rows = select_q12("L8H15", cell="NAMED_PERSON_NEW", dose_mag=0.8)
    lex_rows = select_q12("L8H15", cell="LEX_ASSISTANT_CTRL", dose_mag=0.8)
    if len(role_rows) != 15 or len(name_rows) != 15 or len(lex_rows) != 10:
        raise ValueError(f"Unexpected SD.4 row counts: role={len(role_rows)}, name={len(name_rows)}, lex={len(lex_rows)}")
    role_sum = _summarize([float(r["spec"]["d_pos"]) for r in role_rows])
    name_sum = _summarize([float(r["spec"]["d_pos"]) for r in name_rows])
    lex_sum = _summarize([float(r["spec"]["d_pos"]) for r in lex_rows])
    diff_rn = role_sum["mean"] - name_sum["mean"]

    rand_l8_names = [f"L{l}H{h}" for (l, h) in manifest["matched_random_heads"] if int(l) == 8]
    if len(rand_l8_names) != 3:
        raise ValueError(f"Expected 3 matched L8 random heads, got {rand_l8_names}")
    rand_l8_role_means: dict[str, float] = {}
    for rh in rand_l8_names:
        rrs = select_q12(rh, cell="ROLE_DEFINITE_NEW", dose_mag=0.8)
        if len(rrs) != 15:
            raise ValueError(f"Expected 15 ROLE_DEFINITE_NEW rows for {rh}, got {len(rrs)}")
        rand_l8_role_means[rh] = float(sum(float(r["spec"]["d_pos"]) for r in rrs) / len(rrs))
    min_rand_l8 = min(rand_l8_role_means.values())

    sd4_cond_contrast = diff_rn <= -0.25
    sd4_cond_rand = role_sum["mean"] < (min_rand_l8 - 0.15)
    sd4_falsifier = abs(diff_rn) <= 0.15
    sd4_result = gate_wrap(
        sd4_cond_contrast and sd4_cond_rand,
        {
            "scope_note": "Frame-matched noun-form contrast on these prompts (definite role nouns vs bare proper names).",
            "role_definite_spec": role_sum,
            "named_person_spec": name_sum,
            "lex_assistant_ctrl_spec": lex_sum,
            "role_definite_raw": _summarize([float(r["raw"]["d_pos"]) for r in role_rows]),
            "named_person_raw": _summarize([float(r["raw"]["d_pos"]) for r in name_rows]),
            "lex_assistant_ctrl_raw": _summarize([float(r["raw"]["d_pos"]) for r in lex_rows]),
            "role_minus_named_mean_spec": float(diff_rn),
            "rand_l8_role_means_spec": rand_l8_role_means,
            "min_rand_l8_role_mean_spec": float(min_rand_l8),
            "checks": {
                "role_minus_named_le_neg025": sd4_cond_contrast,
                "beats_min_rand_l8_by_015": sd4_cond_rand,
                "falsifier_abs_diff_le_015_triggered": sd4_falsifier,
            },
        },
    )

    # SD.5: Counterbalanced Person-vs-Object Control (O1–O2; L8H4 10 tuples/cell, L5H4 6 tuples/cell)
    sd5_heads: dict[str, Any] = {}
    for h, exp_n in (("L8H4", 10), ("L5H4", 6)):
        p_rows = select_q12(h, cell="PERSON_OBJFRAME", dose_mag=0.8)
        o_rows = select_q12(h, cell="OBJECT_OBJFRAME", dose_mag=0.8)
        if len(p_rows) != exp_n or len(o_rows) != exp_n:
            raise ValueError(f"Expected {exp_n} SD.5 rows for {h}, got person={len(p_rows)}, object={len(o_rows)}")
        p_sum = _summarize([float(r["spec"]["d_pos"]) for r in p_rows])
        o_sum = _summarize([float(r["spec"]["d_pos"]) for r in o_rows])
        h_pass = o_sum["mean"] <= (p_sum["mean"] + 0.10)
        sd5_heads[h] = {
            "passed": h_pass,
            "n_tuples_per_cell": exp_n,
            "person_objframe_spec": p_sum,
            "object_objframe_spec": o_sum,
            "person_objframe_raw": _summarize([float(r["raw"]["d_pos"]) for r in p_rows]),
            "object_objframe_raw": _summarize([float(r["raw"]["d_pos"]) for r in o_rows]),
            "object_minus_person_mean_spec": float(o_sum["mean"] - p_sum["mean"]),
        }
    sd5_result = gate_wrap(all(v["passed"] for v in sd5_heads.values()), {"by_head": sd5_heads})

    # -----------------------------------------------------------------------------------------------------------------
    # 5. Evaluate Q3 / SD.6 (Independent Estimand, Exact SELF/THIRD Pairing, Paired I_pre, Null Gate, Production Floor)
    # -----------------------------------------------------------------------------------------------------------------
    # Index Q3 rows by (dialogue_frame, cell, story, unit, mask)
    q3_idx: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    masks_by_frame: dict[str, set[str]] = {}
    for row in q3_by_key.values():
        key5 = (row["dialogue_frame"], row["cell"], row["story"], row["unit"], row["mask"])
        if key5 in q3_idx:
            raise ValueError(f"Duplicate Q3 5-key: {key5}")
        q3_idx[key5] = row
        masks_by_frame.setdefault(row["dialogue_frame"], set()).add(row["mask"])

    # Independent Q3 null gate: |mean R_z_wdown(pre_user_union)| <= 0.15 in one_turn and two_turn_asst_user
    q3_null_diag: dict[str, Any] = {}
    for df in ("one_turn", "two_turn_asst_user", "no_boundary_concat"):
        rz_all = [
            float(q3_idx[(df, cell, story, uid, "pre_user_union")]["spec"]["r_z_wdown_med"])
            for cell in ("SELF_HELD_NEW", "THIRD_GENERIC_NEW")
            for story in Q3_STORIES
            for uid in CORE_UNITS
        ]
        mean_rz = sum(rz_all) / len(rz_all)
        q3_null_diag[df] = {
            "n_rows": len(rz_all),
            "mean_r_z_wdown_spec": float(mean_rz),
            "summary_r_z_wdown_spec": _summarize(rz_all),
            "passed": abs(mean_rz) <= Q3_NULL_THRESHOLD,
            "is_descriptive_only": (df == "no_boundary_concat"),
        }
    q3_null_pass = q3_null_diag["one_turn"]["passed"] and q3_null_diag["two_turn_asst_user"]["passed"]

    # Compute exact paired Delta_ST(frame, mask) across all 10 (unit, story) pairs
    q3_delta_st: dict[str, dict[str, Any]] = {}
    q3_delta_pairs: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    for df, mset in sorted(masks_by_frame.items()):
        q3_delta_st[df] = {}
        for m in sorted(mset):
            pair_vals_spec: dict[tuple[str, str], float] = {}
            pair_vals_raw: list[float] = []
            pair_vals_iso: list[float] = []
            for uid in CORE_UNITS:
                for story in Q3_STORIES:
                    rs = q3_idx[(df, "SELF_HELD_NEW", story, uid, m)]
                    rt = q3_idx[(df, "THIRD_GENERIC_NEW", story, uid, m)]
                    pair_vals_spec[(uid, story)] = float(rs["spec"]["l_z_wdown"]) - float(rt["spec"]["l_z_wdown"])
                    pair_vals_raw.append(float(rs["raw"]["l_z_wdown"]) - float(rt["raw"]["l_z_wdown"]))
                    pair_vals_iso.append(float(rs["spec"]["l_z_iso"]) - float(rt["spec"]["l_z_iso"]))
            q3_delta_pairs[(df, m)] = pair_vals_spec
            q3_delta_st[df][m] = {
                "delta_st_wdown_spec": _summarize(list(pair_vals_spec.values())),
                "delta_st_wdown_raw": _summarize(pair_vals_raw),
                "delta_st_iso_spec": _summarize(pair_vals_iso),
            }

    # Exact paired interaction I_pre per frame
    def paired_interaction(df: str, union_m: str, part_a: str, part_b: str) -> dict[str, Any]:
        vals = [
            q3_delta_pairs[(df, union_m)][(uid, story)]
            - (q3_delta_pairs[(df, part_a)][(uid, story)] + q3_delta_pairs[(df, part_b)][(uid, story)])
            for uid in CORE_UNITS
            for story in Q3_STORIES
        ]
        return _summarize(vals)

    q3_interactions = {
        "one_turn_pre_user_union_minus_bos_plus_uhdr": paired_interaction(
            "one_turn", "pre_user_union", "bos_tok", "user_hdr_turn_final"
        ),
        "two_turn_turn2_union_minus_mideotnl_plus_uhdr": paired_interaction(
            "two_turn_asst_user", "pre_user_union", "mid_eot_nl", "user_hdr_turn_final"
        ),
        "two_turn_turn1_union_minus_bos_plus_uhdr1": paired_interaction(
            "two_turn_asst_user", "pre_user_turn1_union", "bos_tok", "user_hdr_turn1"
        ),
        "no_boundary_concat_descriptive_union_minus_bos_plus_uhdr": paired_interaction(
            "no_boundary_concat", "pre_user_union", "bos_tok", "user_hdr_turn_final"
        ),
    }

    # Production call to evaluate_sd6a_bos_share
    one_bos = q3_delta_st["one_turn"]["bos_tok"]["delta_st_wdown_spec"]["mean"]
    one_union = q3_delta_st["one_turn"]["pre_user_union"]["delta_st_wdown_spec"]["mean"]
    sd6a_eval = evaluate_sd6a_bos_share(delta_st_bos=one_bos, delta_st_union=one_union, floor=SD6A_UNION_FLOOR)

    one_post_diff = (
        q3_delta_st["one_turn"]["post_user_delim_final"]["delta_st_wdown_spec"]["mean"]
        - q3_delta_st["one_turn"]["user_tail_5"]["delta_st_wdown_spec"]["mean"]
    )
    two_post_diff = (
        q3_delta_st["two_turn_asst_user"]["post_user_delim_final"]["delta_st_wdown_spec"]["mean"]
        - q3_delta_st["two_turn_asst_user"]["user_tail_5"]["delta_st_wdown_spec"]["mean"]
    )
    nb_post_diff_desc = (
        q3_delta_st["no_boundary_concat"]["post_user_delim_final"]["delta_st_wdown_spec"]["mean"]
        - q3_delta_st["no_boundary_concat"]["user_tail_5"]["delta_st_wdown_spec"]["mean"]
    )
    sd6b_pass = (one_post_diff >= 0.25) and (two_post_diff >= 0.25)

    if not q3_null_pass:
        sd6_status = "unreadable (q3 null gate failed)"
        sd6_passed = False
        clause_a_out = {**sd6a_eval, "status": "unreadable (q3 null gate failed)", "passed": False}
        clause_b_out = {
            "status": "unreadable (q3 null gate failed)",
            "passed": False,
            "one_turn_post_minus_tail5": float(one_post_diff),
            "two_turn_post_minus_tail5": float(two_post_diff),
            "no_boundary_concat_post_minus_tail5_descriptive": float(nb_post_diff_desc),
        }
    else:
        clause_a_out = sd6a_eval
        clause_b_out = {
            "status": "held" if sd6b_pass else "failed",
            "passed": bool(sd6b_pass),
            "one_turn_post_minus_tail5": float(one_post_diff),
            "two_turn_post_minus_tail5": float(two_post_diff),
            "no_boundary_concat_post_minus_tail5_descriptive": float(nb_post_diff_desc),
        }
        if sd6a_eval["status"] == "not_read (union below floor)":
            sd6_status = "not_read (clause_a union below floor)"
            sd6_passed = False
        elif sd6a_eval["passed"] and sd6b_pass:
            sd6_status = "held"
            sd6_passed = True
        else:
            sd6_status = "failed"
            sd6_passed = False

    sd6_result = {
        "status": sd6_status,
        "passed": sd6_passed,
        "q3_null_gate": {"passed": q3_null_pass, "threshold": Q3_NULL_THRESHOLD, "by_frame": q3_null_diag},
        "clause_a": clause_a_out,
        "clause_b": clause_b_out,
        "paired_interactions_I_pre": q3_interactions,
        "geometry_note": (
            "Active question-tail distance is matched across all 3 frames; two_turn_asst_user read_pos is "
            "+12 tokens later than no_boundary_concat on all 20 matched rows, so no_boundary_concat is descriptive."
        ),
        "delta_st_by_frame_and_mask": q3_delta_st,
    }

    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "verdict_type": "complete_full_scorecard",
            "complete_join_verified": True,
            "manifest_sha256": manifest_sha,
            "design_sha256": design_sha,
            "results_sha256": results_sha,
            "runner_sha256": rmeta.get("runner_sha256"),
            "scorer_sha256": scorer_sha,
            "q12_rows_joined": len(q12_by_key),
            "q12_signed_keys_joined": len(observed_signed_keys),
            "q3_rows_joined": len(q3_by_key),
        },
        "decisions": {
            "SD.1": sd1_result,
            "SD.2": sd2_result,
            "SD.3": sd3_result,
            "SD.4": sd4_result,
            "SD.5": sd5_result,
            "SD.6": sd6_result,
        },
    }


# ---------------------------------------------------------------------------------------------------------------------
# End-to-End Scorer Self-Tests (Exercises REAL `score_results(...)` on Passes, Refusals, Gates & Negative-Union Witness)
# ---------------------------------------------------------------------------------------------------------------------

def _make_metric_q12(d_pos: float, d_neg: float, d0: float = 0.0, sham: float = 1e-7, null_val: float = 0.0) -> dict[str, Any]:
    even = 0.5 * (d_pos + d_neg) - d0
    odd = 0.5 * (d_pos - d_neg)
    np = {
        "d_pos_draws": [null_val] * N_NULL_DRAWS,
        "d_neg_draws": [null_val] * N_NULL_DRAWS,
        "even_draws": [null_val - d0] * N_NULL_DRAWS,
        "odd_draws": [0.0] * N_NULL_DRAWS,
        "d_pos_med": null_val,
        "d_neg_med": null_val,
        "even_med": null_val - d0,
        "odd_med": 0.0,
    }
    return {
        "d0": d0,
        "d_pos": d_pos,
        "d_neg": d_neg,
        "even": even,
        "odd": odd,
        "max_abs_sham": sham,
        "wdown_orth": copy.deepcopy(np),
        "iso_resid": copy.deepcopy(np),
    }


def _make_metric_q3(l_z: float, r_z: float = 0.0) -> dict[str, Any]:
    return {
        "n_z": l_z + r_z,
        "r_z_wdown_med": r_z,
        "l_z_wdown": l_z,
        "r_z_iso_med": r_z,
        "l_z_iso": l_z,
        "wdown_draws": [r_z] * N_NULL_DRAWS,
        "iso_draws": [r_z] * N_NULL_DRAWS,
    }


def _build_synthetic_passing_full_receipt(manifest: dict[str, Any], manifest_sha: str, design_sha: str) -> dict[str, Any]:
    q12_rows: list[dict[str, Any]] = []
    m_doses = tuple(float(x) for x in manifest["meta"]["dose_magnitudes"])
    for pr in manifest["q12_prompts"]:
        mags = m_doses if pr["block_a"] else (0.8,)
        for ep in manifest["eligible_pairs"]:
            for mag in mags:
                h = ep["head_name"]
                cell = pr["cell"]
                # Construct clean synthetic values that satisfy SD.1–SD.5
                if h in ("L8H15", "L8H4") and cell == "THIRD_GENERIC_NEW":
                    d_pos = -0.60 * mag
                    d_neg = -0.20 * mag
                elif h in ("L8H11", "L7H5") and cell == "THIRD_GENERIC_NEW":
                    d_pos = -0.50 * mag
                    d_neg = +0.45 * mag
                elif h == "L8H15" and cell == "ROLE_DEFINITE_NEW":
                    d_pos, d_neg = -0.80, -0.10
                elif h == "L8H15" and cell == "NAMED_PERSON_NEW":
                    d_pos, d_neg = -0.20, -0.05
                elif h in ("L8H4", "L5H4") and cell == "OBJECT_OBJFRAME":
                    d_pos, d_neg = -0.65, -0.05
                elif h in ("L8H4", "L5H4") and cell == "PERSON_OBJFRAME":
                    d_pos, d_neg = -0.50, -0.05
                else:
                    d_pos, d_neg = -0.05, 0.0
                blk = _make_metric_q12(d_pos=d_pos, d_neg=d_neg, d0=0.0, sham=1e-7, null_val=0.01)
                q12_rows.append({
                    "unit": ep["unit"],
                    "unit_layer": ep["unit_layer"],
                    "prompt_id": pr["id"],
                    "cell": pr["cell"],
                    "frame": pr["frame"],
                    "block_a": bool(pr["block_a"]),
                    "dose_mag": float(mag),
                    "head_name": h,
                    "head_layer": ep["head_layer"],
                    "head_idx": ep["head_idx"],
                    "head_kind": ep["head_kind"],
                    "spec": copy.deepcopy(blk),
                    "raw": copy.deepcopy(blk),
                })

    q3_rows: list[dict[str, Any]] = []
    for pr in manifest["q3_prompts"]:
        for uid in CORE_UNITS:
            pu = pr["per_unit"][uid]
            is_self = pr["cell"] == "SELF_HELD_NEW"
            for mname, pidx in pu["masks"].items():
                if mname == "pre_user_union":
                    lz = 0.80 if is_self else 0.20  # Delta_ST = +0.60 >= +0.30
                elif mname == "bos_tok":
                    lz = 0.50 if is_self else 0.10  # Delta_ST = +0.40 > 0.50 * 0.60
                elif mname == "post_user_delim_final":
                    lz = 0.70 if is_self else 0.10  # Delta_ST = +0.60
                elif mname == "user_tail_5":
                    lz = 0.10 if is_self else 0.05  # Delta_ST = +0.05 -> diff = +0.55 >= +0.25
                else:
                    lz = 0.10 if is_self else 0.05
                blk3 = _make_metric_q3(l_z=lz, r_z=0.01)
                q3_rows.append({
                    "unit": uid,
                    "prompt_id": pr["id"],
                    "dialogue_frame": pr["dialogue_frame"],
                    "story": pr["story"],
                    "cell": pr["cell"],
                    "mask": mname,
                    "pos_indices": pidx,
                    "spec": copy.deepcopy(blk3),
                    "raw": copy.deepcopy(blk3),
                })

    return {
        "meta": {
            **manifest["meta"],
            "schema_version": SCHEMA_VERSION,
            "run_mode": "full",
            "artifact_type": "full_receipt",
            "complete": True,
            "smoke_1unit": False,
            "manifest_sha256": manifest_sha,
            "design_sha256": design_sha,
            "runner_sha256": "synthetic_self_test",
            "units_completed": list(CORE_UNITS),
        },
        "denominators": manifest["denominators"],
        "q12_rows": q12_rows,
        "q3_rows": q3_rows,
    }


def run_scorer_self_tests(manifest_path: Path, design_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    m_sha = sha256_file(manifest_path)
    d_sha = sha256_file(design_path)
    base = _build_synthetic_passing_full_receipt(manifest, m_sha, d_sha)

    with tempfile.TemporaryDirectory(prefix="sd_scorer_test_") as td:
        tmp = Path(td)

        def write_and_score(obj: dict[str, Any], name: str) -> dict[str, Any]:
            p = tmp / f"{name}.json"
            p.write_text(json.dumps(obj), encoding="utf-8")
            return score_results(p, manifest_path, design_path)

        # 1. Passing full receipt -> all SD.1–SD.6 hold
        v_pass = write_and_score(base, "pass")
        assert v_pass["meta"]["q12_rows_joined"] == 3240
        assert v_pass["meta"]["q12_signed_keys_joined"] == 6480
        assert v_pass["meta"]["q3_rows_joined"] == 740
        for sd in ("SD.1", "SD.2", "SD.3", "SD.4", "SD.5", "SD.6"):
            assert v_pass["decisions"][sd]["passed"] is True, f"Expected {sd} to pass: {v_pass['decisions'][sd]}"

        # 2. Refusal: smoke receipt
        bad_smoke = copy.deepcopy(base)
        bad_smoke["meta"]["artifact_type"] = "smoke_receipt"
        bad_smoke["meta"]["run_mode"] = "smoke_1unit"
        bad_smoke["meta"]["smoke_1unit"] = True
        try:
            write_and_score(bad_smoke, "smoke")
            raise AssertionError("Smoke receipt was not refused!")
        except ValueError:
            pass

        # 3. Refusal: partial checkpoint
        bad_ckpt = copy.deepcopy(base)
        bad_ckpt["meta"]["artifact_type"] = "partial_checkpoint"
        bad_ckpt["meta"]["complete"] = False
        try:
            write_and_score(bad_ckpt, "ckpt")
            raise AssertionError("Partial checkpoint was not refused!")
        except ValueError:
            pass

        # 4. Refusal: missing Q1+Q2 row
        bad_miss = copy.deepcopy(base)
        bad_miss["q12_rows"].pop()
        try:
            write_and_score(bad_miss, "missing")
            raise AssertionError("Missing Q1+Q2 row was not refused!")
        except ValueError:
            pass

        # 5. Refusal: duplicate Q1+Q2 row
        bad_dup = copy.deepcopy(base)
        bad_dup["q12_rows"].append(copy.deepcopy(bad_dup["q12_rows"][0]))
        try:
            write_and_score(bad_dup, "dup")
            raise AssertionError("Duplicate Q1+Q2 row was not refused!")
        except ValueError:
            pass

        # 6. Refusal: non-finite scalar
        bad_nan = copy.deepcopy(base)
        bad_nan["q12_rows"][0]["spec"]["d_pos"] = float("nan")
        try:
            write_and_score(bad_nan, "nan")
            raise AssertionError("NaN row was not refused!")
        except ValueError:
            pass

        # 7. Refusal: tampered Q3 pos_indices mask
        bad_q3m = copy.deepcopy(base)
        for r in bad_q3m["q3_rows"]:
            if r["mask"] == "bos_tok":
                r["pos_indices"] = [1]
                break
        try:
            write_and_score(bad_q3m, "bad_q3m")
            raise AssertionError("Tampered Q3 pos_indices was not refused!")
        except ValueError:
            pass

        # 8. Refusal: manifest digest mismatch
        bad_sha = copy.deepcopy(base)
        bad_sha["meta"]["manifest_sha256"] = "0" * 64
        try:
            write_and_score(bad_sha, "bad_sha")
            raise AssertionError("Mismatched manifest SHA was not refused!")
        except ValueError:
            pass

        # 9. Gate propagation: SD.1(a) sham failure -> invalidates SD.2–SD.5
        bad_sham = copy.deepcopy(base)
        bad_sham["q12_rows"][0]["spec"]["max_abs_sham"] = 2e-4
        v_sham = write_and_score(bad_sham, "bad_sham")
        assert v_sham["decisions"]["SD.1"]["passed"] is False
        assert v_sham["decisions"]["SD.1"]["q12_validity"] == "invalidated (sham gate failed)"
        for sd in ("SD.2", "SD.3", "SD.4", "SD.5"):
            assert v_sham["decisions"][sd]["passed"] is False
            assert v_sham["decisions"][sd]["status"] == "not_read (invalidated: sham gate failed)"

        # 10. Gate propagation: SD.1(b) null failure -> marks SD.2–SD.5 unreadable
        bad_null = copy.deepcopy(base)
        for r in bad_null["q12_rows"]:
            if r["head_name"] == "L8H15" and r["prompt_id"] in BLOCK_A_THIRD_PROMPTS and r["dose_mag"] == 0.8:
                r["spec"]["wdown_orth"]["d_pos_draws"] = [0.25] * N_NULL_DRAWS
                r["spec"]["wdown_orth"]["d_neg_draws"] = [0.25] * N_NULL_DRAWS
                r["spec"]["wdown_orth"]["even_draws"] = [0.25] * N_NULL_DRAWS
                r["spec"]["wdown_orth"]["d_pos_med"] = 0.25
                r["spec"]["wdown_orth"]["d_neg_med"] = 0.25
                r["spec"]["wdown_orth"]["even_med"] = 0.25
        v_null = write_and_score(bad_null, "bad_null")
        assert v_null["decisions"]["SD.1"]["passed"] is False
        assert v_null["decisions"]["SD.1"]["q12_validity"] == "unreadable (null gate failed)"
        for sd in ("SD.2", "SD.3", "SD.4", "SD.5"):
            assert v_null["decisions"][sd]["passed"] is False
            assert v_null["decisions"][sd]["status"] == "unreadable (null gate failed)"

        # 11. Gate propagation: Q3 null failure -> marks SD.6 unreadable
        bad_q3null = copy.deepcopy(base)
        for r in bad_q3null["q3_rows"]:
            if r["dialogue_frame"] == "one_turn" and r["mask"] == "pre_user_union":
                r["spec"] = _make_metric_q3(l_z=r["spec"]["l_z_wdown"], r_z=0.25)
        v_q3null = write_and_score(bad_q3null, "bad_q3null")
        assert v_q3null["decisions"]["SD.6"]["passed"] is False
        assert v_q3null["decisions"]["SD.6"]["status"] == "unreadable (q3 null gate failed)"

        # 12. Negative-union witness (Alden's bos=-0.05, union=-0.20) exercised through REAL scorer!
        neg_union = copy.deepcopy(base)
        for r in neg_union["q3_rows"]:
            if r["dialogue_frame"] == "one_turn" and r["mask"] == "bos_tok":
                lz = -0.05 if r["cell"] == "SELF_HELD_NEW" else 0.0
                r["spec"] = _make_metric_q3(l_z=lz, r_z=0.0)
            elif r["dialogue_frame"] == "one_turn" and r["mask"] == "pre_user_union":
                lz = -0.20 if r["cell"] == "SELF_HELD_NEW" else 0.0
                r["spec"] = _make_metric_q3(l_z=lz, r_z=0.0)
        v_negu = write_and_score(neg_union, "neg_union")
        assert v_negu["decisions"]["SD.6"]["clause_a"]["passed"] is False
        assert v_negu["decisions"]["SD.6"]["clause_a"]["status"] == "not_read (union below floor)"
        assert v_negu["decisions"]["SD.6"]["passed"] is False

    return {
        "status": "PASSED",
        "all_pass_fixture_verified": True,
        "refused_smoke_receipt": True,
        "refused_partial_checkpoint": True,
        "refused_missing_row": True,
        "refused_duplicate_row": True,
        "refused_non_finite_row": True,
        "refused_tampered_q3_mask": True,
        "refused_manifest_sha_mismatch": True,
        "propagated_sd1a_sham_invalidation": True,
        "propagated_sd1b_null_unreadability": True,
        "propagated_q3_null_unreadability": True,
        "negative_union_witness_not_read": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinned Offline Scorer for Signed-Dose Study (v2.1)")
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=HERE / "signed_dose_manifest.json")
    parser.add_argument("--design", type=Path, default=HERE / "SIGNED-DOSE-DESIGN.md")
    parser.add_argument("--expected-manifest-sha256", type=str, default=None)
    parser.add_argument("--expected-design-sha256", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.self_test_only:
        report = run_scorer_self_tests(args.manifest, args.design)
        print(json.dumps({"scorer_self_test": report}, indent=2))
        return

    if args.results is None:
        raise SystemExit("Must pass --results <path> or --self-test-only")

    verdict = score_results(
        results_path=args.results,
        manifest_path=args.manifest,
        design_path=args.design,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_design_sha256=args.expected_design_sha256,
    )
    txt = json.dumps(verdict, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
