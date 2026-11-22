#!/usr/bin/env python3
"""OLMo 2 1B Instruct — 2x2 Bank x Physical-Norm Bridge Offline Scorer (Part A).

Implements all pre-registered gates and total machine-readable decision maps
(`BR.0`–`BR.4`, including `part_b_norm_decision` and `p92_09_part_b_track`) from
`explore_olmo2_attn_mlp/INSTRUCT-BRIDGE-DESIGN.md` v3
(SHA-256: a1db3bd6743f3fe60571d4c8575788f7f8b4ddc38ba7b5a5f39d25137a6fbaf8).

Hardening & contracts:
- Verifies all 4 external SHA-256 package pins, both reference anchor receipt SHA-256s,
  and per-source literal field bindings BEFORE scoring.
- Hard-refuses non-full receipts (`artifact_type != "instruct_bridge_full_receipt"` or
  `complete != True`), emitting zero scorecard file.
- Canonical row-key & metadata binding BEFORE grouping (`#70` discipline).
- Head-row-only sham contract (`3,220/3,220` head rows present/finite `<= 1e-6`;
  `520/520` unpatched rows including `CAL_SELF` strictly omit sham keys).
- Per-variant / per-draw linear decomposition check (`<= 1e-6`) without assuming
  separately reduced medians subtract (Alden #97).
- Unified anchor-failure disposition: on `BR.0 == failed`, science endpoints (`BR.1`–`BR.4`)
  are `uninformative (anchor_replay_failed)`, `part_b_norm_decision` is
  `decline_norm (anchor_replay_failed)`, and `p92_09_part_b_track` is
  `not_selected (anchor_replay_failed)`.
- `BR.3` `cell_collapsed` is explicitly `None` (`not_applicable`) at `d = 0.8`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

# Import canonical constants & validators from sibling runner.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_instruct_bridge as rib


ANCHOR_TOL = 1e-3
SHAM_TOL = 1e-6
DECOMP_TOL = 1e-6

ANCHOR_FAIL_SCI = "uninformative (anchor_replay_failed)"
ANCHOR_FAIL_NORM = "decline_norm (anchor_replay_failed)"
ANCHOR_FAIL_TRACK = "not_selected (anchor_replay_failed)"

CORE_CELLS_BY_BANK: dict[str, tuple[str, str, str]] = {
    "OLD_BANK": ("SELF_HELD_NEW", "ROLE_DEFINITE_NEW", "NAMED_PERSON_NEW"),
    "NEW_BANK": ("SELF", "ROLE_BARE", "NAME_BARE"),
}
HIGH_DOSE_PAIR_BY_BANK: dict[str, tuple[str, str]] = {
    "OLD_BANK": ("SELF_HELD_NEW", "THIRD_GENERIC_NEW"),
    "NEW_BANK": ("CAL_SELF", "CAL_THIRD"),
}
BR4_PRIMARY_CELL: dict[str, str] = {
    "NEW_BANK": "ROLE_BARE",
    "OLD_BANK": "ROLE_DEFINITE_NEW",
}
L3_UNITS: tuple[str, ...] = ("p92-02", "p92-05", "p92-13")
L7_CTRL_HEADS: tuple[str, str] = ("L7H4", "L7H9")


def _mean(xs: list[float]) -> float:
    if not xs:
        raise ValueError("Cannot take mean of empty list")
    return float(sum(xs) / len(xs))


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _has_any_sham_key(obj: Any) -> bool:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "sham" in str(k).lower():
                return True
            if _has_any_sham_key(v):
                return True
    elif isinstance(obj, list):
        return any(_has_any_sham_key(v) for v in obj)
    return False


# ---------------------------------------------------------------------------
# Strict Canonical Row-Key, Metadata, Sham & Decomposition Validation
# ---------------------------------------------------------------------------

def validate_unpatched_row_integrity(row: dict[str, Any]) -> float:
    """Enforce zero sham keys, finiteness, and per-variant/per-draw linear decomposition."""
    if _has_any_sham_key(row):
        raise ValueError(f"Forbidden sham key found on unpatched row: {(row.get('bank'), row.get('prompt_id'))}")

    t = row.get("transport")
    if not isinstance(t, dict):
        raise ValueError("Unpatched row missing transport dict")
    for k in ("n_write_spec", "n_write_raw", "n_write_peer_med", "n_write_neg_spec", "n_write_neg_raw", "n_write_neg_peer_med"):
        if not _is_finite_number(t.get(k)):
            raise ValueError(f"Non-finite/missing transport.{k} on {row.get('prompt_id')}")
    for k in (
        "wdown_pos_draws_spec", "wdown_pos_draws_raw", "wdown_pos_draws_peer_med",
        "iso_pos_draws_spec", "iso_pos_draws_raw", "iso_pos_draws_peer_med",
    ):
        arr = t.get(k)
        if not isinstance(arr, list) or len(arr) != rib.N_NULL_DRAWS or not all(_is_finite_number(v) for v in arr):
            raise ValueError(f"Invalid transport.{k} on {row.get('prompt_id')}")

    for k in ("norm_multiplier", "injected_norm", "read_pos_resid_norm", "norm_ratio"):
        if not _is_finite_number(row.get(k)):
            raise ValueError(f"Non-finite/missing {k} on unpatched row {row.get('prompt_id')}")

    errs = [
        abs(float(t["n_write_spec"]) - (float(t["n_write_raw"]) - float(t["n_write_peer_med"]))),
        abs(float(t["n_write_neg_spec"]) - (float(t["n_write_neg_raw"]) - float(t["n_write_neg_peer_med"]))),
        *[
            abs(float(t["wdown_pos_draws_spec"][k]) - (float(t["wdown_pos_draws_raw"][k]) - float(t["wdown_pos_draws_peer_med"][k])))
            for k in range(rib.N_NULL_DRAWS)
        ],
        *[
            abs(float(t["iso_pos_draws_spec"][k]) - (float(t["iso_pos_draws_raw"][k]) - float(t["iso_pos_draws_peer_med"][k])))
            for k in range(rib.N_NULL_DRAWS)
        ],
    ]
    max_err = max(errs)
    if max_err > DECOMP_TOL:
        raise ValueError(f"Unpatched linear decomposition failed ({max_err} > {DECOMP_TOL}) on {row.get('prompt_id')}")
    return max_err


def admit_scorer_package(
    design_path: Path,
    manifest_path: Path,
    runner_path: Path | None,
    scorer_path: Path,
    expected_design_sha: str,
    expected_manifest_sha: str,
    expected_runner_sha: str,
    expected_scorer_sha: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Bind external runner pin STRICTLY to the actually imported `rib` module (Alden #104 item 3)."""
    loaded_runner = Path(rib.__file__).resolve()
    if runner_path is not None and runner_path.resolve() != loaded_runner:
        raise ValueError(
            f"--runner path ({runner_path.resolve()}) does not match actually imported runner module ({loaded_runner}); refused."
        )
    loaded_scorer = Path(__file__).resolve()
    if scorer_path.resolve() != loaded_scorer:
        raise ValueError(
            f"Scorer path ({scorer_path.resolve()}) does not match executing scorer ({loaded_scorer}); refused."
        )
    return rib.admit_frozen_package(
        design_path=design_path,
        manifest_path=manifest_path,
        runner_path=loaded_runner,
        scorer_path=loaded_scorer,
        expected_design_sha=expected_design_sha,
        expected_manifest_sha=expected_manifest_sha,
        expected_runner_sha=expected_runner_sha,
        expected_scorer_sha=expected_scorer_sha,
    )


def validate_head_row_integrity(
    row: dict[str, Any],
    expected_peers: list[int] | None = None,
) -> tuple[float, float, float, float | None]:
    """Enforce sham <= 1e-6, finiteness, per-metric derived summary integrity, linear decomp & 19-peer deltas."""
    for metric_key in ("spec", "raw", "peer_med"):
        m = row.get(metric_key)
        if not isinstance(m, dict):
            raise ValueError(f"Head row missing metric '{metric_key}'")
        for sk in ("d0", "d_pos", "d_neg", "even", "odd"):
            if not _is_finite_number(m.get(sk)):
                raise ValueError(f"Head row missing/non-finite {metric_key}.{sk}")
        # Verify derived even/odd from d0, d_pos, d_neg within this metric.
        d0 = float(m["d0"])
        dp = float(m["d_pos"])
        dn = float(m["d_neg"])
        if abs(float(m["even"]) - (0.5 * (dp + dn) - d0)) > DECOMP_TOL:
            raise ValueError(f"Derived summary mismatch on {metric_key}.even")
        if abs(float(m["odd"]) - (0.5 * (dp - dn))) > DECOMP_TOL:
            raise ValueError(f"Derived summary mismatch on {metric_key}.odd")
        if metric_key in ("spec", "raw"):
            if "max_abs_sham" not in m or not _is_finite_number(m.get("max_abs_sham")):
                raise ValueError(f"Head row missing/non-finite {metric_key}.max_abs_sham on {row.get('head_name')}")
        for grp in ("wdown_orth", "iso_resid"):
            g = m.get(grp)
            if not isinstance(g, dict):
                raise ValueError(f"Head row missing {metric_key}.{grp}")
            for dk in ("d_pos_draws", "d_neg_draws", "even_draws", "odd_draws"):
                arr = g.get(dk)
                if not isinstance(arr, list) or len(arr) != rib.N_NULL_DRAWS or not all(_is_finite_number(v) for v in arr):
                    raise ValueError(f"Invalid {metric_key}.{grp}.{dk}")
            for mk in ("d_pos_med", "d_neg_med", "even_med", "odd_med"):
                if not _is_finite_number(g.get(mk)):
                    raise ValueError(f"Invalid {metric_key}.{grp}.{mk}")
            # Recompute this metric's OWN draw-based even/odd draws and medians (Alden #104 item 1).
            # Deliberately per-metric (never subtracting separately reduced medians across metrics!).
            dp_d = [float(x) for x in g["d_pos_draws"]]
            dn_d = [float(x) for x in g["d_neg_draws"]]
            ev_d = [float(x) for x in g["even_draws"]]
            od_d = [float(x) for x in g["odd_draws"]]
            for k in range(rib.N_NULL_DRAWS):
                if abs(ev_d[k] - (0.5 * (dp_d[k] + dn_d[k]) - d0)) > DECOMP_TOL:
                    raise ValueError(f"Derived null summary mismatch on {metric_key}.{grp}.even_draws[{k}]")
                if abs(od_d[k] - (0.5 * (dp_d[k] - dn_d[k]))) > DECOMP_TOL:
                    raise ValueError(f"Derived null summary mismatch on {metric_key}.{grp}.odd_draws[{k}]")
            for mk, draws in (
                ("d_pos_med", dp_d),
                ("d_neg_med", dn_d),
                ("even_med", ev_d),
                ("odd_med", od_d),
            ):
                exp_med = float(statistics.median(draws))
                if abs(float(g[mk]) - exp_med) > DECOMP_TOL:
                    raise ValueError(
                        f"Derived null summary mismatch on {metric_key}.{grp}.{mk}: "
                        f"stored={g[mk]}, recomputed={exp_med}"
                    )
            if "even_max_abs" in g:
                if not _is_finite_number(g["even_max_abs"]) or abs(float(g["even_max_abs"]) - max(abs(x) for x in ev_d)) > DECOMP_TOL:
                    raise ValueError(f"Derived null summary mismatch on {metric_key}.{grp}.even_max_abs")
            if "odd_max_abs" in g:
                if not _is_finite_number(g["odd_max_abs"]) or abs(float(g["odd_max_abs"]) - max(abs(x) for x in od_d)) > DECOMP_TOL:
                    raise ValueError(f"Derived null summary mismatch on {metric_key}.{grp}.odd_max_abs")

    sham_spec = float(row["spec"]["max_abs_sham"])
    sham_raw = float(row["raw"]["max_abs_sham"])
    if sham_spec > SHAM_TOL or sham_raw > SHAM_TOL:
        raise ValueError(
            f"Head sham exceeded {SHAM_TOL}: spec={sham_spec}, raw={sham_raw} on "
            f"{(row.get('bank'), row.get('norm_cond'), row.get('unit'), row.get('prompt_id'), row.get('head_name'))}"
        )

    decomp_err = rib.check_head_linear_decomposition(row["spec"], row["raw"], row["peer_med"])

    derived_frac_pos: float | None = None
    if expected_peers is not None:
        pd = row.get("peer_deltas")
        exp_keys = {str(tid) for tid in expected_peers}
        if not isinstance(pd, dict) or set(pd.keys()) != exp_keys:
            raise ValueError(
                f"Required 19 peer_deltas IDs missing/mismatched on "
                f"{(row.get('bank'), row.get('norm_cond'), row.get('unit'), row.get('prompt_id'), row.get('head_name'))}"
            )
        dpos_vals: list[float] = []
        for tk in exp_keys:
            ent = pd[tk]
            if not isinstance(ent, dict) or not _is_finite_number(ent.get("d_pos")) or not _is_finite_number(ent.get("d_neg")):
                raise ValueError(f"Invalid peer_deltas[{tk}] entry")
            dpos_vals.append(float(ent["d_pos"]))
        derived_frac_pos = float(sum(1 for v in dpos_vals if v > 0.0) / len(dpos_vals))
        derived_mean_dp = float(sum(dpos_vals) / len(dpos_vals))
        if "frac_peer_tokens_pos" in row:
            if not _is_finite_number(row["frac_peer_tokens_pos"]) or abs(float(row["frac_peer_tokens_pos"]) - derived_frac_pos) > DECOMP_TOL:
                raise ValueError("Stored frac_peer_tokens_pos disagrees with derived peer_deltas fraction")
        if "peer_mean_d_pos" in row:
            if not _is_finite_number(row["peer_mean_d_pos"]) or abs(float(row["peer_mean_d_pos"]) - derived_mean_dp) > DECOMP_TOL:
                raise ValueError("Stored peer_mean_d_pos disagrees with derived peer_deltas mean")

    return sham_spec, sham_raw, decomp_err, derived_frac_pos


def bind_and_validate_rows(
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    expect_full: bool,
    pins: dict[str, str] | None = None,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Bind receipt meta, timing, provenance, and every row canonically BEFORE grouping."""
    meta = receipt.get("meta", {})
    exp_type = "instruct_bridge_full_receipt" if expect_full else "instruct_bridge_smoke_receipt"
    exp_mode = "full" if expect_full else "smoke_1unit"
    exp_units = list(rib.CORE_UNIT_IDS) if expect_full else [rib.SMOKE_UNIT_ID]
    den = manifest["meta"]["denominators"]["full" if expect_full else "smoke"]

    if meta.get("artifact_type") != exp_type or meta.get("complete") is not True:
        raise ValueError(
            f"Refused receipt: artifact_type={meta.get('artifact_type')!r}, complete={meta.get('complete')!r}, "
            f"expected {exp_type!r} with complete=True"
        )
    if meta.get("schema_version") != rib.SCHEMA_VERSION:
        raise ValueError(f"Receipt schema_version mismatch: {meta.get('schema_version')!r}")
    if meta.get("run_mode") != exp_mode or meta.get("smoke_1unit") is not (not expect_full):
        raise ValueError(f"Receipt run_mode/smoke_1unit mismatch: {meta.get('run_mode')!r}, {meta.get('smoke_1unit')!r}")
    if meta.get("model_repo") != rib.MODEL_REPO or meta.get("model_commit") != rib.MODEL_COMMIT:
        raise ValueError(f"Receipt model_repo/model_commit mismatch: {meta.get('model_repo')!r}@{meta.get('model_commit')!r}")
    if meta.get("dtype") != rib.DTYPE_STR or meta.get("attn_implementation") != rib.ATTN_IMPL:
        raise ValueError(f"Receipt dtype/attn_implementation mismatch")
    if meta.get("units_run") != exp_units:
        raise ValueError(f"Receipt units_run mismatch: {meta.get('units_run')!r} != {exp_units!r}")
    if meta.get("n_unpatched_rows") != den["n_unpatched_rows"]:
        raise ValueError(f"Receipt meta n_unpatched_rows={meta.get('n_unpatched_rows')} != {den['n_unpatched_rows']}")
    if meta.get("n_head_rows") != den["n_head_rows"]:
        raise ValueError(f"Receipt meta n_head_rows={meta.get('n_head_rows')} != {den['n_head_rows']}")
    if meta.get("n_b19_forwards") != den["n_b19_forwards"]:
        raise ValueError(f"Receipt meta n_b19_forwards={meta.get('n_b19_forwards')} != {den['n_b19_forwards']}")
    if "max_abs_null_vec_diff" not in meta or not _is_finite_number(meta["max_abs_null_vec_diff"]) or float(meta["max_abs_null_vec_diff"]) != 0.0:
        raise ValueError(f"Missing or non-zero max_abs_null_vec_diff in receipt meta: {meta.get('max_abs_null_vec_diff')!r}")
    if pins is not None:
        if meta.get("pins") != pins:
            raise ValueError(f"Receipt embedded pins mismatch: {meta.get('pins')} != {pins}")
        for pk, pv in pins.items():
            if meta.get(pk) != pv:
                raise ValueError(f"Receipt meta.{pk} mismatch: {meta.get(pk)} != {pv}")

    # Validate execution provenance & finite measured timing breakdown (Alden #104 item 2).
    prov = meta.get("execution_provenance")
    if not allow_fixture:
        if prov != "measured_gpu_run":
            raise ValueError(
                f"Receipt execution_provenance={prov!r} refused for launch/study verification "
                "(requires 'measured_gpu_run'; fixture-only verification must set allow_fixture=True)."
            )
    else:
        if prov not in ("measured_gpu_run", "cpu_or_dummy_run", "synthetic_anchor_fixture"):
            raise ValueError(f"Invalid execution_provenance: {prov!r}")
    rib.validate_timing_breakdown(meta.get("timing"), den["n_b19_forwards"])

    units = exp_units
    schedule = manifest["schedule_full"] if expect_full else manifest["schedule_smoke"]
    norms_map = manifest["meta"]["norms"]

    # Build expected canonical key -> expected metadata maps.
    exp_unp: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    exp_hd: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for uid in units:
        u = rib.UNIT_BY_ID[uid]
        for e in schedule:
            dk = rib.dose_key(e["dose_mag"])
            bkey = f"{e['bank']}|{e['prompt_id']}"
            peer_ids = manifest["bindings"][bkey][uid]["peers"]
            for nc in rib.NORM_ORDER:
                uk = (e["bank"], nc, uid, e["prompt_id"], dk)
                exp_unp[uk] = {
                    "unit_layer": u["layer"],
                    "cell": e["cell"],
                    "story_or_frame": e["story_or_frame"],
                    "block_a": e["block_a"],
                    "head_patched": bool(e["head_patched"]),
                    "norm_multiplier": float(norms_map[nc][str(u["layer"])]),
                }
                if e["head_patched"]:
                    for h in rib.CANONICAL_HEADS:
                        hk = (e["bank"], nc, uid, e["prompt_id"], dk, h["name"])
                        exp_hd[hk] = {
                            "unit_layer": u["layer"],
                            "cell": e["cell"],
                            "story_or_frame": e["story_or_frame"],
                            "block_a": e["block_a"],
                            "head_layer": h["layer"],
                            "head_idx": h["head"],
                            "head_kind": h["kind"],
                            "expected_peers": peer_ids if (bool(e["record_peer_deltas"]) and h["name"] in rib.PEER_DELTA_HEADS) else None,
                        }

    unp_rows = receipt.get("unpatched_rows", [])
    hd_rows = receipt.get("head_rows", [])
    if len(unp_rows) != len(exp_unp):
        raise ValueError(f"Unpatched row count mismatch: got {len(unp_rows)}, expected {len(exp_unp)}")
    if len(hd_rows) != len(exp_hd):
        raise ValueError(f"Head row count mismatch: got {len(hd_rows)}, expected {len(exp_hd)}")

    unp_by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    max_decomp = 0.0
    basis_sha_by_tuple: dict[tuple[str, str, str, str], dict[str, str]] = {}

    for r in unp_rows:
        dk = rib.dose_key(r["dose_mag"])
        uk = (r["bank"], r["norm_cond"], r["unit"], r["prompt_id"], dk)
        if uk not in exp_unp:
            raise ValueError(f"Unexpected unpatched row key: {uk}")
        if uk in unp_by_key:
            raise ValueError(f"Duplicate unpatched row key: {uk}")
        em = exp_unp[uk]
        for fk in ("unit_layer", "cell", "story_or_frame", "block_a", "head_patched"):
            if r.get(fk) != em[fk]:
                raise ValueError(f"Unpatched metadata tamper on {uk}.{fk}: {r.get(fk)!r} != {em[fk]!r}")
        if abs(float(r["norm_multiplier"]) - em["norm_multiplier"]) > 1e-12:
            raise ValueError(f"Unpatched norm_multiplier tamper on {uk}")
        max_decomp = max(max_decomp, validate_unpatched_row_integrity(r))
        bsha = r.get("null_unit_vec_sha256")
        if not isinstance(bsha, str) or len(bsha) != 64:
            raise ValueError(f"Missing/invalid null_unit_vec_sha256 on {uk}")
        basis_sha_by_tuple.setdefault((r["bank"], r["unit"], r["prompt_id"], dk), {})[r["norm_cond"]] = bsha
        unp_by_key[uk] = r

    # Verify identical fixed null basis across OLD_NORM and NEW_NORM for every (bank, unit, prompt_id, dose_mag).
    for tk, nmap in basis_sha_by_tuple.items():
        if set(nmap.keys()) != set(rib.NORM_ORDER) or nmap["OLD_NORM"] != nmap["NEW_NORM"]:
            raise ValueError(f"Fixed null basis mismatch across norms on {tk}: {nmap}")

    hd_by_key: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    derived_frac_pos_by_key: dict[tuple[str, str, str, str, str, str], float] = {}
    max_sham_spec = 0.0
    max_sham_raw = 0.0
    for r in hd_rows:
        dk = rib.dose_key(r["dose_mag"])
        hk = (r["bank"], r["norm_cond"], r["unit"], r["prompt_id"], dk, r["head_name"])
        if hk not in exp_hd:
            raise ValueError(f"Unexpected head row key (e.g. CAL_SELF or extra): {hk}")
        if hk in hd_by_key:
            raise ValueError(f"Duplicate head row key: {hk}")
        em = exp_hd[hk]
        for fk in ("unit_layer", "cell", "story_or_frame", "block_a", "head_layer", "head_idx", "head_kind"):
            if r.get(fk) != em[fk]:
                raise ValueError(f"Head metadata tamper on {hk}.{fk}: {r.get(fk)!r} != {em[fk]!r}")
        if r.get("null_unit_vec_sha256") != basis_sha_by_tuple[(r["bank"], r["unit"], r["prompt_id"], dk)][r["norm_cond"]]:
            raise ValueError(f"Head row null_unit_vec_sha256 mismatch with unpatched row on {hk}")
        ss, sr, de, dfp = validate_head_row_integrity(r, expected_peers=em["expected_peers"])
        max_sham_spec = max(max_sham_spec, ss)
        max_sham_raw = max(max_sham_raw, sr)
        max_decomp = max(max_decomp, de)
        if dfp is not None:
            derived_frac_pos_by_key[hk] = dfp
        hd_by_key[hk] = r

    return {
        "unp_by_key": unp_by_key,
        "hd_by_key": hd_by_key,
        "derived_frac_pos_by_key": derived_frac_pos_by_key,
        "execution_provenance": prov,
        "max_abs_sham_spec": max_sham_spec,
        "max_abs_sham_raw": max_sham_raw,
        "max_abs_decomp_err": max_decomp,
        "null_basis_pairs_verified": len(basis_sha_by_tuple),
    }


# ---------------------------------------------------------------------------
# BR.0 — Separate Exact Reference Receipt Anchor Joins
# ---------------------------------------------------------------------------

def evaluate_br0_anchors(
    bound: dict[str, Any],
    manifest: dict[str, Any],
    old_receipt_path: Path,
    new_receipt_path: Path,
    expect_full: bool,
) -> dict[str, Any]:
    """Verify full hashes of both reference receipts and run exact separate anchor joins."""
    old_bytes = old_receipt_path.read_bytes()
    new_bytes = new_receipt_path.read_bytes()
    old_sha = rib.sha256_bytes(old_bytes)
    new_sha = rib.sha256_bytes(new_bytes)
    if old_sha != rib.SOURCE_PINS["old_receipt_sha256"]:
        raise ValueError(f"OLD_RECEIPT SHA-256 mismatch: {old_sha} != {rib.SOURCE_PINS['old_receipt_sha256']}")
    if new_sha != rib.SOURCE_PINS["new_receipt_sha256"]:
        raise ValueError(f"NEW_RECEIPT SHA-256 mismatch: {new_sha} != {rib.SOURCE_PINS['new_receipt_sha256']}")

    old_rcpt = json.loads(old_bytes.decode("utf-8"))
    new_rcpt = json.loads(new_bytes.decode("utf-8"))

    # Per-source literal check on NEW_RECEIPT["unit_vectors"] (map_key, layer, neuron, cal_resid_median).
    uvecs = new_rcpt.get("unit_vectors")
    if not isinstance(uvecs, dict) or set(uvecs.keys()) != set(rib.CORE_UNIT_IDS):
        raise ValueError("NEW_RECEIPT unit_vectors keys mismatch during BR.0")
    for u in rib.CANONICAL_CORE_UNITS:
        uv = uvecs[u["id"]]
        if int(uv["layer"]) != u["layer"] or int(uv["neuron"]) != u["neuron"]:
            raise ValueError(f"NEW_RECEIPT unit_vectors[{u['id']}] coordinate drift")
        if float(uv["cal_resid_median"]) != float(rib.CANONICAL_NORMS["NEW_NORM"][str(u["layer"])]):
            raise ValueError(f"NEW_RECEIPT unit_vectors[{u['id']}] norm drift")

    # Index OLD_RECEIPT["q12_rows"] on exact join key (unit, prompt_id, cell, dose_key, head_name).
    old_idx: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for r in old_rcpt["q12_rows"]:
        k = (r["unit"], r["prompt_id"], r["cell"], rib.dose_key(r["dose_mag"]), r["head_name"])
        if k in old_idx:
            raise ValueError(f"Duplicate key in OLD_RECEIPT q12_rows: {k}")
        old_idx[k] = r

    # Index NEW_RECEIPT head & unpatched tables separately.
    new_hd_idx: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for tbl in ("eval_arm_a_head_rows", "cal_head_rows"):
        for r in new_rcpt[tbl]:
            k = (tbl, r["unit"], r["prompt_id"], r["cell"], rib.dose_key(r["dose_mag"]), r["head_name"])
            if k in new_hd_idx:
                raise ValueError(f"Duplicate key in NEW_RECEIPT {tbl}: {k}")
            new_hd_idx[k] = r

    new_unp_idx: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for tbl in ("eval_arm_a_unpatched_rows", "cal_unpatched_rows"):
        for r in new_rcpt[tbl]:
            k = (tbl, r["unit"], r["prompt_id"], r["cell"], rib.dose_key(r["dose_mag"]))
            if k in new_unp_idx:
                raise ValueError(f"Duplicate key in NEW_RECEIPT {tbl}: {k}")
            new_unp_idx[k] = r

    # Join 1: OLD_BANK x OLD_NORM head rows (480 full / 24 smoke; 0 unpatched rows).
    old_checked = 0
    old_max_diff = 0.0
    old_worst_key = None
    old_non_anchor_unp = 0
    old_non_anchor_l7h9 = 0
    old_non_anchor_d12 = 0

    for r in bound["unp_by_key"].values():
        if r["bank"] == "OLD_BANK" and r["norm_cond"] == "OLD_NORM":
            old_non_anchor_unp += 1

    for r in bound["hd_by_key"].values():
        if r["bank"] == "OLD_BANK" and r["norm_cond"] == "OLD_NORM":
            dk = rib.dose_key(r["dose_mag"])
            if r["head_name"] not in rib.OLD_OVERLAP_HEADS:
                old_non_anchor_l7h9 += 1
                continue
            if dk == "1.2":
                old_non_anchor_d12 += 1
                continue
            jk = (r["unit"], r["prompt_id"], r["cell"], dk, r["head_name"])
            if jk not in old_idx:
                raise ValueError(f"OLD anchor key missing from OLD_RECEIPT: {jk}")
            ref = old_idx[jk]
            for mk in ("spec", "raw"):
                for fld in ("d_pos", "d_neg"):
                    diff = abs(float(r[mk][fld]) - float(ref[mk][fld]))
                    if diff > old_max_diff:
                        old_max_diff = diff
                        old_worst_key = (jk, f"{mk}.{fld}", diff)
            old_checked += 1

    # Join 2a: NEW_BANK x NEW_NORM head rows (910 full / 21 smoke).
    new_hd_checked = 0
    new_hd_max_diff = 0.0
    new_hd_worst_key = None
    for r in bound["hd_by_key"].values():
        if r["bank"] == "NEW_BANK" and r["norm_cond"] == "NEW_NORM":
            dk = rib.dose_key(r["dose_mag"])
            tbl = "eval_arm_a_head_rows" if r["prompt_id"].startswith("eval_") else "cal_head_rows"
            jk = (tbl, r["unit"], r["prompt_id"], r["cell"], dk, r["head_name"])
            if jk not in new_hd_idx:
                raise ValueError(f"NEW head anchor key missing from NEW_RECEIPT: {jk}")
            ref = new_hd_idx[jk]
            for mk in ("spec", "raw"):
                for fld in ("d_pos", "d_neg"):
                    diff = abs(float(r[mk][fld]) - float(ref[mk][fld]))
                    if diff > new_hd_max_diff:
                        new_hd_max_diff = diff
                        new_hd_worst_key = (jk, f"{mk}.{fld}", diff)
            new_hd_checked += 1

    # Join 2b: NEW_BANK x NEW_NORM unpatched transport rows (160 full / 4 smoke).
    new_unp_checked = 0
    new_unp_max_diff = 0.0
    new_unp_worst_key = None
    for r in bound["unp_by_key"].values():
        if r["bank"] == "NEW_BANK" and r["norm_cond"] == "NEW_NORM":
            dk = rib.dose_key(r["dose_mag"])
            tbl = "eval_arm_a_unpatched_rows" if r["prompt_id"].startswith("eval_") else "cal_unpatched_rows"
            jk = (tbl, r["unit"], r["prompt_id"], r["cell"], dk)
            if jk not in new_unp_idx:
                raise ValueError(f"NEW unpatched anchor key missing from NEW_RECEIPT: {jk}")
            ref = new_unp_idx[jk]
            for fld in ("n_write_spec", "n_write_raw", "n_write_neg_spec", "n_write_neg_raw"):
                diff = abs(float(r["transport"][fld]) - float(ref["transport"][fld]))
                if diff > new_unp_max_diff:
                    new_unp_max_diff = diff
                    new_unp_worst_key = (jk, fld, diff)
            new_unp_checked += 1

    den = manifest["meta"]["denominators"]["full" if expect_full else "smoke"]
    counts_ok = (
        old_checked == den["old_anchor_head_rows"]
        and new_hd_checked == den["new_anchor_head_rows"]
        and new_unp_checked == den["new_anchor_unpatched_rows"]
    )
    tols_ok = (
        old_max_diff <= ANCHOR_TOL
        and new_hd_max_diff <= ANCHOR_TOL
        and new_unp_max_diff <= ANCHOR_TOL
    )
    passed = bool(counts_ok and tols_ok)
    return {
        "status": "passed" if passed else "failed",
        "tolerance": ANCHOR_TOL,
        "old_receipt_sha256": old_sha,
        "new_receipt_sha256": new_sha,
        "join_1_old_head": {
            "n_rows_joined": old_checked,
            "expected_rows": den["old_anchor_head_rows"],
            "max_abs_diff": old_max_diff,
            "passed": bool(old_checked == den["old_anchor_head_rows"] and old_max_diff <= ANCHOR_TOL),
            "worst_key": str(old_worst_key) if old_worst_key else None,
            "non_anchor_segregated": {
                "old_unpatched_rows_new_measurement": old_non_anchor_unp,
                "old_l7h9_head_rows": old_non_anchor_l7h9,
                "old_d12_overlap_head_rows": old_non_anchor_d12,
            },
        },
        "join_2a_new_head": {
            "n_rows_joined": new_hd_checked,
            "expected_rows": den["new_anchor_head_rows"],
            "max_abs_diff": new_hd_max_diff,
            "passed": bool(new_hd_checked == den["new_anchor_head_rows"] and new_hd_max_diff <= ANCHOR_TOL),
            "worst_key": str(new_hd_worst_key) if new_hd_worst_key else None,
        },
        "join_2b_new_unpatched": {
            "n_rows_joined": new_unp_checked,
            "expected_rows": den["new_anchor_unpatched_rows"],
            "max_abs_diff": new_unp_max_diff,
            "passed": bool(new_unp_checked == den["new_anchor_unpatched_rows"] and new_unp_max_diff <= ANCHOR_TOL),
            "worst_key": str(new_unp_worst_key) if new_unp_worst_key else None,
        },
    }


# ---------------------------------------------------------------------------
# BR.1, BR.2, BR.3, BR.4 Total Machine-Readable Endpoints
# ---------------------------------------------------------------------------

def _cell_transport_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ms = _mean([float(r["transport"]["n_write_spec"]) for r in rows])
    mr = _mean([float(r["transport"]["n_write_raw"]) for r in rows])
    mp = _mean([float(r["transport"]["n_write_peer_med"]) for r in rows])
    null_means = [
        _mean([float(r["transport"]["wdown_pos_draws_spec"][k]) for r in rows])
        for k in range(rib.N_NULL_DRAWS)
    ]
    max_null = max(null_means)
    passed = bool(ms >= 0.50 and mr >= 0.50 and ms > max_null)
    return {
        "n_rows": len(rows),
        "mean_spec": ms,
        "mean_raw": mr,
        "mean_peer_med": mp,
        "max_wdown_null_spec": max_null,
        "cell_transport_pass": passed,
    }


def _cell_head_null_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ev = _mean([float(r["spec"]["wdown_orth"]["even_med"]) for r in rows])
    od = _mean([float(r["spec"]["wdown_orth"]["odd_med"]) for r in rows])
    passed = bool(abs(ev) <= 0.15 and abs(od) <= 0.15)
    return {
        "n_rows": len(rows),
        "mean_wdown_even_med_spec": ev,
        "mean_wdown_odd_med_spec": od,
        "head_null_pass": passed,
    }


def evaluate_br1(
    unp_rows: list[dict[str, Any]],
    hd_rows: list[dict[str, Any]],
    br0_passed: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for bank in ("OLD_BANK", "NEW_BANK"):
        core_cells = CORE_CELLS_BY_BANK[bank]
        for nc in rib.NORM_ORDER:
            ckey = f"{bank}|{nc}"
            t_by_cell: dict[str, Any] = {}
            for cell in core_cells:
                sub = [
                    r for r in unp_rows
                    if r["bank"] == bank and r["norm_cond"] == nc and r["cell"] == cell and rib.dose_key(r["dose_mag"]) == "0.8"
                ]
                t_by_cell[cell] = _cell_transport_stats(sub)
            t_valid = all(t_by_cell[c]["cell_transport_pass"] for c in core_cells)

            h_by_head: dict[str, Any] = {}
            for hname in rib.HEAD_NAMES:
                h_cells: dict[str, Any] = {}
                for cell in core_cells:
                    sub_h = [
                        r for r in hd_rows
                        if r["bank"] == bank and r["norm_cond"] == nc and r["cell"] == cell
                        and rib.dose_key(r["dose_mag"]) == "0.8" and r["head_name"] == hname
                    ]
                    h_cells[cell] = _cell_head_null_stats(sub_h)
                hn_valid = all(h_cells[c]["head_null_pass"] for c in core_cells)
                if not br0_passed:
                    verdict = ANCHOR_FAIL_SCI
                elif not t_valid:
                    bad_t = [c for c in core_cells if not t_by_cell[c]["cell_transport_pass"]]
                    verdict = f"uninformative (transport_failed:{','.join(bad_t)})"
                elif not hn_valid:
                    bad_h = [c for c in core_cells if not h_cells[c]["head_null_pass"]]
                    verdict = f"uninformative (head_null_failed:{','.join(bad_h)})"
                else:
                    verdict = "valid"
                h_by_head[hname] = {
                    "verdict": verdict,
                    "head_null_valid": hn_valid,
                    "cells": h_cells,
                }

            out[ckey] = {
                "transport_valid": t_valid,
                "transport_cells": t_by_cell,
                "heads": h_by_head,
            }
    return out


def classify_br2(
    br0_passed: bool,
    br1_valid_map: dict[str, bool],
    g_oo: float,
    g_on: float,
    g_no: float,
    g_nn: float,
) -> dict[str, Any]:
    """Exhaustive 7-branch BR.2 decision partition (v3 §BR.2)."""
    d_norm_old = g_on - g_oo
    d_norm_new = g_nn - g_no
    d_bank_old = g_no - g_oo
    d_bank_new = g_nn - g_on
    did = d_norm_new - d_norm_old

    if not br0_passed:
        verdict = ANCHOR_FAIL_SCI
    else:
        failed_cells = [k for k in ("OLD_BANK|OLD_NORM", "OLD_BANK|NEW_NORM", "NEW_BANK|OLD_NORM", "NEW_BANK|NEW_NORM") if not br1_valid_map[k]]
        if failed_cells:
            verdict = f"uninformative (cell_gate_failed:{','.join(failed_cells)})"
        elif (
            d_norm_old >= 0.25 and d_norm_new >= 0.25 and g_no <= -0.30
            and abs(d_bank_old) <= 0.20 and abs(d_bank_new) <= 0.20
        ):
            verdict = "norm_dominant"
        elif (
            d_bank_old >= 0.30 and d_bank_new >= 0.30
            and abs(d_norm_old) <= 0.15 and abs(d_norm_new) <= 0.15
        ):
            verdict = "bank_dominant"
        elif abs(did) >= 0.25:
            verdict = "interaction_present"
        elif (abs(d_norm_old) >= 0.20 or abs(d_norm_new) >= 0.20) and (abs(d_bank_old) >= 0.20 or abs(d_bank_new) >= 0.20):
            verdict = "additive_mixed"
        else:
            verdict = "neither_or_inconclusive"

    return {
        "verdict": verdict,
        "G_OO": g_oo,
        "G_ON": g_on,
        "G_NO": g_no,
        "G_NN": g_nn,
        "d_norm_OLD": d_norm_old,
        "d_norm_NEW": d_norm_new,
        "d_bank_OLD": d_bank_old,
        "d_bank_NEW": d_bank_new,
        "DiD": did,
    }


def evaluate_br2(
    hd_rows: list[dict[str, Any]],
    br1: dict[str, Any],
    br0_passed: bool,
    derived_frac_pos_by_key: dict[tuple[str, str, str, str, str, str], float],
) -> dict[str, Any]:
    def cell_mean(bank: str, nc: str, cell: str, hname: str, mk: str) -> float:
        sub = [
            float(r[mk]["d_pos"]) for r in hd_rows
            if r["bank"] == bank and r["norm_cond"] == nc and r["cell"] == cell
            and rib.dose_key(r["dose_mag"]) == "0.8" and r["head_name"] == hname
        ]
        return _mean(sub)

    def g_for(hname: str, mk: str) -> tuple[float, float, float, float]:
        g_oo = cell_mean("OLD_BANK", "OLD_NORM", "ROLE_DEFINITE_NEW", hname, mk) - cell_mean("OLD_BANK", "OLD_NORM", "NAMED_PERSON_NEW", hname, mk)
        g_on = cell_mean("OLD_BANK", "NEW_NORM", "ROLE_DEFINITE_NEW", hname, mk) - cell_mean("OLD_BANK", "NEW_NORM", "NAMED_PERSON_NEW", hname, mk)
        g_no = cell_mean("NEW_BANK", "OLD_NORM", "ROLE_BARE", hname, mk) - cell_mean("NEW_BANK", "OLD_NORM", "NAME_BARE", hname, mk)
        g_nn = cell_mean("NEW_BANK", "NEW_NORM", "ROLE_BARE", hname, mk) - cell_mean("NEW_BANK", "NEW_NORM", "NAME_BARE", hname, mk)
        return g_oo, g_on, g_no, g_nn

    valid_l8h15 = {k: (br1[k]["heads"]["L8H15"]["verdict"] == "valid") for k in br1}
    primary = classify_br2(br0_passed, valid_l8h15, *g_for("L8H15", "spec"))

    # Parallel decomposed tables for L8H15, L8H2, L8H3, L8H13.
    l8_tables: dict[str, Any] = {}
    for hname in ("L8H15", "L8H2", "L8H3", "L8H13"):
        ht: dict[str, Any] = {}
        for mk in ("spec", "raw", "peer_med"):
            g_oo, g_on, g_no, g_nn = g_for(hname, mk)
            c_sr_oo = cell_mean("OLD_BANK", "OLD_NORM", "SELF_HELD_NEW", hname, mk) - cell_mean("OLD_BANK", "OLD_NORM", "ROLE_DEFINITE_NEW", hname, mk)
            c_sr_on = cell_mean("OLD_BANK", "NEW_NORM", "SELF_HELD_NEW", hname, mk) - cell_mean("OLD_BANK", "NEW_NORM", "ROLE_DEFINITE_NEW", hname, mk)
            c_sr_no = cell_mean("NEW_BANK", "OLD_NORM", "SELF", hname, mk) - cell_mean("NEW_BANK", "OLD_NORM", "ROLE_BARE", hname, mk)
            c_sr_nn = cell_mean("NEW_BANK", "NEW_NORM", "SELF", hname, mk) - cell_mean("NEW_BANK", "NEW_NORM", "ROLE_BARE", hname, mk)
            g_alias_no = cell_mean("NEW_BANK", "OLD_NORM", "ALIAS_TO_ROLE", hname, mk) - cell_mean("NEW_BANK", "OLD_NORM", "ALIAS_TO_NAME", hname, mk)
            g_alias_nn = cell_mean("NEW_BANK", "NEW_NORM", "ALIAS_TO_ROLE", hname, mk) - cell_mean("NEW_BANK", "NEW_NORM", "ALIAS_TO_NAME", hname, mk)
            ht[mk] = {
                "G_OO": g_oo, "G_ON": g_on, "G_NO": g_no, "G_NN": g_nn,
                "C_SR_OO": c_sr_oo, "C_SR_ON": c_sr_on, "C_SR_NO": c_sr_no, "C_SR_NN": c_sr_nn,
                "G_alias_NO": g_alias_no, "G_alias_NN": g_alias_nn,
            }
        if hname in rib.PEER_DELTA_HEADS:
            # Derive mean_frac_peer_tokens_pos strictly across 100% of required d=0.8 rows (never skip missing!).
            fp_summary: dict[str, dict[str, float]] = {}
            for bank, cells, exp_n in (
                ("OLD_BANK", ("SELF_HELD_NEW", "ROLE_DEFINITE_NEW", "NAMED_PERSON_NEW", "THIRD_GENERIC_NEW"), 15),
                ("NEW_BANK", ("SELF", "ROLE_BARE", "NAME_BARE", "ALIAS_TO_ROLE", "ALIAS_TO_NAME"), 20),
            ):
                for nc in rib.NORM_ORDER:
                    fp_cells: dict[str, float] = {}
                    for cell in cells:
                        cell_rows = [
                            r for r in hd_rows
                            if r["bank"] == bank and r["norm_cond"] == nc and r["cell"] == cell
                            and rib.dose_key(r["dose_mag"]) == "0.8" and r["head_name"] == hname
                        ]
                        if len(cell_rows) != exp_n:
                            raise ValueError(f"Unexpected row count for peer_deltas on {(bank, nc, cell, hname)}: {len(cell_rows)} != {exp_n}")
                        vals = [
                            derived_frac_pos_by_key[(r["bank"], r["norm_cond"], r["unit"], r["prompt_id"], rib.dose_key(r["dose_mag"]), r["head_name"])]
                            for r in cell_rows
                        ]
                        fp_cells[cell] = _mean(vals)
                    fp_summary[f"{bank}|{nc}"] = fp_cells
            ht["mean_frac_peer_tokens_pos"] = fp_summary
        l8_tables[hname] = ht

    return {
        "primary_L8H15_spec": primary,
        "l8_decomposed": l8_tables,
    }


def classify_br3_states(
    br0_passed: bool,
    p_old: bool,
    p_new: bool,
    c_old: bool,
    c_new: bool,
) -> tuple[str, str]:
    """Orthogonal 4-state disjoint `gate_state` and `collapse_state` (v3 §BR.3)."""
    if not br0_passed:
        return ANCHOR_FAIL_SCI, ANCHOR_FAIL_SCI
    if p_old and p_new:
        gs = "both_pass"
    elif p_old and (not p_new):
        gs = "old_pass_new_miss"
    elif (not p_old) and p_new:
        gs = "old_miss_new_pass"
    else:
        gs = "both_miss"

    if (not c_old) and (not c_new):
        cs = "neither_collapsed"
    elif (not c_old) and c_new:
        cs = "new_only_collapsed"
    elif c_old and (not c_new):
        cs = "old_only_collapsed"
    else:
        cs = "both_collapsed"
    return gs, cs


def evaluate_br3(
    unp_rows: list[dict[str, Any]],
    hd_rows: list[dict[str, Any]],
    br0_passed: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "design_note": "3-dose high-dose check on {0.8, 1.2, 1.6}, strictly within each bank (NOT the original 6-dose d_cal).",
        "banks": {},
    }

    for bank in ("OLD_BANK", "NEW_BANK"):
        pair_cells = HIGH_DOSE_PAIR_BY_BANK[bank]
        third_cell = pair_cells[1]
        # Filter OLD_BANK strictly to block_a F1/F2 prompts so {0.8, 1.2, 1.6} share identical 2 prompts!
        allowed_pids = (
            {"self_F1", "self_F2", "third_F1", "third_F2"}
            if bank == "OLD_BANK"
            else {"cal_self_C1", "cal_self_C2", "cal_third_C1", "cal_third_C2"}
        )

        cells_by_norm_dose: dict[str, dict[str, Any]] = {}
        pair_status: dict[str, dict[str, Any]] = {}
        rebound_l8h15: dict[str, Any] = {}

        for nc in rib.NORM_ORDER:
            cells_by_norm_dose[nc] = {}
            pair_status[nc] = {}
            base_raw_by_cell: dict[str, float] = {}

            for dk in ("0.8", "1.2", "1.6"):
                c_stats: dict[str, Any] = {}
                for cell in pair_cells:
                    sub = [
                        r for r in unp_rows
                        if r["bank"] == bank and r["norm_cond"] == nc and r["cell"] == cell
                        and r["prompt_id"] in allowed_pids and rib.dose_key(r["dose_mag"]) == dk
                    ]
                    st = _cell_transport_stats(sub)
                    if dk == "0.8":
                        base_raw_by_cell[cell] = st["mean_raw"]
                        st["cell_collapsed"] = None  # Explicitly null (not_applicable) at d=0.8 (Alden #93 item 3).
                    else:
                        braw = base_raw_by_cell[cell]
                        st["cell_collapsed"] = bool(
                            (st["mean_raw"] <= 0.0) or (braw >= 0.50 and st["mean_raw"] <= 0.25 * braw)
                        )
                    if bank == "OLD_BANK" and nc == "OLD_NORM":
                        st["provenance_tag"] = "new_measurement_not_anchor"
                    c_stats[cell] = st
                cells_by_norm_dose[nc][dk] = c_stats

                p_pass = all(c_stats[c]["cell_transport_pass"] for c in pair_cells)
                if dk == "0.8":
                    pair_status[nc][dk] = {
                        "pair_pass": p_pass,
                        "pair_collapsed": None,
                        "status": "baseline_pass" if p_pass else "baseline_miss",
                    }
                else:
                    p_coll = any(bool(c_stats[c]["cell_collapsed"]) for c in pair_cells)
                    if p_pass and (not p_coll):
                        st_lbl = "pass_intact"
                    elif p_pass and p_coll:
                        st_lbl = "pass_with_reduction"
                    elif (not p_pass) and (not p_coll):
                        st_lbl = "miss_no_collapse"
                    else:
                        st_lbl = "miss_and_collapsed"
                    pair_status[nc][dk] = {
                        "pair_pass": p_pass,
                        "pair_collapsed": p_coll,
                        "status": st_lbl,
                    }

            # L8H15 rebound deltas on THIRD_GENERIC_NEW / CAL_THIRD across {0.8, 1.2, 1.6}.
            reb_nc: dict[str, Any] = {}
            for mk in ("spec", "raw", "peer_med"):
                d_vals = {}
                for dk in ("0.8", "1.2", "1.6"):
                    sub_h = [
                        float(r[mk]["d_pos"]) for r in hd_rows
                        if r["bank"] == bank and r["norm_cond"] == nc and r["cell"] == third_cell
                        and r["prompt_id"] in allowed_pids and rib.dose_key(r["dose_mag"]) == dk
                        and r["head_name"] == "L8H15"
                    ]
                    d_vals[dk] = _mean(sub_h)
                reb_nc[mk] = {
                    "d0.8": d_vals["0.8"],
                    "d1.2": d_vals["1.2"],
                    "d1.6": d_vals["1.6"],
                    "delta_1.2_minus_0.8": d_vals["1.2"] - d_vals["0.8"],
                    "delta_1.6_minus_0.8": d_vals["1.6"] - d_vals["0.8"],
                }
            rebound_l8h15[nc] = reb_nc

        cross_norm: dict[str, Any] = {}
        for dk in ("1.2", "1.6"):
            gs, cs = classify_br3_states(
                br0_passed=br0_passed,
                p_old=bool(pair_status["OLD_NORM"][dk]["pair_pass"]),
                p_new=bool(pair_status["NEW_NORM"][dk]["pair_pass"]),
                c_old=bool(pair_status["OLD_NORM"][dk]["pair_collapsed"]),
                c_new=bool(pair_status["NEW_NORM"][dk]["pair_collapsed"]),
            )
            cross_norm[dk] = {"gate_state": gs, "collapse_state": cs}

        out["banks"][bank] = {
            "pair_cells": list(pair_cells),
            "cells_by_norm_dose": cells_by_norm_dose,
            "pair_status": pair_status,
            "cross_norm": cross_norm,
            "l8h15_rebound": rebound_l8h15,
        }

    return out


def classify_part_b_decision(
    br0_passed: bool,
    q_old: bool,
    q_new: bool,
    br4b_old: bool,
    br4b_new: bool,
) -> tuple[str, str]:
    """Total 5-state `part_b_norm_decision` & coupled `p92_09_part_b_track` (v3 §BR.4 & Alden #93)."""
    if not br0_passed:
        return ANCHOR_FAIL_NORM, ANCHOR_FAIL_TRACK
    if q_old and q_new:
        return (
            "pin_OLD_NORM (both_qualify_lower_perturbation_default)",
            "include_spec_only_track" if br4b_old else "diagnostic_only",
        )
    if q_old and (not q_new):
        return (
            "pin_OLD_NORM (old_only_qualifies)",
            "include_spec_only_track" if br4b_old else "diagnostic_only",
        )
    if (not q_old) and q_new:
        return (
            "pin_NEW_NORM (new_only_qualifies)",
            "include_spec_only_track" if br4b_new else "diagnostic_only",
        )
    return "decline_norm (neither_norm_qualifies_hold_part_b)", "not_selected (no_norm_pinned)"


def evaluate_br4(
    unp_rows: list[dict[str, Any]],
    hd_rows: list[dict[str, Any]],
    br1: dict[str, Any],
    br0_passed: bool,
) -> dict[str, Any]:
    def unit_head_stats(bank: str, nc: str, cell: str, hname: str, uid: str, mk: str) -> dict[str, float]:
        sub = [
            r for r in hd_rows
            if r["bank"] == bank and r["norm_cond"] == nc and r["cell"] == cell
            and rib.dose_key(r["dose_mag"]) == "0.8" and r["head_name"] == hname and r["unit"] == uid
        ]
        dp = _mean([float(r[mk]["d_pos"]) for r in sub])
        dn = _mean([float(r[mk]["d_neg"]) for r in sub])
        s_odd = 0.5 * (dn - dp)
        return {"d_pos": dp, "d_neg": dn, "S_odd": s_odd}

    cells_report: dict[str, Any] = {}
    for bank in ("NEW_BANK", "OLD_BANK"):
        c_prim = BR4_PRIMARY_CELL[bank]
        for nc in rib.NORM_ORDER:
            ckey = f"{bank}|{nc}"
            # 1. Primary cell 5-unit unpatched transport.
            sub_u5 = [
                r for r in unp_rows
                if r["bank"] == bank and r["norm_cond"] == nc and r["cell"] == c_prim and rib.dose_key(r["dose_mag"]) == "0.8"
            ]
            t5 = _cell_transport_stats(sub_u5)

            # 2. Individual L5/L6 late-write unit transport on primary cell.
            u_trans: dict[str, Any] = {}
            for uid in ("p92-46", "p92-09"):
                sub_uu = [r for r in sub_u5 if r["unit"] == uid]
                ms = _mean([float(r["transport"]["n_write_spec"]) for r in sub_uu])
                mr = _mean([float(r["transport"]["n_write_raw"]) for r in sub_uu])
                u_trans[uid] = {"mean_spec": ms, "mean_raw": mr, "pass": bool(ms >= 0.50 and mr >= 0.50)}

            # 3. 5-unit head-null gate for L7H5, L7H4, L7H9 on primary cell.
            l7_nulls: dict[str, Any] = {}
            for hname in ("L7H5", "L7H4", "L7H9"):
                sub_h5 = [
                    r for r in hd_rows
                    if r["bank"] == bank and r["norm_cond"] == nc and r["cell"] == c_prim
                    and rib.dose_key(r["dose_mag"]) == "0.8" and r["head_name"] == hname
                ]
                l7_nulls[hname] = _cell_head_null_stats(sub_h5)

            br4_valid = bool(
                br0_passed
                and t5["cell_transport_pass"]
                and u_trans["p92-46"]["pass"]
                and u_trans["p92-09"]["pass"]
                and all(l7_nulls[h]["head_null_pass"] for h in ("L7H5", "L7H4", "L7H9"))
            )

            # Compute exact signed statistics for p92-46, p92-09, and L3 across L7H5/L7H4/L7H9.
            stats_grid: dict[str, dict[str, dict[str, float]]] = {}
            for mk in ("raw", "spec"):
                stats_grid[mk] = {}
                for hname in ("L7H5", "L7H4", "L7H9"):
                    for uid in ("p92-46", "p92-09"):
                        stats_grid[mk][f"{hname}|{uid}"] = unit_head_stats(bank, nc, c_prim, hname, uid, mk)
                    l3_sodd = _mean([unit_head_stats(bank, nc, c_prim, hname, u3, mk)["S_odd"] for u3 in L3_UNITS])
                    stats_grid[mk][f"{hname}|L3_mean"] = {"S_odd": l3_sodd}

            # BR.4a (p92-46 raw sign flip)
            p46_r = stats_grid["raw"]["L7H5|p92-46"]
            p46_s = stats_grid["spec"]["L7H5|p92-46"]
            ctrl46_r = max(stats_grid["raw"][f"{ch}|p92-46"]["S_odd"] for ch in L7_CTRL_HEADS)
            ctrl46_s = max(stats_grid["spec"][f"{ch}|p92-46"]["S_odd"] for ch in L7_CTRL_HEADS)
            l3_r = stats_grid["raw"]["L7H5|L3_mean"]["S_odd"]
            br4a_holds = bool(
                br4_valid
                and p46_r["d_pos"] <= -0.25 and p46_r["d_neg"] >= 0.25
                and p46_s["d_pos"] <= -0.25 and p46_s["d_neg"] >= 0.25
                and (p46_r["S_odd"] - ctrl46_r) >= 0.15
                and (p46_s["S_odd"] - ctrl46_s) >= 0.15
                and (p46_r["S_odd"] - l3_r) >= 0.20
            )

            # BR.4b (p92-09 SpecLogit flip + orthogonal raw tag)
            p09_s = stats_grid["spec"]["L7H5|p92-09"]
            p09_r = stats_grid["raw"]["L7H5|p92-09"]
            ctrl09_s = max(stats_grid["spec"][f"{ch}|p92-09"]["S_odd"] for ch in L7_CTRL_HEADS)
            l3_s = stats_grid["spec"]["L7H5|L3_mean"]["S_odd"]
            br4b_holds = bool(
                br4_valid
                and p09_s["d_pos"] <= -0.50 and p09_s["d_neg"] >= 0.25
                and (p09_s["S_odd"] - ctrl09_s) >= 0.25
                and (p09_s["S_odd"] - l3_s) >= 0.25
            )

            if not br0_passed:
                raw_tag = ANCHOR_FAIL_SCI
                br4_verdict = ANCHOR_FAIL_SCI
                br4a_verdict = ANCHOR_FAIL_SCI
                br4b_verdict = ANCHOR_FAIL_SCI
            else:
                if p09_r["d_pos"] <= -0.25 and p09_r["d_neg"] >= 0.25:
                    raw_tag = "raw_present"
                elif p09_r["d_neg"] <= 0.05:
                    raw_tag = "raw_absent_or_neg"
                else:
                    raw_tag = "raw_intermediate"
                br4_verdict = "valid" if br4_valid else "uninformative (br4_gate_failed)"
                br4a_verdict = ("holds" if br4a_holds else "fails") if br4_valid else "uninformative (br4_gate_failed)"
                br4b_verdict = ("holds" if br4b_holds else "fails") if br4_valid else "uninformative (br4_gate_failed)"

            cells_report[ckey] = {
                "primary_cell": c_prim,
                "br4_valid": br4_valid,
                "br4_verdict": br4_verdict,
                "br4a_holds": br4a_holds,
                "br4a_verdict": br4a_verdict,
                "br4b_holds": br4b_holds,
                "br4b_verdict": br4b_verdict,
                "p92_09_raw_tag": raw_tag,
                "transport_5unit": t5,
                "late_write_unit_transport": u_trans,
                "l7_nulls": l7_nulls,
                "margins": {
                    "p92_46_raw_over_ctrl": p46_r["S_odd"] - ctrl46_r,
                    "p92_46_spec_over_ctrl": p46_s["S_odd"] - ctrl46_s,
                    "p92_46_raw_over_L3": p46_r["S_odd"] - l3_r,
                    "p92_09_spec_over_ctrl": p09_s["S_odd"] - ctrl09_s,
                    "p92_09_spec_over_L3": p09_s["S_odd"] - l3_s,
                },
                "stats": stats_grid,
            }

    # Emit descriptive per-cell and multi-cell third-person pools alongside (never overriding primary).
    descriptive_pools: dict[str, Any] = {}
    pool_specs = {
        "NEW_BANK": {
            "pool_all_third": ("ROLE_BARE", "NAME_BARE", "ALIAS_TO_ROLE", "ALIAS_TO_NAME"),
            "ROLE_BARE": ("ROLE_BARE",),
            "NAME_BARE": ("NAME_BARE",),
            "ALIAS_TO_ROLE": ("ALIAS_TO_ROLE",),
            "ALIAS_TO_NAME": ("ALIAS_TO_NAME",),
            "SELF": ("SELF",),
        },
        "OLD_BANK": {
            "pool_all_third": ("ROLE_DEFINITE_NEW", "NAMED_PERSON_NEW", "THIRD_GENERIC_NEW"),
            "ROLE_DEFINITE_NEW": ("ROLE_DEFINITE_NEW",),
            "NAMED_PERSON_NEW": ("NAMED_PERSON_NEW",),
            "THIRD_GENERIC_NEW": ("THIRD_GENERIC_NEW",),
            "SELF_HELD_NEW": ("SELF_HELD_NEW",),
        },
    }
    for bank, pmap in pool_specs.items():
        for nc in rib.NORM_ORDER:
            dkey = f"{bank}|{nc}"
            descriptive_pools[dkey] = {}
            for pname, pcells in pmap.items():
                pdict: dict[str, Any] = {}
                for hname in ("L7H5", "L7H4", "L7H9"):
                    for gname, gunits in (
                        ("p92-46", ("p92-46",)),
                        ("p92-09", ("p92-09",)),
                        ("L5_L6", ("p92-46", "p92-09")),
                        ("L3", L3_UNITS),
                    ):
                        sub = [
                            r for r in hd_rows
                            if r["bank"] == bank and r["norm_cond"] == nc and r["cell"] in pcells
                            and rib.dose_key(r["dose_mag"]) == "0.8" and r["head_name"] == hname and r["unit"] in gunits
                        ]
                        pdict[f"{hname}|{gname}"] = {
                            "n_rows": len(sub),
                            "spec_d_pos": _mean([float(r["spec"]["d_pos"]) for r in sub]),
                            "spec_d_neg": _mean([float(r["spec"]["d_neg"]) for r in sub]),
                            "raw_d_pos": _mean([float(r["raw"]["d_pos"]) for r in sub]),
                            "raw_d_neg": _mean([float(r["raw"]["d_neg"]) for r in sub]),
                        }
                descriptive_pools[dkey][pname] = pdict

    # Total deterministic Part B operating-norm decision on fixed NEW_BANK benchmark.
    q_old = bool(
        br0_passed
        and (br1["NEW_BANK|OLD_NORM"]["heads"]["L8H15"]["verdict"] == "valid")
        and cells_report["NEW_BANK|OLD_NORM"]["br4a_holds"]
    )
    q_new = bool(
        br0_passed
        and (br1["NEW_BANK|NEW_NORM"]["heads"]["L8H15"]["verdict"] == "valid")
        and cells_report["NEW_BANK|NEW_NORM"]["br4a_holds"]
    )
    decision, p09_track = classify_part_b_decision(
        br0_passed=br0_passed,
        q_old=q_old,
        q_new=q_new,
        br4b_old=bool(cells_report["NEW_BANK|OLD_NORM"]["br4b_holds"]),
        br4b_new=bool(cells_report["NEW_BANK|NEW_NORM"]["br4b_holds"]),
    )

    return {
        "cells": cells_report,
        "descriptive_cells_and_pools": descriptive_pools,
        "qualification_NEW_BANK": {"Q_OLD_NORM": q_old, "Q_NEW_NORM": q_new},
        "part_b_norm_decision": decision,
        "p92_09_part_b_track": p09_track,
    }


def score_full_receipt(
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    pins: dict[str, str],
    old_receipt_path: Path,
    new_receipt_path: Path,
    out_path: Path | None = None,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Score a verified full receipt (`520` unpatched, `3,220` head rows) and emit scorecard."""
    bound = bind_and_validate_rows(receipt, manifest, expect_full=True, pins=pins, allow_fixture=allow_fixture)
    br0 = evaluate_br0_anchors(bound, manifest, old_receipt_path, new_receipt_path, expect_full=True)
    br0_passed = br0["status"] == "passed"

    unp_rows = list(bound["unp_by_key"].values())
    hd_rows = list(bound["hd_by_key"].values())

    br1 = evaluate_br1(unp_rows, hd_rows, br0_passed)
    br2 = evaluate_br2(hd_rows, br1, br0_passed, bound["derived_frac_pos_by_key"])
    br3 = evaluate_br3(unp_rows, hd_rows, br0_passed)
    br4 = evaluate_br4(unp_rows, hd_rows, br1, br0_passed)

    scorecard = {
        "meta": {
            "schema_version": rib.SCHEMA_VERSION,
            "artifact_type": "instruct_bridge_scorecard",
            "execution_provenance": bound["execution_provenance"],
            "provenance_mode": "measured_gpu_run" if bound["execution_provenance"] == "measured_gpu_run" else "fixture_only_not_launch_evidence",
            "pins": dict(pins),
            "external_pins_verified": True,
            "complete_join_verified": True,
            "max_abs_sham_spec": bound["max_abs_sham_spec"],
            "max_abs_sham_raw": bound["max_abs_sham_raw"],
            "max_abs_decomp_err": bound["max_abs_decomp_err"],
            "null_basis_pairs_verified": bound["null_basis_pairs_verified"],
        },
        "BR0_anchor_replay": br0,
        "BR1_validity_gates": br1,
        "BR2_bank_x_norm_L8H15": br2,
        "BR3_high_dose_transport": br3,
        "BR4_L7H5_and_part_b_norm": br4,
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
        os.replace(tmp, out_path)

    return scorecard


def verify_smoke_receipt_only(
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    pins: dict[str, str],
    old_receipt_path: Path,
    new_receipt_path: Path,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Technical verification of smoke receipt (zero study verdicts emitted)."""
    bound = bind_and_validate_rows(receipt, manifest, expect_full=False, pins=pins, allow_fixture=allow_fixture)
    br0_sub = evaluate_br0_anchors(bound, manifest, old_receipt_path, new_receipt_path, expect_full=False)
    is_gpu = bound["execution_provenance"] == "measured_gpu_run"
    return {
        "smoke_technical_check_passed": bool(br0_sub["status"] == "passed" and (is_gpu or allow_fixture)),
        "execution_provenance": bound["execution_provenance"],
        "provenance_mode": "measured_gpu_run" if is_gpu else "fixture_only_not_launch_evidence",
        "max_abs_sham_spec": bound["max_abs_sham_spec"],
        "max_abs_sham_raw": bound["max_abs_sham_raw"],
        "max_abs_decomp_err": bound["max_abs_decomp_err"],
        "null_basis_pairs_verified": bound["null_basis_pairs_verified"],
        "anchor_subset_check": br0_sub,
        "timing": receipt["meta"]["timing"],
    }


# ---------------------------------------------------------------------------
# Exhaustive CPU Self-Tests (Alden #93 & #97 Controls)
# ---------------------------------------------------------------------------

def _make_full_receipt_from_real_anchors(
    manifest: dict[str, Any],
    pins: dict[str, str],
    old_rcpt: dict[str, Any],
    new_rcpt: dict[str, Any],
) -> dict[str, Any]:
    """Build a valid full synthetic receipt whose anchor rows match REAL disk receipts 100%."""
    old_idx = {
        (r["unit"], r["prompt_id"], r["cell"], rib.dose_key(r["dose_mag"]), r["head_name"]): r
        for r in old_rcpt["q12_rows"]
    }
    new_hd_idx: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for tbl in ("eval_arm_a_head_rows", "cal_head_rows"):
        for r in new_rcpt[tbl]:
            new_hd_idx[(tbl, r["unit"], r["prompt_id"], r["cell"], rib.dose_key(r["dose_mag"]), r["head_name"])] = r
    new_unp_idx: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for tbl in ("eval_arm_a_unpatched_rows", "cal_unpatched_rows"):
        for r in new_rcpt[tbl]:
            new_unp_idx[(tbl, r["unit"], r["prompt_id"], r["cell"], rib.dose_key(r["dose_mag"]))] = r

    def synth_metric(dp: float, dn: float) -> dict[str, Any]:
        return {
            "d0": 0.0,
            "d_pos": dp,
            "d_neg": dn,
            "even": 0.5 * (dp + dn),
            "odd": 0.5 * (dp - dn),
            "max_abs_sham": 0.0,
            "wdown_orth": {
                "d_pos_draws": [0.0, 0.0, 0.0, 0.0],
                "d_neg_draws": [0.0, 0.0, 0.0, 0.0],
                "even_draws": [0.0, 0.0, 0.0, 0.0],
                "odd_draws": [0.0, 0.0, 0.0, 0.0],
                "d_pos_med": 0.0, "d_neg_med": 0.0, "even_med": 0.0, "odd_med": 0.0,
                "even_max_abs": 0.0, "odd_max_abs": 0.0,
            },
            "iso_resid": {
                "d_pos_draws": [0.0, 0.0, 0.0, 0.0],
                "d_neg_draws": [0.0, 0.0, 0.0, 0.0],
                "even_draws": [0.0, 0.0, 0.0, 0.0],
                "odd_draws": [0.0, 0.0, 0.0, 0.0],
                "d_pos_med": 0.0, "d_neg_med": 0.0, "even_med": 0.0, "odd_med": 0.0,
                "even_max_abs": 0.0, "odd_max_abs": 0.0,
            },
        }

    def synth_pmed_from_raw_spec(raw_m: dict[str, Any], spec_m: dict[str, Any]) -> dict[str, Any]:
        pm: dict[str, Any] = {k: float(raw_m[k]) - float(spec_m[k]) for k in ("d0", "d_pos", "d_neg", "even", "odd")}
        pm["max_abs_sham"] = 0.0
        for grp in ("wdown_orth", "iso_resid"):
            g = {}
            for dk in ("d_pos_draws", "d_neg_draws", "even_draws", "odd_draws"):
                g[dk] = [float(r) - float(s) for r, s in zip(raw_m[grp][dk], spec_m[grp][dk])]
            g["d_pos_med"] = float(statistics.median(g["d_pos_draws"]))
            g["d_neg_med"] = float(statistics.median(g["d_neg_draws"]))
            g["even_med"] = float(statistics.median(g["even_draws"]))
            g["odd_med"] = float(statistics.median(g["odd_draws"]))
            pm[grp] = g
        return pm

    norms_map = manifest["meta"]["norms"]
    unp_rows: list[dict[str, Any]] = []
    hd_rows: list[dict[str, Any]] = []

    for uid in rib.CORE_UNIT_IDS:
        u = rib.UNIT_BY_ID[uid]
        for e in manifest["schedule_full"]:
            bank = e["bank"]
            pid = e["prompt_id"]
            cell = e["cell"]
            d = float(e["dose_mag"])
            dk = rib.dose_key(d)
            bsha = rib.sha256_bytes(f"{bank}|{uid}|{pid}|{dk}".encode())
            for nc in rib.NORM_ORDER:
                m_L = float(norms_map[nc][str(u["layer"])])
                if bank == "NEW_BANK" and nc == "NEW_NORM":
                    tbl = "eval_arm_a_unpatched_rows" if pid.startswith("eval_") else "cal_unpatched_rows"
                    rt = copy.deepcopy(new_unp_idx[(tbl, uid, pid, cell, dk)]["transport"])
                    # Ensure peer_med fields exist linearly (01bd41d9 stores n_write_peer_med, add neg & draw peer_meds linearly)
                    rt["n_write_neg_peer_med"] = float(rt["n_write_neg_raw"]) - float(rt["n_write_neg_spec"])
                    rt["wdown_pos_draws_peer_med"] = [float(r) - float(s) for r, s in zip(rt["wdown_pos_draws_raw"], rt["wdown_pos_draws_spec"])]
                    rt["iso_pos_draws_peer_med"] = [float(r) - float(s) for r, s in zip(rt["iso_pos_draws_raw"], rt["iso_pos_draws_spec"])]
                else:
                    rt = {
                        "n_write_spec": 2.0, "n_write_raw": 1.5, "n_write_peer_med": -0.5,
                        "n_write_neg_spec": -2.0, "n_write_neg_raw": -1.5, "n_write_neg_peer_med": 0.5,
                        "wdown_pos_draws_spec": [0.0, 0.0, 0.0, 0.0],
                        "wdown_pos_draws_raw": [0.0, 0.0, 0.0, 0.0],
                        "wdown_pos_draws_peer_med": [0.0, 0.0, 0.0, 0.0],
                        "iso_pos_draws_spec": [0.0, 0.0, 0.0, 0.0],
                        "iso_pos_draws_raw": [0.0, 0.0, 0.0, 0.0],
                        "iso_pos_draws_peer_med": [0.0, 0.0, 0.0, 0.0],
                    }
                unp_rows.append({
                    "bank": bank, "norm_cond": nc, "unit": uid, "unit_layer": u["layer"],
                    "prompt_id": pid, "cell": cell, "story_or_frame": e["story_or_frame"],
                    "block_a": e["block_a"], "dose_mag": d, "head_patched": bool(e["head_patched"]),
                    "norm_multiplier": m_L, "injected_norm": d * m_L, "read_pos_resid_norm": m_L,
                    "norm_ratio": d, "null_unit_vec_sha256": bsha, "transport": rt,
                })
                if not e["head_patched"]:
                    continue
                for h in rib.CANONICAL_HEADS:
                    hn = h["name"]
                    if bank == "OLD_BANK" and nc == "OLD_NORM" and e["in_old_anchor_heads"] and hn in rib.OLD_OVERLAP_HEADS:
                        ref = old_idx[(uid, pid, cell, dk, hn)]
                        sp = copy.deepcopy(ref["spec"])
                        rw = copy.deepcopy(ref["raw"])
                        pm = synth_pmed_from_raw_spec(rw, sp)
                    elif bank == "NEW_BANK" and nc == "NEW_NORM":
                        tbl = "eval_arm_a_head_rows" if pid.startswith("eval_") else "cal_head_rows"
                        ref = new_hd_idx[(tbl, uid, pid, cell, dk, hn)]
                        sp = copy.deepcopy(ref["spec"])
                        rw = copy.deepcopy(ref["raw"])
                        pm = copy.deepcopy(ref["peer_med"])
                    else:
                        sp = synth_metric(-0.1, 0.1)
                        rw = synth_metric(-0.1, 0.1)
                        pm = synth_metric(0.0, 0.0)
                    hrow_f: dict[str, Any] = {
                        "bank": bank, "norm_cond": nc, "unit": uid, "unit_layer": u["layer"],
                        "prompt_id": pid, "cell": cell, "story_or_frame": e["story_or_frame"],
                        "block_a": e["block_a"], "dose_mag": d, "head_name": hn,
                        "head_layer": h["layer"], "head_idx": h["head"], "head_kind": h["kind"],
                        "null_unit_vec_sha256": bsha, "spec": sp, "raw": rw, "peer_med": pm,
                    }
                    if e["record_peer_deltas"] and hn in rib.PEER_DELTA_HEADS:
                        pids_req = manifest["bindings"][f"{bank}|{pid}"][uid]["peers"]
                        if bank == "NEW_BANK" and nc == "NEW_NORM" and "peer_deltas" in ref:
                            p_deltas = copy.deepcopy(ref["peer_deltas"])
                        else:
                            p_deltas = {str(tid): {"d_pos": 0.1, "d_neg": -0.1} for tid in pids_req}
                        dpos_list = [float(p_deltas[str(tid)]["d_pos"]) for tid in pids_req]
                        hrow_f["peer_deltas"] = p_deltas
                        hrow_f["peer_mean_d_pos"] = float(sum(dpos_list) / len(dpos_list))
                        hrow_f["frac_peer_tokens_pos"] = float(sum(1 for v in dpos_list if v > 0.0) / len(dpos_list))
                    hd_rows.append(hrow_f)

    den_full = manifest["meta"]["denominators"]["full"]
    return {
        "meta": {
            "schema_version": rib.SCHEMA_VERSION,
            "artifact_type": "instruct_bridge_full_receipt",
            "run_mode": "full",
            "smoke_1unit": False,
            "execution_provenance": "synthetic_anchor_fixture",
            "device_name": "synthetic_fixture",
            "complete": True,
            "model_repo": rib.MODEL_REPO,
            "model_commit": rib.MODEL_COMMIT,
            "dtype": rib.DTYPE_STR,
            "attn_implementation": rib.ATTN_IMPL,
            "design_sha256": pins["design_sha256"],
            "manifest_sha256": pins["manifest_sha256"],
            "runner_sha256": pins["runner_sha256"],
            "scorer_sha256": pins["scorer_sha256"],
            "pins": dict(pins),
            "units_run": list(rib.CORE_UNIT_IDS),
            "n_unpatched_rows": len(unp_rows),
            "n_head_rows": len(hd_rows),
            "n_b19_forwards": den_full["n_b19_forwards"],
            "max_abs_null_vec_diff": 0.0,
            "timing": {
                "model_load_sec": 1.0,
                "b19_elapsed_sec": 696.0,
                "non_b19_runner_sec": 5.0,
                "total_runner_sec": 701.0,
                "n_b19_done": den_full["n_b19_forwards"],
                "ms_per_b19": 100.0,
                "projected_full_b19_sec": 696.0,
            },
        },
        "unpatched_rows": unp_rows,
        "head_rows": hd_rows,
    }


def run_self_tests(repo_root: Path) -> dict[str, Any]:
    design_p = repo_root / rib.DESIGN_REL_PATH
    manifest_p = repo_root / "explore_olmo2_attn_mlp/instruct_bridge_manifest.json"
    runner_p = Path(rib.__file__).resolve()
    scorer_p = Path(__file__).resolve()
    old_rcpt_p = repo_root / "explore_olmo2_attn_mlp/results_signed_dose_full_d13fc3dd.json"
    new_rcpt_p = repo_root / "explore_olmo2_attn_mlp/results_stage_sweep_instruct_01bd41d9.json"

    d_sha, m_sha, r_sha, s_sha = (
        rib.sha256_file(design_p), rib.sha256_file(manifest_p), rib.sha256_file(runner_p), rib.sha256_file(scorer_p)
    )
    manifest, pins = admit_scorer_package(design_p, manifest_p, runner_p, scorer_p, d_sha, m_sha, r_sha, s_sha)

    # 0. Red/green module-binding test for --runner (Alden #104 item 3).
    with tempfile.TemporaryDirectory() as td:
        other_runner = Path(td) / "other_runner.py"
        other_runner.write_bytes(runner_p.read_bytes())
        try:
            admit_scorer_package(
                design_p, manifest_p, other_runner, scorer_p,
                d_sha, m_sha, rib.sha256_file(other_runner), s_sha,
            )
            raise AssertionError("Expected refusal when --runner path != actually imported rib module")
        except ValueError as e:
            assert "does not match actually imported runner module" in str(e)

    # 1. Valid literal-source control.
    lit = rib.verify_literal_source_bindings(
        manifest=manifest,
        old_manifest_bytes=rib.load_git_blob(repo_root, rib.OLD_GIT_COMMIT, "explore_olmo2_attn_mlp/signed_dose_manifest.json"),
        old_runner_bytes=rib.load_git_blob(repo_root, rib.OLD_GIT_COMMIT, "explore_olmo2_attn_mlp/run_signed_dose.py"),
        new_manifest_bytes=rib.load_git_blob(repo_root, rib.NEW_GIT_COMMIT, "explore_olmo2_attn_mlp/stage_sweep_manifest.json"),
        new_runner_bytes=rib.load_git_blob(repo_root, rib.NEW_GIT_COMMIT, "explore_olmo2_attn_mlp/run_stage_sweep.py"),
        new_receipt_bytes=new_rcpt_p.read_bytes(),
        old_receipt_bytes=old_rcpt_p.read_bytes(),
    )
    assert lit["literal_source_bindings_verified"] is True

    old_rcpt = json.loads(old_rcpt_p.read_text(encoding="utf-8"))
    new_rcpt = json.loads(new_rcpt_p.read_text(encoding="utf-8"))

    # 2. Anchor-backed synthetic fixture joins (offline fixture check, NOT measured GPU replay).
    good = _make_full_receipt_from_real_anchors(manifest, pins, old_rcpt, new_rcpt)
    # Red: synthetic fixture refused when allow_fixture=False!
    try:
        score_full_receipt(good, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=False)
        raise AssertionError("Expected refusal of synthetic_anchor_fixture with allow_fixture=False")
    except ValueError as e:
        assert "measured_gpu_run" in str(e)
    # Green: synthetic fixture explicitly marked fixture_only_not_launch_evidence when allow_fixture=True.
    sc_good = score_full_receipt(good, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=True)
    assert sc_good["meta"]["provenance_mode"] == "fixture_only_not_launch_evidence"
    assert sc_good["BR0_anchor_replay"]["status"] == "passed"
    assert sc_good["BR0_anchor_replay"]["join_1_old_head"]["n_rows_joined"] == 480
    assert sc_good["BR0_anchor_replay"]["join_1_old_head"]["max_abs_diff"] == 0.0
    assert sc_good["BR0_anchor_replay"]["join_2a_new_head"]["n_rows_joined"] == 910
    assert sc_good["BR0_anchor_replay"]["join_2a_new_head"]["max_abs_diff"] == 0.0
    assert sc_good["BR0_anchor_replay"]["join_2b_new_unpatched"]["n_rows_joined"] == 160
    assert sc_good["BR0_anchor_replay"]["join_2b_new_unpatched"]["max_abs_diff"] == 0.0
    assert sc_good["BR3_high_dose_transport"]["banks"]["NEW_BANK"]["cells_by_norm_dose"]["NEW_NORM"]["0.8"]["CAL_SELF"]["cell_collapsed"] is None
    assert sc_good["BR3_high_dose_transport"]["banks"]["OLD_BANK"]["cells_by_norm_dose"]["OLD_NORM"]["0.8"]["SELF_HELD_NEW"]["cell_collapsed"] is None
    assert "NEW_BANK|NEW_NORM" in sc_good["BR2_bank_x_norm_L8H15"]["l8_decomposed"]["L8H15"]["mean_frac_peer_tokens_pos"]

    # 3. Smoke receipt hard refusal on full scorer (zero output written).
    with tempfile.TemporaryDirectory() as td:
        out_p = Path(td) / "should_not_exist.json"
        smoke_bad = copy.deepcopy(good)
        smoke_bad["meta"]["artifact_type"] = "instruct_bridge_smoke_receipt"
        try:
            score_full_receipt(smoke_bad, manifest, pins, old_rcpt_p, new_rcpt_p, out_path=out_p, allow_fixture=True)
            raise AssertionError("Expected smoke receipt refusal")
        except ValueError as e:
            assert "Refused receipt" in str(e)
        assert not out_p.exists()

    # 4. Sham contract tests (Alden #93 item 2):
    miss_sham = copy.deepcopy(good)
    miss_sham["head_rows"][0]["spec"]["max_abs_sham"] = None
    try:
        score_full_receipt(miss_sham, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=True)
        raise AssertionError("Expected rejection of head row with None sham")
    except ValueError as e:
        assert "max_abs_sham" in str(e)

    fab_sham = copy.deepcopy(good)
    cal_self_idx = next(i for i, r in enumerate(fab_sham["unpatched_rows"]) if r["cell"] == "CAL_SELF")
    fab_sham["unpatched_rows"][cal_self_idx]["max_abs_sham"] = 0.0
    try:
        score_full_receipt(fab_sham, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=True)
        raise AssertionError("Expected rejection of manufactured sham on unpatched CAL_SELF row")
    except ValueError as e:
        assert "Forbidden sham key" in str(e)

    # 5. Metadata tamper witness (#70 discipline).
    tamper = copy.deepcopy(good)
    tamper["head_rows"][10]["cell"] = "WRONG_CELL"
    try:
        score_full_receipt(tamper, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=True)
        raise AssertionError("Expected metadata tamper refusal")
    except ValueError as e:
        assert "metadata tamper" in str(e)

    # 5b. Alden #104 Witness 1 (Paired Red/Green): Derived null summary tamper + 19-peer IDs validation.
    # Red 1a: Exact Alden #104 witness — change ONLY spec.wdown_orth.odd_med by +4 on p92-46/L8H15/eval_role_bare_E1.
    odd_med_tamper = copy.deepcopy(good)
    idx_alden = next(
        i for i, r in enumerate(odd_med_tamper["head_rows"])
        if r["bank"] == "NEW_BANK" and r["norm_cond"] == "NEW_NORM"
        and r["unit"] == "p92-46" and r["head_name"] == "L8H15" and r["prompt_id"] == "eval_role_bare_E1"
    )
    odd_med_tamper["head_rows"][idx_alden]["spec"]["wdown_orth"]["odd_med"] += 4.0
    try:
        score_full_receipt(odd_med_tamper, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=True)
        raise AssertionError("Expected refusal on tampered derived spec.wdown_orth.odd_med (+4)")
    except ValueError as e:
        assert "Derived null summary mismatch" in str(e)

    # Red 1b: Drop 1 peer ID from required 19 peer_deltas on L8H15 d=0.8 -> must hard-refuse!
    drop_peer = copy.deepcopy(good)
    some_key = next(iter(drop_peer["head_rows"][idx_alden]["peer_deltas"].keys()))
    del drop_peer["head_rows"][idx_alden]["peer_deltas"][some_key]
    try:
        score_full_receipt(drop_peer, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=True)
        raise AssertionError("Expected refusal on missing required peer_deltas ID")
    except ValueError as e:
        assert "Required 19 peer_deltas IDs" in str(e)

    # Red 1c: Tamper stored frac_peer_tokens_pos without changing peer_deltas -> must hard-refuse!
    tamper_fp = copy.deepcopy(good)
    tamper_fp["head_rows"][idx_alden]["frac_peer_tokens_pos"] = 0.9999
    try:
        score_full_receipt(tamper_fp, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=True)
        raise AssertionError("Expected refusal on tampered frac_peer_tokens_pos")
    except ValueError as e:
        assert "frac_peer_tokens_pos disagrees" in str(e)

    # 6. Unified BR.0 anchor-failure test (Alden #93 item 3):
    # Perturb 1 OLD anchor row's d_pos by +0.005 (> 1e-3), updating derived even/odd linearly so row integrity passes!
    anc_fail = copy.deepcopy(good)
    old_h_idx = next(
        i for i, r in enumerate(anc_fail["head_rows"])
        if r["bank"] == "OLD_BANK" and r["norm_cond"] == "OLD_NORM" and r["head_name"] == "L8H15" and rib.dose_key(r["dose_mag"]) == "0.8"
    )
    for mk in ("spec", "raw"):
        m = anc_fail["head_rows"][old_h_idx][mk]
        m["d_pos"] += 0.005
        m["even"] = 0.5 * (m["d_pos"] + m["d_neg"]) - m["d0"]
        m["odd"] = 0.5 * (m["d_pos"] - m["d_neg"])
    sc_fail = score_full_receipt(anc_fail, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=True)
    assert sc_fail["BR0_anchor_replay"]["status"] == "failed"
    assert sc_fail["BR1_validity_gates"]["NEW_BANK|NEW_NORM"]["heads"]["L8H15"]["verdict"] == ANCHOR_FAIL_SCI
    assert sc_fail["BR2_bank_x_norm_L8H15"]["primary_L8H15_spec"]["verdict"] == ANCHOR_FAIL_SCI
    assert sc_fail["BR3_high_dose_transport"]["banks"]["NEW_BANK"]["cross_norm"]["1.6"]["gate_state"] == ANCHOR_FAIL_SCI
    assert sc_fail["BR3_high_dose_transport"]["banks"]["NEW_BANK"]["cross_norm"]["1.6"]["collapse_state"] == ANCHOR_FAIL_SCI
    assert sc_fail["BR4_L7H5_and_part_b_norm"]["cells"]["NEW_BANK|NEW_NORM"]["br4_verdict"] == ANCHOR_FAIL_SCI
    assert sc_fail["BR4_L7H5_and_part_b_norm"]["cells"]["NEW_BANK|NEW_NORM"]["p92_09_raw_tag"] == ANCHOR_FAIL_SCI
    assert sc_fail["BR4_L7H5_and_part_b_norm"]["part_b_norm_decision"] == ANCHOR_FAIL_NORM
    assert sc_fail["BR4_L7H5_and_part_b_norm"]["p92_09_part_b_track"] == ANCHOR_FAIL_TRACK

    # 7. Exhaustive branch coverage of BR.2 (7 states), BR.3 (4x4 states + narrow miss), BR.4 (5 states).
    all_v = {k: True for k in ("OLD_BANK|OLD_NORM", "OLD_BANK|NEW_NORM", "NEW_BANK|OLD_NORM", "NEW_BANK|NEW_NORM")}
    assert classify_br2(False, all_v, -0.6, -0.1, -0.6, -0.1)["verdict"] == ANCHOR_FAIL_SCI
    assert classify_br2(True, {**all_v, "OLD_BANK|NEW_NORM": False}, -0.6, -0.1, -0.6, -0.1)["verdict"].startswith("uninformative (cell_gate_failed:")
    assert classify_br2(True, all_v, -0.60, -0.10, -0.55, -0.05)["verdict"] == "norm_dominant"
    assert classify_br2(True, all_v, -0.60, -0.55, -0.05, -0.05)["verdict"] == "bank_dominant"
    assert classify_br2(True, all_v, -0.60, -0.10, -0.10, -0.10)["verdict"] == "interaction_present"
    assert classify_br2(True, all_v, -0.60, -0.35, -0.35, -0.10)["verdict"] == "additive_mixed"
    assert classify_br2(True, all_v, -0.20, -0.15, -0.15, -0.10)["verdict"] == "neither_or_inconclusive"

    for (po, pn), exp_g in [
        ((True, True), "both_pass"), ((True, False), "old_pass_new_miss"),
        ((False, True), "old_miss_new_pass"), ((False, False), "both_miss"),
    ]:
        for (co, cn), exp_c in [
            ((False, False), "neither_collapsed"), ((False, True), "new_only_collapsed"),
            ((True, False), "old_only_collapsed"), ((True, True), "both_collapsed"),
        ]:
            assert classify_br3_states(True, po, pn, co, cn) == (exp_g, exp_c)
    assert classify_br3_states(False, True, True, False, False) == (ANCHOR_FAIL_SCI, ANCHOR_FAIL_SCI)
    assert not ((0.49 <= 0.0) or (1.0 >= 0.50 and 0.49 <= 0.25 * 1.0))

    assert classify_part_b_decision(False, True, True, True, True) == (ANCHOR_FAIL_NORM, ANCHOR_FAIL_TRACK)
    assert classify_part_b_decision(True, True, True, True, False) == ("pin_OLD_NORM (both_qualify_lower_perturbation_default)", "include_spec_only_track")
    assert classify_part_b_decision(True, True, False, False, True) == ("pin_OLD_NORM (old_only_qualifies)", "diagnostic_only")
    assert classify_part_b_decision(True, False, True, False, True) == ("pin_NEW_NORM (new_only_qualifies)", "include_spec_only_track")
    assert classify_part_b_decision(True, False, False, True, True) == ("decline_norm (neither_norm_qualifies_hold_part_b)", "not_selected (no_norm_pinned)")

    # 7b. Green 1 (Alden #97 & #104): Nonlinear-median positive control (each metric matches its OWN draw median
    # while median(spec) != median(raw) - median(peer_med)), paired with broken per-draw linear rejection.
    nl = copy.deepcopy(good)
    idx_non_anc = next(
        i for i, r in enumerate(nl["head_rows"])
        if r["bank"] == "OLD_BANK" and r["norm_cond"] == "NEW_NORM"
    )
    r_nl = nl["head_rows"][idx_non_anc]
    raw_d = [1.0, 5.0, 2.0, 4.0]
    pm_d = [0.2, 4.9, 0.1, 0.3]
    sp_d = [r - p for r, p in zip(raw_d, pm_d)]
    assert abs(statistics.median(sp_d) - (statistics.median(raw_d) - statistics.median(pm_d))) > 0.1
    for mobj, vals in ((r_nl["raw"], raw_d), (r_nl["peer_med"], pm_d), (r_nl["spec"], sp_d)):
        mobj["wdown_orth"]["d_pos_draws"] = list(vals)
        mobj["wdown_orth"]["d_neg_draws"] = [-v for v in vals]
        mobj["wdown_orth"]["even_draws"] = [0.0] * 4
        mobj["wdown_orth"]["odd_draws"] = list(vals)
        mobj["wdown_orth"]["d_pos_med"] = float(statistics.median(vals))
        mobj["wdown_orth"]["d_neg_med"] = float(statistics.median([-v for v in vals]))
        mobj["wdown_orth"]["even_med"] = 0.0
        mobj["wdown_orth"]["odd_med"] = float(statistics.median(vals))
        mobj["wdown_orth"]["even_max_abs"] = 0.0
        mobj["wdown_orth"]["odd_max_abs"] = max(abs(x) for x in vals)
    sc_nl = score_full_receipt(nl, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=True)
    assert sc_nl["BR0_anchor_replay"]["status"] == "passed"

    nl["head_rows"][idx_non_anc]["spec"]["wdown_orth"]["d_pos_draws"][0] += 0.01
    # Even if we also updated d_pos_med/even/odd for spec, linear cross-metric identity must still refuse!
    sp_w = nl["head_rows"][idx_non_anc]["spec"]["wdown_orth"]
    sp_w["even_draws"][0] += 0.005
    sp_w["odd_draws"][0] += 0.005
    sp_w["d_pos_med"] = float(statistics.median(sp_w["d_pos_draws"]))
    sp_w["even_med"] = float(statistics.median(sp_w["even_draws"]))
    sp_w["odd_med"] = float(statistics.median(sp_w["odd_draws"]))
    sp_w["even_max_abs"] = max(abs(x) for x in sp_w["even_draws"])
    sp_w["odd_max_abs"] = max(abs(x) for x in sp_w["odd_draws"])
    try:
        score_full_receipt(nl, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=True)
        raise AssertionError("Expected rejection of broken per-draw linear decomposition")
    except AssertionError as e:
        if "Head linear decomposition violated" not in str(e):
            raise

    # 7c. Alden #104 Witness 2 (Paired Red/Green): Smoke verifier requires measured counts, mode/model/units, timing & provenance.
    smoke_keys_set = {(e["bank"], e["prompt_id"], rib.dose_key(e["dose_mag"])) for e in manifest["schedule_smoke"]}
    den_sm = manifest["meta"]["denominators"]["smoke"]
    smoke_base = {
        "meta": {
            "schema_version": rib.SCHEMA_VERSION,
            "artifact_type": "instruct_bridge_smoke_receipt",
            "run_mode": "smoke_1unit",
            "smoke_1unit": True,
            "execution_provenance": "synthetic_anchor_fixture",
            "device_name": "synthetic_fixture",
            "complete": True,
            "model_repo": rib.MODEL_REPO,
            "model_commit": rib.MODEL_COMMIT,
            "dtype": rib.DTYPE_STR,
            "attn_implementation": rib.ATTN_IMPL,
            "design_sha256": pins["design_sha256"],
            "manifest_sha256": pins["manifest_sha256"],
            "runner_sha256": pins["runner_sha256"],
            "scorer_sha256": pins["scorer_sha256"],
            "pins": dict(pins),
            "units_run": [rib.SMOKE_UNIT_ID],
            "n_unpatched_rows": den_sm["n_unpatched_rows"],
            "n_head_rows": den_sm["n_head_rows"],
            "n_b19_forwards": den_sm["n_b19_forwards"],
            "max_abs_null_vec_diff": 0.0,
            "timing": {
                "model_load_sec": 2.0,
                "b19_elapsed_sec": 24.2,
                "non_b19_runner_sec": 3.0,
                "total_runner_sec": 27.2,
                "n_b19_done": den_sm["n_b19_forwards"],
                "ms_per_b19": 100.0,
                "projected_full_b19_sec": 696.0,
            },
        },
        "unpatched_rows": [
            r for r in good["unpatched_rows"]
            if r["unit"] == rib.SMOKE_UNIT_ID and (r["bank"], r["prompt_id"], rib.dose_key(r["dose_mag"])) in smoke_keys_set
        ],
        "head_rows": [
            r for r in good["head_rows"]
            if r["unit"] == rib.SMOKE_UNIT_ID and (r["bank"], r["prompt_id"], rib.dose_key(r["dose_mag"])) in smoke_keys_set
        ],
    }

    # Red 2a: Exact Alden #104 witness — n_b19_forwards=0, timing=None -> must hard-refuse!
    sm_no_timing = copy.deepcopy(smoke_base)
    sm_no_timing["meta"]["execution_provenance"] = "measured_gpu_run"
    sm_no_timing["meta"]["n_b19_forwards"] = 0
    sm_no_timing["meta"]["timing"] = None
    try:
        verify_smoke_receipt_only(sm_no_timing, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=False)
        raise AssertionError("Expected refusal of smoke with n_b19_forwards=0 and timing=None")
    except ValueError:
        pass

    # Red 2b: Synthetic fixture refused for launch verification (allow_fixture=False)!
    try:
        verify_smoke_receipt_only(smoke_base, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=False)
        raise AssertionError("Expected refusal of synthetic_anchor_fixture for launch smoke verification")
    except ValueError as e:
        assert "measured_gpu_run" in str(e)

    # Red 2c: Wrong model_commit refused!
    sm_bad_model = copy.deepcopy(smoke_base)
    sm_bad_model["meta"]["execution_provenance"] = "measured_gpu_run"
    sm_bad_model["meta"]["model_commit"] = "0" * 40
    try:
        verify_smoke_receipt_only(sm_bad_model, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=False)
        raise AssertionError("Expected refusal of smoke with wrong model_commit")
    except ValueError as e:
        assert "model_repo/model_commit" in str(e)

    # Green 2a: Fixture-only verification clearly labeled fixture_only_not_launch_evidence.
    sm_fix = verify_smoke_receipt_only(smoke_base, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=True)
    assert sm_fix["provenance_mode"] == "fixture_only_not_launch_evidence"

    # Green 2b: Complete measured_gpu_run smoke with exact 18/112/242, finite timing, and 24/21/4 anchors passes!
    sm_gpu = copy.deepcopy(smoke_base)
    sm_gpu["meta"]["execution_provenance"] = "measured_gpu_run"
    sm_gpu["meta"]["device_name"] = "NVIDIA L4"
    sm_chk = verify_smoke_receipt_only(sm_gpu, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=False)
    assert sm_chk["smoke_technical_check_passed"] is True
    assert sm_chk["provenance_mode"] == "measured_gpu_run"
    assert sm_chk["anchor_subset_check"]["join_1_old_head"]["n_rows_joined"] == 24
    assert sm_chk["anchor_subset_check"]["join_2a_new_head"]["n_rows_joined"] == 21
    assert sm_chk["anchor_subset_check"]["join_2b_new_unpatched"]["n_rows_joined"] == 4

    return {
        "scorer_self_tests_passed": True,
        "witness_1_derived_null_and_peers_red_green_verified": True,
        "witness_2_smoke_meta_timing_provenance_red_green_verified": True,
        "witness_3_loaded_runner_module_binding_red_green_verified": True,
        "offline_fixture_anchor_joins_verified": {
            "old_head_rows": 480,
            "new_head_rows": 910,
            "new_unpatched_rows": 160,
            "smoke_old_head_rows": 24,
            "smoke_new_head_rows": 21,
            "smoke_new_unpatched_rows": 4,
        },
        "sham_controls_verified": True,
        "nonlinear_median_decomposition_control_verified": True,
        "anchor_failure_unified_disposition_verified": True,
        "br2_br3_br4_exhaustive_branches_verified": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OLMo 2 1B Instruct 2x2 Bank x Norm Bridge Offline Scorer")
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--verify-smoke-only", action="store_true", help="Technical check of measured GPU smoke receipt (zero verdicts)")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--receipt", type=Path, default=None)
    ap.add_argument("--old-receipt", type=Path, default=None)
    ap.add_argument("--new-receipt", type=Path, default=None)
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--runner", type=Path, default=None)
    ap.add_argument("--design-sha256", type=str, default="")
    ap.add_argument("--manifest-sha256", type=str, default="")
    ap.add_argument("--runner-sha256", type=str, default="")
    ap.add_argument("--scorer-sha256", type=str, default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    if args.self_test_only:
        print(json.dumps(run_self_tests(repo_root), indent=2))
        return

    if args.receipt is None:
        raise ValueError("--receipt is required")
    design_p = args.design or (repo_root / rib.DESIGN_REL_PATH)
    manifest_p = args.manifest or (repo_root / "explore_olmo2_attn_mlp/instruct_bridge_manifest.json")
    scorer_p = Path(__file__).resolve()
    old_rcpt_p = args.old_receipt or (repo_root / "explore_olmo2_attn_mlp/results_signed_dose_full_d13fc3dd.json")
    new_rcpt_p = args.new_receipt or (repo_root / "explore_olmo2_attn_mlp/results_stage_sweep_instruct_01bd41d9.json")

    manifest, pins = admit_scorer_package(
        design_path=design_p,
        manifest_path=manifest_p,
        runner_path=args.runner,
        scorer_path=scorer_p,
        expected_design_sha=args.design_sha256,
        expected_manifest_sha=args.manifest_sha256,
        expected_runner_sha=args.runner_sha256,
        expected_scorer_sha=args.scorer_sha256,
    )
    rib.verify_literal_source_bindings(
        manifest=manifest,
        old_manifest_bytes=rib.load_git_blob(repo_root, rib.OLD_GIT_COMMIT, "explore_olmo2_attn_mlp/signed_dose_manifest.json"),
        old_runner_bytes=rib.load_git_blob(repo_root, rib.OLD_GIT_COMMIT, "explore_olmo2_attn_mlp/run_signed_dose.py"),
        new_manifest_bytes=rib.load_git_blob(repo_root, rib.NEW_GIT_COMMIT, "explore_olmo2_attn_mlp/stage_sweep_manifest.json"),
        new_runner_bytes=rib.load_git_blob(repo_root, rib.NEW_GIT_COMMIT, "explore_olmo2_attn_mlp/run_stage_sweep.py"),
        new_receipt_bytes=new_rcpt_p.read_bytes(),
        old_receipt_bytes=old_rcpt_p.read_bytes(),
    )
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))

    if args.verify_smoke_only:
        res = verify_smoke_receipt_only(receipt, manifest, pins, old_rcpt_p, new_rcpt_p, allow_fixture=False)
        print(json.dumps(res, indent=2))
        if not res["smoke_technical_check_passed"]:
            sys.exit(1)
        return

    scorecard = score_full_receipt(
        receipt=receipt,
        manifest=manifest,
        pins=pins,
        old_receipt_path=old_rcpt_p,
        new_receipt_path=new_rcpt_p,
        out_path=args.out,
        allow_fixture=False,
    )
    print(json.dumps({
        "BR0": scorecard["BR0_anchor_replay"]["status"],
        "BR2": scorecard["BR2_bank_x_norm_L8H15"]["primary_L8H15_spec"]["verdict"],
        "BR3_OLD_1.6": scorecard["BR3_high_dose_transport"]["banks"]["OLD_BANK"]["cross_norm"]["1.6"],
        "BR3_NEW_1.6": scorecard["BR3_high_dose_transport"]["banks"]["NEW_BANK"]["cross_norm"]["1.6"],
        "BR4_part_b_norm_decision": scorecard["BR4_L7H5_and_part_b_norm"]["part_b_norm_decision"],
        "BR4_p92_09_part_b_track": scorecard["BR4_L7H5_and_part_b_norm"]["p92_09_part_b_track"],
        "out": str(args.out) if args.out else None,
    }, indent=2))


if __name__ == "__main__":
    main()
