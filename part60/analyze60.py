#!/usr/bin/env python3
"""Analyze Part 60 sentence tiers, held-out predictions, costs, and parse failures."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
BASIN = Path(os.environ.get("PART60_BASIN", HERE.parent)).resolve()
DEFAULT_ROOT = BASIN / "results" / "part60"


class AnalysisError(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise AnalysisError(f"{path}:{line_no} is not an object")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def auc(pos: list[float], neg: list[float]) -> float | None:
    pos = [float(x) for x in pos if math.isfinite(float(x))]
    neg = [float(x) for x in neg if math.isfinite(float(x))]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def phase_for_metric(method: str) -> str:
    """M3 is evaluated on fresh sentences; M5 is evaluated on round 2 only."""
    if method == "M3":
        return "fresh"
    if method == "M5":
        return "round2"
    return "primary"


def tier_statistics(strong: list[float], weak: list[float], none: list[float], threshold: float,
                    *, parsed: bool = True, m0: bool = False) -> dict[str, Any]:
    """Return per-unit tier hits. Missing or unparsable output is never verified."""
    if not parsed or not strong:
        return {"parsed": False, "verified": False, "h_strong": None, "h_weak": None, "h_none": None,
                "ordering": None, "auc": None, "any_strong_hit": False, "strong_marked": None}
    strong = [float(x) for x in strong]
    weak = [float(x) for x in weak]
    none = [float(x) for x in none]
    if any(not math.isfinite(x) for x in strong + weak + none):
        return {"parsed": False, "verified": False, "h_strong": None, "h_weak": None, "h_none": None,
                "ordering": None, "auc": None, "any_strong_hit": False, "strong_marked": None}
    h_s = sum(x > threshold for x in strong) / len(strong)
    h_w = sum(x > threshold for x in weak) / len(weak) if weak else None
    h_n = sum(x > threshold for x in none) / len(none) if none else None
    verified = h_s >= 0.75 if m0 else h_s >= 0.75 and h_n is not None and h_n <= 0.25
    ordering = (h_s > h_w > h_n) if h_w is not None and h_n is not None else None
    return {"parsed": True, "verified": bool(verified), "h_strong": float(h_s), "h_weak": h_w,
            "h_none": h_n, "ordering": ordering, "auc": auc(strong, none),
            "any_strong_hit": any(x > threshold for x in strong)}


def tier_statistics_with_marked(strong: list[float], weak: list[float], none: list[float],
                                marked: list[float], threshold: float, *, parsed: bool = True) -> dict[str, Any]:
    result = tier_statistics(strong, weak, none, threshold, parsed=parsed)
    if not result["parsed"] or len(marked) != len(strong) or any(not math.isfinite(float(x)) for x in marked):
        result["strong_marked"] = None
        result["marked_verified"] = None
        result["whole_strong_verified"] = None
        result["marked_comparable"] = False
        return result
    h_marked = sum(float(x) > threshold for x in marked) / len(marked)
    result["strong_marked"] = float(h_marked)
    result["marked_verified"] = bool(h_marked >= 0.75)
    result["whole_strong_verified"] = bool(result["h_strong"] >= 0.75)
    result["marked_comparable"] = True
    return result


def _score_file(root: Path, split: str, method: str, phase: str) -> Path:
    import score60
    return score60.score_path(root, split, method, phase)


def load_acts(path: Path) -> tuple[Any, dict[str, int], dict[str, int]]:
    data = np.load(path, allow_pickle=False)
    unit_keys = [str(x) for x in data["unit_keys"]]
    sids = [str(sid) for sid in data["sids"]]
    if len(set(sids)) != len(sids) or len(set(unit_keys)) != len(unit_keys):
        raise AnalysisError(f"activation matrix has duplicate sentence or unit keys: {path}")
    if data["max"].ndim != 2 or data["max"].shape != (len(sids), len(unit_keys)):
        raise AnalysisError(f"activation matrix dimensions do not match its keys: {path}")
    if data["marked"].shape != data["max"].shape:
        raise AnalysisError(f"marked activation dimensions do not match whole-text values: {path}")
    sid_rows = {sid: i for i, sid in enumerate(sids)}
    unit_cols = {key: j for j, key in enumerate(unit_keys)}
    return data, sid_rows, unit_cols


def _matrix_index(path: Path) -> tuple[Any, dict[str, int], dict[str, int]]:
    return load_acts(path)


def _tier_values(data: Any, sid_rows: dict[str, int], unit_cols: dict[str, int],
                 record_index: int, unit_key: str, method: str, phase: str,
                 tier: str) -> tuple[list[float], list[float]]:
    values: list[float] = []
    marked_values: list[float] = []
    col = unit_cols.get(unit_key)
    if col is None:
        return [], []
    for j in range(8):
        sid = f"{record_index}|{method}|{phase}|{tier}|{j}"
        row = sid_rows.get(sid)
        if row is None:
            break
        value = float(data["max"][row, col])
        if not math.isfinite(value):
            break
        values.append(value)
        marked_values.append(float(data["marked"][row, col]))
    return values, marked_values


def unit_metric(row: dict[str, Any] | None, data: Any | None, sid_rows: dict[str, int],
                unit_cols: dict[str, int], record_index: int, unit_key: str, method: str,
                phase: str, threshold: float) -> dict[str, Any]:
    if method == "M3":
        output = (row or {}).get("fresh", {})
    elif method == "M5":
        output = (row or {}).get(phase, {})
    else:
        output = row or {}
    parsed = isinstance(output, dict) and output.get("parse_status") == "ok"
    if data is None or not parsed:
        return tier_statistics([], [], [], threshold, parsed=False, m0=method == "M0")
    strong, marked = _tier_values(data, sid_rows, unit_cols, record_index, unit_key, method, phase, "strong")
    weak, _ = _tier_values(data, sid_rows, unit_cols, record_index, unit_key, method, phase, "weak")
    none, _ = _tier_values(data, sid_rows, unit_cols, record_index, unit_key, method, phase, "none")
    if len(strong) != 4 or (method != "M0" and len(none) != 4):
        return tier_statistics([], [], [], threshold, parsed=False, m0=method == "M0")
    if ((method in ("M1", "M2", "M6", "M7") and phase == "primary") or
            (method == "M5" and phase == "round1")) and len(weak) != 4:
        return tier_statistics([], [], [], threshold, parsed=False)
    if method == "M4":
        return tier_statistics_with_marked(strong, weak, none, marked, threshold, parsed=True)
    return tier_statistics(strong, weak, none, threshold, parsed=True, m0=method == "M0")


def cost_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = output_tokens = calls = 0
    for row in rows:
        usage = row.get("usage_tokens", {})
        input_tokens += int(usage.get("input", 0))
        output_tokens += int(usage.get("output", 0))
        calls += int(usage.get("calls", 0))
    usd = (input_tokens * 0.05 + output_tokens * 0.25) / 1_000_000
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "calls": calls,
            "usd": usd, "usd_per_unit": usd / len(rows) if rows else None}


def _paired_hstrong(metrics: dict[str, dict[str, Any]]) -> tuple[float | None, float | None, int]:
    methods = {method: {row["unit_key"]: row for row in metrics.get(method, {}).get("unit_rows", [])
                        if row.get("type") == "neuron"} for method in ("M0", "M1")}
    paired = [(methods["M1"][key]["h_strong"], methods["M0"][key]["h_strong"])
              for key in methods["M1"].keys() & methods["M0"].keys()
              if methods["M1"][key].get("parsed") and methods["M0"][key].get("parsed")
              and methods["M1"][key].get("h_strong") is not None
              and methods["M0"][key].get("h_strong") is not None]
    if not paired:
        return None, None, 0
    return float(np.mean([x for x, _ in paired])), float(np.mean([y for _, y in paired])), len(paired)


def aggregate_method(root: Path, split: str, method: str, units: list[dict[str, Any]],
                     thresholds: dict[str, float]) -> dict[str, Any] | None:
    gen_path = root / "out" / split / f"{method}.jsonl"
    if not gen_path.exists():
        return None
    generation_rows = read_jsonl(gen_path)
    by_record = {int(row["record_index"]): row for row in generation_rows}
    phase = phase_for_metric(method)
    activation_path = _score_file(root, split, method, phase)
    if not activation_path.exists():
        return None
    data, sid_rows, unit_cols = load_acts(activation_path)
    metrics: list[dict[str, Any]] = []
    for unit in units:
        rec = int(unit["record_index"])
        row = by_record.get(rec)
        threshold = thresholds.get(unit["unit_key"])
        if threshold is None:
            metric = tier_statistics([], [], [], 0.0, parsed=False, m0=method == "M0")
            parse_status = "threshold_missing"
        else:
            metric = unit_metric(row, data, sid_rows, unit_cols, rec, unit["unit_key"], method, phase, threshold)
            parse_status = (row or {}).get("parse_status", "missing")
        metrics.append({**unit, **metric, "parse_status": parse_status})
    total = len(units)
    out: dict[str, Any] = {"method": method, "split": split, "n_units": total,
                           "n_parseable": sum(m["parsed"] for m in metrics),
                           "n_failed": total - sum(m["parsed"] for m in metrics),
                           "n_scored": int(data["max"].shape[0]) if data is not None else 0,
                           "phase_evaluated": phase, "cost": cost_for(generation_rows), "by_type": {},
                           "unit_rows": metrics}
    for typ in ("neuron", "control"):
        selected = [m for m in metrics if m["type"] == typ]
        valid = [m for m in selected if m["parsed"]]
        marked_comparable = [m for m in selected if m.get("marked_comparable")]
        def mean_field(name: str) -> float | None:
            values = [m[name] for m in valid if m.get(name) is not None]
            return float(np.mean(values)) if values else None
        out["by_type"][typ] = {
            "n": len(selected), "n_parseable": len(valid),
            "verified_rate": sum(bool(m["verified"]) for m in selected) / len(selected) if selected else None,
            "mean_h_strong": mean_field("h_strong"), "mean_h_weak": mean_field("h_weak"),
            "mean_h_none": mean_field("h_none"), "mean_auc": mean_field("auc"),
            "mean_strong_marked": mean_field("strong_marked"),
            "marked_comparable_n": len(marked_comparable) if method == "M4" else None,
            "marked_verified_rate": sum(bool(m.get("marked_verified")) for m in marked_comparable) / len(marked_comparable) if marked_comparable else None,
            "whole_strong_verified_rate": sum(bool(m.get("whole_strong_verified")) for m in marked_comparable) / len(marked_comparable) if marked_comparable else None,
        }
    eligible = [m for m in metrics if m["type"] == "neuron" and m["parsed"] and m["any_strong_hit"] and m["ordering"] is not None]
    out["real_tier_ordering_share"] = sum(bool(m["ordering"]) for m in eligible) / len(eligible) if eligible else None
    out["real_tier_ordering_n"] = len(eligible)
    out["all_units_verified"] = sum(bool(m["verified"]) for m in metrics) / total if total else None
    return out


def aggregate_m5_round1(root: Path, split: str, units: list[dict[str, Any]],
                        thresholds: dict[str, float]) -> dict[str, Any] | None:
    gen_path = root / "out" / split / "M5.jsonl"
    if not gen_path.exists():
        return None
    rows = read_jsonl(gen_path)
    by_record = {int(row["record_index"]): row for row in rows}
    act_path = _score_file(root, split, "M5", "round1")
    if not act_path.exists():
        return None
    data, sid_rows, unit_cols = load_acts(act_path)
    result = []
    for unit in units:
        row = by_record.get(int(unit["record_index"]), {})
        threshold = thresholds.get(unit["unit_key"])
        if threshold is None:
            metric = tier_statistics([], [], [], 0.0, parsed=False)
        else:
            metric = unit_metric(row, data, sid_rows, unit_cols, int(unit["record_index"]), unit["unit_key"],
                                 "M5", "round1", threshold)
        result.append({**unit, **metric})
    out: dict[str, Any] = {"method": "M5-round1", "split": split, "phase_evaluated": "round1", "n_units": len(units),
                           "n_parseable": sum(m["parsed"] for m in result), "n_failed": len(units)-sum(m["parsed"] for m in result),
                           "by_type": {}, "unit_rows": result}
    for typ in ("neuron", "control"):
        selected = [m for m in result if m["type"] == typ]
        valid = [m for m in selected if m["parsed"]]
        out["by_type"][typ] = {"n": len(selected), "n_parseable": len(valid),
                               "verified_rate": sum(bool(m["verified"]) for m in selected)/len(selected) if selected else None,
                               "mean_h_strong": float(np.mean([m["h_strong"] for m in valid])) if valid else None}
    return out


def generation_failure_report(rows: list[dict[str, Any]], method: str, expected: int) -> list[dict[str, Any]]:
    """Report complete generation phases over the registered failure limit."""
    import gen60
    if method == "M7" or expected <= 0:
        return []
    phases = {"M3": ("selection", "fresh"), "M5": ("round1", "round2")}.get(method, ("primary",))
    by_record = {int(row["record_index"]): row for row in rows if "record_index" in row}
    if len(by_record) != expected:
        return []
    failures = []
    for phase in phases:
        if method in ("M3", "M5"):
            entries = [by_record[i].get(phase) for i in sorted(by_record)]
        else:
            entries = [by_record[i] for i in sorted(by_record)]
        # A phase that has not run yet is pending, not a generation failure.
        if not all(isinstance(entry, dict) and "parse_status" in entry for entry in entries):
            continue
        successes = sum(entry.get("parse_status") == "ok" for entry in entries)
        if gen60.generation_failure_exceeds_ten_percent(successes, expected):
            failures.append({"method": method, "phase": phase, "successes": successes,
                             "expected": expected, "failure_rate": (expected-successes)/expected,
                             "reason": "generation parse/refusal failure exceeded 10%"})
    return failures


def truncation_report(rows: list[dict[str, Any]], method: str) -> dict[str, int]:
    phases = {"M3": ("selection", "fresh"), "M5": ("round1", "round2")}.get(method, ("primary",))
    result: dict[str, int] = {}
    for phase in phases:
        count = 0
        for row in rows:
            entry = row.get(phase, {}) if method in ("M3", "M5") else row
            if isinstance(entry, dict) and (entry.get("raw_parse_status") == "max_tokens" or entry.get("stop_reason") == "max_tokens"):
                count += 1
        if count:
            result[phase] = count
    return result


def _percent(value: float | None) -> str:
    return "unscored" if value is None else f"{100*value:.1f}%"


def prediction_table(metrics: dict[str, dict[str, Any]], m5_round1: dict[str, Any] | None,
                     split: str, best_dev: str | None = None) -> list[dict[str, Any]]:
    """Registered predictions: 60a-60i use dev; 60j uses test."""
    dev = metrics if split == "dev" else {}
    test = metrics if split == "test" else {}

    def get(source: dict[str, dict[str, Any]], method: str, typ: str, field: str) -> Any:
        return source.get(method, {}).get("by_type", {}).get(typ, {}).get(field)

    rows: list[dict[str, Any]] = []

    def add(pid: str, description: str, condition: bool | None, evidence: str) -> None:
        rows.append({"id": pid, "prediction": description,
                     "status": "unscored" if condition is None else "held" if condition else "failed",
                     "evidence": evidence})

    m0r, m0c = get(dev, "M0", "neuron", "verified_rate"), get(dev, "M0", "control", "verified_rate")
    a_ok = None if m0r is None or m0c is None else abs(m0r - 0.281) <= 0.03 and abs(m0c - 0.062) <= 0.02
    add("60a", "M0 pass rates within 3 pp of 28.1% real and 2 pp of 6.2% control", a_ok,
        f"real={_percent(m0r)}; control={_percent(m0c)}")

    m1r, m1c = get(dev, "M1", "neuron", "verified_rate"), get(dev, "M1", "control", "verified_rate")
    b_ok = None if m1r is None or m1c is None else m1r >= 0.15 and m1c <= 0.03
    add("60b", "M1 verified rate at least 15% real and at most 3% control", b_ok,
        f"real={_percent(m1r)}; control={_percent(m1c)}")

    hs1, hs0, paired_n = _paired_hstrong(dev)
    c_ok = None if hs1 is None or hs0 is None else hs1 - hs0 >= 0.10
    diff = hs1 - hs0 if hs1 is not None and hs0 is not None else None
    add("60c", "M1 real strong-tier hit rate exceeds M0 by at least 10 pp", c_ok,
        f"M1={_percent(hs1)}; M0={_percent(hs0)}; difference={_percent(diff)} on the same {paired_n} real units")

    order = dev.get("M1", {}).get("real_tier_ordering_share")
    add("60d", "M1 strong > weak > none ordering in at least 70% of eligible real units",
        None if order is None else order >= 0.70,
        f"ordering={_percent(order)}; n={dev.get('M1', {}).get('real_tier_ordering_n', 0)}")

    hn2, hn1 = get(dev, "M2", "neuron", "mean_h_none"), get(dev, "M1", "neuron", "mean_h_none")
    e_ok = None if hn2 is None or hn1 is None or hn1 <= 0 else hn2 <= (2 / 3) * hn1
    add("60e", "M2 lowers real mean H_none by at least one third relative to M1", e_ok,
        f"M2={_percent(hn2)}; M1={_percent(hn1)}")

    m3r = get(dev, "M3", "neuron", "verified_rate")
    f_ok = None if m3r is None or m1r is None else abs(m3r - m1r) <= 0.05
    add("60f", "M3 fresh-sentence real verified rate is within 5 pp of M1", f_ok,
        f"M3 fresh={_percent(m3r)}; M1={_percent(m1r)}")

    g_r1 = (m5_round1 or {}).get("by_type", {}).get("neuron", {}).get("verified_rate")
    g_c1 = (m5_round1 or {}).get("by_type", {}).get("control", {}).get("verified_rate")
    g_r2, g_c2 = get(dev, "M5", "neuron", "verified_rate"), get(dev, "M5", "control", "verified_rate")
    g_ok = None if any(x is None for x in (g_r1, g_c1, g_r2, g_c2)) else g_r2-g_r1 >= 0.05 and g_c2-g_c1 > 0.02
    add("60g", "M5 round 2 raises real verified rate by at least 5 pp and control rate by more than 2 pp", g_ok,
        f"real round1={_percent(g_r1)}, round2={_percent(g_r2)}; control round1={_percent(g_c1)}, round2={_percent(g_c2)}")

    m4m, m4w = get(dev, "M4", "neuron", "marked_verified_rate"), get(dev, "M4", "neuron", "whole_strong_verified_rate")
    h_ok = None if m4m is None or m4w is None else m4m-m4w >= 0.10
    add("60h", "M4 marked-token threshold pass exceeds whole-text pass by at least 10 pp", h_ok,
        f"marked={_percent(m4m)}; whole-text={_percent(m4w)}")

    m6r = get(dev, "M6", "neuron", "verified_rate")
    c6 = dev.get("M6", {}).get("cost", {}).get("usd_per_unit")
    c1 = dev.get("M1", {}).get("cost", {}).get("usd_per_unit")
    i_ok = None if m6r is None or m1r is None or c6 is None or c1 is None or c1 <= 0 else abs(m6r-m1r) <= 0.03 and c6 >= 2*c1
    add("60i", "M6 differs from M1 by at most 3 pp in real verified rate and costs at least twice as much", i_ok,
        f"M6={_percent(m6r)}; M1={_percent(m1r)}; cost/unit M6={c6}; M1={c1}")

    if split != "test":
        add("60j", "No tested method beats both M1 and M0 test gaps by at least 5 pp", None,
            "unscored until the registered test phase")
    else:
        m0r_t, m0c_t = get(test, "M0", "neuron", "verified_rate"), get(test, "M0", "control", "verified_rate")
        m1r_t, m1c_t = get(test, "M1", "neuron", "verified_rate"), get(test, "M1", "control", "verified_rate")
        cr = get(test, best_dev, "neuron", "verified_rate") if best_dev else None
        cc = get(test, best_dev, "control", "verified_rate") if best_dev else None
        vals = (m0r_t, m0c_t, m1r_t, m1c_t, cr, cc)
        if any(x is None for x in vals):
            j_ok = None
            evidence = f"test candidate={best_dev or 'unknown'}; M0, M1, or candidate rates missing"
        else:
            gap0, gap1, gapx = m0r_t-m0c_t, m1r_t-m1c_t, cr-cc
            improved = gapx >= gap0 + 0.05 and (best_dev == "M1" or gapx >= gap1 + 0.05)
            j_ok = not improved
            comparator = "M0 only (candidate is M1)" if best_dev == "M1" else "M0 and M1"
            evidence = f"tested candidate={best_dev}; gaps M0={_percent(gap0)}, M1={_percent(gap1)}, candidate={_percent(gapx)}; required lead over {comparator}; recommendation={'candidate' if improved else 'M0'}"
        add("60j", "No tested method beats both M1 and M0 test gaps by at least 5 pp", j_ok, evidence)
    return rows


def choose_dev_leader(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = metrics.get("M0", {}).get("by_type", {})
    base_r = base.get("neuron", {}).get("verified_rate")
    base_c = base.get("control", {}).get("verified_rate")
    base_gap = base_r-base_c if base_r is not None and base_c is not None else None
    candidates = []
    for method, value in metrics.items():
        if method == "M0" or value.get("n_units", 0) <= 0:
            continue
        if value.get("n_parseable", 0) / value["n_units"] < 0.90:
            continue
        by = value.get("by_type", {})
        real = by.get("neuron", {}).get("verified_rate")
        control = by.get("control", {}).get("verified_rate")
        if real is not None and control is not None:
            candidates.append((real-control, method, real, control))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    if not candidates or base_gap is None:
        return {"leader": None, "best_variant": None, "base_gap": base_gap, "best_gap": None,
                "gain": None, "stop_rule": "unscored", "test_candidate": None}
    best_gap, method, real, control = candidates[0]
    gain = best_gap-base_gap
    chosen = method if gain >= 0.05 else "M0"
    return {"leader": chosen, "best_variant": method, "base_gap": base_gap, "best_gap": best_gap,
            "gain": gain, "stop_rule": "continue_to_test" if chosen != "M0" else "keep_M0_under_5pp_gain",
            "best_variant_real": real, "best_variant_control": control,
            "test_candidate": chosen if chosen != "M0" else None}


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Prediction | Status | Evidence |", "|---|---|---|"]
    for row in rows:
        evidence = str(row["evidence"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['id']} — {row['prediction']} | {row['status']} | {evidence} |")
    return "\n".join(lines)


def run_analysis(root: Path, split: str) -> dict[str, Any]:
    import gen60
    import score60
    thresholds = score60.ensure_thresholds(root)
    units = [{k: u[k] for k in ("record_index", "unit_key", "type", "layer")} for u in gen60.load_splits()[split]]
    metrics: dict[str, dict[str, Any]] = {}
    m5_round1 = None
    skipped = []
    for method in [f"M{i}" for i in range(8)]:
        state_path = root / "out" / split / f"{method}.state.json"
        state = read_json(state_path) if state_path.is_file() else {}
        if state.get("status") == "dropped":
            skipped.append({"method": method, "reason": state.get("reason", "method marked dropped")})
            continue
        value = aggregate_method(root, split, method, units, thresholds)
        if value is not None:
            metrics[method] = value
        if method == "M5":
            m5_round1 = aggregate_m5_round1(root, split, units, thresholds)
    dropped_generation = []
    for method in [f"M{i}" for i in range(1, 8)]:
        path = root / "out" / split / f"{method}.jsonl"
        if path.exists():
            rows = read_jsonl(path)
            dropped_generation.extend(generation_failure_report(rows, method, len(units)))
    dropped_methods = {item["method"] for item in dropped_generation}
    for method in dropped_methods:
        metrics.pop(method, None)
        if method == "M5":
            m5_round1 = None
    prediction_metrics = {method: value for method, value in metrics.items() if method not in dropped_methods}
    prediction_m5_round1 = None if "M5" in dropped_methods else m5_round1
    leader = choose_dev_leader(prediction_metrics) if split == "dev" else None
    predictions = prediction_table(prediction_metrics, prediction_m5_round1, split,
                                   leader.get("test_candidate") if leader else None)
    for method in [f"M{i}" for i in range(1, 8)]:
        if any(item["method"] == method for item in skipped):
            continue
        path = root / "out" / split / f"{method}.jsonl"
        if path.exists():
            rows = read_jsonl(path)
            if rows and all(row.get("parse_status") == "skipped" for row in rows):
                skipped.append({"method": method, "reason": rows[0].get("skip_reason", "skipped")})
    truncations = {}
    for method in [f"M{i}" for i in range(1, 8)]:
        path = root / "out" / split / f"{method}.jsonl"
        if path.is_file():
            counts = truncation_report(read_jsonl(path), method)
            if counts:
                truncations[method] = counts
    return {"split": split, "metrics": metrics, "m5_round1": m5_round1, "dev_selection": leader,
            "predictions": predictions, "skipped_methods": skipped,
            "dropped_generation": dropped_generation,
            "max_tokens_truncations": truncations,
            "definitions_file": "part60/DEFINITIONS-60.md"}


def write_reports(root: Path, analyses: list[dict[str, Any]]) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    payload = {"analyses": analyses, "predictions": [p for a in analyses for p in a["predictions"]]}
    json_path, md_path = root / "analysis-60.json", root / "analysis-60.md"
    write_json(json_path, payload)
    dev = next((analysis for analysis in analyses if analysis.get("split") == "dev"), None)
    if dev is not None:
        selection = dev.get("dev_selection") or {}
        candidate = selection.get("test_candidate")
        methods = ["M0", "M1"]
        if candidate and candidate not in methods:
            methods.append(candidate)
        write_json(root / "test-selection.json", {
            "status": "ready", "created_at": datetime.now(timezone.utc).isoformat(),
            "candidate": candidate, "methods": methods, "dev_selection": selection,
            "rule": "generated by analyze60.py --split dev; M0, M1, and the single candidate selected by the registered five-point dev rule",
        })
    lines = ["# Part 60 analysis", "", "Metrics use the frozen definitions in `part60/DEFINITIONS-60.md`.", ""]
    for analysis in analyses:
        split = analysis["split"]
        lines.extend([f"## {split}", "", "| Method | Parsed | Failed | Real verified | Control verified | Real Hstrong | Control Hstrong | Real-control gap | Real Hweak | Real Hnone | Real tier order | Real AUC | Calls | Max-token truncations | Cost / unit |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for method, result in analysis["metrics"].items():
            real, control = result["by_type"]["neuron"], result["by_type"]["control"]
            pct = lambda value: "—" if value is None else f"{value:.3f}"
            cost = result["cost"].get("usd_per_unit")
            cost_text = "—" if cost is None else f"${cost:.5f}"
            gap = None if real["verified_rate"] is None or control["verified_rate"] is None else real["verified_rate"] - control["verified_rate"]
            truncations = sum(analysis.get("max_tokens_truncations", {}).get(method, {}).values())
            lines.append(f"| {method} | {result['n_parseable']}/{result['n_units']} | {result['n_failed']} | {pct(real['verified_rate'])} | {pct(control['verified_rate'])} | {pct(real['mean_h_strong'])} | {pct(control['mean_h_strong'])} | {pct(gap)} | {pct(real['mean_h_weak'])} | {pct(real['mean_h_none'])} | {pct(result['real_tier_ordering_share'])} | {pct(real['mean_auc'])} | {result['cost'].get('calls', 0)} | {truncations} | {cost_text} |")
        if analysis.get("m5_round1"):
            r1 = analysis["m5_round1"]["by_type"]
            lines.extend(["", f"M5 round 1 verified: real {pct(r1['neuron']['verified_rate'])}, control {pct(r1['control']['verified_rate'])}; M5 main metrics above use round 2 only."])
        if analysis.get("skipped_methods"):
            lines.extend(["", "Skipped: " + "; ".join(f"{x['method']}: {x['reason']}" for x in analysis["skipped_methods"])])
        if analysis.get("dropped_generation"):
            lines.extend(["", "Dropped by generation failure rule: " + "; ".join(
                f"{x['method']}/{x['phase']} {x['successes']}/{x['expected']} parsed ({100*x['failure_rate']:.1f}% failed)"
                for x in analysis["dropped_generation"])])
        if analysis.get("max_tokens_truncations"):
            lines.extend(["", "Max-token truncations: " + "; ".join(
                f"{method}: " + ", ".join(f"{phase}={count}" for phase, count in phases.items())
                for method, phases in analysis["max_tokens_truncations"].items())])
        if analysis.get("dev_selection"):
            lines.extend(["", f"Dev stop-rule selection: `{json.dumps(analysis['dev_selection'], sort_keys=True)}`"])
        lines.extend(["", "### Predictions", "", markdown_table(analysis["predictions"]), ""])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    args = parser.parse_args(argv)
    try:
        splits = ("dev", "test") if args.split == "all" else (args.split,)
        analyses = [run_analysis(args.root, split) for split in splits]
        if len(analyses) == 2:
            leader = analyses[0].get("dev_selection") or {}
            dropped = {item["method"] for item in analyses[1].get("dropped_generation", [])}
            test_metrics = {method: value for method, value in analyses[1]["metrics"].items() if method not in dropped}
            test_m5_round1 = None if "M5" in dropped else analyses[1].get("m5_round1")
            analyses[1]["predictions"] = prediction_table(test_metrics, test_m5_round1, "test", leader.get("test_candidate"))
        md_path, json_path = write_reports(args.root / "analysis", analyses)
        print(f"wrote {md_path} and {json_path}")
        return 0
    except (AnalysisError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
