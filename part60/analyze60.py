#!/usr/bin/env python3
"""Analyze Part 60 sentence tiers, held-out predictions, costs, and parse failures."""
from __future__ import annotations

import argparse
import hashlib
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
        return {"parsed": False, "positive_only_verified": False, "verified": False, "h_strong": None, "h_weak": None, "h_none": None,
                "ordering": None, "auc": None, "any_strong_hit": False, "strong_marked": None}
    strong = [float(x) for x in strong]
    weak = [float(x) for x in weak]
    none = [float(x) for x in none]
    if any(not math.isfinite(x) for x in strong + weak + none):
        return {"parsed": False, "positive_only_verified": False, "verified": False, "h_strong": None, "h_weak": None, "h_none": None,
                "ordering": None, "auc": None, "any_strong_hit": False, "strong_marked": None}
    h_s = sum(x > threshold for x in strong) / len(strong)
    h_w = sum(x > threshold for x in weak) / len(weak) if weak else None
    h_n = sum(x > threshold for x in none) / len(none) if none else None
    positive_only_verified = h_s >= 0.75
    verified = positive_only_verified if m0 else positive_only_verified and h_n is not None and h_n <= 0.25
    ordering = (h_s > h_w > h_n) if h_w is not None and h_n is not None else None
    return {"parsed": True, "positive_only_verified": bool(positive_only_verified),
            "verified": bool(verified), "h_strong": float(h_s), "h_weak": h_w,
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


def _phases_for_method(method: str) -> tuple[str, ...]:
    return {"M3": ("selection", "fresh"), "M5": ("round1", "round2")}.get(method, ("primary",))


def _raw_results_by_custom_id(root: Path, split: str, method: str,
                              phase: str | None = None) -> dict[str, dict[str, Any]]:
    """Load durable raw API results; custom_id is the deduplication key."""
    import gen60
    phases = (phase,) if phase else _phases_for_method(method)
    results: dict[str, dict[str, Any]] = {}
    for current_phase in phases:
        path = gen60.manifest_path(root, split, method, current_phase)
        if not path.is_file():
            continue
        manifest = read_json(path)
        for chunk in manifest.get("chunks", []):
            raw_path = Path(chunk["directory"]) / "raw-results.jsonl"
            if not raw_path.is_file():
                continue
            for result in read_jsonl(raw_path):
                custom_id = result.get("custom_id")
                if custom_id is not None:
                    results[str(custom_id)] = result
    return results


def _usage_from_raw_result(result: dict[str, Any]) -> dict[str, int] | None:
    wrapper = result.get("result")
    message = wrapper.get("message") if isinstance(wrapper, dict) else None
    usage = message.get("usage") if isinstance(message, dict) else None
    if not isinstance(usage, dict) or "input_tokens" not in usage or "output_tokens" not in usage:
        return None
    try:
        input_tokens, output_tokens = int(usage["input_tokens"]), int(usage["output_tokens"])
    except (TypeError, ValueError):
        return None
    if input_tokens < 0 or output_tokens < 0:
        return None
    return {"input": input_tokens, "output": output_tokens, "calls": 1}


def _sum_raw_usage(results: dict[str, dict[str, Any]]) -> dict[str, int]:
    unique_usage = {custom_id: usage for custom_id, result in results.items()
                    if (usage := _usage_from_raw_result(result)) is not None}
    return {key: sum(value[key] for value in unique_usage.values())
            for key in ("input", "output", "calls")}


def cost_for(rows: list[dict[str, Any]], root: Path | None = None,
             split: str | None = None, method: str | None = None) -> dict[str, Any]:
    import gen60
    batch_usage = {"input": 0, "output": 0, "calls": 0}
    direct_usage = {"input": 0, "output": 0, "calls": 0}
    reserved_usd = 0.0
    if root is not None and split is not None and method is not None:
        batch_usage = _sum_raw_usage(_raw_results_by_custom_id(root, split, method))
        if method == "M6" and split == "dev":
            smoke_path = root / "out" / "smoke-M6-results.jsonl"
            if smoke_path.is_file():
                smoke_results = {}
                for result in read_jsonl(smoke_path):
                    if result.get("custom_id") is not None:
                        smoke_results[str(result["custom_id"])] = result
                direct_usage = _sum_raw_usage(smoke_results)
        ledger_path = gen60.fixed_spend_path(root)
        if ledger_path.is_file():
            _, ledger = gen60.spend_state(ledger_path)
            reserved_usd = sum(float(entry.get("reserved_usd", 0.0)) for entry in ledger.values()
                               if entry.get("event") == "reserve"
                               and entry.get("method") == method and entry.get("split") == split)
    input_tokens = batch_usage["input"] + direct_usage["input"]
    output_tokens = batch_usage["output"] + direct_usage["output"]
    calls = batch_usage["calls"] + direct_usage["calls"]
    usd = ((batch_usage["input"] * gen60.PRICE_INPUT_BATCH
            + batch_usage["output"] * gen60.PRICE_OUTPUT_BATCH
            + direct_usage["input"] * gen60.PRICE_INPUT_DIRECT
            + direct_usage["output"] * gen60.PRICE_OUTPUT_DIRECT) / 1_000_000)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "calls": calls,
            "usd": usd, "reserved_usd": reserved_usd,
            "usd_per_unit": usd / len(rows) if rows else None}


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
                     thresholds: dict[str, float],
                     activation_path: Path | None = None) -> dict[str, Any] | None:
    gen_path = root / "out" / split / f"{method}.jsonl"
    if not gen_path.exists():
        return None
    generation_rows = read_jsonl(gen_path)
    by_record = {int(row["record_index"]): row for row in generation_rows}
    phase = phase_for_metric(method)
    activation_path = activation_path or _score_file(root, split, method, phase)
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
                           "phase_evaluated": phase, "cost": cost_for(generation_rows, root, split, method), "by_type": {},
                           "n_threshold_missing": sum(m["parse_status"] == "threshold_missing" for m in metrics),
                           "unit_rows": metrics}
    for typ in ("neuron", "control"):
        selected = [m for m in metrics if m["type"] == typ]
        valid = [m for m in selected if m["parsed"]]
        marked_comparable = [m for m in selected if m.get("marked_comparable")]
        def mean_field(name: str) -> float | None:
            values = [m[name] for m in valid if m.get(name) is not None]
            return float(np.mean(values)) if values else None
        def mean_count(name: str) -> int:
            return sum(m.get(name) is not None for m in valid)
        out["by_type"][typ] = {
            "n": len(selected), "n_parseable": len(valid),
            "n_threshold_missing": sum(m["parse_status"] == "threshold_missing" for m in selected),
            "positive_only_verified_rate": sum(bool(m["positive_only_verified"]) for m in selected) / len(selected) if selected else None,
            "verified_rate": sum(bool(m["verified"]) for m in selected) / len(selected) if selected else None,
            "mean_h_strong": mean_field("h_strong"), "mean_h_weak": mean_field("h_weak"),
            "mean_h_none": mean_field("h_none"), "mean_auc": mean_field("auc"),
            "mean_strong_marked": mean_field("strong_marked"),
            "n_mean_h_strong": mean_count("h_strong"), "n_mean_h_weak": mean_count("h_weak"),
            "n_mean_h_none": mean_count("h_none"), "n_mean_auc": mean_count("auc"),
            "n_mean_strong_marked": mean_count("strong_marked"),
            "marked_comparable_n": len(marked_comparable) if method == "M4" else None,
            "marked_verified_rate": sum(bool(m.get("marked_verified")) for m in marked_comparable) / len(marked_comparable) if marked_comparable else None,
            "whole_strong_verified_rate": sum(bool(m.get("whole_strong_verified")) for m in marked_comparable) / len(marked_comparable) if marked_comparable else None,
        }
    eligible = [m for m in metrics if m["type"] == "neuron" and m["parsed"] and m["any_strong_hit"] and m["ordering"] is not None]
    eligible_two_hits = [m for m in metrics if m["type"] == "neuron" and m["parsed"]
                         and m.get("h_strong") is not None and m["h_strong"] >= 0.5
                         and m["ordering"] is not None]
    out["real_tier_ordering_share"] = sum(bool(m["ordering"]) for m in eligible) / len(eligible) if eligible else None
    out["real_tier_ordering_n"] = len(eligible)
    out["real_tier_ordering_two_hits_share"] = (sum(bool(m["ordering"]) for m in eligible_two_hits) / len(eligible_two_hits)
                                                if eligible_two_hits else None)
    out["real_tier_ordering_two_hits_n"] = len(eligible_two_hits)
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
                               "positive_only_verified_rate": sum(bool(m["positive_only_verified"]) for m in selected)/len(selected) if selected else None,
                               "verified_rate": sum(bool(m["verified"]) for m in selected)/len(selected) if selected else None,
                               "mean_h_strong": float(np.mean([m["h_strong"] for m in valid])) if valid else None}
    eligible = [m for m in result if m["type"] == "neuron" and m["parsed"]
                and m["any_strong_hit"] and m["ordering"] is not None]
    eligible_two_hits = [m for m in result if m["type"] == "neuron" and m["parsed"]
                         and m.get("h_strong") is not None and m["h_strong"] >= 0.5
                         and m["ordering"] is not None]
    out["real_tier_ordering_share"] = sum(bool(m["ordering"]) for m in eligible) / len(eligible) if eligible else None
    out["real_tier_ordering_n"] = len(eligible)
    out["real_tier_ordering_two_hits_share"] = (sum(bool(m["ordering"]) for m in eligible_two_hits) / len(eligible_two_hits)
                                                if eligible_two_hits else None)
    out["real_tier_ordering_two_hits_n"] = len(eligible_two_hits)
    return out


def generation_failure_report(rows: list[dict[str, Any]], method: str, expected: int) -> list[dict[str, Any]]:
    """Apply the failure limit to units eligible for each phase; missing phase rows fail."""
    import gen60
    if method == "M7" or expected <= 0:
        return []
    phases = {"M3": ("selection", "fresh"), "M5": ("round1", "round2")}.get(method, ("primary",))
    by_record = {int(row["record_index"]): row for row in rows if "record_index" in row}
    failures = []
    for phase in phases:
        if method == "M3" and phase == "fresh":
            eligible = [row for row in by_record.values()
                        if isinstance(row.get("selection"), dict)
                        and row["selection"].get("parse_status") == "ok"]
            denominator = len(eligible)
            entries = [row.get("fresh") for row in eligible]
            phase_started = any(isinstance(entry, dict) and "parse_status" in entry for entry in entries)
        elif method == "M5" and phase == "round2":
            eligible = [row for row in by_record.values()
                        if isinstance(row.get("round1"), dict)
                        and row["round1"].get("parse_status") == "ok"]
            denominator = len(eligible)
            entries = [row.get("round2") for row in eligible]
            phase_started = any(isinstance(entry, dict) and "parse_status" in entry for entry in entries)
        else:
            denominator = expected
            entries = [row.get(phase) if method in ("M3", "M5") else row
                       for row in by_record.values()]
            phase_started = any(isinstance(entry, dict) and "parse_status" in entry for entry in entries)
        # A phase with no result records at all has not run yet.
        if denominator <= 0 or not phase_started:
            continue
        successes = sum(isinstance(entry, dict) and entry.get("parse_status") == "ok" for entry in entries)
        if gen60.generation_failure_exceeds_ten_percent(successes, denominator):
            failures.append({"method": method, "phase": phase, "successes": successes,
                             "expected": denominator, "eligible": denominator,
                             "failure_rate": (denominator-successes)/denominator,
                             "reason": "generation parse/refusal failure exceeded 10%"})
    return failures


def truncation_report(rows: list[dict[str, Any]], method: str, root: Path | None = None,
                      split: str | None = None) -> dict[str, int]:
    if root is None or split is None:
        return {}
    phases = _phases_for_method(method)
    result: dict[str, int] = {}
    for phase in phases:
        raw_results = _raw_results_by_custom_id(root, split, method, phase)
        count = sum(isinstance(raw.get("result"), dict)
                    and isinstance(raw["result"].get("message"), dict)
                    and raw["result"]["message"].get("stop_reason") == "max_tokens"
                    for raw in raw_results.values())
        if count:
            result[phase] = count
    return result


def _percent(value: float | None) -> str:
    return "unscored" if value is None else f"{100*value:.1f}%"


def prediction_table(metrics: dict[str, dict[str, Any]], m5_round1: dict[str, Any] | None,
                     split: str, best_dev: str | None = None,
                     paired_bootstrap: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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
    m4n = dev.get("M4", {}).get("by_type", {}).get("neuron", {}).get("marked_comparable_n")
    h_diff = m4m-m4w if m4m is not None and m4w is not None else None
    h_ok = None if h_diff is None else h_diff <= 0.0
    m4_pos = get(dev, "M4", "neuron", "positive_only_verified_rate")
    m1_pos = get(dev, "M1", "neuron", "positive_only_verified_rate")
    m4_full = get(dev, "M4", "neuron", "verified_rate")
    m1_full = get(dev, "M1", "neuron", "verified_rate")
    add("60h", "M4 marked-token pass rate minus whole-text pass rate is at most 0 pp", h_ok,
        f"marked={_percent(m4m)}; whole-text={_percent(m4w)}; difference={_percent(h_diff)}; positive difference is a scoring bug; same marked-comparable denominator n={m4n}; M4−M1 real pass-rate differences: positive-only={_percent(m4_pos-m1_pos if m4_pos is not None and m1_pos is not None else None)}, full-rule={_percent(m4_full-m1_full if m4_full is not None and m1_full is not None else None)}")

    m6r = get(dev, "M6", "neuron", "verified_rate")
    c6 = dev.get("M6", {}).get("cost", {}).get("usd_per_unit")
    c1 = dev.get("M1", {}).get("cost", {}).get("usd_per_unit")
    i_ok = None if m6r is None or m1r is None or c6 is None or c1 is None or c1 <= 0 else abs(m6r-m1r) <= 0.03 and c6 >= 2*c1
    add("60i", "M6 differs from M1 by at most 3 pp in real verified rate and costs at least twice as much", i_ok,
        f"M6={_percent(m6r)}; M1={_percent(m1r)}; cost/unit M6={c6}; M1={c1}")

    if split != "test":
        add("60j", "No tested method beats both M1 and M0 test gaps by at least 5 pp", None,
            "unscored until the registered test phase")
        rows[-1]["status"] = "unscored"
    else:
        if not best_dev:
            j_ok = None
            evidence = "no dev candidate advanced; no test candidate was selected"
        else:
            comparisons = ("M0",) if best_dev == "M1" else ("M0", "M1")
            leads: dict[str, dict[str, float | None]] = {}
            missing = False
            improved = True
            for metric, field in (("positive_only", "positive_only_verified_rate"), ("full_rule", "verified_rate")):
                cr = get(test, best_dev, "neuron", field)
                cc = get(test, best_dev, "control", field)
                if cr is None or cc is None:
                    missing = True
                    continue
                candidate_gap = cr-cc
                leads[metric] = {}
                for comparator in comparisons:
                    br = get(test, comparator, "neuron", field)
                    bc = get(test, comparator, "control", field)
                    ci = (((paired_bootstrap or {}).get("comparisons", {}).get(best_dev, {}).get(comparator, {})
                           .get(f"gap_{metric}") or {}).get("ci95"))
                    if br is None or bc is None or not ci:
                        missing = True
                        leads[metric][comparator] = None
                        improved = False
                    else:
                        point_lead = candidate_gap-(br-bc)
                        leads[metric][comparator] = {"point": point_lead, "ci95": list(ci)}
                        if point_lead < 0.05:
                            improved = False
            j_ok = None if missing else not improved
            evidence = f"tested candidate={best_dev}; only this single dev-selected candidate is tested; point leads over {','.join(comparisons)} for positive-only/full-rule={json.dumps(leads, sort_keys=True)}; advancement requires a point lead of >=5 pp for both (the registered rule); the 95% interval is reported beside it, not required; recommendation={'candidate' if improved and not missing else 'M0'}"
        add("60j", "No tested method beats both M1 and M0 test gaps by at least 5 pp", j_ok, evidence)
        if not best_dev:
            rows[-1]["status"] = "held"
        elif j_ok is not False:
            rows[-1]["status"] = "unscored (one candidate tested)"
    return rows


def choose_dev_leader(metrics: dict[str, dict[str, Any]],
                      bootstrap: dict[str, Any] | None = None) -> dict[str, Any]:
    base = metrics.get("M0", {}).get("by_type", {})
    base_gaps = {}
    for name, field in (("positive_only", "positive_only_verified_rate"), ("full_rule", "verified_rate")):
        real = base.get("neuron", {}).get(field)
        control = base.get("control", {}).get(field)
        base_gaps[name] = real-control if real is not None and control is not None else None
    candidates = []
    for method, value in metrics.items():
        if method == "M0" or value.get("n_units", 0) <= 0:
            continue
        if value.get("n_parseable", 0) / value["n_units"] < 0.90:
            continue
        by = value.get("by_type", {})
        gaps = {}
        improvements = {}
        lower_bounds = {}
        for name, field in (("positive_only", "positive_only_verified_rate"), ("full_rule", "verified_rate")):
            real = by.get("neuron", {}).get(field)
            control = by.get("control", {}).get(field)
            gap = real-control if real is not None and control is not None else None
            gaps[name] = gap
            improvements[name] = gap-base_gaps[name] if gap is not None and base_gaps[name] is not None else None
            interval = (((bootstrap or {}).get("comparisons", {}).get(method, {}).get("M0", {})
                         .get(f"gap_{name}") or {}).get("ci95"))
            lower_bounds[name] = interval[0] if interval else None
        if all(improvements[name] is not None and improvements[name] >= 0.05
               for name in ("positive_only", "full_rule")):
            candidates.append((min(improvements.values()), method, gaps, improvements, lower_bounds))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    if not candidates:
        return {"leader": "M0", "best_variant": None, "base_gaps": base_gaps,
                "best_gaps": None, "improvements": None, "lower_bounds": None,
                "stop_rule": "keep_M0_unless_positive_only_and_full_rule_each_lead_by_5pp",
                "test_candidate": None}
    _, method, gaps, improvements, lower_bounds = candidates[0]
    return {"leader": method, "best_variant": method, "base_gaps": base_gaps,
            "best_gaps": gaps, "improvements": improvements, "lower_bounds": lower_bounds,
            "stop_rule": "continue_to_test_when_both_positive_only_and_full_rule_point_leads_are_at_least_5pp",
            "test_candidate": method}


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Prediction | Status | Evidence |", "|---|---|---|"]
    for row in rows:
        evidence = str(row["evidence"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['id']} — {row['prediction']} | {row['status']} | {evidence} |")
    return "\n".join(lines)


def analysis_input_fingerprint(root: Path) -> str:
    """Hash the inputs and implementation that determine the dev selection marker."""
    import gen60
    labeled: list[tuple[str, Path]] = []
    for name in ("gen60.py", "score60.py", "analyze60.py", "DEFINITIONS-60.md"):
        labeled.append((f"part60/{name}", HERE / name))
    labeled.append(("results/part56-pilot/pilot.jsonl", gen60.PILOT))
    for base_name in ("out/dev", "acts/dev"):
        base = root / base_name
        if base.is_dir():
            for path in sorted(base.iterdir()):
                if path.is_file() and (path.suffix in (".jsonl", ".npz") or path.name.endswith(".state.json")):
                    labeled.append((f"results/part60/{base_name}/{path.name}", path))
    for relative in ("thresholds.json", "reproduction/dev.json"):
        path = root / relative
        if path.is_file():
            labeled.append((f"results/part60/{relative}", path))
    digest = hashlib.sha256()
    for label, path in sorted(labeled, key=lambda item: item[0]):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


BOOTSTRAP_SEED = 6003
BOOTSTRAP_RESAMPLES = 2000


def paired_bootstrap_differences(metrics: dict[str, dict[str, Any]],
                                seed: int = BOOTSTRAP_SEED,
                                resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Paired CIs for method-minus-baseline type rates and real-control gaps."""
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {"seed": seed, "resamples": resamples, "comparisons": {}}
    measures = {
        "real_positive_only": ("neuron", "positive_only_verified"),
        "control_positive_only": ("control", "positive_only_verified"),
        "gap_positive_only": ("gap", "positive_only_verified"),
        "real_full_rule": ("neuron", "verified"),
        "control_full_rule": ("control", "verified"),
        "gap_full_rule": ("gap", "verified"),
    }
    for method, method_value in metrics.items():
        result["comparisons"][method] = {}
        method_rows = {row["unit_key"]: row for row in method_value.get("unit_rows", [])}
        for comparator in ("M0", "M1"):
            if method == comparator or comparator not in metrics:
                continue
            baseline_rows = {row["unit_key"]: row for row in metrics[comparator].get("unit_rows", [])}
            common = set(method_rows) & set(baseline_rows)
            by_type_keys = {
                typ: sorted(key for key in common if method_rows[key].get("type") == typ
                            and baseline_rows[key].get("type") == typ)
                for typ in ("neuron", "control")
            }
            if not all(by_type_keys.values()):
                result["comparisons"][method][comparator] = {name: None for name in measures}
                continue
            indices = {typ: rng.integers(0, len(keys), size=(resamples, len(keys)))
                       for typ, keys in by_type_keys.items()}
            arrays: dict[str, dict[str, np.ndarray]] = {}
            for name, (typ, field) in measures.items():
                if typ == "gap":
                    groups = ("neuron", "control")
                else:
                    groups = (typ,)
                for key_method, key_rows in ((method, method_rows), (comparator, baseline_rows)):
                    if key_method not in arrays:
                        arrays[key_method] = {}
                    if typ == "gap":
                        arrays[key_method][f"gap_{field}"] = {
                            group: np.asarray([bool(key_rows[k].get(field, False)) for k in by_type_keys[group]], dtype=np.float64)
                            for group in groups
                        }
                    else:
                        arrays[key_method][name] = np.asarray(
                            [bool(key_rows[k].get(field, False)) for k in by_type_keys[typ]], dtype=np.float64)
            comparison: dict[str, Any] = {}
            for name, (typ, field) in measures.items():
                if typ == "gap":
                    m_real = arrays[method][f"gap_{field}"]["neuron"]
                    m_control = arrays[method][f"gap_{field}"]["control"]
                    b_real = arrays[comparator][f"gap_{field}"]["neuron"]
                    b_control = arrays[comparator][f"gap_{field}"]["control"]
                    m_draw = m_real[indices["neuron"]].mean(axis=1) - m_control[indices["control"]].mean(axis=1)
                    b_draw = b_real[indices["neuron"]].mean(axis=1) - b_control[indices["control"]].mean(axis=1)
                    point = float(m_real.mean() - m_control.mean() - b_real.mean() + b_control.mean())
                else:
                    m_values = arrays[method][name]
                    b_values = arrays[comparator][name]
                    draws = indices[typ]
                    m_draw = m_values[draws].mean(axis=1)
                    b_draw = b_values[draws].mean(axis=1)
                    point = float(m_values.mean() - b_values.mean())
                difference = m_draw - b_draw
                comparison[name] = {"estimate": point,
                                    "ci95": [float(x) for x in np.percentile(difference, [2.5, 97.5])],
                                    "n_real": len(by_type_keys["neuron"]),
                                    "n_control": len(by_type_keys["control"])}
            result["comparisons"][method][comparator] = comparison
    return result


def unjudgeable_units(root: Path, units: list[dict[str, Any]],
                      thresholds: dict[str, float]) -> dict[str, Any]:
    threshold_file = root / "thresholds.json"
    saved = read_json(threshold_file) if threshold_file.is_file() else {}
    ceilings = saved.get("observed_ceilings", {})
    by_type: dict[str, int] = {}
    known = 0
    unknown = 0
    keys = []
    for unit in units:
        key = str(unit["unit_key"])
        threshold = thresholds.get(key)
        ceiling = ceilings.get(key)
        if threshold is None or ceiling is None:
            unknown += 1
            continue
        known += 1
        if float(threshold) >= float(ceiling):
            by_type[unit["type"]] = by_type.get(unit["type"], 0) + 1
            keys.append(key)
    return {"count": len(keys), "known_ceiling_units": known,
            "unknown_ceiling_units": unknown, "by_type": by_type,
            "unit_keys": keys,
            "definition": "empirically unjudgeable (threshold at or above the pool maximum)"}


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
        if method == "M0" and value is not None:
            archive_path = score60.archive_m0_path(root, split)
            if archive_path.is_file():
                archive = aggregate_method(root, split, method, units, thresholds,
                                           activation_path=archive_path)
                if archive is not None:
                    value["archive_crosscheck"] = {
                        "n_parseable": archive["n_parseable"],
                        "by_type": {typ: {"n": archive["by_type"][typ]["n"],
                                          "verified_rate": archive["by_type"][typ]["verified_rate"]}
                                    for typ in ("neuron", "control")},
                    }
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
    bootstrap = paired_bootstrap_differences(prediction_metrics)
    leader = choose_dev_leader(prediction_metrics, bootstrap) if split == "dev" else None
    predictions = prediction_table(prediction_metrics, prediction_m5_round1, split,
                                   leader.get("test_candidate") if leader else None,
                                   paired_bootstrap=bootstrap)
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
            counts = truncation_report(read_jsonl(path), method, root, split)
            if counts:
                truncations[method] = counts
    return {"split": split, "metrics": metrics, "m5_round1": m5_round1, "dev_selection": leader,
            "predictions": predictions, "skipped_methods": skipped,
            "dropped_generation": dropped_generation,
            "max_tokens_truncations": truncations,
            "unjudgeable": unjudgeable_units(root, units, thresholds),
            "paired_bootstrap": bootstrap,
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
        analyzed = set(dev.get("metrics", {}))
        analyzed.update(str(item.get("method")) for item in dev.get("skipped_methods", []))
        expected = {f"M{i}" for i in range(8)}
        selection_status = "ready" if expected.issubset(analyzed) else "incomplete"
        write_json(root / "test-selection.json", {
            "status": selection_status, "created_at": datetime.now(timezone.utc).isoformat(),
            "candidate": candidate, "methods": methods, "dev_selection": selection,
            "missing_dev_methods": sorted(expected - analyzed),
            "analysis_input_sha256": analysis_input_fingerprint(root),
            "rule": "generated by analyze60.py --split dev; advancement uses at least five percentage points on both positive-only and full-rule gap leads over M0; 95% paired-bootstrap intervals are reported and are not a gate",
        })
    lines = ["# Part 60 analysis", "", "Metrics use the frozen definitions in `part60/DEFINITIONS-60.md`.", ""]
    for analysis in analyses:
        split = analysis["split"]
        lines.extend([f"## {split}", "", "| Method | Parsed | Failed | Threshold missing | Real positive-only | Real full-rule | Control positive-only | Control full-rule | Real Hstrong | Control Hstrong | Full-rule gap | Real Hweak | Real Hnone | Control Hnone | 60d all eligible | 60d >=2 strong hits | Real AUC | Calls | Max-token truncations | Cost / unit |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for method, result in analysis["metrics"].items():
            real, control = result["by_type"]["neuron"], result["by_type"]["control"]
            pct = lambda value: "—" if value is None else f"{value:.3f}"
            cost = result["cost"].get("usd_per_unit")
            cost_text = "—" if cost is None else f"${cost:.5f}"
            gap = None if real["verified_rate"] is None or control["verified_rate"] is None else real["verified_rate"] - control["verified_rate"]
            truncations = sum(analysis.get("max_tokens_truncations", {}).get(method, {}).values())
            mean = lambda value, n: f"{pct(value)} (n={n})"
            real_n = int(real.get("n_parseable", result["n_parseable"]))
            control_n = int(control.get("n_parseable", result["n_parseable"]))
            lines.append(f"| {method} | {result['n_parseable']}/{result['n_units']} | {result['n_failed']} | {result['n_threshold_missing']} | {pct(real['positive_only_verified_rate'])} | {pct(real['verified_rate'])} | {pct(control['positive_only_verified_rate'])} | {pct(control['verified_rate'])} | {mean(real['mean_h_strong'], real.get('n_mean_h_strong', real_n))} | {mean(control['mean_h_strong'], control.get('n_mean_h_strong', control_n))} | {pct(gap)} | {mean(real['mean_h_weak'], real.get('n_mean_h_weak', real_n))} | {mean(real['mean_h_none'], real.get('n_mean_h_none', real_n))} | {mean(control['mean_h_none'], control.get('n_mean_h_none', control_n))} | {pct(result['real_tier_ordering_share'])} (n={result['real_tier_ordering_n']}) | {pct(result.get('real_tier_ordering_two_hits_share'))} (n={result.get('real_tier_ordering_two_hits_n', 0)}) | {mean(real['mean_auc'], real.get('n_mean_auc', real_n))} | {result['cost'].get('calls', 0)} | {truncations} | {cost_text} |")
            if method == "M0" and result.get("archive_crosscheck"):
                archive = result["archive_crosscheck"]["by_type"]
                lines.append(f"| M0 archive cross-check | {result['archive_crosscheck']['n_parseable']}/{result['n_units']} | — | — | — | {pct(archive['neuron']['verified_rate'])} | — | {pct(archive['control']['verified_rate'])} | — | — | — | — | — | — | — | — | — | 0 | 0 | $0.00000 |")
        unjudgeable = analysis.get("unjudgeable", {})
        lines.extend(["", f"Empirically unjudgeable (threshold at or above the pool maximum): {unjudgeable.get('count', 0)}; known ceilings {unjudgeable.get('known_ceiling_units', 0)}, unknown ceilings {unjudgeable.get('unknown_ceiling_units', 0)}; by type {json.dumps(unjudgeable.get('by_type', {}), sort_keys=True)}."])
        lines.extend(["", "### Paired bootstrap differences", "", f"2,000 paired resamples; seed {analysis.get('paired_bootstrap', {}).get('seed')}.", "", "| Method | Comparator | Measure | Difference | 95% CI |", "|---|---|---|---:|---:|"])
        measures = ("real_positive_only", "control_positive_only", "gap_positive_only",
                    "real_full_rule", "control_full_rule", "gap_full_rule")
        comparisons = analysis.get("paired_bootstrap", {}).get("comparisons", {})
        for method, comparators in comparisons.items():
            for comparator, values in comparators.items():
                for measure in measures:
                    value = values.get(measure)
                    if not value:
                        continue
                    interval = value["ci95"]
                    lines.append(f"| {method} | {comparator} | {measure} | {value['estimate']:.3f} | [{interval[0]:.3f}, {interval[1]:.3f}] |")
        lines.extend(["", "Interpretation limits: control hits are not a false-discovery rate for labels because Part 58 notes that some random directions carry real features; only 247 real test units with at least 20 top, 10 random, and 5 weak windows are covered, so no atlas-wide claim follows; the test split is prospective for new outputs but not an untouched holdout of earlier decisions; M3 selects a hypothesis-plus-probe distribution and uses about twice M1's generations."])
        if analysis.get("m5_round1"):
            r1 = analysis["m5_round1"]["by_type"]
            lines.extend(["", f"M5 round 1 positive-only/full-rule: real {pct(r1['neuron']['positive_only_verified_rate'])}/{pct(r1['neuron']['verified_rate'])}, control {pct(r1['control']['positive_only_verified_rate'])}/{pct(r1['control']['verified_rate'])}; M5 main metrics above use round 2 only."])
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
            analyses[1]["predictions"] = prediction_table(test_metrics, test_m5_round1, "test",
                                                           leader.get("test_candidate"),
                                                           paired_bootstrap=analyses[1].get("paired_bootstrap"))
        md_path, json_path = write_reports(args.root / "analysis", analyses)
        print(f"wrote {md_path} and {json_path}")
        return 0
    except (AnalysisError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
