#!/usr/bin/env python3
"""Score registered Part 198 JSONL records with the frozen design's gate and forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import run198 as R

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE / "out"


def registered_rows(path: Path, kind: str, key: str) -> list[dict[str, Any]]:
    """Keep the first registered terminal row; conflicting repeats make the stream unscorable."""
    first = {}
    for row in R.read_jsonl(path):
        if row.get("run_mode", "registered") != "registered":
            continue
        if row.get("record_type") not in {kind, "failure"}:
            continue
        value = row.get(key)
        if value is not None:
            identity = str(value)
            if identity not in first:
                first[identity] = row
                continue
            if _terminal_payload(first[identity]) != _terminal_payload(row):
                raise ValueError(f"conflicting terminal rows for {key}={identity!r} in {path}: "
                                 f"{first[identity].get('record_type')} then {row.get('record_type')}")
    return list(first.values())


def _terminal_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"run_id", "timestamp"}}


def all_registered(path: Path) -> list[dict[str, Any]]:
    return [r for r in R.read_jsonl(path) if r.get("run_mode", "registered") == "registered"]


def flag(row: dict[str, Any], *keys: str) -> bool:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return False
        value = value.get(key)
    return value is True


def count_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {"attempted": len(rows), "took": sum(bool(r.get("took")) for r in rows),
            "already_satisfied": sum(bool(r.get("already_satisfied")) for r in rows),
            "name_eligible": sum(bool(r.get("name_eligible")) for r in rows)}


def metric_result(label: str, rows: list[dict[str, Any]], success: Callable[[dict[str, Any]], bool],
                  denom_rule: str, threshold: int | None, gate_ok: bool,
                  qualifier: str | None = None) -> dict[str, Any]:
    summary = count_summary(rows)
    if denom_rule == "took":
        selected = [r for r in rows if r.get("took") is True]
    elif denom_rule == "took_and_name_eligible":
        selected = [r for r in rows if r.get("took") is True and r.get("name_eligible") is True]
    elif denom_rule == "name_eligible":
        selected = [r for r in rows if r.get("name_eligible") is True]
    else:
        selected = rows
    successes = sum(bool(success(r)) for r in selected)
    n = len(selected)
    ci = R.cp_interval(successes, n)
    if not gate_ok or n == 0:
        status = "NOT EVALUABLE"
    elif threshold is None:
        status = "DESCRIPTIVE"
    elif threshold >= 0:
        status = "PASS" if successes <= threshold else "FAIL"
    else:
        status = "PASS" if successes >= -threshold else "FAIL"
    out = {"forecast": label, **summary, "denominator_rule": denom_rule,
           "rate_denominator": n, "successes": successes,
           "clopper_pearson_95": list(ci) if ci else None, "status": status}
    if qualifier:
        out["qualification"] = qualifier
    return out


def gate_check(out_dir: Path, files: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    gate_path = out_dir / "gate.json"
    gate_file = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    prerequisites = {"fixtures": gate_file.get("fixtures_pass") is True,
                     "synthetic_tests": gate_file.get("synthetic_tests_pass") is True,
                     "stats_crosscheck": gate_file.get("crosscheck_pass") is True}
    reasons = [f"pre-data {k} did not pass" for k, passed in prerequisites.items() if not passed]
    take_requirements = {}
    for method in ("ROME", "MEMIT", "AlphaEdit"):
        rows = files.get(f"armA-{method}", [])
        took = sum(r.get("took") is True for r in rows)
        ok = len(rows) == 24 and took >= 21
        take_requirements[f"armA_{method}"] = {"attempted": len(rows), "took": took,
                                                "required": 21, "passed": ok}
        if not ok:
            reasons.append(f"Arm A {method} took gate not met ({took}/24)")
    b_rows = files.get("armB", [])
    b_took = sum(r.get("took") is True for r in b_rows)
    b_ok = len(b_rows) == 24 and b_took >= 21
    take_requirements["armB_wsp"] = {"attempted": len(b_rows), "took": b_took,
                                     "required": 21, "passed": b_ok}
    if not b_ok:
        reasons.append(f"Arm B WSP took gate not met ({b_took}/24)")
    restoration_failures = []
    for method in ("ROME", "MEMIT", "AlphaEdit"):
        rows = files.get(f"all-armA-{method}", [])
        cases = [r for r in rows if r.get("record_type") == "case"]
        verifies = [r for r in rows if r.get("record_type") == "state_verification"]
        failures = [r for r in rows if r.get("record_type") == "failure" and r.get("case_id") is not None]
        if any(r.get("restored_bit_exact") is not True for r in cases):
            restoration_failures.append(f"Arm A {method} case restoration missing/false")
        if not verifies or any(not flag(r, "restored_bit_exact") or not flag(r, "update_reproduced")
                               for r in verifies):
            restoration_failures.append(f"Arm A {method} first-update state check missing/false")
        if failures:
            restoration_failures.append(f"Arm A {method} contains failure records")
    for method in ("MEMIT", "AlphaEdit"):
        rows = files.get(f"all-armA-mass-{method}", [])
        batches = [r for r in rows if r.get("record_type") == "batch"]
        if any(r.get("restored_bit_exact") is not True for r in batches):
            restoration_failures.append(f"Arm A mass {method} restoration missing/false")
        if any(r.get("record_type") == "failure" and r.get("batch_id") is not None for r in rows):
            restoration_failures.append(f"Arm A mass {method} contains failure records")
    b_all = files.get("all-armB", [])
    b_cases = [r for r in b_all if r.get("record_type") == "case"]
    if len(b_cases) != 24 or any(r.get("undo_zero_exact") is not True for r in b_cases):
        restoration_failures.append("Arm B WSP undo zero check missing/false")
    if any(r.get("record_type") == "failure" and r.get("case_id") is not None for r in b_all):
        restoration_failures.append("Arm B WSP contains failure records")
    reasons.extend(restoration_failures)
    return {"passed": not reasons, "prerequisites": prerequisites,
            "take_requirements": take_requirements, "restoration_failures": restoration_failures,
            "reasons": reasons}


def mass_forecast(label: str, rows: list[dict[str, Any]], stat: str, gate_ok: bool) -> dict[str, Any]:
    batches = [r for r in rows if r.get("record_type") == "batch"]
    countable = [r for r in batches if r.get("take_gate_passed") is True]
    successes = sum(flag(r, "localization", stat, "localized") for r in countable)
    facts = [f for r in batches for f in r.get("facts", [])]
    eligible = sum(bool(f.get("name_eligible")) for f in facts)
    took_facts = sum(bool(f.get("took")) for f in facts)
    already = sum(bool(f.get("already_satisfied")) for f in facts)
    n = len(countable)
    status = "NOT EVALUABLE" if not gate_ok or not R.mass_gate_status(n) else (
        "PASS" if successes == 0 else "FAIL")
    return {"forecast": label, "attempted": len(rows), "took": took_facts,
            "already_satisfied": already, "name_eligible": eligible, "counted_batches": n,
            "successes": successes, "dropped_batches": len(batches) - len(countable),
            "denominator_rule": "batches with at least 8 of 10 facts taking",
            "rate_denominator": n, "clopper_pearson_95": list(R.cp_interval(successes, n)) if n else None,
            "status": status}


def final_sequence_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the sequence state with the most completed writes, independent of file order."""
    return max(rows, key=lambda r: int(r.get("n_written_so_far", 0))) if rows else None


def score_from_dir(out_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    allfiles: dict[str, Any] = {}
    for method in ("ROME", "MEMIT", "AlphaEdit"):
        path = out_dir / f"armA-{method.lower()}.jsonl"
        allfiles[f"all-armA-{method}"] = all_registered(path)
        files[f"armA-{method}"] = registered_rows(path, "case", "case_id")
    for method in ("MEMIT", "AlphaEdit"):
        path = out_dir / f"armA-mass-{method.lower()}.jsonl"
        allfiles[f"all-armA-mass-{method}"] = all_registered(path)
        files[f"armA-mass-{method}"] = registered_rows(path, "batch", "batch_id")
    bpath = out_dir / "armB-wsp.jsonl"
    allfiles["all-armB"] = all_registered(bpath)
    files["armB"] = registered_rows(bpath, "case", "case_id")
    seqpath = out_dir / "armB-seq-wsp.jsonl"
    files["armB-seq"] = registered_rows(seqpath, "case", "case_id")
    gate = gate_check(out_dir, {**files, **allfiles})
    tstar_path = out_dir / "tstar.json"
    out: dict[str, Any] = {"created_utc": R.utc_now(), "gate": gate,
        "tstar": json.loads(tstar_path.read_text(encoding="utf-8")) if tstar_path.is_file() else None,
        "forecasts": [], "sensitivity_checks": {}, "descriptive": {}}
    gate_ok = bool(gate["passed"])
    for method in ("ROME", "MEMIT", "AlphaEdit"):
        rows = files[f"armA-{method}"]
        for stat, label in (("f1", "198.1"), ("pcs", "198.3")):
            out["forecasts"].append(metric_result(f"{label} {method} single", rows,
                lambda r, s=stat: flag(r, "localization", s, "localized"), "took", 1, gate_ok))
        named_rows = [r for r in rows if r.get("took") is True and r.get("name_eligible") is True]
        oracle = metric_result(f"198.4 {method} oracle naming", rows,
            lambda r: flag(r, "oracle_reader", "named_strict"), "took_and_name_eligible", 2,
            gate_ok, "S1 must pass for this forecast to be informative")
        update = metric_result(f"198.4 {method} update-top naming", rows,
            lambda r: any(flag(v, "update_top_names_strict")
                          for v in r.get("edited_layer_stats", {}).values()),
            "took_and_name_eligible", 2, gate_ok)
        oracle["loose_successes"] = sum(flag(r, "oracle_reader", "named_loose") for r in named_rows)
        update["loose_successes"] = sum(any(flag(v, "update_top_names_loose")
            for v in r.get("edited_layer_stats", {}).values()) for r in named_rows)
        out["forecasts"].extend([oracle, update])
    # S1 is conditional on a took, name-eligible ROME record; missing controls remain failures.
    rome = files["armA-ROME"]
    s1_rows = [r for r in rome if r.get("took") is True and r.get("name_eligible") is True]
    s1_success = sum(flag(r, "aimed_positive_control", "oracle_named_strict") for r in s1_rows)
    s1_missing_controls = sum(not isinstance(r.get("aimed_positive_control"), dict) for r in s1_rows)
    s1_status = ("NOT EVALUABLE" if not gate_ok else
                 ("PASS" if s1_success >= 20 else ("FAIL" if s1_rows else "NOT EVALUABLE")))
    out["sensitivity_checks"]["S1"] = {"attempted": len(rome),
        "took": sum(bool(r.get("took")) for r in rome),
        "already_satisfied": sum(bool(r.get("already_satisfied")) for r in rome),
        "name_eligible": len(s1_rows), "missing_controls": s1_missing_controls, "successes": s1_success,
        "clopper_pearson_95": list(R.cp_interval(s1_success, len(s1_rows))) if s1_rows else None,
        "threshold": 20, "status": s1_status}
    # S1 failure makes the 198.4 oracle result uninformative, so suppress its verdict.
    if s1_status != "PASS":
        for item in out["forecasts"]:
            if item["forecast"].startswith("198.4 ") and item["forecast"].endswith("oracle naming"):
                item["status"] = "NOT EVALUABLE"
                item["qualification"] = "S1 failed or is unevaluable; not informative about direction"
    for method in ("MEMIT", "AlphaEdit"):
        rows = files[f"armA-mass-{method}"]
        out["forecasts"].append(mass_forecast(f"198.1 {method} mass", rows, "f1", gate_ok))
        out["forecasts"].append(mass_forecast(f"198.3 {method} mass", rows, "pcs", gate_ok))
    b = files["armB"]
    random_rows = [r for r in b if isinstance(r.get("random_control"), dict)]
    random_local = [flag(r, "random_control", "localization", "t_word", "localized") for r in random_rows]
    random_success = sum(random_local)
    s2_status = ("NOT EVALUABLE" if not gate_ok or len(random_rows) != 24 else
                 ("PASS" if random_success <= 2 else "FAIL"))
    out["sensitivity_checks"]["S2"] = {"attempted": len(b),
        "took": sum(bool(r.get("took")) for r in b),
        "already_satisfied": sum(bool(r.get("already_satisfied")) for r in b),
        "name_eligible": sum(bool(r.get("name_eligible")) for r in b),
        "successes": random_success, "denominator": len(random_rows),
        "clopper_pearson_95": list(R.cp_interval(random_success, len(random_rows))) if random_rows else None,
        "threshold": 2, "status": s2_status}
    s2_note = None if s2_status == "PASS" else "S2 failed or is unevaluable: not specific to aimed writes"
    out["forecasts"].append(metric_result("198.6 WSP T_word localization", b,
        lambda r: flag(r, "localization", "t_word", "localized"), "took", -20, gate_ok, s2_note))
    out["forecasts"].append(metric_result("198.7 WSP blind joint success", b,
        lambda r: flag(r, "blind_reader", "localized") and flag(r, "blind_reader", "named_strict"),
        "took_and_name_eligible", -18, gate_ok, s2_note))
    oracle_b = metric_result("WSP oracle naming diagnostic", b,
        lambda r: flag(r, "oracle_reader", "named_strict"), "took_and_name_eligible", None, gate_ok)
    oracle_b["loose_successes"] = sum(flag(r, "oracle_reader", "named_loose") for r in b
                                       if r.get("took") and r.get("name_eligible"))
    out["descriptive"]["wsp_oracle_naming"] = oracle_b
    real_yes = control_yes = both = real_only = control_only = neither = 0
    for row in b:
        real = flag(row, "localization", "t_word", "localized")
        control = flag(row, "random_control", "localization", "t_word", "localized")
        real_yes += int(real)
        control_yes += int(control)
        both += int(real and control)
        real_only += int(real and not control)
        control_only += int(control and not real)
        neither += int(not real and not control)
    out["sensitivity_checks"]["paired_random_vs_real"] = {
        "attempted": len(b), "took": sum(bool(r.get("took")) for r in b),
        "already_satisfied": sum(bool(r.get("already_satisfied")) for r in b),
        "name_eligible": sum(bool(r.get("name_eligible")) for r in b),
        "real_localized": real_yes, "random_localized": control_yes,
        "both": both, "real_only": real_only, "random_only": control_only, "neither": neither,
        "exact_mcnemar_two_sided_p": R.exact_mcnemar(real_only, control_only) if gate_ok and b else None}
    seq = files["armB-seq"]
    final_names = {}
    final_row = final_sequence_row(seq)
    if final_row:
        final_names = {str(k): bool(v.get("named_strict"))
                       for k, v in final_row.get("names_after_write", {}).items()}
    eligible_seq_ids = {str(r["case_id"]) for r in seq if r.get("name_eligible")}
    seq_success = sum(final_names.get(cid, False) for cid in eligible_seq_ids)
    seq_n = len(eligible_seq_ids)
    seq_ci = R.cp_interval(seq_success, seq_n)
    seq_summary = count_summary(seq)
    seq_status = "NOT EVALUABLE" if not gate_ok or len(seq) != 10 or seq_n == 0 else (
        "PASS" if seq_success >= 5 else "FAIL")
    out["forecasts"].append({"forecast": "198.8 WSP sequence final strict names", **seq_summary,
        "name_eligible": seq_n, "rate_denominator": seq_n, "successes": seq_success,
        "denominator_rule": "name-eligible objects among the 10 sequence writes",
        "clopper_pearson_95": list(seq_ci) if seq_ci else None, "status": seq_status})
    out["descriptive"]["198.2_gradient_tword_localization"] = {
        method: {"attempted": len(files[f"armA-{method}"]),
            "took": sum(bool(r.get("took")) for r in files[f"armA-{method}"]),
            "already_satisfied": sum(bool(r.get("already_satisfied")) for r in files[f"armA-{method}"]),
            "name_eligible": sum(bool(r.get("name_eligible")) for r in files[f"armA-{method}"]),
            "successes": sum(flag(r, "localization", "t_word", "localized")
                             for r in files[f"armA-{method}"]),
            "status": "DESCRIPTIVE"} for method in ("ROME", "MEMIT", "AlphaEdit")}
    out["descriptive"]["198.5_aimed_control_localization"] = {
        "attempted": len(s1_rows), "took": sum(bool(r.get("took")) for r in s1_rows),
        "already_satisfied": sum(bool(r.get("already_satisfied")) for r in s1_rows),
        "name_eligible": len(s1_rows),
        "successes": sum(flag(r, "aimed_positive_control", "localization", "localized") for r in s1_rows),
        "status": "DESCRIPTIVE"}
    return out


def print_result(result: dict[str, Any]) -> None:
    print("GATE", "PASS" if result["gate"]["passed"] else "FAIL")
    for reason in result["gate"].get("reasons", []):
        print("  ", reason)
    for item in result.get("forecasts", []):
        ci = item.get("clopper_pearson_95")
        ci_text = "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        print(f"{item['forecast']}: attempted={item.get('attempted', 0)} took={item.get('took', 0)} "
              f"already_satisfied={item.get('already_satisfied', 0)} name_eligible={item.get('name_eligible', 0)} "
              f"successes={item.get('successes', 0)} denominator={item.get('rate_denominator', 0)} "
              f"CP95={ci_text} {item['status']}")
        if item.get("qualification"):
            print("  ", item["qualification"])
    print("DESCRIPTIVE (no verdict)")
    for name, value in result.get("descriptive", {}).items():
        if isinstance(value, dict) and "attempted" in value:
            rows = [(name, value)]
        elif isinstance(value, dict):
            rows = [(f"{name} {key}", row) for key, row in value.items()
                    if isinstance(row, dict) and "attempted" in row]
        else:
            rows = []
        for label, row in rows:
            print(f"{label}: attempted={row.get('attempted', 0)} took={row.get('took', 0)} "
                  f"already_satisfied={row.get('already_satisfied', 0)} "
                  f"name_eligible={row.get('name_eligible', 0)} successes={row.get('successes', 0)}")
    for name, item in result.get("sensitivity_checks", {}).items():
        if not isinstance(item, dict):
            continue
        print(f"{name}: attempted={item.get('attempted', 0)} took={item.get('took', 0)} "
              f"already_satisfied={item.get('already_satisfied', 0)} name_eligible={item.get('name_eligible', 0)} "
              f"successes={item.get('successes', 0)} {item.get('status', '')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    out_dir = R.root_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = score_from_dir(out_dir)
    path = out_dir / "score.json"
    path.write_text(json.dumps(R.safe_json(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_result(result)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

