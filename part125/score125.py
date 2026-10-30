#!/usr/bin/env python3
"""Score Part 125 gates and registered predictions from saved raw files."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (HERE, ROOT / "part92", ROOT / "part92b", ROOT / "part119", ROOT / "part123"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import label119  # noqa: E402
import label123  # noqa: E402
import label125  # noqa: E402
import session125  # noqa: E402

PREDICTIONS = {
    "125.1": {"text": "At turn 5, the coherent-persona counts of arm 2 and arm 3 are within 5 units of each other.", "confidence": "medium", "author": "Claude"},
    "125.2": {"text": "At turn 5, arm 3 shows a coherent persona in at least 6 of 29 units, so text alone carries the persona across four turns.", "confidence": "medium", "author": "Claude"},
    "125.3": {"text": "At turn 2, delta_2 is above the mean of its three delta_r_2 in at least 20 of 29 units.", "confidence": "medium", "author": "Claude"},
    "125.4": {"text": "The count in 125.3 is at least 5 units lower at turn 5 than at turn 2.", "confidence": "low", "author": "Claude"},
    "125.5": {"text": "At turn 5, arm 1 has at least 6 more incoherent answers than arm 2. Pushing every turn breaks answers more than it holds a persona.", "confidence": "medium", "author": "Claude"},
    "125.6": {"text": "At turn 5, arm 4 shows a coherent target persona in at most 2 of 29 units.", "confidence": "medium", "author": "Claude"},
}


def read_json(path: str | Path, *, optional: bool = False):
    source = Path(path)
    if optional and not source.is_file():
        return None
    with source.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path, *, optional: bool = False) -> list[dict]:
    source = Path(path)
    if optional and not source.is_file():
        return []
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = []
    lines = source.read_bytes().splitlines(keepends=True)
    for line_no, raw in enumerate(lines, 1):
        try:
            line = raw.decode("utf-8")
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row is not an object")
                rows.append(row)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            torn = line_no == len(lines) and not raw.endswith((b"\n", b"\r"))
            if not torn:
                raise ValueError(f"invalid JSONL in {source}:{line_no}: {exc}") from exc
            print(f"skipping torn final JSONL line in {source}:{line_no}", file=sys.stderr, flush=True)
            break
    return rows


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, value: dict, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def _valid_label(row: dict | None) -> bool:
    return bool(row is not None and label123.valid_label(row))


def _known_result(reader: str, path: str | Path) -> dict:
    rows = read_jsonl(path, optional=True)
    result = label119.gate_result(label119.read_known(), rows)
    result["required_correct"] = 11
    result["passed"] = bool(result.get("correct", 0) >= 11)
    result["reader"] = reader
    return result


def _label_lookup(answer_rows: list[dict], keys: list[dict], labels: list[dict], *,
                  require_all: bool):
    key_map = {}
    for row in keys:
        key = (str(row.get("arm", "")), str(row.get("unit_id", "")), int(row.get("turn", 0)))
        key_map[key] = row
    latest = label123.latest_label_rows(labels)
    result = {}
    missing = []
    for answer in answer_rows:
        key = (str(answer["arm"]), str(answer["unit_id"]), int(answer["turn"]))
        question = label125._reader_question(answer)
        expected_hash = label123.answer_hash(question, str(answer.get("answer", "")))
        private = key_map.get(key)
        label = None
        reason = None
        if private is None:
            if require_all:
                reason = "private reader key missing"
            else:
                continue
        elif str(private.get("answer_hash", "")) != expected_hash:
            reason = "private key hash mismatch"
        else:
            item_id = str(private.get("item_id", ""))
            candidate = latest.get((item_id, int(answer["turn"]) - 1))
            if not _valid_label(candidate):
                reason = "label missing or invalid"
            elif str(candidate.get("answer_hash", "")) != expected_hash:
                reason = "label answer hash mismatch"
            else:
                label = candidate
        result[key] = label
        if label is None:
            missing.append({"arm": key[0], "unit_id": key[1], "turn": key[2], "reason": reason})
    return result, missing


def _cohen_kappa(ds_by_item: dict, luna_by_item: dict,
                 sample_keys: list[dict]) -> dict:
    pairs = []
    unresolved = []
    expected = set()
    for row in sample_keys:
        key = (str(row.get("item_id", "")), int(row.get("question_index", -1)))
        answer_hash = str(row.get("answer_hash", ""))
        if key in expected:
            unresolved.append({"item_id": key[0], "reason": "duplicate sample key"})
            continue
        expected.add(key)
        ds = ds_by_item.get(key)
        luna = luna_by_item.get(key)
        if (not _valid_label(ds) or not _valid_label(luna) or
                str(ds.get("answer_hash", "")) != answer_hash or
                str(luna.get("answer_hash", "")) != answer_hash):
            unresolved.append({"item_id": key[0], "reason": "missing, invalid, or stale sample label"})
            continue
        pairs.append((bool(ds["shows_persona"]), bool(luna["shows_persona"])))
    n = len(pairs)
    if not n:
        return {"n": 0, "expected_n": len(expected), "unresolved": unresolved,
                "agreement": None, "expected_agreement": None, "kappa": None,
                "single_class": False, "gate_method": "kappa", "gate_value": None,
                "threshold": 0.60, "agreement_threshold": 0.80,
                "gate_passed": False}
    observed = sum(a == b for a, b in pairs) / n
    p_ds = sum(a for a, _b in pairs) / n
    p_luna = sum(b for _a, b in pairs) / n
    expected_agreement = p_ds * p_luna + (1 - p_ds) * (1 - p_luna)
    one_class = len({a for a, _b in pairs}) == 1 or len({b for _a, b in pairs}) == 1
    kappa = (None if one_class or expected_agreement == 1 else
             (observed - expected_agreement) / (1 - expected_agreement))
    gate_method = "raw_agreement_single_class" if one_class else "cohen_kappa"
    gate_value = observed if one_class else kappa
    passed = (len(pairs) == len(sample_keys) and not unresolved and
              observed >= 0.80 and (kappa is None or kappa >= 0.60))
    return {"n": n, "expected_n": len(sample_keys), "unresolved": unresolved,
            "agreement": observed, "expected_agreement": expected_agreement, "kappa": kappa,
            "single_class": bool(one_class), "gate_method": gate_method,
            "gate_value": gate_value, "threshold": 0.60,
            "kappa_threshold": 0.60, "agreement_threshold": 0.80,
            "gate_passed": bool(passed)}


def _count_for_arm(labels_by_key: dict, units: list[str], arm: str, turn: int,
                   answers_by_key: dict | None = None) -> dict:
    known = []
    missing = []
    for unit in units:
        label = labels_by_key.get((arm, unit, turn))
        if _valid_label(label):
            answer = (answers_by_key or {}).get((arm, unit, turn), {})
            forced_broken = bool(answer.get("cap_hit")) or not str(answer.get("answer", "")).strip()
            known.append({**label, "coherent": bool(label["coherent"]) and not forced_broken})
        else:
            missing.append(unit)
    return {
        "coherent_persona": sum(bool(row["shows_persona"]) and bool(row["coherent"]) for row in known),
        "shows_persona": sum(bool(row["shows_persona"]) for row in known),
        "incoherent": sum(not bool(row["coherent"]) for row in known),
        "known_labels": len(known), "missing_labels": len(missing), "missing_units": missing,
    }


def _gate(number: int, name: str, passed: bool, evidence: dict) -> dict:
    return {"number": number, "name": name, "passed": bool(passed), **evidence}


def _prediction_thresholds(unit_count: int) -> dict[str, int]:
    scale = int(unit_count) / 29
    return {
        "125.2_minimum": math.ceil(6 * scale),
        "125.3_minimum": math.ceil(20 * scale),
        "125.5_minimum": math.ceil(6 * scale),
        "125.6_maximum": math.floor(2 * scale),
    }


def _write_report(out: Path, result: dict) -> str:
    lines = ["# Part 125 score", "", f"All gates passed: **{result['all_gates_passed']}**", "",
             "## Gates", "", "| # | Gate | Result | Evidence |", "|---:|---|---|---|"]
    for gate in result["gates"]:
        lines.append(f"| {gate['number']} | {gate['name']} | {'PASS' if gate['passed'] else 'FAIL'} | "
                     f"`{json.dumps({k: v for k, v in gate.items() if k not in ('number', 'name', 'passed')}, ensure_ascii=False)}` |")
    lines.extend(["", "## Turn 5 reader counts", "",
                  "Missing labels are excluded from counts and are never treated as false.", "",
                  "| Arm | Coherent persona | Shows persona | Incoherent | Known | Missing |",
                  "|---|---:|---:|---:|---:|---:|"])
    for arm in ("arm1", "arm2", "arm3", "arm4"):
        row = result["turn5_counts"][arm]
        lines.append(f"| {arm} | {row['coherent_persona']} | {row['shows_persona']} | {row['incoherent']} | "
                     f"{row['known_labels']} | {row['missing_labels']} |")
    lines.extend(["", "## Predictions", ""])
    for prediction_id, prediction in result["predictions"].items():
        lines.append(f"### {prediction_id} ({prediction['confidence']})")
        lines.append("")
        lines.append(prediction["text"])
        if "registered_text" in prediction:
            lines.append("")
            lines.append(f"Registered wording: {prediction['registered_text']}")
        lines.append("")
        lines.append(f"Verdict: **{prediction['verdict']}**. {prediction.get('observed', '')}")
        lines.append("")
    lines.extend(["## Interpretation", ""])
    lines.extend(result["interpretation"])
    lines.extend(["", "## Reader-free per-turn summaries", "",
                  "Positive delta means arm 2 assigns more probability to the persona sentence than the twin. KL is `KL(arm 2 || twin)` for the next-token distributions.", "",
                  "| Turn | Comparisons | Mean delta | Median delta | Mean KL | Median KL |",
                  "|---:|---:|---:|---:|---:|---:|"])
    for turn, row in result["reader_free_summary"].items():
        fmt = lambda value: "n/a" if value is None else f"{value:.6g}"
        lines.append(f"| {turn} | {row['comparisons']} | {fmt(row['mean_delta'])} | {fmt(row['median_delta'])} | "
                     f"{fmt(row['mean_kl_arm2_to_twin'])} | {fmt(row['median_kl_arm2_to_twin'])} |")
    lines.extend(["", "## Cache-sanity distribution", "",
                  f"{json.dumps(result['cache_sanity_distribution'], ensure_ascii=False)}", "",
                  "## Missing labels", ""])
    if result["missing_labels"]:
        for row in result["missing_labels"][:50]:
            lines.append(f"- {row['arm']} {row['unit_id']} turn {row['turn']}: {row['reason']}")
        if len(result["missing_labels"]) > 50:
            lines.append(f"- … and {len(result['missing_labels']) - 50} more")
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, *, validate_inventory: bool = True) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--reader-free", required=True)
    parser.add_argument("--random-reader-free", required=True)
    parser.add_argument("--cache-sanity", required=True)
    parser.add_argument("--determinism", required=True)
    parser.add_argument("--ledger-checks", required=True)
    parser.add_argument("--call-sequences", required=True)
    parser.add_argument("--zero-hook", required=True)
    parser.add_argument("--labels-ds", required=True)
    parser.add_argument("--keys-ds", required=True)
    parser.add_argument("--labels-luna", required=True)
    parser.add_argument("--keys-luna", required=True)
    parser.add_argument("--known-ds", required=True)
    parser.add_argument("--known-luna", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--slot-only", action="store_true")
    args = parser.parse_args(argv)

    raw_dir = Path(args.answers).parent
    pilot_meta_path = raw_dir / "pilot-meta.json"
    run_meta_path = raw_dir / "run-meta.json"
    if not run_meta_path.is_file():
        raise ValueError("run-meta.json is required before scoring")
    run_meta = read_json(run_meta_path)
    expected_unit_parameters = run_meta.get("parameters", {}).get("unit_parameters")
    if not isinstance(expected_unit_parameters, dict):
        raise ValueError("run-meta.json lacks unit parameter provenance")
    if validate_inventory:
        if not pilot_meta_path.is_file():
            raise ValueError("pilot and run metadata are required before labels are read")
        pilot_meta = read_json(pilot_meta_path)
        required_cut = bool((pilot_meta.get("pilot_size_decision") or {}).get("over_90_minutes"))
        if bool(args.slot_only) != required_cut:
            raise ValueError("scorer cut differs from the pilot decision")
        if bool(run_meta.get("slot_only")) != bool(args.slot_only):
            raise ValueError("run metadata does not match the scorer cut")
        if bool(run_meta.get("parameters", {}).get("slot_only")) != bool(args.slot_only):
            raise ValueError("run parameter record does not match the scorer cut")
    answers = read_jsonl(args.answers)
    reader_free = read_jsonl(args.reader_free)
    random_reader_free = read_jsonl(args.random_reader_free)
    cache_sanity = read_jsonl(args.cache_sanity)
    determinism = read_jsonl(args.determinism)
    ledger_checks = read_jsonl(args.ledger_checks)
    call_sequences = read_jsonl(args.call_sequences)
    call_by_unit = {str(row.get("unit_id")): row for row in call_sequences}
    zero_checks = read_json(args.zero_hook)
    ds_labels = read_jsonl(args.labels_ds, optional=True)
    ds_keys = read_jsonl(args.keys_ds, optional=True)
    luna_labels = read_jsonl(args.labels_luna, optional=True)
    luna_keys = read_jsonl(args.keys_luna, optional=True)

    units = sorted({str(row["unit_id"]) for row in answers})
    unit_kinds = {str(row["unit_id"]): str(row.get("kind", "")) for row in answers}
    expected_n = 21 if args.slot_only else 29
    if validate_inventory and len(units) != expected_n:
        raise ValueError(f"expected {expected_n} units, found {len(units)}")
    used_layers = sorted({int(row["layer"]) for row in answers})
    zero_by_key = {(int(row.get("layer", -1)), int(row.get("turn", -1))): row for row in zero_checks}
    zero_expected = {(layer, turn) for layer in used_layers for turn in range(1, 6)}
    zero_ok = (set(zero_by_key) >= zero_expected and
               all(zero_by_key[key].get("batch_size") == 1 and
                   isinstance(zero_by_key[key].get("plain_token_ids"), list) and
                   zero_by_key[key].get("plain_token_ids") == zero_by_key[key].get("hooked_token_ids")
                   for key in zero_expected))

    required_pilot_ids = {"p92-02", "p92-08"}
    determinism_ids = {str(row.get("unit_id")) for row in determinism}
    def det_row_passes(row):
        sequence = call_by_unit.get(str(row.get("unit_id")), {})
        pushed_run = [call.get("logit_hash") for call in sequence.get("arm2_calls", [])]
        twin_run = [call.get("logit_hash") for call in sequence.get("arm3_calls", [])]
        pushed_replay = row.get("pushed_logit_hashes_replay", [])
        twin_replay = row.get("twin_logit_hashes_replay", [])
        count = int(row.get("calls", -1))
        return (len(pushed_run) == count == len(twin_run) == len(pushed_replay) == len(twin_replay) and
                all(isinstance(value, str) and value for value in pushed_run + twin_run) and
                pushed_run == pushed_replay and twin_run == twin_replay and
                row.get("first_differing_call") is None and
                row.get("pushed_bitwise_equal") is True and
                row.get("twin_bitwise_equal") is True)

    det_ok = bool(determinism) and all(det_row_passes(row) for row in determinism)
    if validate_inventory and not args.slot_only:
        det_ok = det_ok and required_pilot_ids <= determinism_ids
    cache_top1 = sum(row.get("cached_top1_id") == row.get("full_top1_id") for row in cache_sanity)
    cache_count = len(cache_sanity)
    cache_needed = int(math.ceil(0.95 * cache_count))
    per_check_max = [max((float(value) for value in row.get("top100_abs_diffs", [])), default=math.inf)
                     for row in cache_sanity]
    cache_max = max(per_check_max, default=math.inf)
    required_cache = len(units) * 5
    cache_ok = cache_count >= required_cache
    cache_ok = cache_ok and cache_top1 >= cache_needed and cache_max <= 1.0

    ds_known = _known_result("ds", args.known_ds)
    luna_known = _known_result("luna", args.known_luna)
    ds_by_key, missing_ds = _label_lookup(answers, ds_keys, ds_labels, require_all=True)
    luna_by_key, missing_luna = _label_lookup(answers, luna_keys, luna_labels, require_all=False)
    ds_item = label123.latest_label_rows(ds_labels)
    luna_item = label123.latest_label_rows(luna_labels)
    sample_policy_ok = False
    try:
        expected_luna_items = label125.luna_sample(label125.make_items(answers), ds_labels)
        expected_luna_keys = {
            (str(item["public"]["item_id"]), int(item["public"]["question_index"]),
             str(item["public"]["answer_hash"])) for item in expected_luna_items
        }
        actual_luna_keys = {
            (str(row.get("item_id", "")), int(row.get("question_index", -1)),
             str(row.get("answer_hash", ""))) for row in luna_keys
        }
        sample_policy_ok = expected_luna_keys == actual_luna_keys
    except (ValueError, KeyError, TypeError):
        expected_luna_keys = set()
    agreement = _cohen_kappa(ds_item, luna_item, luna_keys)
    agreement["sample_policy_match"] = bool(sample_policy_ok)
    agreement["gate_passed"] = bool(agreement["gate_passed"] and sample_policy_ok)
    kappa_ok = bool(agreement["gate_passed"])
    label_complete = not missing_ds and not missing_luna

    answer_index = {(str(row["unit_id"]), int(row["turn"])): row
                    for row in answers if row["arm"] == "arm2"}
    ledger_by_unit = defaultdict(list)
    ledger_ok = bool(ledger_checks)
    provenance_rows = (answers + reader_free + random_reader_free + cache_sanity +
                       determinism + ledger_checks + zero_checks)
    provenance_calls = []
    for sequence in call_sequences:
        provenance_calls.extend(sequence.get(key, []) for key in
                                ("arm2_calls", "arm3_calls", "arm1_calls", "arm4_calls"))
        for pair in sequence.get("random_pairs", []):
            provenance_calls.extend((pair.get("random_calls", []), pair.get("twin_calls", [])))
    provenance_ok = (all(
        row.get("parameters") == expected_unit_parameters for row in provenance_rows
    ) and all(row.get("parameters") == expected_unit_parameters for row in call_sequences) and all(
        call.get("parameters") == expected_unit_parameters
        for calls in provenance_calls for call in calls
    ) and all(isinstance(call.get("logit_hash"), str) and call.get("logit_hash")
              for calls in provenance_calls for call in calls))
    for row in ledger_checks:
        left = row.get("arm2_ledger_token_ids")
        twin = row.get("arm3_ledger_token_ids")
        template = row.get("template_token_ids")
        ledger_ok = ledger_ok and isinstance(left, list) and left == twin == template
        if int(row.get("turn", 0)) == 1:
            ledger_ok = ledger_ok and row.get("arm2_hook_removed") is True
        expected_hook_count = int(row.get("arm2_calls_this_turn", 0)) if int(row.get("turn", 0)) == 1 else 0
        ledger_ok = ledger_ok and row.get("arm2_hook_state_ok") is True
        ledger_ok = ledger_ok and row.get("twin_hook_state_ok") is True
        ledger_ok = ledger_ok and int(row.get("arm2_hook_calls", -1)) == expected_hook_count
        ledger_ok = ledger_ok and row.get("call_logs_equal") is True
        answer = answer_index.get((str(row.get("unit_id")), int(row.get("turn", 0))))
        if answer is None or not isinstance(left, list):
            ledger_ok = False
        else:
            token_ids = answer.get("token_ids", [])
            ledger_ok = ledger_ok and len(left) >= len(token_ids) and left[-len(token_ids):] == token_ids
        ledger_by_unit[str(row.get("unit_id"))].append(row)
    for rows in ledger_by_unit.values():
        rows.sort(key=lambda row: int(row["turn"]))
        for previous, current in zip(rows, rows[1:]):
            prior_ids = previous.get("arm2_ledger_token_ids", [])
            current_ids = current.get("arm2_ledger_token_ids", [])
            ledger_ok = ledger_ok and current_ids[:len(prior_ids)] == prior_ids
    ledger_ok = ledger_ok and set(call_by_unit) == set(units)
    def paired_calls_match(left, right):
        fields = ("token_ids", "shape", "cache_length_before", "cache_length_after", "turn")
        return len(left) > 0 and len(left) == len(right) and all(
            all(a.get(field) == b.get(field) for field in fields) and
            a.get("shape") == [1, len(a.get("token_ids", []))]
            for a, b in zip(left, right)
        )

    for unit in units:
        row = call_by_unit.get(unit, {})
        left = row.get("arm2_calls", [])
        right = row.get("arm3_calls", [])
        ledger_ok = ledger_ok and paired_calls_match(left, right)
        ledger_ok = ledger_ok and all(call.get("hook_on") is (int(call.get("turn", 0)) == 1)
                                      for call in left)
        ledger_ok = ledger_ok and all(call.get("hook_on") is False for call in right)
        arm1_calls, arm4_calls = row.get("arm1_calls", []), row.get("arm4_calls", [])
        ledger_ok = ledger_ok and {int(call.get("turn", 0)) for call in arm1_calls} == {2, 3, 4, 5}
        ledger_ok = ledger_ok and all(call.get("hook_on") is True for call in arm1_calls)
        ledger_ok = ledger_ok and {int(call.get("turn", 0)) for call in arm4_calls} == {1, 2, 3, 4, 5}
        ledger_ok = ledger_ok and all(call.get("hook_on") is False for call in arm4_calls)
        pairs = row.get("random_pairs", [])
        ledger_ok = ledger_ok and len(pairs) == 3
        for pair in pairs:
            random_calls = pair.get("random_calls", [])
            twin_calls = pair.get("twin_calls", [])
            ledger_ok = ledger_ok and paired_calls_match(random_calls, twin_calls)
            ledger_ok = ledger_ok and paired_calls_match(left, random_calls)
            ledger_ok = ledger_ok and all(
                call.get("hook_on") is (int(call.get("turn", 0)) == 1)
                for call in random_calls
            )
            ledger_ok = ledger_ok and all(call.get("hook_on") is False for call in twin_calls)
    if validate_inventory:
        expected_ledger = expected_n * 5
        ledger_ok = ledger_ok and len(ledger_checks) == expected_ledger
    ledger_ok = ledger_ok and provenance_ok

    gates = [
        _gate(1, "Zero-vector hook has same token IDs at batch size one", zero_ok,
              {"checks": len(zero_expected & set(zero_by_key)), "expected": len(zero_expected)}),
        _gate(2, "Pushed path and no-hook twin are bitwise deterministic on both pilot units", det_ok,
              {"units": sorted(determinism_ids), "rows": len(determinism), "all_equal": all(
                  row.get("pushed_bitwise_equal") is True and
                  row.get("twin_bitwise_equal") is True for row in determinism)}),
        _gate(3, "Twin cache agrees with one-forward full re-encoding", cache_ok,
              {"comparisons": cache_count, "required_comparisons": required_cache,
               "top1_agree": cache_top1,
               "top1_required": cache_needed, "max_top100_abs_diff": cache_max,
               "max_required": 1.0}),
        _gate(4, "DeepSeek and Luna each pass 11 of 12 known examples", bool(ds_known["passed"] and luna_known["passed"]),
              {"deepseek": ds_known, "luna": luna_known}),
        _gate(5, "DeepSeek/Luna agreement is at least 0.80 and kappa is at least 0.60 when defined", kappa_ok,
              {"agreement": agreement}),
        _gate(6, "Template ledger, paired call plans and hook coverage pass at every turn", ledger_ok,
              {"checks": len(ledger_checks), "expected": expected_n * 5,
               "parameter_rows_match": provenance_ok}),
        _gate(7, "All current DeepSeek labels and private joins match answer hashes", label_complete,
              {"missing_or_stale_deepseek": len(missing_ds),
               "missing_or_stale_luna_sample": len(missing_luna)}),
    ]
    all_gates = all(gate["passed"] for gate in gates)
    answers_by_key = {(str(row["arm"]), str(row["unit_id"]), int(row["turn"])): row
                      for row in answers}
    turn5 = {arm: _count_for_arm(ds_by_key, units, arm, 5, answers_by_key)
             for arm in ("arm1", "arm2", "arm3", "arm4")}

    random_by = defaultdict(dict)
    for row in random_reader_free:
        random_by[(str(row["unit_id"]), int(row["turn"]))][int(row["direction"])] = float(row["delta"])
    reader_free_by = {(str(row["unit_id"]), int(row["turn"])): row
                      for row in reader_free}
    above_by_turn = {}
    comparative = {}
    no_larger_random_by_turn = {}
    for turn in range(1, 6):
        above = []
        within = []
        for unit in units:
            deltas = random_by.get((unit, turn), {})
            measure = reader_free_by.get((unit, turn))
            if measure is None or len(deltas) != 3:
                continue
            real = float(measure["delta"])
            random_mean = sum(deltas.values()) / 3
            comparative[(unit, turn)] = {"real_delta": real, "random_mean": random_mean,
                                         "above_random": real > random_mean,
                                         "kl_arm2_to_twin": float(measure["kl_arm2_to_twin"])}
            if real > random_mean:
                above.append(unit)
            else:
                within.append(unit)
        above_by_turn[turn] = above
        no_larger_random_by_turn[turn] = within
    no_larger_random_at_turn5 = no_larger_random_by_turn.get(5, [])

    counts_t2 = {arm: _count_for_arm(ds_by_key, units, arm, 2, answers_by_key)
                 for arm in ("arm1", "arm2", "arm3", "arm4")}
    incoherent_diff = turn5["arm1"]["incoherent"] - turn5["arm2"]["incoherent"]
    counts_for_p3_t2 = len(above_by_turn.get(2, []))
    counts_for_p3_t5 = len(above_by_turn.get(5, []))
    diff_2_3 = turn5["arm2"]["coherent_persona"] - turn5["arm3"]["coherent_persona"]
    prediction_thresholds = _prediction_thresholds(len(units))
    verdicts = {}
    if not all_gates:
        first_failed = next(gate for gate in gates if not gate["passed"])
        for key, prediction in PREDICTIONS.items():
            verdicts[key] = {**prediction, "verdict": f"not read (gate {first_failed['number']})",
                             "observed": "At least one pre-read gate failed."}
    else:
        verdicts = {
            "125.1": {**PREDICTIONS["125.1"], "verdict": "held" if abs(diff_2_3) <= 5 else "not held",
                      "observed": f"arm 2 {turn5['arm2']['coherent_persona']}, arm 3 {turn5['arm3']['coherent_persona']} of {len(units)} selected units; difference {diff_2_3}."},
            "125.2": {**PREDICTIONS["125.2"], "verdict": "held" if turn5["arm3"]["coherent_persona"] >= prediction_thresholds["125.2_minimum"] else "not held",
                      "observed": f"arm 3 has {turn5['arm3']['coherent_persona']} coherent-persona labels at turn 5, among {len(units)} selected units."},
            "125.3": {**PREDICTIONS["125.3"], "verdict": "held" if counts_for_p3_t2 >= prediction_thresholds["125.3_minimum"] else "not held",
                      "observed": f"{counts_for_p3_t2}/{len(units)} units have delta_2 above their three-direction mean."},
            "125.4": {**PREDICTIONS["125.4"], "verdict": "held" if counts_for_p3_t5 <= counts_for_p3_t2 - 5 else "not held",
                      "observed": f"The above-random count changes from {counts_for_p3_t2}/{len(units)} at turn 2 to {counts_for_p3_t5}/{len(units)} at turn 5."},
            "125.5": {**PREDICTIONS["125.5"], "verdict": "held" if incoherent_diff >= prediction_thresholds["125.5_minimum"] else "not held",
                      "observed": f"arm 1 has {turn5['arm1']['incoherent']} and arm 2 {turn5['arm2']['incoherent']} known incoherent answers; difference {incoherent_diff}."},
            "125.6": {**PREDICTIONS["125.6"], "verdict": "held" if turn5["arm4"]["coherent_persona"] <= prediction_thresholds["125.6_maximum"] else "not held",
                      "observed": f"arm 4 has {turn5['arm4']['coherent_persona']} coherent-persona labels at turn 5, among {len(units)} selected units."},
        }

    interpretation = []
    if not all_gates:
        interpretation.append(f"Predictions are withheld because gate {next(g['number'] for g in gates if not g['passed'])} failed.")
    else:
        if verdicts["125.1"]["verdict"] == "held" and verdicts["125.3"]["verdict"] == "held":
            interpretation.append("The pushed cache moves the model in the persona's direction, and the reader cannot see that at turn 5. This is not a null result.")
        if diff_2_3 >= 6:
            interpretation.append("Arm 2 beats arm 3 by at least six units, so pushed memory carried more than the words on this reader.")
        if turn5["arm3"]["coherent_persona"] >= 20 and abs(diff_2_3) <= 5:
            interpretation.append("Arm 3 is near arm 2 at a high count. Text may be saturating the reader, so the reader-free measure is the better evidence.")
        if no_larger_random_at_turn5:
            named_units = ", ".join(no_larger_random_at_turn5)
            interpretation.append(f"At turn 5, the real cache delta is no larger than the mean random-direction delta for {len(no_larger_random_at_turn5)} units ({named_units}). For these units, the kept cache changes the model but not toward this persona on this measure.")
        if verdicts["125.5"]["verdict"] == "held":
            interpretation.append(f"Arm 1 has at least {prediction_thresholds['125.5_minimum']} more incoherent answers than arm 2 at turn 5. Pushing every turn breaks answers more than it holds a persona.")
        if not interpretation:
            interpretation.append("No registered interpretation rule above was triggered; report the counts and deltas without a broader claim.")

    if args.slot_only:
        notes = {
            "125.2": f"Slot-only criterion: at least {prediction_thresholds['125.2_minimum']} of {len(units)} units, scaled up from 6 of 29.",
            "125.3": f"Slot-only criterion: at least {prediction_thresholds['125.3_minimum']} of {len(units)} units, scaled up from 20 of 29.",
            "125.5": f"Slot-only criterion: at least {prediction_thresholds['125.5_minimum']} units, scaled up from 6 of 29.",
            "125.6": f"Slot-only criterion: at most {prediction_thresholds['125.6_maximum']} of {len(units)} units, scaled down from 2 of 29.",
        }
        for key, note in notes.items():
            verdicts[key]["slot_only_note"] = note

    cache_diffs = per_check_max
    ordered = sorted(cache_diffs)
    cache_distribution = {"n": len(ordered), "min": min(ordered) if ordered else None,
                          "median": median(ordered) if ordered else None,
                          "p90": ordered[min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1)] if ordered else None,
                          "max": max(ordered) if ordered else None,
                          "top1_agree": cache_top1, "comparisons": cache_count}
    reader_free_summary = {}
    for turn in range(1, 6):
        rows = [row for (unit, row_turn), row in reader_free_by.items() if row_turn == turn]
        deltas = [float(row["delta"]) for row in rows]
        kls = [float(row["kl_arm2_to_twin"]) for row in rows]
        reader_free_summary[str(turn)] = {
            "comparisons": len(rows),
            "mean_delta": mean(deltas) if deltas else None,
            "median_delta": median(deltas) if deltas else None,
            "mean_kl_arm2_to_twin": mean(kls) if kls else None,
            "median_kl_arm2_to_twin": median(kls) if kls else None,
        }
    result = {
        "part": 125, "prediction_author": "Claude", "slot_only": bool(args.slot_only),
        "prediction_thresholds": prediction_thresholds,
        "unit_count": len(units), "unit_ids": units, "unit_kinds": unit_kinds,
        "all_gates_passed": all_gates, "gates": gates,
        "predictions": verdicts,
        "turn5_counts": turn5, "turn2_counts": counts_t2,
        "reader_free_above_random": {str(turn): len(rows) for turn, rows in above_by_turn.items()},
        "reader_free_not_above_random": {str(turn): len(rows) for turn, rows in no_larger_random_by_turn.items()},
        "reader_free_summary": reader_free_summary,
        "reader_free_comparisons": {f"{unit}:turn{turn}": values
                                    for (unit, turn), values in comparative.items()},
        "missing_labels": missing_ds + missing_luna,
        "missing_or_stale_deepseek_labels": len(missing_ds),
        "missing_or_stale_luna_labels": len(missing_luna),
        "turn5_arm2_minus_arm3": diff_2_3,
        "turn5_arm1_minus_arm2_incoherent": incoherent_diff,
        "cache_sanity_distribution": cache_distribution,
        "interpretation": interpretation,
    }
    out = Path(args.out)
    _write_json(out / "score125.json", result)
    _write_markdown(out / "SCORE.md", result, _write_report(out, result))
    print(json.dumps({"all_gates_passed": all_gates, "predictions": verdicts,
                      "output": str(out / "score125.json")}, ensure_ascii=False), flush=True)
    return 0 if all_gates else 1


if __name__ == "__main__":
    raise SystemExit(main())
