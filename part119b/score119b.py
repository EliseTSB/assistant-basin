#!/usr/bin/env python3
"""CPU-only Part 119b scorer. Every prediction and report value is computed here."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT / "part119", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rows119  # noqa: E402
import score119  # noqa: E402 -- registered broken-answer and bootstrap definitions
import rows119b as rb  # noqa: E402
import label119 as reader_shared  # noqa: E402 -- same known-example gate definitions
from part86 import triage86  # noqa: E402 -- Part 86's registered model-free break rule

DOSES = rb.DOSES
KIND_CLASSES = ("agreed_slot", "agreed_concept", "disputed")


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path, *, optional: bool = False) -> dict | None:
    path = repo_path(path)
    if optional and not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path, *, optional: bool = False) -> list[dict]:
    path = repo_path(path)
    if optional and not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL in {path}:{line_no}")
            result.append(row)
    return result


def keyed(rows: list[dict], field: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = str(row[field])
        if key in result:
            raise ValueError(f"duplicate {field} {key!r}")
        result[key] = row
    return result


def compose_answers(local_answers: list[dict], part119_answers: list[dict],
                    planned: list[dict]) -> tuple[list[dict], dict]:
    """Join only the registered 0.8 Part 119 rows into Part 119b dose ids."""
    local = keyed(local_answers, "row_id")
    prior = keyed(part119_answers, "row_id")
    combined = {}
    if "plain" in prior:
        combined["plain"] = prior["plain"]
    missing_reused = []
    for row in planned:
        row_id = row["row_id"]
        if row["reused"]:
            source = prior.get(row["source_row_id"])
            if source is None:
                missing_reused.append(row_id)
                continue
            combined[row_id] = {**source, "row_id": row_id,
                                "row_type": row["row_type"], "dose": row["dose"],
                                "source_row_id": row["source_row_id"]}
        elif row_id in local:
            combined[row_id] = local[row_id]
    return list(combined.values()), {"local_rows": len(local),
                                     "expected_planned_rows": len(planned),
                                     "combined_rows": len(combined),
                                     "missing_planned_rows": sorted(
                                         {row["row_id"] for row in planned} - set(combined)),
                                     "missing_reused": missing_reused,
                                     "extra_local": sorted(set(local) -
                                                            {row["row_id"] for row in planned
                                                             if not row["reused"]})}


def resolved(position: dict, threshold: float = 1.0) -> bool:
    """Part 119b's registered floor: all three centered norms reach 1.0."""
    # The runner records this from unrounded norms; use it to avoid a 1e-5
    # serialization round-up turning an actually unresolved point into a pass.
    if threshold == 1.0 and type(position.get("resolved")) is bool:
        return position["resolved"]
    return all(float(position.get(field, 0.0)) >= threshold
               for field in ("nA", "n2", "nPair"))


def interaction_share(position: dict) -> float | None:
    n_pair = float(position.get("nPair", 0.0))
    if n_pair <= 0:
        return None
    return (float(position.get("nI", 0.0)) / n_pair) ** 2


def _position_map(row: dict) -> dict[int, dict]:
    return {int(position["pos"]): position for position in row.get("positions", [])}


def validate_shared_support(interactions: list[dict]) -> dict:
    """Check the runner reused the same top-10 union T at all four doses."""
    by_base = defaultdict(dict)
    problems = []
    for row in interactions:
        key = (str(row["kind"]), str(row["first"]), str(row["second"]))
        dose = float(row["dose"])
        if dose in by_base[key]:
            problems.append({"row": key, "dose": dose, "problem": "duplicate dose"})
        by_base[key][dose] = row
    checked = 0
    for key, dose_rows in by_base.items():
        if not all(dose in dose_rows for dose in DOSES):
            continue
        maps = {dose: _position_map(dose_rows[dose]) for dose in DOSES}
        common = set.intersection(*(set(value) for value in maps.values())) if maps else set()
        for pos in common:
            hashes = [maps[dose][pos].get("t_hash") for dose in DOSES]
            sizes = [maps[dose][pos].get("t_size") for dose in DOSES]
            if any(value is None for value in hashes) or len(set(hashes)) != 1 or len(set(sizes)) != 1:
                problems.append({"row": key, "pos": pos, "problem": "T differs across doses"})
            checked += 1
    return {"passed": not problems, "positions_checked": checked,
            "problems": problems[:25], "problem_count": len(problems)}


def path2_index(interactions: list[dict]) -> dict[tuple[str, str, str], dict[float, dict]]:
    result = defaultdict(dict)
    for row in interactions:
        key = (str(row["kind"]), str(row["first"]), str(row["second"]))
        dose = float(row["dose"])
        if dose in result[key]:
            raise ValueError(f"duplicate Path 2 row/dose {key} at {dose}")
        result[key][dose] = row
    return result


def _row_shares(row: dict, *, filter_resolved: bool = False) -> list[float]:
    values = []
    for position in row.get("positions", []):
        if filter_resolved and not resolved(position):
            continue
        value = interaction_share(position)
        if value is not None:
            values.append(value)
    return values


def _common_support(path_rows: dict[tuple[str, str, str], dict[float, dict]]) -> dict:
    """Map each directed row to positions present and resolved at every dose."""
    support = {}
    for key, dose_rows in path_rows.items():
        if not all(dose in dose_rows for dose in DOSES):
            support[key] = set()
            continue
        maps = {dose: _position_map(dose_rows[dose]) for dose in DOSES}
        present = set.intersection(*(set(value) for value in maps.values()))
        support[key] = {pos for pos in present
                        if all(resolved(maps[dose][pos]) for dose in DOSES)}
    return support


def path2_coverage(interactions: list[dict], expected: dict[str, set[tuple[str, str, str]]] | None = None) -> dict:
    index = path2_index(interactions)
    expected = expected or {
        "real": {(kind, a, b) for kind, a, b in index if kind == "real"},
        "control": {(kind, a, b) for kind, a, b in index if kind == "control"},
    }
    common = _common_support(index)
    output = {}
    for kind in ("real", "control"):
        row_keys = expected.get(kind, set())
        row_counts = {dose: 0 for dose in DOSES}
        position_counts = {dose: 0 for dose in DOSES}
        total_positions = {dose: 0 for dose in DOSES}
        resolved_positions = {dose: 0 for dose in DOSES}
        common_rows = common_positions = 0
        for key in row_keys:
            dose_rows = index.get(key, {})
            row_common = common.get(key, set())
            if row_common:
                common_rows += 1
                common_positions += len(row_common)
            for dose in DOSES:
                row = dose_rows.get(dose)
                if row is None:
                    continue
                positions = row.get("positions", [])
                total_positions[dose] += len(positions)
                valid = [position for position in positions if resolved(position)]
                resolved_positions[dose] += len(valid)
                if valid:
                    row_counts[dose] += 1
                position_counts[dose] += len(valid)
        output[kind] = {
            "expected_rows": len(row_keys),
            "rows_present_by_dose": {dose_text: sum(
                dose in index.get(key, {}) for key in row_keys)
                for dose_text, dose in ((rb.dose_text(d), d) for d in DOSES)},
            "rows_with_resolved_position": {rb.dose_text(dose): row_counts[dose]
                                            for dose in DOSES},
            "row_coverage": {rb.dose_text(dose):
                             row_counts[dose] / len(row_keys) if row_keys else None
                             for dose in DOSES},
            "positions_total": {rb.dose_text(dose): total_positions[dose] for dose in DOSES},
            "positions_resolved": {rb.dose_text(dose): resolved_positions[dose]
                                   for dose in DOSES},
            "position_coverage": {rb.dose_text(dose):
                                  resolved_positions[dose] / total_positions[dose]
                                  if total_positions[dose] else None for dose in DOSES},
            "rows_with_resolved_position_all_four_doses": common_rows,
            "all_four_resolved_positions": common_positions,
            "all_four_row_coverage": common_rows / len(row_keys) if row_keys else None,
        }
    return output


def summarize_path2(interactions: list[dict], expected: dict[str, set[tuple[str, str, str]]] | None = None) -> dict:
    index = path2_index(interactions)
    expected = expected or {
        "real": {(kind, a, b) for kind, a, b in index if kind == "real"},
        "control": {(kind, a, b) for kind, a, b in index if kind == "control"},
    }
    support = _common_support(index)
    summary = {"coverage": path2_coverage(interactions, expected), "by_dose": {}}
    for dose in DOSES:
        cell = {}
        for kind in ("real", "control"):
            all_shares, resolved_shares, common_row_shares = [], [], []
            row_b_a, row_b_2 = [], []
            all_position_shares, resolved_position_shares = [], []
            for key in expected.get(kind, set()):
                row = index.get(key, {}).get(dose)
                if row is None:
                    continue
                shares = _row_shares(row)
                all_position_shares.extend(shares)
                if shares:
                    all_shares.append(float(median(shares)))
                positions = _position_map(row)
                resolved_values = [interaction_share(positions[pos]) for pos in positions
                                   if resolved(positions[pos])]
                resolved_values = [value for value in resolved_values if value is not None]
                resolved_position_shares.extend(resolved_values)
                if resolved_values:
                    resolved_shares.append(float(median(resolved_values)))
                common_positions = support.get(key, set())
                common_values = [interaction_share(positions[pos]) for pos in common_positions
                                 if pos in positions]
                common_values = [value for value in common_values if value is not None]
                if common_values:
                    common_row_shares.append(float(median(common_values)))
                    b_a = [positions[pos].get("b_A") for pos in common_positions
                           if pos in positions and positions[pos].get("b_A") is not None]
                    b_2 = [positions[pos].get("b_2") for pos in common_positions
                           if pos in positions and positions[pos].get("b_2") is not None]
                    if b_a:
                        row_b_a.append(float(median(b_a)))
                    if b_2:
                        row_b_2.append(float(median(b_2)))
            coverage = summary["coverage"][kind]
            cell[kind] = {
                "rows_with_any_share": len(all_shares),
                "median_row_share_all_positions": _median_or_none(all_shares),
                "median_share_all_positions_pooled": _median_or_none(all_position_shares),
                "median_row_share_resolved_at_dose": _median_or_none(resolved_shares),
                "median_share_resolved_at_dose_pooled": _median_or_none(resolved_position_shares),
                "median_row_share_common_resolved_all_doses": _median_or_none(common_row_shares),
                "rows_with_common_resolved_positions": len(common_row_shares),
                "median_row_b_A_common_resolved": _median_or_none(row_b_a),
                "median_row_b_2_common_resolved": _median_or_none(row_b_2),
                "row_coverage": coverage["row_coverage"][rb.dose_text(dose)],
                "position_coverage": coverage["position_coverage"][rb.dose_text(dose)],
                "common_row_coverage": coverage["all_four_row_coverage"],
                "common_position_count": coverage["all_four_resolved_positions"],
            }
        summary["by_dose"][rb.dose_text(dose)] = cell
    return summary


def _median_or_none(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def gate3_result(summary: dict, interactions: list[dict], path_support: dict) -> dict:
    real = summary["by_dose"]["0.2"]["real"]
    coverage = summary["coverage"]["real"]
    common_row_coverage = coverage["all_four_row_coverage"] or 0.0
    median_share = real["median_row_share_common_resolved_all_doses"]
    enough_rows = common_row_coverage >= .70
    share_small = median_share is not None and median_share <= .15
    return {"passed": bool(path_support["passed"] and enough_rows and share_small),
            "read": True, "rows_kept": coverage["rows_with_resolved_position_all_four_doses"],
            "rows_expected": coverage["expected_rows"],
            "row_coverage_all_four_doses": common_row_coverage,
            "positions_kept_all_four_doses": coverage["all_four_resolved_positions"],
            "median_row_share_0.2_common_resolved": median_share,
            "coverage_clause_passed": enough_rows,
            "share_clause_passed": share_small,
            "shared_T_passed": path_support["passed"],
            "thresholds": {"row_coverage_at_least": .70,
                           "median_share_at_most": .15}}


def gate2_result(interactions: list[dict], reference: list[dict],
                 expected: dict[str, set[tuple[str, str, str]]]) -> dict:
    index = path2_index(interactions)
    reference_by_key = {(str(row["kind"]), str(row["first"]), str(row["second"])): row
                        for row in reference}
    class_rows = {"real": [], "control": []}
    relative_npair = []
    compared_rows = {"real": 0, "control": 0}
    position_pairs = 0
    missing = []
    mismatch_positions = 0
    for kind in ("real", "control"):
        for key in expected.get(kind, set()):
            current = index.get(key, {}).get(.8)
            old = reference_by_key.get(key)
            if current is None or old is None:
                missing.append({"kind": kind, "first": key[1], "second": key[2]})
                continue
            compared_rows[kind] += 1
            current_pos = _position_map(current)
            old_pos = _position_map(old)
            common = sorted(set(current_pos) & set(old_pos))
            if set(current_pos) != set(old_pos):
                mismatch_positions += 1
            old_shares, new_shares = [], []
            for pos in common:
                before, after = old_pos[pos], current_pos[pos]
                old_pair, new_pair = float(before["nPair"]), float(after["nPair"])
                if old_pair > 0:
                    relative_npair.append(abs(new_pair - old_pair) / old_pair)
                old_share = interaction_share(before)
                new_share = interaction_share(after)
                if old_share is not None:
                    old_shares.append(old_share)
                if new_share is not None:
                    new_shares.append(new_share)
                position_pairs += 1
            if old_shares and new_shares:
                class_rows[kind].append({"old": float(median(old_shares)),
                                         "new": float(median(new_shares))})
    observed = {}
    class_ok = {}
    for kind, target in (("real", .4707), ("control", .3686)):
        new_values = [row["new"] for row in class_rows[kind]]
        value = _median_or_none(new_values)
        observed[kind] = {"median_row_share_0.8": value, "target": target,
                          "rows_compared": compared_rows[kind],
                          "reference_row_share_median": _median_or_none(
                              [row["old"] for row in class_rows[kind]])}
        class_ok[kind] = (compared_rows[kind] == len(expected.get(kind, set()))
                          and value is not None and abs(value - target) <= .01)
    median_relative = _median_or_none(relative_npair)
    complete = not missing and mismatch_positions == 0
    return {"passed": bool(complete and all(class_ok.values())
                           and median_relative is not None and median_relative < .01),
            "complete": complete, "missing_rows": missing[:25],
            "missing_row_count": len(missing), "position_list_mismatch_rows": mismatch_positions,
            "position_pairs_compared": position_pairs, "row_medians": observed,
            "class_thresholds_passed": class_ok,
            "median_relative_nPair_difference": median_relative,
            "nPair_threshold": .01,
            "rows_compared": compared_rows}


def _broken_stats(answers: list[dict], planned: list[dict]) -> dict:
    answer_map = keyed(answers, "row_id")
    by_group = defaultdict(lambda: {"broken": 0, "three_word_repeat": 0, "total": 0})
    missing = defaultdict(list)
    for row in planned:
        answer = answer_map.get(row["row_id"])
        group = (row["row_type"], float(row["dose"]))
        if answer is None:
            missing[group].append(row["row_id"])
            continue
        text = str(answer.get("text", ""))
        status = triage86.triage_answer(text)
        words = [word.casefold() for word in triage86.WORDS.findall(text)]
        trigrams = Counter(tuple(words[index:index + 3]) for index in range(len(words) - 2))
        by_group[group]["broken"] += int(status == "broken")
        by_group[group]["three_word_repeat"] += int(any(count >= 3 for count in trigrams.values()))
        by_group[group]["total"] += 1
    output = {}
    for (row_type, dose), counts in by_group.items():
        expected = sum(row["row_type"] == row_type and row["dose"] == dose for row in planned)
        output.setdefault(rb.dose_text(dose), {})[row_type] = {
            **counts, "part86_triage_broken": counts["broken"],
            "expected": expected,
            "share": counts["broken"] / counts["total"] if counts["total"] else None,
            "part86_triage_share": counts["broken"] / counts["total"] if counts["total"] else None,
            "three_word_repeat_share": (counts["three_word_repeat"] / counts["total"]
                                        if counts["total"] else None),
            "complete": counts["total"] == expected,
            "missing": len(missing[(row_type, dose)]),
        }
    # Keep empty/missing cells visible too.
    for row in planned:
        dose, row_type = rb.dose_text(row["dose"]), row["row_type"]
        if row_type not in output.setdefault(dose, {}):
            expected = sum(item["row_type"] == row_type and item["dose"] == row["dose"]
                           for item in planned)
            output[dose][row_type] = {
                "broken": 0, "part86_triage_broken": 0, "three_word_repeat": 0,
                "total": 0, "expected": expected, "share": None,
                "part86_triage_share": None, "three_word_repeat_share": None,
                "complete": False, "missing": expected}
    return output


def _answer_cell(broken: dict, row_type: str, dose: float) -> dict:
    return broken.get(rb.dose_text(dose), {}).get(row_type,
        {"broken": 0, "part86_triage_broken": 0, "three_word_repeat": 0,
         "total": 0, "expected": 0, "share": None, "part86_triage_share": None,
         "three_word_repeat_share": None, "complete": False, "missing": 0})


def _latest_label_by_id(labels: list[dict]) -> dict[str, dict]:
    latest = {}
    for row in labels:
        item_id = str(row.get("item_id", ""))
        if not item_id:
            continue
        old = latest.get(item_id, {})
        valid = type(row.get("coherent")) is bool and "error" not in row
        old_valid = type(old.get("coherent")) is bool and "error" not in old
        if item_id not in latest or valid or not old_valid:
            latest[item_id] = row
    return latest


def cohen_kappa_coherent(first: dict[str, dict], second: dict[str, dict]) -> dict:
    """Cohen kappa and raw agreement for the one field the design reads."""
    ids = sorted(set(first) & set(second))
    valid_ids = [item_id for item_id in ids
                 if type(first[item_id].get("coherent")) is bool
                 and type(second[item_id].get("coherent")) is bool]
    if not valid_ids:
        return {"kappa": None, "matched_valid": 0, "observed_agreement": None,
                "undefined_reason": "no valid paired labels"}
    first_values = [first[item_id]["coherent"] for item_id in valid_ids]
    second_values = [second[item_id]["coherent"] for item_id in valid_ids]
    observed = sum(a == b for a, b in zip(first_values, second_values)) / len(valid_ids)
    first_rate = sum(first_values) / len(valid_ids)
    second_rate = sum(second_values) / len(valid_ids)
    expected = first_rate * second_rate + (1 - first_rate) * (1 - second_rate)
    if expected >= 1:
        return {"kappa": None, "matched_valid": len(valid_ids),
                "observed_agreement": observed,
                "undefined_reason": "one reader gave one value throughout"}
    return {"kappa": float((observed - expected) / (1 - expected)),
            "matched_valid": len(valid_ids), "observed_agreement": observed,
            "undefined_reason": None}


def _prior_row_id(dose_row_id: str, row_type: str) -> str:
    prefix = f"{row_type}@{rb.dose_text(.4)}:"
    if not dose_row_id.startswith(prefix):
        raise ValueError(f"not a Part 119b 0.4 {row_type} row id: {dose_row_id!r}")
    return f"{row_type}:{dose_row_id[len(prefix):]}"


def _coherence_counts_by_class(key_rows: list[dict], labels: dict[str, dict]) -> dict:
    groups = defaultdict(lambda: {"coherent": 0, "incoherent": 0, "missing": 0,
                                  "expected": 0})
    for key_row in key_rows:
        name = str(key_row.get("sample_class", "unknown"))
        item_id = str(key_row["item_id"])
        row = labels.get(item_id, {})
        group = groups[name]
        group["expected"] += 1
        if type(row.get("coherent")) is not bool or "error" in row:
            group["missing"] += 1
        elif row["coherent"]:
            group["coherent"] += 1
        else:
            group["incoherent"] += 1
    output = {}
    for name, counts in groups.items():
        labeled = counts["coherent"] + counts["incoherent"]
        output[name] = {**counts, "labeled": labeled,
                        "coherent_share_of_expected": (counts["coherent"] / counts["expected"]
                                                        if counts["expected"] else None),
                        "coherent_share_of_labeled": counts["coherent"] / labeled if labeled else None}
    return output


def _paired_coherence_by_class(key_rows: list[dict], labels_04: dict[str, dict],
                               prior_keys: list[dict], labels_08: dict[str, dict]) -> dict:
    prior_item_by_row = {str(row["row_id"]): str(row["item_id"]) for row in prior_keys
                         if row.get("row_id") and row.get("item_id")}
    groups = defaultdict(lambda: {"expected": 0, "paired": 0, "missing_04": 0,
                                  "missing_08": 0, "coherent_04": 0,
                                  "coherent_08": 0, "same": 0})
    for key_row in key_rows:
        row_type = str(key_row.get("row_type", ""))
        sample_class = str(key_row.get("sample_class", "unknown"))
        source_id = _prior_row_id(str(key_row["row_id"]), row_type)
        new = labels_04.get(str(key_row["item_id"]), {})
        old_item_id = prior_item_by_row.get(source_id)
        old = labels_08.get(old_item_id, {}) if old_item_id is not None else {}
        group = groups[sample_class]
        group["expected"] += 1
        new_valid = type(new.get("coherent")) is bool and "error" not in new
        old_valid = type(old.get("coherent")) is bool and "error" not in old
        group["missing_04"] += int(not new_valid)
        group["missing_08"] += int(not old_valid)
        if not (new_valid and old_valid):
            continue
        group["paired"] += 1
        group["coherent_04"] += int(new["coherent"])
        group["coherent_08"] += int(old["coherent"])
        group["same"] += int(new["coherent"] is old["coherent"])
    output = {}
    for name, counts in groups.items():
        paired = counts["paired"]
        output[name] = {**counts,
                        "agreement": counts["same"] / paired if paired else None,
                        "coherent_share_04": counts["coherent_04"] / paired if paired else None,
                        "coherent_share_08": counts["coherent_08"] / paired if paired else None,
                        "share_difference_04_minus_08": (
                            (counts["coherent_04"] - counts["coherent_08"]) / paired
                            if paired else None)}
    return output


def reader_coherence_analysis(key_rows: list[dict], labels_ds: list[dict],
                              labels_luna: list[dict], prior_keys: list[dict],
                              prior_labels: list[dict], known_ds: list[dict],
                              known_luna: list[dict], *,
                              sample_ds_present: bool = True,
                              sample_luna_present: bool = True,
                              keys_present: bool = True,
                              known_ds_present: bool = True,
                              known_luna_present: bool = True) -> dict:
    """Score only coherent labels; absent/error rows remain unclassified."""
    key_by_id = keyed(key_rows, "item_id")
    ds_by_id = _latest_label_by_id(labels_ds)
    luna_by_id = _latest_label_by_id(labels_luna)
    prior_by_id = _latest_label_by_id(prior_labels)
    known = reader_shared.read_known()
    ds_gate = reader_shared.gate_result(known, known_ds)
    luna_gate = reader_shared.gate_result(known, known_luna)
    ds_gate["present"] = known_ds_present
    luna_gate["present"] = known_luna_present
    ds_gate["passed"] = bool(ds_gate["passed"] and known_ds_present)
    luna_gate["passed"] = bool(luna_gate["passed"] and known_luna_present)
    kappa = cohen_kappa_coherent(ds_by_id, luna_by_id)
    withheld_reasons = []
    unavailable_reasons = []
    if not ds_gate["passed"]:
        withheld_reasons.append("DeepSeek known-example gate did not pass 10 of 12")
    if not luna_gate["passed"]:
        withheld_reasons.append("Luna known-example gate did not pass 10 of 12")
    if kappa["kappa"] is not None and kappa["kappa"] < .6:
        withheld_reasons.append("defined DeepSeek/Luna kappa on coherent is below 0.6")
    if kappa["kappa"] is None and kappa["undefined_reason"] != "one reader gave one value throughout":
        unavailable_reasons.append("kappa is undefined without a constant reader's shared valid labels")
    if not sample_ds_present:
        unavailable_reasons.append("DeepSeek sample label file is missing")
    if not sample_luna_present:
        unavailable_reasons.append("Luna sample label file is missing")
    if not keys_present:
        unavailable_reasons.append("opaque reader key file is missing")
    read_count = not withheld_reasons and not unavailable_reasons

    counts = _coherence_counts_by_class(key_rows, ds_by_id)
    expected_pairs = sum(row.get("row_type") == "pair" for row in key_rows)
    pair_counts = {"coherent": 0, "incoherent": 0, "missing": 0,
                   "expected": expected_pairs, "labeled": 0}
    for row in key_rows:
        if row.get("row_type") != "pair":
            continue
        label = ds_by_id.get(str(row["item_id"]), {})
        if type(label.get("coherent")) is not bool or "error" in label:
            pair_counts["missing"] += 1
        elif label["coherent"]:
            pair_counts["coherent"] += 1
        else:
            pair_counts["incoherent"] += 1
    pair_counts["labeled"] = pair_counts["coherent"] + pair_counts["incoherent"]
    pair_counts["coherent_share_of_expected"] = (
        pair_counts["coherent"] / expected_pairs if expected_pairs else None)
    pair_counts["coherent_share_of_labeled"] = (
        pair_counts["coherent"] / pair_counts["labeled"] if pair_counts["labeled"] else None)
    paired = _paired_coherence_by_class(key_rows, ds_by_id, prior_keys, prior_by_id)
    return {
        "read": read_count,
        "withheld": bool(withheld_reasons),
        "withheld_reasons": withheld_reasons,
        "unavailable_reasons": unavailable_reasons,
        "known_ds": ds_gate,
        "known_luna": luna_gate,
        "sample_files_present": {"ds": sample_ds_present, "luna": sample_luna_present},
        "keys_present": keys_present,
        "cohen_kappa_on_coherent": kappa,
        "coherent_counts_0.4_by_class": counts,
        "real_pair_coherent_count_0.4": pair_counts,
        "paired_0.4_vs_part119_0.8_by_class": paired,
        "expected_rows": len(key_by_id),
        "rule": ("The reader's coherent count is withheld if a reader scores below 10 of 12 on the known examples, "
                 "or if the DeepSeek and Luna kappa on coherent is defined and below 0.6. If kappa is undefined "
                 "because a reader gave one value throughout, raw agreement is reported and the count is read."),
    }


def reader_119b1_clause(reader: dict) -> dict:
    count = reader["real_pair_coherent_count_0.4"]
    available = bool(reader.get("read") and count.get("expected") == 108)
    holds = count.get("coherent", 0) >= 0.60 * 108 if available else None
    return {"coherent": count.get("coherent", 0), "incoherent": count.get("incoherent", 0),
            "missing": count.get("missing", 0), "labeled": count.get("labeled", 0),
            "expected": count.get("expected", 0),
            "coherent_share_of_expected": count.get("coherent_share_of_expected"),
            "threshold": 0.60, "holds": holds, "read": available}


def _prediction(number: str, threshold: str, observed, available: bool,
                holds: bool | None, reason: str | None = None,
                unavailable_verdict: str = "not read") -> dict:
    if available:
        verdict = "held" if holds else "failed"
        final_reason = None
    else:
        verdict = unavailable_verdict
        final_reason = reason
    return {"id": number, "threshold": threshold, "observed": observed,
            "verdict": verdict, "reason": final_reason}


def prediction_119b1(broken_cell: dict, reader: dict, gate1_pass: bool) -> dict:
    """Read 119b.1 only when both its answer and reader clauses are available."""
    reader_clause = reader_119b1_clause(reader)
    expected_reader_pairs = reader_clause["expected"] == 108
    reader_clause_available = reader_clause["read"]
    triage_holds = (broken_cell["share"] is not None and broken_cell["share"] <= .10)
    observed = ({
        "part86_triage_clause": {"broken": broken_cell["broken"],
                                 "total": broken_cell["total"],
                                 "share": broken_cell["part86_triage_share"],
                                 "threshold": 0.10, "holds": triage_holds},
        "reader_coherence_clause": reader_clause,
    } if gate1_pass and reader_clause_available else None)
    available = (gate1_pass and broken_cell["complete"] and broken_cell["share"] is not None
                 and reader_clause_available)
    reason = ("Gate 1 did not pass or was not run" if not gate1_pass else
              "reader coherent count was withheld" if reader["withheld"] else
              "reader keys do not identify the registered 108 real pair sample"
              if not expected_reader_pairs else
              "0.4 real-pair answer rows are incomplete")
    return _prediction(
        "119b.1", "At 0.4, at most 10% of 741 real pair answers are broken and the reader marks at least 60% of the 108 sampled real pair answers coherent.",
        observed, available,
        bool(triage_holds and reader_clause["holds"]), reason,
        unavailable_verdict="withheld" if reader["withheld"] else "not read")


def _row_share_values(path_rows, support, kind: str, dose: float) -> list[float]:
    output = []
    for key, dose_rows in path_rows.items():
        if key[0] != kind:
            continue
        row = dose_rows.get(dose)
        if row is None:
            continue
        posmap = _position_map(row)
        values = [interaction_share(posmap[pos]) for pos in support.get(key, set())
                  if pos in posmap]
        values = [value for value in values if value is not None]
        if values:
            output.append(float(median(values)))
    return output


def _row_slope_values(path_rows, support, kind: str, dose: float) -> tuple[list[float], list[float], dict]:
    a_values, two_values, by_row = [], [], {}
    for key, dose_rows in path_rows.items():
        if key[0] != kind:
            continue
        row = dose_rows.get(dose)
        if row is None:
            continue
        posmap = _position_map(row)
        positions = [posmap[pos] for pos in support.get(key, set()) if pos in posmap]
        a = [float(p["b_A"]) for p in positions if p.get("b_A") is not None]
        two = [float(p["b_2"]) for p in positions if p.get("b_2") is not None]
        row_a, row_2 = _median_or_none(a), _median_or_none(two)
        if row_a is not None:
            a_values.append(row_a)
        if row_2 is not None:
            two_values.append(row_2)
        by_row[(key[1], key[2])] = {"b_A": row_a, "b_2": row_2,
                                    "positions": len(positions)}
    return a_values, two_values, by_row


def _unit_bootstrap_means(rows: list[dict], units: list[dict], draws: int = 2000) -> tuple[float | None, list[float | None]]:
    unit_ids = [unit["id"] for unit in units]
    values = [float(row["value"]) for row in rows if row.get("value") is not None]
    point = float(np.mean(values)) if values else None
    bootstrap_draws = score119.make_unit_bootstrap_draws(unit_ids, draws=draws,
                                                        seed=score119.BOOTSTRAP_SEED)
    samples = []
    for counts in bootstrap_draws:
        numerator = denominator = 0.0
        for row in rows:
            if row.get("value") is None:
                continue
            weight = int(counts.get(str(row["first"]), 0)) * int(counts.get(str(row["second"]), 0))
            numerator += float(row["value"]) * weight
            denominator += weight
        if denominator:
            samples.append(numerator / denominator)
    ci = ([float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]
          if samples else [None, None])
    return point, ci


def _unit_bootstrap_sum_weighted_ci(rows: list[dict], units: list[dict],
                                    draws: int = 2000) -> list[float | None]:
    """Sensitivity interval using each row's two unit draw counts added together."""
    unit_ids = [unit["id"] for unit in units]
    bootstrap_draws = score119.make_unit_bootstrap_draws(
        unit_ids, draws=draws, seed=score119.BOOTSTRAP_SEED)
    samples = []
    for counts in bootstrap_draws:
        numerator = denominator = 0.0
        for row in rows:
            if row.get("value") is None:
                continue
            weight = int(counts.get(str(row["first"]), 0)) + int(counts.get(str(row["second"]), 0))
            numerator += float(row["value"]) * weight
            denominator += weight
        if denominator:
            samples.append(numerator / denominator)
    return ([float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]
            if samples else [None, None])


def pair_top_agreement(interactions: list[dict], units: list[dict]) -> dict:
    index = path2_index(interactions)
    rows = []
    shared_positions = 0
    for key, dose_rows in index.items():
        if key[0] != "real" or .4 not in dose_rows or .8 not in dose_rows:
            continue
        at04, at08 = _position_map(dose_rows[.4]), _position_map(dose_rows[.8])
        common = set(at04) & set(at08)
        target_positions = []
        for pos in common:
            low, high = at04[pos], at08[pos]
            if int(low["top_first"]) == int(low["top_second"]):
                continue
            if int(high["top_first"]) == int(high["top_second"]):
                continue
            predicted = int(low["top_additive"]) == int(low["top_both"])
            plain = int(low["top_none"]) == int(low["top_both"])
            winner = score119.winner_take_all(
                low["top_first"], low["top_second"],
                low["margin_first"], low["margin_second"])
            wta = winner is not None and int(winner) == int(low["top_both"])
            add_high = int(high["top_additive"]) == int(high["top_both"])
            target_positions.append((predicted, plain, wta, add_high))
        if target_positions:
            shared_positions += len(target_positions)
            rows.append({"first": key[1], "second": key[2],
                         "add_04": float(np.mean([item[0] for item in target_positions])),
                         "plain_04": float(np.mean([item[1] for item in target_positions])),
                         "wta_04": float(np.mean([item[2] for item in target_positions])),
                         "add_08": float(np.mean([item[3] for item in target_positions])),
                         "positions": len(target_positions)})
    diffs = [{**row, "value": row["add_04"] - row["add_08"]} for row in rows]
    add_point, add_ci = _unit_bootstrap_means(diffs, units)
    add_sum_ci = _unit_bootstrap_sum_weighted_ci(diffs, units)
    improvements = [{**row, "value": row["add_04"] - row["wta_04"]} for row in rows]
    wta_point, wta_ci = _unit_bootstrap_means(improvements, units)
    wta_sum_ci = _unit_bootstrap_sum_weighted_ci(improvements, units)
    return {"rows_with_shared_contested_positions": len(rows),
            "shared_contested_positions": shared_positions,
            "additive_agreement_0.4": _weighted_row_mean(rows, "add_04"),
            "additive_agreement_0.8": _weighted_row_mean(rows, "add_08"),
            "plain_agreement_0.4": _weighted_row_mean(rows, "plain_04"),
            "winner_take_all_agreement_0.4": _weighted_row_mean(rows, "wta_04"),
            "additive_minus_0.8_difference": add_point,
            "additive_minus_0.8_ci95": add_ci,
            "additive_minus_0.8_ci95_sum_weighted": add_sum_ci,
            "additive_minus_winner_take_all_0.4": wta_point,
            "additive_minus_winner_take_all_ci95": wta_ci,
            "additive_minus_winner_take_all_ci95_sum_weighted": wta_sum_ci,
            "rows": rows}


def _weighted_row_mean(rows: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def effect_matched_at_04(interactions: list[dict], units: list[dict],
                         path_rows, support: dict, expected_real_rows: int = 1482) -> dict:
    records = []
    for key, dose_rows in path_rows.items():
        row = dose_rows.get(.4)
        if row is None:
            continue
        positions = _position_map(row)
        shared = support.get(key, set())
        records.append({"kind": key[0], "first": key[1], "second": key[2],
                        "positions": [{name: position.get(name)
                                       for name in ("pos", "nI", "nA", "n2", "nPair", "tv")}
                                      for pos, position in positions.items() if pos in shared]})
    draws = score119.make_unit_bootstrap_draws([unit["id"] for unit in units],
                                               draws=score119.BOOTSTRAP_DRAWS,
                                               seed=score119.BOOTSTRAP_SEED)
    result = score119.effect_matched_comparison(records, [unit["id"] for unit in units],
                                               draws, expected_real_rows)
    residual_rows = result.get("row_residuals", [])
    result["mean_residual_ci95_product_weighted"] = result.get("mean_residual_ci95")
    result["mean_residual_ci95_sum_weighted"] = _unit_bootstrap_sum_weighted_ci(
        residual_rows, units)
    all_real_positions = sum(len(row.get("positions", [])) for row in interactions
                             if row.get("kind") == "real" and math.isclose(float(row["dose"]), .4))
    kept = int(result.get("real_positions_kept", 0))
    result["real_positions_total_all_at_0.4"] = all_real_positions
    result["share_real_positions_kept_all_at_0.4"] = kept / all_real_positions if all_real_positions else None
    return result


def p9_read_status(path2_available: bool, effect_match: dict) -> dict:
    """Apply the registered 70% position-coverage clause before reading 119b.9."""
    coverage = effect_match.get("share_real_positions_kept_all_at_0.4")
    if not path2_available:
        return {"available": False, "verdict": "not read",
                "reason": "Path 2 gates or completeness stopped this prediction", "coverage": coverage}
    if coverage is None or coverage < .70:
        return {"available": False, "verdict": "withheld",
                "reason": "less than 70% of all real positions are kept", "coverage": coverage}
    ci = effect_match.get("mean_residual_ci95", [None, None])
    return {"available": bool(effect_match.get("fit_available") and len(ci) == 2
                               and ci[0] is not None and ci[1] is not None),
            "verdict": "not read" if not effect_match.get("fit_available") or len(ci) != 2
            or ci[0] is None or ci[1] is None else None,
            "reason": "effect-matched fit or interval unavailable" if not effect_match.get("fit_available")
            or len(ci) != 2 or ci[0] is None or ci[1] is None else None,
            "coverage": coverage}


def kind_gap_summary(path_rows, support, units: list[dict]) -> dict:
    output = {"agreed": {}, "haiku": {}}
    for scheme in ("agreed", "haiku"):
        for dose in (.4, .8):
            pairs = []
            if scheme == "agreed":
                slots = [u["id"] for u in units if u["group"] == "agreed_slot"]
                concepts = [u["id"] for u in units if u["group"] == "agreed_concept"]
            else:
                slots = [u["id"] for u in units if u["haiku_kind"] == "slot"]
                concepts = [u["id"] for u in units if u["haiku_kind"] == "concept"]
            for slot in slots:
                for concept in concepts:
                    forward = path_rows.get(("real", slot, concept), {}).get(dose)
                    reverse = path_rows.get(("real", concept, slot), {}).get(dose)
                    if forward is None or reverse is None:
                        gap_values = []
                    else:
                        f = _row_slope_for_key(path_rows, support, "real", slot, concept, dose)
                        r = _row_slope_for_key(path_rows, support, "real", concept, slot, dose)
                        # Forward order has slot in A and concept in 2; reverse order has
                        # concept in A and slot in 2. Each row contributes concept - slot.
                        gap_values = []
                        if f is not None:
                            gap_values.append(f["b_2"] - f["b_A"])
                        rev = _row_slope_for_key(path_rows, support, "real", concept, slot, dose)
                        if rev is not None:
                            gap_values.append(rev["b_A"] - rev["b_2"])
                    if gap_values:
                        pairs.append({"slot": slot, "concept": concept,
                                      "gap": float(np.mean(gap_values)),
                                      "ordered_rows": len(gap_values)})
                    else:
                        pairs.append({"slot": slot, "concept": concept, "gap": None,
                                      "ordered_rows": 0})
            values = [item["gap"] for item in pairs if item["gap"] is not None]
            output[scheme][rb.dose_text(dose)] = {
                "pairs_expected": len(slots) * len(concepts),
                "pairs_with_gap": len(values),
                "positive_pairs": sum(value > 0 for value in values),
                "positive_share_of_expected": (sum(value > 0 for value in values) /
                                                (len(slots) * len(concepts))
                                                if slots and concepts else None),
                "median_gap": _median_or_none(values),
                "median_absolute_gap": _median_or_none([abs(value) for value in values]),
                "pairs": pairs}
    output["directions_agree_by_dose"] = {}
    for dose in (.4, .8):
        med_agreed = output["agreed"][rb.dose_text(dose)]["median_gap"]
        med_haiku = output["haiku"][rb.dose_text(dose)]["median_gap"]
        output["directions_agree_by_dose"][rb.dose_text(dose)] = (
            _direction(med_agreed) * _direction(med_haiku) >= 0)
    output["directions_agree"] = all(output["directions_agree_by_dose"].values())
    output["direction_medians"] = {
        f"agreed_{dose}": output["agreed"][dose]["median_gap"]
        for dose in ("0.4", "0.8")}
    output["direction_medians"].update({
        f"haiku_{dose}": output["haiku"][dose]["median_gap"]
        for dose in ("0.4", "0.8")})
    return output


def _row_slope_for_key(path_rows, support, kind, first, second, dose):
    row = path_rows.get((kind, first, second), {}).get(dose)
    if row is None:
        return None
    posmap = _position_map(row)
    positions = [posmap[pos] for pos in support.get((kind, first, second), set()) if pos in posmap]
    a = [float(pos["b_A"]) for pos in positions if pos.get("b_A") is not None]
    two = [float(pos["b_2"]) for pos in positions if pos.get("b_2") is not None]
    if not a or not two:
        return None
    return {"b_A": float(median(a)), "b_2": float(median(two))}


def _direction(value: float | None) -> int:
    if value is None or value == 0:
        return 0
    return 1 if value > 0 else -1


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for offset in range(cursor, end):
            ranks[order[offset]] = rank
        cursor = end
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    rx, ry = np.asarray(_rankdata(x)), np.asarray(_rankdata(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def effect_size_by_unit(interactions: list[dict], units: list[dict], tournament: dict) -> dict:
    index = path2_index(interactions)
    n_a = defaultdict(list)
    for (kind, first, _second), dose_rows in index.items():
        if kind != "real" or .8 not in dose_rows:
            continue
        for position in dose_rows[.8].get("positions", []):
            n_a[first].append(float(position["nA"]))
    e_values = {unit["id"]: _median_or_none(n_a[unit["id"]]) for unit in units}
    dominance = {str(row["id"]): row.get("dominance") for row in tournament.get("units", [])}
    ids = [unit["id"] for unit in units if e_values[unit["id"]] is not None
           and dominance.get(unit["id"]) is not None]
    rho = _spearman([e_values[unit_id] for unit_id in ids],
                    [float(dominance[unit_id]) for unit_id in ids])
    return {"n_units": len(ids), "spearman_rho": rho,
            "units": [{"id": unit["id"], "E": e_values[unit["id"]],
                       "dominance": dominance.get(unit["id"])} for unit in units]}


def _complete_expected_path2(interactions: list[dict], units: list[dict]) -> dict:
    expected_base = {(row["kind"], row["first"], row["second"])
                     for row in rb.all_path2_rows(units)}
    found = {(str(row["kind"]), str(row["first"]), str(row["second"]), float(row["dose"]))
             for row in interactions}
    missing = [(kind, first, second, dose) for kind, first, second in expected_base
               for dose in DOSES if (kind, first, second, dose) not in found]
    extra = [(str(row["kind"]), str(row["first"]), str(row["second"]), float(row["dose"]))
             for row in interactions if (str(row["kind"]), str(row["first"]),
                                        str(row["second"])) not in expected_base
             or float(row["dose"]) not in DOSES]
    return {"complete": not missing and not extra,
            "expected_rows": len(expected_base) * len(DOSES),
            "actual_rows": len(found), "missing": missing[:25],
            "missing_count": len(missing), "extra": extra[:25], "extra_count": len(extra)}


def score_data(local_answers: list[dict], interactions: list[dict], path1: list[dict],
               gate1: dict | None, part119_answers: list[dict], part119_interactions: list[dict],
               part119_score: dict, tournament: dict, push_norms: dict,
               units: list[dict] | None = None, *,
               reader_key_rows: list[dict] | None = None,
               reader_labels_ds: list[dict] | None = None,
               reader_labels_luna: list[dict] | None = None,
               reader_known_ds: list[dict] | None = None,
               reader_known_luna: list[dict] | None = None,
               part119_reader_keys: list[dict] | None = None,
               part119_reader_labels: list[dict] | None = None,
               reader_files_present: dict[str, bool] | None = None) -> dict:
    units = rows119.load_units() if units is None else units
    planned = rb.all_answer_rows(units)
    answers, answer_coverage = compose_answers(local_answers, part119_answers, planned)
    answer_map = keyed(answers, "row_id")
    broken = _broken_stats(answers, planned)
    reader_files_present = reader_files_present or {}
    reader = reader_coherence_analysis(
        reader_key_rows or [], reader_labels_ds or [], reader_labels_luna or [],
        part119_reader_keys or [], part119_reader_labels or [],
        reader_known_ds or [], reader_known_luna or [],
        sample_ds_present=reader_files_present.get("ds", False),
        sample_luna_present=reader_files_present.get("luna", False),
        keys_present=reader_files_present.get("keys", False),
        known_ds_present=reader_files_present.get("known_ds", False),
        known_luna_present=reader_files_present.get("known_luna", False))
    expected_path_rows = rb.all_path2_rows(units)
    expected_by_kind = {
        "real": {("real", row["first"], row["second"]) for row in expected_path_rows
                 if row["kind"] == "real"},
        "control": {("control", row["first"], row["second"]) for row in expected_path_rows
                    if row["kind"] == "control"},
    }
    path_complete = _complete_expected_path2(interactions, units)
    path_support = validate_shared_support(interactions)

    gate1_read = gate1 is not None and "no_hook_tokens" in gate1 and "zero_hook_tokens" in gate1
    gate1_exact = (gate1.get("no_hook_tokens") == gate1.get("zero_hook_tokens")
                   if gate1_read else None)
    gate1_pass = bool(gate1 is not None and gate1.get("passed") is True and gate1_exact is True)
    gate1_result = {"passed": gate1_pass, "exact_token_match": gate1_exact,
        "read": gate1_read,
        "no_hook_tokens_present": gate1 is not None and "no_hook_tokens" in gate1,
        "reason": None if gate1_pass else ("gate1.json is missing; zero-hook gate was not run"
                                            if not gate1_read else
                                            "zero-hook output did not exactly match no-hook output")}
    if gate1_pass:
        gate2 = gate2_result(interactions, part119_interactions, expected_by_kind)
        gate2["read"] = True
    else:
        gate2 = {"passed": False, "read": False,
                 "reason": "Gate 1 did not pass or was not run; the Path 2 reproduction gate was not read"}
    core_passed = gate1_pass and gate2["passed"]

    summary = summarize_path2(interactions, expected_by_kind)
    gate3 = (gate3_result(summary, interactions, path_support) if core_passed else
             {"passed": False, "read": False,
              "reason": "Gate 1 or Gate 2 stopped Path 2 reading"})
    path_analysis = path2_index(interactions)
    answer_groups_complete = {}
    for row_type in ("pair", "control", "randpair", "single", "random"):
        for dose in sorted({row["dose"] for row in planned if row["row_type"] == row_type}):
            cell = _answer_cell(broken, row_type, dose)
            answer_groups_complete[(row_type, dose)] = cell["complete"]

    predictions = []
    cell = _answer_cell(broken, "pair", .4)
    real_pair_broken_share_04 = cell["share"]
    predictions.append(prediction_119b1(cell, reader, gate1_pass))
    cell = _answer_cell(broken, "randpair", .8)
    predictions.append(_prediction(
        "119b.2", "At least 20% of 741 R_A+R_B answers are broken at 0.8 + 0.8.",
        ({"broken": cell["broken"], "total": cell["total"], "share": cell["share"],
          "expected": cell["expected"]} if gate1_pass else None),
        gate1_pass and cell["complete"] and cell["share"] is not None,
        cell["share"] is not None and cell["share"] >= .20,
        "Gate 1 did not pass or was not run" if not gate1_pass else "0.8 random-pair answer rows are incomplete"))
    cell = _answer_cell(broken, "single", 1.13)
    predictions.append(_prediction(
        "119b.3", "At least 30% of 39 neuron single answers are broken at alpha 1.13.",
        ({"broken": cell["broken"], "total": cell["total"], "share": cell["share"],
          "expected": cell["expected"], "combined_orthogonal_0.8_norm": 0.8 * math.sqrt(2)}
         if gate1_pass else None),
        gate1_pass and cell["complete"] and cell["share"] is not None,
        cell["share"] is not None and cell["share"] >= .30,
        "Gate 1 did not pass or was not run" if not gate1_pass else "1.13 neuron-single answer rows are incomplete"))

    shares_04 = _row_share_values(path_analysis, _common_support(path_analysis), "real", .4)
    shares_02 = _row_share_values(path_analysis, _common_support(path_analysis), "real", .2)
    common_support = _common_support(path_analysis)
    real04 = _median_or_none(shares_04)
    real02 = _median_or_none(shares_02)
    path4_rows = summary["coverage"]["real"]["rows_present_by_dose"]["0.4"]
    path02_rows = summary["coverage"]["real"]["rows_present_by_dose"]["0.2"]
    path_metrics_available = (gate1_pass and gate2["passed"]
                              and path_complete["complete"] and path_support["passed"])
    gate3_stopped = core_passed and not gate3["passed"]
    predictions.append(_prediction(
        "119b.4", "Median row interaction share over real ordered pairs at 0.4 is at most 0.25.",
        ({"median_row_share_0.4_common_resolved": real04,
          "rows_with_common_resolved_positions": len(shares_04), "expected_rows": 1482}
         if path_metrics_available and not gate3_stopped else None),
        path_metrics_available and not gate3_stopped and path4_rows == 1482
        and real04 is not None,
        real04 is not None and real04 <= .25,
        _path_reason(gate1_pass, gate2, path_complete, gate3_stopped)))
    predictions.append(_prediction(
        "119b.5", "Median real row share at 0.2 is at most half the median at 0.4.",
        ({"median_row_share_0.2_common_resolved": real02,
          "median_row_share_0.4_common_resolved": real04,
          "rows_0.2": len(shares_02), "rows_0.4": len(shares_04)}
         if path_metrics_available and not gate3_stopped else None),
        path_metrics_available and not gate3_stopped and path02_rows == 1482
        and len(shares_04) == 1482 and real02 is not None and real04 is not None,
        real02 is not None and real04 is not None and real02 <= .5 * real04,
        _path_reason(gate1_pass, gate2, path_complete, gate3_stopped)))

    b_a, b_2, _ = _row_slope_values(path_analysis, common_support, "real", .4)
    median_b_a, median_b_2 = _median_or_none(b_a), _median_or_none(b_2)
    predictions.append(_prediction(
        "119b.6", "At 0.4, median row b_A and b_2 over 1,482 real ordered pairs are each 0.75–1.25.",
        ({"median_row_b_A": median_b_a, "median_row_b_2": median_b_2,
          "rows_with_b_A": len(b_a), "rows_with_b_2": len(b_2), "expected_rows": 1482}
         if path_metrics_available else None),
        path_metrics_available and median_b_a is not None and median_b_2 is not None,
        median_b_a is not None and median_b_2 is not None
        and .75 <= median_b_a <= 1.25 and .75 <= median_b_2 <= 1.25,
        _path_reason(gate1_pass, gate2, path_complete, False)))

    agreements = pair_top_agreement(interactions, units)
    diff_ci = agreements["additive_minus_0.8_ci95"]
    p7_available = (path_metrics_available and not gate3_stopped
                    and agreements["shared_contested_positions"] > 0
                    and diff_ci[0] is not None)
    p7_holds = (agreements["additive_minus_0.8_difference"] is not None
                and agreements["additive_minus_0.8_difference"] >= .20
                and diff_ci[0] is not None and diff_ci[0] > 0
                and agreements["additive_agreement_0.4"] is not None
                and agreements["plain_agreement_0.4"] is not None
                and agreements["additive_agreement_0.4"] -
                    agreements["plain_agreement_0.4"] >= .10)
    predictions.append(_prediction(
        "119b.7", "Additive pair-top agreement rises at least 20 points from 0.8 to 0.4, 95% unit-bootstrap CI > 0; 0.4 is at least 10 points above plain.",
        ({key: value for key, value in agreements.items() if key != "rows"}
         if path_metrics_available and not gate3_stopped else None),
        p7_available, p7_holds,
        _path_reason(gate1_pass, gate2, path_complete, gate3_stopped)))
    wta_ci = agreements["additive_minus_winner_take_all_ci95"]
    p8_available = (path_metrics_available and not gate3_stopped
                    and agreements["shared_contested_positions"] > 0
                    and wta_ci[0] is not None)
    p8_holds = (agreements["additive_minus_winner_take_all_0.4"] is not None
                and agreements["additive_minus_winner_take_all_0.4"] >= .05
                and wta_ci[0] is not None and wta_ci[0] > 0)
    predictions.append(_prediction(
        "119b.8", "At 0.4, additive agreement exceeds winner-take-all by at least 5 points, 95% unit-bootstrap CI > 0.",
        ({key: value for key, value in agreements.items() if key != "rows"}
         if path_metrics_available and not gate3_stopped else None),
        p8_available, p8_holds,
        _path_reason(gate1_pass, gate2, path_complete, gate3_stopped)))

    effect_match = effect_matched_at_04(interactions, units, path_analysis, common_support)
    coverage9 = effect_match.get("share_real_positions_kept_all_at_0.4")
    ci9 = effect_match.get("mean_residual_ci95", [None, None])
    bound = math.log(1.25)
    p9_status = p9_read_status(path_metrics_available, effect_match)
    p9_coverage = coverage9 is not None and coverage9 >= .70
    p9_ci_available = len(ci9) == 2 and ci9[0] is not None and ci9[1] is not None
    p9_available = p9_status["available"]
    p9_holds = p9_ci_available and ci9[0] >= -bound and ci9[1] <= bound
    predictions.append(_prediction(
        "119b.9", "Effect-matched mean real residual 95% unit-bootstrap interval is inside +/- log(1.25), with at least 70% real-position coverage.",
        ({"mean_residual": effect_match.get("mean_residual"), "ci95": ci9,
         "ci95_product_weighted": effect_match.get("mean_residual_ci95_product_weighted"),
         "ci95_sum_weighted": effect_match.get("mean_residual_ci95_sum_weighted"),
         "positions_kept": effect_match.get("real_positions_kept"),
         "positions_total": effect_match.get("real_positions_total_all_at_0.4"),
         "coverage": coverage9, "fit_available": effect_match.get("fit_available")}
         if path_metrics_available else None),
        p9_available, p9_holds,
        (p9_status["reason"] or "effect-matched interval is available"),
        unavailable_verdict=p9_status["verdict"] or "not read"))

    gaps = kind_gap_summary(path_analysis, common_support, units)
    gap_08 = gaps["agreed"]["0.8"]
    gap_04 = gaps["agreed"]["0.4"]
    med_abs_08, med_abs_04 = gap_08["median_absolute_gap"], gap_04["median_absolute_gap"]
    p10_available = (path_metrics_available and path_support["passed"]
                     and gaps["directions_agree"] and gap_08["pairs_expected"] == 168
                     and gap_08["pairs_with_gap"] == 168 and gap_04["pairs_with_gap"] == 168
                     and med_abs_08 is not None and med_abs_04 is not None)
    p10_holds = (gap_08["positive_pairs"] >= .60 * 168
                 and med_abs_04 <= .5 * med_abs_08)
    predictions.append(_prediction(
        "119b.10", "Positive kind gap in at least 60% of 168 agreed slot×concept pairs at 0.8; median absolute gap at 0.4 is at most half the 0.8 value.",
        ({"positive_pairs_0.8": gap_08["positive_pairs"],
         "expected_pairs": gap_08["pairs_expected"],
         "positive_share_of_expected": gap_08["positive_share_of_expected"],
         "median_absolute_gap_0.4": med_abs_04,
         "median_absolute_gap_0.8": med_abs_08,
         "agreed_haiku_directions_agree": gaps["directions_agree"],
         "median_gap_haiku_0.8": gaps["haiku"]["0.8"]["median_gap"]}
         if path_metrics_available and gaps["directions_agree"] else None),
        p10_available, p10_holds,
        ("Gate 1 did not pass or Gate 2 failed" if not core_passed else
         "agreed and Haiku kind-gap directions differ" if not gaps["directions_agree"] else
         "Path 2 kind-gap coverage or slopes are incomplete")))

    e_summary = effect_size_by_unit(interactions, units, tournament)
    p11_available = (path_metrics_available and e_summary["n_units"] == 39
                     and e_summary["spearman_rho"] is not None)
    rho = e_summary["spearman_rho"]
    predictions.append(_prediction(
        "119b.11", "Spearman correlation between Path 2 E and Part 119 dominance is at least 0.5 across 39 units.",
        ({"n_units": e_summary["n_units"], "spearman_rho": rho}
         if path_metrics_available else None),
        p11_available, rho is not None and rho >= .5,
        _path_reason(gate1_pass, gate2, path_complete, False)))

    same_diff = same_different_layer_summary(answers, planned, units)
    single_08 = {"neuron": _answer_cell(broken, "single", .8),
                 "random": _answer_cell(broken, "random", .8)}
    path1_summary = summarize_path1(path1, part119_score)
    path2_report = path_metrics_available
    effect_match_report = (effect_match if path2_report else {
        "read": False, "reason": _path_reason(gate1_pass, gate2, path_complete, False)})
    headline = headline_by_rule(predictions, real04, gate1_pass, gate2, gate3,
                                broken_share_04=real_pair_broken_share_04)
    extra_lines = []
    if (p9_coverage and p9_ci_available and ci9[0] > bound
            and next(row for row in predictions if row["id"] == "119b.9")["verdict"] == "failed"):
        extra_lines.append("At this dose, real pairs interact more than a neuron and a random direction with the same single effects.")
    if (next(row for row in predictions if row["id"] == "119b.2")["verdict"] == "held"
            and next(row for row in predictions if row["id"] == "119b.3")["verdict"] == "held"):
        same_layer_norm = push_norms.get("median_same_layer_combined_norm")
        extra_lines.append(
            "Breaking tracks how much is pushed more than which neurons are pushed. "
            f"The 0.8 random-single broken share is {_fmt(single_08['random']['share'])}, "
            f"the 0.8 neuron-single broken share is {_fmt(single_08['neuron']['share'])}; "
            f"the measured median combined norm of same-layer 0.8 pairs is {_fmt(same_layer_norm)} "
            f"over {push_norms.get('same_layer_pairs', 0)} pairs, and broken shares are "
            f"{_fmt(same_diff.get('same_layer', {}).get('share'))} same-layer vs "
            f"{_fmt(same_diff.get('different_layer', {}).get('share'))} different-layer.")

    return {"part": "119b", "stage": "main", "gates": {
                "gate1": gate1_result, "gate2": gate2, "gate3": gate3,
                "core_passed": core_passed},
            "coverage": {"answers": answer_coverage,
                         "all_path2": path_complete,
                         "shared_T": path_support},
            "predictions": predictions, "headline": headline,
            "additional_readings": extra_lines,
            "broken_answers_by_dose_and_class": broken,
            "reader_coherence": reader,
            "path2_summary": summary if path2_report else None,
            "path2_pair_top_agreement": ({key: value for key, value in agreements.items()
                                          if key != "rows"} if path2_report else None),
            "effect_matched_0.4": effect_match_report,
            "kind_gap": gaps if path2_report else {
                "read": False, "directions_agree": "not read",
                "agreed": {"0.4": {"positive_pairs": "not read", "pairs_expected": 168,
                                    "median_absolute_gap": None, "median_gap": None},
                           "0.8": {"positive_pairs": "not read", "pairs_expected": 168,
                                    "median_absolute_gap": None, "median_gap": None}},
                "haiku": {"0.8": {"median_gap": None}},
            },
            "effect_size_dominance": e_summary if path2_report else {
                "read": False, "spearman_rho": None, "n_units": 0},
            "same_different_layer_breaking": same_diff,
            "part119b_path1_description": path1_summary,
            "push_norms": {key: value for key, value in push_norms.items() if key != "rows"},
            "single_0.8_broken_shares": {key: value["share"] for key, value in single_08.items()},
            "per_dose_table": make_dose_table(broken, summary, path2_report)}


def _path_reason(gate1_pass, gate2, path_complete, gate3_stopped):
    if not gate1_pass:
        return "Gate 1 did not pass or was not run; predictions were not read"
    if not gate2.get("passed"):
        return "Gate 2 failed; Path 2 predictions were not read"
    if not path_complete.get("complete"):
        return "Path 2 rows are incomplete"
    if gate3_stopped:
        return "Gate 3 stopped this registered prediction"
    return "Path 2 support unavailable"


def same_different_layer_summary(answers, planned, units):
    by_id = {unit["id"]: unit for unit in units}
    answer_by_id = {row["row_id"]: row for row in answers}
    groups = {"same_layer": {"broken": 0, "total": 0},
              "different_layer": {"broken": 0, "total": 0}}
    for row in planned:
        if row["row_type"] != "pair" or not math.isclose(row["dose"], .8):
            continue
        a, b = (push["unit_id"] for push in row["pushes"])
        name = "same_layer" if by_id[a]["layer"] == by_id[b]["layer"] else "different_layer"
        answer = answer_by_id.get(row["row_id"])
        if answer is None:
            continue
        groups[name]["broken"] += int(score119.triage_answer(str(answer.get("text", ""))) == "broken")
        groups[name]["total"] += 1
    return {name: {**value, "share": value["broken"] / value["total"] if value["total"] else None}
            for name, value in groups.items()}


def summarize_path1(path1: list[dict], part119_score: dict) -> dict:
    row_rates = [score119.row_hit_rates(row) for row in path1]
    additive = [row["additive"] for row in row_rates if row["additive"] is not None]
    wta = [row["winner_take_all"] for row in row_rates if row["winner_take_all"] is not None]
    old = (part119_score.get("logit_summary", {}).get("real_pairs", {})
           if part119_score else {})
    return {"read": bool(path1), "rows": len(path1), "rows_with_contested_positions": len(additive),
            "contested_positions": sum(row["n_contested"] for row in row_rates),
            "additive_mean_hit_rate_0.4_own_path": float(np.mean(additive)) if additive else None,
            "winner_take_all_mean_hit_rate_0.4_own_path": float(np.mean(wta)) if wta else None,
            "part119_additive_mean_hit_rate_0.8_own_path": old.get("additive_mean"),
            "comparison_rule": "descriptive only; answers and contested positions differ from Part 119"}


def headline_by_rule(predictions, median_share_04, gate1_pass, gate2, gate3,
                     broken_share_04=None):
    by_id = {row["id"]: row for row in predictions}
    if not gate1_pass:
        return "Not read: Gate 1 did not pass or was not run."
    if not gate2.get("passed"):
        return "Not read for Path 2 because Gate 2 failed."
    p1 = by_id["119b.1"]
    observed_p1 = p1.get("observed") or {}
    triage_clause = observed_p1.get("part86_triage_clause", {})
    reader_clause = observed_p1.get("reader_coherence_clause", {})
    if (p1["verdict"] == "held"
            and triage_clause.get("holds") is True
            and reader_clause.get("read") is True
            and reader_clause.get("holds") is True
            and by_id["119b.4"]["verdict"] == "held"
            and by_id["119b.6"]["verdict"] == "held"):
        return ("At 0.4, on prefixes from each neuron's own 0.8 answer, the two pushes' effects on next-word scores "
                "mostly add, both pushes contribute, and the pair's own answers stay intact; Part 119's failure "
                "to add does not extend to this dose.")
    if (gate3.get("passed") and by_id["119b.4"]["verdict"] in ("held", "failed")
            and median_share_04 is not None and median_share_04 >= .4):
        return (f"The failure to add extends to 0.4; the 0.4 real-pair broken share was "
                f"{_fmt(broken_share_04)}.")
    return "No registered headline reading; report the numbers."


def make_dose_table(broken: dict, summary: dict, path2_read_allowed: bool = True) -> list[dict]:
    rows = []
    for dose in DOSES:
        dose_key = rb.dose_text(dose)
        cell = summary["by_dose"].get(dose_key, {}) if path2_read_allowed else {}
        real, control = cell.get("real", {}), cell.get("control", {})
        rows.append({"dose": dose,
                     "broken_share_by_row_class": {
                         "real_pair": _answer_cell(broken, "pair", dose)["share"],
                         "A_plus_random": _answer_cell(broken, "control", dose)["share"],
                         "random_pair": _answer_cell(broken, "randpair", dose)["share"],
                         "neuron_single": _answer_cell(broken, "single", dose)["share"],
                         "random_single": _answer_cell(broken, "random", dose)["share"]},
                     "three_word_repeat_share_by_row_class": {
                         "real_pair": _answer_cell(broken, "pair", dose)["three_word_repeat_share"],
                         "A_plus_random": _answer_cell(broken, "control", dose)["three_word_repeat_share"],
                         "random_pair": _answer_cell(broken, "randpair", dose)["three_word_repeat_share"],
                         "neuron_single": _answer_cell(broken, "single", dose)["three_word_repeat_share"],
                         "random_single": _answer_cell(broken, "random", dose)["three_word_repeat_share"]},
                     "median_share_all_positions": {
                         "real": real.get("median_row_share_all_positions"),
                         "control": control.get("median_row_share_all_positions")},
                     "median_share_resolved_positions": {
                         "real": real.get("median_row_share_resolved_at_dose"),
                         "control": control.get("median_row_share_resolved_at_dose")},
                     "median_b_A": {"real": real.get("median_row_b_A_common_resolved"),
                                    "control": control.get("median_row_b_A_common_resolved")},
                     "median_b_2": {"real": real.get("median_row_b_2_common_resolved"),
                                    "control": control.get("median_row_b_2_common_resolved")},
                     "coverage": {"real_rows": real.get("row_coverage"),
                                  "real_positions": real.get("position_coverage"),
                                  "control_rows": control.get("row_coverage"),
                                  "control_positions": control.get("position_coverage"),
                                  "all_four_real_rows": real.get("common_row_coverage"),
                                  "all_four_control_rows": control.get("common_row_coverage")}})
    return rows


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(result: dict) -> str:
    gates = result["gates"]
    lines = ["# Part 119b score", "",
             "Scores are generated by `part119b/score119b.py`; Part 119's 0.8 answers and interaction rows are read-only inputs.", "",
             "## Gates", "",
             "| Gate | Verdict | Observed |", "| --- | --- | --- |",
             f"| 1 exact zero push | {'pass' if gates['gate1']['passed'] else 'fail' if gates['gate1'].get('read') else 'not run'} | "
             f"exact token match={gates['gate1']['exact_token_match'] if gates['gate1']['read'] else 'not run'} |",
             f"| 2 Part 119 Path 2 reproduction | {'pass' if gates['gate2']['passed'] else 'not read' if not gates['gate2'].get('read') else 'fail'} | "
             f"complete={gates['gate2'].get('complete')}; real/control medians="
             f"{_fmt(gates['gate2'].get('row_medians', {}).get('real', {}).get('median_row_share_0.8'))}/"
             f"{_fmt(gates['gate2'].get('row_medians', {}).get('control', {}).get('median_row_share_0.8'))}; "
             f"median relative nPair error={_fmt(gates['gate2'].get('median_relative_nPair_difference'))} |",
             f"| 3 low-dose signal and coverage | {'pass' if gates['gate3']['passed'] else 'not read' if not gates['gate3'].get('read') else 'fail'} | "
             f"rows kept={gates['gate3'].get('rows_kept')}/{gates['gate3'].get('rows_expected')}; "
             f"common row coverage={_fmt(gates['gate3'].get('row_coverage_all_four_doses'))}; "
             f"0.2 median share={_fmt(gates['gate3'].get('median_row_share_0.2_common_resolved'))} |",
             "", "Intervals for 119b.7, 119b.8, and 119b.9 show the registered product of the two unit draw counts and the sum-weighted sensitivity interval.",
             "", "## Registered predictions", "",
             "| ID | Threshold | Observed | Verdict | Reason |",
             "| --- | --- | --- | --- | --- |"]
    for item in result["predictions"]:
        observed = json.dumps(item["observed"], ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"))
        lines.append(f"| {item['id']} | {item['threshold']} | `{observed}` | {item['verdict']} | {item.get('reason') or '—'} |")
    lines.extend(["", "## Headline", "", result["headline"]])
    if result["additional_readings"]:
        lines.extend(["", "Additional registered lines:"])
        lines.extend(f"- {line}" for line in result["additional_readings"])
    reader = result["reader_coherence"]
    kappa = reader["cohen_kappa_on_coherent"]
    lines.extend(["", "## Blind reader", "",
                  f"Reader coherent count read: {reader['read']}; withheld: {reader['withheld']}; "
                  f"DeepSeek known gate={reader['known_ds']['correct']}/{reader['known_ds']['total']} "
                  f"(pass={reader['known_ds']['passed']}); Luna known gate="
                  f"{reader['known_luna']['correct']}/{reader['known_luna']['total']} "
                  f"(pass={reader['known_luna']['passed']}).",
                  f"DeepSeek/Luna kappa on coherent={_fmt(kappa['kappa'])}; raw agreement="
                  f"{_fmt(kappa['observed_agreement'])}; paired valid={kappa['matched_valid']}; "
                  f"undefined reason={kappa['undefined_reason'] or '—'}.",
                  reader["rule"],
                  f"At 0.4, real-pair coherent={reader['real_pair_coherent_count_0.4']['coherent']}/"
                  f"{reader['real_pair_coherent_count_0.4']['expected']} expected; incoherent="
                  f"{reader['real_pair_coherent_count_0.4']['incoherent']}; missing="
                  f"{reader['real_pair_coherent_count_0.4']['missing']}.",
                  "The paired comparison below joins the same row at 0.4 and Part 119's 0.8 labels; missing labels are excluded from paired agreement.", "",
                  "| Sample class | Paired valid | Coherent 0.4 | Coherent 0.8 | Raw agreement | Missing 0.4 / 0.8 |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for name, item in reader["paired_0.4_vs_part119_0.8_by_class"].items():
        lines.append(f"| {name} | {item['paired']} | {item['coherent_04']} | {item['coherent_08']} | "
                     f"{_fmt(item['agreement'])} | {item['missing_04']} / {item['missing_08']} |")
    if reader["withheld_reasons"]:
        lines.extend(["", "Withheld because: " + "; ".join(reader["withheld_reasons"]) + "."])
    if reader["unavailable_reasons"]:
        lines.extend(["", "Not read because: " + "; ".join(reader["unavailable_reasons"]) + "."])
    lines.extend(["", "## Per-dose answer breaking", "",
                  "Part 86's `triage86.py` rule is reported as written, beside the three-word-repeat condition alone. Missing answers are not included in either count.", "",
                  "Shares for the resolved column use positions resolved at that dose. Path 2 interaction-share predictions use positions resolved at all four doses; the all-position column is descriptive.", "",
                  "| Dose | Part 86 triage broken: real / A+R / random pair / neuron single / random single | Three-word repeat alone: real / A+R / random pair / neuron single / random single | Median share all positions (real/control) | Median share resolved (real/control) | Median b_A (real/control) | Median b_2 (real/control) | Coverage rows (real/control), positions (real/control) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for row in result["per_dose_table"]:
        b = row["broken_share_by_row_class"]
        repeats = row["three_word_repeat_share_by_row_class"]
        all_s, res_s, ba, b2, cov = (row[name] for name in
                                     ("median_share_all_positions", "median_share_resolved_positions",
                                      "median_b_A", "median_b_2", "coverage"))
        triage_values = "/".join(_fmt(b[name]) for name in
                                  ("real_pair", "A_plus_random", "random_pair", "neuron_single", "random_single"))
        repeat_values = "/".join(_fmt(repeats[name]) for name in
                                  ("real_pair", "A_plus_random", "random_pair", "neuron_single", "random_single"))
        lines.append(
            f"| {rb.dose_text(row['dose'])} | {triage_values} | {repeat_values} | "
            f"{_fmt(all_s['real'])}/{_fmt(all_s['control'])} | {_fmt(res_s['real'])}/{_fmt(res_s['control'])} | "
            f"{_fmt(ba['real'])}/{_fmt(ba['control'])} | {_fmt(b2['real'])}/{_fmt(b2['control'])} | "
            f"{_fmt(cov['real_rows'])}/{_fmt(cov['control_rows'])}, "
            f"{_fmt(cov['real_positions'])}/{_fmt(cov['control_positions'])} |")
    if not result["part119b_path1_description"].get("read"):
        lines.extend(["", "## Path 1 description", "", "Part 119b Path 1 at 0.4 was not run.",
                      f"Part 119's 0.8 own-answer additive hit rate: "
                      f"{_fmt(result['part119b_path1_description']['part119_additive_mean_hit_rate_0.8_own_path'])}."])
    else:
        lines.extend(["", "## Path 1 description", "",
                  f"Rows with contested positions: {result['part119b_path1_description']['rows_with_contested_positions']}/741; "
                  f"contested positions: {result['part119b_path1_description']['contested_positions']}; "
                  f"0.4 own-answer additive hit rate: {_fmt(result['part119b_path1_description']['additive_mean_hit_rate_0.4_own_path'])}; "
                  f"Part 119's 0.8 own-answer additive hit rate: "
                  f"{_fmt(result['part119b_path1_description']['part119_additive_mean_hit_rate_0.8_own_path'])}. "
                  "This is descriptive: the answer paths and contested positions differ."])
    lines.extend([
                  "", "## Shared support and coverage", "",
                  f"Shared token-set integrity: {result['coverage']['shared_T']['passed']}; "
                  f"positions checked: {result['coverage']['shared_T']['positions_checked']}; "
                  f"problems: {result['coverage']['shared_T']['problem_count']}.", "",
                  "Path 2 uses the row-specific union of the four 0.8 top-10 lists at every dose. Resolved positions require nA, n2, and nPair each to be at least 1.0. Coverage by dose and row class is recorded in score119b.json.",
                  "", "## Effect-matched read", "",
                  f"At 0.4, real positions kept: {result['effect_matched_0.4'].get('real_positions_kept')}/"
                  f"{result['effect_matched_0.4'].get('real_positions_total_all_at_0.4')} "
                  f"({_fmt(result['effect_matched_0.4'].get('share_real_positions_kept_all_at_0.4'))}); "
                  f"mean residual={_fmt(result['effect_matched_0.4'].get('mean_residual'))}; "
                  f"95% product-weight interval={result['effect_matched_0.4'].get('mean_residual_ci95_product_weighted')}; "
                  f"sum-weight interval={result['effect_matched_0.4'].get('mean_residual_ci95_sum_weighted')}.",
                  "", "## Kind gap and effect size", "",
                  f"119b.10 agreed/Haiku directions agree: {result['kind_gap']['directions_agree']}; "
                  f"agreed 0.8 positive pairs: {result['kind_gap']['agreed']['0.8']['positive_pairs']}/"
                  f"{result['kind_gap']['agreed']['0.8']['pairs_expected']}; "
                  f"median absolute gap 0.4/0.8: "
                  f"{_fmt(result['kind_gap']['agreed']['0.4']['median_absolute_gap'])}/"
                  f"{_fmt(result['kind_gap']['agreed']['0.8']['median_absolute_gap'])}.",
                  "", "Effect size E is the median nA over real ordered rows on each unit's own 0.8 path. "
                  f"E/dominance Spearman rho={_fmt(result['effect_size_dominance']['spearman_rho'])} "
                  f"(n={result['effect_size_dominance']['n_units']}).",
                  "", "## Answer breaking controls", "",
                  f"Same-layer real pair broken share at 0.8: {_fmt(result['same_different_layer_breaking']['same_layer']['share'])} "
                  f"({result['same_different_layer_breaking']['same_layer']['broken']}/"
                  f"{result['same_different_layer_breaking']['same_layer']['total']}); different-layer: "
                  f"{_fmt(result['same_different_layer_breaking']['different_layer']['share'])} "
                  f"({result['same_different_layer_breaking']['different_layer']['broken']}/"
                  f"{result['same_different_layer_breaking']['different_layer']['total']}).",
                  f"Median measured combined norm for same-layer 0.8 pairs: "
                  f"{_fmt(result['push_norms'].get('median_same_layer_combined_norm'))} "
                  f"over {result['push_norms'].get('same_layer_pairs', 0)} pairs.",
                  "", "## Scope", "",
                  "One prompt, one OLMo checkpoint, greedy decoding, and four shared doses. Path 2 is on prefixes from the single unit's selected 0.8 answer; it measures logits on those prefixes, not whether lower doses generate the same persona. T is selected at 0.8 and logits are rounded to five decimals. The 1.13 and 1.6 single rows are size comparisons, not direct two-push norm matches for cross-layer pairs.", ""])
    return "\n".join(lines)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def score_files(args: argparse.Namespace) -> dict:
    units = rows119.load_units(args.neurons, args.kinds, args.kinds_ds)
    local_answers = read_jsonl(args.answers, optional=True)
    interactions = read_jsonl(args.interactions, optional=True)
    path1 = read_jsonl(args.path1, optional=True)
    gate1 = read_json(args.gate1, optional=True)
    part119_answers = read_jsonl(args.part119_answers, optional=True)
    part119_interactions = read_jsonl(args.part119_interactions, optional=True)
    part119_score = read_json(args.part119_score, optional=True) or {}
    tournament = read_json(args.part119_tournament, optional=True) or {}
    push_norms = read_json(args.push_norms, optional=True) or {}
    reader_key_path = repo_path(args.reader_keys)
    reader_ds_path = repo_path(args.reader_ds)
    reader_luna_path = repo_path(args.reader_luna)
    reader_known_ds_path = repo_path(args.reader_known_ds)
    reader_known_luna_path = repo_path(args.reader_known_luna)
    part119_reader_keys_path = repo_path(args.part119_reader_keys)
    part119_reader_labels_path = repo_path(args.part119_reader_labels)
    result = score_data(local_answers, interactions, path1, gate1,
                        part119_answers, part119_interactions, part119_score,
                        tournament, push_norms, units,
                        reader_key_rows=read_jsonl(reader_key_path, optional=True),
                        reader_labels_ds=read_jsonl(reader_ds_path, optional=True),
                        reader_labels_luna=read_jsonl(reader_luna_path, optional=True),
                        reader_known_ds=read_jsonl(reader_known_ds_path, optional=True),
                        reader_known_luna=read_jsonl(reader_known_luna_path, optional=True),
                        part119_reader_keys=read_jsonl(part119_reader_keys_path, optional=True),
                        part119_reader_labels=read_jsonl(part119_reader_labels_path, optional=True),
                        reader_files_present={
                            "keys": reader_key_path.is_file(),
                            "ds": reader_ds_path.is_file(),
                            "luna": reader_luna_path.is_file(),
                            "known_ds": reader_known_ds_path.is_file(),
                            "known_luna": reader_known_luna_path.is_file()})
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(result)
    (output_dir / "score119b.json").write_text(
        json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "SCORE.md").write_text(render_markdown(safe), encoding="utf-8")
    print(json.dumps({"headline": safe["headline"],
                      "gates": {key: value.get("passed") for key, value in safe["gates"].items()
                                if isinstance(value, dict)},
                      "predictions": [{"id": item["id"], "verdict": item["verdict"]}
                                      for item in safe["predictions"]],
                      "output": str(output_dir / "SCORE.md")}, ensure_ascii=False), flush=True)
    return safe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="results/part119b/answers.jsonl")
    parser.add_argument("--interactions", default="results/part119b/interaction.jsonl")
    parser.add_argument("--path1", default="results/part119b/path1.jsonl")
    parser.add_argument("--gate1", default="results/part119b/gate1.json")
    parser.add_argument("--push-norms", default="results/part119b/push_norms.json")
    parser.add_argument("--part119-answers", default="results/part119/answers.jsonl")
    parser.add_argument("--part119-interactions", default="results/part119/interaction.jsonl")
    parser.add_argument("--part119-score", default="results/part119/score119.json")
    parser.add_argument("--part119-tournament", default="results/part119/tournament.json")
    parser.add_argument("--reader-keys", default="results/part119b/labels-ds.jsonl.keys.jsonl")
    parser.add_argument("--reader-ds", default="results/part119b/labels-ds.jsonl")
    parser.add_argument("--reader-luna", default="results/part119b/labels-luna.jsonl")
    parser.add_argument("--reader-known-ds", default="results/part119b/known-ds.jsonl")
    parser.add_argument("--reader-known-luna", default="results/part119b/known-luna.jsonl")
    parser.add_argument("--part119-reader-keys", default="results/part119/labels-ds.jsonl.keys.jsonl")
    parser.add_argument("--part119-reader-labels", default="results/part119/labels-ds.jsonl")
    parser.add_argument("--output-dir", default="results/part119b")
    parser.add_argument("--neurons", default=rows119.DEFAULT_NEURONS)
    parser.add_argument("--kinds", default=rows119.DEFAULT_HAIKU_KINDS)
    parser.add_argument("--kinds-ds", default=rows119.DEFAULT_DS_KINDS)
    args = parser.parse_args(argv)
    score_files(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
