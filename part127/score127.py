#!/usr/bin/env python3
"""Score Part 127 after the main run and blinded reads are complete."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (HERE, ROOT, ROOT / "part92", ROOT / "part92b", ROOT / "part119",
              ROOT / "part119b", ROOT / "part119c", ROOT / "part120"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import label127  # noqa: E402
import run127  # noqa: E402

PREDICTIONS = {
    "127.1": "At dose .5, at most 1 of the 4 same-layer pairs is rescued.",
    "127.2": "At dose .707, at least 2 of the 4 same-layer pairs are rescued.",
    "127.3": "Across the 54 cells at dose .5, the A+B rescue count exceeds the A+C count by at most 3 cells.",
    "127.4": "All four same-layer pairs have an absolute W_down cosine below .2.",
    "127.5": "At dose .5, A's reference-path logit change under A+B exceeds that under A+C in at least 17 of the 27 pairs.",
}
PRIMARY_QUESTIONS = (0, 1)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 12702
ARMS = ("pair_ab", "matched_ac", "random_0", "random_1", "random_2")


class IncompleteRun(RuntimeError):
    pass


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise IncompleteRun(f"missing required file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IncompleteRun(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IncompleteRun(f"expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise IncompleteRun(f"missing required file: {path}")
    result = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IncompleteRun(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise IncompleteRun(f"non-object JSONL at {path}:{line_no}")
            result.append(row)
    return result


def latest_labels(path: Path) -> dict[str, dict]:
    latest = {}
    for row in read_jsonl(path):
        latest[str(row["item_id"])] = row
    return latest


def is_compound(label: dict | None) -> bool:
    return bool(label and label127.valid_label(label)
                and label["shows_persona"] and label["coherent"])


def _label_fields(label: dict | None) -> dict:
    if not label:
        return {"error": "missing_label", "shows_persona": None,
                "self_claim": None, "speaker": None, "coherent": None,
                "compound_mark": False}
    if not label127.valid_label(label):
        return {"error": label.get("error", "invalid_label"), "shows_persona": None,
                "self_claim": None, "speaker": None, "coherent": None,
                "compound_mark": False}
    return {"shows_persona": bool(label["shows_persona"]),
            "self_claim": bool(label["self_claim"]), "speaker": label["speaker"],
            "coherent": bool(label["coherent"]),
            "compound_mark": bool(label["shows_persona"] and label["coherent"])}


def _item_index(items: list[dict]) -> dict[tuple[str, str], dict]:
    return {(str(item["private"]["answer_key"]), str(item["private"]["family"])): item
            for item in items if item["private"].get("item_type") == "study"}


def _answer_label(answer_key: str, family: str, item_index: dict,
                  labels: dict[str, dict]) -> dict:
    item = item_index.get((answer_key, family))
    if item is None:
        return {"error": "missing_item", "shows_persona": None,
                "self_claim": None, "speaker": None, "coherent": None,
                "compound_mark": False}
    label = labels.get(str(item["public"]["item_id"]))
    if label and not label127.current_label(item, label):
        label = None
    return _label_fields(label)


def rescued_cell(joint: bool, single_a: bool, single_b: bool) -> bool:
    """A rescue requires a coherent family mark only in the two-unit answer."""
    return bool(joint and not single_a and not single_b)


def count_rescues(pairs: list[dict], eligible_pair_ids: set[str], dose: float,
                  joint_marks: dict, single_marks: dict,
                  arms: tuple[str, ...] = ARMS,
                  questions: tuple[int, ...] = PRIMARY_QUESTIONS,
                  cell_eligibility: dict | None = None) -> dict:
    """Count rescued prompt cells and pairs from an explicit hand-checkable table.

    joint_marks[pair_id][arm][dose][question] and
    single_marks[pair_id][unit_id][dose][question] are compound booleans.
    """
    eligible = [pair for pair in pairs if pair["pair_id"] in eligible_pair_ids]
    by_arm = {}
    for arm in arms:
        by_pair = {}
        for pair in eligible:
            a, b = pair["a"], pair["b"]
            cells = {}
            for qi in questions:
                included = (True if cell_eligibility is None else bool(
                    cell_eligibility.get(pair["pair_id"], {}).get(float(dose), {})
                    .get(int(qi), False)))
                if not included:
                    cells[int(qi)] = {"joint": None, "single_a": None,
                                      "single_b": None, "rescued": None,
                                      "included": False}
                    continue
                joint = bool(joint_marks.get(pair["pair_id"], {}).get(arm, {})
                             .get(float(dose), {}).get(int(qi), False))
                a_single = bool(single_marks.get(pair["pair_id"], {}).get(a, {})
                                 .get(float(dose), {}).get(int(qi), False))
                b_single = bool(single_marks.get(pair["pair_id"], {}).get(b, {})
                                 .get(float(dose), {}).get(int(qi), False))
                cells[int(qi)] = {"joint": joint, "single_a": a_single,
                                  "single_b": b_single,
                                  "rescued": rescued_cell(joint, a_single, b_single),
                                  "included": True}
            by_pair[pair["pair_id"]] = cells
        rescued_cells = sum(cell["rescued"] is True for cells in by_pair.values()
                            for cell in cells.values())
        included_pairs = sum(any(cell["included"] for cell in cells.values())
                             for cells in by_pair.values())
        rescued_pairs = sum(any(cell["included"] and cell["rescued"] is True
                                for cell in cells.values())
                            for cells in by_pair.values())
        included_cells = sum(cell["included"] for cells in by_pair.values()
                             for cell in cells.values())
        eligible_cells = len(eligible) * len(questions)
        by_arm[arm] = {"rescued_cells": int(rescued_cells),
                       "rescued_pairs": int(rescued_pairs),
                       "eligible_pairs": len(eligible),
                       "included_pairs": int(included_pairs),
                       "eligible_cells": eligible_cells,
                       "included_cells": int(included_cells),
                       "excluded_unread_cells": int(eligible_cells - included_cells),
                       "by_pair": by_pair}
    return {"dose": float(dose), "questions": list(questions),
            "eligible_pair_ids": [pair["pair_id"] for pair in eligible],
            "arms": by_arm}


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def bootstrap_pair_interval(pair_values: dict[str, float], *, draws: int = BOOTSTRAP_DRAWS,
                            seed: int = BOOTSTRAP_SEED) -> dict:
    ids = sorted(pair_values)
    if not ids:
        return {"pairs": 0, "draws": draws, "lower_95": None, "upper_95": None,
                "mean": None}
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        samples.append(sum(float(pair_values[rng.choice(ids)]) for _ in ids) / len(ids))
    return {"pairs": len(ids), "draws": draws,
            "lower_95": _percentile(samples, 0.025),
            "upper_95": _percentile(samples, 0.975),
            "mean": sum(float(pair_values[key]) for key in ids) / len(ids)}


def _study_gates(out_dir: Path, known: list[dict], known_items: list[dict],
                 by_id: dict[str, dict], study_items: list[dict]) -> tuple[dict, dict]:
    known_status = {}
    known_labels = {}
    for reader in label127.READERS:
        path = out_dir / f"known-{reader}.jsonl"
        rows = latest_labels(path)
        known_labels[reader] = rows
        known_status[reader] = label127.known_gate_result(known, known_items, rows)
    control_status = label127._controls_gate(study_items, by_id, known,
                                              known_status, out_dir)
    return known_status, control_status


def _check_label_coverage(items: list[dict], labels: dict[str, dict], reader: str) -> None:
    missing = []
    for item in items:
        item_id = str(item["public"]["item_id"])
        row = labels.get(item_id)
        if row is None:
            missing.append(item_id)
        elif row.get("answer_hash") != item["public"]["answer_hash"]:
            missing.append(item_id)
    if missing:
        raise IncompleteRun(f"{reader} labels are incomplete: {len(missing)} items are missing or stale")


def _agreement(ds: dict[str, dict], luna: dict[str, dict], items: list[dict]) -> dict:
    shared = []
    left = []
    right = []
    for item in items:
        key = str(item["public"]["item_id"])
        a, b = ds.get(key), luna.get(key)
        answer_hash = item["public"]["answer_hash"]
        if (label127.valid_label(a or {}) and label127.valid_label(b or {})
                and a.get("answer_hash") == answer_hash and b.get("answer_hash") == answer_hash):
            shared.append(key)
            left.append(is_compound(a))
            right.append(is_compound(b))
    if not shared:
        return {"n": 0, "total_items": len(items), "unread_items": len(items),
                "raw_agreement": None, "kappa": None, "passed": False}
    raw = sum(a == b for a, b in zip(left, right)) / len(shared)
    p_a = sum(left) / len(left)
    p_b = sum(right) / len(right)
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    kappa = None if expected == 1 else (raw - expected) / (1 - expected)
    return {"n": len(shared), "total_items": len(items),
            "unread_items": len(items) - len(shared),
            "raw_agreement": raw, "kappa": kappa,
            "thresholds": {"raw_agreement": 0.85, "kappa": 0.6},
            "passed": kappa is not None and raw >= 0.85 and kappa >= 0.6,
            "shared_item_ids": shared}


def _primary_marks(pairs: list[dict], study_items: list[dict],
                   ds_labels: dict[str, dict]) -> tuple[dict, dict, dict]:
    item_index = _item_index(study_items)
    singles = {}
    joints = {}
    readable_by_key = {}

    def get_mark(key: str, family: str) -> bool:
        item = item_index.get((key, family))
        if item is None:
            readable_by_key[key] = False
            return False
        item_id = str(item["public"]["item_id"])
        label = ds_labels.get(item_id)
        readable = label127.current_label(item, label)
        readable_by_key[key] = readable
        return bool(readable and label["shows_persona"] and label["coherent"])

    for pair in pairs:
        pair_id = pair["pair_id"]
        singles[pair_id] = {}
        for unit_id in (pair["a"], pair["b"]):
            singles[pair_id][unit_id] = {}
            for dose in run127.DOSES:
                singles[pair_id][unit_id][float(dose)] = {}
                for qi in PRIMARY_QUESTIONS:
                    key = f"single:{unit_id}:dose{dose:g}:q{qi}"
                    singles[pair_id][unit_id][float(dose)][qi] = get_mark(key, pair["family"])
        joints[pair_id] = {}
        for arm in ARMS:
            joints[pair_id][arm] = {}
            for dose in run127.DOSES:
                joints[pair_id][arm][float(dose)] = {}
                for qi in PRIMARY_QUESTIONS:
                    if arm == "pair_ab":
                        key = f"pair:{pair_id}:dose{dose:g}:q{qi}"
                    elif arm == "matched_ac":
                        key = f"matched:{pair_id}:dose{dose:g}:q{qi}"
                    else:
                        random_index = int(arm.rsplit("_", 1)[1])
                        key = f"random:{pair_id}:{random_index}:dose{dose:g}:q{qi}"
                    joints[pair_id][arm][float(dose)][qi] = get_mark(key, pair["family"])
    cell_eligibility = {}
    for pair in pairs:
        pair_id, family = pair["pair_id"], pair["family"]
        cell_eligibility[pair_id] = {}
        for dose in run127.DOSES:
            cell_eligibility[pair_id][float(dose)] = {}
            for qi in PRIMARY_QUESTIONS:
                required_keys = [f"single:{pair['a']}:dose{dose:g}:q{qi}",
                                 f"single:{pair['b']}:dose{dose:g}:q{qi}"]
                for arm in ARMS:
                    if arm == "pair_ab":
                        required_keys.append(f"pair:{pair_id}:dose{dose:g}:q{qi}")
                    elif arm == "matched_ac":
                        required_keys.append(f"matched:{pair_id}:dose{dose:g}:q{qi}")
                    else:
                        random_index = int(arm.rsplit("_", 1)[1])
                        required_keys.append(f"random:{pair_id}:{random_index}:dose{dose:g}:q{qi}")
                cell_eligibility[pair_id][float(dose)][qi] = all(
                    readable_by_key.get(key, False) for key in required_keys)
    return singles, joints, cell_eligibility


def _rescue_data(pairs: list[dict], eligible_ids: set[str],
                 singles: dict, joints: dict, cell_eligibility: dict) -> dict:
    output = {}
    pair_outcomes = {}
    for dose in run127.DOSES:
        report = count_rescues(pairs, eligible_ids, dose, joints, singles,
                               cell_eligibility=cell_eligibility)
        output[str(dose)] = report
        pair_outcomes[str(dose)] = {}
        for arm in ARMS:
            pair_outcomes[str(dose)][arm] = {
                pair_id: int(sum(cell["rescued"] is True for cell in cells.values()))
                for pair_id, cells in report["arms"][arm]["by_pair"].items()
                if any(cell["included"] for cell in cells.values())}
    return {"counts": output, "pair_outcomes": pair_outcomes}


def _bootstrap_rescues(pairs: list[dict], eligible_ids: set[str], rescue: dict) -> dict:
    report = {}
    for dose_text, arms in rescue["pair_outcomes"].items():
        report[dose_text] = {}
        for arm, values in arms.items():
            report[dose_text][arm] = bootstrap_pair_interval(values)
        pair_values = arms.get("pair_ab", {})
        for control in ("matched_ac", "random_0", "random_1", "random_2"):
            common = sorted(set(pair_values) & set(arms.get(control, {})))
            diff = {pair_id: pair_values[pair_id] - arms[control][pair_id]
                    for pair_id in common}
            report[dose_text][f"pair_ab_minus_{control}"] = bootstrap_pair_interval(diff)
    return report


def _fixed_emitted_measure(position: dict, state: dict) -> tuple[float, str]:
    support = [int(token) for token in position.get("support_ids", [])]
    centered = state.get("centered_delta_on_support")
    emitted = int(position["emitted_token"])
    if emitted in support:
        if not isinstance(centered, list) or len(centered) != len(support):
            raise IncompleteRun("fixed-path centered support does not match its token ids")
        field = "centered_delta_on_support"
        value = float(centered[support.index(emitted)])
    else:
        if "emitted_logit_delta" not in state:
            raise IncompleteRun("fixed-path emitted token is outside support and has no raw fallback")
        field = "emitted_logit_delta_fallback_outside_support"
        value = float(state["emitted_logit_delta"])
    recorded_field = state.get("emitted_logit_delta_field")
    recorded_value = state.get("emitted_logit_delta_for_scoring")
    if recorded_field is not None and recorded_field != field:
        raise IncompleteRun("fixed-path emitted logit field metadata does not match its support")
    if recorded_value is not None and not math.isclose(float(recorded_value), value,
                                                       rel_tol=1e-6, abs_tol=1e-7):
        raise IncompleteRun("fixed-path emitted logit value does not match its selected field")
    return value, field


def _fixed_measure(fixed_rows: list[dict], pairs: list[dict]) -> dict:
    buckets = defaultdict(lambda: defaultdict(list))
    field_counts = Counter()
    seen = set()
    expected_group_keys = set()
    for pair in pairs:
        for qi in range(len(run127.QUESTIONS)):
            for dose in run127.DOSES:
                expected_group_keys.add(f"group:{pair['pair_id']}:q{qi}:dose{dose:g}")
    for row in fixed_rows:
        key = str(row.get("fixed_key"))
        if key in seen:
            raise IncompleteRun(f"duplicate fixed-path group row: {key}")
        seen.add(key)
        if key not in expected_group_keys:
            continue
        pair_id, dose = str(row["pair_id"]), float(row["dose"])
        for ref_tag in ("A", "B"):
            positions = row["reference_paths"][ref_tag]
            for state_name in ("pair_ab", "matched_ac", "random_0", "random_1", "random_2"):
                for position in positions:
                    if state_name not in position["states"]:
                        continue
                    value, field = _fixed_emitted_measure(position,
                                                          position["states"][state_name])
                    buckets[(pair_id, dose, ref_tag)][state_name].append(value)
                    field_counts[field] += 1
    if seen != expected_group_keys:
        missing = sorted(expected_group_keys - seen)
        raise IncompleteRun(f"fixed-path table is incomplete: {len(missing)} groups missing")
    by_pair = {pair["pair_id"]: {"dose": {}} for pair in pairs}
    for pair in pairs:
        pair_id = pair["pair_id"]
        for dose in run127.DOSES:
            dose_row = {}
            for ref_tag in ("A", "B"):
                group = buckets[(pair_id, float(dose), ref_tag)]
                dose_row[ref_tag] = {condition: (sum(values) / len(values) if values else None)
                                     for condition, values in group.items()}
            by_pair[pair_id]["dose"][str(dose)] = dose_row
    dose05 = {pair_id: int((values["dose"]["0.5"]["A"].get("pair_ab") is not None)
                           and (values["dose"]["0.5"]["A"].get("matched_ac") is not None)
                           and values["dose"]["0.5"]["A"]["pair_ab"]
                               > values["dose"]["0.5"]["A"]["matched_ac"])
              for pair_id, values in by_pair.items()}
    return {"by_pair": by_pair, "prediction_127_5_wins": sum(dose05.values()),
            "prediction_127_5_total_pairs": len(dose05),
            "prediction_127_5_pair_wins": dose05,
            "emitted_logit_delta_fields": dict(field_counts),
            "bootstrap_win_rate": bootstrap_pair_interval(dose05)}


def _pair_question_table(pairs: list[dict], answer_by_key: dict,
                         study_items: list[dict], ds_labels: dict[str, dict],
                         luna_labels: dict[str, dict]) -> list[dict]:
    item_index = _item_index(study_items)
    result = []
    for pair in pairs:
        pair_id, family = pair["pair_id"], pair["family"]
        for qi, question in enumerate(run127.QUESTIONS):
            components = {}
            keys = {
                "single_a_dose1": f"single:{pair['a']}:dose1:q{qi}",
                "single_b_dose1": f"single:{pair['b']}:dose1:q{qi}",
                "single_a_dose0.5": f"single:{pair['a']}:dose0.5:q{qi}",
                "single_b_dose0.5": f"single:{pair['b']}:dose0.5:q{qi}",
                "single_a_dose0.707": f"single:{pair['a']}:dose0.707:q{qi}",
                "single_b_dose0.707": f"single:{pair['b']}:dose0.707:q{qi}",
                "no_push": f"plain:q{qi}",
            }
            for dose in run127.DOSES:
                dose_text = f"{dose:g}"
                keys[f"pair_ab_dose{dose_text}"] = f"pair:{pair_id}:dose{dose_text}:q{qi}"
                keys[f"matched_ac_dose{dose_text}"] = f"matched:{pair_id}:dose{dose_text}:q{qi}"
                for random_index in range(3):
                    keys[f"random_{random_index}_dose{dose_text}"] = (
                        f"random:{pair_id}:{random_index}:dose{dose_text}:q{qi}")
            for name, key in keys.items():
                answer = answer_by_key.get(key)
                reader_rows = {}
                for reader, labels in (("ds", ds_labels), ("luna", luna_labels)):
                    label_fields = _answer_label(key, family, item_index, labels)
                    reader_rows[reader] = label_fields
                available_reader_fields = [fields for fields in reader_rows.values()
                                           if fields.get("error") != "missing_label"]
                components[name] = {"answer_key": key,
                                    "answer": answer.get("answer") if answer else None,
                                    "broken_answer": (not answer or not str(answer.get("answer", "")).strip()
                                                      or any(fields.get("error")
                                                             or not fields.get("coherent", False)
                                                             for fields in available_reader_fields)),
                                    "readers": reader_rows}
            result.append({"pair_id": pair_id, "family": family,
                           "same_layer": pair["same_layer"], "question_index": qi,
                           "question": question, "components": components})
    return result


def _broken_counts(study_items: list[dict], answer_by_key: dict,
                   labels_by_reader: dict[str, dict],
                   included_item_ids: dict[str, set[str]]) -> dict:
    item_by_id = {str(item["public"]["item_id"]): item for item in study_items}
    result = {}
    for reader, labels in labels_by_reader.items():
        by_arm = defaultdict(lambda: {"items": 0, "broken": 0, "label_errors": 0,
                                      "missing_labels": 0, "stale_labels": 0,
                                      "unread_labels": 0, "empty_answers": 0,
                                      "incoherent": 0, "retries": 0,
                                      "truncated": 0, "truncated_attempts": 0})
        for item_id, item in item_by_id.items():
            if item_id not in included_item_ids[reader]:
                continue
            arm = item["private"]["arm"]
            label = labels.get(item_id)
            answer = item["a"]
            counts = by_arm[arm]
            counts["items"] += 1
            if not answer.strip():
                counts["empty_answers"] += 1
            if label is None:
                counts["missing_labels"] += 1
            elif label.get("answer_hash") != item["public"]["answer_hash"]:
                counts["stale_labels"] += 1
            label_is_current = label127.current_label(item, label)
            if not label_is_current:
                counts["label_errors"] += 1
                counts["unread_labels"] += 1
            elif not label["coherent"]:
                counts["incoherent"] += 1
            truncated_attempts = max(0, int((label or {}).get("truncated_attempts", 0)))
            counts["truncated"] += int(truncated_attempts > 0)
            counts["truncated_attempts"] += truncated_attempts
            counts["retries"] += max(0, int((label or {}).get("retry_count", 0)))
            if not answer.strip() or not label_is_current or not label["coherent"]:
                counts["broken"] += 1
        result[reader] = dict(by_arm)
    return result


def _raw_label_records(study_items: list[dict], ds_labels: dict[str, dict],
                       luna_labels: dict[str, dict]) -> list[dict]:
    records = []
    for item in study_items:
        item_id = str(item["public"]["item_id"])
        row = {"item_id": item_id, "answer_hash": item["public"]["answer_hash"],
               **item["private"], "answer": item["a"], "question": item["q"]}
        for reader, labels in (("ds", ds_labels), ("luna", luna_labels)):
            label = labels.get(item_id)
            if label and label.get("answer_hash") != item["public"]["answer_hash"]:
                label = None
            row[reader] = _label_fields(label)
        records.append(row)
    return records


def _prediction_rows(*, known_gates: dict, controls: dict, agreement: dict,
                     rescue: dict, cosine_rows: list[dict], fixed: dict,
                     rig_passed: bool) -> tuple[list[dict], bool]:
    known_passed = all(known_gates[reader]["passed"] for reader in label127.READERS)
    negative_passed = controls["negative_control_passed"]
    label_gate = bool(rig_passed and known_passed and negative_passed and agreement["passed"])
    same_layer_ids = set(controls["same_layer_eligible_pair_ids"])
    same_layer_rescues = {}
    for dose in run127.DOSES:
        arm = rescue["counts"][str(dose)]["arms"]["pair_ab"]["by_pair"]
        same_layer_rescues[str(dose)] = sum(
            int(any(cell["rescued"] for cell in arm[pair_id].values()))
            for pair_id in same_layer_ids if pair_id in arm)
    eligible_any = bool(controls["eligible_pair_ids"])
    diff_05 = (rescue["counts"]["0.5"]["arms"]["pair_ab"]["rescued_cells"]
               - rescue["counts"]["0.5"]["arms"]["matched_ac"]["rescued_cells"])
    unread_primary_cells = {
        "127.1": sum(not cell["included"]
                     for pair_id in same_layer_ids
                     for cell in rescue["counts"]["0.5"]["arms"]["pair_ab"]["by_pair"].get(
                         pair_id, {}).values()),
        "127.2": sum(not cell["included"]
                     for pair_id in same_layer_ids
                     for cell in rescue["counts"]["0.707"]["arms"]["pair_ab"]["by_pair"].get(
                         pair_id, {}).values()),
        "127.3": sum(not cell["included"]
                     for cells in rescue["counts"]["0.5"]["arms"]["pair_ab"]["by_pair"].values()
                     for cell in cells.values()),
    }
    readable_primary_cells = {
        prediction_id: sum(cell["included"]
                           for pair_id in (same_layer_ids if prediction_id in ("127.1", "127.2")
                                           else set(controls["eligible_pair_ids"]))
                           for cell in rescue["counts"]["0.5" if prediction_id != "127.2" else "0.707"]
                           ["arms"]["pair_ab"]["by_pair"].get(pair_id, {}).values())
        for prediction_id in ("127.1", "127.2", "127.3")}
    same_cos = [row for row in cosine_rows if row["same_layer"]]
    all_same_cosines = len(same_cos) == 4 and all(abs(float(row["cosine"])) < 0.2 for row in same_cos)
    fixed_wins = fixed["prediction_127_5_wins"]
    all_fixed = fixed["prediction_127_5_total_pairs"] == 27
    predictions = []
    for prediction_id, text in PREDICTIONS.items():
        if prediction_id in ("127.1", "127.2", "127.3"):
            if not rig_passed:
                verdict, reason = "not read (rig gate)", "do-nothing hook gate failed"
            elif not known_passed:
                verdict, reason = "not read (reader gate)", "a known-example reader check failed"
            elif not negative_passed:
                verdict, reason = "not read (negative-control gate)", "a reader's no-push false-positive gate failed"
            elif not agreement["passed"]:
                verdict, reason = "not read (reader agreement)", "agreement threshold failed or kappa was undefined"
            elif prediction_id in ("127.1", "127.2") and not same_layer_ids:
                verdict, reason = "not read (positive controls)", "no same-layer pair passed its positive-control gate"
            elif prediction_id == "127.3" and not eligible_any:
                verdict, reason = "not read (positive controls)", "no pair passed its positive-control gate"
            elif unread_primary_cells[prediction_id]:
                verdict, reason = ("not read (unread labels)",
                                   f"{unread_primary_cells[prediction_id]} primary cells lack a current DeepSeek label")
            elif prediction_id == "127.1":
                verdict, reason = ("held" if same_layer_rescues["0.5"] <= 1 else "failed"), None
            elif prediction_id == "127.2":
                verdict, reason = ("held" if same_layer_rescues["0.707"] >= 2 else "failed"), None
            else:
                verdict, reason = ("held" if diff_05 <= 3 else "failed"), None
        elif prediction_id == "127.4":
            verdict, reason = ("held" if all_same_cosines else "failed"), None
        else:
            if not rig_passed:
                verdict, reason = "not read (rig gate)", "do-nothing hook gate failed"
            elif not all_fixed:
                verdict, reason = "not read (fixed-path data)", "fixed-path table is incomplete"
            else:
                verdict, reason = ("held" if fixed_wins >= 17 else "failed"), None
        if prediction_id in ("127.1", "127.2"):
            observation = {"same_layer_rescued_pairs_dose_0.5": same_layer_rescues["0.5"],
                           "same_layer_rescued_pairs_dose_0.707": same_layer_rescues["0.707"],
                           "same_layer_eligible_pairs": len(same_layer_ids),
                           "readable_cells": readable_primary_cells[prediction_id]}
        elif prediction_id == "127.3":
            observation = {"pair_ab_minus_matched_ac_cells_dose_0.5": diff_05,
                           "eligible_pairs": len(controls["eligible_pair_ids"]),
                           "readable_cells": readable_primary_cells[prediction_id]}
        elif prediction_id == "127.4":
            observation = {"same_layer_absolute_cosines_below_0.2": all_same_cosines,
                           "same_layer_pair_count": len(same_cos)}
        else:
            observation = {"pairs_ab_exceeds_ac": fixed_wins,
                           "pairs": fixed["prediction_127_5_total_pairs"]}
        predictions.append({"id": prediction_id, "prediction": text,
                            "observation": observation,
                            "verdict": verdict, "withheld_reason": reason})
    headline_withdrawn = (not label_gate or not controls["same_layer_gate_passed"]
                          or any(unread_primary_cells.values()))
    return predictions, headline_withdrawn


def score_data(out_dir: Path, *, neurons_path: str | Path = run127.DEFAULT_NEURONS) -> dict:
    run_meta = read_json(out_dir / "run_meta.json")
    if run_meta.get("status") != "complete" or run_meta.get("stage") != "main":
        raise IncompleteRun("run127 is incomplete: a complete main stage is required")
    if int(run_meta.get("answer_count", -1)) != run127.EXPECTED_ANSWER_COUNT:
        raise IncompleteRun("run127 is incomplete: main arm count is not 1,914")
    by_id = run127.load_inventory(neurons_path)
    pairs = run127.selected_pairs(by_id, "main")
    answer_rows = read_jsonl(out_dir / "answers.jsonl")
    answer_by_key = {}
    for row in answer_rows:
        key = str(row.get("key", ""))
        if not key or key in answer_by_key:
            raise IncompleteRun(f"duplicate or empty answer key in answers.jsonl: {key!r}")
        answer_by_key[key] = row
    expected_rows = run127.build_arm_list(by_id, pairs, require_registered_counts=True)
    expected_keys = {row["key"] for row in expected_rows}
    if set(answer_by_key) != expected_keys:
        missing = sorted(expected_keys - set(answer_by_key))
        extra = sorted(set(answer_by_key) - expected_keys)
        raise IncompleteRun(f"run127 is incomplete: {len(missing)} answers missing and {len(extra)} unexpected")
    counts = dict(Counter(row["arm"] for row in answer_rows))
    if counts != run127.EXPECTED_ARMS:
        raise IncompleteRun(f"run127 is incomplete: answer arm counts differ: {counts}")
    rig = read_json(out_dir / "rig_gate.json")
    cosine = read_json(out_dir / "cosines.json")
    cosine_rows = cosine.get("pairs", [])
    if cosine.get("pair_ids") != [pair["pair_id"] for pair in pairs] or len(cosine_rows) != 27:
        raise IncompleteRun("run127 is incomplete: the 27-pair cosine table is missing")
    fixed_meta = read_json(out_dir / "fixed_path_meta.json")
    expected_fixed_counts = run127.fixed_path_counts(
        pairs, sorted({unit_id for pair in pairs for unit_id in (pair["a"], pair["b"])}))
    if fixed_meta.get("status") != "complete" or fixed_meta.get("counts") != expected_fixed_counts:
        raise IncompleteRun("run127 is incomplete: fixed-path score count does not match the design")
    fixed_rows = read_jsonl(out_dir / "fixed_path.jsonl")
    fixed_group_expected = len(pairs) * len(run127.QUESTIONS) * len(run127.DOSES)
    if len(fixed_rows) != fixed_group_expected:
        raise IncompleteRun(f"run127 is incomplete: expected {fixed_group_expected} fixed-path groups, "
                            f"found {len(fixed_rows)}")
    known = label127.read_known(repo_path(label127.KNOWN_PATH))
    known_items = label127.known_items(known)
    # This file proves the full-dose controls and no-push controls were read
    # before study labels were opened by label127's staged interface.
    saved_control_gates = read_json(out_dir / "control_gates.json")
    study_items = label127._study_items(answer_rows, by_id)
    expected_ds_items = len(study_items)
    if expected_ds_items != run127.EXPECTED_ANSWER_COUNT + 24:
        # The six shared no-push answers each have five family-target labels.
        raise IncompleteRun(f"reader inventory count differs from the registered expansion: {expected_ds_items}")
    ds_labels = latest_labels(out_dir / "labels-ds.jsonl")
    luna_labels = latest_labels(out_dir / "labels-luna.jsonl")
    selected_luna_items, luna_sample = label127._choose_luna_items(study_items, ds_labels)
    saved_luna_sample = read_json(out_dir / "luna_sample.json")
    if (saved_luna_sample.get("selected_items") != luna_sample["selected_items"]
            or saved_luna_sample.get("negative_sample_ids") != luna_sample["negative_sample_ids"]
            or saved_luna_sample.get("positive_item_ids") != luna_sample["positive_item_ids"]
            or saved_luna_sample.get("unread_item_ids") != luna_sample["unread_item_ids"]):
        raise IncompleteRun("Luna sample metadata does not match the current DeepSeek labels")
    known_gates, controls = _study_gates(out_dir, known, known_items, by_id, study_items)
    if (saved_control_gates.get("eligible_pair_ids") != controls["eligible_pair_ids"]
            or saved_control_gates.get("same_layer_eligible_pair_ids")
            != controls["same_layer_eligible_pair_ids"]):
        raise IncompleteRun("saved control gates do not match the current full-dose reader labels")
    agreement = _agreement(ds_labels, luna_labels, study_items)
    singles, joints, cell_eligibility = _primary_marks(pairs, study_items, ds_labels)
    eligible = set(controls["eligible_pair_ids"])
    rescue = _rescue_data(pairs, eligible, singles, joints, cell_eligibility)
    bootstrap = _bootstrap_rescues(pairs, eligible, rescue)
    fixed = _fixed_measure(fixed_rows, pairs)
    table = _pair_question_table(pairs, answer_by_key, study_items, ds_labels, luna_labels)
    broken = _broken_counts(
        study_items, answer_by_key, {"ds": ds_labels, "luna": luna_labels},
        {"ds": {str(item["public"]["item_id"]) for item in study_items},
         "luna": {str(item["public"]["item_id"]) for item in selected_luna_items}})
    raw_records = _raw_label_records(study_items, ds_labels, luna_labels)
    predictions, headline_withdrawn = _prediction_rows(
        known_gates=known_gates, controls=controls, agreement=agreement,
        rescue=rescue, cosine_rows=cosine_rows, fixed=fixed,
        rig_passed=bool(rig.get("passed")))
    result = {
        "status": "complete", "stage": "main",
        "run": {"answer_count": len(answer_rows), "arm_counts": counts,
                "pair_count": len(pairs), "unit_count": len(run_meta["selected_units"]),
                "questions": list(run127.QUESTIONS), "doses": list(run127.DOSES),
                "batch": run_meta.get("batch")},
        "gates": {"rig": rig, "known_examples": known_gates,
                  "positive_controls": controls["positive_controls"],
                  "eligible_pair_ids": controls["eligible_pair_ids"],
                  "same_layer_eligible_pair_ids": controls["same_layer_eligible_pair_ids"],
                  "negative_control_gates": controls["negative_control_gates"],
                  "negative_control_passed": controls["negative_control_passed"],
                  "reader_agreement": agreement},
        "luna_sample": luna_sample,
        "cosines": cosine_rows,
        "rescues": rescue["counts"],
        "pair_bootstrap": bootstrap,
        "fixed_path": {"meta": fixed_meta, "by_pair": fixed["by_pair"],
                       "prediction_127_5_wins": fixed["prediction_127_5_wins"],
                       "prediction_127_5_total_pairs": fixed["prediction_127_5_total_pairs"],
                       "prediction_127_5_pair_wins": fixed["prediction_127_5_pair_wins"],
                       "emitted_logit_delta_fields": fixed["emitted_logit_delta_fields"],
                       "bootstrap_win_rate": fixed["bootstrap_win_rate"]},
        "broken_answers": broken,
        "pair_question_rows": table,
        "raw_reader_labels": raw_records,
        "headline_withdrawn": headline_withdrawn,
        "predictions": predictions,
    }
    return result


def render_markdown(result: dict) -> str:
    lines = ["# Part 127 score", "",
             f"Run: {result['run']['answer_count']} generated answers; "
             f"{result['run']['pair_count']} pairs; batch {result['run']['batch']}.",
             f"Headline withdrawn: {'yes' if result['headline_withdrawn'] else 'no'}.", "",
             "## Registered predictions", "",
             "| Prediction | Observation | Verdict |", "| --- | --- | --- |"]
    for row in result["predictions"]:
        observation = json.dumps(row["observation"], ensure_ascii=False, sort_keys=True)
        lines.append(f"| {row['id']} {row['prediction']} | `{observation}` | {row['verdict']} |")
    agreement = result["gates"]["reader_agreement"]
    lines.extend(["", "## Reader gate and agreement", "",
                  "| Reader | Known examples |", "| --- | ---: |"])
    for reader in label127.READERS:
        gate = result["gates"]["known_examples"][reader]
        lines.append(f"| {label127.READER_NAMES[reader]} | {gate['correct']}/{gate['total']} "
                     f"({'pass' if gate['passed'] else 'fail'}) |")
    lines.extend(["", f"Compound-flag agreement: {agreement['raw_agreement']} raw over {agreement['n']} shared of "
                  f"{agreement['total_items']} items; {agreement['unread_items']} unread; "
                  f"kappa {agreement['kappa']}; {'pass' if agreement['passed'] else 'fail'}.", "",
                  "## Rescue counts", "",
                  "A rescue is a coherent DeepSeek family mark on a joint arm when both half-dose single answers lack it. "
                  "Only pairs that passed both readers' full-dose positive controls enter these counts.", "",
                  "| Dose | Arm | Rescued cells | Rescued pairs | Included cells | Excluded unread | Eligible pairs |",
                  "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
    for dose in run127.DOSES:
        for arm in ARMS:
            row = result["rescues"][str(dose)]["arms"][arm]
            lines.append(f"| {dose:g} | {arm} | {row['rescued_cells']} | {row['rescued_pairs']} | "
                         f"{row['included_cells']} | {row['excluded_unread_cells']} | {row['eligible_pairs']} |")
    lines.extend(["", "## Pair bootstrap intervals", "",
                  "Pairs are the resampling unit. Intervals show mean rescued cells per pair, "
                  "or the pair-level difference for A+B versus each control.", "",
                  "| Dose | Measure | Pairs | Mean | 95% interval |",
                  "| ---: | --- | ---: | ---: | ---: |"])
    for dose in run127.DOSES:
        for measure, interval in result["pair_bootstrap"][str(dose)].items():
            low, high = interval["lower_95"], interval["upper_95"]
            bounds = "n/a" if low is None else f"[{low:.4f}, {high:.4f}]"
            mean = "n/a" if interval["mean"] is None else f"{interval['mean']:.4f}"
            lines.append(f"| {dose:g} | {measure} | {interval['pairs']} | {mean} | {bounds} |")
    lines.extend(["", "## Pair cosine table", "",
                  "Cross-layer cosines are descriptive. Same-layer pairs are the direct geometric checks.", "",
                  "| Pair | Family | Layers | Same layer | Cosine |", "| --- | --- | --- | --- | ---: |"])
    for row in result["cosines"]:
        lines.append(f"| {row['pair_id']} | {row['family']} | L{row['layer_a']}/L{row['layer_b']} | "
                     f"{'yes' if row['same_layer'] else 'no'} | {row['cosine']:.5f} |")
    lines.extend(["", "## Fixed-path logit measure", "",
                  "The values are mean centered emitted-token logit changes against the no-push path over six prompts and up to 24 positions. Raw changes are used only when an emitted token falls outside the fixed support.", "",
                  "| Pair | Dose | A: A+B | A: A+C | B: A+B |", "| --- | ---: | ---: | ---: | ---: |"])
    for pair_id, row in result["fixed_path"]["by_pair"].items():
        for dose in run127.DOSES:
            values = row["dose"][str(dose)]
            a_ab = values["A"].get("pair_ab")
            a_ac = values["A"].get("matched_ac")
            b_ab = values["B"].get("pair_ab")
            fmt = lambda value: "n/a" if value is None else f"{value:.5f}"
            lines.append(f"| {pair_id} | {dose:g} | {fmt(a_ab)} | {fmt(a_ac)} | {fmt(b_ab)} |")
    lines.extend(["", "## Broken answers", "", "Unread labels are reported and excluded from rescue denominators. Empty outputs and incoherent outputs count as broken.", "",
                  "| Reader | Arm | Items | Broken | Unread labels | Retries | Truncated items | Truncated attempts | Empty | Incoherent |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for reader in ("ds", "luna"):
        for arm, counts in result["broken_answers"][reader].items():
            lines.append(f"| {label127.READER_NAMES[reader]} | {arm} | {counts['items']} | {counts['broken']} | "
                         f"{counts['unread_labels']} | {counts['retries']} | {counts['truncated']} | "
                         f"{counts['truncated_attempts']} | {counts['empty_answers']} | {counts['incoherent']} |")
    lines.extend(["", "## Pair and question rows", "",
                  f"The JSON file contains all {len(result['pair_question_rows'])} pair by question records, "
                  "including both readers' raw flags for every arm and dose.", "",
                  "## Reader-specific negative controls", "",
                  "| Reader | False-positive rate | No-push marks | Unread | Threshold | Result |",
                  "| --- | ---: | ---: | ---: | --- |"])
    for reader in label127.READERS:
        gate = result["gates"]["negative_control_gates"][reader]
        rate = gate["known_false_positive_rate"]
        lines.append(f"| {label127.READER_NAMES[reader]} | {rate:.4f} | {gate['no_push_marks']} | "
                     f"{gate['unread_items']} | {gate['threshold']} | {'pass' if gate['passed'] else 'fail'} |")
    lines.append("")
    return "\n".join(lines)


def score_files(out_dir: str | Path, *, neurons: str | Path = run127.DEFAULT_NEURONS) -> dict:
    directory = repo_path(out_dir)
    result = score_data(directory, neurons_path=neurons)
    (directory / "score127.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (directory / "SCORE.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="completed main run directory")
    parser.add_argument("--neurons", default=run127.DEFAULT_NEURONS)
    args = parser.parse_args(argv)
    try:
        result = score_files(args.out, neurons=args.neurons)
    except IncompleteRun as exc:
        print(f"INCOMPLETE: {exc}", file=sys.stderr, flush=True)
        return 2
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"score127: {exc}", file=sys.stderr, flush=True)
        return 2
    print(f"PART127_SCORE_COMPLETE predictions={len(result['predictions'])} "
          f"score={repo_path(args.out) / 'score127.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
