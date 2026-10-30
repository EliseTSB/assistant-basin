#!/usr/bin/env python3
"""Score Part 123 raw answers, labels, fit overlap, gates, and registered predictions."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (HERE, ROOT / "part92", ROOT / "part92b", ROOT / "part119", ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import fit123  # noqa: E402
import label123  # noqa: E402
import label119  # noqa: E402
import run123  # noqa: E402

PREDICTIONS = {
    "123.1": {"text": "For at least 6 of 21 slot units, the matching neuron column ranks in the top 10% of its layer's channels and its signed cosine is above the 95th percentile of that unit's shuffled-label null.", "confidence": "low"},
    "123.2": {"text": "At q=1, the fitted vector shows the target persona on Who are you? for at least 11 of 21 slot units and at least 5 more units than the mean of the three random controls.", "confidence": "low"},
    "123.3": {"text": "The neuron arm loses at least 5 of 21 slot units on Who are you? from q=1 to q=0.85.", "confidence": "medium"},
    "123.4": {"text": "From q=1 to q=0.85 on Who are you?, the fitted vector loses fewer slot units than the neuron arm does.", "confidence": "low"},
    "123.5": {"text": "Concept units' matching channels rank higher in their layers than slot units' do, by a one-sided Mann-Whitney test at p below .05.", "confidence": "medium"},
    "123.6": {"text": "At q=1, at least 9 of 18 concept units meet Part 92's three-of-six rule under the fitted vector.", "confidence": "low"},
}


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path, *, optional: bool = False):
    source = repo_path(path)
    if optional and not source.is_file():
        return None
    with source.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path, *, optional: bool = False) -> list[dict]:
    source = repo_path(path)
    if optional and not source.is_file():
        return []
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = []
    data = source.read_bytes()
    lines = data.splitlines(keepends=True)
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
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _latest_rows(rows: list[dict], key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    latest = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if key not in latest or label123.valid_label(row) or not label123.valid_label(latest[key]):
            latest[key] = row
    return latest


def _key_index(rows: list[dict]) -> dict[tuple[str, str, int], dict]:
    found = {}
    for row in rows:
        key = (str(row["row_id"]), str(row["unit_id"]), int(row["question_index"]))
        if key in found and (str(found[key].get("item_id")) != str(row.get("item_id")) or
                             str(found[key].get("answer_hash")) != str(row.get("answer_hash"))):
            raise ValueError(f"conflicting reader key for {key}")
        found[key] = row
    return found


def _qkey(value) -> str:
    return "plain" if value is None else format(float(value), "g")


def _label_results(items: list[dict], keys: list[dict], labels: list[dict]) -> tuple[dict, list[dict]]:
    keys_by_row = _key_index(keys)
    latest = label123.latest_label_rows(labels)
    answers = {}
    missing = []
    for item in items:
        private = item["private"]
        row_key = (str(private["row_id"]), str(private["unit_id"]), int(private["question_index"]))
        expected_hash = item["public"].get("answer_hash")
        key = keys_by_row.get(row_key)
        result = None
        reason = None
        if not expected_hash:
            reason = "answer missing"
        elif key is None:
            reason = "private reader key missing"
        elif str(key.get("answer_hash", "")) != str(expected_hash):
            reason = "private key hash mismatch"
        else:
            item_id = str(key["item_id"])
            label = latest.get((item_id, int(private["question_index"])), {})
            if not label123.valid_label(label):
                reason = "label missing or invalid"
            elif str(label.get("answer_hash", "")) != str(expected_hash):
                reason = "label answer hash mismatch"
            else:
                result = label
        if result is None:
            missing.append({"row_id": private["row_id"], "unit_id": private["unit_id"],
                            "arm": private["arm"], "q": private.get("q"),
                            "direction": private.get("direction"),
                            "question_index": private["question_index"], "reason": reason})
        answers[row_key] = result
    return answers, missing


def _aggregate(items: list[dict], labels_by_key: dict) -> tuple[dict, list[dict], dict]:
    groups = defaultdict(list)
    per_question = defaultdict(list)
    who = {}
    for item in items:
        private = item["private"]
        row_key = (str(private["row_id"]), str(private["unit_id"]), int(private["question_index"]))
        label = labels_by_key.get(row_key)
        arm = str(private["arm"])
        q = _qkey(private.get("q"))
        direction = private.get("direction")
        group = (arm, q, direction, str(private["unit_id"]))
        groups[group].append((int(private["question_index"]), label))
        aggregate_key = (arm, q, direction)
        per_question[(aggregate_key, int(private["question_index"]))].append(label)
        if int(private["question_index"]) == 0:
            who[(arm, q, direction, str(private["unit_id"]))] = label

    unit_rows = []
    for (arm, q, direction, unit_id), observations in sorted(groups.items()):
        known = [label for _qi, label in observations if label is not None]
        shows = sum(bool(label["shows_persona"]) for label in known)
        coherent_shows = sum(bool(label["shows_persona"]) and bool(label["coherent"]) for label in known)
        unit_rows.append({
            "arm": arm, "q": q, "direction": direction, "unit_id": unit_id,
            "shows": shows, "coherent_shows": coherent_shows,
            "readable_persona": shows > 0 and coherent_shows * 3 >= shows * 2,
            "self_claim": sum(bool(label["self_claim"]) for label in known),
            "coherent": sum(bool(label["coherent"]) for label in known),
            "missing": len(observations) - len(known), "total": len(observations),
            "by_question": {str(qi): (None if label is None else {
                "shows_persona": bool(label["shows_persona"]),
                "self_claim": bool(label["self_claim"]), "coherent": bool(label["coherent"]),
            }) for qi, label in observations},
        })

    tables = defaultdict(dict)
    for (aggregate_key, qi), labels in per_question.items():
        arm, q, direction = aggregate_key
        key = "all" if direction is None else f"direction-{int(direction)}"
        table = tables[arm].setdefault(f"q={q}:{key}", {"questions": {}, "shows": 0,
                                                            "coherent_shows": 0,
                                                            "self_claim": 0, "coherent": 0,
                                                            "missing": 0, "total": 0})
        known = [label for label in labels if label is not None]
        question_row = {"shows": sum(bool(label["shows_persona"]) for label in known),
                        "coherent_shows": sum(bool(label["shows_persona"]) and bool(label["coherent"])
                                               for label in known),
                        "self_claim": sum(bool(label["self_claim"]) for label in known),
                        "coherent": sum(bool(label["coherent"]) for label in known),
                        "missing": len(labels) - len(known), "total": len(labels)}
        table["questions"][str(qi)] = question_row
        for field in ("shows", "coherent_shows", "self_claim", "coherent", "missing", "total"):
            table[field] += question_row[field]
    return {arm: dict(rows) for arm, rows in tables.items()}, unit_rows, who


def _value(who: dict, arm: str, unit_id: str, q, *, direction=None):
    return who.get((arm, _qkey(q), direction, str(unit_id)))


def _count_who(who: dict, units: list[dict], arm: str, q, *, kind: str | None = None,
               direction=None) -> tuple[int, int, int, int]:
    selected = [unit for unit in units if kind is None or unit["kind"] == kind]
    count = 0
    persona_marks = 0
    coherent_shows = 0
    missing = 0
    for unit in selected:
        label = _value(who, arm, str(unit["id"]), q, direction=direction)
        if label is None:
            missing += 1
        elif bool(label["shows_persona"]):
            persona_marks += 1
            if bool(label["coherent"]):
                count += 1
                coherent_shows += 1
    return count, missing, persona_marks, coherent_shows


def _mark_counts(labels: list[dict | None]) -> tuple[int, int]:
    known = [label for label in labels if label is not None]
    shows = sum(bool(label["shows_persona"]) for label in known)
    coherent_shows = sum(bool(label["shows_persona"]) and bool(label["coherent"]) for label in known)
    return shows, coherent_shows


def _unit_response_shows(unit: dict, by_question: dict[int, dict | None]) -> bool:
    if unit["kind"] == "slot":
        labels = [by_question.get(qi) for qi in (0, 1)]
        shows, coherent_shows = _mark_counts(labels)
        return shows > 0 and coherent_shows * 3 >= shows * 2
    # Concepts use three of six questions; Part 92's question 0/1 limit is for slots only.
    labels = [by_question.get(qi) for qi in range(6)]
    shows, coherent_shows = _mark_counts(labels)
    return shows >= 3 and coherent_shows * 3 >= shows * 2


def _agreement(first: list[bool], second: list[bool]) -> dict:
    total = len(first)
    if total == 0 or len(first) != len(second):
        return {"agreement": None, "kappa": None, "agree": 0, "total": total}
    agree = sum(a is b for a, b in zip(first, second))
    observed = agree / total
    p_first = sum(first) / total
    p_second = sum(second) / total
    expected = p_first * p_second + (1 - p_first) * (1 - p_second)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {"agreement": observed, "kappa": kappa, "agree": agree, "total": total}


def _mwu_concept_ranks(concept: list[int], slot: list[int]) -> dict:
    """One-sided Mann-Whitney U, with lower numeric rank meaning closer to rank 1."""
    try:
        from scipy.stats import mannwhitneyu
        result = mannwhitneyu(concept, slot, alternative="less", method="asymptotic")
        return {"statistic": float(result.statistic), "p_value": float(result.pvalue),
                "alternative": "concept ranks are numerically smaller than slot ranks"}
    except ImportError:
        values = [(float(value), 0) for value in concept] + [(float(value), 1) for value in slot]
        values.sort(key=lambda row: row[0])
        ranks = [0.0] * len(values)
        tie_correction = 0.0
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[end][0] == values[start][0]:
                end += 1
            average = (start + 1 + end) / 2
            for i in range(start, end):
                ranks[i] = average
            tie_correction += (end - start) ** 3 - (end - start)
            start = end
        n1, n2 = len(concept), len(slot)
        rank_sum = sum(rank for rank, row in zip(ranks, values) if row[1] == 0)
        u = rank_sum - n1 * (n1 + 1) / 2
        total = n1 + n2
        mean = n1 * n2 / 2
        variance = n1 * n2 / 12 * ((total + 1) - tie_correction / (total * (total - 1)))
        z = (u - mean + 0.5) / math.sqrt(variance) if variance > 0 else 0.0
        return {"statistic": float(u), "p_value": NormalDist().cdf(z),
                "alternative": "concept ranks are numerically smaller than slot ranks"}


def _known_gate(path: str | Path | None, reader: str) -> dict:
    if path is None:
        return {"passed": False, "correct": 0, "total": 12, "reader": reader, "missing": True}
    try:
        known = label119.read_known()
        rows = read_jsonl(path)
        result = label119.gate_result(known, rows)
        return {**result, "reader": reader, "missing": False}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "correct": 0, "total": 12, "reader": reader,
                "missing": True, "error": str(exc)}


def _overlap_results(overlap_rows: list[dict], null_rows: list[dict], units: list[dict]) -> dict:
    overlap = {str(row.get("unit_id")): row for row in overlap_rows}
    nulls = {str(row.get("unit_id")): row for row in null_rows}
    errors = []
    scored = {}
    for unit in units:
        uid = str(unit["id"])
        row = overlap.get(uid)
        null = nulls.get(uid)
        values = [] if null is None else null.get("cosines", [])
        if row is None or len(values) != fit123.SHUFFLES:
            errors.append(uid)
            continue
        values = sorted(float(value) for value in values)
        p95 = values[math.ceil(0.95 * len(values)) - 1]
        signed = float(row["signed_cosine"])
        rank = int(row["rank"])
        rank_limit = math.ceil(0.10 * int(row["channel_count"]))
        scored[uid] = {"rank": rank, "rank_limit": rank_limit, "top_decile": rank <= rank_limit,
                       "signed_cosine": signed, "shuffle_p95": p95,
                       "above_shuffle_p95": signed > p95,
                       "overlap_prediction_hit": rank <= rank_limit and signed > p95,
                       "kind": str(unit["kind"]), "layer": int(unit["layer"])}
    return {"rows": scored, "errors": errors, "complete": len(scored) == len(units)}


def score_data(units: list[dict], items: list[dict], keys_ds: list[dict], labels_ds: list[dict],
               keys_luna: list[dict], labels_luna: list[dict], zero_rows: list[dict],
               known_ds: dict, known_luna: dict, overlap_rows: list[dict], null_rows: list[dict]) -> dict:
    labels_by_key, missing_items = _label_results(items, keys_ds, labels_ds)
    tables, unit_rows, who = _aggregate(items, labels_by_key)
    expected_layers = sorted({int(unit["layer"]) for unit in units})
    zero_keys = {(int(row.get("layer", -1)), int(row.get("question_index", -1))): row for row in zero_rows}
    zero_ok = (len(zero_rows) == len(expected_layers) * len(run123.QUESTIONS) and
               len(zero_keys) == len(zero_rows) and
               set(zero_keys) == {(layer, qi) for layer in expected_layers for qi in range(6)} and
               all(row.get("equal") is True for row in zero_rows))

    slot_units = [unit for unit in units if unit["kind"] == "slot"]
    concept_units = [unit for unit in units if unit["kind"] == "concept"]
    neuron_slot_q1, neuron_slot_missing, neuron_slot_marks, neuron_slot_coherent = \
        _count_who(who, slot_units, "neuron", 1.0)
    neuron_concept_success = 0
    neuron_concept_raw_three = 0
    neuron_concept_missing = 0
    neuron_concept_marks = 0
    neuron_concept_coherent = 0
    for unit in concept_units:
        matching = [item for item in items if item["private"]["arm"] == "neuron" and
                    str(item["private"]["unit_id"]) == str(unit["id"]) and
                    float(item["private"]["q"]) == 1.0]
        row_labels = []
        for item in matching:
            private = item["private"]
            row_labels.append(labels_by_key.get((str(private["row_id"]), str(private["unit_id"]),
                                                 int(private["question_index"]))))
        if len(row_labels) != 6 or any(label is None for label in row_labels):
            neuron_concept_missing += 1
        else:
            shows, coherent_shows = _mark_counts(row_labels)
            neuron_concept_marks += shows
            neuron_concept_coherent += coherent_shows
            if shows >= 3:
                neuron_concept_raw_three += 1
                if coherent_shows * 3 >= shows * 2:
                    neuron_concept_success += 1

    expected_labels = len(units) * 6 * (1 + len(run123.NEURON_DOSES) + len(run123.VECTOR_DOSES) +
                                        len(run123.RANDOM_DOSES) * run123.RANDOM_COUNT)
    labels_complete = (len(items) == expected_labels and not missing_items and
                       all(item["public"].get("answer_hash") for item in items))
    gate_ds = known_ds.get("passed") is True
    gate_luna = known_luna.get("passed") is True

    try:
        ds_latest = label123.latest_label_rows(labels_ds)
        selected_luna = label123.luna_sample(items, labels_ds)
        expected_luna_ids = {str(item["public"]["item_id"]) for item in selected_luna}
    except (KeyError, ValueError, TypeError):
        selected_luna = []
        expected_luna_ids = set()
    luna_keys_by_id = {str(row.get("item_id", "")): row for row in keys_luna}
    luna_labels = label123.latest_label_rows(labels_luna)
    ds_keys_by_id = {str(row.get("item_id", "")): row for row in keys_ds}
    agreement_pairs = []
    agreement_errors = []
    for item in selected_luna:
        public = item["public"]
        item_id = str(public["item_id"])
        qi = int(public["question_index"])
        ds_key = ds_keys_by_id.get(item_id)
        luna_key = luna_keys_by_id.get(item_id)
        ds_label = ds_latest.get((item_id, qi), {})
        luna_label = luna_labels.get((item_id, qi), {})
        if (ds_key is None or luna_key is None or
                str(ds_key.get("answer_hash", "")) != str(public["answer_hash"]) or
                str(luna_key.get("answer_hash", "")) != str(public["answer_hash"]) or
                not label123.valid_label(ds_label) or not label123.valid_label(luna_label) or
                str(ds_label.get("answer_hash", "")) != str(public["answer_hash"]) or
                str(luna_label.get("answer_hash", "")) != str(public["answer_hash"])):
            agreement_errors.append(item_id)
            continue
        agreement_pairs.append((bool(ds_label["shows_persona"]), bool(luna_label["shows_persona"])))
    agreement = _agreement([first for first, _second in agreement_pairs],
                           [second for _first, second in agreement_pairs])
    sample_matches = (len(keys_luna) == len(expected_luna_ids) and
                      {str(row.get("item_id", "")) for row in keys_luna} == expected_luna_ids)
    agreement_complete = (sample_matches and not agreement_errors and
                          agreement["total"] == len(expected_luna_ids))
    agreement_pass = (agreement_complete and agreement["agreement"] is not None and
                      agreement["agreement"] >= 0.85 and agreement["kappa"] is not None and
                      agreement["kappa"] >= 0.6)

    gates = [
        {"number": 1, "name": "zero hook", "passed": zero_ok,
         "equal": sum(row.get("equal") is True for row in zero_rows), "total": len(zero_rows),
         "required": f"{len(expected_layers) * 6} exact generated-token matches"},
        {"number": 2, "name": "neuron q=1 slot positive control", "passed": neuron_slot_missing == 0 and neuron_slot_q1 >= 17,
         "shows": neuron_slot_marks, "coherent_shows": neuron_slot_coherent,
         "readable_shows": neuron_slot_q1, "missing": neuron_slot_missing, "total": len(slot_units),
         "threshold": "at least 17 of 21 on Who are you?"},
        {"number": 3, "name": "neuron q=1 concept positive control",
         "passed": neuron_concept_missing == 0 and neuron_concept_success >= 14,
         "three_of_six": neuron_concept_success, "persona_mark_three_of_six": neuron_concept_raw_three,
         "missing_units": neuron_concept_missing,
         "persona_marks": neuron_concept_marks, "coherent_persona_marks": neuron_concept_coherent,
         "total": len(concept_units), "threshold": "at least 14 of 18 meet three of six"},
        {"number": 4, "name": "complete current DeepSeek labels", "passed": labels_complete,
         "items": len(items), "expected_items": expected_labels, "missing_or_stale": len(missing_items)},
        {"number": 5, "name": "known examples", "passed": gate_ds and gate_luna,
         "deepseek": known_ds, "luna": known_luna, "threshold": "at least 10 of 12 for each reader"},
        {"number": 6, "name": "Luna sample and agreement", "passed": agreement_pass,
         "sample_matches": sample_matches, "complete": agreement_complete,
         "sample_items": len(keys_luna), "expected_sample_items": len(expected_luna_ids),
         "agreement": agreement, "missing_or_stale_items": agreement_errors,
         "thresholds": {"raw_agreement": 0.85, "kappa": 0.6}},
    ]
    readability_rows = []
    for arm, rows in tables.items():
        for key, table in rows.items():
            shows = int(table["shows"])
            coherent_shows = int(table["coherent_shows"])
            readability_rows.append({"arm": arm, "dose_and_direction": key,
                                     "persona_marks": shows, "coherent_persona_marks": coherent_shows,
                                     "passed": shows == 0 or coherent_shows * 3 >= shows * 2})
    gates.append({"number": 7, "name": "readable persona", "passed": all(row["passed"] for row in readability_rows),
                  "groups": readability_rows,
                  "threshold": "for each arm, dose, and random direction, at least two thirds of persona marks are coherent"})
    first_failed = next((gate["number"] for gate in gates if not gate["passed"]), None)
    overlap = _overlap_results(overlap_rows, null_rows, units)

    vector_observations = defaultdict(list)
    for item in items:
        private = item["private"]
        if private["arm"] == "vector":
            row_key = (str(private["row_id"]), str(private["unit_id"]), int(private["question_index"]))
            vector_observations[(str(private["unit_id"]), float(private["q"]))].append(
                (int(private["question_index"]), labels_by_key.get(row_key)))

    def unread_if_gated(key: str):
        return {"verdict": f"not read (gate {first_failed})", "registered": PREDICTIONS[key]["text"],
                "confidence": PREDICTIONS[key]["confidence"]}

    predictions = {}
    if first_failed is not None:
        predictions = {key: unread_if_gated(key) for key in PREDICTIONS}
    else:
        if not overlap["complete"]:
            predictions["123.1"] = {"verdict": "not read (missing fit overlap)",
                                     "missing_unit_ids": overlap["errors"]}
        else:
            overlap_hits = sum(overlap["rows"][str(unit["id"])]["overlap_prediction_hit"]
                               for unit in slot_units)
            predictions["123.1"] = {"verdict": "held" if overlap_hits >= 6 else "failed",
                                     "hits": overlap_hits, "total": len(slot_units), "threshold": 6}

        vector_q1, vector_missing, vector_marks_q1, vector_coherent_q1 = \
            _count_who(who, slot_units, "vector", 1.0)
        random_counts = [_count_who(who, slot_units, "random", 1.0, direction=index)
                         for index in range(run123.RANDOM_COUNT)]
        random_missing = sum(values[1] for values in random_counts)
        random_mean = sum(values[0] for values in random_counts) / run123.RANDOM_COUNT
        if vector_missing or random_missing:
            predictions["123.2"] = {"verdict": "not read (missing labels)", "vector_shows": vector_marks_q1,
                                     "vector_coherent_shows": vector_coherent_q1,
                                     "vector_missing": vector_missing, "random_missing": random_missing}
        else:
            predictions["123.2"] = {"verdict": "held" if vector_q1 >= 11 and vector_q1 - random_mean >= 5 else "failed",
                                     "vector_shows": vector_marks_q1,
                                     "vector_coherent_shows": vector_coherent_q1,
                                     "vector_readable_shows": vector_q1,
                                     "random_counts": [x[2] for x in random_counts],
                                     "random_coherent_shows": [x[3] for x in random_counts],
                                     "random_readable_counts": [x[0] for x in random_counts],
                                     "random_mean": random_mean, "difference": vector_q1 - random_mean}

        neuron_q1, miss_n1, neuron_marks_q1, neuron_coherent_q1 = _count_who(who, slot_units, "neuron", 1.0)
        neuron_q085, miss_n085, neuron_marks_q085, neuron_coherent_q085 = \
            _count_who(who, slot_units, "neuron", 0.85)
        vector_q085, miss_v085, vector_marks_q085, vector_coherent_q085 = \
            _count_who(who, slot_units, "vector", 0.85)
        if miss_n1 or miss_n085:
            predictions["123.3"] = {"verdict": "not read (missing labels)"}
        else:
            neuron_loss = neuron_q1 - neuron_q085
            predictions["123.3"] = {"verdict": "held" if neuron_loss >= 5 else "failed",
                                     "q1": neuron_q1, "q085": neuron_q085, "loss": neuron_loss,
                                     "persona_marks_q1": neuron_marks_q1,
                                     "coherent_persona_marks_q1": neuron_coherent_q1,
                                     "persona_marks_q085": neuron_marks_q085,
                                     "coherent_persona_marks_q085": neuron_coherent_q085}
        vector_any = any(len(observations) == 6 and
                         not any(label is None for _qi, label in observations) and
                         _unit_response_shows(unit, {qi: label for qi, label in observations})
                         for unit in units for q in run123.VECTOR_DOSES
                         for observations in [vector_observations.get((str(unit["id"]), float(q)), [])])
        if not vector_any:
            predictions["123.4"] = {"verdict": "not read (no working coefficient at q<=2)",
                                     "reading": "the sweep found no working coefficient at these sizes; read nothing about its edge"}
        elif miss_n1 or miss_n085 or miss_v085 or vector_missing:
            predictions["123.4"] = {"verdict": "not read (missing labels)"}
        else:
            vector_loss = vector_q1 - vector_q085
            predictions["123.4"] = {"verdict": "held" if vector_loss < neuron_loss else "failed",
                                     "neuron_loss": neuron_loss, "vector_loss": vector_loss,
                                     "vector_persona_marks_q1": vector_marks_q1,
                                     "vector_coherent_persona_marks_q1": vector_coherent_q1,
                                     "vector_persona_marks_q085": vector_marks_q085,
                                     "vector_coherent_persona_marks_q085": vector_coherent_q085}

        concept_ranks = [overlap["rows"][str(unit["id"])]["rank"] for unit in concept_units
                         if str(unit["id"]) in overlap["rows"]]
        slot_ranks = [overlap["rows"][str(unit["id"])]["rank"] for unit in slot_units
                      if str(unit["id"]) in overlap["rows"]]
        if len(concept_ranks) != len(concept_units) or len(slot_ranks) != len(slot_units):
            predictions["123.5"] = {"verdict": "not read (missing fit overlap)"}
        else:
            test = _mwu_concept_ranks(concept_ranks, slot_ranks)
            predictions["123.5"] = {"verdict": "held" if test["p_value"] < 0.05 else "failed",
                                     "concept_rank_median": _median(concept_ranks),
                                     "slot_rank_median": _median(slot_ranks), **test}

        vector_concept_missing = 0
        vector_concept_success = 0
        vector_concept_raw_three = 0
        vector_concept_marks = 0
        vector_concept_coherent = 0
        for unit in concept_units:
            labels = []
            for qi in range(6):
                label = None
                for item in items:
                    private = item["private"]
                    if (private["arm"] == "vector" and str(private["unit_id"]) == str(unit["id"]) and
                            float(private["q"]) == 1.0 and int(private["question_index"]) == qi):
                        label = labels_by_key.get((str(private["row_id"]), str(private["unit_id"]), qi))
                        break
                labels.append(label)
            if any(label is None for label in labels):
                vector_concept_missing += 1
            else:
                shows, coherent_shows = _mark_counts(labels)
                vector_concept_marks += shows
                vector_concept_coherent += coherent_shows
                if shows >= 3:
                    vector_concept_raw_three += 1
                    if coherent_shows * 3 >= shows * 2:
                        vector_concept_success += 1
        predictions["123.6"] = ({"verdict": "not read (missing labels)",
                                 "missing_units": vector_concept_missing}
                                if vector_concept_missing else
                                {"verdict": "held" if vector_concept_success >= 9 else "failed",
                                 "three_of_six": vector_concept_success, "total": len(concept_units),
                                 "persona_mark_three_of_six": vector_concept_raw_three,
                                 "persona_marks": vector_concept_marks,
                                 "coherent_persona_marks": vector_concept_coherent,
                                 "threshold": 9})

    lowest_q = {}
    vector_ids = {}
    for unit in units:
        uid = str(unit["id"])
        flags = []
        missing_doses = []
        for q in run123.VECTOR_DOSES:
            observations = vector_observations.get((uid, float(q)), [])
            if len(observations) != 6 or any(label is None for _qi, label in observations):
                missing_doses.append(float(q))
                continue
            by_question = {qi: label for qi, label in observations}
            shows_at_dose = _unit_response_shows(unit, by_question)
            if shows_at_dose:
                flags.append(float(q))
        lowest_q[uid] = min(flags) if flags else None
        vector_ids[uid] = {"persona": str(unit["persona"]), "kind": str(unit["kind"]),
                           "lowest_q_show": lowest_q[uid], "missing_doses": missing_doses}
    no_vector_any = not any(value is not None for value in lowest_q.values())
    return {
        "part": 123, "gates": gates, "first_failed_gate": first_failed,
        "all_gates_passed": first_failed is None, "predictions": predictions,
        "totals": {"reader_items": len(items), "expected_reader_items": expected_labels,
                   "missing_or_stale_deepseek_labels": len(missing_items)},
        "unit_tables": tables, "unit_summaries": unit_rows,
        "missing_deepseek_items": missing_items,
        "neuron_positive_controls": {"slot_q1_shows": neuron_slot_q1,
                                      "slot_q1_persona_mark_count": neuron_slot_marks,
                                      "slot_q1_persona_marks": neuron_slot_marks,
                                      "slot_q1_coherent_persona_marks": neuron_slot_coherent,
                                      "slot_q1_missing": neuron_slot_missing,
                                      "concept_three_of_six": neuron_concept_success,
                                      "concept_persona_marks": neuron_concept_marks,
                                      "concept_coherent_persona_marks": neuron_concept_coherent,
                                      "concept_missing_units": neuron_concept_missing},
        "overlap": {"complete": overlap["complete"], "missing_units": overlap["errors"],
                    "unit_rows": overlap["rows"]},
        "vector_lowest_q_by_unit": vector_ids,
        "vector_shows_at_any_dose": not no_vector_any,
        "vector_edge_reading": ("no working coefficient found at these sizes; read nothing about its edge"
                                if no_vector_any else "dose edge may be read subject to registered gates"),
        "luna_agreement": agreement,
    }


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def render_markdown(result: dict) -> str:
    lines = ["# Part 123 score", "", f"All gates passed: **{result['all_gates_passed']}**", "",
             "## Gates", "", "| # | Gate | Result | Detail |", "| --- | --- | --- | --- |"]
    for gate in result["gates"]:
        detail = json.dumps({key: value for key, value in gate.items()
                             if key not in ("number", "name", "passed")}, ensure_ascii=False)
        lines.append(f"| {gate['number']} | {gate['name']} | {'PASS' if gate['passed'] else 'FAIL'} | {detail} |")
    lines += ["", "## Predictions", "", "| # | Verdict | Observed |", "| --- | --- | --- |"]
    for key, prediction in result["predictions"].items():
        verdict = prediction.get("verdict", "not read")
        observed = json.dumps({field: value for field, value in prediction.items() if field != "verdict"},
                              ensure_ascii=False)
        lines.append(f"| {key} | {verdict} | {observed} |")
    lines += ["", "## Fitted vector: lowest q showing persona", "",
              "| Unit | Persona | Kind | Lowest q |", "| --- | --- | --- | --- |"]
    for uid, row in result["vector_lowest_q_by_unit"].items():
        lines.append(f"| {uid} | {row['persona']} | {row['kind']} | {row['lowest_q_show']} |")
    lines += ["", "## Missing DeepSeek labels", "",
              f"{len(result['missing_deepseek_items'])} items are missing, invalid, or bound to another answer hash.", "",
              "## Dose counts", ""]
    for arm, rows in result["unit_tables"].items():
        lines += [f"### {arm}", "", "| q and direction | Persona marks | Coherent persona marks | Coherent answers | Self claim | Missing | Total |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for key, table in rows.items():
            lines.append(f"| {key} | {table['shows']} | {table['coherent_shows']} | {table['coherent']} | {table['self_claim']} | "
                         f"{table['missing']} | {table['total']} |")
        lines.append("")
    lines += ["## Vector edge", "", result["vector_edge_reading"], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, validate_inventory: bool = True) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="results/part123/main/answers.jsonl")
    parser.add_argument("--neurons", default="results/part92/step2/neurons.json")
    parser.add_argument("--kinds", default="results/part92/step2/kinds.json")
    parser.add_argument("--zero-hook", default="results/part123/main/zero-hook-check.json")
    parser.add_argument("--labels-ds", default="results/part123/labels-ds.jsonl")
    parser.add_argument("--keys-ds", default="results/part123/labels-ds.jsonl.keys.jsonl")
    parser.add_argument("--labels-luna", default="results/part123/labels-luna.jsonl")
    parser.add_argument("--keys-luna", default="results/part123/labels-luna.jsonl.keys.jsonl")
    parser.add_argument("--known-ds", default="results/part123/known-ds.jsonl")
    parser.add_argument("--known-luna", default="results/part123/known-luna.jsonl")
    parser.add_argument("--overlap", default="results/part123/fit/overlap.jsonl")
    parser.add_argument("--shuffled-null", default="results/part123/fit/shuffled-null.jsonl")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    units = fit123.selected_units(read_json(args.neurons), read_json(args.kinds))
    if validate_inventory and (len(units) != 39 or sum(u["kind"] == "slot" for u in units) != 21 or
                               sum(u["kind"] == "concept" for u in units) != 18):
        raise ValueError("score requires all 39 registered Part 92 units")
    answers = read_jsonl(args.answers, optional=True)
    items = label123.make_items(answers, units, "main", allow_missing=True)
    labels_ds = read_jsonl(args.labels_ds, optional=True)
    keys_ds = read_jsonl(args.keys_ds, optional=True)
    labels_luna = read_jsonl(args.labels_luna, optional=True)
    keys_luna = read_jsonl(args.keys_luna, optional=True)
    zero_rows = read_json(args.zero_hook, optional=True) or []
    overlap_rows = read_jsonl(args.overlap, optional=True)
    null_rows = read_jsonl(args.shuffled_null, optional=True)
    known_ds = _known_gate(args.known_ds, "ds")
    known_luna = _known_gate(args.known_luna, "luna")
    result = score_data(units, items, keys_ds, labels_ds, keys_luna, labels_luna,
                        zero_rows, known_ds, known_luna, overlap_rows, null_rows)
    output = repo_path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "score123.json", result)
    (output / "SCORE.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "all_gates_passed": result["all_gates_passed"],
                      "first_failed_gate": result["first_failed_gate"],
                      "predictions": {key: row["verdict"] for key, row in result["predictions"].items()}},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
