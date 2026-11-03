#!/usr/bin/env python3
"""Score Part 177's reader labels, tables, gates and registered predictions."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUT = ROOT / "results" / "part177"
BOOTSTRAP_REPS = 10_000
SEED = 177
READING_W3 = "a"
LABELS = ("asserted", "intermediate", "quoted from the problem", "rejected or doubted", "ambiguous")
GATE4_KNOWN_MINIMUM = 17
GATE4_MARGIN_BAR = 0.25
PREDICTION_177_1_BAR = 0.15
PREDICTION_177_2_BAR = 0.25
PREDICTION_177_3_BAR = 0.50
PREDICTION_177_4_BAR = 0.80
PREDICTION_177_6_BAR = 0.15
PREDICTION_W1_GAP_BAR = 0.15
PREDICTION_W1_EARLY_HOLD = 0.20
PREDICTION_W1_EARLY_FLOOR = 0.10
PREDICTION_W1_LATE_GAP_BAR = 0.15
PREDICTION_W2_HOLD = 0.40
PREDICTION_W2_FAIL = 0.20
PREDICTION_W3_HOLD = 1 / 3
PREDICTION_W3_FAIL = 0.15
MIN_N_177_4 = 20
MIN_N_177_6 = 15
MIN_N_W3 = 20


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def percentile(values, q):
    values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not values:
        return None
    return values[round(float(q) * (len(values) - 1))]


def bootstrap_interval(clusters, statistic, reps=BOOTSTRAP_REPS, seed=SEED):
    """Percentile interval resampling problem clusters with all their arms/draws attached."""
    clusters = list(clusters)
    if not clusters:
        return {"lower": None, "upper": None, "replicates": 0, "seed": int(seed)}
    rng = random.Random(int(seed))
    values = []
    for _ in range(max(0, int(reps))):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        value = statistic(sample)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return {"lower": percentile(values, 0.025), "upper": percentile(values, 0.975),
            "replicates": len(values), "seed": int(seed)}


def _rate(numerator, denominator, clusters=None, reps=BOOTSTRAP_REPS, seed=SEED):
    numerator, denominator = int(numerator), int(denominator)
    value = numerator / denominator if denominator else None
    interval = None
    if clusters is not None and denominator and clusters:
        interval = bootstrap_interval(clusters,
            lambda sample: (sum(int(row["numerator"]) for row in sample) /
                            sum(int(row["denominator"]) for row in sample))
            if sum(int(row["denominator"]) for row in sample) else None,
            reps=reps, seed=seed)
    return {"numerator": numerator, "denominator": denominator, "rate": value,
            "percent": (100.0 * numerator / denominator) if denominator else None, "ci95": interval}


def _difference(left, right, clusters=None, reps=BOOTSTRAP_REPS, seed=SEED):
    left_n, left_d = int(left["numerator"]), int(left["denominator"])
    right_n, right_d = int(right["numerator"]), int(right["denominator"])
    left_rate = left_n / left_d if left_d else None
    right_rate = right_n / right_d if right_d else None
    diff = left_rate - right_rate if left_rate is not None and right_rate is not None else None
    ci = None
    if clusters and left_d and right_d:
        ci = bootstrap_interval(clusters,
            lambda sample: ((sum(row["left_n"] for row in sample) / sum(row["left_d"] for row in sample)) -
                            (sum(row["right_n"] for row in sample) / sum(row["right_d"] for row in sample)))
            if sum(row["left_d"] for row in sample) and sum(row["right_d"] for row in sample) else None,
            reps=reps, seed=seed)
    return {"left": left, "right": right, "difference": diff,
            "difference_percentage_points": 100.0 * diff if diff is not None else None,
            "difference_ci95": ci}


def _valid_label(row):
    return bool(row and "error" not in row and not row.get("pending_dns") and row.get("label") in LABELS)


def _kappa(left, right):
    ids = sorted(set(left) & set(right))
    pairs = [(left[item_id], right[item_id]) for item_id in ids]
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    labels = set(value for pair in pairs for value in pair)
    expected = sum((sum(a == value for a, _ in pairs) / n) *
                   (sum(b == value for _, b in pairs) / n) for value in labels)
    return 1.0 if expected >= 1.0 and observed >= 1.0 else \
        None if expected >= 1.0 else (observed - expected) / (1.0 - expected)


def _known_gate(rows, examples):
    by_id = {str(row.get("id")): row for row in rows}
    known = []
    possible = 0
    unknown = 0
    for example in examples:
        row = by_id.get(str(example["id"]))
        if _valid_label(row):
            match = row["label"] == example["expected"]
            known.append({"id": str(example["id"]), "expected": example["expected"],
                          "observed": row["label"], "correct": match})
            possible += int(match)
        else:
            unknown += 1
            known.append({"id": str(example["id"]), "expected": example["expected"],
                          "observed": None, "correct": None})
    correct = sum(row["correct"] is True for row in known)
    lower, upper = correct, correct + unknown
    status = "passed" if lower >= GATE4_KNOWN_MINIMUM else \
        "failed" if upper < GATE4_KNOWN_MINIMUM else "not read"
    return {"status": status, "passed": status == "passed", "correct": correct, "unknown": unknown,
            "possible_correct": [lower, upper], "total": len(examples),
            "minimum": GATE4_KNOWN_MINIMUM, "rows": known}


def _expected_labels(keys, label_rows):
    labels = {str(row.get("id")): row for row in label_rows}
    if len(labels) != len(label_rows):
        raise ValueError("duplicate reader label IDs")
    return {str(key["id"]): labels.get(str(key["id"])) for key in keys}


def _hit_state(keys, label_by_id, problem_id, condition, donor_available=True):
    these = [key for key in keys if str(key["problem_id"]) == str(problem_id) and key["condition"] == condition]
    if not these:
        unavailable_donor = condition == "shuffled" and not donor_available
        return {"state": None if unavailable_donor else 0, "known_yes": False,
                "unknown": unavailable_donor, "first_asserted_position": None}
    valid = [label_by_id.get(str(key["id"])) for key in these]
    asserted_positions = [int(key["hit_start"]) for key, row in zip(these, valid)
                          if _valid_label(row) and row["label"] == "asserted"]
    if asserted_positions:
        return {"state": 1, "known_yes": True, "unknown": False,
                "first_asserted_position": min(asserted_positions)}
    missing = any(not _valid_label(row) for row in valid)
    return {"state": None if missing else 0, "known_yes": False, "unknown": missing,
            "first_asserted_position": None}


def _window_counts(keys, label_by_id, condition, target_only=False, c_ids=None):
    selected = [key for key in keys if key["condition"] == condition and
                (not target_only or key.get("target_T")) and
                (c_ids is None or str(key["problem_id"]) in c_ids)]
    asserted = sum(_valid_label(label_by_id.get(str(key["id"]))) and
                   label_by_id[str(key["id"])]["label"] == "asserted" for key in selected)
    unknown = sum(not _valid_label(label_by_id.get(str(key["id"]))) for key in selected)
    return {"numerator": int(asserted), "denominator": len(selected), "unknown": int(unknown),
            "observed_rate_valid_only": asserted / (len(selected) - unknown)
            if len(selected) - unknown else None}


def _window_share(keys, label_by_id, condition, problem_ids, reps, seed):
    problem_ids = sorted(map(str, problem_ids))
    grouped = {problem_id: {"numerator": 0, "denominator": 0, "unknown": 0}
               for problem_id in problem_ids}
    for key in keys:
        problem_id = str(key["problem_id"])
        if key["condition"] != condition or problem_id not in grouped:
            continue
        label = label_by_id.get(str(key["id"]))
        grouped[problem_id]["denominator"] += 1
        if not _valid_label(label):
            grouped[problem_id]["unknown"] += 1
        else:
            grouped[problem_id]["numerator"] += int(label["label"] == "asserted")
    counts = {name: sum(row[name] for row in grouped.values())
              for name in ("numerator", "denominator", "unknown")}
    lower, upper = _rate_bounds(counts)
    complete = counts["unknown"] == 0
    point = counts["numerator"] / counts["denominator"] if complete and counts["denominator"] else None
    ci = (bootstrap_interval(list(grouped.values()),
        lambda sample: sum(row["numerator"] for row in sample) / sum(row["denominator"] for row in sample)
        if sample and sum(row["denominator"] for row in sample) else None,
        reps=reps, seed=seed) if complete and counts["denominator"] else None)
    return {**counts, "point": point, "bounds": [lower, upper], "ci95": ci,
            "problem_clusters": len(grouped), "weighting": "one observation per selected hit window"}


def _window_margin(target_ids, keys, label_by_id, reps, seed):
    target_ids = sorted(map(str, target_ids))
    real = _window_share(keys, label_by_id, "real", target_ids, reps, seed)
    shuffled = _window_share(keys, label_by_id, "shuffled", target_ids, reps, seed + 1)
    difference = real["point"] - shuffled["point"] \
        if real["point"] is not None and shuffled["point"] is not None else None
    difference_bounds = _margin_bounds(real, shuffled)
    complete = real["unknown"] == 0 and shuffled["unknown"] == 0
    clusters = []
    real_by_problem, shuffled_by_problem = {}, {}
    for condition, target in (("real", real_by_problem), ("shuffled", shuffled_by_problem)):
        for key in keys:
            problem_id = str(key["problem_id"])
            if problem_id not in set(target_ids) or key["condition"] != condition:
                continue
            label = label_by_id.get(str(key["id"]))
            row = target.setdefault(problem_id, {"numerator": 0, "denominator": 0})
            row["denominator"] += 1
            row["numerator"] += int(_valid_label(label) and label["label"] == "asserted")
    for problem_id in target_ids:
        left = real_by_problem.get(problem_id, {"numerator": 0, "denominator": 0})
        right = shuffled_by_problem.get(problem_id, {"numerator": 0, "denominator": 0})
        clusters.append({"left_n": left["numerator"], "left_d": left["denominator"],
                         "right_n": right["numerator"], "right_d": right["denominator"]})
    ci = (bootstrap_interval(clusters,
        lambda sample: (sum(row["left_n"] for row in sample) / sum(row["left_d"] for row in sample) -
                        sum(row["right_n"] for row in sample) / sum(row["right_d"] for row in sample))
        if sample and sum(row["left_d"] for row in sample) and sum(row["right_d"] for row in sample) else None,
        reps=reps, seed=seed + 2) if complete and real["denominator"] and shuffled["denominator"] else None)
    return {"real": real, "shuffled": shuffled, "difference": difference,
            "difference_bounds": difference_bounds, "difference_ci95": ci,
            "denominator_windows": {"real": real["denominator"], "shuffled": shuffled["denominator"]}}


def _rate_bounds(counts):
    denominator = int(counts["denominator"])
    if not denominator:
        return None, None
    known = int(counts["numerator"])
    unknown = int(counts["unknown"])
    return known / denominator, (known + unknown) / denominator


def _margin_bounds(real, shuffled):
    real_bounds = _rate_bounds(real)
    shuffled_bounds = _rate_bounds(shuffled)
    if real_bounds is None or shuffled_bounds is None:
        return None, None
    return real_bounds[0] - shuffled_bounds[1], real_bounds[1] - shuffled_bounds[0]


def _unknown_boolean_range(states):
    known_yes = sum(value == 1 for value in states)
    unknown = sum(value is None for value in states)
    denominator = len(states)
    if not denominator:
        return {"known_yes": known_yes, "unknown": unknown, "denominator": denominator,
                "bounds": [None, None], "point": None}
    return {"known_yes": known_yes, "unknown": unknown, "denominator": denominator,
            "bounds": [known_yes / denominator, (known_yes + unknown) / denominator],
            "point": known_yes / denominator if not unknown else None}


def _threshold_verdict(bounds, threshold, relation="at_least"):
    if not bounds or bounds[0] is None or bounds[1] is None:
        return "not read"
    if relation == "at_least":
        if bounds[0] >= threshold:
            return "held"
        if bounds[1] < threshold:
            return "failed"
        return "not read"
    if relation == "greater":
        if bounds[0] > threshold:
            return "held"
        if bounds[1] <= threshold:
            return "failed"
        return "not read"
    raise ValueError(relation)


def _status_from_gates(read_result, reader_gate):
    gates = read_result.get("gates_1_to_3", {})
    data_ok = all(gates.get(name, {}).get("passed") for name in ("gate1", "gate2", "gate3"))
    if not data_ok:
        return False, "a Gate 1, 2 or 3 check failed"
    if reader_gate.get("status") != "passed":
        return False, "Gate 4 did not pass completely"
    return True, None


def _assertion_onset_distribution(target_ids, keys, label_by_id, traces_by_problem,
                                 reps=BOOTSTRAP_REPS, seed=SEED):
    """All possible (asserted trace count, first-assertion-before-onset count) totals."""
    totals = {(0, 0)}
    unknown_windows = 0
    eligible_if_known = 0
    before_if_known = 0
    onset_problem_n = 0
    for problem_id in sorted(map(str, target_ids)):
        trace = traces_by_problem.get(problem_id, {}).get("unedited", {})
        onset = trace.get("loop_onset")
        if onset is None:
            continue
        onset_problem_n += 1
        real = sorted([key for key in keys if str(key["problem_id"]) == problem_id and
                       key["condition"] == "real"], key=lambda row: int(row["hit_start"]))
        known_positions, unknown_positions = [], []
        for key in real:
            label = label_by_id.get(str(key["id"]))
            if _valid_label(label):
                if label["label"] == "asserted":
                    known_positions.append(int(key["hit_start"]))
            else:
                unknown_positions.append(int(key["hit_start"]))
        unknown_windows += len(unknown_positions)
        if known_positions:
            eligible_if_known += 1
            before_if_known += int(min(known_positions) < int(onset["char_index"]))
        options = set()
        for mask in range(1 << len(unknown_positions)):
            possible = list(known_positions) + [position for index, position in enumerate(unknown_positions)
                                                if mask & (1 << index)]
            if not possible:
                options.add((0, 0))
            else:
                options.add((1, int(min(possible) < int(onset["char_index"]))))
        totals = {(den + possible_den, before + possible_before)
                  for den, before in totals for possible_den, possible_before in options}
    min_den = min((den for den, _ in totals), default=0)
    max_den = max((den for den, _ in totals), default=0)
    rates = [before / den for den, before in totals if den]
    bounds = [min(rates), max(rates)] if rates and min_den >= 20 else [None, None]
    point = before_if_known / eligible_if_known if unknown_windows == 0 and eligible_if_known else None
    complete_clusters = []
    if unknown_windows == 0:
        for problem_id in sorted(map(str, target_ids)):
            trace = traces_by_problem.get(problem_id, {}).get("unedited", {})
            onset = trace.get("loop_onset")
            if onset is None:
                continue
            real = [key for key in keys if str(key["problem_id"]) == problem_id and key["condition"] == "real"]
            positions = [int(key["hit_start"]) for key in real
                         if _valid_label(label_by_id.get(str(key["id"]))) and
                         label_by_id[str(key["id"])]["label"] == "asserted"]
            if positions:
                complete_clusters.append({"eligible": 1,
                                          "before": int(min(positions) < int(onset["char_index"]))})
    ci = (bootstrap_interval(complete_clusters,
            lambda sample: sum(row["before"] for row in sample) / sum(row["eligible"] for row in sample)
            if sample and sum(row["eligible"] for row in sample) else None,
            reps=reps, seed=seed + 17)
          if complete_clusters else None)
    return {"onset_traces": onset_problem_n, "known_asserted_traces": eligible_if_known,
            "before_onset_known": before_if_known, "unknown_windows": unknown_windows,
            "possible_eligible_counts": [min_den, max_den], "share_bounds": bounds,
            "point": point, "ci95": ci, "all_admissible_completions_meet_20": min_den >= 20}


def _group_trace_assertions(target_ids, keys, label_by_id, traces_by_problem=None):
    ids = sorted(map(str, target_ids))
    rows = []
    for problem_id in ids:
        trace = (traces_by_problem or {}).get(problem_id, {}).get("unedited", {})
        real = _hit_state(keys, label_by_id, problem_id, "real")
        shuffled = _hit_state(keys, label_by_id, problem_id, "shuffled",
                              donor_available=bool(trace.get("donor_draw_n", 0)))
        rows.append({"problem_id": problem_id, "real": real, "shuffled": shuffled})
    return rows


def _paired_assertion_bounds(rows, reps=BOOTSTRAP_REPS, seed=SEED):
    n = len(rows)
    if not n:
        return {"bounds": [None, None], "point": None, "denominator": 0,
                "known_left": 0, "known_right": 0, "unknown_left": 0, "unknown_right": 0,
                "bootstrap_ci95": None}
    low, high, left_yes, right_yes, ul, ur = 0.0, 0.0, 0, 0, 0, 0
    values = []
    for row in rows:
        left = row["real"]["state"]
        right = row["shuffled"]["state"]
        left_min, left_max = (0, 1) if left is None else (left, left)
        right_min, right_max = (0, 1) if right is None else (right, right)
        low += left_min - right_max
        high += left_max - right_min
        left_yes += left == 1
        right_yes += right == 1
        ul += left is None
        ur += right is None
        values.append((left, right))
    complete = not ul and not ur
    return {"bounds": [low / n, high / n],
            "point": (left_yes - right_yes) / n if complete else None,
            "denominator": n, "known_left": left_yes, "known_right": right_yes,
            "unknown_left": ul, "unknown_right": ur,
            "bootstrap_ci95": (bootstrap_interval(
                [{"left": left, "right": right} for left, right in values],
                lambda sample: sum(row["left"] - row["right"] for row in sample) / len(sample)
                if sample and all(row["left"] is not None and row["right"] is not None for row in sample) else None,
                reps=reps, seed=seed) if complete else None)}


def _cross_table(rows, left_field, right_field, allow_unknown=False):
    cells = {"0,0": 0, "0,1": 0, "1,0": 0, "1,1": 0}
    unknown = 0
    for row in rows:
        left, right = row.get(left_field), row.get(right_field)
        if left is None or right is None:
            unknown += 1
            continue
        cells[f"{int(bool(left))},{int(bool(right))}"] += 1
    return {"cells_left_then_right": cells, "known_total": sum(cells.values()), "unknown_rows": unknown,
            "total_rows": len(rows)}


def _rate_by_binary_group(rows, group_field, outcome_field, reps, seed):
    groups = {}
    for value in (False, True):
        subset = [row for row in rows if row.get(group_field) is value]
        numerator = sum(bool(row.get(outcome_field)) for row in subset)
        clusters = [{"numerator": int(bool(row.get(outcome_field))), "denominator": 1} for row in subset]
        groups[str(value).lower()] = _rate(numerator, len(subset), clusters=clusters, reps=reps, seed=seed)
    return groups


def _box_tables(traces_by_id, reps, seed):
    out = {}
    for arm in ("unedited", "p50c"):
        traces = [row for row in traces_by_id.values() if row["arm"] == arm and row.get("literal_hit")]
        den = len(traces)
        counts = {cell: sum(row.get("box_cell") == cell for row in traces)
                  for cell in ("correct[0]", "pred_first wrong box", "pred_first null")}
        clusters = [{"problem_id": row["problem_id"],
                     "numerator": int(row.get("pred_first") is None), "denominator": 1} for row in traces]
        out[arm] = {"denominator_literal_hit_traces": den, "cells": {
                        cell: _rate(count, den, clusters=([{"problem_id": row["problem_id"],
                                                           "numerator": int(row.get("box_cell") == cell),
                                                           "denominator": 1} for row in traces]),
                                    reps=reps, seed=seed + i)
                        for i, (cell, count) in enumerate(counts.items())},
                    "pred_first_null_rate": _rate(sum(row.get("pred_first") is None for row in traces), den,
                                                   clusters=clusters, reps=reps, seed=seed + 10),
                    "pred_first_nonnull_share": _rate(sum(row.get("pred_first") is not None for row in traces), den,
                                                       clusters=[{"problem_id": row["problem_id"],
                                                                   "numerator": int(row.get("pred_first") is not None),
                                                                   "denominator": 1} for row in traces],
                                                       reps=reps, seed=seed + 11),
                    "earlier_gold_box_last_differs": {
                        "numerator": sum(row.get("earlier_gold_box_last_differs") for row in traces),
                        "denominator": den}}
    return out


def _onset_order_table(cohort, reps, seed):
    eligible = [row for row in cohort if row.get("literal_hit")]
    categories = {"hit before onset": 0, "onset before hit": 0, "no onset detected": 0}
    for row in eligible:
        onset = row.get("loop_onset")
        if onset is None:
            categories["no onset detected"] += 1
        elif int(row["first_hit_char"]) < int(onset["char_index"]):
            categories["hit before onset"] += 1
        else:
            categories["onset before hit"] += 1
    clusters = {name: [{"problem_id": row["problem_id"],
                        "numerator": int((row.get("loop_onset") is None and name == "no onset detected") or
                                         (row.get("loop_onset") is not None and name == "hit before onset" and
                                          int(row["first_hit_char"]) < int(row["loop_onset"]["char_index"])) or
                                         (row.get("loop_onset") is not None and name == "onset before hit" and
                                          int(row["first_hit_char"]) >= int(row["loop_onset"]["char_index"]))),
                        "denominator": 1} for row in eligible] for name in categories}
    return {name: _rate(count, len(eligible), clusters=clusters[name], reps=reps, seed=seed + i)
            for i, (name, count) in enumerate(categories.items())}


def _cluster_by_problem(traces):
    grouped = defaultdict(dict)
    for row in traces:
        grouped[str(row["problem_id"])][row["arm"]] = row
    return grouped


def _prediction(prediction, estimate, verdict, rule, confidence, author, extra=None):
    return {"prediction": prediction, "estimate": estimate, "verdict": verdict, "rule": rule,
            "confidence": confidence, "author": author, **(extra or {})}


def _make_predictions(read_result, traces_by_problem, keys, label_by_id, key_by_id,
                      gate4, gate4_passed, reps, seed):
    predictions = {}
    readable, blocked_reason = _status_from_gates(read_result, gate4)
    target_ids = list(map(str, read_result.get("target_T_ids", [])))
    assertion_rows = _group_trace_assertions(target_ids, keys, label_by_id, traces_by_problem)
    assertion_bounds = _paired_assertion_bounds(assertion_rows, reps=reps, seed=seed)

    # 177.1: trace-level paired assertions on the fixed target set T.
    v = _threshold_verdict([assertion_bounds["point"], assertion_bounds["point"]],
                           PREDICTION_177_1_BAR, "at_least") if readable else "not read"
    predictions["177.1"] = _prediction(
        "On T, the share of traces with an asserted real gold exceeds the same share for shuffled draw 1 by at least 15 points",
        {"difference": assertion_bounds["point"], "difference_bounds": assertion_bounds["bounds"],
         "denominator_problems": assertion_bounds["denominator"],
         "known_real_asserted": assertion_bounds["known_left"],
         "known_shuffled_asserted": assertion_bounds["known_right"],
         "unknown_real": assertion_bounds["unknown_left"], "unknown_shuffled": assertion_bounds["unknown_right"],
         "bootstrap_ci95": assertion_bounds["bootstrap_ci95"]}, v,
        "Point estimate of the paired trace-rate difference must be at least 0.15; report the problem-bootstrap interval.",
        "medium", "Claude", {"read_block": blocked_reason} if not readable else {})

    # 177.2: assertion share among target traces; unlabelled windows are allowed either value for the verdict.
    real_states = [row["real"]["state"] for row in assertion_rows]
    real_range = _unknown_boolean_range(real_states)
    real_ci = (bootstrap_interval([{"yes": int(value)} for value in real_states],
                 lambda sample: sum(row["yes"] for row in sample) / len(sample) if sample else None,
                 reps=reps, seed=seed + 2) if real_range["point"] is not None and real_states else None)
    v = _threshold_verdict([real_range["point"], real_range["point"]],
                           PREDICTION_177_2_BAR, "at_least") if readable else "not read"
    predictions["177.2"] = _prediction(
        "On T, at least 25% of traces have an asserted real gold",
        {"share": real_range["point"], "share_bounds": real_range["bounds"], "ci95": real_ci,
         "numerator_known": real_range["known_yes"], "unknown_traces": real_range["unknown"],
         "denominator": real_range["denominator"]}, v,
        "Point estimate of the trace share must be at least 0.25; report the problem-bootstrap interval.",
        "medium", "Claude",
        {"read_block": blocked_reason} if not readable else {})

    # 177.3: preserve each trace's first asserted position and enumerate unknown labels' outcomes.
    onset_summary = _assertion_onset_distribution(target_ids, keys, label_by_id, traces_by_problem,
                                                  reps=reps, seed=seed)
    onset_share = onset_summary["point"]
    onset_share_bounds = onset_summary["share_bounds"]
    if not onset_summary["all_admissible_completions_meet_20"]:
        v = "not read"
    else:
        v = _threshold_verdict([onset_share, onset_share], PREDICTION_177_3_BAR, "at_least") \
            if readable else "not read"
    predictions["177.3"] = _prediction(
        "Among T traces with an asserted real gold and a detected onset, at least half have the first asserted hit before the onset; not read below 20 traces",
        {"share": onset_share, **onset_summary}, v,
        "At least 20 eligible traces and the point estimate must be at least 0.5; report the interval.",
        "medium", "Claude",
        {"read_block": blocked_reason} if not readable else {})

    p50c = [row for row in traces_by_problem.values() if row.get("p50c")]
    residual = [bundle["p50c"] for bundle in traces_by_problem.values()
                if bundle.get("p50c", {}).get("capped") and not bundle["p50c"].get("think_closed")]
    detected = sum(row.get("loop_onset") is not None for row in residual)
    share = detected / len(residual) if residual else None
    ci = bootstrap_interval([{"yes": int(row.get("loop_onset") is not None)} for row in residual],
                            lambda sample: sum(row["yes"] for row in sample) / len(sample) if sample else None,
                            reps=reps, seed=seed + 4) if residual else None
    v = "not read" if len(residual) < MIN_N_177_4 else \
        _threshold_verdict([share, share], PREDICTION_177_4_BAR, "at_least") \
        if readable and share is not None else "not read"
    predictions["177.4"] = _prediction(
        "At least 80% of p50c traces that are capped with no </think> carry a detected onset",
        {"numerator": detected, "denominator": len(residual), "share": share, "ci95": ci}, v,
        f"At least {MIN_N_177_4} traces; point estimate of detected onset share must be at least 0.80.",
        "medium", "Claude",
        {"read_block": blocked_reason} if not readable else {})

    paired_both = []
    for problem_id, bundle in traces_by_problem.items():
        left, right = bundle.get("unedited"), bundle.get("p50c")
        if not left or not right:
            continue
        if not (left["capped"] and not left["think_closed"] and right["capped"] and not right["think_closed"]):
            continue
        if left.get("loop_onset") is None or right.get("loop_onset") is None:
            continue
        if not left.get("text_chars") or not right.get("text_chars"):
            continue
        paired_both.append({"problem_id": problem_id,
                            "unedited": left["loop_onset"]["char_index"] / left["text_chars"],
                            "p50c": right["loop_onset"]["char_index"] / right["text_chars"]})
    if len(paired_both) < MIN_N_177_6:
        median_diff, med_u, med_p, median_ci = None, None, None, None
        v = "not read"
    else:
        med_u = statistics.median(row["unedited"] for row in paired_both)
        med_p = statistics.median(row["p50c"] for row in paired_both)
        median_diff = med_p - med_u
        median_ci = bootstrap_interval(paired_both,
            lambda sample: (statistics.median(row["p50c"] for row in sample) -
                            statistics.median(row["unedited"] for row in sample)) if sample else None,
            reps=reps, seed=seed + 5)
        v = ("held" if median_diff > 0 else "failed") if readable else "not read"
    predictions["177.5"] = _prediction(
        "On problems in the same cohort in both arms with a detected onset in both, median onset as a share of text is later in p50c; not read below 15 pairs",
        {"pairs": len(paired_both), "unedited_median": med_u, "p50c_median": med_p,
         "median_difference": median_diff, "difference_ci95": median_ci}, v,
        "At least 15 paired cohort problems and p50c median onset fraction greater than unedited.",
        "medium", "Claude", {"read_block": blocked_reason} if not readable else {})

    long_pairs = []
    for problem_id, bundle in traces_by_problem.items():
        left, right = bundle.get("unedited"), bundle.get("p50c")
        if left and right and left["text_chars"] >= 50_000 and right["text_chars"] >= 50_000:
            long_pairs.append({"problem_id": problem_id,
                               "left": int(left.get("loop_onset_50k") is not None),
                               "right": int(right.get("loop_onset_50k") is not None)})
    left_yes = sum(row["left"] for row in long_pairs)
    right_yes = sum(row["right"] for row in long_pairs)
    diff = (left_yes - right_yes) / len(long_pairs) if long_pairs else None
    diff_ci = bootstrap_interval(long_pairs,
        lambda sample: sum(row["left"] - row["right"] for row in sample) / len(sample) if sample else None,
        reps=reps, seed=seed + 6) if long_pairs else None
    v = "not read" if len(long_pairs) < MIN_N_177_6 else \
        _threshold_verdict([diff, diff], PREDICTION_177_6_BAR, "at_least") \
        if readable and diff is not None else "not read"
    predictions["177.6"] = _prediction(
        "Over the 200 paired problems, with onset searched only in the first 50,000 characters of each text and pairs with a shorter text left out, the rate of a detected onset is at least 15 points lower in p50c",
        {"unedited": _rate(left_yes, len(long_pairs)), "p50c": _rate(right_yes, len(long_pairs)),
         "difference_unedited_minus_p50c": diff, "difference_ci95": diff_ci,
         "eligible_pairs": len(long_pairs)}, v,
        f"At least {MIN_N_177_6} pairs; point rate difference unedited minus p50c must be at least 0.15.",
        "medium", "Claude", {"read_block": blocked_reason} if not readable else {})

    # Weaver 177.W1: use C traces with an available draw 1; supplemental C\T draws never enter T predictions.
    c_traces = [row for row in traces_by_problem.values() if row.get("unedited", {}).get("cohort_C")]
    c_ids = {bundle["unedited"]["problem_id"] for bundle in c_traces}
    key_by_trace_condition = defaultdict(list)
    for key in keys:
        key_by_trace_condition[(str(key["problem_id"]), key["condition"])].append(key)
    paired_c = []
    early_real = 0
    real_hit_total = 0
    for bundle in c_traces:
        trace = bundle["unedited"]
        problem_id = str(trace["problem_id"])
        if not trace.get("donor_draw_n", 0):
            continue
        real_positions = [int(hit["start"]) for hit in trace.get("hit_spans", [])]
        shuffled_keys = key_by_trace_condition[(problem_id, "shuffled")]
        shuffled_positions = list(map(int, shuffled_keys[0].get("all_hit_starts", []))) if shuffled_keys else []
        real_early_n = sum(position / trace["text_chars"] <= 0.15 for position in real_positions) \
            if trace.get("text_chars") else 0
        real_hit_total += len(real_positions)
        early_real += real_early_n
        first_position_early = bool(real_positions and trace.get("text_chars") and
                                    min(real_positions) / trace["text_chars"] <= 0.15)
        real_after = bool(trace.get("text_chars") and any(
            position / trace["text_chars"] > 0.15 for position in real_positions))
        shuffled_after = bool(trace.get("text_chars") and any(
            position / trace["text_chars"] > 0.15 for position in shuffled_positions))
        paired_c.append({"problem_id": problem_id, "real": int(bool(real_positions)),
                         "shuffled": int(bool(shuffled_positions)), "early_real_n": real_early_n,
                         "real_hit_n": len(real_positions), "real_after": int(real_after),
                         "first_hit_early": int(first_position_early),
                         "shuffled_after": int(shuffled_after)})
    real_n = sum(row["real"] for row in paired_c)
    shuffled_n = sum(row["shuffled"] for row in paired_c)
    c_n = len(paired_c)
    # Amendment 1: the exclusion drops hits, so every trace in C stays in the denominator.
    late_real = sum(row["real_after"] for row in paired_c)
    late_shuffled = sum(row["shuffled_after"] for row in paired_c)
    first_hit_early = sum(row["first_hit_early"] for row in paired_c)
    first_share = early_real / real_hit_total if real_hit_total else None
    gap = (real_n - shuffled_n) / c_n if c_n else None
    late_gap = (late_real - late_shuffled) / c_n if c_n else None
    w1_ci = bootstrap_interval(paired_c,
        lambda sample: (sum(row["real"] - row["shuffled"] for row in sample) / len(sample)) if sample else None,
        reps=reps, seed=seed + 7) if paired_c else None
    early_ci = bootstrap_interval(paired_c,
        lambda sample: sum(row["early_real_n"] for row in sample) / sum(row["real_hit_n"] for row in sample)
        if sample and sum(row["real_hit_n"] for row in sample) else None,
        reps=reps, seed=seed + 8) if real_hit_total else None
    late_ci = bootstrap_interval(paired_c,
        lambda sample: (sum(row["real_after"] - row["shuffled_after"] for row in sample) / len(sample))
        if sample else None, reps=reps, seed=seed + 9) if paired_c else None
    if not readable:
        w1_verdict = "not read"
    elif not paired_c or first_share is None or late_gap is None:
        w1_verdict = "not read"
    elif first_share < PREDICTION_W1_EARLY_FLOOR:
        w1_verdict = "failed"
    elif gap > PREDICTION_W1_GAP_BAR and first_share >= PREDICTION_W1_EARLY_HOLD and \
            late_gap < PREDICTION_W1_LATE_GAP_BAR:
        w1_verdict = "held"
    else:
        # Amendment 1: as in every Weaver bar, only the stated falsifier gives `failed`.
        w1_verdict = "inconclusive"
    predictions["177.W1"] = _prediction(
        "The real-against-shuffled literal-hit gap on C exceeds 15 points, but at least 20% of real hits sit in the first 15% of text, and excluding them the gap falls under 15; falsifier: fewer than 10% of hits are that early",
        {"paired_C_n": c_n, "real_hit_rate": _rate(real_n, c_n), "shuffled_hit_rate": _rate(shuffled_n, c_n),
         "gap": gap, "gap_ci95": w1_ci, "early_real_hits": early_real,
         "real_hit_denominator": real_hit_total, "early_real_share": first_share, "early_real_share_ci95": early_ci,
         "first_hit_early_traces": _rate(first_hit_early, real_n),
         "late_C_n": c_n, "late_real_hit_rate": _rate(late_real, c_n),
         "late_shuffled_hit_rate": _rate(late_shuffled, c_n), "late_gap": late_gap,
         "late_gap_ci95": late_ci,
         "donor_available_C_n": c_n}, w1_verdict,
        "Hold requires point gap >0.15, early hit share >=0.20, and late trace gap <0.15 over all of C; fail below 0.10; otherwise inconclusive.",
        "medium", "Weaver", {"read_block": blocked_reason} if not readable else {})

    # Weaver 177.W2: Gate 4's shuffled-window asserted share over C, kept separate from the T gate margin.
    shuffled_c = _window_share(keys, label_by_id, "shuffled", c_ids, reps, seed + 10)
    w2_bounds = shuffled_c["bounds"]
    if not readable or not shuffled_c["denominator"]:
        w2_verdict = "not read"
    elif shuffled_c["point"] is not None and shuffled_c["point"] >= PREDICTION_W2_HOLD:
        w2_verdict = "held"
    elif shuffled_c["point"] is not None and shuffled_c["point"] < PREDICTION_W2_FAIL:
        w2_verdict = "failed"
    elif shuffled_c["point"] is not None:
        w2_verdict = "inconclusive"
    else:
        w2_verdict = "not read"
    predictions["177.W2"] = _prediction(
        "Shuffled windows clear 40% yes on Gate 4; falsifier: under 20%",
        {"asserted_windows": shuffled_c["numerator"], "total_C_windows": shuffled_c["denominator"],
         "unknown_windows": shuffled_c["unknown"], "share_bounds": w2_bounds,
         "share": shuffled_c["point"], "ci95": shuffled_c["ci95"]}, w2_verdict,
        "Point share is over shuffled windows; report a problem-cluster bootstrap interval.", "medium", "Weaver",
        {"read_block": blocked_reason} if not readable else {})

    # Weaver 177.W3, reading (a) is the verdict key. Both registered denominator variants are printed.
    c_literal = [bundle["unedited"] for bundle in c_traces if bundle["unedited"].get("literal_hit")]
    denom = len(c_literal)
    a_num = sum(row.get("pred_first") is not None and int(row["correct0"]) != 1 for row in c_literal)
    b_num = sum(int(row["correct0"]) != 1 for row in c_literal)
    a_rate = _rate(a_num, denom, clusters=[{"problem_id": row["problem_id"],
                                           "numerator": int(row.get("pred_first") is not None and
                                                             int(row["correct0"]) != 1),
                                           "denominator": 1} for row in c_literal],
                    reps=reps, seed=seed + 8)
    b_rate = _rate(b_num, denom, clusters=[{"problem_id": row["problem_id"],
                                           "numerator": int(int(row["correct0"]) != 1), "denominator": 1}
                                          for row in c_literal], reps=reps, seed=seed + 9)
    selected_rate = a_rate if READING_W3 == "a" else b_rate
    if not readable or denom < MIN_N_W3:
        w3_verdict = "not read"
    elif selected_rate["rate"] >= PREDICTION_W3_HOLD:
        w3_verdict = "held"
    elif selected_rate["rate"] < PREDICTION_W3_FAIL:
        w3_verdict = "failed"
    else:
        w3_verdict = "inconclusive"
    predictions["177.W3"] = _prediction(
        "Among C traces with a literal hit, at least a third have a pred_first that does not match the gold; falsifier: under 15%",
        {"reading_key": READING_W3, "reading_a_box_fails_correct0": a_rate,
         "reading_b_any_no_correct_box_including_null": b_rate,
         "denominator_C_literal_hit_traces": denom}, w3_verdict,
        "Verdict key a counts only non-null pred_first with correct[0] = 0; b additionally counts null pred_first.",
        "medium", "Weaver", {"read_block": blocked_reason} if not readable else {})
    return predictions


def _gate4(read_result, keys, ds_rows, haiku_rows, known_rows, examples, reader_meta, out_dir, reps, seed):
    label_by_id = {str(row.get("id")): row for row in ds_rows}
    if len(label_by_id) != len(ds_rows):
        raise ValueError("duplicate first-reader label IDs")
    haiku_by_id = {str(row.get("id")): row for row in haiku_rows}
    if len(haiku_by_id) != len(haiku_rows):
        raise ValueError("duplicate second-reader label IDs")
    expected_ids = [str(row["id"]) for row in keys]
    valid_ds = sum(_valid_label(label_by_id.get(item_id)) for item_id in expected_ids)
    ds_errors = sum(item_id in label_by_id and not _valid_label(label_by_id[item_id]) and
                    not label_by_id[item_id].get("pending_dns") for item_id in expected_ids)
    ds_pending = sum(item_id not in label_by_id or not _valid_label(label_by_id.get(item_id))
                     for item_id in expected_ids)
    error_rate = (len(expected_ids) - valid_ds) / len(expected_ids) if expected_ids else None

    examples = list(examples or [])
    known = _known_gate(known_rows, examples)
    sample = read_json(Path(out_dir) / "second-reader-sample.json", {}) or {}
    sample_ids = list(map(str, sample.get("ids", [])))
    second_missing = [item_id for item_id in sample_ids if not _valid_label(haiku_by_id.get(item_id))]
    first_missing_sample = [item_id for item_id in sample_ids if not _valid_label(label_by_id.get(item_id))]
    shared = [item_id for item_id in sample_ids if _valid_label(label_by_id.get(item_id)) and
              _valid_label(haiku_by_id.get(item_id))]
    kappa = _kappa({item_id: label_by_id[item_id]["label"] for item_id in shared},
                   {item_id: haiku_by_id[item_id]["label"] for item_id in shared})
    kappa_complete = bool(sample_ids) and not second_missing and not first_missing_sample and len(shared) == len(sample_ids)
    if kappa_complete:
        kappa_status = "not read" if kappa is None else \
            "passed" if kappa >= 0.6 else "failed"
    else:
        kappa_status = "not read"

    target = set(map(str, read_result.get("target_T_ids", [])))
    traces_by_problem = _cluster_by_problem(read_result.get("trace_results", []))
    window_margin = _window_margin(target, keys, label_by_id, reps, seed + 20)
    margin_bounds = window_margin["difference_bounds"]
    margin_point = window_margin["difference"]
    real = window_margin["real"]
    shuffled = window_margin["shuffled"]
    margin_ci = window_margin["difference_ci95"]
    if margin_point is None or ds_pending:
        margin_status = "not read"
    elif margin_point >= GATE4_MARGIN_BAR:
        margin_status = "passed"
    elif margin_point < GATE4_MARGIN_BAR:
        margin_status = "failed"

    second_valid = sum(_valid_label(haiku_by_id.get(item_id)) for item_id in sample_ids)
    second_errors = sum(item_id in haiku_by_id and not _valid_label(haiku_by_id[item_id]) and
                        not haiku_by_id[item_id].get("pending_dns") for item_id in sample_ids)
    second_pending = len(sample_ids) - second_valid
    second_error_rate = second_pending / len(sample_ids) if sample_ids else None
    if error_rate is None or second_error_rate is None:
        error_status = "not read"
    elif ds_pending or second_pending:
        error_status = "not read"
    elif error_rate > 0.02 or second_error_rate > 0.02:
        error_status = "failed"
    else:
        error_status = "passed"
    components = (known["status"], kappa_status, margin_status, error_status)
    status = "failed" if "failed" in components else "not read" if "not read" in components else "passed"
    return {"status": status, "passed": status == "passed",
            "components": {"known_examples": known, "second_reader_kappa": {
                                "kappa": kappa, "pairs": len(shared), "expected_pairs": len(sample_ids),
                                "missing_first": len(first_missing_sample), "missing_second": len(second_missing),
                                "complete": kappa_complete, "threshold": 0.6, "status": kappa_status},
                            "asserted_margin_T": {"real_window_share": real, "shuffled_window_share": shuffled,
                                                  "difference": margin_point, "difference_bounds": margin_bounds,
                                                  "difference_ci95": margin_ci, "threshold": 0.25,
                                                  "denominator_problems": len(target),
                                                  "denominator_windows": window_margin["denominator_windows"],
                                                  "weighting": "each selected window contributes one binary label; bootstrap clusters by problem",
                                                  "status": margin_status},
                            "reader_errors": {"expected_windows": len(expected_ids), "valid": valid_ds,
                                              "errors": ds_errors, "pending_or_missing": ds_pending,
                                              "error_rate": error_rate, "second_reader_errors": second_errors,
                                              "second_reader_expected": len(sample_ids),
                                              "second_reader_error_rate": second_error_rate, "maximum": 0.02,
                                              "status": error_status}},
            "reader_metadata": reader_meta}


def _read_tables(read_result, traces, keys, labels, reps, seed):
    trace_by_problem = _cluster_by_problem(traces)
    trace_by_arm_id = {(str(row["problem_id"]), row["arm"]): row for row in traces}
    cohort = [row for row in traces if row.get("cohort_C")]
    box = _box_tables(trace_by_arm_id, reps, seed)
    onset_order = _onset_order_table(cohort, reps, seed + 30)
    label_by_id = {str(row.get("id")): row for row in labels}
    target_ids = set(map(str, read_result.get("target_T_ids", [])))
    key_by_trace_condition = defaultdict(list)
    for key in keys:
        key_by_trace_condition[(str(key["problem_id"]), key["condition"])].append(key)

    paired_rows = []
    literal_cross = []
    asserted_cross_c = []
    asserted_cross_t = []
    for problem_id, bundle in sorted(trace_by_problem.items()):
        left, right = bundle.get("unedited"), bundle.get("p50c")
        if not left or not right:
            continue
        correct_p = int(right["correct0"]) == 1
        literal = bool(left.get("literal_hit"))
        real_state = _hit_state(keys, label_by_id, problem_id, "real")
        asserted = real_state["state"]
        paired_rows.append({"problem_id": problem_id, "unedited_literal_hit": int(literal),
                            "unedited_asserted": asserted, "p50c_correct": int(correct_p),
                            "in_T": problem_id in target_ids})
        literal_cross.append({"unedited_literal_hit": literal, "p50c_correct": correct_p})
        if left.get("cohort_C"):
            asserted_cross_c.append({"problem_id": problem_id,
                                     "unedited_asserted": None if asserted is None else bool(asserted),
                                     "p50c_correct": correct_p})
        if problem_id in target_ids:
            asserted_cross_t.append({"problem_id": problem_id,
                                     "unedited_asserted": None if asserted is None else bool(asserted),
                                     "p50c_correct": correct_p})

    literal_known = [{"literal_hit": bool(row["unedited_literal_hit"]), "p50c_correct": bool(row["p50c_correct"]),
                      "problem_id": row["problem_id"]} for row in paired_rows]
    assertion_c_known = [{"asserted": row["unedited_asserted"] is True,
                          "p50c_correct": bool(row["p50c_correct"]), "problem_id": row["problem_id"]}
                         for row in asserted_cross_c if row["unedited_asserted"] is not None]
    assertion_t_known = [{"asserted": row["unedited_asserted"] is True, "p50c_correct": bool(row["p50c_correct"]),
                          "problem_id": row["problem_id"]}
                         for row in paired_rows if row["in_T"] and row["unedited_asserted"] is not None]
    return {"box_cell_by_arm": box, "onset_order_C_literal_hit": onset_order,
            "paired_tables": {
                "unedited_literal_hit_vs_p50c_correct_all_200": {
                    "cross_table": _cross_table(literal_cross, "unedited_literal_hit", "p50c_correct"),
                    "p50c_correct_by_unedited_literal_hit": _rate_by_binary_group(
                        literal_known, "literal_hit", "p50c_correct", reps, seed + 40)},
                "unedited_asserted_vs_p50c_correct_C": {
                    "cross_table": _cross_table(asserted_cross_c, "unedited_asserted", "p50c_correct"),
                    "p50c_correct_by_unedited_asserted": _rate_by_binary_group(
                        assertion_c_known, "asserted", "p50c_correct", reps, seed + 41)},
                "unedited_asserted_vs_p50c_correct_T": {
                    "cross_table": _cross_table(asserted_cross_t, "unedited_asserted", "p50c_correct"),
                    "p50c_correct_by_unedited_asserted": _rate_by_binary_group(
                        assertion_t_known, "asserted", "p50c_correct", reps, seed + 42)},
                "pairs": len(paired_rows)},
            "paired_observations": paired_rows}


def _boolean_readout(states, reps, seed):
    known_yes = sum(value == 1 for value in states)
    unknown = sum(value is None for value in states)
    denominator = len(states)
    lower = known_yes / denominator if denominator else None
    upper = (known_yes + unknown) / denominator if denominator else None
    complete = unknown == 0
    interval = (bootstrap_interval([{"yes": int(value)} for value in states],
                                   lambda sample: sum(row["yes"] for row in sample) / len(sample)
                                   if sample else None, reps=reps, seed=seed) if complete and denominator else None)
    return {"numerator_known_yes": known_yes, "unknown": unknown, "denominator": denominator,
            "rate": known_yes / denominator if denominator and complete else None,
            "rate_bounds": [lower, upper], "ci95": interval}


def _reader_readouts(read_result, keys, ds_rows, reps, seed):
    label_by_id = {str(row.get("id")): row for row in ds_rows}
    target_ids = sorted(map(str, read_result.get("target_T_ids", [])))
    traces_by_problem = _cluster_by_problem(read_result.get("trace_results", []))
    real_keys = [key for key in keys if key["condition"] == "real" and key.get("target_T")]
    by_problem = defaultdict(list)
    for key in real_keys:
        by_problem[str(key["problem_id"])].append(key)

    def first_last(which):
        outcomes, clusters = [], []
        for problem_id in target_ids:
            these = sorted(by_problem.get(problem_id, []), key=lambda row: int(row["hit_start"]))
            if not these:
                continue
            key = these[0] if which == "first" else these[-1]
            label = label_by_id.get(str(key["id"]))
            state = 1 if _valid_label(label) and label["label"] == "asserted" else \
                0 if _valid_label(label) else None
            outcomes.append(state)
            if state is not None:
                clusters.append({"problem_id": problem_id, "yes": int(state)})
        return _boolean_readout(outcomes, reps, seed), len(outcomes)

    first, first_traces = first_last("first")
    last, last_traces = first_last("last")
    all_real_keys = real_keys
    rejected = sum(_valid_label(label_by_id.get(str(key["id"]))) and
                   label_by_id[str(key["id"])]["label"] == "rejected or doubted" for key in all_real_keys)
    valid_hits = sum(_valid_label(label_by_id.get(str(key["id"]))) for key in all_real_keys)
    unknown_hits = len(all_real_keys) - valid_hits
    rejected_bounds = [rejected / len(all_real_keys), (rejected + unknown_hits) / len(all_real_keys)] \
        if all_real_keys else [None, None]
    rejected_clusters = []
    by_problem_hit_counts = defaultdict(lambda: {"yes": 0, "den": 0, "unknown": 0})
    for key in all_real_keys:
        group = by_problem_hit_counts[str(key["problem_id"])]
        label = label_by_id.get(str(key["id"]))
        if _valid_label(label):
            group["den"] += 1
            group["yes"] += int(label["label"] == "rejected or doubted")
        else:
            group["unknown"] += 1
    if not unknown_hits:
        rejected_clusters = [{"yes": row["yes"], "den": row["den"]}
                             for row in by_problem_hit_counts.values()]
    rejected_ci = (bootstrap_interval(rejected_clusters,
        lambda sample: sum(row["yes"] for row in sample) / sum(row["den"] for row in sample)
        if sample and sum(row["den"] for row in sample) else None,
        reps=reps, seed=seed + 1) if rejected_clusters else None)

    before_onset_states = []
    for problem_id in target_ids:
        trace = traces_by_problem.get(problem_id, {}).get("unedited", {})
        onset = trace.get("loop_onset")
        if onset is None:
            continue
        candidates = by_problem.get(problem_id, [])
        known_before = any(_valid_label(label_by_id.get(str(key["id"]))) and
                           label_by_id[str(key["id"])]["label"] == "asserted" and
                           int(key["hit_start"]) < int(onset["char_index"]) for key in candidates)
        unknown_before = any(not _valid_label(label_by_id.get(str(key["id"]))) and
                             int(key["hit_start"]) < int(onset["char_index"]) for key in candidates)
        state = 1 if known_before else None if unknown_before else 0
        before_onset_states.append(state)

    cohort = [row for row in read_result.get("trace_results", []) if row.get("cohort_C")]
    group_stats = {}
    for group_name in ("looped", "merely_long"):
        group_rows = [row for row in cohort if row.get(group_name)]
        states = [_hit_state(keys, label_by_id, row["problem_id"], "real")["state"] for row in group_rows]
        hit_rows = [row for row in group_rows if row.get("literal_hit")]
        hit_states = [_hit_state(keys, label_by_id, row["problem_id"], "real")["state"] for row in hit_rows]
        group_stats[group_name] = {"all_C_traces": _boolean_readout(states, reps, seed + (2 if group_name == "looped" else 3)),
                                   "literal_hit_C_traces": _boolean_readout(hit_states, reps, seed + (4 if group_name == "looped" else 5))}

    return {"first_read_hit_asserted": {"traces_with_read_hit": first_traces, **first},
            "last_read_hit_asserted": {"traces_with_read_hit": last_traces, **last},
            "rejected_or_doubted_read_hit_share": {"numerator": rejected, "valid_read_hit_denominator": valid_hits,
                                                   "expected_read_hits": len(all_real_keys),
                                                   "unknown_read_hits": unknown_hits,
                                                   "share_valid_only": rejected / valid_hits if valid_hits else None,
                                                   "share_bounds": rejected_bounds, "ci95": rejected_ci},
            "asserted_before_detected_onset": _boolean_readout(before_onset_states, reps, seed + 6),
            "asserted_by_C_loop_class": group_stats}


def _paired_correctness(read_result, reps, seed):
    summaries = read_result.get("gate1_problem_summaries", {})
    by_arm = {arm: {str(row["problem_id"]): row for row in summaries.get(arm, [])}
              for arm in ("unedited", "p50c")}
    shared = sorted(set(by_arm["unedited"]) & set(by_arm["p50c"]))
    if set(by_arm["unedited"]) != set(by_arm["p50c"]):
        raise ValueError("the two arms have different per-problem ID sets")
    clusters = [{"problem_id": problem_id,
                 "unedited": list(by_arm["unedited"][problem_id]["correct"]),
                 "p50c": list(by_arm["p50c"][problem_id]["correct"])} for problem_id in shared]
    result = {"same_problem_ids": True, "problems": len(shared), "per_sample": {}, "any_of_four": {},
              "any_of_four_cross_table": None}
    for sample_index in range(4):
        paired = [{"problem_id": row["problem_id"], "unedited": row["unedited"][sample_index],
                   "p50c": row["p50c"][sample_index]} for row in clusters]
        left_n = sum(row["unedited"] for row in paired)
        right_n = sum(row["p50c"] for row in paired)
        left_rate = _rate(left_n, len(paired),
                          clusters=[{"numerator": row["unedited"], "denominator": 1} for row in paired],
                          reps=reps, seed=seed + sample_index)
        right_rate = _rate(right_n, len(paired),
                           clusters=[{"numerator": row["p50c"], "denominator": 1} for row in paired],
                           reps=reps, seed=seed + 4 + sample_index)
        diff_ci = bootstrap_interval(paired,
            lambda sample: sum(row["p50c"] - row["unedited"] for row in sample) / len(sample) if sample else None,
            reps=reps, seed=seed + 8 + sample_index)
        result["per_sample"][str(sample_index)] = {"unedited": left_rate, "p50c": right_rate,
                                                    "p50c_minus_unedited": (right_n - left_n) / len(paired)
                                                    if paired else None,
                                                    "difference_ci95": diff_ci}
    any_rows = [{"problem_id": row["problem_id"], "unedited": int(any(row["unedited"])),
                 "p50c": int(any(row["p50c"]))} for row in clusters]
    for arm in ("unedited", "p50c"):
        numerator = sum(row[arm] for row in any_rows)
        result["any_of_four"][arm] = _rate(
            numerator, len(any_rows),
            clusters=[{"problem_id": row["problem_id"], "numerator": row[arm], "denominator": 1}
                      for row in any_rows], reps=reps, seed=seed + (12 if arm == "unedited" else 13))
    result["any_of_four_cross_table"] = _cross_table(
        [{"unedited": bool(row["unedited"]), "p50c": bool(row["p50c"])} for row in any_rows],
        "unedited", "p50c")
    return result


def score(out_dir=DEFAULT_OUT, reps=BOOTSTRAP_REPS, seed=SEED):
    out = Path(out_dir)
    read_result = read_json(out / "read.json")
    traces = list(read_result.get("trace_results", []))
    read_gates = read_result.get("gates_1_to_3", {})
    data_gates_passed = all(read_gates.get(name, {}).get("passed")
                            for name in ("gate1", "gate2", "gate3"))
    keys = read_jsonl(out / "window-key.jsonl")
    windows = read_jsonl(out / "windows.jsonl")
    if data_gates_passed and {str(row["id"]) for row in windows} != {str(row["id"]) for row in keys}:
        raise ValueError("window-key IDs do not exactly match windows.jsonl")
    if not data_gates_passed:
        keys, windows = [], []
    ds_rows = read_jsonl(out / "labels-ds.jsonl")
    haiku_rows = read_jsonl(out / "labels-haiku.jsonl")
    known_rows = read_jsonl(out / "known-labels-ds.jsonl")
    known_file = read_json(HERE / "known177.json", {}) or {}
    examples = known_file.get("examples", [])
    reader_meta = read_json(out / "reader-metadata.json", {}) or {}
    gate4 = _gate4(read_result, keys, ds_rows, haiku_rows, known_rows, examples, reader_meta,
                   out, reps, seed) if data_gates_passed else {
                       "status": "not read", "passed": False,
                       "blocked": "Gate 1, 2, or 3 failed; reader and score readouts stopped"}

    # Reader gate includes all four requirements and separately blocks results behind Gates 1-3.
    predictions = _make_predictions(read_result, _cluster_by_problem(traces), keys,
                                    {str(row.get("id")): row for row in ds_rows},
                                    {str(row["id"]): row for row in keys}, gate4,
                                    gate4.get("status") == "passed", reps, seed)
    tables = _read_tables(read_result, traces, keys, ds_rows, reps, seed) if data_gates_passed else None
    reported_readouts = None
    if data_gates_passed:
        target_ids = set(map(str, read_result.get("target_T_ids", [])))
        trace_by_problem = _cluster_by_problem(traces)
        p50_closed_target = [bundle["p50c"] for problem_id, bundle in trace_by_problem.items()
                             if problem_id in target_ids and bundle.get("p50c", {}).get("think_closed")]
        p50_closed_hit_n = sum(bool(row.get("literal_hit")) for row in p50_closed_target)
        p50_closed_hit = _rate(p50_closed_hit_n, len(p50_closed_target),
                               clusters=[{"problem_id": row["problem_id"],
                                          "numerator": int(bool(row.get("literal_hit"))), "denominator": 1}
                                         for row in p50_closed_target], reps=reps, seed=seed + 60)
        reported_readouts = {"reader": _reader_readouts(read_result, keys, ds_rows, reps, seed + 70),
                             "correctness_same_500_problems": _paired_correctness(read_result, reps, seed + 80),
                             "p50c_literal_hit_when_thinking_closed_on_T": {
                                 "numerator": p50_closed_hit_n, "denominator": len(p50_closed_target),
                                 "rate": p50_closed_hit}}

    # Attach problem-bootstrap intervals to the three upstream gates; each cluster retains all four samples or arms.
    per_arm_gates = read_result.get("gates_1_to_3", {})
    gate_intervals = {}
    for arm in ("unedited", "p50c"):
        rows = [row for row in traces if row["arm"] == arm]
        all_rows = read_result.get("gate1_problem_summaries", {}).get(arm, [])
        gate_intervals[arm] = {"gate1_all_500": _rate(
                                   sum(row["correct_n"] for row in all_rows),
                                   sum(row["sample_n"] for row in all_rows),
                                   clusters=[{"problem_id": row["problem_id"],
                                              "numerator": row["correct_n"],
                                              "denominator": row["sample_n"]} for row in all_rows],
                                   reps=reps, seed=seed + (48 if arm == "unedited" else 49)),
                               "gate2_capped_first_200": _rate(
                                   per_arm_gates["gate2"]["by_arm"][arm]["numerator"],
                                   per_arm_gates["gate2"]["by_arm"][arm]["denominator"],
                                   clusters=[{"problem_id": row["problem_id"],
                                              "numerator": int(row["capped"]), "denominator": 1} for row in rows],
                                   reps=reps, seed=seed + (50 if arm == "unedited" else 51))}
    p50_rows = [row for row in traces if row["arm"] == "p50c" and row["correct0"] == 1]
    gate_intervals["p50c_gate3"] = _rate(
        per_arm_gates["gate3"]["numerator"], per_arm_gates["gate3"]["denominator"],
        clusters=[{"problem_id": row["problem_id"], "numerator": int(row["literal_hit"]), "denominator": 1}
                  for row in p50_rows], reps=reps, seed=seed + 52)

    score_value = {"part": 177, "reading_w3": READING_W3,
                   "read_gates_1_to_3": read_result.get("gates_1_to_3"), "gate4": gate4,
                   "gate_intervals": gate_intervals, "tables": tables,
                   "reported_readouts": reported_readouts,
                   "kept_going_reading_blocked": predictions["177.1"]["verdict"] == "failed",
                   "predictions": predictions,
                   "bootstrap": {"replicates": int(reps), "seed": int(seed),
                                 "unit": "problem ID; paired arms and all within-problem windows/draws stay attached"},
                   "null_rule": "Missing or pending labels prevent Gate 4 from passing; assertion summaries retain their bounds.",
                   "scope": "The first 200 sample-0 traces per arm and the registered MATH-500 test inputs."}
    write_json(out / "score.json", score_value)
    write_json(out / "prediction_table.json", predictions)
    return score_value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    result = score(args.out_dir, args.replicates, args.seed)
    print(json.dumps({"gate4": result["gate4"]["status"],
                      "predictions": {key: row["verdict"] for key, row in result["predictions"].items()},
                      "reading_w3": READING_W3}, ensure_ascii=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
