#!/usr/bin/env python3
"""Part 174 scoring, direction-level summaries, uncertainty and registered verdicts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part171"), str(ROOT / "part172"), str(ROOT / "part173")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import label173
import label174
import names173
import run171
import run174
import score173

# Import the frozen Part 173 scoring, modal/support/stability implementation and entity rule.
score_answer = score173.score_answer
modal = score173._modal
summarize_direction = score173.summarize_cell
resolve_entity = names173.resolve_entity
normalize_name = names173.normalize_name
READER_PROMPT = label173.READER_PROMPT

REPS = int(run174.DESIGN["bootstrap"]["replicates"])
SEED = int(run174.DESIGN["bootstrap"]["seed"])


def gate9_known(known):
    """Check the Part 173 known-answer bars for one reader."""
    total = int((known or {}).get("total", 0) or 0)
    relation = int((known or {}).get("relation_correct", 0) or 0)
    primary = int((known or {}).get("primary_correct", 0) or 0)
    relation_min = run174.DESIGN["gate9"]["minimum_relation"]
    primary_min = run174.DESIGN["gate9"]["minimum_primary"]
    expected = run174.DESIGN["gate9"]["known_examples"]
    return {"passed": bool(total == expected and relation >= relation_min and primary >= primary_min),
            "total": total, "relation_correct": relation, "primary_correct": primary,
            "minimum_total": expected, "minimum_relation": relation_min,
            "minimum_primary": primary_min}


def read_jsonl(path):
    return run171.read_jsonl(path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _label_index(rows):
    by_id = {}
    duplicates = []
    for row in rows:
        item_id = str(row.get("id"))
        if item_id in by_id:
            duplicates.append(item_id)
        by_id[item_id] = row
    if duplicates:
        raise ValueError(f"reader output has duplicate pair ids: {duplicates[:3]}")
    return by_id


def _reader_map(scored, first_rows):
    by_id = _label_index(first_rows)
    result = {}
    for row in scored:
        pair_id = label173.pair_id(row.get("question", ""), row.get("answer", ""))
        result[(str(row.get("arm")), str(row.get("prompt_id")))] = by_id.get(pair_id)
    return result


def _reader_gate(answers_path, answers, labels_dir):
    labels_dir = Path(labels_dir)
    first = read_jsonl(labels_dir / "labels-ds.jsonl") if (labels_dir / "labels-ds.jsonl").is_file() else []
    second = read_jsonl(labels_dir / "labels-haiku.jsonl") if (labels_dir / "labels-haiku.jsonl").is_file() else []
    items = label173.items_for(answers_path)
    sample = read_json(labels_dir / "second_reader.json", {}) or {}
    pair_gate = label173.gate9(first, second, sample, items)
    known = {}
    for reader in ("ds", "haiku"):
        value = read_json(labels_dir / f"known-{reader}.json", {}) or {}
        known[reader] = gate9_known(value)
        known[reader]["provenance_matches"] = (
            value.get("provenance") == label173.known_cache_provenance(reader) and
            len(value.get("rows", [])) == run174.DESIGN["gate9"]["known_examples"])
        known[reader]["passed"] = bool(known[reader]["passed"] and known[reader]["provenance_matches"])
    first_path = labels_dir / "labels-ds.jsonl"
    sample_sha_matches = (first_path.is_file() and
                          sample.get("first_reader_sha256") == hashlib.sha256(first_path.read_bytes()).hexdigest())
    pair_gate["first_reader_sample_sha_matches"] = bool(sample_sha_matches)
    pair_gate["passed"] = bool(pair_gate.get("passed") and sample_sha_matches)
    if not sample_sha_matches and pair_gate.get("status") == "passed":
        pair_gate["status"] = "failed"
    error_cap = label174.check_error_cap(answers_path, labels_dir)
    passed = bool(pair_gate.get("passed") and error_cap.get("passed") and
                  all(value["passed"] for value in known.values()))
    status = ("passed" if passed else "failed" if not error_cap.get("passed") or
              any(not value["passed"] for value in known.values()) or
              pair_gate.get("status") == "failed" else "unread")
    return {"passed": passed, "status": status,
            "known_examples": known, "second_reader": pair_gate, "error_cap": error_cap}


def _arm_row_map(scored):
    result = {}
    for row in scored:
        result.setdefault((str(row.get("set")), float(row.get("dose", 0.0))), []).append(row)
    return result


def _direction_summaries(scored, reader_by_row):
    by_arm = _arm_row_map(scored)
    directions = []
    for set_name in run174.SET_NAMES:
        for dose in run174.DESIGN["signs"]:
            arm = run174.dose_arm(set_name, dose)
            rows = by_arm.get((set_name, float(dose)), [])
            expected_rows = int(run174.DESIGN["gate1"]["rows_per_arm"])
            if len(rows) != expected_rows:
                raise ValueError(f"direction {arm} has {len(rows)} rows; expected {expected_rows}")
            summary = summarize_direction(rows, reader_by_row, mode="canonical")
            clean_rows = [row for row in rows if row.get("clean")]
            labelled_clean_answers = sum(
                (record := reader_by_row.get((str(row.get("arm")), str(row.get("prompt_id"))))) is not None and
                label173.valid(record)
                for row in clean_rows)
            unlabelled_clean_answers = len(clean_rows) - labelled_clean_answers
            reader_available = labelled_clean_answers > 0
            google_claims = (None if unlabelled_clean_answers else
                             int(summary["full_name_distribution"].get("google", 0)))
            directions.append({"set": set_name, "family": run174.FAMILY_BY_SET[set_name],
                               "seed": int(set_name.rsplit("d", 1)[1]), "dose": float(dose),
                               "arm": arm, "google_claim_count": google_claims,
                               "reader_available": bool(reader_available),
                               "unlabelled_clean_answers": int(unlabelled_clean_answers),
                               "reader_coverage_status": ("incomplete" if unlabelled_clean_answers else
                                                           "complete"),
                               "supported_status_basis": "labelled_answers_only",
                               "supported_status_flag": ("incomplete_reader_coverage"
                                                          if unlabelled_clean_answers else None),
                               "supported": bool(summary["eligible"]),
                               "stable": bool(summary["stable"]),
                               "summary": summary})
    return directions


def _draw_summaries(directions):
    by_set = {}
    for direction in directions:
        by_set.setdefault(direction["set"], {})[float(direction["dose"])] = direction
    draws = []
    for set_name in run174.SET_NAMES:
        positive = by_set[set_name][0.5]
        negative = by_set[set_name][-0.5]
        both = positive["supported"] and negative["supported"]
        if not positive["supported"] and not negative["supported"]:
            kind = "neither"
        elif positive["supported"] != negative["supported"]:
            kind = "one sign"
        elif positive["summary"]["modal"] == negative["summary"]["modal"]:
            kind = "both same"
        else:
            kind = "both different"
        google_plus = positive["google_claim_count"]
        google_minus = negative["google_claim_count"]
        raw_google_plus = int(positive["summary"].get("raw_word_hits", {}).get("google", 0))
        raw_google_minus = int(negative["summary"].get("raw_word_hits", {}).get("google", 0))
        draws.append({"set": set_name, "family": run174.FAMILY_BY_SET[set_name],
                      "seed": int(set_name.rsplit("d", 1)[1]), "draw_class": kind,
                      "both_supported": bool(both),
                      "google_claim_count_plus": google_plus,
                      "google_claim_count_minus": google_minus,
                      "D": (google_plus - google_minus if google_plus is not None and
                            google_minus is not None else None),
                      "raw_google_word_hits_plus": raw_google_plus,
                      "raw_google_word_hits_minus": raw_google_minus,
                      "raw_google_word_hit_difference": raw_google_plus - raw_google_minus,
                      "cf": None, "cr": None,
                      "positive_supported": bool(positive["supported"]),
                      "negative_supported": bool(negative["supported"]),
                      "positive_modal": positive["summary"]["modal"],
                      "negative_modal": negative["summary"]["modal"]})
    return draws


def _wilson(k, n, z=1.959963984540054):
    k, n = int(k), int(n)
    if n <= 0:
        return {"status": "not read", "k": k, "n": n, "rate": None,
                "interval_95": [None, None]}
    p = k / n
    z2 = z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return {"status": "read", "k": k, "n": n, "rate": p,
            "interval_95": [max(0.0, center - radius), min(1.0, center + radius)]}


def _exact_either_sign_bound(values, confidence=0.95):
    values = np.asarray(values, dtype=np.int8).reshape(-1)
    n = int(values.size)
    k = int(values.sum())
    if n == 0:
        return {"status": "not read", "label": "exact one-sided bound on either-sign rate",
                "k": 0, "n": 0, "rate": None, "interval_95": [None, None]}
    alpha = 1.0 - float(confidence)
    if k == 0:
        upper = 1.0 - alpha ** (1.0 / n)
        return {"status": "read", "label": "exact one-sided 95% upper bound on either-sign rate",
                "k": k, "n": n, "rate": 0.0, "interval_95": [0.0, upper],
                "upper_bound": upper}
    if k == n:
        lower = alpha ** (1.0 / n)
        return {"status": "read", "label": "exact one-sided 95% lower bound on either-sign rate",
                "k": k, "n": n, "rate": 1.0, "interval_95": [lower, 1.0],
                "lower_bound": lower}
    return {"status": "not applicable", "label": "exact one-sided bound used only at zero or all success",
            "k": k, "n": n, "rate": k / n, "interval_95": [None, None]}


def _bootstrap_draw_interval(values, reps=REPS, seed=SEED):
    values = np.asarray(values, dtype=np.int8)
    if values.ndim != 2 or values.shape[1] != 2 or not values.shape[0]:
        return {"status": "not read", "label": "draw-level; signs paired within draw",
                "replicates": int(reps),
                "interval_95": [None, None]}
    if np.all(values == values.flat[0]):
        return {"status": "not bootstrapped", "label": "draw-level; signs paired within draw",
                "reason": "zero or all-success outcome; see exact either-sign bound",
                "replicates": 0, "point_rate": float(values.mean()),
                "interval_95": [None, None]}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.shape[0], size=(int(reps), values.shape[0]))
    rates = values[indices].mean(axis=(1, 2))
    return {"status": "read", "label": "draw-level; signs paired within draw; prompts held fixed",
            "seed": int(seed), "replicates": int(reps),
            "point_rate": float(values.mean()),
            "interval_95": [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]}


def _bootstrap_stratified_draw_interval(values_by_family, reps=REPS, seed=SEED):
    arrays = {family: np.asarray(values, dtype=np.int8)
              for family, values in values_by_family.items() if np.asarray(values).size}
    if not arrays or any(values.ndim != 2 or values.shape[1] != 2 for values in arrays.values()):
        return {"status": "not read", "label": "draw-level stratified bootstrap; signs paired within draw",
                "replicates": int(reps), "interval_95": [None, None]}
    combined = np.concatenate(list(arrays.values()), axis=0)
    if np.all(combined == combined.flat[0]):
        return {"status": "not bootstrapped", "label": "draw-level stratified bootstrap; signs paired within draw",
                "reason": "zero or all-success outcome; see exact either-sign bound",
                "replicates": 0, "point_rate": float(combined.mean()),
                "interval_95": [None, None]}
    rng = np.random.default_rng(int(seed))
    rates = np.empty(int(reps), dtype=np.float64)
    for replicate in range(int(reps)):
        resampled = []
        for family in sorted(arrays):
            values = arrays[family]
            indices = rng.integers(0, values.shape[0], size=values.shape[0])
            resampled.append(values[indices])
        rates[replicate] = np.concatenate(resampled, axis=0).mean()
    return {"status": "read", "label": "draw-level stratified bootstrap; signs paired within draw",
            "seed": int(seed), "replicates": int(reps), "families_resampled_within": {
                family: int(values.shape[0]) for family, values in sorted(arrays.items())},
            "point_rate": float(combined.mean()),
            "interval_95": [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]}


def _draw_event_interval(events, reps=REPS, seed=SEED):
    events = np.asarray(events, dtype=np.int8).reshape(-1)
    exact = _exact_either_sign_bound(events)
    if exact["status"] == "not read" or exact["status"] == "read":
        return exact
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, events.size, size=(int(reps), events.size))
    rates = events[indices].mean(axis=1)
    return {"status": "read", "label": "draw-level bootstrap of either-sign events",
            "seed": int(seed), "replicates": int(reps), "k": int(events.sum()),
            "n": int(events.size), "rate": float(events.mean()),
            "interval_95": [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]}


def _stratified_event_interval(events_by_family, reps=REPS, seed=SEED):
    arrays = {family: np.asarray(events, dtype=np.int8).reshape(-1)
              for family, events in events_by_family.items() if np.asarray(events).size}
    if not arrays:
        return {"status": "not read", "label": "draw-level stratified bootstrap of either-sign events",
                "replicates": int(reps), "rate": None, "interval_95": [None, None]}
    combined = np.concatenate(list(arrays.values()))
    exact = _exact_either_sign_bound(combined)
    if exact["status"] == "read":
        exact["families_resampled_within"] = {family: int(events.size)
                                               for family, events in sorted(arrays.items())}
        return exact
    rng = np.random.default_rng(int(seed))
    rates = np.empty(int(reps), dtype=np.float64)
    for replicate in range(int(reps)):
        values = []
        for family in sorted(arrays):
            events = arrays[family]
            indices = rng.integers(0, events.size, size=events.size)
            values.append(events[indices])
        rates[replicate] = np.concatenate(values).mean()
    return {"status": "read", "label": "draw-level stratified bootstrap of either-sign events",
            "seed": int(seed), "replicates": int(reps), "k": int(combined.sum()),
            "n": int(combined.size), "rate": float(combined.mean()),
            "families_resampled_within": {family: int(events.size)
                                           for family, events in sorted(arrays.items())},
            "interval_95": [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]}


def _prompt_bootstrap(family, directions, rows_by_arm, reader_by_row, reps=REPS, seed=SEED):
    """Resample the 44 questions and recompute Part 173's support threshold per direction."""
    selected = [row for row in directions if row["family"] == family]
    if not selected:
        return {"label": "question-level; says nothing about new draws", "replicates": int(reps),
                "interval_95": [None, None]}
    rng = np.random.default_rng(int(seed))
    prompt_draws = rng.integers(0, run174.PROMPTS, size=(int(reps), run174.PROMPTS), dtype=np.int16)
    support_by_rep = np.zeros(int(reps), dtype=np.int16)
    rows_by_key = {}
    for direction in selected:
        key = (direction["set"], float(direction["dose"]))
        rows = rows_by_arm.get(key, [])
        row_by_prompt = {int(str(row.get("prompt_id", "new44b:-1")).rsplit(":", 1)[1]): row
                         for row in rows if str(row.get("prompt_id", "")).startswith("new44b:")}
        codes = np.full(run174.PROMPTS, -1, dtype=np.int16)
        broken = np.zeros(run174.PROMPTS, dtype=np.int8)
        entity_ids = {}
        for index in range(run174.PROMPTS):
            row = row_by_prompt.get(index)
            if row is None:
                continue
            broken[index] = int(bool(row.get("broken")))
            record = reader_by_row.get((str(row.get("arm")), str(row.get("prompt_id"))))
            entity = score173._primary_entity(record, mode="canonical")
            canonical = score173._canonical_entity(record)
            if entity is None or canonical == names173.QWEN_ENTITY:
                continue
            if entity not in entity_ids:
                entity_ids[entity] = len(entity_ids)
            codes[index] = entity_ids[entity]
        sampled = codes[prompt_draws]
        top_count = np.zeros(int(reps), dtype=np.int16)
        for code in range(len(entity_ids)):
            top_count = np.maximum(top_count, (sampled == code).sum(axis=1).astype(np.int16))
        broken_count = broken[prompt_draws].sum(axis=1)
        support_by_rep += ((top_count >= run174.DESIGN["cell"]["modal_count_min"]) &
                           (broken_count <= run174.DESIGN["cell"]["broken_max"])).astype(np.int16)
    rates = support_by_rep / len(selected)
    point = float(np.mean([row["supported"] for row in selected]))
    return {"label": "question-level bootstrap; held directions fixed; says nothing about new draws",
            "seed": int(seed), "replicates": int(reps), "point_rate": point,
            "interval_95": [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]}


def rate_uncertainty(directions, rows_by_arm, reader_by_row, reps=REPS, seed=SEED):
    output, paired_by_metric, events_by_metric = {}, {}, {}
    metric_fns = {
        "supported": lambda row: bool(row["supported"]),
        "google_led": lambda row: bool(row["supported"] and row["summary"]["modal"] == "google"),
        "other_led": lambda row: bool(row["supported"] and row["summary"]["modal"] != "google"),
    }
    for family in ("u100d", "f100d"):
        chosen = [row for row in directions if row["family"] == family]
        output[family] = {"signed_directions": len(chosen)}
        for offset, (metric, predicate) in enumerate(metric_fns.items()):
            by_draw = {}
            for row in chosen:
                by_draw.setdefault(row["seed"], {})[float(row["dose"])] = int(predicate(row))
            paired = np.asarray([[by_draw[seed_value][0.5], by_draw[seed_value][-0.5]]
                                 for seed_value in sorted(by_draw)], dtype=np.int8)
            events = np.any(paired, axis=1).astype(np.int8) if paired.size else np.asarray([], dtype=np.int8)
            paired_by_metric.setdefault(metric, {})[family] = paired
            events_by_metric.setdefault(metric, {})[family] = events
            output[family].setdefault("rates", {})[metric] = {
                "at_each_sign_wilson": {
                    str(sign): _wilson(sum(predicate(row) for row in chosen if row["dose"] == sign),
                                       sum(row["dose"] == sign for row in chosen))
                    for sign in run174.DESIGN["signs"]},
                "pooled_signed_rate": _bootstrap_draw_interval(
                    paired, reps=reps, seed=int(seed) + offset),
                "either_sign_rate": _draw_event_interval(
                    events, reps=reps, seed=int(seed) + offset),
            }
        output[family]["prompt_bootstrap"] = _prompt_bootstrap(
            family, chosen, rows_by_arm, reader_by_row, reps=reps, seed=int(seed))
    output["combined"] = {"label": "families stratified; resampled within u100d (40) and f100d (20) draws"}
    for offset, metric in enumerate(metric_fns):
        family_values = paired_by_metric.get(metric, {})
        metric_seed = int(seed) if metric == "supported" else int(seed) + 300 + offset
        output["combined"][metric] = {
            "pooled_signed_rate": _bootstrap_stratified_draw_interval(
                family_values, reps=reps, seed=metric_seed),
            "either_sign_rate": _stratified_event_interval(
                events_by_metric.get(metric, {}), reps=reps, seed=int(seed) + 200 + offset),
        }
    return output


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_test(xs, ys, sd=None, minimum_sd=0.01):
    try:
        xs = np.asarray(xs, dtype=np.float64).reshape(-1)
        ys = np.asarray(ys, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return {"status": "not read", "rho": None, "interval_95": [None, None],
                "reason": "non-numeric correlation input"}
    n = int(min(xs.size, ys.size))
    observed_sd = (float(np.std(xs, ddof=1)) if sd is None and xs.size > 1 else sd)
    base = {"n": n, "standard_deviation": observed_sd,
            "minimum_standard_deviation": float(minimum_sd), "rho": None,
            "p_value": None, "interval_95": [None, None]}
    if n == 0:
        return {**base, "status": "not read", "reason": "empty correlation denominator"}
    if xs.size != ys.size:
        return {**base, "status": "not read", "reason": "mismatched correlation observations"}
    if xs.size < 3:
        return {**base, "status": "not read", "reason": "too few observations for Spearman correlation"}
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        return {**base, "status": "not read", "reason": "nonfinite correlation input"}
    if observed_sd is None or not math.isfinite(float(observed_sd)):
        return {**base, "status": "not read", "reason": "predictor standard deviation is nonfinite"}
    if float(observed_sd) < minimum_sd:
        return {**base, "status": "not read", "reason": "predictor standard deviation is under 0.01"}
    counts = Counter(ys.tolist())
    if counts and max(counts.values()) > ys.size / 2:
        return {**base, "status": "not read", "reason": "one D value occurs on more than half of draws"}
    rx, ry = _rankdata(xs), _rankdata(ys)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return {**base, "status": "not read", "reason": "constant ranks"}
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if not math.isfinite(rho):
        return {**base, "status": "not read", "reason": "nonfinite Spearman correlation"}
    try:
        from scipy.stats import spearmanr
        p_value = float(spearmanr(xs, ys).pvalue)
        if not math.isfinite(p_value):
            p_value = None
    except ImportError:
        p_value = None
    return {**base, "status": "read", "rho": rho, "p_value": p_value}


def spearman_with_interval(xs, ys, reps=REPS, seed=SEED, strata=None, sd=None):
    try:
        xs = np.asarray(xs, dtype=np.float64).reshape(-1)
        ys = np.asarray(ys, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return {"status": "not read", "rho": None, "interval_95": [None, None],
                "reason": "missing or non-numeric correlation input"}
    observed_sd = (float(np.std(xs, ddof=1)) if sd is None and xs.size > 1 else sd)
    result = spearman_test(xs, ys, sd=observed_sd)
    result.update({"interval_method": ("draw-level percentile bootstrap stratified by family"
                                       if strata is not None else "draw-level percentile bootstrap"),
                   "replicates": int(reps),
                   "valid_replicates": 0})
    if result["status"] != "read":
        return result
    rng = np.random.default_rng(int(seed))
    correlations = []
    if strata is None:
        groups = [np.arange(xs.size)]
    else:
        strata = np.asarray(strata)
        if strata.size != xs.size:
            return {**result, "status": "not read", "rho": None,
                    "reason": "strata length does not match draw observations"}
        groups = [np.flatnonzero(strata == family) for family in dict.fromkeys(strata.tolist())]
        result["families_resampled_within"] = {
            str(family): int(np.count_nonzero(strata == family))
            for family in dict.fromkeys(strata.tolist())}
    for _ in range(int(reps)):
        sample = np.concatenate([rng.choice(group, size=group.size, replace=True) for group in groups])
        rho = _spearman_value(xs[sample], ys[sample])
        if rho is not None and math.isfinite(rho):
            correlations.append(rho)
    if not correlations:
        return {**result, "status": "not read", "rho": None,
                "reason": "no finite draw-bootstrap Spearman samples"}
    result["valid_replicates"] = len(correlations)
    result["interval_95"] = [float(np.percentile(correlations, 2.5)),
                             float(np.percentile(correlations, 97.5))]
    result["seed"] = int(seed)
    return result


def _spearman_value(xs, ys):
    xs = np.asarray(xs, dtype=np.float64).reshape(-1)
    ys = np.asarray(ys, dtype=np.float64).reshape(-1)
    if xs.size != ys.size or xs.size < 3 or not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        return None
    rx, ry = _rankdata(xs), _rankdata(ys)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if math.isfinite(value) else None


def correlation_table(draws, outcome_key="D", reps=REPS, seed=SEED,
                      projection_sd_60=None):
    output = {}
    for projection_index, projection in enumerate(("cf", "cr")):
        rows_by_family = {family: [row for row in draws if row["family"] == family]
                          for family in ("u100d", "f100d")}
        family_results = {}
        for family_index, family in enumerate(("u100d", "f100d")):
            rows = rows_by_family[family]
            try:
                if any(row.get(projection) is None or row.get(outcome_key) is None for row in rows):
                    raise ValueError("missing correlation input")
                xs = [float(row[projection]) for row in rows]
                ys = [float(row[outcome_key]) for row in rows]
            except (KeyError, TypeError, ValueError):
                family_results[family] = {"status": "not read", "rho": None,
                                          "interval_95": [None, None], "reason": "missing correlation input"}
                continue
            family_sd = None
            if projection_sd_60 is not None:
                family_sd = projection_sd_60.get(projection)
                try:
                    family_sd = float(family_sd)
                except (TypeError, ValueError):
                    family_sd = None
                if family_sd is None or not math.isfinite(family_sd):
                    family_results[family] = {
                        "status": "not read", "rho": None, "interval_95": [None, None],
                        "reason": "missing or nonfinite 60-draw projection standard deviation"}
                    continue
            family_results[family] = spearman_with_interval(
                xs, ys, reps=reps, seed=int(seed) + 100 * projection_index + family_index + 1,
                sd=family_sd)
        pooled_rows = [*rows_by_family["u100d"], *rows_by_family["f100d"]]
        try:
            if any(row.get(projection) is None or row.get(outcome_key) is None
                   for row in pooled_rows):
                raise ValueError("missing correlation input")
            xs = [float(row[projection]) for row in pooled_rows]
            ys = [float(row[outcome_key]) for row in pooled_rows]
            strata = [row["family"] for row in pooled_rows]
            pooled = spearman_with_interval(
                xs, ys, reps=reps, seed=int(seed) + 100 * projection_index + 3, strata=strata)
        except (KeyError, TypeError, ValueError):
            pooled = {"status": "not read", "rho": None,
                      "interval_95": [None, None], "reason": "missing correlation input"}
        output[projection] = {"pooled": pooled, **family_results}
    return output


def _class_counts(draws, family):
    counts = Counter(row["draw_class"] for row in draws if row["family"] == family)
    return {kind: int(counts.get(kind, 0)) for kind in ("neither", "one sign", "both same", "both different")}


def _family_distributions(scored, reader_by_row, directions, draws):
    output = {}
    for family in ("u100d", "f100d"):
        family_rows = [row for row in scored if row.get("set") in run174.SET_NAMES and
                       run174.FAMILY_BY_SET.get(str(row.get("set"))) == family]
        relation = Counter()
        mentions = Counter()
        primary = Counter()
        raw_hits = Counter()
        for row in family_rows:
            raw_hits.update({key: int(value) for key, value in row.get("raw_word_hits", {}).items()})
            if not row.get("clean"):
                continue
            record = reader_by_row.get((str(row.get("arm")), str(row.get("prompt_id"))))
            if record and record.get("relation") in ("claim", "none"):
                relation[record["relation"]] += 1
                entities = {normalize_name(name, mode="canonical") for name in record.get("names", [])}
                mentions.update(entity for entity in entities if entity is not None)
                entity = score173._primary_entity(record, mode="canonical")
                if entity is not None:
                    primary[entity] += 1
        output[family] = {"answers": len(family_rows), "relation_distribution": dict(sorted(relation.items())),
                          "primary_distribution": dict(sorted(primary.items())),
                          "mention_distribution": dict(sorted(mentions.items())),
                          "raw_word_hits": dict(sorted(raw_hits.items())),
                          "draw_class_distribution": _class_counts(draws, family)}
    return output


def _direction_floor_rates(directions):
    incomplete_directions = [row for row in directions if row.get("unlabelled_clean_answers", 0)]
    available_google = [row for row in directions if row["google_claim_count"] is not None]
    supported = [row for row in directions if row["supported"]]
    output = {"registered_174_1": {
        "k": len(supported), "n": len(directions),
        "rate": len(supported) / len(directions) if directions else None,
        "reader_incomplete_directions": len(incomplete_directions),
        "status": "incomplete" if incomplete_directions else "read",
    }}
    for floor in (5, 10, 15):
        modal_floor = [row for row in directions
                       if row["summary"]["modal_count"] >= floor and
                       row["summary"]["broken"] <= run174.DESIGN["cell"]["broken_max"]]
        google_floor = [row for row in available_google
                        if row["google_claim_count"] >= floor and
                        row["summary"]["broken"] <= run174.DESIGN["cell"]["broken_max"]]
        output[str(floor)] = {
            "modal_count_floor_supported": {
                "k": len(modal_floor), "n": len(directions),
                "rate": len(modal_floor) / len(directions) if directions else None,
                "reader_incomplete_directions": len(incomplete_directions),
                "status": "incomplete" if incomplete_directions else "read",
            },
            "google_claim_count_floor": {
                "k": len(google_floor), "n": len(directions),
                "rate": (len(google_floor) / len(directions)
                         if directions and not incomplete_directions else None),
                "labelled_directions": len(available_google),
                "unlabelled_directions": len(incomplete_directions),
                "labelled_direction_rate": (len(google_floor) / len(available_google)
                                             if available_google else None),
                "status": "not read" if incomplete_directions else "read",
            },
        }
    return output


def _other_led(directions):
    return [row for row in directions if row["supported"] and row["summary"]["modal"] != "google"]


def prediction_verdicts(directions, draws, reader_read, stage0, spearman):
    by_family = {name: [row for row in directions if row["family"] == name]
                 for name in ("u100d", "f100d")}
    u = by_family["u100d"]
    f = by_family["f100d"]
    u_supported = sum(row["supported"] for row in u)
    f_supported = sum(row["supported"] for row in f)
    total_supported = sum(row["supported"] for row in directions)
    stable_supported = sum(row["supported"] and row["stable"] for row in directions)
    u_other = _other_led(u)
    all_other = _other_led(directions)
    u_google = sum(row["supported"] and row["summary"]["modal"] == "google" for row in u)
    f_google = sum(row["supported"] and row["summary"]["modal"] == "google" for row in f)
    google_share = u_google / u_supported if u_supported else None
    f_google_share = f_google / f_supported if f_supported else None
    u_draws = [row for row in draws if row["family"] == "u100d"]
    qualifying = [row for row in u_draws if row["positive_supported"] or row["negative_supported"]]
    both_count = sum(row["draw_class"] in ("both same", "both different") for row in qualifying)
    u_rate = u_supported / 80
    f_rate = f_supported / 40
    stable_share = stable_supported / total_supported if total_supported else None
    known_other = sum(names173.is_known_entity(row["summary"]["modal"]) and
                      row["summary"]["modal"] != names173.QWEN_ENTITY for row in all_other)
    known_share = known_other / len(all_other) if all_other else None
    abs_difference = abs(f_rate - u_rate)
    p = run174.DESIGN["predictions"]

    def result(key, prediction, confidence, author, estimate, verdict, rule):
        return {"prediction": prediction, "confidence": confidence, "author": author,
                "estimate": estimate, "verdict": verdict, "registered_rule": rule}

    unavailable = "unread" if not reader_read else None
    labels_unavailable = "unread" if not reader_read else None
    values = {}
    values["174.1"] = result("174.1", "At least 20 of 80 u100d signed directions are supported",
                               "medium", "Claude", {"supported": int(u_supported), "total": 80},
                               unavailable or ("held" if u_supported >= 20 else "failed"),
                               "supported count >= 20")
    if unavailable:
        v2 = unavailable
    elif u_supported < p["174.2"]["not_read_below"]:
        v2 = "not read"
    else:
        v2 = "held" if google_share >= p["174.2"]["google_share_min"] else "failed"
    values["174.2"] = result("174.2", "Google is modal on at least 80% of supported u100d directions",
                               "medium", "Claude", {"google_led": int(u_google), "supported": int(u_supported),
                                                       "share": google_share}, v2,
                               "Google-led share >= 0.80; not read below 10 supported")
    values["174.3"] = result("174.3", "At most 4 of 80 u100d signed directions are other-led",
                               "medium", "Claude", {"other_led": len(u_other), "total": 80},
                               unavailable or ("held" if len(u_other) <= p["174.3"]["u100d_other_led_max"]
                                               else "failed"),
                               "other-led count <= 4")
    if unavailable:
        v4 = unavailable
    elif len(qualifying) < p["174.4"]["not_read_below_draws"]:
        v4 = "not read"
    else:
        v4 = "held" if both_count / len(qualifying) <= p["174.4"]["both_supported_share_max"] else "failed"
    values["174.4"] = result("174.4", "Among u100d draws with a supported sign, at most 25% are both same or both different",
                               "low", "Claude", {"qualifying_draws": len(qualifying), "both_supported": both_count,
                                                  "share": both_count / len(qualifying) if qualifying else None}, v4,
                               "both-same plus both-different share <= 0.25; not read below 8 qualifying draws")
    if unavailable:
        v5 = unavailable
    elif u_supported < p["174.5"]["not_read_below_u100d_supported"]:
        v5 = "uninformative"
    else:
        v5 = "held" if f_rate >= p["174.5"]["f100d_share_min_of_u100d"] * u_rate else "failed"
    values["174.5"] = result("174.5", "f100d supported share is at least half of u100d's",
                               "low", "Claude", {"f100d_supported": int(f_supported), "f100d_total": 40,
                                                  "f100d_share": f_rate, "u100d_supported": int(u_supported),
                                                  "u100d_total": 80, "u100d_share": u_rate},
                               v5,
                               "literal share comparison always reported; retention read only with at least 10 u100d supported")
    if unavailable:
        v10 = unavailable
    elif f_supported < p["174.10"]["not_read_below"]:
        v10 = "not read"
    else:
        v10 = "held" if f_google_share >= p["174.10"]["google_share_min"] else "failed"
    values["174.10"] = result("174.10", "Google is modal on at least 80% of supported f100d directions",
                                "medium", "Claude", {"google_led": int(f_google), "supported": int(f_supported),
                                                        "share": f_google_share}, v10,
                                "Google-led share >= 0.80; not read below 5 supported")
    for prediction, projection in (("174.6", "cf"), ("174.9", "cr")):
        test = spearman.get(projection, {})
        if "pooled" in test:
            test = test.get("pooled", {})
        if unavailable:
            test = {}
            verdict = unavailable
        elif test.get("status") == "not read":
            verdict = "not read"
        elif test.get("status") != "read":
            verdict = "unread"
        else:
            verdict = ("held" if abs(test["rho"]) <
                       p[prediction]["absolute_spearman_max"] else "failed")
        wording = ("Across 60 draws, |Spearman(cf, D)| is under 0.30" if projection == "cf" else
                   "Across 60 draws, |Spearman(cr, D)| is under 0.30")
        values[prediction] = result(prediction, wording,
                                    "medium" if projection == "cf" else "low", "Claude",
                                    {"rho": test.get("rho"), "interval_95": test.get("interval_95"),
                                     "p_value": test.get("p_value"), "n": test.get("n"),
                                     "standard_deviation": test.get("standard_deviation")},
                                    verdict, "absolute observed Spearman rho < 0.30; not read if group D has a repeated value on more than half its draws, projection sd < 0.01, or rho is nonfinite")
    broken = [row["summary"]["broken"] for row in u]
    median_broken = float(np.median(broken)) if broken else None
    values["174.7"] = result("174.7", "Median broken of the 80 u100d signed directions is at most 8 of 44",
                               "medium", "Claude", {"median_broken": median_broken, "directions": len(broken)},
                               "held" if median_broken is not None and median_broken <= 8 else "failed",
                               "median broken <= 8")
    if unavailable:
        v8 = unavailable
    elif total_supported < p["174.8"]["not_read_below_supported"]:
        v8 = "not read"
    else:
        v8 = "held" if stable_share >= p["174.8"]["stable_share_min"] else "failed"
    values["174.8"] = result("174.8", "At least 60% of all supported directions are stable",
                               "medium", "Claude", {"stable_supported": int(stable_supported),
                                                     "supported": int(total_supported),
                                                     "share": stable_share}, v8,
                               "stable share >= 0.60; not read below 5 supported")

    if labels_unavailable:
        w2 = labels_unavailable
    elif len(all_other) == 0:
        w2 = "not testable"
    elif known_share is None or known_share >= p["174.W2"]["known_non_qwen_falsifier_share"]:
        w2 = "failed"
    else:
        w2 = "held"
    values["174.W1"] = result("174.W1", "At least 30 of the 80 u100d signed directions are supported; falsifier: fewer than 20",
                                "medium", "Weaver", {"supported": int(u_supported), "total": 80},
                                labels_unavailable or ("failed" if u_supported < 20 else
                                                       "held" if u_supported >= 30 else "inconclusive"),
                                "held >= 30; falsified below 20; otherwise inconclusive")
    values["174.W2"] = result("174.W2", "Other-led directions are present, and a majority name invented entities",
                                "medium", "Weaver", {"other_led": len(all_other), "known_non_qwen": int(known_other),
                                                     "invented": len(all_other) - int(known_other),
                                                     "known_share": known_share,
                                                     "falsifier_met": bool(
                                                         known_share is not None and
                                                         known_share >= p["174.W2"]["known_non_qwen_falsifier_share"])},
                                w2, "not testable with no other-led direction; held iff a strict majority are invented; failed at least half known non-Qwen")
    if labels_unavailable:
        w3 = labels_unavailable
    elif u_rate == 0:
        w3 = "not testable"
    elif f_rate < p["174.W3"]["falsifier_f_below_u_fraction"] * u_rate:
        w3 = "failed"
    elif abs_difference <= p["174.W3"]["share_absolute_difference_max"] + 1e-12:
        w3 = "held"
    else:
        w3 = "inconclusive"
    values["174.W3"] = result("174.W3", "f100d supported share is within 20 percentage points of u100d's; falsifier: f100d below half",
                                "medium", "Weaver", {"u100d_share": u_rate, "f100d_share": f_rate,
                                                     "absolute_difference": abs_difference,
                                                     "falsifier_met": f_rate <
                                                     p["174.W3"]["falsifier_f_below_u_fraction"] * u_rate},
                                w3, "u100d share zero is not testable; below half falsifies before the 0.20-point hold bar")
    return values


def fallback_reading(predictions, reader_read, generation_readable):
    prerequisites = {key: predictions.get(key, {}).get("verdict") for key in ("174.2", "174.10")}
    if not reader_read or not generation_readable:
        return {"status": "unread", "verdict": "unread", "prerequisites": prerequisites,
                "rule": "read only when 174.2 and 174.10 are both read and both hold"}
    if all(value == "held" for value in prerequisites.values()):
        return {"status": "read", "verdict": "held", "prerequisites": prerequisites,
                "rule": "read only when 174.2 and 174.10 are both read and both hold"}
    return {"status": "not read", "verdict": "not read", "prerequisites": prerequisites,
            "rule": "read only when 174.2 and 174.10 are both read and both hold"}


def score(answers_path="results/part174/answers.jsonl", gates_path="results/part174/gates.json",
          labels_dir="results/part174/labels", out_dir="results/part174", reps=REPS, seed=SEED):
    answers_path = Path(answers_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_answers = read_jsonl(answers_path)
    scored = score173.score_rows(raw_answers)
    ds_rows = read_jsonl(Path(labels_dir) / "labels-ds.jsonl") if (Path(labels_dir) / "labels-ds.jsonl").is_file() else []
    reader_by_row = _reader_map(scored, ds_rows)
    stage0 = read_json(out / "stage0.json", {}) or {}
    gate_file = read_json(gates_path, {}) or {}
    reader_gate = _reader_gate(answers_path, raw_answers, labels_dir)
    reader_read = bool(reader_gate["passed"])
    required_gates = [f"gate{index}" for index in range(1, 9)]
    generation_readable = bool(gate_file.get("passed") and all(
        gate_file.get(name, {}).get("passed") for name in required_gates))
    directions = _direction_summaries(scored, reader_by_row)
    rows_by_arm = _arm_row_map(scored)
    draws = _draw_summaries(directions)
    metrics = stage0.get("random_set_metrics", {})
    for row in draws:
        row["cf"] = metrics.get(row["set"], {}).get("cf")
        row["cr"] = metrics.get(row["set"], {}).get("cr")
    if reader_read and generation_readable:
        projection_stats = stage0.get("projection_standard_deviations", {}) or {}
        spearman = correlation_table(
            draws, outcome_key="D", reps=int(reps), seed=int(seed),
            projection_sd_60={"cf": projection_stats.get("cf_sd"),
                              "cr": projection_stats.get("cr_sd")})
    else:
        spearman = {projection: {group: {"status": "unread", "rho": None,
                                         "interval_95": [None, None],
                                         "reason": "reader gate or generation gate did not pass"}
                                 for group in ("pooled", "u100d", "f100d")}
                    for projection in ("cf", "cr")}
    raw_word_hit_correlations = correlation_table(
        draws, outcome_key="raw_google_word_hit_difference", reps=int(reps), seed=int(seed) + 1000)
    # Allow small replicate counts in fixture tests while the registered production run stays at 10,000.
    uncertainty = rate_uncertainty(directions, rows_by_arm, reader_by_row,
                                   reps=int(reps), seed=int(seed))

    by_family = {family: [row for row in directions if row["family"] == family]
                 for family in ("u100d", "f100d")}
    google_hist = Counter(row["google_claim_count"] for row in directions
                          if row["google_claim_count"] is not None)
    unavailable_google_claim_directions = sum(row["google_claim_count"] is None for row in directions)
    raw_google_hist = Counter(int(row["summary"]["raw_word_hits"].get("google", 0)) for row in directions)
    supported_count = sum(row["supported"] for row in directions)
    stable_count = sum(row["supported"] and row["stable"] for row in directions)
    u_other = _other_led(by_family["u100d"])
    family_rate_counts = {
        family: {
            "status": "read" if reader_read and generation_readable else "unread",
            "signed_directions": len(rows),
            "supported": int(sum(row["supported"] for row in rows)),
            "google_led": int(sum(row["supported"] and row["summary"]["modal"] == "google"
                                   for row in rows)),
            "other_led": int(sum(row["supported"] and row["summary"]["modal"] != "google"
                                  for row in rows)),
        }
        for family, rows in by_family.items()
    }
    predictions = prediction_verdicts(directions, draws, reader_read, stage0, spearman)
    if not generation_readable:
        for value in predictions.values():
            value["verdict"] = "unread"
            value["read_block"] = "a generation gate failed or was missing"
    separate_fallback = fallback_reading(predictions, reader_read, generation_readable)
    summary = {
        "part": 174, "prediction_read_status": "read" if reader_read and generation_readable else "unread",
        "reader_gate9": reader_gate, "generation_gates": gate_file,
        "counts": {"answers": len(raw_answers), "scored_non_replay_answers": len(scored),
                   "directions": len(directions), "u100d_signed": len(by_family["u100d"]),
                   "f100d_signed": len(by_family["f100d"]),
                   "supported": supported_count, "stable_supported": stable_count,
                   "other_led_u100d": len(u_other),
                   "families": family_rate_counts},
        "direction_classes": {family: _class_counts(draws, family) for family in ("u100d", "f100d")},
        "google_claim_count_histogram_over_120_directions": {
            "status": "read" if unavailable_google_claim_directions == 0 else "unread",
            "directions_with_reader_count": 120 - unavailable_google_claim_directions,
            "directions_without_reader_count": unavailable_google_claim_directions,
            "counts": {str(key): int(value) for key, value in sorted(google_hist.items())}},
        "raw_google_word_hit_histogram_over_120_directions": {
            "label": "descriptive fallback; word hits are not reader-extracted claims",
            "counts": {str(key): int(value) for key, value in sorted(raw_google_hist.items())}},
        "support_floor_sensitivity": {
            "note": ("174.1 and floors 5, 10, and 15 are shown over each family's signed directions. "
                     "Incomplete reader coverage is flagged and never treated as zero Google claims."),
            "families": {family: _direction_floor_rates(rows)
                         for family, rows in by_family.items()}},
        "families": _family_distributions(scored, reader_by_row, directions, draws),
        "family_rate_counts": family_rate_counts,
        "direction_rate_uncertainty": uncertainty,
        "draw_sign_contrasts": draws,
        "spearman_tests": spearman,
        "descriptive_fallback": {
            "label": "raw word hits are descriptive; they are not reader-extracted claims or prediction endpoints",
            "raw_word_hit_correlations": raw_word_hit_correlations,
        },
        "predictions": predictions,
        "fallback_reading": separate_fallback,
        "eligibility_screen": {
            **(stage0.get("eligibility_screen", {}) or {}),
            "length_relative_tolerance": ((stage0.get("gate8") or {}).get(
                "length_relative_tolerance")),
            "readings_conditional_on_all_60_draws_passing": bool(
                (stage0.get("gate8") or {}).get("passed") and
                (stage0.get("eligibility_screen") or {}).get(
                    "readings_conditional_on_all_60_draws_passing")),
            "condition": "Reader-based readings are conditional on all 60 planned draws passing Gate 8's stated |cf| <= 0.079 eligibility screen and length tolerance.",
        },
        "scope": "This census describes these two dense draw generators, this model snapshot, dose, and NEW44B prompts.",
        "bootstrap_note": "Draw intervals quantify direction uncertainty; question-level intervals hold directions fixed.",
        "fallback": ("Reader gate and error cap passed; reader-extracted entities may score the predictions." if reader_read else
                     "Reader gate failed, was unread, or exceeded the error cap; raw word hits remain descriptive only."),
    }
    (out / "scored_directions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n" for row in directions),
        encoding="utf-8", newline="\n")
    (out / "score.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2,
                                                  allow_nan=False) + "\n",
                                     encoding="utf-8", newline="\n")
    (out / "prediction_table.json").write_text(json.dumps(summary["predictions"], ensure_ascii=True,
                                                            indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8", newline="\n")
    print(json.dumps({"prediction_read_status": summary["prediction_read_status"],
                      "predictions": {key: value["verdict"] for key, value in summary["predictions"].items()},
                      "directions": summary["counts"]["directions"]}, ensure_ascii=True,
                     allow_nan=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="results/part174/answers.jsonl")
    parser.add_argument("--gates", default="results/part174/gates.json")
    parser.add_argument("--labels-dir", default="results/part174/labels")
    parser.add_argument("--out-dir", default="results/part174")
    parser.add_argument("--replicates", type=int, default=REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    summary = score(args.answers, args.gates, args.labels_dir, args.out_dir,
                    args.replicates, args.seed)
    generation_gates = summary.get("generation_gates", {}) or {}
    generation_passed = bool(generation_gates.get("passed") and all(
        generation_gates.get(f"gate{index}", {}).get("passed") for index in range(1, 9)))
    reader_passed = bool((summary.get("reader_gate9") or {}).get("passed"))
    prediction_unread = any(value.get("verdict") == "unread"
                            for value in (summary.get("predictions") or {}).values())
    return 0 if (summary.get("prediction_read_status") == "read" and
                 generation_passed and reader_passed and not prediction_unread) else 1


if __name__ == "__main__":
    raise SystemExit(main())
