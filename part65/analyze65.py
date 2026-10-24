#!/usr/bin/env python3
"""Analyze the registered Part 65 predictions 65a-65g.
Reads blinded judge answers, Part 64 Haiku-1x hypotheses and sentence-test
pass/fail rows, plus Part 65 write summaries. Writes analysis-65.json and a
human-readable analysis-65.md. All uncertainty intervals resample units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

from rank65 import layer_band, words


BOOTSTRAP_REPS = 5000
SHUFFLE_REPS = 1000
EMBEDDING_DIMENSIONS = 512
TRIALS_PER_UNIT = 2


class AnalysisError(RuntimeError):
    """Incomplete or inconsistent Part 65 analysis inputs."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise AnalysisError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def read_labels(neuron_path: Path, control_path: Path) -> dict[str, str]:
    from judge65 import read_label_rows
    rows = read_label_rows(neuron_path, control_path)
    return {key: row["read_label"] for key, row in rows.items()}


def read_write_words(path: Path) -> dict[str, list[str]]:
    output = {}
    for row in read_jsonl(path):
        key, values = row.get("unit_key"), row.get("write_words")
        if not isinstance(key, str) or key in output or not isinstance(values, list):
            raise AnalysisError(f"{path}: invalid or duplicate write summary")
        output[key] = [str(value) for value in values]
    return output


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AnalysisError("cannot take a percentile of an empty sample")
    position = (len(ordered) - 1) * probability
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_mean(values: list[float], *, seed: int = 6501,
                   reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    if not values:
        raise AnalysisError("bootstrap group is empty")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return {"estimate": sum(values) / n,
            "ci95": [_percentile(means, 0.025), _percentile(means, 0.975)],
            "n": n}


def bootstrap_difference(left: list[float], right: list[float], *,
                         seed: int = 6501, reps: int = BOOTSTRAP_REPS) -> dict[str, Any]:
    if not left or not right:
        raise AnalysisError("bootstrap difference requires two nonempty groups")
    rng = random.Random(seed)
    ln, rn = len(left), len(right)
    diffs = []
    for _ in range(reps):
        a = sum(left[rng.randrange(ln)] for _ in range(ln)) / ln
        b = sum(right[rng.randrange(rn)] for _ in range(rn)) / rn
        diffs.append(a - b)
    return {"estimate": sum(left) / ln - sum(right) / rn,
            "ci95": [_percentile(diffs, 0.025), _percentile(diffs, 0.975)],
            "n_left": ln, "n_right": rn}


def _hashed_embedding(text: str, dimensions: int = EMBEDDING_DIMENSIONS) -> dict[int, float]:
    tokens = words(text)
    features = tokens + [tokens[i] + "\0" + tokens[i + 1] for i in range(len(tokens) - 1)]
    vector: dict[int, float] = {}
    for feature in features:
        raw = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(raw, "little")
        index = value % dimensions
        sign = 1.0 if (value >> 63) == 0 else -1.0
        vector[index] = vector.get(index, 0.0) + sign
    return vector


def _vector_cosine(a: dict[int, float], b: dict[int, float],
                   norm_a: float | None = None, norm_b: float | None = None) -> float:
    if norm_a is None:
        norm_a = math.sqrt(sum(value * value for value in a.values()))
    if norm_b is None:
        norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return sum(value * b.get(index, 0.0) for index, value in a.items()) / (norm_a * norm_b)


def cosine(left: str, right: str) -> float:
    return _vector_cosine(_hashed_embedding(left), _hashed_embedding(right))


def _derangement(size: int, rng: random.Random) -> list[int]:
    if size < 2:
        raise AnalysisError("a shuffled-pair null needs at least two units per stratum")
    values = list(range(size))
    for _ in range(100):
        rng.shuffle(values)
        if all(index != values[index] for index in range(size)):
            return values
    # A cyclic shift of a shuffled ordering is a guaranteed no-self mapping.
    offset = rng.randrange(1, size)
    return [(index + offset) % size for index in range(size)]


def shuffled_null_cosines(records: list[dict[str, Any]], *, seed: int = 6501,
                          reps: int = SHUFFLE_REPS) -> dict[str, float]:
    """Mean unit-level read/write cosine over within-type, within-band derangements."""
    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(records):
        groups.setdefault((row["type"], row["layer_band"]), []).append(index)
    sums = [0.0] * len(records)
    counts = [0] * len(records)
    read_vectors = [_hashed_embedding(row["read_label"]) for row in records]
    write_vectors = [_hashed_embedding(row["write_text"]) for row in records]
    read_norms = [math.sqrt(sum(value * value for value in vector.values())) for vector in read_vectors]
    write_norms = [math.sqrt(sum(value * value for value in vector.values())) for vector in write_vectors]
    rng = random.Random(seed)
    for indices in groups.values():
        if len(indices) < 2:
            raise AnalysisError("a shuffled-pair null needs at least two units per type and band")
        for _ in range(reps):
            mapping = _derangement(len(indices), rng)
            for position, left_index in enumerate(indices):
                right_index = indices[mapping[position]]
                value = _vector_cosine(read_vectors[left_index], write_vectors[right_index],
                                       read_norms[left_index], write_norms[right_index])
                sums[left_index] += value
                counts[left_index] += 1
    return {row["unit_key"]: sums[index] / counts[index]
            for index, row in enumerate(records)}


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _missing_trial_scenario(unit_map: dict[str, dict[str, Any]],
                            answers_by_pair: dict[tuple[str, int], dict[str, Any]],
                            missing_score: float) -> dict[str, Any]:
    """Score every registered unit after assigning one score to each missing trial."""
    scores: dict[str, float] = {}
    for key in unit_map:
        trial_scores = []
        for trial in (1, 2):
            answer = answers_by_pair.get((key, trial))
            correct = answer.get("correct") if answer is not None else None
            trial_scores.append(float(correct) if correct is not None else missing_score)
        scores[key] = sum(trial_scores) / TRIALS_PER_UNIT

    neurons = [scores[key] for key, unit in unit_map.items() if unit["type"] == "neuron"]
    controls = [scores[key] for key, unit in unit_map.items() if unit["type"] == "control"]
    if not neurons or not controls:
        raise AnalysisError("missing-trial bounds need registered neurons and controls")
    bands = {
        band: [scores[key] for key, unit in unit_map.items()
               if unit["type"] == "neuron" and layer_band(int(unit["layer"])) == band]
        for band in ("first", "middle", "last")
    }
    if any(not values for values in bands.values()):
        raise AnalysisError("missing-trial bounds need neurons in all three layer bands")

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    return {
        "missing_trial_score": missing_score,
        "65a": {"estimate": mean(neurons), "n": len(neurons)},
        "65b": {
            "estimate": mean(neurons) - mean(controls),
            "n_neuron": len(neurons),
            "n_control": len(controls),
        },
        "neuron_agreement_by_layer_band": {
            band: {"estimate": mean(values), "n": len(values)}
            for band, values in bands.items()
        },
    }


def analyze_rows(answer_rows: list[dict[str, Any]], units: list[dict[str, Any]],
                 read_labels: dict[str, str], write_words: dict[str, list[str]], *,
                 seed: int = 6501, bootstrap_reps: int = BOOTSTRAP_REPS,
                 shuffle_reps: int = SHUFFLE_REPS) -> dict[str, Any]:
    unit_map = {row["unit_key"]: row for row in units}
    expected_keys = set(unit_map)
    if len(unit_map) != len(units):
        raise AnalysisError("collection includes duplicate unit keys")
    if set(read_labels) != expected_keys or set(write_words) != expected_keys:
        raise AnalysisError("read labels and write words must cover every registered unit")
    if answer_rows:
        run_plan_pairs = set()
        run_roots = set()
        for row_number, answer in enumerate(answer_rows, 1):
            judge_run_id = answer.get("judge_run_id")
            plan_sha256 = answer.get("plan_sha256")
            run_root = answer.get("run_root")
            if not all(isinstance(value, str) and value.strip()
                       for value in (judge_run_id, plan_sha256, run_root)):
                raise AnalysisError(
                    f"judge answer row {row_number} lacks run ID, plan hash, or output directory"
                )
            run_plan_pairs.add((judge_run_id, plan_sha256))
            run_roots.add(run_root)
        if len(run_plan_pairs) != 1 or len(run_roots) != 1:
            raise AnalysisError("judge answers combine more than one run or plan")
    # judge65.build_trials registers two numbered attempts for every frozen unit.
    # Build the expected pairs from that plan shape, never from answers.jsonl.
    registered_pairs = {
        (key, trial)
        for key in unit_map
        for trial in range(1, TRIALS_PER_UNIT + 1)
    }
    total_trial_count = len(registered_pairs)
    answers_by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    for answer in answer_rows:
        key = answer.get("unit_key")
        if key not in expected_keys:
            raise AnalysisError(f"judge answer has unknown unit {key!r}")
        trial = answer.get("trial")
        if isinstance(trial, bool) or not isinstance(trial, int):
            raise AnalysisError(f"{key}: judge trial number must be 1 or 2")
        pair = (key, trial)
        if pair not in registered_pairs:
            raise AnalysisError(f"{key}: judge answer has unregistered trial {trial!r}")
        if pair in answers_by_pair:
            raise AnalysisError(f"{key}: duplicate judge answer for trial {trial}")
        correct = answer.get("correct")
        if correct is not None and not isinstance(correct, bool):
            raise AnalysisError(f"{key}: judge correct field must be true, false, or null")
        answers_by_pair[pair] = answer

    absent_trial_count = len(registered_pairs - set(answers_by_pair))
    unparseable_trial_count = sum(answer.get("correct") is None
                                  for answer in answers_by_pair.values())
    missing_trial_count = absent_trial_count + unparseable_trial_count
    if missing_trial_count * 20 > total_trial_count:
        raise AnalysisError(
            f"analysis stopped: {missing_trial_count}/{total_trial_count} registered judge trials "
            "are missing (above the registered 5% limit)"
        )

    records = []
    units_without_valid_trial = 0
    excluded_units_by_type_and_layer_band = {
        kind: {band: 0 for band in ("first", "middle", "last")}
        for kind in ("neuron", "control")
    }
    excluded_units_by_missing_trial_count = {"one": 0, "two": 0}
    for key, unit in unit_map.items():
        rows = [answers_by_pair.get((key, trial))
                for trial in range(1, TRIALS_PER_UNIT + 1)]
        present_rows = [row for row in rows if row is not None]
        if any(row.get("type") != unit["type"]
               or int(row.get("layer", -1)) != int(unit["layer"])
               for row in present_rows):
            raise AnalysisError(f"{key}: judge metadata disagrees with the frozen collection")
        foils = [row.get("foil_unit_key") for row in present_rows]
        if (len(set(foils)) != len(foils) or key in foils
                or any(foil not in unit_map for foil in foils)):
            raise AnalysisError(f"{key}: foil assignment must name distinct other units")
        band = layer_band(int(unit["layer"]))
        if any(unit_map[foil]["type"] != unit["type"]
               or layer_band(int(unit_map[foil]["layer"])) != band for foil in foils):
            raise AnalysisError(f"{key}: foil type or layer band does not match")
        for trial, row in enumerate(rows, 1):
            if row is not None and row.get("own_position") != ("A" if trial == 1 else "B"):
                raise AnalysisError(f"{key}: trial order was not swapped A then B")
        sentence_values = {row.get("sentence_test_pass") for row in present_rows}
        if (len(sentence_values) > 1
                or any(value not in (True, False, None) for value in sentence_values)):
            raise AnalysisError(f"{key}: sentence-test status differs across trials")
        valid_rows = [row for row in present_rows if row.get("correct") is not None]
        missing_for_unit = TRIALS_PER_UNIT - len(valid_rows)
        if missing_for_unit:
            excluded_units_by_type_and_layer_band[unit["type"]][band] += 1
            excluded_units_by_missing_trial_count["one" if missing_for_unit == 1 else "two"] += 1
        if not valid_rows:
            units_without_valid_trial += 1
            continue
        if missing_for_unit:
            continue
        agreement = sum(bool(row["correct"]) for row in valid_rows) / TRIALS_PER_UNIT
        records.append({
            "unit_key": key,
            "type": unit["type"],
            "layer": int(unit["layer"]),
            "layer_band": band,
            "agreement": agreement,
            "valid_trial_count": TRIALS_PER_UNIT,
            "missing_trial_count": 0,
            "sentence_test_pass": next(iter(sentence_values)),
            "read_label": read_labels[key],
            "write_text": " ".join(write_words[key]),
        })

    neurons = [row for row in records if row["type"] == "neuron"]
    controls = [row for row in records if row["type"] == "control"]
    if not neurons or not controls:
        raise AnalysisError(f"analysis needs valid trials for both types, found {len(neurons)} neurons and {len(controls)} controls")
    by_band = {
        band: [row for row in neurons if row["layer_band"] == band]
        for band in ("first", "middle", "last")
    }
    if any(not by_band[band] for band in ("first", "middle", "last")):
        raise AnalysisError("65c/65g need valid real-neuron trials in all three layer bands")
    pass_rows = [row for row in neurons if row["sentence_test_pass"] is True]
    fail_rows = [row for row in neurons if row["sentence_test_pass"] is False]
    if not pass_rows or not fail_rows:
        raise AnalysisError("65d needs both Part 64 sentence-test pass and fail neurons")

    missing_trial_bounds = {
        "missing_trials_scored_0_5": _missing_trial_scenario(
            unit_map, answers_by_pair, 0.5),
        "missing_trials_scored_wrong": _missing_trial_scenario(
            unit_map, answers_by_pair, 0.0),
    }

    actual_cosines = {row["unit_key"]: cosine(row["read_label"], row["write_text"])
                      for row in records}
    null_cosines = shuffled_null_cosines(records, seed=seed, reps=shuffle_reps)
    for row in records:
        row["embedding_cosine"] = actual_cosines[row["unit_key"]]
        row["shuffled_cosine"] = null_cosines[row["unit_key"]]
        row["embedding_above_shuffled"] = row["embedding_cosine"] - row["shuffled_cosine"]

    real_agreement = [row["agreement"] for row in neurons]
    control_agreement = [row["agreement"] for row in controls]
    last_agreement = [row["agreement"] for row in by_band["last"]]
    first_agreement = [row["agreement"] for row in by_band["first"]]
    pass_agreement = [row["agreement"] for row in pass_rows]
    fail_agreement = [row["agreement"] for row in fail_rows]
    real_cosine = [row["embedding_cosine"] for row in neurons]
    control_cosine = [row["embedding_cosine"] for row in controls]
    real_cosine_lift = [row["embedding_above_shuffled"] for row in neurons]
    control_cosine_lift = [row["embedding_above_shuffled"] for row in controls]
    band_agreement = {
        band: bootstrap_mean([row["agreement"] for row in by_band[band]],
                             seed=seed + offset, reps=bootstrap_reps)
        for band, offset in (("first", 10), ("middle", 11), ("last", 12))
    }

    real = bootstrap_mean(real_agreement, seed=seed, reps=bootstrap_reps)
    control = bootstrap_mean(control_agreement, seed=seed + 1, reps=bootstrap_reps)
    # Amendment 3 sensitivity (reported, not pass/fail): relative doses may not transfer to layers 0-2 (Lyra).
    late_real = [row["agreement"] for row in neurons if row["layer"] > 2]
    late_control = [row["agreement"] for row in controls if row["layer"] > 2]
    if not late_real or not late_control:
        raise AnalysisError("the layers 0-2 sensitivity needs units above layer 2 of both types")
    layers_0_2_left_out = {
        "neuron_count": len(late_real),
        "control_count": len(late_control),
        "65a": bootstrap_mean(late_real, seed=seed + 20, reps=bootstrap_reps),
        "65b": bootstrap_difference(late_real, late_control, seed=seed + 21, reps=bootstrap_reps),
    }
    type_difference = bootstrap_difference(real_agreement, control_agreement,
                                           seed=seed + 2, reps=bootstrap_reps)
    depth_difference = bootstrap_difference(last_agreement, first_agreement,
                                            seed=seed + 3, reps=bootstrap_reps)
    sentence_difference = bootstrap_difference(pass_agreement, fail_agreement,
                                               seed=seed + 4, reps=bootstrap_reps)
    real_embedding = bootstrap_mean(real_cosine, seed=seed + 5, reps=bootstrap_reps)
    control_embedding = bootstrap_mean(control_cosine, seed=seed + 6, reps=bootstrap_reps)
    embedding_difference = bootstrap_difference(real_cosine, control_cosine,
                                                  seed=seed + 7, reps=bootstrap_reps)
    real_null_lift = bootstrap_mean(real_cosine_lift, seed=seed + 8, reps=bootstrap_reps)
    control_null_lift = bootstrap_mean(control_cosine_lift, seed=seed + 9, reps=bootstrap_reps)
    null_adjusted_embedding_difference = bootstrap_difference(
        real_cosine_lift, control_cosine_lift, seed=seed + 14, reps=bootstrap_reps
    )

    predictions = {
        "65a": {
            "registered": "Real neurons: blind-matching agreement at least 60%, with the 95% unit-bootstrap interval above 50%.",
            "agreement": real,
            "pass": real["estimate"] >= 0.60 and real["ci95"][0] > 0.50,
        },
        "65b": {
            "registered": "Real neurons beat random directions by at least 8 percentage points.",
            "agreement_difference_neuron_minus_control": type_difference,
            "pass": type_difference["estimate"] >= 0.08,
        },
        "65c": {
            "registered": "Among real neurons, the last layer third agrees at least 10 points more than the first third.",
            "exploratory": True,
            "agreement_difference_last_minus_first": depth_difference,
            "pass": depth_difference["estimate"] >= 0.10,
        },
        "65d": {
            "registered": "Real neurons that passed their Part 64 Haiku-1x sentence test agree at least 10 points more than failures.",
            "agreement_difference_pass_minus_fail": sentence_difference,
            "pass": sentence_difference["estimate"] >= 0.10,
        },
        "65e": {
            "registered": "Random directions agree above 50%, with the lower 95% unit-bootstrap bound above 50%.",
            "agreement": control,
            "pass": control["estimate"] > 0.50 and control["ci95"][0] > 0.50,
        },
        "65f": {
            "registered": "The embedding measure ranks real above control in the same direction as the primary, with no threshold.",
            "measure_used": "Registered paired difference in observed read/write cosine (neuron mean minus control mean).",
            "cosine_difference_neuron_minus_control": embedding_difference,
            "null_adjusted_cosine_difference_neuron_minus_control": null_adjusted_embedding_difference,
            "primary_agreement_difference_neuron_minus_control": type_difference["estimate"],
            "cosine_difference_above_shuffled_neuron": real_null_lift,
            "cosine_difference_above_shuffled_control": control_null_lift,
            "pass": (_sign(embedding_difference["estimate"]) ==
                     _sign(type_difference["estimate"]) and type_difference["estimate"] != 0),
        },
        "65g": {
            "registered": "Among real neurons, the middle third agrees less than both the first and last thirds by point estimate.",
            "exploratory": True,
            "band_agreement": band_agreement,
            "pass": (band_agreement["middle"]["estimate"] < band_agreement["first"]["estimate"]
                     and band_agreement["middle"]["estimate"] < band_agreement["last"]["estimate"]),
        },
    }
    return {
        "schema": "part65-analysis-v1",
        "seed": seed,
        "bootstrap_replicates": bootstrap_reps,
        "shuffled_pair_replicates": shuffle_reps,
        "trial_count": total_trial_count,
        "missing_trial_count": missing_trial_count,
        "absent_trial_count": absent_trial_count,
        "unparseable_trial_count": unparseable_trial_count,
        "unit_without_valid_trial_count": units_without_valid_trial,
        "excluded_unit_count": sum(sum(bands.values())
                                    for bands in excluded_units_by_type_and_layer_band.values()),
        "excluded_units_by_type_and_layer_band": excluded_units_by_type_and_layer_band,
        "excluded_units_by_missing_trial_count": excluded_units_by_missing_trial_count,
        "missing_trial_bounds": missing_trial_bounds,
        "sensitivity_layers_0_2_left_out": layers_0_2_left_out,
        "exploratory_predictions": ["65c", "65g"],
        "embedding": {
            "method": "512-dimensional signed Blake2b feature hashing of word unigrams and adjacent word bigrams; cosine",
            "null": "within-type, within-layer-band write-summary derangement",
            "65f_measure_used": predictions["65f"]["measure_used"],
            "65f_null_adjusted_measure": "Neurons minus controls in unit-level observed cosine minus shuffled-pair null cosine.",
        },
        "unit_count": len(records),
        "registered_unit_count": len(units),
        "unit_agreement": {
            "neuron": real,
            "control": control,
            "neuron_minus_control": type_difference,
            "first_third_neuron": band_agreement["first"],
            "middle_third_neuron": band_agreement["middle"],
            "last_third_neuron": band_agreement["last"],
            "sentence_pass_neuron": bootstrap_mean(pass_agreement, seed=seed + 12, reps=bootstrap_reps),
            "sentence_fail_neuron": bootstrap_mean(fail_agreement, seed=seed + 13, reps=bootstrap_reps),
        },
        "embedding_cosine": {
            "neuron": real_embedding,
            "control": control_embedding,
            "neuron_minus_control": embedding_difference,
            "null_adjusted_neuron_minus_control": null_adjusted_embedding_difference,
            "neuron_minus_shuffled_null": real_null_lift,
            "control_minus_shuffled_null": control_null_lift,
        },
        "predictions": predictions,
        "units": records,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Part 65 analysis",
        "",
        f"Units analyzed: {result['unit_count']}; bootstrap replicates: {result['bootstrap_replicates']}; "
        f"shuffled-pair replicates: {result['shuffled_pair_replicates']}; "
        f"missing trials: {result['missing_trial_count']}/{result['trial_count']} "
        f"(absent: {result['absent_trial_count']}, unparseable: {result['unparseable_trial_count']}); "
        f"units excluded for any missing trial: {result['excluded_unit_count']}.",
        "",
        "| Prediction | Estimate | 95% unit-bootstrap interval | Pass |",
        "|---|---:|---:|:---:|",
    ]
    predictions = result["predictions"]
    embedding_difference = predictions["65f"]["cosine_difference_neuron_minus_control"]
    rows = (
        ("65a neuron agreement", predictions["65a"]["agreement"]),
        ("65b neuron minus control", predictions["65b"]["agreement_difference_neuron_minus_control"]),
        ("65c last minus first third", predictions["65c"]["agreement_difference_last_minus_first"]),
        ("65d sentence pass minus fail", predictions["65d"]["agreement_difference_pass_minus_fail"]),
        ("65e control agreement", predictions["65e"]["agreement"]),
        ("65f cosine neuron minus control", predictions["65f"]["cosine_difference_neuron_minus_control"]),
    )
    for name, metric in rows:
        estimate = metric["estimate"]
        low, high = metric["ci95"]
        passing = predictions[name[:3]]["pass"]
        lines.append(f"| {name} | {estimate:.4f} | [{low:.4f}, {high:.4f}] | {str(passing).lower()} |")
    lines.extend([
        "",
        "65c and 65g are exploratory depth predictions.",
        "",
        "## 65g exploratory depth agreement",
        "",
        f"Middle below both first and last by point estimate: {str(predictions['65g']['pass']).lower()}.",
        "",
        "| Band | Estimate | 95% unit-bootstrap interval | n |",
        "|---|---:|---:|---:|",
    ])
    for band in ("first", "middle", "last"):
        metric = predictions["65g"]["band_agreement"][band]
        lines.append(f"| {band} | {metric['estimate']:.4f} | [{metric['ci95'][0]:.4f}, {metric['ci95'][1]:.4f}] | {metric['n']} |")
    null_adjusted = predictions["65f"]["null_adjusted_cosine_difference_neuron_minus_control"]
    lines.extend([
        "",
        "## 65f embedding measure",
        "",
        predictions["65f"]["measure_used"],
        f"Registered observed cosine difference: {embedding_difference['estimate']:.4f} "
        f"[{embedding_difference['ci95'][0]:.4f}, {embedding_difference['ci95'][1]:.4f}].",
        f"Null-adjusted neuron-minus-control difference: {null_adjusted['estimate']:.4f} "
        f"[{null_adjusted['ci95'][0]:.4f}, {null_adjusted['ci95'][1]:.4f}].",
        "",
        "## Embedding and shuffled-pair null",
        "",
        result["embedding"]["method"] + ".",
        result["embedding"]["null"] + ".",
        "",
        "| Type | Paired cosine | Paired minus shuffled null |",
        "|---|---:|---:|",
    ])
    for label, actual, lift in (
        ("Neuron", result["embedding_cosine"]["neuron"],
         result["embedding_cosine"]["neuron_minus_shuffled_null"]),
        ("Control", result["embedding_cosine"]["control"],
         result["embedding_cosine"]["control_minus_shuffled_null"]),
    ):
        lines.append(f"| {label} | {actual['estimate']:.4f} | {lift['estimate']:.4f} |")
    lines.extend([
        "",
        "## Missing-trial handling and bounds",
        "",
        "A unit enters the primary predictions only when both registered trials have valid answers. Missing-trial bounds cover all registered units and are not used for pass or fail.",
        "",
        "| Excluded units | First | Middle | Last | Total |",
        "|---|---:|---:|---:|---:|",
    ])
    for kind in ("neuron", "control"):
        by_band = result["excluded_units_by_type_and_layer_band"][kind]
        lines.append(f"| {kind} | {by_band['first']} | {by_band['middle']} | {by_band['last']} | {sum(by_band.values())} |")
    lines.extend([
        "",
        "| Missing-trial score | 65a neuron agreement | 65b neuron minus control | First-third neuron | Middle-third neuron | Last-third neuron |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for label, key in (("0.5 (chance)", "missing_trials_scored_0_5"),
                       ("0 (wrong)", "missing_trials_scored_wrong")):
        scenario = result["missing_trial_bounds"][key]
        bands = scenario["neuron_agreement_by_layer_band"]
        lines.append(
            f"| {label} | {scenario['65a']['estimate']:.4f} | {scenario['65b']['estimate']:.4f} | "
            f"{bands['first']['estimate']:.4f} | {bands['middle']['estimate']:.4f} | "
            f"{bands['last']['estimate']:.4f} |"
        )
    late = result["sensitivity_layers_0_2_left_out"]
    lines.extend([
        "",
        "## Sensitivity: layers 0 to 2 left out (Amendment 3; not used for pass or fail)",
        "",
        f"65a neuron agreement {late['65a']['estimate']:.4f} (n={late['neuron_count']}); "
        f"65b neuron minus control {late['65b']['estimate']:.4f} (controls n={late['control_count']}).",
    ])
    lines.append("")
    return "\n".join(lines)


def analyze_files(answers: Path, collection: Path, read_neuron: Path,
                  read_control: Path, ranked: Path, *, seed: int = 6501,
                  bootstrap_reps: int = BOOTSTRAP_REPS,
                  shuffle_reps: int = SHUFFLE_REPS) -> dict[str, Any]:
    from steer65 import load_units
    result = analyze_rows(
        read_jsonl(answers),
        load_units(collection),
        read_labels(read_neuron, read_control),
        read_write_words(ranked),
        seed=seed,
        bootstrap_reps=bootstrap_reps,
        shuffle_reps=shuffle_reps,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--collection", type=Path,
                        default=Path("/home/elisepc/green/worktrees/ab-part64/results/part64/collection.jsonl"))
    parser.add_argument("--read-neuron", type=Path,
                        default=Path("/home/elisepc/green/worktrees/ab-part64/results/part64/gen64/outputs/haiku-1x-neuron.jsonl"))
    parser.add_argument("--read-control", type=Path,
                        default=Path("/home/elisepc/green/worktrees/ab-part64/results/part64/gen64/outputs/haiku-1x-control.jsonl"))
    parser.add_argument("--ranked", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("analysis-65.json"))
    parser.add_argument("--seed", type=int, default=6501)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--shuffle-reps", type=int, default=SHUFFLE_REPS)
    args = parser.parse_args(argv)
    try:
        result = analyze_files(args.answers, args.collection, args.read_neuron,
                               args.read_control, args.ranked, seed=args.seed,
                               bootstrap_reps=args.bootstrap_reps,
                               shuffle_reps=args.shuffle_reps)
    except AnalysisError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    markdown_path = args.out.with_suffix(".md")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"analysis written to {args.out} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
