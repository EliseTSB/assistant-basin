#!/usr/bin/env python3
"""Score Part 124 activation rows, gates, permutations and predictions."""
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
for candidate in (HERE, ROOT, ROOT / "part119"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import passages124  # noqa: E402

FRAMES = ("T", "F", "C")
TOP_K = 3
PERMUTATIONS = 10000
PERMUTATION_SEED = passages124.PERMUTATION_SEED
DEFAULT_OUT = "results/part124"
DEFAULT_PASSAGES = "part124/passages124.json"

PREDICTIONS = {
    "124.1": {
        "text": "In T, at least 9 of 39 units read their persona, and the count is above the 95th percentile of the permutation null.",
        "confidence": "low",
        "author": "Claude",
    },
    "124.2": {
        "text": "In T, the concept read rate is at least 15 percentage points above the slot read rate.",
        "confidence": "low",
        "author": "Claude",
    },
    "124.3": {
        "text": "In T, at most 4 of 21 slot units read their persona.",
        "confidence": "medium",
        "author": "Claude",
    },
    "124.4": {
        "text": "Of the units that read in T, at least half also read in C.",
        "confidence": "low",
        "author": "Claude",
    },
}


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path):
    with repo_path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path) -> list[dict]:
    source = repo_path(path)
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith("\n"):
                print(f"skipping torn final JSONL line: {source}:{index + 1}",
                      file=sys.stderr, flush=True)
                break
            raise ValueError(f"invalid activation JSONL at {source}:{index + 1}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"activation row at {source}:{index + 1} is not an object")
        rows.append(row)
    return rows


def finite_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _activation_key(row: dict) -> tuple[str, int, str, str]:
    return (str(row["frame"]), int(row["template_index"]),
            str(row["unit_id"]), str(row["side"]))


def validate_inputs(passages: dict, read_set: dict, run_meta: dict,
                    activation_rows: list[dict]) -> tuple[list[dict], dict]:
    units = passages.get("phrases")
    if not isinstance(units, list) or not units:
        raise ValueError("passage manifest has no phrase units")
    by_id = {str(unit["id"]): unit for unit in units}
    if len(by_id) != len(units):
        raise ValueError("passage manifest has duplicate unit ids")
    if len({str(unit["persona"]) for unit in units}) != len(units):
        raise ValueError("passage manifest has duplicate persona phrases")
    length_counts = Counter(int(unit["slot_token_count"]) for unit in units)
    expected_strata, _merge_history = passages124.resolve_length_strata(dict(length_counts))
    for unit in units:
        length = int(unit["slot_token_count"])
        if int(unit["length_stratum"]) != expected_strata[length]:
            raise ValueError(f"length stratum is stale for {unit['persona']!r}")
    feasibility = passages.get("length_feasibility")
    if not isinstance(feasibility, dict):
        raise ValueError("passage manifest has no length feasibility record")
    if feasibility.get("canonical_length_context") != "T template 1":
        raise ValueError("passage manifest uses an unexpected canonical phrase-length context")
    declared_counts = {int(length): int(count) for length, count in
                       feasibility.get("counts_by_token_length", {}).items()}
    if declared_counts != dict(length_counts):
        raise ValueError("declared phrase-length counts do not match the manifest")
    declared_strata = {int(length): int(label) for length, label in
                       feasibility.get("length_to_stratum", {}).items()}
    if declared_strata != expected_strata:
        raise ValueError("declared length strata do not match the rare-length merge rule")
    construction = passages.get("construction_check")
    if not isinstance(construction, dict):
        raise ValueError("passage manifest has no construction check")

    random_pools = read_set.get("random_pools")
    random_units = read_set.get("units")
    if not isinstance(random_pools, dict) or not isinstance(random_units, dict):
        raise ValueError("read set is missing random pools or unit channels")
    layer_keys = {str(int(unit["layer"])) for unit in units}
    if set(random_pools) != layer_keys:
        raise ValueError("read-set layer pools do not match the focal units")
    for layer, pool in random_pools.items():
        if not isinstance(pool, list) or len(pool) != int(read_set.get("random_count_per_layer", -1)):
            raise ValueError(f"random pool size is wrong for layer {layer}")
        if len(pool) != len(set(int(channel) for channel in pool)):
            raise ValueError(f"random pool repeats a channel in layer {layer}")
    focal_channels_by_layer = defaultdict(set)
    for unit in units:
        focal_channels_by_layer[str(int(unit["layer"]))].add(int(unit["neuron"]))
    for layer, pool in random_pools.items():
        if set(int(channel) for channel in pool) & focal_channels_by_layer[layer]:
            raise ValueError(f"random pool includes a focal channel at layer {layer}")
    for unit in units:
        unit_id = str(unit["id"])
        config = random_units.get(unit_id)
        if not isinstance(config, dict):
            raise ValueError(f"read set is missing unit {unit_id}")
        if int(config.get("layer", -1)) != int(unit["layer"]) or \
                int(config.get("channel", -1)) != int(unit["neuron"]):
            raise ValueError(f"read-set focal channel is stale for unit {unit_id}")
        controls = config.get("random_controls")
        if not isinstance(controls, list) or len(controls) != 3:
            raise ValueError(f"unit {unit_id} needs three random channels")
        if len({int(control["channel"]) for control in controls}) != 3:
            raise ValueError(f"unit {unit_id} repeats a random channel")
        if any(int(control["channel"]) not in random_pools[str(unit["layer"])]
               for control in controls):
            raise ValueError(f"unit {unit_id} has a control outside its layer pool")
        if any(control.get("target_phrase") != unit["persona"] for control in controls):
            raise ValueError(f"unit {unit_id} random controls lack its pseudo-target phrase")

    records = {}
    expected = {
        (frame, template_index, unit_id, side)
        for frame in FRAMES
        for template_index in range(8)
        for unit_id in by_id
        for side in ("persona", "neutral")
    }
    for row in activation_rows:
        try:
            key = _activation_key(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("activation row has an invalid frame/template/unit/side key") from exc
        if key not in expected:
            raise ValueError(f"unexpected activation row {key}")
        if key in records:
            raise ValueError(f"duplicate activation row {key}")
        frame, template_index, unit_id, side = key
        unit = by_id[unit_id]
        expected_phrase = unit["persona"] if side == "persona" else unit["neutral"]
        if row.get("phrase") != expected_phrase or row.get("persona") != unit["persona"]:
            raise ValueError(f"activation phrase does not match the manifest for {key}")
        if row.get("kind") != unit["kind"]:
            raise ValueError(f"activation kind does not match the manifest for {key}")
        slot_counts = unit.get("slot_token_counts_by_frame_template")
        expected_slot_counts = slot_counts.get(frame) if isinstance(slot_counts, dict) else None
        if not isinstance(expected_slot_counts, list) or len(expected_slot_counts) != 8:
            raise ValueError(f"manifest lacks context slot counts for {unit_id}")
        if int(row.get("slot_token_count", -1)) != int(expected_slot_counts[template_index]):
            raise ValueError(f"activation slot token count is stale for {key}")
        focal = row.get("focal")
        if not isinstance(focal, dict) or set(focal) != layer_keys:
            raise ValueError(f"missing focal layer activations for {key}")
        for layer in layer_keys:
            layer_values = focal.get(layer)
            if not isinstance(layer_values, dict):
                raise ValueError(f"missing focal layer values for {key} at layer {layer}")
            expected_ids = {str(unit["id"]) for unit in units if str(unit["layer"]) == layer}
            if set(layer_values) != expected_ids:
                raise ValueError(f"missing focal channels for {key} at layer {layer}")
            if any(finite_number(value) is None for value in layer_values.values()):
                raise ValueError(f"nonfinite focal activation for {key} at layer {layer}")
        for field in ("focal_mean", "focal_post_max"):
            diagnostic = row.get(field)
            if not isinstance(diagnostic, dict) or set(diagnostic) != layer_keys:
                raise ValueError(f"missing {field} diagnostics for {key}")
            for layer in layer_keys:
                layer_values = diagnostic.get(layer)
                expected_ids = {str(unit["id"]) for unit in units
                                if str(unit["layer"]) == layer}
                if not isinstance(layer_values, dict) or set(layer_values) != expected_ids:
                    raise ValueError(f"missing {field} channels for {key} at layer {layer}")
                if any(finite_number(value) is None for value in layer_values.values()):
                    raise ValueError(f"nonfinite {field} diagnostic for {key} at layer {layer}")
        if side == "persona":
            random_max = row.get("random_max")
            if not isinstance(random_max, dict) or set(random_max) != layer_keys:
                raise ValueError(f"missing random channel activations for {key}")
            for layer in layer_keys:
                values = random_max.get(layer)
                if not isinstance(values, list) or len(values) != len(random_pools[layer]):
                    raise ValueError(f"random activation vector has the wrong size for {key} at {layer}")
                if any(finite_number(value) is None for value in values):
                    raise ValueError(f"nonfinite random activation for {key} at layer {layer}")
        elif "random_max" in row:
            raise ValueError(f"neutral activation unexpectedly includes random channels for {key}")
        records[key] = row
    missing = expected - set(records)
    if missing:
        example = sorted(missing)[:3]
        raise ValueError(f"missing {len(missing)} activation rows, including {example}")
    if len(records) != len(expected):
        raise ValueError("activation coverage does not match the expected passages")

    do_nothing = run_meta.get("do_nothing")
    if not isinstance(do_nothing, dict):
        raise ValueError("run metadata has no do-nothing check")
    return units, {"records": records, "construction": construction,
                   "do_nothing": do_nothing, "by_id": by_id}


def _top3(scores: dict[str, float]) -> tuple[list[str], dict[str, int]]:
    ordered = sorted(scores, key=lambda phrase: (-scores[phrase], phrase))
    return ordered[:TOP_K], {phrase: index + 1 for index, phrase in enumerate(ordered)}


def _focal_results(units: list[dict], records: dict) -> tuple[dict, dict]:
    by_id = {str(unit["id"]): unit for unit in units}
    phrase_to_id = {str(unit["persona"]): str(unit["id"]) for unit in units}
    output = {frame: {} for frame in FRAMES}
    scores = {frame: {} for frame in FRAMES}
    for frame in FRAMES:
        for unit in units:
            unit_id = str(unit["id"])
            layer = str(int(unit["layer"]))
            phrase_scores = {}
            per_template = {}
            per_template_mean = {}
            per_template_post_max = {}
            for phrase, phrase_unit_id in phrase_to_id.items():
                owner = by_id[phrase_unit_id]
                values = [float(records[(frame, template_index, phrase_unit_id, "persona")]
                                ["focal"][layer][unit_id])
                          for template_index in range(8)]
                phrase_scores[phrase] = sum(values) / len(values)
                per_template[phrase] = values
                per_template_mean[phrase] = [
                    float(records[(frame, template_index, phrase_unit_id, "persona")]
                          ["focal_mean"][layer][unit_id])
                    for template_index in range(8)
                ]
                per_template_post_max[phrase] = [
                    float(records[(frame, template_index, phrase_unit_id, "persona")]
                          ["focal_post_max"][layer][unit_id])
                    for template_index in range(8)
                ]
            top, ranks = _top3(phrase_scores)
            own_phrase = str(unit["persona"])
            own_rank = ranks[own_phrase]
            output[frame][unit_id] = {
                "unit_id": unit_id,
                "persona": own_phrase,
                "kind": str(unit["kind"]),
                "slot_token_count": int(unit["slot_token_count"]),
                "length_stratum": int(unit["length_stratum"]),
                "reads": own_phrase in top,
                "own_rank": own_rank,
                "top3": top,
                "phrase_scores": phrase_scores,
                "template_maxima": per_template,
                "template_mean_slot_activation": per_template_mean,
                "template_post_slot_maxima": per_template_post_max,
            }
            scores[frame][unit_id] = {
                "phrase_scores": phrase_scores,
                "top3": set(top),
                "ranks": ranks,
            }
    return output, scores


def _random_results(units: list[dict], read_set: dict, records: dict) -> tuple[dict, dict]:
    phrases = [str(unit["persona"]) for unit in units]
    phrase_index = {phrase: index for index, phrase in enumerate(phrases)}
    pools = read_set["random_pools"]
    rates = {}
    details = {}
    for frame in FRAMES:
        reads = 0
        observed = 0
        unit_details = []
        for unit in units:
            unit_id = str(unit["id"])
            layer = str(int(unit["layer"]))
            pool = [int(channel) for channel in pools[layer]]
            control_rows = read_set["units"][unit_id]["random_controls"]
            for control in control_rows:
                channel = int(control["channel"])
                target = str(control["target_phrase"])
                channel_index = pool.index(channel)
                phrase_scores = {}
                for phrase in phrases:
                    owner_id = next(str(row["id"]) for row in units if row["persona"] == phrase)
                    vals = [float(records[(frame, template_index, owner_id, "persona")]
                                  ["random_max"][layer][channel_index])
                            for template_index in range(8)]
                    phrase_scores[phrase] = sum(vals) / 8
                top, _ranks = _top3(phrase_scores)
                passed = target in top
                reads += int(passed)
                observed += 1
                unit_details.append({"unit_id": unit_id, "channel": channel,
                                     "target_phrase": target, "reads": passed})
        rates[frame] = {"reads": reads, "observed": observed,
                        "rate": reads / observed if observed else None,
                        "threshold": 0.15,
                        "passed": bool(observed and reads / observed <= 0.15)}
        details[frame] = unit_details
    return rates, details


def _instrument_results(units: list[dict], read_set: dict, records: dict) -> tuple[dict, dict]:
    phrases = [str(unit["persona"]) for unit in units]
    by_phrase_id = {str(unit["persona"]): str(unit["id"]) for unit in units}
    per_frame = {}
    per_layer = {}
    for frame in FRAMES:
        total_repeat = 0
        total_channels = 0
        layer_rows = {}
        for layer, pool_value in read_set["random_pools"].items():
            pool = [int(channel) for channel in pool_value]
            repeat = 0
            for channel_index, _channel in enumerate(pool):
                odd_scores, even_scores = {}, {}
                for phrase in phrases:
                    owner_id = by_phrase_id[phrase]
                    odd = [float(records[(frame, template_index, owner_id, "persona")]
                                 ["random_max"][layer][channel_index])
                           for template_index in (0, 2, 4, 6)]
                    even = [float(records[(frame, template_index, owner_id, "persona")]
                                  ["random_max"][layer][channel_index])
                            for template_index in (1, 3, 5, 7)]
                    odd_scores[phrase] = sum(odd) / 4
                    even_scores[phrase] = sum(even) / 4
                odd_top, _ = _top3(odd_scores)
                odd_best = odd_top[0]
                even_top, _ = _top3(even_scores)
                repeat += int(odd_best in even_top)
            count = len(pool)
            layer_rows[layer] = {"repeat": repeat, "channels": count,
                                 "rate": repeat / count if count else None}
            total_repeat += repeat
            total_channels += count
        per_frame[frame] = {
            "repeat": total_repeat,
            "channels": total_channels,
            "rate": total_repeat / total_channels if total_channels else None,
            "chance": 3 / 39,
            "threshold": 0.15,
            "gate_applies": frame == "T",
            "passed": (bool(total_channels and total_repeat / total_channels >= 0.15)
                       if frame == "T" else None),
        }
        per_layer[frame] = layer_rows
    return per_frame, per_layer


def _neutral_results(units: list[dict], records: dict) -> dict:
    results = {}
    for frame in FRAMES:
        results[frame] = {}
        for kind in ("concept", "slot"):
            kind_units = [unit for unit in units if unit["kind"] == kind]
            wins = 0
            comparisons = 0
            by_unit = []
            for unit in kind_units:
                unit_id = str(unit["id"])
                layer = str(int(unit["layer"]))
                unit_wins = 0
                for template_index in range(8):
                    persona = float(records[(frame, template_index, unit_id, "persona")]
                                    ["focal"][layer][unit_id])
                    neutral = float(records[(frame, template_index, unit_id, "neutral")]
                                    ["focal"][layer][unit_id])
                    unit_wins += int(persona > neutral)
                wins += unit_wins
                comparisons += 8
                by_unit.append({"unit_id": unit_id, "wins": unit_wins, "templates": 8,
                                "share": unit_wins / 8})
            results[frame][kind] = {
                "wins": wins,
                "comparisons": comparisons,
                "share": wins / comparisons if comparisons else None,
                "by_unit": by_unit,
            }
    return results


def _permutation_results(units: list[dict], focal: dict, *, count: int = PERMUTATIONS,
                         seed: int = PERMUTATION_SEED) -> dict:
    groups: dict[int, list[dict]] = defaultdict(list)
    for unit in units:
        groups[int(unit["length_stratum"])].append(unit)
    rng = random.Random(seed)
    null_all = []
    null_by_kind = {"concept": [], "slot": []}
    for _iteration in range(count):
        assigned = {}
        for stratum_units in groups.values():
            targets = [str(unit["persona"]) for unit in stratum_units]
            rng.shuffle(targets)
            assigned.update((str(unit["id"]), phrase)
                            for unit, phrase in zip(stratum_units, targets))
        read_flags = {
            str(unit["id"]): assigned[str(unit["id"])] in
            focal["T"][str(unit["id"])]["top3"]
            for unit in units
        }
        null_all.append(sum(read_flags.values()))
        for kind in ("concept", "slot"):
            null_by_kind[kind].append(sum(read_flags[str(unit["id"])]
                                          for unit in units if unit["kind"] == kind))

    def describe(values: list[int], observed: int) -> dict:
        ordered = sorted(values)
        p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
        tail = sum(value >= observed for value in values)
        return {
            "observed": observed,
            "median": (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2,
            "p95": p95,
            "above_p95": observed > p95,
            "tail_count": tail,
            "permutations": len(values),
            "tail_rate": tail / len(values),
            "minimum_excess": 6,
            "six_unit_excess_met": observed - ((ordered[(len(ordered) - 1) // 2] +
                                                  ordered[len(ordered) // 2]) / 2) >= 6,
            "null_counts": values,
        }

    observed_all = sum(str(unit["persona"]) in focal["T"][str(unit["id"])]["top3"]
                       for unit in units)
    observed_kind = {
        kind: sum(str(unit["persona"]) in focal["T"][str(unit["id"])]["top3"]
                  for unit in units if unit["kind"] == kind)
        for kind in ("concept", "slot")
    }
    return {
        "seed": seed,
        "permutations": count,
        "all_units": describe(null_all, observed_all),
        "by_kind": {kind: describe(null_by_kind[kind], observed_kind[kind])
                    for kind in ("concept", "slot")},
    }


def _family_diagnostics(units: list[dict], focal_scores: dict, frame: str = "T") -> list[dict]:
    family_by_phrase = {}
    for family, phrases in passages124.FAMILIES.items():
        for phrase in phrases:
            family_by_phrase[phrase] = family
    results = []
    for unit in units:
        unit_id = str(unit["id"])
        row = focal_scores[frame][unit_id]
        own = str(unit["persona"])
        if row["reads"]:
            continue
        family = family_by_phrase.get(own)
        if family is None:
            continue
        siblings = set(passages124.FAMILIES[family]) - {own}
        ordered = sorted(row["phrase_scores"],
                         key=lambda phrase: (-row["phrase_scores"][phrase], phrase))
        siblings_ahead = [phrase for phrase in ordered[:row["own_rank"] - 1] if phrase in siblings]
        remaining = [phrase for phrase in ordered if phrase not in siblings]
        compressed_rank = remaining.index(own) + 1
        if compressed_rank <= TOP_K:
            results.append({
                "unit_id": unit_id,
                "persona": own,
                "family": family,
                "own_rank": row["own_rank"],
                "siblings_ahead": siblings_ahead,
                "rank_without_siblings": compressed_rank,
                "counts_as_read": False,
            })
    return results


def score_results(passages: dict, read_set: dict, run_meta: dict,
                  activation_rows: list[dict], *, permutations: int = PERMUTATIONS) -> dict:
    units, validated = validate_inputs(passages, read_set, run_meta, activation_rows)
    records = validated["records"]
    focal_rows, focal_scores = _focal_results(units, records)
    random_rates, random_rows = _random_results(units, read_set, records)
    instrument, instrument_by_layer = _instrument_results(units, read_set, records)
    neutral = _neutral_results(units, records)
    permutation = _permutation_results(units, focal_scores, count=permutations)

    construction_data = validated["construction"]
    construction_ok = bool(
        construction_data.get("passed") is True and
        construction_data.get("frame_template_checks") == 24 and
        construction_data.get("matched_persona_neutral_pairs") == len(units) * 24 and
        construction_data.get("slots_checked") == len(units) * 48 and
        construction_data.get("outside_token_ids_identical_within_each_frame_template") is True and
        construction_data.get("persona_neutral_slot_counts_match") is True and
        construction_data.get("quote_boundary_merges") == 0
    )
    do_nothing = validated["do_nothing"]
    do_nothing_ok = bool(do_nothing.get("passed") is True and
                         do_nothing.get("identical_logits") is True)
    random_gate_ok = all(row["passed"] for row in random_rates.values())
    instrument_gate_ok = instrument["T"]["passed"]
    gates = {
        "construction": {"passed": construction_ok, "details": construction_data},
        "do_nothing": {"passed": do_nothing_ok, "details": do_nothing},
        "random_channels": {"passed": random_gate_ok, "by_frame": random_rates},
        "instrument_T": {"passed": instrument_gate_ok, "by_frame": instrument,
                         "by_layer": instrument_by_layer},
    }
    all_gates_passed = all(row["passed"] for row in gates.values())

    kind_counts = {}
    for kind in ("concept", "slot"):
        kind_units = [unit for unit in units if unit["kind"] == kind]
        reads = sum(focal_rows["T"][str(unit["id"])]["reads"] for unit in kind_units)
        kind_counts[kind] = {"reads": reads, "units": len(kind_units),
                             "rate": reads / len(kind_units) if kind_units else None,
                             "permutation": permutation["by_kind"][kind]}
    concept_rate = kind_counts["concept"]["rate"]
    slot_rate = kind_counts["slot"]["rate"]
    kind_rate_difference = (concept_rate - slot_rate
                            if concept_rate is not None and slot_rate is not None else None)
    t_count = permutation["all_units"]["observed"]
    t_rows = focal_rows["T"]
    c_rows = focal_rows["C"]
    t_read_ids = {unit_id for unit_id, row in t_rows.items() if row["reads"]}
    persisted = sum(c_rows[unit_id]["reads"] for unit_id in t_read_ids)
    persistence_rate = persisted / len(t_read_ids) if t_read_ids else None
    concept_rate = kind_counts["concept"]["rate"]
    slot_rate = kind_counts["slot"]["rate"]
    predictions = {}
    if not all_gates_passed:
        for key, value in PREDICTIONS.items():
            predictions[key] = {**value, "status": "withdrawn: gate miss"}
    else:
        predictions = {
            "124.1": {**PREDICTIONS["124.1"], "status": bool(
                len(units) == 39 and t_count >= 9 and permutation["all_units"]["above_p95"])},
            "124.2": {**PREDICTIONS["124.2"], "status": bool(
                concept_rate is not None and slot_rate is not None and concept_rate - slot_rate >= 0.15)},
            "124.3": {**PREDICTIONS["124.3"], "status": bool(
                len([unit for unit in units if unit["kind"] == "slot"]) == 21 and
                kind_counts["slot"]["reads"] <= 4)},
            "124.4": {**PREDICTIONS["124.4"], "status": (
                None if not t_read_ids else persistence_rate >= 0.5)},
        }

    by_length = {}
    for frame in FRAMES:
        by_length[frame] = {}
        for unit in units:
            length = str(int(unit["slot_token_count"]))
            row = by_length[frame].setdefault(length, {"reads": 0, "units": 0})
            row["units"] += 1
            row["reads"] += int(focal_rows[frame][str(unit["id"])]["reads"])
        for row in by_length[frame].values():
            row["rate"] = row["reads"] / row["units"] if row["units"] else None
    by_stratum = {}
    for frame in FRAMES:
        by_stratum[frame] = {}
        for unit in units:
            stratum = str(int(unit["length_stratum"]))
            row = by_stratum[frame].setdefault(stratum, {"reads": 0, "units": 0})
            row["units"] += 1
            row["reads"] += int(focal_rows[frame][str(unit["id"])]["reads"])
        for row in by_stratum[frame].values():
            row["rate"] = row["reads"] / row["units"] if row["units"] else None

    interpretations = []
    if not all_gates_passed:
        missed = [name for name, gate in gates.items() if not gate["passed"]]
        interpretations.append("Headline withdrawn because these gates missed: " + ", ".join(missed) + ".")
    else:
        kinds_beat = [kind for kind in ("concept", "slot")
                      if kind_counts[kind]["permutation"]["above_p95"]]
        if not kinds_beat:
            interpretations.append(
                "No own-persona word preference was found under these quoted templates. "
                "Part 122's prose reads may come from context words, or the quotes may blunt the word. "
                "This tests mentioned text, not a description."
            )
        else:
            interpretations.append("Kinds above their length-stratified permutation null: " +
                                   ", ".join(kinds_beat) + ".")
    preamble = _preamble(gates, random_rates, instrument)
    return {
        "schema_version": 1,
        "frames": list(FRAMES),
        "top_k": TOP_K,
        "chance_rate": 3 / 39,
        "all_gates_passed": all_gates_passed,
        "preamble": preamble,
        "gates": gates,
        "random_controls": random_rows,
        "instrument": {"by_frame": instrument, "by_layer": instrument_by_layer},
        "focal": focal_rows,
        "kind_counts": kind_counts,
        "concept_minus_slot_rate": kind_rate_difference,
        "reads_by_token_length": by_length,
        "reads_by_length_stratum": by_stratum,
        "permutations": permutation,
        "neutral": neutral,
        "family_crowding_near_misses_T": _family_diagnostics(units, focal_rows),
        "persistence_T_to_C": {"reads_in_T": len(t_read_ids), "also_read_in_C": persisted,
                                "rate": persistence_rate},
        "predictions": predictions,
        "interpretations": interpretations,
    }


def _rate(row: dict) -> str:
    if row.get("rate") is None:
        return "not estimable"
    return f"{row['reads']}/{row['observed']} ({row['rate']:.1%})"


def _preamble(gates: dict, random_rates: dict, instrument: dict) -> list[str]:
    random_parts = []
    false_parts = []
    for frame in FRAMES:
        row = random_rates[frame]
        random_parts.append(f"{frame} {row['reads']}/{row['observed']}")
        false_parts.append(f"{frame} {row['reads']}/{row['observed']}" +
                           (f" ({row['rate']:.1%})" if row["rate"] is not None else ""))
    random_gate = "pass" if gates["random_channels"]["passed"] else "fail"
    lines = [
        f"random reads: {', '.join(random_parts)}; observed false-positive rates: "
        f"{', '.join(false_parts)}; gate: {random_gate}"
    ]
    instrument_parts = []
    for frame in FRAMES:
        row = instrument[frame]
        instrument_parts.append(f"{frame} {row['repeat']}/{row['channels']}" +
                                (f" ({row['rate']:.1%})" if row["rate"] is not None else ""))
    instrument_gate = "pass" if gates["instrument_T"]["passed"] else "fail"
    lines.append(f"instrument repeat (chance 7.7%): {', '.join(instrument_parts)}; "
                 f"T gate: {instrument_gate}")
    for name in ("construction", "do_nothing", "random_channels", "instrument_T"):
        status = "pass" if gates[name]["passed"] else "fail"
        lines.append(f"gate {name}: {status}")
    return lines


def format_score(result: dict) -> str:
    lines = ["# Part 124 score", "", "## Gates", ""]
    lines.extend(result["preamble"])
    lines.append("")
    lines.append("### Instrument repeat by layer")
    lines.append("")
    lines.append("| Frame | Layer | Repeats | Rate |")
    lines.append("| --- | ---: | ---: | ---: |")
    for frame in FRAMES:
        for layer, row in sorted(result["instrument"]["by_layer"][frame].items(),
                                 key=lambda item: int(item[0])):
            lines.append(f"| {frame} | {layer} | {row['repeat']}/{row['channels']} | {row['rate']:.1%} |")
    lines.append("")
    lines.append("## Focal reads")
    lines.append("")
    if not result["all_gates_passed"]:
        lines.append("The headline is withdrawn because at least one gate failed. The counts below are gated results.")
        lines.append("")
    for frame in FRAMES:
        rows = result["focal"][frame]
        reads = sum(row["reads"] for row in rows.values())
        lines.append(f"### {frame}: {reads}/{len(rows)} reads")
        lines.append("")
        lines.append("| Unit | Kind | Slot tokens | Stratum | Own rank | Read | Top three | Own maxima by template 1-8 | Mean slot avg | Post-slot maxima by template 1-8 |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- |")
        for unit_id, row in rows.items():
            own_maxima = row["template_maxima"][row["persona"]]
            maxima_text = ", ".join(f"{value:.6g}" for value in own_maxima)
            own_means = row["template_mean_slot_activation"][row["persona"]]
            mean_text = f"{sum(own_means) / len(own_means):.6g}"
            own_post_maxima = row["template_post_slot_maxima"][row["persona"]]
            post_maxima_text = ", ".join(f"{value:.6g}" for value in own_post_maxima)
            lines.append(f"| {unit_id} ({row['persona']}) | {row['kind']} | "
                         f"{row['slot_token_count']} | {row['length_stratum']} | "
                         f"{row['own_rank']} | {'yes' if row['reads'] else 'no'} | "
                         f"{', '.join(row['top3'])} | {maxima_text} | {mean_text} | "
                         f"{post_maxima_text} |")
        lines.append("")
    lines.append("## Length and kind summaries")
    lines.append("")
    lines.append("| Frame | Token length | Reads | Rate |")
    lines.append("| --- | ---: | ---: | ---: |")
    for frame in FRAMES:
        for length, row in sorted(result["reads_by_token_length"][frame].items(), key=lambda item: int(item[0])):
            lines.append(f"| {frame} | {length} | {row['reads']}/{row['units']} | {row['rate']:.1%} |")
    lines.append("")
    for kind in ("concept", "slot"):
        row = result["kind_counts"][kind]
        perm = row["permutation"]
        lines.append(f"- {kind}: {row['reads']}/{row['units']} reads in T; "
                     f"rate {row['rate']:.1%}; permutation p95 {perm['p95']}/{row['units']}.")
    difference = result["concept_minus_slot_rate"]
    difference_text = "not estimable" if difference is None else f"{difference:+.1%}"
    lines.append(f"- Concept minus slot read-rate difference in T: {difference_text}.")
    lines.append("")
    lines.append("## Permutation null")
    lines.append("")
    all_null = result["permutations"]["all_units"]
    lines.append(f"T true map: {all_null['observed']}/{len(result['focal']['T'])}; "
                 f"null median {all_null['median']:.1f}; p95 {all_null['p95']}; "
                 f"empirical tail {all_null['tail_count']}/{all_null['permutations']} "
                 f"({all_null['tail_rate']:.4f}).")
    lines.append(f"Six-unit excess over the median: "
                 f"{'met' if all_null['six_unit_excess_met'] else 'not met'}.")
    lines.append("")
    lines.append("## Neutral foils")
    lines.append("")
    lines.append("| Frame | Kind | Own phrase beats neutral | Share |")
    lines.append("| --- | --- | ---: | ---: |")
    for frame in FRAMES:
        for kind in ("concept", "slot"):
            row = result["neutral"][frame][kind]
            share = "not estimable" if row["share"] is None else f"{row['share']:.1%}"
            lines.append(f"| {frame} | {kind} | {row['wins']}/{row['comparisons']} | {share} |")
    lines.append("")
    lines.append("## Family-crowding near misses")
    lines.append("")
    crowding = result["family_crowding_near_misses_T"]
    if not crowding:
        lines.append("None.")
    else:
        lines.append("These phrases enter the top three only after their family siblings are removed. They are not reads.")
        lines.append("")
        lines.append("| Unit | Persona | Family | Original rank | Siblings ahead | Rank without siblings |")
        lines.append("| --- | --- | --- | ---: | --- | ---: |")
        for row in crowding:
            lines.append(f"| {row['unit_id']} | {row['persona']} | {row['family']} | {row['own_rank']} | "
                         f"{', '.join(row['siblings_ahead'])} | {row['rank_without_siblings']} |")
    lines.append("")
    lines.append("## Predictions")
    lines.append("")
    lines.append("| # | Status | Confidence | Author | Prediction |")
    lines.append("| --- | --- | --- | --- | --- |")
    for key, row in result["predictions"].items():
        status = "not estimable" if row["status"] is None else str(row["status"])
        lines.append(f"| {key} | {status} | {row['confidence']} | {row['author']} | {row['text']} |")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.extend(result["interpretations"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", default=DEFAULT_PASSAGES)
    parser.add_argument("--activations", default="results/part124/activations.jsonl")
    parser.add_argument("--read-set", default="results/part124/read_set.json")
    parser.add_argument("--run-meta", default="results/part124/run_meta.json")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = parser.parse_args(argv)
    if args.permutations < 1:
        parser.error("--permutations must be positive")

    passages = read_json(args.passages)
    read_set = read_json(args.read_set)
    run_meta = read_json(args.run_meta)
    activation_rows = read_jsonl(args.activations)
    result = score_results(passages, read_set, run_meta, activation_rows,
                           permutations=args.permutations)
    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = format_score(result)
    with (out_dir / "SCORE.md").open("w", encoding="utf-8") as stream:
        stream.write(markdown)
    with (out_dir / "score124.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    for line in result["preamble"]:
        print(line, flush=True)
    print("focal results follow in SCORE.md", flush=True)
    print(json.dumps({"score": str(out_dir / "score124.json"),
                      "markdown": str(out_dir / "SCORE.md"),
                      "all_gates_passed": result["all_gates_passed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
