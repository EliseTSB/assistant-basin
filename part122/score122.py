#!/usr/bin/env python3
"""Score Part 122 activation rows, gates, predictions, and registered readings."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "part119") not in sys.path:
    sys.path.insert(0, str(ROOT / "part119"))

import rows119  # noqa: E402

FRAMES = ("T", "F", "C")
DEFAULT_OUT = "results/part122"
READ_WIN_MIN = 7
PAIRS_PER_FRAME = 8
PREDICTION_CONFIDENCE = {
    "122.1": "medium", "122.2": "medium", "122.3": "medium",
    "122.4": "low", "122.5": "low", "122.6": "medium", "122.7": "low",
}
PREDICTION_TEXT = {
    "122.1": "In frame T, at least 9 of the 18 concept units read their persona.",
    "122.2": "In frame T, at most 5 of the 21 slot units read their persona.",
    "122.3": "In frame T, the concept rate is above the slot rate, with a two-sided Fisher p below .05.",
    "122.4": "In frame C, at least 5 more of the 21 slot units read their persona than in frame T.",
    "122.5": "The gain from T to C is larger for slot units than for concept units, as a share of each kind.",
    "122.6": "In every frame, the shuffled control read rate is at most 10% of units.",
    "122.7": "Of the 11 concept units whose persona the J-lens matches, at least 7 read in frame T.",
}


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path):
    with repo_path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with repo_path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row in {path}:{line_no}")
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


def valid_count(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def evaluate_gate1(pair_counts: list[tuple[object, object] | None], expected: int) -> dict:
    within = 0
    observed = 0
    for pair in pair_counts:
        if pair is None or len(pair) != 2:
            continue
        left, right = valid_count(pair[0]), valid_count(pair[1])
        if left is None or right is None:
            continue
        observed += 1
        within += abs(left - right) <= 2
    rate = within / observed if observed else None
    complete = observed == expected
    return {"observed_pairs": observed, "expected_pairs": expected,
            "within_two": within, "rate": rate, "threshold": 0.90,
            "passed": bool(complete and rate is not None and rate >= 0.90)}


def evaluate_gate2(gate2: object, expected: int = 16) -> dict:
    pairs = gate2.get("pairs") if isinstance(gate2, dict) else None
    pairs = pairs if isinstance(pairs, list) else []
    wins = 0
    observed = 0
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        space = pair.get("space")
        ordinary = pair.get("ordinary")
        if not isinstance(space, dict) or not isinstance(ordinary, dict):
            continue
        smax = finite_number(space.get("max"))
        omax = finite_number(ordinary.get("max"))
        if smax is None or omax is None:
            continue
        observed += 1
        wins += smax > omax
    return {"observed_pairs": observed, "expected_pairs": expected, "space_max_wins": wins,
            "threshold": 9, "passed": bool(len(pairs) == expected and observed == expected and wins >= 9)}


def evaluate_gate3(frame_results: dict[str, list[bool | None]], expected_cells: int) -> dict:
    by_frame = {}
    for frame in FRAMES:
        values = frame_results.get(frame, [])
        complete = len(values) == expected_cells and all(type(value) is bool for value in values)
        reads = sum(value is True for value in values if type(value) is bool)
        rate = reads / expected_cells if expected_cells else None
        by_frame[frame] = {"read_controls": reads, "observed_cells": sum(type(value) is bool for value in values),
                           "expected_cells": expected_cells, "rate": rate, "threshold": 0.15,
                           "passed": bool(complete and rate is not None and rate <= 0.15)}
    return {"by_frame": by_frame, "passed": all(row["passed"] for row in by_frame.values())}


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float | None:
    """Two-sided Fisher exact p using probability ordering, matching scipy's default."""
    values = (a, b, c, d)
    if any(valid_count(value) is None for value in values):
        return None
    first_n, second_n = a + b, c + d
    successes, total = a + c, a + b + c + d
    if total == 0:
        return None
    low = max(0, successes - second_n)
    high = min(first_n, successes)

    def probability(x: int) -> float:
        return (math.comb(successes, x) * math.comb(total - successes, first_n - x) /
                math.comb(total, first_n))

    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(low, high + 1)
                         if probability(x) <= observed + 1e-12))


def _valid_stats(row: object) -> dict | None:
    if not isinstance(row, dict):
        return None
    maximum, mean = finite_number(row.get("max")), finite_number(row.get("mean"))
    if maximum is None or mean is None:
        return None
    return {"max": maximum, "mean": mean}


def unit_read_verdict(max_wins: int, pair_count: int = PAIRS_PER_FRAME) -> bool | None:
    """Apply the registered 7 of 8 max-activation rule to a complete pair set."""
    if pair_count != PAIRS_PER_FRAME or valid_count(max_wins) is None:
        return None
    return max_wins >= READ_WIN_MIN


def _lookup_records(activation_rows: list[dict]):
    records = {}
    duplicates = set()
    for row in activation_rows:
        if not isinstance(row, dict):
            continue
        try:
            key = (str(row["frame"]), str(row["source_unit_id"]),
                   int(row["pair_index"]), str(row["side"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key in records:
            duplicates.add(key)
        else:
            records[key] = row
    for key in duplicates:
        records.pop(key, None)
    return records, duplicates


def _record(records: dict, duplicates: set, frame: str, source: str, pair_index: int, side: str):
    key = (frame, source, pair_index, side)
    return None if key in duplicates else records.get(key)


def _token_pair(records: dict, duplicates: set, frame: str, source: str, pair_index: int):
    persona = _record(records, duplicates, frame, source, pair_index, "persona")
    ordinary = _record(records, duplicates, frame, source, pair_index, "ordinary")
    if persona is None or ordinary is None:
        return None
    left, right = valid_count(persona.get("token_count")), valid_count(ordinary.get("token_count"))
    return (left, right) if left is not None and right is not None else None


def _channel_stats(record: dict | None, unit_id: str, control: str,
                   random_id: int | None = None) -> dict | None:
    if record is None or not isinstance(record.get("values"), dict):
        return None
    value = record["values"].get(unit_id)
    if not isinstance(value, dict):
        return None
    if control == "unit":
        return _valid_stats(value.get("unit"))
    random_rows = value.get("random")
    if not isinstance(random_rows, list):
        return None
    match = [row for row in random_rows if isinstance(row, dict) and row.get("random_id") == random_id]
    return _valid_stats(match[0]) if len(match) == 1 else None


def _compare_series(records: dict, duplicates: set, unit_id: str, source_id: str,
                    frame: str, control: str, random_id: int | None = None) -> dict:
    wins = losses = ties = mean_wins = mean_losses = mean_ties = 0
    max_persona, max_ordinary, mean_persona, mean_ordinary = [], [], [], []
    complete = True
    for pair_index in range(PAIRS_PER_FRAME):
        persona_row = _record(records, duplicates, frame, source_id, pair_index, "persona")
        ordinary_row = _record(records, duplicates, frame, source_id, pair_index, "ordinary")
        p = _channel_stats(persona_row, unit_id, control, random_id)
        o = _channel_stats(ordinary_row, unit_id, control, random_id)
        if p is None or o is None:
            complete = False
            continue
        wins += p["max"] > o["max"]
        losses += p["max"] < o["max"]
        ties += p["max"] == o["max"]
        mean_wins += p["mean"] > o["mean"]
        mean_losses += p["mean"] < o["mean"]
        mean_ties += p["mean"] == o["mean"]
        max_persona.append(p["max"])
        max_ordinary.append(o["max"])
        mean_persona.append(p["mean"])
        mean_ordinary.append(o["mean"])
    valid = complete and len(max_persona) == PAIRS_PER_FRAME
    return {"pairs": len(max_persona), "max_wins": wins, "max_losses": losses, "max_ties": ties,
            "read": unit_read_verdict(wins, PAIRS_PER_FRAME) if valid else None,
            "mean_wins": mean_wins if valid else None,
            "mean_losses": mean_losses if valid else None,
            "mean_ties": mean_ties if valid else None,
            "max_persona_average": sum(max_persona) / len(max_persona) if valid else None,
            "max_ordinary_average": sum(max_ordinary) / len(max_ordinary) if valid else None,
            "mean_persona_average": sum(mean_persona) / len(mean_persona) if valid else None,
            "mean_ordinary_average": sum(mean_ordinary) / len(mean_ordinary) if valid else None,
            "mean_delta_average": (sum(mean_persona) - sum(mean_ordinary)) / len(mean_persona) if valid else None}


def score_readings(activation_rows: list[dict], units: list[dict], read_set: object) -> dict:
    records, duplicates = _lookup_records(activation_rows)
    unit_results = []
    random_results = []
    shuffled_results = []
    gate1_pairs = []
    expected = len(units) * len(FRAMES) * PAIRS_PER_FRAME
    for frame in FRAMES:
        for unit in units:
            unit_id = str(unit["id"])
            for pair_index in range(PAIRS_PER_FRAME):
                gate1_pairs.append(_token_pair(records, duplicates, frame, unit_id, pair_index))
            unit_result = _compare_series(records, duplicates, unit_id, unit_id, frame, "unit")
            unit_results.append({"unit_id": unit_id, "frame": frame,
                                 "haiku_kind": unit["haiku_kind"], "ds_kind": unit["ds_kind"],
                                 "agreement_group": unit["group"], **unit_result})

            config = (read_set.get("units", {}).get(unit_id)
                      if isinstance(read_set, dict) and isinstance(read_set.get("units"), dict) else None)
            random_rows = config.get("random", []) if isinstance(config, dict) else []
            by_random_id = {row.get("random_id"): row for row in random_rows if isinstance(row, dict)}
            for random_id in range(3):
                compared = _compare_series(records, duplicates, unit_id, unit_id, frame,
                                           "random", random_id)
                random_results.append({"unit_id": unit_id, "random_id": random_id,
                                       "channel": by_random_id.get(random_id, {}).get("channel"),
                                       "frame": frame, **compared})

            partner_map = read_set.get("shuffled_partner", {}) if isinstance(read_set, dict) else {}
            partner = partner_map.get(unit_id) if isinstance(partner_map, dict) else None
            compared = (_compare_series(records, duplicates, unit_id, str(partner), frame, "unit")
                        if partner is not None else
                        {"pairs": 0, "max_wins": 0, "max_losses": 0, "max_ties": 0, "read": None,
                         "mean_wins": None, "mean_losses": None, "mean_ties": None,
                         "max_persona_average": None, "max_ordinary_average": None,
                         "mean_persona_average": None, "mean_ordinary_average": None,
                         "mean_delta_average": None})
            shuffled_results.append({"unit_id": unit_id, "partner_unit_id": partner,
                                     "frame": frame, **compared})

    gate1 = evaluate_gate1(gate1_pairs, expected)
    random_by_frame = {frame: [row["read"] for row in random_results if row["frame"] == frame]
                       for frame in FRAMES}
    gate3 = evaluate_gate3(random_by_frame, len(units) * 3)
    return {"unit_frames": unit_results, "random_controls": random_results,
            "shuffled_controls": shuffled_results, "gate1": gate1, "gate3": gate3,
            "duplicate_activation_keys": len(duplicates)}


def evaluate_gates(activation_rows: list[dict], gate2: object, read_set: object,
                   units: list[dict]) -> tuple[dict, dict]:
    readings = score_readings(activation_rows, units, read_set)
    pairs = []
    records, duplicates = _lookup_records(activation_rows)
    for frame in FRAMES:
        for unit in units:
            for pair_index in range(PAIRS_PER_FRAME):
                pairs.append(_token_pair(records, duplicates, frame, str(unit["id"]), pair_index))
    gate1 = evaluate_gate1(pairs, len(units) * len(FRAMES) * PAIRS_PER_FRAME)
    gate2_result = evaluate_gate2(gate2)
    gate3_result = readings["gate3"]
    gates = {"gate1": gate1, "gate2": gate2_result, "gate3": gate3_result,
             "all_passed": bool(gate1["passed"] and gate2_result["passed"] and gate3_result["passed"]),
             "first_failed": next((key for key in ("gate1", "gate2", "gate3") if not {
                 "gate1": gate1, "gate2": gate2_result, "gate3": gate3_result}[key]["passed"]), None)}
    return gates, readings


def _kind_rates(unit_frames: list[dict], units: list[dict]) -> list[dict]:
    source_by_id = {str(unit["id"]): unit for unit in units}
    output = []
    for frame in FRAMES:
        for kind in ("concept", "slot"):
            selected = [row for row in unit_frames if row["frame"] == frame and row["haiku_kind"] == kind]
            agreed = [row for row in selected
                      if source_by_id[row["unit_id"]]["ds_kind"] == kind]
            readable = [row for row in selected if type(row["read"]) is bool]
            agreed_readable = [row for row in agreed if type(row["read"]) is bool]
            reads = sum(row["read"] is True for row in readable)
            agreed_reads = sum(row["read"] is True for row in agreed_readable)
            output.append({"frame": frame, "haiku_kind": kind, "units": len(selected),
                           "complete_units": len(readable), "reads": reads,
                           "rate": reads / len(readable) if readable else None,
                           "agreed_units": len(agreed), "agreed_complete_units": len(agreed_readable),
                           "agreed_reads": agreed_reads,
                           "agreed_rate": agreed_reads / len(agreed_readable) if agreed_readable else None})
    return output


def _count_read(rows: list[dict], frame: str, kind: str) -> tuple[int, int, int]:
    chosen = [row for row in rows if row["frame"] == frame and row.get("haiku_kind") == kind]
    return (sum(row.get("read") is True for row in chosen),
            sum(type(row.get("read")) is bool for row in chosen), len(chosen))


def prediction_verdicts(unit_frames: list[dict], units: list[dict], shuffled: list[dict],
                        jlens_ids: list[str]) -> dict:
    predictions = {}

    def put(key: str, available: bool, passed: bool, metrics: dict):
        predictions[key] = {"prediction": PREDICTION_TEXT[key], "confidence": PREDICTION_CONFIDENCE[key],
                            "available": bool(available), "passed": bool(passed), "metrics": metrics,
                            "verdict": ("supported" if passed else "not supported") if available else
                            "not evaluable (missing value)"}

    concept_t, concept_complete, concept_expected = _count_read(unit_frames, "T", "concept")
    slot_t, slot_complete, slot_expected = _count_read(unit_frames, "T", "slot")
    p1_avail = concept_complete == concept_expected == 18
    p2_avail = slot_complete == slot_expected == 21
    put("122.1", p1_avail, p1_avail and concept_t >= 9,
        {"concept_reads": concept_t, "complete_units": concept_complete, "expected_units": 18, "threshold": 9})
    put("122.2", p2_avail, p2_avail and slot_t <= 5,
        {"slot_reads": slot_t, "complete_units": slot_complete, "expected_units": 21, "threshold": 5})

    fisher = fisher_exact_two_sided(concept_t, 18 - concept_t, slot_t, 21 - slot_t) if p1_avail and p2_avail else None
    p3_avail = p1_avail and p2_avail and fisher is not None
    concept_rate = concept_t / 18 if p1_avail else None
    slot_rate = slot_t / 21 if p2_avail else None
    put("122.3", p3_avail,
        p3_avail and concept_rate > slot_rate and fisher < 0.05,
        {"concept_rate": concept_rate, "slot_rate": slot_rate, "fisher_two_sided_p": fisher,
         "threshold": 0.05, "direction": "concept > slot"})

    slot_c, slot_c_complete, slot_c_expected = _count_read(unit_frames, "C", "slot")
    p4_avail = slot_complete == slot_expected == 21 and slot_c_complete == slot_c_expected == 21
    slot_gain = slot_c - slot_t
    put("122.4", p4_avail, p4_avail and slot_gain >= 5,
        {"slot_T_reads": slot_t, "slot_C_reads": slot_c, "gain": slot_gain,
         "threshold": 5, "expected_units_per_frame": 21})

    concept_c, concept_c_complete, concept_c_expected = _count_read(unit_frames, "C", "concept")
    p5_avail = p1_avail and p4_avail and concept_c_complete == concept_c_expected == 18
    concept_gain = concept_c - concept_t
    slot_gain_share = slot_gain / 21 if p4_avail else None
    concept_gain_share = concept_gain / 18 if p5_avail else None
    put("122.5", p5_avail,
        p5_avail and slot_gain_share > concept_gain_share,
        {"slot_gain": slot_gain, "slot_gain_share": slot_gain_share,
         "concept_gain": concept_gain, "concept_gain_share": concept_gain_share,
         "comparison": "slot share > concept share"})

    shuffled_by_frame = {}
    p6_available = True
    for frame in FRAMES:
        selected = [row for row in shuffled if row.get("frame") == frame]
        complete = len(selected) == len(units) and all(type(row.get("read")) is bool for row in selected)
        count = sum(row.get("read") is True for row in selected)
        rate = count / len(units) if units else None
        shuffled_by_frame[frame] = {"reads": count, "complete_units": sum(type(row.get("read")) is bool for row in selected),
                                    "expected_units": len(units), "rate": rate, "threshold": 0.10,
                                    "passed": bool(complete and rate is not None and rate <= 0.10)}
        p6_available &= complete and rate is not None
    put("122.6", p6_available,
        p6_available and all(row["passed"] for row in shuffled_by_frame.values()),
        {"by_frame": shuffled_by_frame, "threshold": 0.10})

    jlens_ids = [str(unit_id) for unit_id in jlens_ids]
    t_by_id = {row["unit_id"]: row for row in unit_frames if row["frame"] == "T"}
    jlens_rows = [t_by_id.get(unit_id) for unit_id in jlens_ids]
    jlens_complete = len(jlens_ids) == 11 and all(row is not None and type(row.get("read")) is bool
                                                  for row in jlens_rows)
    jlens_reads = sum(row.get("read") is True for row in jlens_rows if row is not None)
    put("122.7", jlens_complete, jlens_complete and jlens_reads >= 7,
        {"jlens_units": len(jlens_ids), "complete_units": sum(row is not None and type(row.get("read")) is bool
                                                             for row in jlens_rows),
         "reads": jlens_reads, "threshold": 7})
    return predictions


def reading_branch(predictions: dict) -> dict:
    """Apply the four registered reading branches to evaluable prediction verdicts."""
    required = ("122.1", "122.2", "122.3")
    if any(not predictions.get(key, {}).get("available", False) for key in required):
        return {"branch": 4, "text": "The numbers are reported; nothing is read from them."}
    if not predictions["122.3"]["passed"]:
        return {"branch": 3,
                "text": "These kinds, sorted on writing, do not differ in reading their persona on these passages."}
    if all(predictions[key]["passed"] for key in required):
        if not predictions.get("122.4", {}).get("available", False):
            return {"branch": 4, "text": "The numbers are reported; nothing is read from them."}
        if predictions["122.4"]["passed"]:
            return {"branch": 1,
                    "text": "Concept neurons read their theme in plain text; slot neurons do not, and read their persona more when the assistant speaks as it."}
        return {"branch": 2,
                "text": "Concept neurons read their theme in plain text and slot neurons do not, in any frame tested."}
    return {"branch": 4, "text": "The numbers are reported; nothing is read from them."}


def score_results(activation_rows: list[dict], gate2: object, read_set: object,
                  units: list[dict], jlens_ids: list[str]) -> dict:
    gates, readings = evaluate_gates(activation_rows, gate2, read_set, units)
    kind_rates = _kind_rates(readings["unit_frames"], units)
    predictions = prediction_verdicts(readings["unit_frames"], units,
                                     readings["shuffled_controls"], jlens_ids)
    if not gates["all_passed"]:
        failed = gates["first_failed"]
        for row in predictions.values():
            row["verdict"] = f"not read (gate {failed[-1]})"
    branch = reading_branch(predictions) if gates["all_passed"] else {
        "branch": "gate_stop", "text": f"Nothing is read from these tables because {gates['first_failed']} failed."}
    losses = [{"unit_id": row["unit_id"], "frame": row["frame"], "max_losses": row["max_losses"],
               "max_wins": row["max_wins"]} for row in readings["unit_frames"]
              if row["max_losses"] > row["max_wins"]]
    control_summary = {}
    for frame in FRAMES:
        random_rows = [row for row in readings["random_controls"] if row["frame"] == frame]
        shuffled_rows = [row for row in readings["shuffled_controls"] if row["frame"] == frame]
        random_complete = [row for row in random_rows if type(row["read"]) is bool]
        shuffled_complete = [row for row in shuffled_rows if type(row["read"]) is bool]
        control_summary[frame] = {
            "random_read_controls": sum(row["read"] for row in random_complete),
            "random_total_controls": len(random_rows),
            "random_rate": sum(row["read"] for row in random_complete) / len(random_complete)
            if random_complete else None,
            "random_complete_controls": len(random_complete),
            "shuffled_reads": sum(row["read"] for row in shuffled_complete),
            "shuffled_total_units": len(shuffled_rows),
            "shuffled_rate": sum(row["read"] for row in shuffled_complete) / len(shuffled_complete)
            if shuffled_complete else None,
            "shuffled_complete_units": len(shuffled_complete),
        }
    return {"schema_version": 1, "gates": gates, "unit_frames": readings["unit_frames"],
            "kind_rates": kind_rates, "random_controls": readings["random_controls"],
            "shuffled_controls": readings["shuffled_controls"], "control_summary": control_summary,
            "losses": losses, "predictions": predictions, "reading_branch": branch,
            "duplicate_activation_keys": readings["duplicate_activation_keys"]}


def _fmt(value, digits: int = 3) -> str:
    return "missing" if value is None else f"{value:.{digits}f}"


def build_score_markdown(score: dict) -> str:
    gates = score["gates"]
    lines = ["# Part 122 score", "", "## Gates", "",
             "| Gate | Result | Observed | Threshold |", "| --- | --- | ---: | ---: |",
             f"| 1 token match | {'PASS' if gates['gate1']['passed'] else 'MISS'} | "
             f"{gates['gate1']['within_two']}/{gates['gate1']['observed_pairs']} within 2 tokens; "
             f"{gates['gate1']['observed_pairs']}/{gates['gate1']['expected_pairs']} complete | 90% |",
             f"| 2 Part 95 positive control | {'PASS' if gates['gate2']['passed'] else 'MISS'} | "
             f"{gates['gate2']['space_max_wins']}/{gates['gate2']['observed_pairs']} max wins; "
             f"{gates['gate2']['observed_pairs']}/{gates['gate2']['expected_pairs']} complete | 9/16 |"]
    for frame in FRAMES:
        row = gates["gate3"]["by_frame"][frame]
        lines.append(f"| 3 random control {frame} | {'PASS' if row['passed'] else 'MISS'} | "
                     f"{row['read_controls']}/{row['observed_cells']} read; {row['observed_cells']}/{row['expected_cells']} complete | 15% |")
    if not gates["all_passed"]:
        lines += ["", f"## Activation tables: nothing is read from them because gate {gates['first_failed'][-1]} failed", ""]
    else:
        lines += ["", "## Activation tables", ""]
    lines += ["### Unit and frame results", "",
              "| Unit | Frame | Haiku | DS | Max wins | Losses | Ties | Read (7/8) | Max persona | Max ordinary | Mean wins | Mean persona | Mean ordinary | Mean delta |",
              "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in score["unit_frames"]:
        lines.append(f"| {row['unit_id']} | {row['frame']} | {row['haiku_kind']} | {row['ds_kind']} | "
                     f"{row['max_wins']}/{row['pairs']} | {row['max_losses']} | {row['max_ties']} | "
                     f"{row['read'] if row['read'] is not None else 'missing'} | "
                     f"{_fmt(row['max_persona_average'])} | {_fmt(row['max_ordinary_average'])} | "
                     f"{row['mean_wins'] if row['mean_wins'] is not None else 'missing'} | "
                     f"{_fmt(row['mean_persona_average'])} | {_fmt(row['mean_ordinary_average'])} | "
                     f"{_fmt(row['mean_delta_average'])} |")
    lines += ["", "### Rates by Haiku kind and agreed kind", "",
              "| Frame | Haiku kind | Reads / complete | Rate | Agreed reads / complete | Agreed rate |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in score["kind_rates"]:
        lines.append(f"| {row['frame']} | {row['haiku_kind']} | {row['reads']}/{row['complete_units']} "
                     f"(n={row['units']}) | {_fmt(row['rate'])} | {row['agreed_reads']}/"
                     f"{row['agreed_complete_units']} (n={row['agreed_units']}) | {_fmt(row['agreed_rate'])} |")
    lines += ["", "### Controls", "", "| Frame | Random reads / cells | Random rate | Shuffled reads / units | Shuffled rate |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for frame, row in score["control_summary"].items():
        lines.append(f"| {frame} | {row['random_read_controls']}/{row['random_complete_controls']} "
                     f"(n={row['random_total_controls']}) | {_fmt(row['random_rate'])} | "
                     f"{row['shuffled_reads']}/{row['shuffled_complete_units']} "
                     f"(n={row['shuffled_total_units']}) | {_fmt(row['shuffled_rate'])} |")
    lines += ["", "### Predictions", "", "| # | Confidence | Result | Verdict |", "| --- | --- | --- | --- |"]
    for key in (f"122.{i}" for i in range(1, 8)):
        pred = score["predictions"][key]
        lines.append(f"| {key} | {pred['confidence']} | {pred['metrics']} | {pred['verdict']} |")
    lines += ["", "### Registered reading", "", score["reading_branch"]["text"], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", default=f"{DEFAULT_OUT}/activations.jsonl")
    parser.add_argument("--gate2", default=f"{DEFAULT_OUT}/gate2.json")
    parser.add_argument("--read-set", default=f"{DEFAULT_OUT}/read_set.json")
    parser.add_argument("--neurons", default=rows119.DEFAULT_NEURONS)
    parser.add_argument("--kinds-haiku", default=rows119.DEFAULT_HAIKU_KINDS)
    parser.add_argument("--kinds-ds", default=rows119.DEFAULT_DS_KINDS)
    parser.add_argument("--jlens", default=str(HERE / "jlens-concept-matches.json"))
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    units = rows119.load_units(args.neurons, args.kinds_haiku, args.kinds_ds)
    activation_rows = read_jsonl(args.activations) if repo_path(args.activations).is_file() else []
    gate2 = read_json(args.gate2) if repo_path(args.gate2).is_file() else None
    read_set = read_json(args.read_set) if repo_path(args.read_set).is_file() else None
    jlens_data = read_json(args.jlens)
    jlens_units = jlens_data.get("units", []) if isinstance(jlens_data, dict) else []
    jlens_ids = [str(row["id"]) for row in jlens_units if isinstance(row, dict) and "id" in row]
    score = score_results(activation_rows, gate2, read_set, units, jlens_ids)
    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "score122.json").open("w", encoding="utf-8") as stream:
        json.dump(score, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    (out_dir / "SCORE.md").write_text(build_score_markdown(score), encoding="utf-8")
    print(json.dumps({"all_gates_passed": score["gates"]["all_passed"],
                      "first_failed_gate": score["gates"]["first_failed"],
                      "predictions": {key: value["verdict"] for key, value in score["predictions"].items()},
                      "score": str(out_dir / "score122.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
