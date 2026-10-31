#!/usr/bin/env python3
"""Apply Part 143 gates and registered predictions to the measured probabilities."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREDICTION_TEXT = {
    "143.1": ("After forced I'm, Set R median prompt-only P(' sorry') is at least 10 times plain.", "medium"),
    "143.2": ("After forced I'm, Set R median answer-only P(' sorry') is at least 10 times plain.", "medium"),
    "143.3": ("After forced I, Set R median answer-only P('m') is below prompt-only.", "low"),
    "143.4": ("Under full, Set R median P(' sorry') is at least twice the larger prompt-only or answer-only median.", "medium"),
    "143.5": ("Under full, Set S median P(' sorry') is at most one tenth of Set R.", "high"),
}


def read_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _prob(row):
    value = row.get("probability", row.get("sorry_probability"))
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"row has missing or non-finite probability: {row.get('key')}")
    return float(value)


def _median(rows):
    if not rows:
        return None
    return statistics.median(_prob(row) for row in rows)


def score(rows, part139_rows, expected_unit_ids=None, expected_r_ids=None):
    readouts = ("im_after_i", "sorry_after_im")
    plain = {row.get("readout"): row for row in rows if row.get("arm") == "plain"}
    noop = {(row.get("readout"), row.get("unit_id")): row for row in rows if row.get("arm") == "noop"}
    if expected_unit_ids is None:
        expected_unit_ids = sorted({row.get("unit_id") for row in rows if row.get("arm") not in (None, "plain") and row.get("unit_id") is not None})
    gate1_differences = []
    for readout in readouts:
        for unit_id in expected_unit_ids:
            row = noop.get((readout, unit_id))
            baseline = plain.get(readout)
            if row is None or baseline is None:
                gate1_differences.append({"readout": readout, "unit_id": unit_id, "absolute_difference": None, "passed": False, "missing_row_or_baseline": True})
                continue
            difference = abs(_prob(row) - _prob(baseline))
            gate1_differences.append({"readout": readout, "unit_id": unit_id, "absolute_difference": difference, "passed": difference <= 0.001})
    gate1_passed = bool(gate1_differences) and all(item["passed"] for item in gate1_differences)

    part139_by_unit = {
        row.get("unit_id"): _prob(row)
        for row in part139_rows
        if row.get("stage") == "A" and row.get("arm") == "forced_prefix" and row.get("unit_id")
        and row.get("unit_id") != "plain" and row.get("set") == "R"
    }
    full_r = [row for row in rows if row.get("readout") == "sorry_after_im" and row.get("arm") == "full" and row.get("set") == "R"]
    expected_ids = set(expected_r_ids) if expected_r_ids is not None else set(part139_by_unit)
    if expected_ids - set(part139_by_unit):
        raise ValueError(f"Part 139 stored rows missing Set R units: {sorted(expected_ids - set(part139_by_unit))}")
    part139_by_unit = {unit_id: part139_by_unit[unit_id] for unit_id in expected_ids}
    full_by_unit = {row.get("unit_id"): row for row in full_r}
    unexpected = set(full_by_unit) - set(part139_by_unit)
    if unexpected:
        raise ValueError(f"Part 139 stored Set R rows missing for full rows: {sorted(unexpected)}")
    gate2_differences = []
    for unit_id, prior_probability in part139_by_unit.items():
        row = full_by_unit.get(unit_id)
        if row is None:
            gate2_differences.append({"unit_id": unit_id, "absolute_difference": None, "passed": False, "missing_current_row": True})
            continue
        difference = abs(_prob(row) - prior_probability)
        gate2_differences.append({"unit_id": unit_id, "absolute_difference": difference, "passed": difference <= 0.05})
    gate2_passed_n = sum(item["passed"] for item in gate2_differences)
    gate2_fraction = gate2_passed_n / len(part139_by_unit) if part139_by_unit else 0.0
    gates = {
        "gate1_noop_matches_plain": {
            "threshold_absolute_difference": 0.001,
            "passed": gate1_passed,
            "rows": sum(item["absolute_difference"] is not None for item in gate1_differences),
            "expected_rows": len(gate1_differences),
            "max_absolute_difference": max((item["absolute_difference"] for item in gate1_differences if item["absolute_difference"] is not None), default=None),
            "comparisons": gate1_differences,
        },
        "gate2_full_matches_part139": {
            "threshold_absolute_difference": 0.05,
            "required_fraction": 0.90,
            "passed": gate2_fraction >= 0.90,
            "n": len(part139_by_unit), "matching_rows": gate2_passed_n,
            "fraction_matching": gate2_fraction,
            "comparisons": gate2_differences,
        },
    }
    if not all(gate["passed"] for gate in gates.values()):
        return {"gates": gates}

    cells = defaultdict(list)
    for row in rows:
        if row.get("arm") == "plain":
            key = (row["readout"], "plain", "plain", "plain")
        else:
            key = (row["readout"], row.get("arm"), row.get("set"), row.get("unit"))
        cells[key].append(_prob(row))
    medians = {}
    for (readout, arm, set_name, unit_type), values in sorted(cells.items()):
        medians.setdefault(readout, {}).setdefault(arm, {}).setdefault(set_name, {})[unit_type] = {
            "n": len(values), "median": statistics.median(values),
            "middle_50_percent": {"q1": _quantile(values, .25), "q3": _quantile(values, .75)},
        }

    # Predictions use all units in the registered set, combining neurons and random directions.
    def set_median(readout, arm, set_name):
        values = [row for row in rows if row.get("readout") == readout and row.get("arm") == arm and row.get("set") == set_name]
        if not values:
            raise ValueError(f"missing values for {readout} {arm} {set_name}")
        return _median(values)

    plain_sorry = _prob(plain["sorry_after_im"])
    prompt_r = set_median("sorry_after_im", "prompt-only", "R")
    answer_r = set_median("sorry_after_im", "answer-only", "R")
    full_r_median = set_median("sorry_after_im", "full", "R")
    full_s_median = set_median("sorry_after_im", "full", "S")
    pred_values = {
        "143.1": {"value": prompt_r, "plain_probability": plain_sorry, "ratio": prompt_r / plain_sorry if plain_sorry else None, "held": bool(plain_sorry) and prompt_r >= 10 * plain_sorry},
        "143.2": {"value": answer_r, "plain_probability": plain_sorry, "ratio": answer_r / plain_sorry if plain_sorry else None, "held": bool(plain_sorry) and answer_r >= 10 * plain_sorry},
        "143.3": {
            "answer_only_median": set_median("im_after_i", "answer-only", "R"),
            "prompt_only_median": set_median("im_after_i", "prompt-only", "R"),
            "held": set_median("im_after_i", "answer-only", "R") < set_median("im_after_i", "prompt-only", "R"),
        },
        "143.4": {"full_median": full_r_median, "prompt_only_median": prompt_r, "answer_only_median": answer_r,
                  "threshold": 2 * max(prompt_r, answer_r), "held": full_r_median >= 2 * max(prompt_r, answer_r)},
        "143.5": {"set_s_full_median": full_s_median, "set_r_full_median": full_r_median,
                  "threshold": full_r_median / 10, "held": full_s_median <= full_r_median / 10},
    }
    for prediction_id, result in pred_values.items():
        result["verdict"] = "held" if result["held"] else "failed"
        result["text"], result["confidence"] = PREDICTION_TEXT[prediction_id]
    return {"gates": gates, "medians": medians, "predictions": pred_values}


def render_markdown(result):
    lines = ["# Part 143 score", ""]
    for gate_id, gate in result["gates"].items():
        lines.append(f"- **{gate_id}: {'passed' if gate['passed'] else 'failed'}** — " +
                     (f"max noop difference {gate['max_absolute_difference'] if gate['max_absolute_difference'] is not None else 'missing'} across {gate['rows']} rows; threshold 0.001."
                      if gate_id.startswith("gate1") else
                      f"{gate['matching_rows']}/{gate['n']} ({gate['fraction_matching']:.1%}) within 0.05; required 90%."))
    if "predictions" not in result:
        lines.extend(["", "Gates failed; registered predictions are withheld."])
        return "\n".join(lines) + "\n"
    lines.extend(["", "## Registered predictions", "", "| Prediction | Result | Values |", "|---|---|---|"])
    for prediction_id, (text, confidence) in PREDICTION_TEXT.items():
        result_row = result["predictions"][prediction_id]
        details = ", ".join(f"{key}={value:.8g}" for key, value in result_row.items() if isinstance(value, float))
        lines.append(f"| {prediction_id} ({confidence}) | {result_row['verdict']} | {details} |")
    lines.extend(["", "## Medians", "", "Each cell gives median and middle 50% interval (Q1–Q3).", ""])
    lines.append("| Readout | Arm | Set | Unit type | n | Median | Q1 | Q3 |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for readout, arms in sorted(result["medians"].items()):
        for arm, sets in sorted(arms.items()):
            for set_name, unit_types in sorted(sets.items()):
                for unit_type, stats in sorted(unit_types.items()):
                    lines.append(f"| {readout} | {arm} | {set_name} | {unit_type} | {stats['n']} | {stats['median']:.8g} | {stats['middle_50_percent']['q1']:.8g} | {stats['middle_50_percent']['q3']:.8g} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "results/part143/results.jsonl"))
    parser.add_argument("--part139", default=str(ROOT / "results/part139/results.jsonl"))
    parser.add_argument("--units", default=str(ROOT / "part139/units139.json"))
    parser.add_argument("--out", default=str(ROOT / "results/part143"))
    args = parser.parse_args(argv)
    units = json.loads(Path(args.units).read_text(encoding="utf-8"))
    expected_r_ids = [unit["id"] for unit in units["sets"]["R"]]
    expected_unit_ids = [unit["id"] for set_name in ("R", "S") for unit in units["sets"][set_name]]
    result = score(read_rows(args.input), read_rows(args.part139), expected_unit_ids, expected_r_ids)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "score.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "SCORE.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
