#!/usr/bin/env python3
"""CPU-only Part 119c gates, summaries, registered predictions, and headline."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT / "part119", ROOT / "part119b", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rows119  # noqa: E402
import rows119c  # noqa: E402
import score119  # noqa: E402 -- registered unit bootstrap, no reader dependency
from gate119b import interaction_share  # noqa: E402 -- registered share definition

PART119B_PATH = "results/part119b/interaction.jsonl"
BF16_PATH = "results/part119c/interaction-bf16.jsonl"
FLOAT32_PATH = "results/part119c/interaction-float32.jsonl"
SCORE_DOSES = (0.05, 0.1, 0.2, 0.4, 0.8)
GATE1_DOSES = (0.2, 0.4, 0.8)
KINDS = ("real", "control")


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: str | Path) -> list[dict]:
    path = repo_path(path)
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


def _row_key(row: dict) -> tuple[str, str, str]:
    return str(row["kind"]), str(row["first"]), str(row["second"])


def _index(rows: list[dict]) -> dict[tuple[str, str, str], dict[float, dict]]:
    result: dict[tuple[str, str, str], dict[float, dict]] = defaultdict(dict)
    for row in rows:
        key, dose = _row_key(row), float(row["dose"])
        if dose in result[key]:
            raise ValueError(f"duplicate Path 2 row/dose {key} at {dose}")
        result[key][dose] = row
    return result


def _position_map(row: dict) -> dict[int, dict]:
    result = {}
    for position in row.get("positions", []):
        pos = int(position["pos"])
        if pos in result:
            raise ValueError(f"duplicate Path 2 position {pos}")
        result[pos] = position
    return result


def _median_or_none(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _reproduction_gate(current: list[dict], reference: list[dict],
                       selected_rows: list[dict], doses: tuple[float, ...],
                       threshold: float, gate_name: str) -> dict:
    """Compare row-median relative nPair error and require complete row support."""
    current_index, reference_index = _index(current), _index(reference)
    by_dose: dict[str, dict] = {}
    all_relative = []
    missing, position_mismatch, zero_reference = [], [], []
    row_medians_by_dose: dict[float, list[float]] = {float(dose): [] for dose in doses}
    missing_by_dose = {float(dose): [] for dose in doses}
    mismatch_by_dose = {float(dose): [] for dose in doses}
    zero_by_dose = {float(dose): [] for dose in doses}
    for dose in doses:
        dose = float(dose)
        for source in selected_rows:
            key = _row_key(source)
            newer = current_index.get(key, {}).get(dose)
            older = reference_index.get(key, {}).get(dose)
            if newer is None or older is None:
                issue = {"kind": key[0], "first": key[1], "second": key[2],
                         "dose": dose,
                         "missing_from": "current" if newer is None else "reference"}
                missing.append(issue)
                missing_by_dose[dose].append(issue)
                continue
            new_pos, old_pos = _position_map(newer), _position_map(older)
            if set(new_pos) != set(old_pos):
                issue = {"kind": key[0], "first": key[1],
                         "second": key[2], "dose": dose}
                position_mismatch.append(issue)
                mismatch_by_dose[dose].append(issue)
            relative = []
            for pos in sorted(set(new_pos) & set(old_pos)):
                before = float(old_pos[pos].get("nPair", 0.0))
                after = float(new_pos[pos].get("nPair", 0.0))
                if before > 0:
                    relative.append(abs(after - before) / before)
                elif after == 0:
                    relative.append(0.0)
                else:
                    issue = {"kind": key[0], "first": key[1],
                             "second": key[2], "dose": dose, "pos": pos}
                    zero_reference.append(issue)
                    zero_by_dose[dose].append(issue)
            row_value = _median_or_none(relative)
            if row_value is not None:
                row_medians_by_dose[dose].append(row_value)
                all_relative.append(row_value)

    for dose in doses:
        value = _median_or_none(row_medians_by_dose[float(dose)])
        complete = (not missing_by_dose[float(dose)]
                    and not mismatch_by_dose[float(dose)]
                    and not zero_by_dose[float(dose)]
                    and len(row_medians_by_dose[float(dose)]) == len(selected_rows))
        by_dose[f"{float(dose):g}"] = {
            "median_relative_nPair_difference": value,
            "rows_compared": len(row_medians_by_dose[float(dose)]),
            "expected_rows": len(selected_rows),
            "complete": complete,
            "passed": bool(complete and value is not None and value < threshold),
        }
    passed = all(value["passed"] for value in by_dose.values())
    return {
        "name": gate_name,
        "passed": bool(passed),
        "threshold": {"median_relative_nPair_difference_below": threshold},
        "by_dose": by_dose,
        "median_relative_nPair_difference_all_checked_rows": _median_or_none(all_relative),
        "missing_rows": missing[:25], "missing_row_count": len(missing),
        "position_mismatch_rows": position_mismatch[:25],
        "position_mismatch_count": len(position_mismatch),
        "zero_reference_nonzero_rows": zero_reference[:25],
        "zero_reference_nonzero_count": len(zero_reference),
    }


def gate1_result(bf16_rows: list[dict], part119b_rows: list[dict],
                 selected_rows: list[dict]) -> dict:
    """119c Gate 1: bf16 reproduces 119b at 0.2, 0.4, and 0.8 within 1%."""
    return _reproduction_gate(bf16_rows, part119b_rows, selected_rows,
                              GATE1_DOSES, 0.01, "bf16 versus Part 119b")


def gate2_result(float32_rows: list[dict], bf16_rows: list[dict],
                 selected_rows: list[dict]) -> dict:
    """119c Gate 2: at 0.8, float32 matches bf16 nPair within 5%."""
    return _reproduction_gate(float32_rows, bf16_rows, selected_rows,
                              (0.8,), 0.05, "float32 versus bf16")


def _row_shares(rows: list[dict]) -> dict[tuple[str, str, str, float], float | None]:
    result = {}
    for row in rows:
        shares = [share for position in row.get("positions", [])
                  if (share := interaction_share(position)) is not None]
        result[(*_row_key(row), float(row["dose"]))] = _median_or_none(shares)
    return result


def summarize(rows_by_dtype: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    """Report position and row medians by dtype, dose, and Path 2 class."""
    table, row_values = [], {}
    for dtype_name in ("bf16", "float32"):
        rows = rows_by_dtype[dtype_name]
        row_share_map = _row_shares(rows)
        row_values[dtype_name] = row_share_map
        for dose in SCORE_DOSES:
            for kind in KINDS:
                class_rows = [row for row in rows
                              if float(row["dose"]) == dose and str(row["kind"]) == kind]
                positions = [position for row in class_rows
                             for position in row.get("positions", [])]
                position_shares = [value for position in positions
                                   if (value := interaction_share(position)) is not None]
                row_shares = [row_share_map[(*_row_key(row), dose)] for row in class_rows]
                row_shares = [value for value in row_shares if value is not None]
                metric_medians = {
                    field: _median_or_none([float(position[field]) for position in positions
                                            if position.get(field) is not None])
                    for field in ("nI", "nPair", "nA", "n2")
                }
                table.append({
                    "dtype": dtype_name, "dose": dose, "kind": kind,
                    "rows": len(class_rows), "rows_with_share": len(row_shares),
                    "positions": len(positions),
                    "positions_with_nPair_above_zero": len(position_shares),
                    "median_nI": metric_medians["nI"],
                    "median_nPair": metric_medians["nPair"],
                    "median_nA": metric_medians["nA"],
                    "median_n2": metric_medians["n2"],
                    "median_share_over_positions": _median_or_none(position_shares),
                    "median_row_share": _median_or_none(row_shares),
                })
    return table, row_values


def _summary_value(table: list[dict], dtype: str, dose: float, kind: str,
                  field: str = "median_row_share") -> float | None:
    for row in table:
        if row["dtype"] == dtype and row["dose"] == dose and row["kind"] == kind:
            return row.get(field)
    return None


def _weighted_difference_ci(differences: list[dict], units: list[dict] | list[str],
                            *, draws: int = 2000) -> dict:
    unit_ids = [str(unit["id"] if isinstance(unit, dict) else unit) for unit in units]
    values = [float(row["value"]) for row in differences]
    point = float(np.mean(values)) if values else None
    bootstrap_draws = score119.make_unit_bootstrap_draws(
        unit_ids, draws=draws, seed=score119.BOOTSTRAP_SEED)
    product_samples, sum_samples = [], []
    for counts in bootstrap_draws:
        for samples, weight_for in (
                (product_samples, lambda first, second: counts.get(first, 0) * counts.get(second, 0)),
                (sum_samples, lambda first, second: counts.get(first, 0) + counts.get(second, 0))):
            numerator = denominator = 0.0
            for row in differences:
                weight = int(weight_for(str(row["first"]), str(row["second"])))
                numerator += float(row["value"]) * weight
                denominator += weight
            if denominator:
                samples.append(numerator / denominator)
    def interval(samples):
        return ([float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]
                if samples else [None, None])
    return {"mean_row_difference": point,
            "ci95_product_weighted_mean_difference": interval(product_samples),
            "ci95_sum_weighted_mean_difference": interval(sum_samples),
            "draws": draws, "seed": score119.BOOTSTRAP_SEED,
            "units": len(unit_ids), "rows": len(differences)}


def predictions(table: list[dict], row_values: dict,
                units: list[dict] | list[str]) -> list[dict]:
    real_bf = row_values["bf16"]
    real_f32 = row_values["float32"]
    diffs = []
    for key, bf16_value in real_bf.items():
        kind, first, second, dose = key
        if kind != "real" or dose != 0.2:
            continue
        f32_value = real_f32.get(key)
        if bf16_value is not None and f32_value is not None:
            diffs.append({"first": first, "second": second,
                          "value": f32_value - bf16_value})
    bootstrap = _weighted_difference_ci(diffs, units)
    bf16_02 = _summary_value(table, "bf16", 0.2, "real")
    f32_02 = _summary_value(table, "float32", 0.2, "real")
    median_difference = (f32_02 - bf16_02
                         if f32_02 is not None and bf16_02 is not None else None)
    product_ci = bootstrap["ci95_product_weighted_mean_difference"]
    pred1_available = (median_difference is not None and product_ci[0] is not None
                       and product_ci[1] is not None)
    pred1_holds = (pred1_available and abs(median_difference) <= .04
                   and product_ci[0] >= -.06 and product_ci[1] <= .06)
    pred1 = {
        "id": "119c.1",
        "threshold": "At 0.2, absolute difference between median row shares <= 0.04 and the product-weighted 95% unit-bootstrap CI lies in [-0.06, 0.06].",
        "observed": {"bf16_median_row_share": bf16_02,
                     "float32_median_row_share": f32_02,
                     "median_difference_float32_minus_bf16": median_difference,
                     "bootstrap": bootstrap},
        "verdict": "holds" if pred1_holds else "does_not_hold" if pred1_available else "not_evaluable",
    }

    f32_01 = _summary_value(table, "float32", 0.1, "real")
    f32_005 = _summary_value(table, "float32", 0.05, "real")
    pred2_available = f32_01 is not None and f32_005 is not None
    pred2_holds = pred2_available and f32_01 <= .10 and f32_005 <= .05
    pred2 = {
        "id": "119c.2",
        "threshold": "Float32 real median row share <= 0.10 at 0.1 and <= 0.05 at 0.05.",
        "observed": {"float32_real_median_row_share_0.1": f32_01,
                     "float32_real_median_row_share_0.05": f32_005},
        "verdict": "holds" if pred2_holds else "does_not_hold" if pred2_available else "not_evaluable",
    }

    slope_points = []
    for dose in (0.05, 0.1, 0.2):
        share = _summary_value(table, "float32", dose, "real")
        if share is None or share <= 0:
            slope_points = []
            break
        slope_points.append((math.log(dose), math.log(share)))
    slope = None
    if len(slope_points) == 3:
        x = np.asarray([point[0] for point in slope_points], dtype=np.float64)
        y = np.asarray([point[1] for point in slope_points], dtype=np.float64)
        design = np.column_stack((np.ones(3), x))
        coeff, _residuals, _rank, _singular = np.linalg.lstsq(design, y, rcond=None)
        slope = float(coeff[1])
    pred3 = {
        "id": "119c.3",
        "threshold": "Float32 real median row share log-log least-squares slope over 0.05, 0.1, and 0.2 >= 1.5.",
        "observed": {"doses": [0.05, 0.1, 0.2],
                     "median_row_shares": [f32_005, f32_01, f32_02],
                     "log_log_slope": slope},
        "verdict": "holds" if slope is not None and slope >= 1.5 else
                   "does_not_hold" if slope is not None else "not_evaluable",
    }

    control_ratios = []
    for dose in (0.05, 0.1, 0.2):
        real = _summary_value(table, "float32", dose, "real")
        control = _summary_value(table, "float32", dose, "control")
        if real is None or control is None:
            factor, matched = None, False
        elif real == 0 and control == 0:
            factor, matched = 1.0, True
        elif real <= 0 or control <= 0:
            factor, matched = None, False
        else:
            factor = max(real / control, control / real)
            matched = factor <= 1.25
        control_ratios.append({"dose": dose, "real_median_row_share": real,
                               "control_median_row_share": control,
                               "larger_to_smaller_factor": factor,
                               "within_factor_1.25": matched})
    pred4_available = all(item["larger_to_smaller_factor"] is not None
                          for item in control_ratios)
    pred4_holds = pred4_available and all(item["within_factor_1.25"]
                                          for item in control_ratios)
    pred4 = {
        "id": "119c.4",
        "threshold": "At each of 0.05, 0.1, and 0.2, float32 real median row share is within a factor of 1.25 of control.",
        "observed": {"dose_comparisons": control_ratios},
        "verdict": "holds" if pred4_holds else "does_not_hold" if pred4_available else "not_evaluable",
    }
    return [pred1, pred2, pred3, pred4]


def headline_by_rule(prediction_rows: list[dict], gates: dict) -> str:
    if not gates["gate1"]["passed"] or not gates["gate2"]["passed"]:
        failed = [name for name in ("gate1", "gate2") if not gates[name]["passed"]]
        return "Stopped by " + " and ".join(failed) + "; no prediction is read."
    by_id = {row["id"]: row for row in prediction_rows}
    if (by_id["119c.1"]["verdict"] == "holds"
            and by_id["119c.2"]["verdict"] == "holds"):
        return ("Part 119b's low-dose share is not a rounding floor; the interaction shrinks "
                "toward zero with dose, and at 0.2 it is already measurable where answers do not break.")
    pred1 = by_id["119c.1"]["observed"]
    difference = pred1.get("median_difference_float32_minus_bf16")
    interval = pred1.get("bootstrap", {}).get("ci95_product_weighted_mean_difference", [None, None])
    if (difference is not None and difference < -.04 and interval[1] is not None
            and interval[1] < 0):
        return "bf16 rounding inflates the low-dose share, and Part 119b's gate 3 miss is at least partly that."
    return "No registered headline reading; report the numbers."


def score_interactions(bf16_rows: list[dict], float32_rows: list[dict],
                       part119b_rows: list[dict], selected_rows: list[dict],
                       units: list[dict] | list[str]) -> dict:
    gate1 = gate1_result(bf16_rows, part119b_rows, selected_rows)
    gate2 = gate2_result(float32_rows, bf16_rows, selected_rows)
    table, row_values = summarize({"bf16": bf16_rows, "float32": float32_rows})
    pred_rows = predictions(table, row_values, units)
    gates = {"gate1": gate1, "gate2": gate2,
             "all_passed": gate1["passed"] and gate2["passed"]}
    if not gates["all_passed"]:
        for row in pred_rows:
            row["verdict"] = "stopped_by_gate"
    return {"part": "119c", "selected_rows_expected": len(selected_rows),
            "selected_rows_by_class": {kind: sum(row["kind"] == kind for row in selected_rows)
                                       for kind in KINDS},
            "gates": gates, "per_dose": table, "predictions": pred_rows,
            "headline": headline_by_rule(pred_rows, gates),
            "bootstrap_definition": "Part 119 draws (2,000, seed 11902); product unit-count weighting is primary and sum weighting is sensitivity.",
            "readout": "Share includes positions with nPair > 0; each row is the median of those positions; summary is the median of row values."}


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def render_markdown(result: dict) -> str:
    lines = ["# Part 119c score", "",
             f"Selected Path 2 rows: {result['selected_rows_expected']} "
             f"({result['selected_rows_by_class']}).", "",
             "## Gates", "",
             "| Gate | Passed | Median relative nPair difference |", "| --- | --- | ---: |"]
    for name in ("gate1", "gate2"):
        gate = result["gates"][name]
        lines.append(f"| {name} | {gate['passed']} | "
                     f"{_fmt(gate['median_relative_nPair_difference_all_checked_rows'])} |")
        for dose, detail in gate["by_dose"].items():
            lines.append(f"| {name} at {dose} | {detail['passed']} | "
                         f"{_fmt(detail['median_relative_nPair_difference'])} |")
    lines.extend(["", "## Predictions", "",
                  "| Prediction | Verdict | Observed |", "| --- | --- | --- |"])
    for prediction in result["predictions"]:
        observed = json.dumps(prediction["observed"], ensure_ascii=False, sort_keys=True)
        lines.append(f"| {prediction['id']} | {prediction['verdict']} | `{observed}` |")
    lines.extend(["", "## Per-dose medians", "",
                  "Share uses positions with nPair > 0; row share is the median of a row's eligible positions; the summary is the median over rows.",
                  "", "| Dtype | Dose | Class | Rows | Positions nPair>0 | Median nI | Median nPair | Median nA | Median n2 | Position share | Row share |",
                  "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in result["per_dose"]:
        lines.append("| " + " | ".join(_fmt(row[key]) for key in (
            "dtype", "dose", "kind", "rows", "positions_with_nPair_above_zero",
            "median_nI", "median_nPair", "median_nA", "median_n2",
            "median_share_over_positions", "median_row_share")) + " |")
    lines.extend(["", "## Headline", "", result["headline"], "",
                  result["readout"], result["bootstrap_definition"], ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bf16", default=BF16_PATH)
    parser.add_argument("--float32", default=FLOAT32_PATH)
    parser.add_argument("--reference", default=PART119B_PATH)
    parser.add_argument("--out", default="results/part119c")
    parser.add_argument("--neurons", default=rows119.DEFAULT_NEURONS)
    parser.add_argument("--kinds", default=rows119.DEFAULT_HAIKU_KINDS)
    parser.add_argument("--kinds-ds", default=rows119.DEFAULT_DS_KINDS)
    args = parser.parse_args(argv)
    units = rows119.load_units(args.neurons, args.kinds, args.kinds_ds)
    selected_rows = rows119c.draw_path2_rows(units)
    result = score_interactions(read_jsonl(args.bf16), read_jsonl(args.float32),
                                read_jsonl(args.reference), selected_rows, units)
    output_dir = repo_path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "score119c.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    (output_dir / "SCORE.md").write_text(render_markdown(result), encoding="utf-8")
    print(result["headline"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
