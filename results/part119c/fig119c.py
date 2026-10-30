#!/usr/bin/env python3
"""Recompute Part 119c summaries from its JSONL rows and draw its figure."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "part119", ROOT / "part119b", ROOT / "part119c", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rows119  # noqa: E402
import rows119c  # noqa: E402
import score119  # noqa: E402
import score119c  # noqa: E402
from gate119b import interaction_share  # noqa: E402
from score119c import read_jsonl, score_interactions  # noqa: E402


DOSES = (0.05, 0.1, 0.2, 0.4, 0.8)
OUT = ROOT / "results" / "part119c"
BF16_FILE = OUT / "interaction-bf16.jsonl"
FLOAT32_FILE = OUT / "interaction-float32.jsonl"
REFERENCE_FILE = ROOT / "results" / "part119b" / "interaction.jsonl"


def f(value: object, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def source(dtype: str) -> str:
    return (f"`results/part119c/interaction-{dtype}.jsonl`: `$.dose`, `$.kind`, `$.first`, `$.second`, "
            "`$.positions[].pos`, `$.positions[].nI`, `$.positions[].nA`, `$.positions[].n2`, "
            "`$.positions[].nPair`")


def find_summary(result: dict, dtype: str, dose: float, kind: str) -> dict:
    return next(row for row in result["per_dose"]
                if row["dtype"] == dtype and row["dose"] == dose and row["kind"] == kind)


def _row_median_share(row: dict, *, resolved_only: bool = False) -> float | None:
    shares = []
    for position in row.get("positions", []):
        if resolved_only and not all(
                position.get(field) is not None and float(position[field]) >= 1.0
                for field in ("nA", "n2", "nPair")):
            continue
        share = interaction_share(position)
        if share is not None:
            shares.append(float(share))
    return float(np.median(shares)) if shares else None


def _product_bootstrap_share_interval(rows: list[dict], dose: float,
                                      units: list[dict] | list[str]) -> dict:
    row_shares = score119c._row_shares(rows)
    selected = []
    for row in rows:
        if float(row["dose"]) != dose or str(row["kind"]) != "real":
            continue
        key = (str(row["kind"]), str(row["first"]), str(row["second"]), dose)
        value = row_shares.get(key)
        if value is not None:
            selected.append({"first": str(row["first"]),
                             "second": str(row["second"]), "value": float(value)})
    values = [row["value"] for row in selected]
    draws = score119.make_unit_bootstrap_draws(
        [str(unit["id"] if isinstance(unit, dict) else unit) for unit in units],
        draws=2000, seed=score119.BOOTSTRAP_SEED)
    samples = []
    for counts in draws:
        weights = [counts.get(row["first"], 0) * counts.get(row["second"], 0)
                   for row in selected]
        value = score119c._weighted_median(values, weights)
        if value is not None:
            samples.append(value)
    return {
        "dose": dose,
        "median_row_share": float(np.median(values)) if values else None,
        "ci95_product_weighted": ([float(np.quantile(samples, .025)),
                                    float(np.quantile(samples, .975))]
                                   if samples else [None, None]),
        "draws": len(samples), "seed": score119.BOOTSTRAP_SEED,
        "units": len(units), "rows": len(selected),
    }


def cold_check_stats(result: dict, bf16_rows: list[dict], float32_rows: list[dict],
                     reference_rows: list[dict], selected_rows: list[dict],
                     units: list[dict] | list[str]) -> dict:
    resolved = {}
    for dtype, rows in (("bf16", bf16_rows), ("float32", float32_rows)):
        resolved[dtype] = {}
        for dose in (0.05, 0.1):
            resolved[dtype][dose] = {}
            for kind in ("real", "control"):
                cell_rows = [row for row in rows
                             if float(row["dose"]) == dose and str(row["kind"]) == kind]
                positions = [position for row in cell_rows
                             for position in row.get("positions", [])]
                row_shares = []
                resolved_positions = 0
                rows_with_resolved = 0
                for row in cell_rows:
                    share = _row_median_share(row, resolved_only=True)
                    if share is not None:
                        row_shares.append(share)
                        rows_with_resolved += 1
                    resolved_positions += sum(
                        all(position.get(field) is not None and float(position[field]) >= 1.0
                            for field in ("nA", "n2", "nPair"))
                        for position in row.get("positions", []))
                summary = find_summary(result, dtype, dose, kind)
                resolved[dtype][dose][kind] = {
                    "registered_all_positions_median_row_share": summary["median_row_share"],
                    "resolved_median_row_share": (float(np.median(row_shares))
                                                   if row_shares else None),
                    "resolved_rows": rows_with_resolved,
                    "total_rows": len(cell_rows),
                    "resolved_positions": resolved_positions,
                    "total_positions": len(positions),
                    "resolved_position_coverage": (resolved_positions / len(positions)
                                                    if positions else None),
                }

    real_control_resolved_factors = {}
    for dose in (0.05, 0.1):
        real = resolved["float32"][dose]["real"]["resolved_median_row_share"]
        control = resolved["float32"][dose]["control"]["resolved_median_row_share"]
        real_control_resolved_factors[dose] = (
            max(real / control, control / real) if real and control else None)

    npair_gaps = score119c._reproduction_gate(
        float32_rows, bf16_rows, selected_rows, DOSES, 1.0,
        "float32 versus bf16 at every dose")
    dtype_n = {}
    for dose in DOSES:
        item = {"median_relative_nPair_gap": npair_gaps["by_dose"][f"{dose:g}"][
            "median_relative_nPair_difference"]}
        for kind in ("real", "control"):
            bf = find_summary(result, "bf16", dose, kind)["median_nI"]
            f32 = find_summary(result, "float32", dose, kind)["median_nI"]
            item[kind] = {"bf16_median_nI": bf,
                          "float32_median_nI": f32,
                          "bf16_to_float32_ratio": bf / f32 if bf is not None and f32 else None,
                          "absolute_excess": bf - f32 if bf is not None and f32 is not None else None}
        dtype_n[dose] = item

    float32_index = {(str(row["kind"]), str(row["first"]), str(row["second"])): row
                     for row in float32_rows if float(row["dose"]) == 0.8}
    bf16_index = {(str(row["kind"]), str(row["first"]), str(row["second"])): row
                  for row in bf16_rows if float(row["dose"]) == 0.8}
    t_differences = []
    for key in sorted(set(float32_index) & set(bf16_index)):
        float_positions = {int(position["pos"]): position
                           for position in float32_index[key].get("positions", [])}
        bf_positions = {int(position["pos"]): position
                        for position in bf16_index[key].get("positions", [])}
        for pos in sorted(set(float_positions) & set(bf_positions)):
            float_t = set(float_positions[pos].get("t_ids", []))
            bf_t = set(bf_positions[pos].get("t_ids", []))
            t_differences.append(len(float_t.symmetric_difference(bf_t)))

    part119b_shares = {}
    for dose in (0.2, 0.4, 0.8):
        row_shares = [_row_median_share(row) for row in reference_rows
                      if float(row["dose"]) == dose and str(row["kind"]) == "real"]
        row_shares = [value for value in row_shares if value is not None]
        part119b_shares[dose] = {
            "real_rows": len(row_shares),
            "median_row_share": float(np.median(row_shares)) if row_shares else None,
        }

    pooled_and_slopes = {}
    for dose in (0.05, 0.1, 0.2):
        positions = [position for row in float32_rows
                     if float(row["dose"]) == dose and str(row["kind"]) == "real"
                     for position in row.get("positions", [])]
        numerator = sum(float(position["nI"]) ** 2 for position in positions
                        if position.get("nI") is not None)
        denominator = sum(float(position["nPair"]) ** 2 for position in positions
                          if position.get("nPair") is not None)
        pooled_and_slopes[dose] = {
            "pooled_ratio": numerator / denominator if denominator else None,
            "row_median_share": find_summary(result, "float32", dose, "real")["median_row_share"],
        }
    local_slopes = {}
    for low, high in ((0.05, 0.1), (0.1, 0.2), (0.05, 0.2)):
        low_share = find_summary(result, "float32", low, "real")["median_row_share"]
        high_share = find_summary(result, "float32", high, "real")["median_row_share"]
        local_slopes[(low, high)] = float(np.log(high_share / low_share) / np.log(high / low))

    low_dose_bootstrap = {
        dose: _product_bootstrap_share_interval(float32_rows, dose, units)
        for dose in (0.05, 0.1)
    }
    records = {
        "bf16": len(bf16_rows), "float32": len(float32_rows),
        "bf16_distinct_keys": len({(str(row["kind"]), str(row["first"]), str(row["second"]))
                                    for row in bf16_rows}),
        "float32_distinct_keys": len({(str(row["kind"]), str(row["first"]), str(row["second"]))
                                       for row in float32_rows}),
    }
    return {
        "resolved": resolved,
        "resolved_float32_real_control_factors": real_control_resolved_factors,
        "dtype_npair_and_ni": dtype_n,
        "t_set_difference": {
            "different_positions": sum(value > 0 for value in t_differences),
            "positions_compared": len(t_differences),
            "median_tokens_different": float(np.median(t_differences)) if t_differences else None,
            "max_tokens_different": max(t_differences) if t_differences else None,
        },
        "part119b_full_real_share": part119b_shares,
        "pooled_ratio_and_row_share": pooled_and_slopes,
        "local_slopes": local_slopes,
        "low_dose_bootstrap": low_dose_bootstrap,
        "records": records,
    }


def render_numbers(result: dict) -> str:
    lines = [
        "# Part 119c numbers",
        "",
        "Path 2 row summaries and the supplemental checks below are recomputed by `fig119c.py` from the JSONL rows. "
        "Gates and registered predictions are computed by `score119c.py`, which `fig119c.py` calls. Shares use "
        "`(nI / nPair)^2` at positions with `nPair > 0`; each row is the median of its eligible positions, "
        "then the summary is the median over rows. Position medians for nI and nPair use all recorded positions. "
        "The nPair-below-one share uses all recorded positions.",
        "",
        "## Inventory and gates",
        "",
        "| Quantity | Recomputed value | Source file and key |",
        "| --- | ---: | --- |",
        f"| Selected row keys | {result['selected_rows_expected']} total; "
        f"{result['selected_rows_by_class']['real']} real and {result['selected_rows_by_class']['control']} control | "
        "Both interaction JSONL files: unique row keys from `$.kind`, `$.first`, `$.second` |",
        f"| Records in each file | bf16 {result['cold_check']['records']['bf16']}; "
        f"float32 {result['cold_check']['records']['float32']}; "
        f"{result['cold_check']['records']['bf16_distinct_keys']} distinct row keys × 5 doses per file | "
        "Both interaction JSONL files: record count; `$.kind`, `$.first`, `$.second`, `$.dose` |",
        f"| Doses | {', '.join(f(dose) for dose in DOSES)} | "
        "Both interaction JSONL files: unique `$.dose` values |",
        "| Gate 1, bf16 against Part 119b | "
        f"{'passed' if result['gates']['gate1']['passed'] else 'failed'}; "
        f"median relative nPair difference {f(result['gates']['gate1']['median_relative_nPair_difference_all_checked_rows'])} | "
        "`results/part119c/interaction-bf16.jsonl` and `results/part119b/interaction.jsonl`: "
        "`$.kind`, `$.first`, `$.second`, `$.dose`, `$.positions[].pos`, `$.positions[].nPair` |",
        "| Gate 2, float32 against bf16 at 0.8 | "
        f"{'passed' if result['gates']['gate2']['passed'] else 'failed'}; "
        f"median relative nPair difference {f(result['gates']['gate2']['median_relative_nPair_difference_all_checked_rows'])}; "
        f"all 0.8 nPair values identical by position: "
        f"{result['gates']['gate2']['all_0.8_nPair_identical_by_position']} | "
        "`results/part119c/interaction-float32.jsonl` and `interaction-bf16.jsonl`: "
        "`$.model_dtype`, `$.model_parameter_dtype`, `$.kind`, `$.first`, `$.second`, `$.dose`, "
        "`$.positions[].pos`, `$.positions[].nPair` |",
        "",
        "Gate 1 also checks the 0.2, 0.4, and 0.8 doses separately. Gate 2 requires the recorded loaded-parameter "
        "dtype evidence and a non-identical 0.8 nPair comparison. The recorded dtype strings are in the JSONL keys "
        "named in the source column.",
        "",
        "## Per-dose summaries",
        "",
        "| Dtype | Dose | Class | Rows | Positions | nPair > 0 | nPair < 1 | nPair < 1 share | Median nI | "
        "Median nPair | Median nA | Median n2 | Position median share | Row median share | Source file and key |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["per_dose"]:
        lines.append("| " + " | ".join((
            row["dtype"], f(row["dose"]), row["kind"], f(row["rows"]), f(row["positions"]),
            f(row["positions_with_nPair_above_zero"]), f(row["positions_nPair_below_one"]),
            f(row["share_positions_nPair_below_one"]), f(row["median_nI"]), f(row["median_nPair"]),
            f(row["median_nA"]), f(row["median_n2"]), f(row["median_share_over_positions"]),
            f(row["median_row_share"]), source(row["dtype"]),
        )) + " |")

    lines.extend([
        "",
        "## Dtype comparison",
        "",
        "| Dose | Class | bf16 row share | float32 row share | float32 minus bf16 | bf16 nPair < 1 share | "
        "float32 nPair < 1 share | Source file and key |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for dose in DOSES:
        for kind in ("real", "control"):
            bf = find_summary(result, "bf16", dose, kind)
            f32 = find_summary(result, "float32", dose, kind)
            lines.append("| " + " | ".join((
                f(dose), kind, f(bf["median_row_share"]), f(f32["median_row_share"]),
                f(f32["median_row_share"] - bf["median_row_share"]),
                f(bf["share_positions_nPair_below_one"]), f(f32["share_positions_nPair_below_one"]),
                source("bf16") + "; " + source("float32"),
            )) + " |")

    lines.extend([
        "",
        "## Real against control",
        "",
        "Factor is the larger row share divided by the smaller row share.",
        "",
        "| Dtype | Dose | Real share | Control share | Larger-to-smaller factor | Source file and key |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for dtype in ("bf16", "float32"):
        for dose in DOSES:
            real = find_summary(result, dtype, dose, "real")["median_row_share"]
            control = find_summary(result, dtype, dose, "control")["median_row_share"]
            factor = max(real / control, control / real) if real > 0 and control > 0 else None
            lines.append(f"| {dtype} | {f(dose)} | {f(real)} | {f(control)} | {f(factor)} | {source(dtype)} |")

    lines.extend([
        "",
        "## Registered predictions",
        "",
        "| Prediction | Recomputed observed values | Verdict | Source file and key |",
        "| --- | --- | --- | --- |",
    ])
    prediction_sources = {
        "119c.1": (source("bf16") + "; " + source("float32") + "; "
                    "`part119/score119.py`: `BOOTSTRAP_SEED`; `part119c/score119c.py`: "
                    "`_weighted_difference_ci`, `predictions`"),
        "119c.2": source("float32"),
        "119c.3": source("float32"),
        "119c.4": source("float32"),
    }
    for pred in result["predictions"]:
        observed = pred["observed"]
        if pred["id"] == "119c.1":
            boot = observed["bootstrap"]
            detail = (f"bf16 share {f(observed['bf16_median_row_share'])}; float32 share "
                      f"{f(observed['float32_median_row_share'])}; difference float32 minus bf16 "
                      f"{f(observed['median_difference_float32_minus_bf16'])}; product-weighted 95% interval "
                      f"{f(boot['ci95_product_weighted_difference_of_median_row_shares'][0])} to "
                      f"{f(boot['ci95_product_weighted_difference_of_median_row_shares'][1])}; "
                      f"sum-weighted sensitivity interval "
                      f"{f(boot['ci95_sum_weighted_difference_of_median_row_shares'][0])} to "
                      f"{f(boot['ci95_sum_weighted_difference_of_median_row_shares'][1])}; "
                      f"{boot['draws']} draws, seed {boot['seed']}, {boot['units']} units, {boot['rows']} paired rows")
        elif pred["id"] == "119c.2":
            detail = (f"float32 real share at 0.05 {f(observed['float32_real_median_row_share_0.05'])}; "
                      f"at 0.1 {f(observed['float32_real_median_row_share_0.1'])}")
        elif pred["id"] == "119c.3":
            detail = (f"float32 real share at 0.05 {f(observed['median_row_shares'][0])}; "
                      f"at 0.2 {f(observed['median_row_shares'][1])}; log-log slope {f(observed['log_log_slope'])}")
        else:
            detail = "; ".join(
                f"dose {f(item['dose'])}: real {f(item['real_median_row_share'])}, "
                f"control {f(item['control_median_row_share'])}, factor {f(item['larger_to_smaller_factor'])}"
                for item in observed["dose_comparisons"])
        verdict = "held" if pred["verdict"] == "holds" else "failed"
        lines.append(f"| {pred['id']} | {detail} | {verdict} | {prediction_sources[pred['id']]} |")

    real_01 = find_summary(result, "float32", 0.1, "real")["median_row_share"]
    real_02 = find_summary(result, "float32", 0.2, "real")["median_row_share"]
    second_order_ratio = (real_02 / real_01) / ((0.2 / 0.1) ** 2)
    lines.extend([
        "",
        "## Dose-growth ratio",
        "",
        f"{f(second_order_ratio)}",
        "",
        "Source: `results/part119c/interaction-float32.jsonl`, `$.dose`, `$.kind`, `$.first`, `$.second`, "
        "`$.positions[].nI`, and `$.positions[].nPair`; the value is the float32 real row-median-share ratio "
        "from 0.1 to 0.2 divided by the squared dose ratio.",
        "",
        "## Resolved-only low-dose check",
        "",
        "Resolved positions require nA, n2, and nPair each to be at least 1.0, matching Part 119b. "
        "The registered reading uses every position with nPair > 0. Coverage counts resolved positions "
        "and rows with at least one resolved position.",
        "",
        "| Dtype | Dose | Class | Registered all-position row median | Resolved-only row median | Resolved rows / all rows | Resolved positions / all positions | Position coverage |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for dtype in ("bf16", "float32"):
        for dose in (0.05, 0.1):
            for kind in ("real", "control"):
                row = result["cold_check"]["resolved"][dtype][dose][kind]
                lines.append("| " + " | ".join((
                    dtype, f(dose), kind,
                    f(row["registered_all_positions_median_row_share"]),
                    f(row["resolved_median_row_share"]),
                    f'{row["resolved_rows"]}/{row["total_rows"]}',
                    f'{row["resolved_positions"]}/{row["total_positions"]}',
                    f(row["resolved_position_coverage"]),
                )) + " |")
    lines.extend([
        "",
        "Float32 resolved real/control factors: " + "; ".join(
            f"dose {f(dose)} {f(value)}" for dose, value in
            result["cold_check"]["resolved_float32_real_control_factors"].items()),
        "",
        "## Part 119b full-sample comparison",
        "",
        "The Part 119b column is recomputed from all real rows with the same all-position row-median rule.",
        "",
        "| Dose | 119c bf16 subset | 119c float32 subset | 119b all real rows | 119b real rows |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for dose, row in result["cold_check"]["part119b_full_real_share"].items():
        lines.append(f"| {f(dose)} | "
                     f"{f(find_summary(result, 'bf16', dose, 'real')['median_row_share'])} | "
                     f"{f(find_summary(result, 'float32', dose, 'real')['median_row_share'])} | "
                     f"{f(row['median_row_share'])} | {row['real_rows']} |")
    lines.extend([
        "",
        "## Dtype gap and nI ratios",
        "",
        "The nPair gap is the median, over rows, of each row's median absolute position-level relative difference, "
        "with bf16 as denominator. The nI ratio is the ratio of pooled position medians, bf16 divided by float32.",
        "",
        "| Dose | Median relative nPair gap | Real bf16 / float32 median nI | Real nI excess | Control bf16 / float32 median nI | Control nI excess |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for dose, row in result["cold_check"]["dtype_npair_and_ni"].items():
        lines.append("| " + " | ".join((
            f(dose), f(row["median_relative_nPair_gap"]),
            f(row["real"]["bf16_to_float32_ratio"]), f(row["real"]["absolute_excess"]),
            f(row["control"]["bf16_to_float32_ratio"]), f(row["control"]["absolute_excess"]),
        )) + " |")
    t = result["cold_check"]["t_set_difference"]
    lines.extend([
        "",
        "At dose 0.8, dtype-specific T sets differ at "
        f"{t['different_positions']} of {t['positions_compared']} matched positions; "
        f"the median symmetric-difference size is {f(t['median_tokens_different'])} tokens and the maximum is "
        f"{f(t['max_tokens_different'])}.",
        "",
        "## Pooled ratio, local slopes, and low-dose intervals",
        "",
        "Pooled ratio is sum(nI²) / sum(nPair²) over float32 real positions. The row-median share is the "
        "registered summary. Bootstrap intervals are product-weighted unit-bootstrap intervals for the "
        "float32 real median row share, using the scorer's seed.",
        "",
        "| Dose | Pooled ratio | Float32 real row-median share |",
        "| ---: | ---: | ---: |",
    ])
    for dose, row in result["cold_check"]["pooled_ratio_and_row_share"].items():
        lines.append(f"| {f(dose)} | {f(row['pooled_ratio'])} | {f(row['row_median_share'])} |")
    lines.extend([
        "",
        "| Dose interval | Local log-log slope |",
        "| --- | ---: |",
    ])
    for (low, high), slope in result["cold_check"]["local_slopes"].items():
        lines.append(f"| {f(low)} to {f(high)} | {f(slope)} |")
    lines.extend([
        "",
        "| Dose | Float32 real row-median share | Product-weighted 95% unit-bootstrap interval | Draws | Seed | Units | Rows |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for dose, row in result["cold_check"]["low_dose_bootstrap"].items():
        lines.append(f"| {f(dose)} | {f(row['median_row_share'])} | "
                     f"{f(row['ci95_product_weighted'][0])} to {f(row['ci95_product_weighted'][1])} | "
                     f"{row['draws']} | {row['seed']} | {row['units']} | {row['rows']} |")
    lines.extend([""])
    return "\n".join(lines)


def draw_figure(result: dict) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), constrained_layout=True)
    colors = {
        ("bf16", "real"): "#2457A7",
        ("float32", "real"): "#18A0A6",
        ("bf16", "control"): "#C64B42",
        ("float32", "control"): "#E28B2D",
    }
    styles = {
        ("bf16", "real"): ("-", "o", 1.10, 0.98),
        ("float32", "real"): ("--", "s", 1.30, 1.22),
        ("bf16", "control"): ("-.", "o", 0.76, 0.80),
        ("float32", "control"): ((0, (4, 1, 1, 1)), "s", 0.95, 1.05),
    }
    panel_specs = (
        (axes[0], "median_row_share", "Median row interaction share", (0.002, 0.8), 3),
        (axes[1], "median_nI", "Median nI", (0.03, 25), 3),
    )
    for ax, field, title, ylim, value_digits in panel_specs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.043, 2.35)
        ax.set_ylim(*ylim)
        ax.set_xticks(DOSES, labels=["0.05", "0.1", "0.2", "0.4", "0.8"])
        ax.set_xlabel("Push dose")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, which="major", color="#C9CFD8", linewidth=0.75)
        ax.grid(True, which="minor", color="#E6E9EE", linewidth=0.4, alpha=0.8)
        for key in (("bf16", "real"), ("float32", "real"),
                    ("bf16", "control"), ("float32", "control")):
            dtype, kind = key
            values = [find_summary(result, dtype, dose, kind)[field] for dose in DOSES]
            linestyle, marker, share_factor, ni_factor = styles[key]
            offset = share_factor if field == "median_row_share" else ni_factor
            color = colors[key]
            ax.plot(DOSES, values, color=color, linestyle=linestyle, marker=marker,
                    linewidth=2.0, markersize=5.2, markeredgecolor="white", markeredgewidth=0.6,
                    zorder=3)
            endpoint = float(values[-1])
            label_y = endpoint * offset
            label_x = 1.02
            ax.plot([0.8, 0.96, label_x - 0.025], [endpoint, endpoint, label_y],
                    color=color, linewidth=1.0, alpha=0.9, clip_on=False, zorder=2)
            label = f"{dtype} {kind} {endpoint:.{value_digits}g}"
            ax.text(label_x, label_y, label, color=color, fontsize=8.4,
                    ha="left", va="center", clip_on=False, zorder=4)
    fig.suptitle("Part 119c | 39 units, 312 rows, one question", fontsize=14, fontweight="bold")
    fig.savefig(OUT / "fig119c.png", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    units = rows119.load_units(rows119.DEFAULT_NEURONS,
                               rows119.DEFAULT_HAIKU_KINDS,
                               rows119.DEFAULT_DS_KINDS)
    selected_rows = rows119c.draw_path2_rows(units)
    bf16_rows = read_jsonl(BF16_FILE)
    float32_rows = read_jsonl(FLOAT32_FILE)
    reference_rows = read_jsonl(REFERENCE_FILE)
    result = score_interactions(
        bf16_rows, float32_rows, reference_rows, selected_rows, units)
    result["cold_check"] = cold_check_stats(
        result, bf16_rows, float32_rows, reference_rows, selected_rows, units)
    (OUT / "NUMBERS.md").write_text(render_numbers(result), encoding="utf-8")
    draw_figure(result)
    print(f"Wrote {OUT / 'NUMBERS.md'} and {OUT / 'fig119c.png'}")


if __name__ == "__main__":
    main()
