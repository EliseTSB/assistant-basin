#!/usr/bin/env python3
"""Recompute Part 120 numbers from its saved vectors and draw the dose figure."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/part120-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part120 import score120

HERE = Path(__file__).resolve().parent
VECTOR_SOURCE = (
    "`results/part120/vectors/*.npz`: `index.doses`, `index.entries[].kind`, "
    "`first`, `second`, `pos`, `t_ids`, `start`, and `values` (A, B, P segments)."
)
COLORS = {"real": "#2457c5", "control": "#d37b14"}
LABELS = {"real": "Real second push", "control": "Random direction"}


def load_result() -> tuple[dict, dict]:
    units = score120.rows119.load_units(
        score120.rows119.DEFAULT_NEURONS,
        score120.rows119.DEFAULT_HAIKU_KINDS,
        score120.rows119.DEFAULT_DS_KINDS,
    )
    selected_rows = score120.rows119c.draw_path2_rows(units)
    vectors, issues = score120.load_vectors(HERE / "vectors", units)
    timing = score120.read_json(HERE / "timing.json")
    score120.require_complete_run(timing, issues, units)
    result = score120.score_data(
        vectors,
        selected_rows,
        units,
        score120.read_jsonl(ROOT / "results/part119c/interaction-float32.jsonl"),
        score120.read_json(HERE / "gate1.json"),
        score120.read_json(HERE / "gate3.json"),
        score120.read_json(ROOT / "results/part119/tournament.json"),
        issues,
    )
    stored = score120.read_json(HERE / "score120.json")
    for key in (
        "selected_rows_expected",
        "selected_rows_by_class",
        "gates",
        "per_dose",
        "positions_dropped_by_dose",
        "float32_floor_ratios",
        "bootstrap_c_0.1_real",
        "u_by_unit",
        "exploratory",
        "predictions",
        "headline",
    ):
        normalized = json.loads(json.dumps(result[key], allow_nan=False, sort_keys=True))
        if normalized != stored[key]:
            raise ValueError(f"recomputed score field differs from score120.json: {key}")
    return result, timing


def fmt(value, digits: int = 9) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}g}"


def make_figure(result: dict, output: Path) -> None:
    fields = (
        ("c", "c"),
        ("q", "q"),
        ("norm_I", "Norm I"),
        ("norm_SA", "Norm S_A"),
        ("norm_SB", "Norm S_B"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), constrained_layout=True)
    for axis, (field, label) in zip(axes.flat, fields):
        for kind in ("real", "control"):
            rows = [row for row in result["per_dose"] if row["kind"] == kind]
            points = [(float(row["dose"]), row[f"median_row_{field}"])
                      for row in rows if row.get(f"median_row_{field}") is not None
                      and float(row[f"median_row_{field}"]) > 0]
            if points:
                x, y = zip(*points)
                axis.plot(x, y, marker="o", linewidth=1.8, markersize=5,
                          color=COLORS[kind], label=LABELS[kind])
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(score120.DOSES)
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xticklabels([f"{dose:g}" for dose in score120.DOSES],
                             rotation=35, ha="right", fontsize=8)
        axis.tick_params(axis="x", which="minor", labelbottom=False)
        axis.set_title(label, loc="left", fontweight="bold")
        axis.set_xlabel("Dose α (log scale)")
        axis.set_ylabel("Median row value (log scale)")
        axis.grid(True, which="major", color="#b8c1ce", alpha=0.55, linewidth=0.7)
        axis.grid(True, which="minor", color="#d8dee7", alpha=0.32, linewidth=0.45)
    axes.flat[0].legend(frameon=False, fontsize=9)
    axes.flat[5].set_visible(False)
    fig.suptitle("Part 120: finite differences by dose", fontsize=15, fontweight="bold")
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def prediction_result(prediction: dict) -> tuple[str, str]:
    observed = prediction["observed"]
    key = prediction["id"]
    if key == "120.1":
        slopes = observed["adjacent_dose_ratios"]
        text = "; ".join(
            f"{item['from_dose']:g} to {item['to_dose']:g}: "
            f"log2 norm-I ratio {fmt(item['log2_norm_ratio'])}"
            for item in slopes
        )
        field = "`observed.adjacent_dose_ratios`"
    elif key == "120.2":
        text = (f"q at 0.05 {fmt(observed['real_median_row_q_0.05'])}; "
                f"q at 0.2 {fmt(observed['real_median_row_q_0.2'])}")
        field = "`observed.real_median_row_q_0.05` and `observed.real_median_row_q_0.2`"
    elif key == "120.3":
        low, high = observed["product_weighted_ci95"]
        text = (f"c at 0.1 {fmt(observed['real_median_row_c_0.1'])}; "
                f"product-weighted 95% interval {fmt(low)} to {fmt(high)}")
        field = "`observed.real_median_row_c_0.1` and `observed.product_weighted_ci95`"
    elif key == "120.4":
        checks = observed["dose_comparisons"]
        text = "; ".join(
            f"{item['dose']:g}: real c {fmt(item['real_median_c'])}, "
            f"control c {fmt(item['control_median_c'])}, factor "
            f"{fmt(item['larger_to_smaller_factor'])}"
            for item in checks
        )
        field = "`observed.dose_comparisons[]`"
    elif key == "120.5":
        text = (f"self share {fmt(observed['real_median_row_self_share_0.4'])}; "
                f"pair share {fmt(observed['real_median_row_pair_share_0.4'])}")
        field = "`observed.real_median_row_self_share_0.4` and `observed.real_median_row_pair_share_0.4`"
    elif key == "120.6":
        text = (f"Haiku slot median u {fmt(observed['haiku_slot_median_u'])} "
                f"(n={observed['haiku_slot_n']}); concept {fmt(observed['haiku_concept_median_u'])} "
                f"(n={observed['haiku_concept_n']}); one-sided p "
                f"{fmt(observed['haiku_one_sided_mann_whitney_p'])}; agreed slot/concept "
                f"{fmt(observed['agreed_slot_median_u'])}/{fmt(observed['agreed_concept_median_u'])}")
        field = "`observed`"
    elif key == "120.7":
        text = (f"Spearman rho {fmt(observed['spearman_rho_u_0.4_vs_dominance'])} "
                f"over {observed['units']} units")
        field = "`observed.spearman_rho_u_0.4_vs_dominance` and `observed.units`"
    else:
        text = (f"real-over-control median unit ratio "
                f"{fmt(observed['median_unit_real_over_control_norm_SB_0.1'])} "
                f"over {observed['units']} units; registered range is 0.8 to 1.25")
        field = "`observed.median_unit_real_over_control_norm_SB_0.1` and `observed.units`"
    source = (f"`results/part120/score120.json`: `$.predictions[]` where `id={key}`, "
              f"fields {field} and `verdict`; recomputed from {VECTOR_SOURCE}")
    return text, source


def make_numbers(result: dict, timing: dict) -> str:
    lines = [
        "# Part 120 numbers",
        "",
        "All statistics below are recomputed by `fig120.py` from the saved per-unit vector arrays. "
        "The script checks the recomputed score against the recorded `score120.json` before writing this file. "
        "The registered position rule takes medians over eligible positions within a row, then medians over rows. "
        "At dose 0.8 there is no 1.6 pass, so only the raw interaction norm is defined.",
        "",
        "Raw vectors are in " + VECTOR_SOURCE,
        "Derived row statistics are computed by `part120/score120.py`: `position_statistics`, "
        "`_row_statistics`, and `_summary_table`.",
        "",
        "## Run scope and gates",
        "",
        "| Item | Value | Source field |",
        "| --- | ---: | --- |",
        f"| Rows | {timing['rows']} ({result['selected_rows_by_class']['real']} real, "
        f"{result['selected_rows_by_class']['control']} control) | `timing.json:$.rows`; "
        "`score120.json:$.selected_rows_by_class` |",
        f"| First units | {timing['first_units']} | `timing.json:$.first_units` |",
        f"| Device and batch | {timing['device']}, {timing['batch']} | `timing.json:$.device`, `$.batch` |",
        f"| Logical passes | {timing['logical_passes']} | `timing.json:$.logical_passes` |",
        f"| Elapsed seconds | {fmt(timing['elapsed_seconds'])} | `timing.json:$.elapsed_seconds` |",
        f"| Gate 1 | {'passed' if result['gates']['gate1']['passed'] else 'failed'} "
        f"({result['gates']['gate1']['paths_checked']}/{result['gates']['gate1']['expected_paths']} paths) | "
        "`score120.json:$.gates.gate1`; `gate1.json:$.paths[]` |",
    ]
    gate2 = result["gates"]["gate2"]
    gate2_max = max(
        cell[stat]["median_relative_difference"]
        for cell in gate2["by_dose"].values()
        for stat in ("nA", "n2", "nPair", "nI")
    )
    lines.extend([
        f"| Gate 2 | {'passed' if gate2['passed'] else 'failed'}; maximum cell median relative difference "
        f"{fmt(gate2_max)} (strict limit 0.005) | `score120.json:$.gates.gate2.by_dose` values' `nA`, `n2`, `nPair`, and `nI` `median_relative_difference` fields |",
        f"| Gate 3 | {'passed' if result['gates']['gate3']['passed'] else 'failed'}; maximum adjacent-dose relative error "
        f"{fmt(result['gates']['gate3']['maximum_relative_error'])} (limit 1e-5) | `score120.json:$.gates.gate3.maximum_relative_error`; `gate3.json:$.maximum_relative_error` |",
        "",
        "## Dose table",
        "",
        "Norms are medians of row medians. At 0.8, the self terms and q/c are n/a because the 1.6 pass was not run.",
        "",
        "| Dose | Class | Rows | Positions considered | Eligible | Dropped | c | q | ‖I‖ | ‖S_A‖ | ‖S_B‖ | Source fields |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in result["per_dose"]:
        dose, kind = row["dose"], row["kind"]
        source_fields = ", ".join(
            f"`median_row_{field}`" for field in ("c", "q", "norm_I", "norm_SA", "norm_SB")
        )
        lines.append(
            f"| {dose:g} | {kind} | {row['rows']} | {row['positions_considered']} | "
            f"{row['positions_eligible'] if row['positions_eligible'] is not None else 'n/a'} | "
            f"{row['positions_dropped'] if row['positions_dropped'] is not None else 'n/a'} | "
            f"{fmt(row['median_row_c'])} | {fmt(row['median_row_q'])} | "
            f"{fmt(row['median_row_norm_I'])} | {fmt(row['median_row_norm_SA'])} | "
            f"{fmt(row['median_row_norm_SB'])} | `results/part120/score120.json`: `$.per_dose[]` "
            f"matching dose={dose:g}, kind={kind}; fields {source_fields} |"
        )
    bootstrap = result["bootstrap_c_0.1_real"]
    lines.extend([
        "",
        "## Bootstrap interval",
        "",
        f"At 0.1 on real rows, median c is {fmt(bootstrap['point_median'])}. The product-weighted "
        f"95% unit-bootstrap interval is {fmt(bootstrap['ci95_product_weighted'][0])} to "
        f"{fmt(bootstrap['ci95_product_weighted'][1])}; the sum-weighted sensitivity interval is "
        f"{fmt(bootstrap['ci95_sum_weighted'][0])} to {fmt(bootstrap['ci95_sum_weighted'][1])}. "
        f"There are {bootstrap['draws']} draws with seed {bootstrap['seed']}, {bootstrap['units']} units, "
        f"and {bootstrap['rows']} real rows. Source fields in `results/part120/score120.json`: "
        "`$.bootstrap_c_0.1_real.point_median`, `ci95_product_weighted`, `ci95_sum_weighted`, "
        "`draws`, `seed`, `units`, and `rows`; "
        f"recomputed from {VECTOR_SOURCE} by `score120.unit_bootstrap_interval`.",
        "",
        "## Registered predictions",
        "",
        "Prediction text is copied from the registration. Values and verdicts are recomputed by this script.",
        "",
        "| # | Prediction (verbatim; author and confidence) | Result | Verdict | Source field |",
        "| --- | --- | --- | --- | --- |",
    ])
    for prediction in result["predictions"]:
        confidence = "medium" if prediction["id"] in {"120.1", "120.2", "120.3", "120.4"} else "low"
        observed, source = prediction_result(prediction)
        verdict = "held" if prediction["verdict"] == "holds" else "failed"
        lines.append(
            f"| {prediction['id']} | Claude ({confidence} confidence): {prediction['prediction']} | "
            f"{observed} | {verdict} | {source} |"
        )
    lines.extend([
        "",
        "## Float32 floor table",
        "",
        "Each ratio is the higher-dose median row norm divided by the lower-dose value. A square-law norm scales by 4 for a doubled dose.",
        "",
        "| Class | Statistic | Dose interval | Higher/lower norm ratio | Source field |",
        "| --- | --- | --- | ---: | --- |",
    ])
    for row in result["float32_floor_ratios"]:
        source = ("`results/part120/score120.json`: `$.float32_floor_ratios[]` matching "
                  f"kind={row['kind']}, statistic={row['statistic']}, "
                  f"from_dose={row['from_dose']:g}, to_dose={row['to_dose']:g}; "
                  "field `higher_over_lower_norm_ratio`")
        statistic_label = {"norm_I": "I", "norm_SA": "S_A"}[row["statistic"]]
        lines.append(
            f"| {row['kind']} | ‖{statistic_label}‖ | "
            f"{row['from_dose']:g} to {row['to_dose']:g} | "
            f"{fmt(row['higher_over_lower_norm_ratio'])} | {source} |"
        )
    lines.extend([
        "",
        "## Exploratory tables",
        "",
        "These cosine and layer-stratified summaries were registered as exploratory. They are not prediction tests.",
        "",
        "| Dose | Class | Median cosine(I,S_A) | Median cosine(I,S_B) | Same-layer rows | Same-layer median c | Different-layer rows | Different-layer median c | Source fields |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in result["exploratory"]:
        dose, kind = row["dose"], row["kind"]
        fields = ("median_row_cosine_I_SA", "median_row_cosine_I_SB",
                  "same_layer_rows", "same_layer_median_row_c",
                  "different_layer_rows", "different_layer_median_row_c")
        sources = ", ".join(f"`{field}`" for field in fields)
        lines.append(
            f"| {dose:g} | {kind} | {fmt(row['median_row_cosine_I_SA'])} | "
            f"{fmt(row['median_row_cosine_I_SB'])} | {row['same_layer_rows']} | "
            f"{fmt(row['same_layer_median_row_c'])} | {row['different_layer_rows']} | "
            f"{fmt(row['different_layer_median_row_c'])} | `results/part120/score120.json`: "
            f"`$.exploratory[]` matching dose={dose:g}, kind={kind}; fields {sources} |"
        )
    lines.extend([
        "",
        "## Source and recomputation notes",
        "",
        "- Gate 1 inputs: `results/part120/gate1.json:$.paths[]`; gate 3 inputs: `results/part120/gate3.json:$.units[]`.",
        "- Gate 2 reference: `results/part119c/interaction-float32.jsonl`, fields `dose`, `kind`, `first`, `second`, `positions[].pos`, `nA`, `n2`, `nPair`, and `nI`.",
        "- Registered predictions, floor ratios, and exploratory source fields are in `results/part120/score120.json`; this script recomputes them from the NPZ vectors before checking equality.",
        "- Timing and scope values come from `results/part120/timing.json`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    result, timing = load_result()
    if not result["gates"]["all_passed"]:
        raise ValueError("Part 120 has a failed gate; registered readings must not be plotted")
    make_figure(result, HERE / "fig120.png")
    (HERE / "NUMBERS.md").write_text(make_numbers(result, timing), encoding="utf-8")
    print("Wrote results/part120/fig120.png and results/part120/NUMBERS.md")


if __name__ == "__main__":
    main()
