#!/usr/bin/env python3
"""Recompute Part 122 tables from the saved run files and draw the summary figure."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT / "part122") not in sys.path:
    sys.path.insert(0, str(ROOT / "part122"))
if str(ROOT / "part119") not in sys.path:
    sys.path.insert(0, str(ROOT / "part119"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-part122")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

import rows119
import score122


def read_run(run_dir: Path, units: list[dict], jlens_ids: list[str]) -> dict:
    activations = score122.read_jsonl(run_dir / "activations.jsonl")
    gate2 = score122.read_json(run_dir / "gate2.json")
    read_set = score122.read_json(run_dir / "read_set.json")
    return score122.score_results(activations, gate2, read_set, units, jlens_ids)


def require_score_matches_saved(run_dir: Path, computed: dict) -> None:
    saved = score122.read_json(run_dir / "score122.json")
    fields = ("gates", "unit_frames", "kind_rates", "control_summary", "predictions",
              "reading_branch", "losses", "duplicate_activation_keys")
    for field in fields:
        if computed[field] != saved[field]:
            raise ValueError(f"recomputed {field} does not match {run_dir / 'score122.json'}")


def rate_text(count: int, total: int, digits: int = 3) -> str:
    return f"{count}/{total} ({100 * count / total:.{digits}f}%)" if total else "missing"


def percent_points(rate: float) -> str:
    return f"{100 * rate:.3f} pp"


def build_numbers(score: dict, first_score: dict, units: list[dict]) -> str:
    lines = [
        "# Part 122 numbers",
        "",
        "All Part 122 run values below were recomputed by fig122.py from the saved activation JSONL, gate 2 JSON, and read-set JSON. The recomputation calls score122.py and checks the result against the saved score JSON. The first-set gate count is recomputed the same way from the first-set run files. Source paths are relative to results/part122.",
        "",
        "## Study size and read rule",
        "",
        f"- Focal units: {len(units)}. Frames: {len(score122.FRAMES)}. Matched pairs per unit and frame: {score122.PAIRS_PER_FRAME}. Total second-set pairs: {len(units) * len(score122.FRAMES) * score122.PAIRS_PER_FRAME}. Source: PART122-DESIGN.md and the raw activation and read-set files.",
        "- A unit meets the registered read rule when persona maximum activation is higher than ordinary maximum activation in at least 7 of 8 pairs. The mean comparison is supplementary.",
        "- Study model: OLMo 2 1B Instruct, revision 48d788eca847d4d7548f375ad03d3c9312f6139e, bf16, Intel A770. Source: PART122-DESIGN.md and run_meta.json.",
        "- Passage writer: DeepSeek V4.1 Flash. Source: part122/BUILD-NOTES.md and PART122-DESIGN.md.",
        "",
        "## Registered gates",
        "",
        "| Check | Recomputed observation | Threshold | Result | Source |",
        "| --- | --- | --- | --- | --- |",
    ]
    g1 = score["gates"]["gate1"]
    lines.append(
        f"| Gate 1, second-set token match | {g1['within_two']}/{g1['observed_pairs']} complete pairs within 2 tokens "
        f"({100*g1['rate']:.3f}%) | at least {100*g1['threshold']:.0f}% | "
        f"{'pass' if g1['passed'] else 'miss'} | activations.jsonl; gate recomputed by score122.py |")
    g2 = score["gates"]["gate2"]
    lines.append(
        f"| Gate 2, Part 95 positive control | {g2['space_max_wins']}/{g2['observed_pairs']} maximum wins; "
        f"{g2['observed_pairs']}/{g2['expected_pairs']} complete | at least {g2['threshold']}/{g2['expected_pairs']} | "
        f"{'pass' if g2['passed'] else 'miss'} | gate2.json |")
    for frame in score122.FRAMES:
        row = score["gates"]["gate3"]["by_frame"][frame]
        lines.append(
            f"| Gate 3, random controls in {frame} | {row['read_controls']}/{row['observed_cells']} read "
            f"({100*row['rate']:.3f}%) | at most {100*row['threshold']:.0f}% | "
            f"{'pass' if row['passed'] else 'miss'} | activations.jsonl; read_set.json |")
    false_pass = sum(math.comb(16, k) for k in range(9, 17)) / (2 ** 16)
    lines += [
        "",
        f"- Gate 2 null false-pass probability at the registered 9-of-16 threshold: {false_pass:.12f} ({100*false_pass:.3f}%). This is the one-sided Binomial(16, 0.5) tail under the fair-win null; the review fold reports it as about 40%. Source: PART122-DESIGN.md and part122/FOLD-r1.md; recomputed here.",
        "",
        "## First passage set, excluded by Amendment 1",
        "",
    ]
    first_g1 = first_score["gates"]["gate1"]
    lines.append(
        f"- {first_g1['within_two']}/{first_g1['observed_pairs']} pairs were within 2 tokens "
        f"({100*first_g1['rate']:.3f}%), below the 90% gate. This set is not used for the result. "
        f"Source: first-set/activations.jsonl, first-set/gate2.json, first-set/read_set.json; recomputed by this script.")
    lines += [
        "- Amendment 1 and the reason for replacement: PART122-DESIGN.md. The first run table was seen before the gate miss was noticed, as recorded in that amendment.",
        "",
        "## Read rate by kind and frame",
        "",
        "| Frame | Haiku kind | Reads / units | Rate | Agreed reads / agreed units | Agreed rate | Source |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in score["kind_rates"]:
        lines.append(
            f"| {row['frame']} | {row['haiku_kind']} | {row['reads']}/{row['complete_units']} | "
            f"{100*row['rate']:.3f}% | {row['agreed_reads']}/{row['agreed_complete_units']} | "
            f"{100*row['agreed_rate']:.3f}% | activations.jsonl; read_set.json; Part 92 and 92b labels |")
    lines += [
        "",
        "Part 92b kind agreement counts for the Haiku groups are also shown above. For concept, 8 of 18 units agree; for slot, 21 of 21 agree. Source: Part 92 and 92b label files loaded by rows119.py.",
        "",
        "## Controls and slot-control gaps",
        "",
        "| Frame | Slot rate | Random controls | Shuffled controls | Slot above random | Slot above shuffled | Source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    kinds = {(row["frame"], row["haiku_kind"]): row for row in score["kind_rates"]}
    for frame in score122.FRAMES:
        slot = kinds[(frame, "slot")]
        controls = score["control_summary"][frame]
        random_gap = slot["rate"] - controls["random_rate"]
        shuffled_gap = slot["rate"] - controls["shuffled_rate"]
        lines.append(
            f"| {frame} | {rate_text(slot['reads'], slot['complete_units'])} | "
            f"{rate_text(controls['random_read_controls'], controls['random_complete_controls'])} | "
            f"{rate_text(controls['shuffled_reads'], controls['shuffled_complete_units'])} | "
            f"{percent_points(random_gap)} | {percent_points(shuffled_gap)} | "
            f"activations.jsonl; read_set.json |")
    lines += [
        "",
        "Random control denominator is 117 per frame (three sampled channels for each of 39 units). Shuffled control denominator is 39 per frame (one non-self partner per unit).",
        "",
        "## Registered predictions",
        "",
        "| # | Result recomputed from the raw run | Verdict | Source |",
        "| --- | --- | --- | --- |",
    ]
    predictions = score["predictions"]
    p1 = predictions["122.1"]["metrics"]
    lines.append(f"| 122.1 | {p1['concept_reads']}/{p1['expected_units']} concept units; threshold at least {p1['threshold']} | {predictions['122.1']['verdict']} | activations.jsonl; read_set.json |")
    p2 = predictions["122.2"]["metrics"]
    lines.append(f"| 122.2 | {p2['slot_reads']}/{p2['expected_units']} slot units; threshold at most {p2['threshold']} | {predictions['122.2']['verdict']} | activations.jsonl; read_set.json |")
    p3 = predictions["122.3"]["metrics"]
    lines.append(f"| 122.3 | {p3['concept_rate']:.12f} vs {p3['slot_rate']:.12f}; two-sided Fisher p = {p3['fisher_two_sided_p']:.12f}; threshold p < {p3['threshold']:.2f} | {predictions['122.3']['verdict']} | activations.jsonl; read_set.json; score122.py |")
    p4 = predictions["122.4"]["metrics"]
    lines.append(f"| 122.4 | slot {p4['slot_T_reads']}/{p4['expected_units_per_frame']} in T to {p4['slot_C_reads']}/{p4['expected_units_per_frame']} in C; gain {p4['gain']}; target at least +{p4['threshold']} | {predictions['122.4']['verdict']} | activations.jsonl; read_set.json |")
    p5 = predictions["122.5"]["metrics"]
    share_gap = 100 * (p5["slot_gain_share"] - p5["concept_gain_share"])
    lines.append(f"| 122.5 | slot gain {p5['slot_gain']}/{21} ({100*p5['slot_gain_share']:.6f}%); concept gain {p5['concept_gain']}/{18} ({100*p5['concept_gain_share']:.6f}%); slot share is {share_gap:.6f} percentage points higher, while both gains are negative | {predictions['122.5']['verdict']} as written | activations.jsonl; read_set.json |")
    p6 = predictions["122.6"]["metrics"]["by_frame"]
    lines.append(f"| 122.6 | shuffled T {rate_text(p6['T']['reads'], p6['T']['complete_units'])}; F {rate_text(p6['F']['reads'], p6['F']['complete_units'])}; C {rate_text(p6['C']['reads'], p6['C']['complete_units'])}; each compared with the 10% ceiling | {predictions['122.6']['verdict']} | activations.jsonl; read_set.json |")
    p7 = predictions["122.7"]["metrics"]
    lines.append(f"| 122.7 | {p7['reads']}/{p7['jlens_units']} J-lens matched units; threshold at least {p7['threshold']} | {predictions['122.7']['verdict']} | activations.jsonl; read_set.json; jlens-concept-matches.json |")
    lines += [
        "",
        f"- Registered reading branch: {score['reading_branch']['text']} Source: score122.py and the prediction values above.",
        "",
        "## Maximum versus mean",
        "",
        "The registered read rule uses maximum activation. For this comparison only, the same 7-of-8 cutoff is applied to mean activation wins. The table lists every unit-frame where the two measures land on different sides of that cutoff.",
        "",
        "| Frame | Unit | Maximum wins / 8 | Mean wins / 8 | Maximum rule | Mean at same cutoff | Source |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    disagreement_rows = []
    for row in score["unit_frames"]:
        max_meets = row["read"] is True
        mean_meets = row["mean_wins"] >= score122.READ_WIN_MIN
        if max_meets != mean_meets:
            disagreement_rows.append(row)
    for row in disagreement_rows:
        lines.append(
            f"| {row['frame']} | {row['unit_id']} | {row['max_wins']}/8 | {row['mean_wins']}/8 | "
            f"{'meets' if row['read'] else 'does not meet'} | "
            f"{'meets' if row['mean_wins'] >= score122.READ_WIN_MIN else 'does not meet'} | "
            f"activations.jsonl; read_set.json |")
    lines += [
        "",
        f"- Total maximum-versus-mean disagreements: {len(disagreement_rows)} unit-frame decisions. Mean is supplementary and does not change the registered result.",
        "",
        "## Source files",
        "",
        "- Second set: activations.jsonl, gate2.json, read_set.json, run_meta.json.",
        "- First set, excluded: first-set/activations.jsonl, first-set/gate2.json, first-set/read_set.json.",
        "- Kind labels: results/part92/step2/neurons.json, results/part92/step2/kinds.json, results/part92b/score92b.json.",
        "- J-lens matched list: part122/jlens-concept-matches.json.",
        "- Registration, scorer, and review: PART122-DESIGN.md, part122/score122.py, part122/FOLD-r1.md.",
        "",
    ]
    return "\n".join(lines)


def draw_figure(score: dict, output: Path) -> None:
    frames = list(score122.FRAMES)
    kinds = {(row["frame"], row["haiku_kind"]): row for row in score["kind_rates"]}
    controls = score["control_summary"]
    labels = ("Concept", "Slot", "Random", "Shuffled")
    colors = ("#32688e", "#d28a27", "#5a9b72", "#9a6c9c")
    width = 0.19
    x = np.arange(len(frames), dtype=float)
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    fig, (ax_rate, ax_heat) = plt.subplots(
        2, 1, figsize=(11, 16), gridspec_kw={"height_ratios": [1, 2.2]})

    series = []
    for frame in frames:
        concept = kinds[(frame, "concept")]
        slot = kinds[(frame, "slot")]
        control = controls[frame]
        series.append((
            (concept["reads"], concept["complete_units"], concept["rate"]),
            (slot["reads"], slot["complete_units"], slot["rate"]),
            (control["random_read_controls"], control["random_complete_controls"], control["random_rate"]),
            (control["shuffled_reads"], control["shuffled_complete_units"], control["shuffled_rate"]),
        ))
    for i, label in enumerate(labels):
        rates = [row[i][2] for row in series]
        bars = ax_rate.bar(x + offsets[i], rates, width, label=label, color=colors[i])
        for bar, frame_row in zip(bars, series):
            count, total, rate = frame_row[i]
            ax_rate.annotate(
                f"{count}/{total}\n{rate:.1%}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8)
    ax_rate.set_xticks(x, frames)
    ax_rate.set_ylim(0, 0.95)
    ax_rate.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_rate.set_ylabel("Units meeting the read rule")
    ax_rate.set_xlabel("Frame: T third person; F first person; C assistant turn")
    ax_rate.set_title("Persona read rate by kind and frame, with both controls")
    ax_rate.legend(ncols=4, loc="upper left", frameon=False)
    ax_rate.grid(axis="y", alpha=0.25)
    ax_rate.set_axisbelow(True)
    ax_rate.text(
        0.995, 0.98, "Read rule: maximum wins in at least 7 of 8 matched pairs",
        transform=ax_rate.transAxes, ha="right", va="top", fontsize=9)

    row_lookup = {(row["unit_id"], row["frame"]): row for row in score["unit_frames"]}
    unit_kinds = {row["unit_id"]: row["haiku_kind"] for row in score["unit_frames"] if row["frame"] == "T"}
    unit_order = sorted(unit_kinds, key=lambda unit_id: (unit_kinds[unit_id] != "concept", unit_id))
    values = []
    for unit_id in unit_order:
        values.append([
            row_lookup[(unit_id, frame)]["max_persona_average"] -
            row_lookup[(unit_id, frame)]["max_ordinary_average"]
            for frame in frames
        ])
    values = np.asarray(values, dtype=float)
    limit = max(float(np.max(np.abs(values))), 1e-9)
    image = ax_heat.imshow(values, aspect="auto", interpolation="nearest",
                           cmap="coolwarm", vmin=-limit, vmax=limit)
    ax_heat.set_xticks(range(len(frames)), ["T\nthird person", "F\nfirst person", "C\nassistant turn"])
    ax_heat.set_yticks(range(len(unit_order)), [
        f"{unit_id} ({unit_kinds[unit_id]})" for unit_id in unit_order])
    ax_heat.tick_params(axis="y", labelsize=7)
    ax_heat.set_xlabel("Frame")
    ax_heat.set_ylabel("Unit, grouped by Part 92 Haiku kind")
    ax_heat.set_title("Per-unit mean maximum activation difference: persona minus ordinary")
    colorbar = fig.colorbar(image, ax=ax_heat, pad=0.015)
    colorbar.set_label("Activation difference")
    for boundary in range(1, len(unit_order)):
        if unit_kinds[unit_order[boundary]] != unit_kinds[unit_order[boundary - 1]]:
            ax_heat.axhline(boundary - 0.5, color="black", linewidth=1)
    fig.suptitle("Part 122: OLMo 2 1B Instruct persona read test", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbers", type=Path, default=HERE / "NUMBERS.md")
    parser.add_argument("--figure", type=Path, default=HERE / "fig122.png")
    args = parser.parse_args()

    units = rows119.load_units()
    jlens_data = score122.read_json(score122.HERE / "jlens-concept-matches.json")
    jlens_ids = [str(row["id"]) for row in jlens_data.get("units", [])
                 if isinstance(row, dict) and "id" in row]

    second = read_run(HERE, units, jlens_ids)
    require_score_matches_saved(HERE, second)
    first_dir = HERE / "first-set"
    first = read_run(first_dir, units, jlens_ids)
    require_score_matches_saved(first_dir, first)

    draw_figure(second, args.figure)
    args.numbers.parent.mkdir(parents=True, exist_ok=True)
    args.numbers.write_text(build_numbers(second, first, units), encoding="utf-8")
    print(json.dumps({
        "figure": str(args.figure),
        "numbers": str(args.numbers),
        "second_set_gates": second["gates"]["all_passed"],
        "first_set_gate1_within_two": first["gates"]["gate1"]["within_two"],
        "second_set_reading_branch": second["reading_branch"]["branch"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

