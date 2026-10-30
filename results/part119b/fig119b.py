#!/usr/bin/env python3
"""Recompute Part 119b summaries from its stored rows and draw the result figure."""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "part119b"))
sys.path.insert(0, str(ROOT / "part86"))
import score119b as scorer  # noqa: E402
import rows119 as rows119  # noqa: E402
import triage86  # noqa: E402


DOSES = (0.2, 0.4, 0.6, 0.8)
ANSWER_DOSES = (0.2, 0.4, 0.6, 0.8, 1.13, 1.6)
OUT = Path(__file__).with_name("fig119b.png")


def load_score() -> dict:
    """Recompute the registered tables from the JSONL inputs, without SCORE.md."""
    return scorer.score_data(
        scorer.read_jsonl("results/part119b/answers.jsonl"),
        scorer.read_jsonl("results/part119b/interaction.jsonl"),
        scorer.read_jsonl("results/part119b/path1.jsonl"),
        scorer.read_json("results/part119b/gate1.json"),
        scorer.read_jsonl("results/part119/answers.jsonl"),
        scorer.read_jsonl("results/part119/interaction.jsonl"),
        scorer.read_json("results/part119/score119.json"),
        scorer.read_json("results/part119/tournament.json"),
        scorer.read_json("results/part119b/push_norms.json"),
        rows119.load_units(),
        reader_key_rows=scorer.read_jsonl("results/part119b/labels-ds.jsonl.keys.jsonl"),
        reader_labels_ds=scorer.read_jsonl("results/part119b/labels-ds.jsonl"),
        reader_labels_luna=scorer.read_jsonl("results/part119b/labels-luna.jsonl"),
        reader_known_ds=scorer.read_jsonl("results/part119b/known-ds.jsonl"),
        reader_known_luna=scorer.read_jsonl("results/part119b/known-luna.jsonl"),
        part119_reader_keys=scorer.read_jsonl("results/part119/labels-ds.jsonl.keys.jsonl"),
        part119_reader_labels=scorer.read_jsonl("results/part119/labels-ds.jsonl"),
        reader_files_present={"keys": True, "ds": True, "luna": True,
                              "known_ds": True, "known_luna": True},
    )


def median_norms_by_dose(interactions: list[dict]) -> dict:
    """Median of per-row position medians, on positions resolved at all doses."""
    index = scorer.path2_index(interactions)
    support = scorer._common_support(index)
    output = {}
    for dose in DOSES:
        dose_text = f"{dose:g}"
        output[dose_text] = {}
        for kind in ("real", "control"):
            row_values = {field: [] for field in ("nI", "nPair", "nA", "n2")}
            for key, dose_rows in index.items():
                if key[0] != kind or dose not in dose_rows:
                    continue
                position_map = scorer._position_map(dose_rows[dose])
                for field, values in row_values.items():
                    observations = [float(position_map[pos][field])
                                    for pos in support.get(key, set())
                                    if pos in position_map and scorer.resolved(position_map[pos])]
                    if observations:
                        values.append(float(statistics.median(observations)))
            output[dose_text][kind] = {
                field: float(statistics.median(values)) if values else None
                for field, values in row_values.items()
            }
    return output


def median_tv_by_dose(interactions: list[dict]) -> dict:
    """Median of each row's all-position TV median, then the median across rows."""
    index = scorer.path2_index(interactions)
    output = {}
    for dose in DOSES:
        dose_text = f"{dose:g}"
        output[dose_text] = {}
        for kind in ("real", "control"):
            row_values = []
            for key, dose_rows in index.items():
                if key[0] != kind or dose not in dose_rows:
                    continue
                values = [float(position["tv"]) for position in dose_rows[dose].get("positions", [])
                          if position.get("tv") is not None]
                if values:
                    row_values.append(float(statistics.median(values)))
            output[dose_text][kind] = float(statistics.median(row_values)) if row_values else None
    return output


def assistant_reply(text: str) -> bool:
    """Use Part 86's wider first-sentence self-description screen."""
    return bool(triage86.WIDER_SELF_DESCRIPTION.search(triage86.first_sentence(text)))


def assistant_reply_summary(answers_b: list[dict], answers_119: list[dict],
                            reader_keys_b: list[dict], reader_labels_b: list[dict],
                            reader_keys_119: list[dict], reader_labels_119: list[dict]) -> dict:
    """Count first-sentence assistant descriptions in all pair answers and reader sample."""
    by_dose = {}
    for dose in ("0.2", "0.4", "0.6", "0.8"):
        source = answers_119 if dose == "0.8" else answers_b
        prefix = "pair:" if dose == "0.8" else f"pair@{dose}:"
        pair_rows = [row for row in source if str(row.get("row_id", "")).startswith(prefix)]
        covered = sum(assistant_reply(str(row.get("text", ""))) for row in pair_rows)
        by_dose[dose] = {"assistant_reply": covered, "pair_answers": len(pair_rows),
                         "share": covered / len(pair_rows) if pair_rows else None}

    def sample_at_dose(dose: str, keys: list[dict], labels: list[dict], answers: list[dict]) -> dict:
        labels_by_id = {row["item_id"]: row for row in labels}
        answers_by_id = {str(row["row_id"]): row for row in answers}
        sample = []
        for key in keys:
            if key.get("row_type") != "pair":
                continue
            row_id = str(key["row_id"])
            _, suffix = row_id.split(":", 1)
            answer_id = row_id if dose == "0.8" else f"pair@{dose}:{suffix}"
            answer = answers_by_id[answer_id]
            label = labels_by_id[key["item_id"]]
            sample.append({"assistant": assistant_reply(str(answer.get("text", ""))),
                           "label": label})
        return {
            "sample_answers": len(sample),
            "assistant_replies": sum(row["assistant"] for row in sample),
            "coherent_assistant_replies": sum(row["assistant"] and row["label"]["coherent"]
                                               for row in sample),
            "coherent_non_assistant_replies": sum(not row["assistant"] and row["label"]["coherent"]
                                                   for row in sample),
            "x_true": sum(row["label"]["x"] for row in sample),
            "y_true": sum(row["label"]["y"] for row in sample),
            "both_true": sum(row["label"]["x"] and row["label"]["y"] for row in sample),
            "coherent": sum(row["label"]["coherent"] for row in sample),
        }

    by_dose["reader_sample"] = {
        "0.4": sample_at_dose("0.4", reader_keys_b, reader_labels_b, answers_b),
        "0.8": sample_at_dose("0.8", reader_keys_119, reader_labels_119, answers_119),
    }
    return by_dose


def fisher_checks(score: dict) -> dict:
    """Two-sided Fisher exact checks for the requested triage contrasts."""
    result = {}
    for dose, neuron_broken, random_broken in (("0.8", 4, 1), ("1.13", 12, 9), ("1.6", 30, 25)):
        result[f"single_{dose}"] = {
            "neuron": [neuron_broken, 39], "random": [random_broken, 39],
            "fisher_two_sided_p": float(fisher_exact(
                [[neuron_broken, 39 - neuron_broken],
                 [random_broken, 39 - random_broken]]).pvalue),
        }
    result["pair_0.8"] = {
        "real": [357, 741], "random": [286, 741],
        "fisher_two_sided_p": float(fisher_exact([[357, 741 - 357], [286, 741 - 286]]).pvalue),
    }
    broken = score["broken_answers_by_dose_and_class"]
    result["A_plus_random"] = {
        dose: {"real_pairs": broken[dose]["pair"]["part86_triage_broken"],
               "real_total": broken[dose]["pair"]["total"],
               "control": broken[dose]["control"]["part86_triage_broken"],
               "control_total": broken[dose]["control"]["total"],
               "fisher_two_sided_p": float(fisher_exact([
                   [broken[dose]["pair"]["part86_triage_broken"],
                    broken[dose]["pair"]["total"] - broken[dose]["pair"]["part86_triage_broken"]],
                   [broken[dose]["control"]["part86_triage_broken"],
                    broken[dose]["control"]["total"] - broken[dose]["control"]["part86_triage_broken"]],
               ]).pvalue)}
        for dose in ("0.4", "0.8")
    }
    return result


def floor_diagnostics(interactions: list[dict]) -> dict:
    """Measure the 0.2 share after excluding positions where the rounding floor is material."""
    index = scorer.path2_index(interactions)
    support = scorer._common_support(index)
    shares_by_npair = []
    for key, dose_rows in index.items():
        if key[0] != "real" or 0.2 not in dose_rows:
            continue
        positions = scorer._position_map(dose_rows[0.2])
        for pos in support.get(key, set()):
            item = positions.get(pos)
            if item and scorer.resolved(item):
                npair = float(item["nPair"])
                interaction_share = (float(item["nI"]) / npair) ** 2
                shares_by_npair.append((npair, interaction_share))
    result = {"common_real_positions_0.2": len(shares_by_npair)}
    for threshold in (2, 3, 5):
        selected = [share for npair, share in shares_by_npair if npair >= threshold]
        result[f"nPair_at_least_{threshold}"] = {
            "positions": len(selected), "median_share": float(statistics.median(selected)) if selected else None,
            "approx_rounding_floor": (0.36 / threshold) ** 2,
        }
    result["nPair_below_2"] = sum(npair < 2 for npair, _ in shares_by_npair)
    result["nPair_below_3"] = sum(npair < 3 for npair, _ in shares_by_npair)
    return result


def path1_diagnostics(path1: list[dict], part119_score: dict) -> dict:
    contested = [position for row in path1 for position in row.get("contested", [])]
    mismatches = sum(position["g"] != position["both_top"] for position in contested)
    return {
        "contested_positions": len(contested),
        "generated_token_differs_from_both_top": mismatches,
        "mismatch_share": mismatches / len(contested) if contested else None,
        "part119_0.8_additive_mean": part119_score["logit_summary"]["real_pairs"]["additive_mean"],
        "part119_0.8_control_additive_mean": part119_score["logit_summary"]["controls"]["additive_mean"],
    }


def answer_share(score: dict, row_type: str, dose: float) -> float | None:
    row = score["broken_answers_by_dose_and_class"].get(f"{dose:g}", {}).get(row_type, {})
    return row.get("part86_triage_share")


def draw(score: dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15.5, 13.2))
    fig.subplots_adjust(left=0.09, right=0.77, top=0.95, bottom=0.06, hspace=0.48)
    palette = plt.get_cmap("tab10").colors

    # Panel 1: answer-level triage plus the paired sample's reader coherence.
    ax = axes[0]
    answer_series = [
        ("pair", "Real pairs", palette[0]),
        ("control", "A + random control", palette[1]),
        ("randpair", "Two random pushes", palette[2]),
        ("single", "Neuron singles", palette[3]),
        ("random", "Random singles", palette[4]),
    ]
    for row_type, label, color in answer_series:
        xs, ys = [], []
        for dose in ANSWER_DOSES:
            value = answer_share(score, row_type, dose)
            if value is not None:
                xs.append(dose)
                ys.append(value)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4.5,
                    color=color, label=f"Triage: {label}")
    reader = score["reader_coherence"]
    coherent_04 = reader["real_pair_coherent_count_0.4"]["coherent_share_of_expected"]
    paired = reader["paired_0.4_vs_part119_0.8_by_class"]
    reader_08 = sum(value["coherent_08"] for name, value in paired.items()
                    if name != "control") / sum(value["paired"] for name, value in paired.items()
                                                  if name != "control")
    ax.plot([0.4, 0.8], [coherent_04, reader_08], linestyle="none", marker="D",
            markersize=6, color="black", label="Reader coherent: real-pair sample")
    ax.set_title("Answer breaking and reader coherence")
    ax.set_ylabel("Share of answers")
    ax.set_xlabel("Push dose α")
    ax.set_xlim(0.15, 1.68)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(ANSWER_DOSES, [f"{dose:g}" for dose in ANSWER_DOSES])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.015, 1), fontsize=8.5)

    # Panel 2: common-support row interaction shares and the Part 119 reference.
    ax = axes[1]
    summary = score["path2_summary"]["by_dose"]
    for kind, label, color in (("real", "Real rows", palette[0]),
                               ("control", "A + random rows", palette[1])):
        values = [summary[f"{dose:g}"][kind]["median_row_share_common_resolved_all_doses"]
                  for dose in DOSES]
        ax.plot(DOSES, values, marker="o", linewidth=1.8, markersize=4.5,
                color=color, label=f"Part 119b: {label}")
    reference = score["gates"]["gate2"]["row_medians"]
    ax.scatter([0.8], [reference["real"]["reference_median_row_share_0.8"]],
               marker="^", s=62, facecolors="white", edgecolors=palette[0], linewidths=1.5,
               label="Part 119 reference: real")
    ax.scatter([0.8], [reference["control"]["reference_median_row_share_0.8"]],
               marker="^", s=62, facecolors="white", edgecolors=palette[1], linewidths=1.5,
               label="Part 119 reference: control")
    ax.axhline(0.15, color="#555555", linestyle="--", linewidth=1.2,
               label="Gate 3 share limit (0.15)")
    ax.set_title("Median interaction share on positions resolved at every dose")
    ax.set_ylabel("Median row interaction share")
    ax.set_xlabel("Dose α")
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0, 0.55)
    ax.set_xticks(DOSES, [f"{dose:g}" for dose in DOSES])
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.015, 1), fontsize=8.5)

    # Panel 3: row-median slope coefficients for both real and matched control rows.
    ax = axes[2]
    for kind, class_label, color in (("real", "real", palette[0]),
                                     ("control", "control", palette[1])):
        values = [summary[f"{dose:g}"][kind]["median_row_b_A_common_resolved"]
                  for dose in DOSES]
        ax.plot(DOSES, values, marker="o", linewidth=1.8, markersize=4.5,
                color=color, linestyle="-", label=f"b_A: {class_label}")
    for kind, class_label, color in (("real", "real", palette[2]),
                                     ("control", "control", palette[3])):
        values = [summary[f"{dose:g}"][kind]["median_row_b_2_common_resolved"]
                  for dose in DOSES]
        ax.plot(DOSES, values, marker="s", linewidth=1.8, markersize=4.3,
                color=color, linestyle="--", label=f"b_2: {class_label}")
    ax.set_title("Median regression slopes on the same common support")
    ax.set_ylabel("Median row slope")
    ax.set_xlabel("Dose α")
    ax.set_xlim(0.15, 0.85)
    ax.axhspan(0.75, 1.25, color="#d9ead3", alpha=0.65, zorder=0,
               label="119b.6 registered band (0.75–1.25)")
    ax.set_ylim(0, 1.35)
    ax.set_xticks(DOSES, [f"{dose:g}" for dose in DOSES])
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.015, 1), fontsize=8.5)

    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    score = load_score()
    interactions = scorer.read_jsonl("results/part119b/interaction.jsonl")
    answers_b = scorer.read_jsonl("results/part119b/answers.jsonl")
    answers_119 = scorer.read_jsonl("results/part119/answers.jsonl")
    reader_keys_b = scorer.read_jsonl("results/part119b/labels-ds.jsonl.keys.jsonl")
    reader_labels_b = scorer.read_jsonl("results/part119b/labels-ds.jsonl")
    reader_keys_119 = scorer.read_jsonl("results/part119/labels-ds.jsonl.keys.jsonl")
    reader_labels_119 = scorer.read_jsonl("results/part119/labels-ds.jsonl")
    norms = median_norms_by_dose(interactions)
    tv = median_tv_by_dose(interactions)
    draw(score)

    part119_rows = scorer.read_jsonl("results/part119/interaction.jsonl")
    part119_tv_summary = scorer.score119.interaction_summaries(part119_rows)
    part119_score = scorer.read_json("results/part119/score119.json")
    path1_rows = scorer.read_jsonl("results/part119b/path1.jsonl")
    units = rows119.load_units()
    agreed_kind_unit_counts = {
        kind: sum(unit.get("haiku_kind") == kind and unit.get("ds_kind") == kind
                  for unit in units)
        for kind in ("slot", "concept")
    }
    saved = scorer.read_json("results/part119b/score119b.json")
    differences = [key for key in score if score[key] != saved.get(key)]
    print(json.dumps({
        "figure": str(OUT.relative_to(ROOT)),
        "figure_bytes": OUT.stat().st_size,
        "recomputed_score_fields_differing_from_score119b_json": differences,
        "gate2_reference_differences": {
            kind: score["gates"]["gate2"]["row_medians"][kind]["absolute_median_difference"]
            for kind in ("real", "control")
        },
        "median_norms_by_dose": norms,
        "common_support_shares_by_dose": {
            dose: {kind: score["path2_summary"]["by_dose"][dose][kind][
                "median_row_share_common_resolved_all_doses"] for kind in ("real", "control")}
            for dose in ("0.2", "0.4", "0.6", "0.8")
        },
        "all_position_shares_for_gate2_only": {
            kind: score["path2_summary"]["by_dose"]["0.8"][kind]["median_row_share_all_positions"]
            for kind in ("real", "control")
        },
        "median_tv_by_dose_all_positions": tv,
        "part119_tv_reference_from_raw": {
            kind: part119_tv_summary[kind]["median_row_tv"]
            for kind in ("real", "control")
        },
        "part119_tv_gap_119b_minus_119": {
            kind: tv["0.8"][kind] - part119_tv_summary[kind]["median_row_tv"]
            for kind in ("real", "control")
        },
        "assistant_reply_by_dose_and_reader_sample": assistant_reply_summary(
            answers_b, answers_119, reader_keys_b, reader_labels_b,
            reader_keys_119, reader_labels_119),
        "fisher_exact_checks": fisher_checks(score),
        "rounding_floor_diagnostics": floor_diagnostics(interactions),
        "path1_diagnostics": path1_diagnostics(path1_rows, part119_score),
        "agreed_kind_unit_counts": agreed_kind_unit_counts,
        "effect_matched_ratio": {
            "mean_residual": score["effect_matched_0.4"]["mean_residual"],
            "ratio": math.exp(score["effect_matched_0.4"]["mean_residual"]),
            "interval": [math.exp(x) for x in score["effect_matched_0.4"][
                "mean_residual_ci95_product_weighted"]],
            "retained_positions": score["effect_matched_0.4"]["real_positions_kept"],
            "common_support_positions": score["effect_matched_0.4"]["real_positions_total"],
        },
        "headline": score["headline"],
        "gate3": score["gates"]["gate3"],
    }, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
