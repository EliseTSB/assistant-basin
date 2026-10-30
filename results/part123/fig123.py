#!/usr/bin/env python3
"""Recompute Part 123 numbers and draw its dose and overlap figure."""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "part123"))
sys.path.insert(0, str(REPO / "part92b"))

import fit123  # noqa: E402
import label123  # noqa: E402
import score123  # noqa: E402
import score92b  # noqa: E402
import run123  # noqa: E402


def read(path: str):
    return score123.read_json(path)


def readl(path: str):
    return score123.read_jsonl(path)


def qkey(q: float) -> str:
    return format(float(q), "g")


def dose_key(arm: str, q: float | None, direction: int | None = None) -> str:
    q_text = "plain" if q is None else qkey(q)
    suffix = "all" if direction is None else f"direction-{direction}"
    return f"q={q_text}:{suffix}"


def build_result():
    units = fit123.selected_units(
        read("results/part92/step2/neurons.json"),
        read("results/part92/step2/kinds.json"),
    )
    answer_rows = readl("results/part123/main/answers.jsonl")
    items = label123.make_items(answer_rows, units, "main", allow_missing=True)
    labels_ds = readl("results/part123/labels-ds.jsonl")
    keys_ds = readl("results/part123/labels-ds.jsonl.keys.jsonl")
    labels_luna = readl("results/part123/labels-luna.jsonl")
    keys_luna = readl("results/part123/labels-luna.jsonl.keys.jsonl")
    zero_rows = read("results/part123/main/zero-hook-check.json")
    overlap_rows = readl("results/part123/fit/overlap.jsonl")
    null_rows = readl("results/part123/fit/shuffled-null.jsonl")
    known_ds = score123._known_gate("results/part123/known-ds.jsonl", "ds")
    known_luna = score123._known_gate("results/part123/known-luna.jsonl", "luna")

    result = score123.score_data(
        units,
        items,
        keys_ds,
        labels_ds,
        keys_luna,
        labels_luna,
        zero_rows,
        known_ds,
        known_luna,
        overlap_rows,
        null_rows,
    )
    labels_by_key, missing = score123._label_results(items, keys_ds, labels_ds)
    tables, unit_rows, who = score123._aggregate(items, labels_by_key)
    overlap = score123._overlap_results(overlap_rows, null_rows, units)
    return {
        "units": units,
        "answers": answer_rows,
        "items": items,
        "labels": labels_by_key,
        "missing": missing,
        "tables": tables,
        "unit_rows": unit_rows,
        "who": who,
        "result": result,
        "overlap": overlap,
        "main_meta": read("results/part123/main/main-meta.json"),
        "fit_meta": read("results/part123/fit/fit-meta.json"),
        "medians": read("results/part123/main/medians.json"),
    }


def prior_concept_count():
    """Rebuild Part 92b's DeepSeek concept-kind count from its labels."""
    neurons = read("results/part92/step2/neurons.json")
    kinds = read("results/part92/step2/kinds.json")
    source_units = fit123.selected_units(neurons, kinds)
    source_kind = {str(unit["id"]): str(unit["kind"]) for unit in source_units}
    ids = [str(row["id"]) for row in neurons]
    labels, _errors = score92b.load_labels(REPO / "results/part92b/labels-ds.jsonl")
    deepseek_kinds = score92b.classify_units(labels, ids)
    concept_ids = [uid for uid, kind in source_kind.items() if kind == "concept"]
    kept = sum(deepseek_kinds.get(uid) == "concept" for uid in concept_ids)
    became_slot = sum(deepseek_kinds.get(uid) == "slot" for uid in concept_ids)
    unclassified = sum(deepseek_kinds.get(uid) == "unclassified" for uid in concept_ids)
    return {"kept": kept, "total": len(concept_ids), "became_slot": became_slot,
            "unclassified": unclassified}


def metrics(data):
    result = data["result"]
    units = data["units"]
    slots = [unit for unit in units if unit["kind"] == "slot"]
    concepts = [unit for unit in units if unit["kind"] == "concept"]
    rows = data["overlap"]["rows"]
    slot_rows = [rows[str(unit["id"])] for unit in slots]
    all_rows = list(rows.values())

    def who(arm, q, direction=None):
        count, missing, marks, coherent = score123._count_who(
            data["who"], slots, arm, q, direction=direction
        )
        return {"readable": count, "missing": missing, "persona_marks": marks,
                "coherent_marks": coherent}

    concept_q1 = [row for row in data["unit_rows"]
                  if row["arm"] == "vector" and row["q"] == "1" and
                  row["unit_id"] in {str(unit["id"]) for unit in concepts}]
    concept_three = [row for row in concept_q1 if row["shows"] >= 3]
    concept_pass = [row for row in concept_three if row["coherent_shows"] * 3 >= row["shows"] * 2]
    slot_vector_working = [row for row in slots
                           if result["vector_lowest_q_by_unit"][str(row["id"])]["lowest_q_show"] is not None]
    vector_working = [(uid, row) for uid, row in result["vector_lowest_q_by_unit"].items()
                      if row["lowest_q_show"] is not None]

    concept_ranks = [rows[str(unit["id"])]["rank"] for unit in concepts]
    slot_ranks = [rows[str(unit["id"])]["rank"] for unit in slots]
    mwu = score123._mwu_concept_ranks(concept_ranks, slot_ranks)
    gates = {int(gate["number"]): gate for gate in result["gates"]}
    failed_gate7 = [group for group in gates[7]["groups"] if not group["passed"]]
    return {
        "slots": slots,
        "concepts": concepts,
        "gates": gates,
        "failed_gate7": failed_gate7,
        "stage_a": {
            "slot_joint": sum(row["overlap_prediction_hit"] for row in slot_rows),
            "slot_total": len(slot_rows),
            "slot_rank_top_decile": sum(row["top_decile"] for row in slot_rows),
            "slot_above_null": sum(row["above_shuffle_p95"] for row in slot_rows),
            "all_joint": sum(row["overlap_prediction_hit"] for row in all_rows),
            "all_total": len(all_rows),
            "all_rank_top_decile": sum(row["top_decile"] for row in all_rows),
            "all_above_null": sum(row["above_shuffle_p95"] for row in all_rows),
        },
        "who_counts": {
            "vector_q1": who("vector", 1.0),
            "vector_q085": who("vector", 0.85),
            "neuron_q1": who("neuron", 1.0),
            "neuron_q085": who("neuron", 0.85),
            "random_q1": [who("random", 1.0, direction=i) for i in range(3)],
        },
        "mwu": {
            "concept_median": score123._median(concept_ranks),
            "slot_median": score123._median(slot_ranks),
            **mwu,
        },
        "vector_concepts_q1": {
            "three_raw": len(concept_three), "readable": len(concept_pass),
            "total_marks": sum(row["shows"] for row in concept_q1),
            "coherent_marks": sum(row["coherent_shows"] for row in concept_q1),
        },
        "slot_vector_working": slot_vector_working,
        "vector_working": vector_working,
        "prior_concepts": prior_concept_count(),
        "fit_settings": {
            "query_roots": len(fit123.QUERY_ROOTS),
            "template_pairs": len(fit123.TEMPLATE_PAIRS),
            "suffix_lengths": list(fit123.SUFFIX_LENGTHS),
            "pairs_per_unit": (len(fit123.QUERY_ROOTS) * len(fit123.TEMPLATE_PAIRS) *
                               len(fit123.SUFFIX_LENGTHS)),
            "shuffles": fit123.SHUFFLES,
            "questions": len(run123.QUESTIONS),
            "neuron_doses": list(run123.NEURON_DOSES),
            "vector_doses": list(run123.VECTOR_DOSES),
            "random_doses": list(run123.RANDOM_DOSES),
            "random_directions": run123.RANDOM_COUNT,
        },
    }


def dose_table_rows(data):
    tables = data["tables"]
    rows = []
    plain = tables["plain"][dose_key("plain", None)]
    rows.append(("Plain", "all", "plain baseline", plain))
    for q in (0.5, 0.85, 1.0):
        rows.append(("Neuron", "all", q, tables["neuron"][dose_key("neuron", q)]))
    for q in (0.5, 0.85, 1.0, 1.5, 2.0):
        rows.append(("Fitted vector", "all", q, tables["vector"][dose_key("vector", q)]))
    for direction in range(3):
        for q in (0.85, 1.0, 2.0):
            rows.append(("Random", str(direction), q,
                         tables["random"][dose_key("random", q, direction)]))
    return rows


def make_figure(data, stats):
    mpl_config = HERE / ".mplconfig"
    os.environ["MPLCONFIGDIR"] = str(mpl_config)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    labels = ["Plain", "0.5", "0.85", "1", "1.5", "2"]
    x = list(range(len(labels)))
    table = data["tables"]
    palette = {
        "plain": "#697386",
        "neuron": "#2864A5",
        "vector": "#C45D15",
        "random": ("#168A7A", "#4AAB98", "#82C3B1"),
    }

    def values(arm, field, direction=None):
        row_values = [math.nan] * len(labels)
        if arm == "plain":
            row = table["plain"][dose_key("plain", None)]
            row_values[0] = row[field]
            return row_values
        q_values = {"neuron": (0.5, 0.85, 1.0),
                    "vector": (0.5, 0.85, 1.0, 1.5, 2.0),
                    "random": (0.85, 1.0, 2.0)}[arm]
        for q in q_values:
            i = labels.index(qkey(q))
            key = dose_key(arm, q, direction)
            row_values[i] = table[arm][key][field]
        return row_values

    fig = plt.figure(figsize=(15.5, 14.2), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.65),
                            left=0.09, right=0.98, top=0.84, bottom=0.07,
                            hspace=0.31, wspace=0.23)
    ax_marks = fig.add_subplot(grid[0, 0])
    ax_coherent = fig.add_subplot(grid[0, 1])
    ax_overlap = fig.add_subplot(grid[1, :])

    lines = []
    for arm, field, axis in (("plain", "shows", ax_marks),
                             ("plain", "coherent", ax_coherent)):
        line, = axis.plot(x, values(arm, field), marker="o", color=palette["plain"],
                          linewidth=2, markersize=6, label="Plain")
        if field == "shows":
            lines.append(line)
    for arm, name in (("neuron", "Neuron"), ("vector", "Fitted vector")):
        for field, axis in (("shows", ax_marks), ("coherent", ax_coherent)):
            line, = axis.plot(x, values(arm, field), marker="o", color=palette[arm],
                              linewidth=2.4, markersize=6, label=name)
            if axis is ax_marks:
                lines.append(line)
    for direction, color in enumerate(palette["random"]):
        for field, axis in (("shows", ax_marks), ("coherent", ax_coherent)):
            line, = axis.plot(x, values("random", field, direction), marker="o",
                              linestyle="--", color=color, linewidth=1.5, markersize=4,
                              label=f"Random direction {direction}")
            if axis is ax_marks:
                lines.append(line)

    for axis in (ax_marks, ax_coherent):
        axis.set_xticks(x, labels)
        axis.set_xlabel("Dose q; plain model shown at left")
        axis.grid(axis="y", color="#d7dce2", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    ax_marks.set_title("Persona marks among 234 answers", loc="left", weight="bold")
    ax_marks.set_ylabel("Reader marks")
    ax_marks.set_ylim(-2, 78)
    ax_coherent.set_title("Coherent answers among 234", loc="left", weight="bold")
    ax_coherent.set_ylabel("Coherent answers")
    ax_coherent.set_ylim(-5, 247)

    ordered = sorted(data["units"], key=lambda row: (int(row["layer"]),
                                                       0 if row["kind"] == "slot" else 1,
                                                       str(row["id"])))
    overlap_rows = data["overlap"]["rows"]
    y_positions = list(range(len(ordered)))
    kind_color = {"slot": "#2864A5", "concept": "#C45D15"}
    for y, unit in zip(y_positions, ordered):
        row = overlap_rows[str(unit["id"])]
        observed = float(row["signed_cosine"])
        null95 = float(row["shuffle_p95"])
        color = kind_color[str(unit["kind"])]
        ax_overlap.plot([null95, observed], [y, y], color=color, alpha=0.35, linewidth=1)
        marker = "o" if unit["kind"] == "slot" else "s"
        ax_overlap.scatter([observed], [y], marker=marker, color=color,
                           edgecolor="white", linewidth=0.6, s=43, zorder=3)
        ax_overlap.scatter([null95], [y], marker="D", facecolor="white",
                           edgecolor=color, linewidth=1.2, s=36, zorder=4)
    ax_overlap.axvline(0, color="#4b5563", linewidth=0.9, alpha=0.75)
    ax_overlap.set_yticks(y_positions, [str(unit["id"]) for unit in ordered])
    ax_overlap.invert_yaxis()
    ax_overlap.tick_params(axis="y", labelsize=7.3, length=0)
    ax_overlap.set_xlabel("Signed cosine with the matching Part 92 neuron")
    ax_overlap.set_title(
        "Stage A: fitted-vector overlap and its shuffled-label reference",
        loc="left", weight="bold", pad=11,
    )
    ax_overlap.text(
        0.995, 1.015,
        f"Filled: observed. Open diamond: shuffle 95th percentile. "
        f"Observed joint slot hits: {stats['stage_a']['slot_joint']}/{stats['stage_a']['slot_total']} (rule: 6).",
        transform=ax_overlap.transAxes, ha="right", va="bottom", fontsize=8.2,
    )
    ax_overlap.grid(axis="x", color="#d7dce2", linewidth=0.8)
    ax_overlap.spines[["top", "right"]].set_visible(False)
    overlap_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=kind_color["slot"],
               label="Slot unit"),
        Line2D([0], [0], marker="s", linestyle="None", color=kind_color["concept"],
               label="Concept unit"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#374151",
               markeredgecolor="white", label="Observed cosine"),
        Line2D([0], [0], marker="D", linestyle="None", markerfacecolor="white",
               markeredgecolor="#374151", label="Shuffle 95th percentile"),
    ]
    ax_overlap.legend(handles=overlap_handles, loc="lower right", frameon=True,
                      fontsize=8, ncol=2)
    fig.suptitle(
        "Part 123 | persona marks, coherence, and fitted-vector overlap",
        x=0.09, y=0.985, ha="left", fontsize=17, weight="bold",
    )
    fig.text(0.09, 0.948,
             "OLMo 2 1B Instruct. DeepSeek labels. Gates 3 and 7 failed, so predictions are not read.",
             ha="left", fontsize=10, color="#4b5563")
    fig.legend(handles=lines, loc="upper center", bbox_to_anchor=(0.55, 0.904),
               ncol=6, frameon=False, fontsize=8)
    fig.savefig(HERE / "fig123.png", dpi=190, facecolor="white")
    plt.close(fig)
    if mpl_config.exists():
        shutil.rmtree(mpl_config)


def fnum(value, digits=6):
    return f"{float(value):.{digits}f}"


def make_numbers(data, stats):
    gate = stats["gates"]
    lines = [
        "# Part 123 numbers",
        "",
        "fig123.py rebuilds these values from the raw Part 123 files. It calls the Part 123 scorer's label joins, dose aggregation, gate logic, and Stage A overlap calculation. The saved score is not an input.",
        "",
        "## Run and labels",
        "",
        "| Measure | Value | Source |",
        "| --- | ---: | --- |",
        f"| Model | {data['fit_meta']['model']} | fit/fit-meta.json and PART123-DESIGN.md |",
        f"| Revision | {data['fit_meta']['revision']} | fit/fit-meta.json and PART123-DESIGN.md |",
        f"| Units and strata | {len(data['units'])} total; {len(stats['slots'])} slots; {len(stats['concepts'])} concepts | fit/fit-meta.json and Part 92 unit-kind files |",
        f"| Layers represented | {len(data['main_meta']['layers'])}: {', '.join(str(x) for x in data['main_meta']['layers'])} | main/main-meta.json |",
        f"| Fit prompt pairs | {stats['fit_settings']['pairs_per_unit']} positive-negative pairs per unit from {stats['fit_settings']['query_roots']} query roots, {stats['fit_settings']['template_pairs']} templates, and {len(stats['fit_settings']['suffix_lengths'])} suffix lengths | part123/fit123.py and PART123-DESIGN.md |",
        f"| Fit prompt sides | {data['fit_meta']['fit_prompts']} last-token states across 39 units | fit/fit-meta.json |",
        f"| Label shuffles per fitted unit | {stats['fit_settings']['shuffles']} | fit/fit-meta.json and part123/fit123.py |",
        f"| Main questions per unit and dose | {stats['fit_settings']['questions']} | part123/run123.py |",
        f"| Registered dose sets | neuron {', '.join(format(q, 'g') for q in stats['fit_settings']['neuron_doses'])}; vector {', '.join(format(q, 'g') for q in stats['fit_settings']['vector_doses'])}; random {', '.join(format(q, 'g') for q in stats['fit_settings']['random_doses'])} with {stats['fit_settings']['random_directions']} directions | part123/run123.py and PART123-DESIGN.md |",
        "| Written push norm | q times 0.8 times the median residual norm; q=1 is Part 92's alpha=0.8 size | PART123-DESIGN.md |",
        "| Precision and device | bf16 on Intel Arc A770 | PART123-DESIGN.md and part123/BUILD-NOTES.md |",
        f"| Main answer rows | {len(data['answers'])} | main/answers.jsonl |",
        f"| Main row target | {data['main_meta']['expected_rows']} | main/main-meta.json |",
        f"| Zero-hook matches | {gate[1]['equal']} of {gate[1]['total']} | main/zero-hook-check.json; recomputed by score123.py |",
        f"| DeepSeek label items | {data['result']['totals']['reader_items']} of {data['result']['totals']['expected_reader_items']} | labels-ds.jsonl, labels-ds.jsonl.keys.jsonl, main/answers.jsonl |",
        f"| Missing or stale DeepSeek labels | {len(data['missing'])} | same raw answer and label files; recomputed by score123.py |",
        f"| Main batch size | {data['main_meta']['batch_size']} | main/main-meta.json |",
        f"| Main token cap | {data['main_meta']['max_new_tokens']} | main/main-meta.json |",
        "",
        "## Gates",
        "",
        "Gate 7 applies to every arm, coefficient, and random direction. A group passes when at least two thirds of its persona marks are coherent; a group with no marks passes.",
        "",
        "| Gate | Observed | Requirement | Result | Source |",
        "| --- | --- | --- | --- | --- |",
        f"| 1, zero hook | {gate[1]['equal']}/{gate[1]['total']} exact token matches | {gate[1]['required']} | {'pass' if gate[1]['passed'] else 'fail'} | main/zero-hook-check.json |",
        f"| 2, neuron slot control at q=1 | {gate[2]['readable_shows']}/{gate[2]['total']} readable on Who are you?; {gate[2]['shows']} marks, {gate[2]['coherent_shows']} coherent marks | at least 17/21 | {'pass' if gate[2]['passed'] else 'fail'} | main/answers.jsonl and DeepSeek labels; score123.py |",
        f"| 3, neuron concept control at q=1 | {gate[3]['three_of_six']}/{gate[3]['total']} meet the coherent three-of-six rule; {gate[3]['persona_mark_three_of_six']}/{gate[3]['total']} have three raw marks; {gate[3]['persona_marks']} marks, {gate[3]['coherent_persona_marks']} coherent | at least 14/18 | {'pass' if gate[3]['passed'] else 'fail'} | main/answers.jsonl and DeepSeek labels; score123.py |",
        f"| 4, complete DeepSeek labels | {data['result']['totals']['reader_items']}/{data['result']['totals']['expected_reader_items']}; {len(data['missing'])} missing or stale | all current labels | {'pass' if gate[4]['passed'] else 'fail'} | answer rows, label keys, and labels; score123.py |",
        f"| 5, known examples | DeepSeek {gate[5]['deepseek']['correct']}/{gate[5]['deepseek']['total']}; Luna {gate[5]['luna']['correct']}/{gate[5]['luna']['total']} | at least 10/12 for each | {'pass' if gate[5]['passed'] else 'fail'} | known-ds.jsonl, known-luna.jsonl; score123.py |",
        f"| 6, Luna agreement | {gate[6]['agreement']['agree']}/{gate[6]['agreement']['total']} ({gate[6]['agreement']['agreement']:.4%}); kappa {gate[6]['agreement']['kappa']:.6f} | at least 85% and kappa at least 0.6 | {'pass' if gate[6]['passed'] else 'fail'} | labels-ds.jsonl, labels-luna.jsonl, both key files; score123.py |",
        f"| 7, readable persona | {len(stats['failed_gate7'])}/{len(gate[7]['groups'])} groups fail | at least two thirds of persona marks coherent in every group | {'pass' if gate[7]['passed'] else 'fail'} | same raw answer and label files; score123.py |",
        "",
        "### Gate 7 failed groups",
        "",
        "| Arm | q and direction | Persona marks | Coherent persona marks | Source |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in stats["failed_gate7"]:
        lines.append(
            f"| {row['arm']} | {row['dose_and_direction']} | {row['persona_marks']} | "
            f"{row['coherent_persona_marks']} | raw answer labels; score123.py |"
        )

    lines += [
        "",
        "## Persona marks and coherent answers by dose",
        "",
        "Persona marks count DeepSeek labels with shows_persona=true. Coherent answers count all coherent answer labels in that 234-answer group, whether or not they show the target persona. Coherent persona marks count the intersection. Each cell is recomputed with score123.py from the main answers, labels, and private label keys.",
        "",
        "| Arm | Direction | q | Persona marks / 234 | Coherent persona marks | Coherent answers / 234 | Self claims | Source |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm, direction, q, row in dose_table_rows(data):
        q_text = q if isinstance(q, str) else format(float(q), "g")
        lines.append(
            f"| {arm} | {direction} | {q_text} | {row['shows']} | {row['coherent_shows']} | "
            f"{row['coherent']} | {row['self_claim']} | main/answers.jsonl, labels-ds.jsonl, labels-ds.jsonl.keys.jsonl; score123.py |"
        )

    w = stats["who_counts"]
    random_who = [entry["readable"] for entry in w["random_q1"]]
    lines += [
        "",
        "## Registered prediction observations",
        "",
        "These are the raw observed values. The verdict column remains not read because gates 3 and 7 failed.",
        "",
        "| Prediction | Observed value | Source |",
        "| --- | --- | --- |",
        f"| 123.1 | {stats['stage_a']['slot_joint']}/{stats['stage_a']['slot_total']} slot units meet both rank and shuffle conditions; {stats['stage_a']['slot_rank_top_decile']}/{stats['stage_a']['slot_total']} rank in the top decile; {stats['stage_a']['slot_above_null']}/{stats['stage_a']['slot_total']} exceed their shuffle p95. | fit/overlap.jsonl, fit/shuffled-null.jsonl, Part 92 unit files; score123.py |",
        f"| 123.2 | Fitted vector {w['vector_q1']['readable']}/21 readable on Who are you?; random directions {', '.join(f'{n}/21' for n in random_who)}. | main/answers.jsonl, DeepSeek label files; score123.py |",
        f"| 123.3 | Neuron readable slot units at q=1 to q=0.85: {w['neuron_q1']['readable']} to {w['neuron_q085']['readable']} of 21; change {w['neuron_q1']['readable'] - w['neuron_q085']['readable']}. | main/answers.jsonl, DeepSeek label files; score123.py |",
        f"| 123.4 | Fitted vector readable slot units at q=1 to q=0.85: {w['vector_q1']['readable']} to {w['vector_q085']['readable']} of 21; neuron: {w['neuron_q1']['readable']} to {w['neuron_q085']['readable']}. | main/answers.jsonl, DeepSeek label files; score123.py |",
        f"| 123.5 | Concept median rank {stats['mwu']['concept_median']}; slot median rank {stats['mwu']['slot_median']}; U={stats['mwu']['statistic']:.1f}, one-sided p={stats['mwu']['p_value']:.6f}. | fit/overlap.jsonl, fit/shuffled-null.jsonl, Part 92 unit files; score123.py |",
        f"| 123.6 | {stats['vector_concepts_q1']['readable']}/18 meet the coherent three-of-six rule at vector q=1; {stats['vector_concepts_q1']['three_raw']}/18 have at least three raw marks; {stats['vector_concepts_q1']['total_marks']} total marks, {stats['vector_concepts_q1']['coherent_marks']} coherent marks. | main/answers.jsonl, DeepSeek label files; score123.py |",
        "",
        "## Stage A overlap, recomputed by the scorer",
        "",
        f"Among slots, {stats['stage_a']['slot_joint']}/{stats['stage_a']['slot_total']} fitted vectors met both the layer top-decile rank rule and the signed-cosine-above-shuffle-p95 rule. Across all units, {stats['stage_a']['all_joint']}/{stats['stage_a']['all_total']} met both. The scorer recomputes each unit's 95th percentile from its 1,000 shuffled cosines using its registered order statistic.",
        "",
        "| Unit | Kind | Layer | Channels | Neuron rank | Top-decile limit | Signed cosine | Shuffle p95 | Top decile | Above p95 | Both | Source |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    overlap_rows = data["overlap"]["rows"]
    for unit in data["units"]:
        row = overlap_rows[str(unit["id"])]
        raw = next(item for item in readl("results/part123/fit/overlap.jsonl")
                   if str(item.get("unit_id")) == str(unit["id"]))
        lines.append(
            f"| {unit['id']} | {unit['kind']} | {row['layer']} | {raw['channel_count']} | {row['rank']} | "
            f"{row['rank_limit']} | {fnum(row['signed_cosine'])} | {fnum(row['shuffle_p95'])} | "
            f"{'yes' if row['top_decile'] else 'no'} | {'yes' if row['above_shuffle_p95'] else 'no'} | "
            f"{'yes' if row['overlap_prediction_hit'] else 'no'} | fit/overlap.jsonl, fit/shuffled-null.jsonl; scorer recomputation |"
        )

    concept_winners = [(uid, row) for uid, row in stats["vector_working"]
                       if row["kind"] == "concept"]
    lines += [
        "",
        "## Unit-level vector show rule",
        "",
        "| Part 92 group | Units meeting the rule at any q | Source |",
        "| --- | ---: | --- |",
        f"| Slots | {len(stats['slot_vector_working'])}/{len(stats['slots'])} | main/answers.jsonl and DeepSeek labels; score123.py |",
        f"| Concepts | {len(concept_winners)}/{len(stats['concepts'])}; "
        f"{', '.join(f'{uid} at q={row['lowest_q_show']:g}' for uid, row in concept_winners) or 'none'} | "
        "main/answers.jsonl and DeepSeek labels; score123.py |",
        f"| All units | {len(stats['vector_working'])}/{len(data['units'])} | main/answers.jsonl and DeepSeek labels; score123.py |",
    ]

    prior = stats["prior_concepts"]
    lines += [
        "",
        "## Earlier DeepSeek concept labels",
        "",
        f"Part 92b's six-question rule classifies {prior['kept']}/{prior['total']} Part 92 concept units as concepts under DeepSeek; {prior['became_slot']} become slots and {prior['unclassified']} are unclassified. fig123.py recomputes these kinds from Part 92b's raw DeepSeek labels and Part 92's unit-kind files using score92b.py. Source: part92b/labels-ds.jsonl and Part 92 unit files.",
        "",
        "## The one vector that met a show rule",
        "",
        "The Part 123 scorer finds one of 39 fitted vectors meeting its unit-level show rule: p92-49, Friendly non-human entity, at q=0.5. The table below gives the DeepSeek label and coherence for each of its six raw answers.",
        "",
        "| Question | Persona mark | Coherent | Source |",
        "| --- | --- | --- | --- |",
    ]
    p92_49 = []
    for item in data["items"]:
        private = item["private"]
        if (private["unit_id"] == "p92-49" and private["arm"] == "vector" and
                float(private["q"]) == 0.5):
            label = data["labels"][(str(private["row_id"]), "p92-49", int(private["question_index"]))]
            p92_49.append((int(private["question_index"]), item["q"], label))
    for qi, question, label in sorted(p92_49):
        lines.append(
            f"| {question} | {'yes' if label['shows_persona'] else 'no'} | "
            f"{'yes' if label['coherent'] else 'no'} | main/answers.jsonl and DeepSeek labels |"
        )
    lines += [
        "",
        "The answer text is in main/answers.jsonl. Its first three replies call itself an AI language model, describe helping people, and say it has no human desires but is available to help. The other replies answer about a perfect day, Paris, and the sea without a persona mark.",
        "",
    ]
    return "\n".join(lines)


def main():
    data = build_result()
    stats = metrics(data)
    make_figure(data, stats)
    (HERE / "NUMBERS.md").write_text(make_numbers(data, stats) + "\n", encoding="utf-8")
    print(json.dumps({
        "figure": str(HERE / "fig123.png"),
        "numbers": str(HERE / "NUMBERS.md"),
        "failed_gates": [number for number, row in stats["gates"].items() if not row["passed"]],
        "stage_a_slot_joint": [stats["stage_a"]["slot_joint"], stats["stage_a"]["slot_total"]],
        "vector_working": stats["vector_working"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
