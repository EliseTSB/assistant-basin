#!/usr/bin/env python3
"""Recompute Part 118's figure and number audit from the saved JSONL files."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/part118-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "part118"))
import score118

OUT = Path(__file__).resolve().parent
ANSWERS_PATH = OUT / "answers.jsonl"
LABELS_PATH = OUT / "labels.jsonl"
DS_PATH = OUT / "labels-ds.jsonl"
NEURONS_PATH = ROOT / "results/part92/step2/neurons.json"
KINDS_PATH = ROOT / "results/part92/step2/kinds.json"
REFERENCE_PATH = ROOT / "results/part92/step2/answers/instruct_chat.jsonl"
PART92_LABELS_PATH = ROOT / "results/part92/step2/labels-instruct_chat.jsonl"
PART92B_DS_PATH = ROOT / "results/part92b/labels-ds.jsonl"
CAP500_LABELS_PATH = OUT / "labels-ds-cap500.jsonl"
CAP500_USAGE_PATH = OUT / "labels-ds-cap500.usage.jsonl"
ALPHAS = (0.4, 0.8, 1.6, 3.2)
QUESTIONS = score118.QUESTIONS


def read_units() -> list[dict]:
    neuron_rows = json.loads(NEURONS_PATH.read_text(encoding="utf-8"))
    kind_data = json.loads(KINDS_PATH.read_text(encoding="utf-8"))
    kind_map = kind_data.get("kinds", kind_data)
    units = [
        {**row, "kind": kind_map[str(row["id"])]}
        for row in neuron_rows
        if kind_map.get(str(row["id"])) in ("concept", "slot")
    ]
    if len(units) != 39:
        raise ValueError(f"expected 39 classified units, found {len(units)}")
    return units


def ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)"


def replay_comparison(answer_rows: list[dict]) -> dict:
    current = {
        (str(row["id"]), int(row["question_index"])): str(row["answer"])
        for row in answer_rows
        if row.get("arm") == "neuron" and math.isclose(
            float(row.get("alpha", -1)), 0.8, rel_tol=0, abs_tol=1e-8
        )
    }
    reference = {
        (str(row["id"]), int(row["question_index"])): str(row["answer"])
        for row in score118.read_jsonl(REFERENCE_PATH)
        if row.get("arm", row.get("unit")) == "neuron"
    }
    keys = sorted(current)
    return {
        "compared": len(keys),
        "exact": sum(current[key] == reference.get(key) for key in keys),
        "first40": sum(current[key][:40] == reference.get(key, "")[:40] for key in keys),
    }


def random_question_reach(units: list[dict], label_map: dict) -> dict:
    result = {}
    for alpha in ALPHAS:
        for qi in range(6):
            showing = sum(
                score118.shows(label_map.get((str(unit["id"]), "random", alpha, qi)))
                for unit in units
            )
            result[(alpha, qi)] = (showing, len(units))
    return result


def reader_dependence(units: list[dict]) -> dict:
    def simple_map(rows: list[dict]) -> dict:
        mapped = {}
        for row in rows:
            arm = row.get("arm", row.get("unit"))
            mapped[(str(row["id"]), arm, int(row["question_index"]))] = row
        return mapped

    haiku_map = simple_map(score118.read_jsonl(PART92_LABELS_PATH))
    ds_map = simple_map(score118.read_jsonl(PART92B_DS_PATH))
    shared = sorted(set(haiku_map) & set(ds_map))
    valid = [key for key in shared
             if not score118.is_error_label(haiku_map[key])
             and not score118.is_error_label(ds_map[key])]
    kappa = score118.cohen_kappa(
        [score118.flag(haiku_map[key], "shows_persona") for key in valid],
        [score118.flag(ds_map[key], "shows_persona") for key in valid],
    )
    unit_counts = {}
    for kind in ("concept", "slot"):
        kind_units = [unit for unit in units if unit["kind"] == kind]
        target_class = "concept-like" if kind == "concept" else "slot-like"
        classes = {
            str(unit["id"]): score118.classify_unit(
                {qi: ds_map.get((str(unit["id"]), "neuron", qi)) for qi in range(6)},
                require_coherent=False,
            )
            for unit in kind_units
        }
        unit_counts[kind] = {
            "same": sum(value == target_class for value in classes.values()),
            "total": len(kind_units),
            "classes": {name: sum(value == name for value in classes.values())
                        for name in ("concept-like", "slot-like", "none")},
        }
        haiku_classes = {
            str(unit["id"]): score118.classify_unit(
                {qi: haiku_map.get((str(unit["id"]), "neuron", qi)) for qi in range(6)},
                require_coherent=True,
            )
            for unit in kind_units
        }
        unit_counts[kind]["coherent_same"] = sum(
            value == target_class for value in haiku_classes.values()
        )
    return {"shared": len(shared), "valid": len(valid), "kappa": kappa,
            "unit_counts": unit_counts}


def old_cap_counts() -> dict:
    cap_rows = score118.read_jsonl(CAP500_LABELS_PATH)
    cap_map = score118._record_map(cap_rows, "DS cap-500 label")
    usage_rows = score118.read_jsonl(CAP500_USAGE_PATH)
    return {
        "sample_labels": len(cap_map),
        "terminal_errors": sum(score118.is_error_label(row) for row in cap_map.values()),
        "usage_records": len(usage_rows),
        "max_tokens": sum(row.get("stop_reason") == "max_tokens" for row in usage_rows),
    }


def draw_figure(question_reach: list[dict], random_reach: dict) -> None:
    colors = {"concept": "#2463a6", "slot": "#d07818", "random": "#666666"}
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.7), sharex=True, sharey=True)
    for qi, ax in enumerate(axes.flat):
        for kind in ("concept", "slot"):
            rows = {
                row["alpha"]: row
                for row in question_reach
                if row["kind"] == kind and row["question_index"] == qi
            }
            values = [rows[alpha]["showing"] / rows[alpha]["total"] for alpha in ALPHAS]
            ax.plot(ALPHAS, values, marker="o", linewidth=2, color=colors[kind],
                    label=f"{kind.title()} units")
        random_values = [random_reach[(alpha, qi)][0] / random_reach[(alpha, qi)][1]
                         for alpha in ALPHAS]
        ax.plot(ALPHAS, random_values, marker="s", linewidth=1.8, linestyle="--",
                color=colors["random"], label="Random directions")
        ax.axvline(0.8, color="#333333", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.set_title(f"Q{qi}. {QUESTIONS[qi]}", fontsize=10, pad=8)
        ax.set_ylim(-0.03, 1.04)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ALPHAS, labels=[f"{alpha:g}" for alpha in ALPHAS])
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Coherent answers with persona marks by question and dose", fontsize=16, y=0.98)
    fig.text(0.5, 0.925,
             "Haiku labels; each point requires a persona mark and a coherent answer. Dotted line: α=0.8 selection dose. At α=1.6+, all neuron answers are incoherent.",
             ha="center", fontsize=10)
    fig.supxlabel("Dose α (multiples of the selection dose; log base 2)", y=0.06)
    fig.supylabel("Share of units with a coherent persona show", x=0.04)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.16,
                        hspace=0.36, wspace=0.12)
    fig.savefig(OUT / "fig118.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def excerpt_rows(answer_map: dict, label_map: dict) -> list[str]:
    cells = (
        ("p92-49", "neuron", 0.4, 0, "low-dose positive"),
        ("p92-02", "neuron", 0.8, 0, "selection-dose positive"),
        ("p92-00", "neuron", 0.8, 1, "coherent, not marked as persona"),
        ("p92-01", "neuron", 1.6, 0, "persona flag, incoherent"),
        ("p92-00", "random", 3.2, 0, "random-direction incoherence"),
    )
    lines = []
    for uid, arm, alpha, qi, note in cells:
        key = (uid, arm, alpha, qi)
        answer = answer_map[key]["answer"].replace("\n", " / ").replace("|", "\\|")
        label = label_map.get(key)
        excerpt = answer[:150].rstrip()
        if len(answer) > 150:
            excerpt += "..."
        lines.append(
            f"| {uid} / {arm} / α={alpha:g} / q{qi} | {QUESTIONS[qi]} | {note} "
            f"(shows_persona={label.get('shows_persona')}, coherent={label.get('coherent')}) | “{excerpt}” |"
        )
    return lines


def write_numbers(result: dict, replay: dict, units: list[dict],
                  answer_rows: list[dict], primary_rows: list[dict],
                  ds_rows: list[dict]) -> None:
    answer_map = score118._record_map(answer_rows, "answer")
    primary_map = score118._record_map(primary_rows, "primary label")
    random_reach = random_question_reach(units, primary_map)
    error_keys = sorted(key for key, row in primary_map.items()
                        if score118.is_error_label(row))
    gate1b_by_kind = {}
    for kind in ("concept", "slot"):
        kind_units = [unit for unit in units if unit["kind"] == kind]
        target_class = "concept-like" if kind == "concept" else "slot-like"
        kept = sum(result["part92_unit_classes"][0.8][str(unit["id"])] == target_class
                   for unit in kind_units)
        gate1b_by_kind[kind] = (kept, len(kind_units))
    reader_audit = reader_dependence(units)
    old_cap = old_cap_counts()
    lines = [
        "# Part 118 number audit",
        "",
        "fig118.py reads the files below, calls the Part 118 scorer, recomputes the replay and random-arm question shares, draws fig118.png, and writes this file.",
        "",
        "## Inputs and rules",
        "",
        "- answers.jsonl: all generated answers, including the six unpushed baselines; baseline rows are excluded from the 1,872 scored cells.",
        "- labels.jsonl: primary Haiku labels. Repeated retry rows are append-only; the last row for a cell is used.",
        "- labels-ds.jsonl: the final 4,000-token DeepSeek V4.1 Flash labels for the seeded Gate 2 sample.",
        "- ../../results/part92/step2/neurons.json and kinds.json: the 39 selected units and their Part 92 kinds.",
        "- ../../results/part92/step2/answers/instruct_chat.jsonl: saved Part 92 answers used for the replay comparison.",
        "- ../../results/part92/step2/labels-instruct_chat.jsonl and ../../results/part92b/labels-ds.jsonl: Haiku and DS labels used to recompute the Part 92b reader comparison.",
        "- labels-ds-cap500.jsonl and labels-ds-cap500.usage.jsonl: the retained, unread 500-token Gate 2 attempt and its usage records.",
        "- ../../part118/score118.py: registered rules, Gate 1b, Gate 2, predictions, and exploratory tables.",
        "",
        "All current-run tables and the figure below are recomputed from these raw files. The 24-answer pilot is not retained here. Its counts and timing are recorded in Amendment 1 of ../../PART118-DESIGN.md; the later cap amendment is cross-checked against the retained usage sidecar below.",
        "",
        "## Amendment 1 pilot record",
        "",
        "These historical values are transcribed from Amendment 1 of ../../PART118-DESIGN.md; the pilot JSONL is not present in results/part118/.",
        "",
        "- Four units at α=0.8 gave 24 pilot answers. Exact replay was 15/24 (62.5%). The other 9 diverged within their first 26 characters.",
        "- The pilot ran on the A770 at batch 24. It measured 24 answers in 18.8 seconds, or 1.28 answers per second (reported in the design as about 1.3).",
        "- Part 92 ran on a rented RTX 3090 at batch 192. No prediction changed in this amendment.",
        "",
        "## Part 92b reader comparison",
        "",
        f"- Shared Haiku and DS labels: {reader_audit['valid']}/{reader_audit['shared']} valid answer cells; Cohen's κ on shows_persona = {reader_audit['kappa']:.6f}.",
        f"- On these selected units, DS keeps {reader_audit['unit_counts']['concept']['same']}/{reader_audit['unit_counts']['concept']['total']} concepts as concept-like and {reader_audit['unit_counts']['slot']['same']}/{reader_audit['unit_counts']['slot']['total']} slots as slot-like. Concept outcomes under DS: {reader_audit['unit_counts']['concept']['classes']}.",
        f"- Reapplying Part 118's coherent rule to Part 92's Haiku labels keeps {reader_audit['unit_counts']['concept']['coherent_same']}/{reader_audit['unit_counts']['concept']['total']} concepts and {reader_audit['unit_counts']['slot']['coherent_same']}/{reader_audit['unit_counts']['slot']['total']} slots, or {reader_audit['unit_counts']['concept']['coherent_same'] + reader_audit['unit_counts']['slot']['coherent_same']}/{len(units)} kinds. This is the count cited in Amendment 2.",
        "Source: Part 92's Haiku labels and Part 92b's DS labels, using Part 92's shows_persona-only classification rule.",
        "",
        "## Amendment 3 cap audit",
        "",
        f"- The saved 500-token usage sidecar has {old_cap['max_tokens']} max_tokens records among {old_cap['usage_records']} rows. The Amendment 3 text records 92 among 365 attempts. The one-record difference is unresolved; the sidecar also contains repeated cell and attempt keys.",
        f"- The retained 500-token label file has {old_cap['terminal_errors']} latest error cells among {old_cap['sample_labels']} sample cells. This matches the retry log before the 4,000-token rerun.",
        "",
        "## Gate 1 replay description",
        "",
        "The original Gate 1 required exact equality on at least 95% of the 234 alpha-0.8 neuron cells. Amendment 1 replaced this gate after the pilot. The full-run rates below are descriptive only.",
        "",
        "| Check | Matches | Cells | Share |",
        "| --- | ---: | ---: | ---: |",
        f"| Exact full answer | {replay['exact']} | {replay['compared']} | {100 * replay['exact'] / replay['compared']:.1f}% |",
        f"| Exact first 40 characters | {replay['first40']} | {replay['compared']} | {100 * replay['first40'] / replay['compared']:.1f}% |",
        "",
        "Source: compare the alpha-0.8 neuron cells in answers.jsonl with the same (id, question_index) in the saved Part 92 answers. Prefix comparison uses the first 40 characters of each answer.",
        "",
        "## Gate and readiness values",
        "",
        f"- Expected scored cells: {result['expected_cells']}; answers present: {result['answers_present']}; primary labels present: {result['primary_labels_present']}; missing answers or primary labels: {len(result['missing_answers']) + len(result['missing_primary_labels'])}.",
        f"- Primary Haiku error cells after retries: {result['primary_label_errors']}. Errors count as not showing and not coherent.",
        f"- Gate 1b: {result['gate1b']['kept']}/{result['gate1b']['units']} units keep their Part 92 kind at α=0.8 using shows_persona alone: {gate1b_by_kind['concept'][0]}/{gate1b_by_kind['concept'][1]} concepts and {gate1b_by_kind['slot'][0]}/{gate1b_by_kind['slot'][1]} slots; threshold {result['gate1b']['threshold']}; {'pass' if result['gate1b']['passed'] else 'fail'}.",
        f"- Gate 2: seeded {score118.DS_SHARE:.0%} sample, seed {result['gate2']['sample_seed']}; {result['ds_sample_present']}/{result['gate2']['sample_size']} DS labels present; {result['ds_label_errors']} final DS errors; Cohen's κ on shows_persona = {result['gate2']['kappa']:.6f}; threshold {result['gate2']['threshold']:.3f}; {'pass' if result['gate2']['passed'] else 'fail'}.",
        "- The original Gate 1 threshold was 95% exact replay over 234 cells. It was withdrawn by Amendment 1, so it is not counted as a pass or failure.",
        "",
        "## Figure data",
        "",
        "Each cell is the number of units with a shows_persona and coherent positive, over all units of that kind. The random arm uses all 39 random directions. The values are also the points in fig118.png.",
        "",
        "| α | Question | Concept | Slot | Random directions |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for alpha in ALPHAS:
        for qi, question in enumerate(QUESTIONS):
            values = {}
            for kind in ("concept", "slot"):
                row = next(row for row in result["question_reach"]
                           if row["alpha"] == alpha and row["kind"] == kind
                           and row["question_index"] == qi)
                values[kind] = (row["showing"], row["total"])
            random_n, random_total = random_reach[(alpha, qi)]
            lines.append(
                f"| {alpha:g} | Q{qi} {question} | {ratio(*values['concept'])} | "
                f"{ratio(*values['slot'])} | {ratio(random_n, random_total)} |"
            )
    lines += [
        "",
        "## Remaining primary label errors",
        "",
        "The latest label row for each cell is used, matching score118.py.",
        "",
        "| Unit / arm / dose / question | Prompt | Source |",
        "| --- | --- | --- |",
    ]
    for key in error_keys:
        uid, arm, alpha, qi = key
        lines.append(f"| {uid} / {arm} / α={alpha:g} / q{qi} | {QUESTIONS[qi]} | labels.jsonl error row |")
    lines += [
        "",
        "## Excerpts read from the answer file",
        "",
        "I inspected 24 answer cells, six at each dose, across the neuron and random arms. These short excerpts retain their source cell and primary flags.",
        "",
        "| Source cell | Prompt | Reading flag | Raw answer excerpt |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(excerpt_rows(answer_map, primary_map))
    lines += [
        "",
        "## Scorer tables",
        "",
        "The report below is rendered from the same score_data result used above. Prediction text is copied from score118.py.",
        "",
    ]
    score_report = score118.render_report(result).replace("DS V4.1 Flash", "DeepSeek V4.1 Flash")
    lines.extend(score_report.splitlines()[1:])
    lines += [
        "",
        "## Cold-check audit",
        "",
        "The α=0.8 classification is in-sample because the units were selected and classified at this dose. Seven concept units have exactly three coherent persona marks; five more have two. Selection alone can make this column look strongest.",
        "",
        "The cold reader contested p92-11 q3 as coherent. If that one answer were marked incoherent, the α=0.8 concept classes would change from 7/4/7 to 6/5/7, the concept Q3 count from 9/18 to 8/18, and coherent persona-marked concept answers from 36/59 to 35/59. Current labels and tables retain Haiku's mark; no registered verdict changes.",
        "",
        "The unresolved p92-42 q0 error could be a coherent persona show. If resolved that way, Gate 1b would move from 34/39 to 35/39 and the slot row from 20/21 to 21/21. It remains an error and is conservatively excluded.",
        "",
        "Gate 1b compares Haiku labels from Part 92 with Haiku labels here, so it checks replication rather than reader independence. Four definite kind changes are p92-00, p92-08, p92-30 and p92-52; p92-42 is a fifth through its unresolved error. The threshold of 26/39 allows 13 kinds to change.",
        "",
        "On the 281-cell Gate 2 sample, Haiku marked 36 persona answers and DeepSeek V4.1 Flash marked 17. All 19 disagreements were Haiku yes and DeepSeek no. At α=1.6 in the neuron arm, the readers marked 14/43 and 4/43. The amended 4/21 slot and 10/18 concept counts are Haiku-only and likely high. The cold reader's after-the-fact bootstrap interval for pooled κ=0.609 was about 0.45 to 0.76 (2,000 cell resamples, seed 1).",
        "",
        "On the same sample, κ was 0.94 for coherence (8 disagreements), 0.85 for self-claim and 0.89 for speaker. Gate 2 uses only the persona mark. Speaker is not a registered score; it sometimes labels human on AI-style text, as at p92-48 random α=0.4 q3. The earlier Part 92b comparison had pooled κ=0.833; its neuron-arm κ=0.818 is the more relevant comparison.",
        "",
        "At α=0.4, the concept-neuron and random-direction question profiles overlap and do not separate. At α=1.6, all 234 neuron answers and 227/234 random answers were incoherent while persona marks persisted: 54/108 concept and 19/126 slot neuron answers were marked, with none coherent. At α=3.2 both arms were 234/234 incoherent.",
        "",
        "There were 20 empty answers: 3 at α=0.8 in the neuron arm, 8 at α=1.6 and 9 at α=3.2. They count as incoherent. The random-arm p92-49 q2 α=0.4 primary label is an error and counts as incoherent; DeepSeek marks it coherent, contributing to the primary 3/234 count.",
        "",
        "The primary Haiku usage sidecar has 2,035 attempts: 84 cap stops (43 at α=0.8 in the neuron arm), 79 HTTP 400 rows (all on p92-00 in the first 79 run rows) and 4 network errors. Each failure was retried. The four remaining error cells returned the cap on every attempt that produced output across three runs.",
        "",
        "The first DeepSeek read at 500 tokens remains unread and its κ was not computed. The final DeepSeek sample used 4,000 tokens; Haiku used 500 tokens.",
    ]
    (OUT / "NUMBERS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    units = read_units()
    answer_rows = score118.read_jsonl(ANSWERS_PATH)
    primary_rows = score118.read_jsonl(LABELS_PATH)
    ds_rows = score118.read_jsonl(DS_PATH)
    result = score118.score_data(answer_rows, primary_rows, ds_rows, units, ALPHAS)
    if not result["ready"]:
        raise RuntimeError("Part 118 is not ready for a write-up")
    result["alphas"] = list(ALPHAS)
    replay = replay_comparison(answer_rows)
    if replay["compared"] != 234:
        raise ValueError(f"expected 234 alpha-0.8 neuron cells, found {replay['compared']}")
    primary_map = score118._record_map(primary_rows, "primary label")
    random_reach = random_question_reach(units, primary_map)
    draw_figure(result["question_reach"], random_reach)
    write_numbers(result, replay, units, answer_rows, primary_rows, ds_rows)
    print(f"wrote {OUT / 'fig118.png'}")
    print(f"wrote {OUT / 'NUMBERS.md'}")


if __name__ == "__main__":
    main()
