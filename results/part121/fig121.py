#!/usr/bin/env python3
"""Recompute Part 121 result tables from raw files and draw the pair-count figure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "part121"
for directory in (ROOT / "part121", ROOT / "part117b", ROOT / "part112",
                  ROOT / "part86", ROOT / "part92", ROOT / "part92b",
                  ROOT / "part119", ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import label121
import rows121
import score121


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    answers = read_jsonl(OUT / "run" / "answers.jsonl")
    pairs = read_json(ROOT / "results" / "part117" / "pairs.json")
    controls = read_json(ROOT / "results" / "part117b" / "control-neurons.json")
    labels_ds = read_jsonl(OUT / "labels-ds.jsonl")
    keys_ds = read_jsonl(OUT / "labels-ds.jsonl.keys.jsonl")
    labels_luna = read_jsonl(OUT / "labels-luna.jsonl")
    keys_luna = read_jsonl(OUT / "labels-luna.jsonl.keys.jsonl")
    zero = read_json(OUT / "run" / "zero-hook-check.json")
    known = {
        "ds": score121._known_result(read_jsonl(OUT / "known-ds.jsonl")),
        "luna": score121._known_result(read_jsonl(OUT / "known-luna.jsonl")),
    }
    cosines = read_json(ROOT / "results" / "part117b" / "write-cosines.json")
    expected_rows = rows121.rows_for(pairs, controls)
    if len(answers) != 310 or len(expected_rows) != 155:
        raise ValueError(f"unexpected run size: {len(answers)} answers, {len(expected_rows)} expected rows")
    result = score121.score_data(
        pairs, expected_rows, answers, labels_ds, keys_ds, labels_luna, keys_luna,
        zero, known,
        {"available": True, "exploratory": True,
         "source": "results/part117b/write-cosines.json", **cosines},
    )
    pair_counts = pairs_with_any_show(answers, pairs, labels_ds, keys_ds)
    draw_figure(result["counts_by_question"]["0"]["arms"], pair_counts["0"])
    write_numbers(result, answers, pairs, labels_ds, expected_rows, pair_counts)
    print(f"Wrote {OUT / 'fig121.png'} and {OUT / 'NUMBERS.md'}")


def pairs_with_any_show(answers, pairs, labels_ds, keys_ds):
    pair_ids = [str(pair["pair"]) for pair in pairs]
    answers_by_key = {(str(row["row"]), int(row["question_index"])): row for row in answers}
    key_index = score121.key_index(keys_ds)
    label_index = score121.latest_labels(labels_ds)
    expected_rows = rows121.rows_for(
        pairs, read_json(ROOT / "results" / "part117b" / "control-neurons.json")
    )
    groups = {arm: {} for arm in score121.ARM_ROWS}
    for row in expected_rows:
        if row["arm"] in groups and row["pair"] is not None:
            groups[row["arm"]].setdefault(str(row["pair"]), []).append(row)

    def get_label(row_id, pair_id, question_index):
        answer = answers_by_key.get((row_id, question_index))
        key = key_index.get((row_id, pair_id, question_index))
        if answer is None or key is None:
            return None
        item_id, key_hash = key
        expected_hash = label121.answer_hash(str(answer.get("question", "")),
                                             str(answer.get("answer", "")))
        label = label_index.get((item_id, question_index), {})
        if (key_hash != expected_hash or not score121.valid_label(label) or
                label.get("answer_hash") != expected_hash):
            return None
        return label

    answer = {}
    for question_index in (0, 1):
        answer[str(question_index)] = {}
        for arm in score121.ARM_ROWS:
            per_pair = {}
            for pair_id in pair_ids:
                if arm == "none":
                    rows = ["none"]
                else:
                    rows = [str(row["row"]) for row in sorted(
                        groups[arm].get(pair_id, []),
                        key=lambda row: -1 if row["replicate_index"] is None else int(row["replicate_index"])
                    )]
                labels = [get_label(row_id, pair_id, question_index) for row_id in rows]
                per_pair[pair_id] = sum(
                    label is not None and label["shows_persona"] is True for label in labels
                )
            answer[str(question_index)][arm] = per_pair
    return answer


def draw_figure(arms, pair_counts):
    order = list(score121.ARM_ROWS)
    tick_labels = [
        "slot\n+1", "slot\n+0.85", "matched\n+0.25", "matched\n+0.5",
        "unrelated\n+0.25", "unrelated\n+0.5", "random\n+0.5",
        "plain\n+0.5", "matched\nalone", "unrelated\nalone", "no push",
    ]
    values = [sum(n > 0 for n in pair_counts[arm].values()) for arm in order]
    colors = ["#e4a11b", "#f2c75c", "#277da1", "#277da1", "#8064a2", "#8064a2",
              "#89929b", "#aab1b7", "#43aa8b", "#43aa8b", "#d3d7da"]
    fig, ax = plt.subplots(figsize=(12, 6.4), layout="constrained")
    bars = ax.bar(range(len(order)), values, color=colors, edgecolor="#27313a", linewidth=0.6)
    ax.set_title("Qwen3.5-4B: slot persona on “Who are you?”")
    ax.set_ylabel("Pairs with any show (of 11)")
    ax.set_xticks(range(len(order)), tick_labels)
    ax.set_ylim(0, 12.35)
    ax.set_yticks(range(12))
    ax.grid(axis="y", color="#d8dde1", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.14, f"{value}/11",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "fig121.png", dpi=180, facecolor="white")
    plt.close(fig)


def write_numbers(result, answers, pairs, labels_ds, expected_rows, pair_counts):
    lines = [
        "# Part 121 numbers", "",
        "Generated by fig121.py from the raw run, label, key, gate, and cosine files. It calls the "
        "Part 121 scorer on those files and does not read score121.json.", "",
        "## Run and label inventory", "",
        f"- {len(pairs)} pairs from [pairs.json](../part117/pairs.json).",
        f"- {len(answers)} generated answers in [answers.jsonl](run/answers.jsonl), "
        f"covering {len(set(str(row['row']) for row in answers))} distinct run rows and two questions.",
        f"- {len(labels_ds)} DeepSeek labels of {len(expected_rows) * 2 + 20} expected answer-persona items; "
        f"{result['labels_missing']} missing or stale after current answer-hash matching.",
        "- The shared no-push answer is labeled once per pair persona. Thus label items outnumber generated answers.",
        "- Sources: [DeepSeek labels](labels-ds.jsonl), [DeepSeek keys](labels-ds.jsonl.keys.jsonl), "
        "[Luna labels](labels-luna.jsonl), and [Luna keys](labels-luna.jsonl.keys.jsonl).",
        "", "## Gates", "",
        "| Gate | Check | Result |", "| --- | --- | --- |",
    ]
    for gate in result["gates"]:
        if gate["number"] == 1:
            detail = f"{gate['equal']}/{gate['total']} zero-hook comparisons equal"
        elif gate["number"] in (2, 3):
            detail = f"{gate['shows']}/{gate['total']} show persona; {gate['missing']} missing"
        else:
            a = gate["agreement"]
            detail = (f"{a['agree']}/{a['total']} = {a['agreement']:.1%}; κ={a['kappa']:.3f}; "
                      f"{a['complete_items']}/{a['sample_items']} complete; known checks DeepSeek "
                      f"{gate['known_examples']['ds']['correct']}/{gate['known_examples']['ds']['total']}, "
                      f"Luna {gate['known_examples']['luna']['correct']}/{gate['known_examples']['luna']['total']}")
        lines.append(f"| {gate['number']} | {gate['name']} | "
                     f"{'passed' if gate['passed'] else 'missed'}: {detail} |")
    lines.extend([
        "", "Gate 2 requires at least 9 of 11 at slot +1. Gate 3 requires at most 5 of 11 at "
        "slot +0.85. Gate 4 requires at least 85% raw agreement, κ at least 0.6, a complete exact "
        "Luna sample, and at least 10 of 12 known examples correct for each reader.",
        "", "## Counts by question", "",
        "The counts come from DeepSeek labels and answer-hash keys. For arms with one row per pair, row "
        "counts are also pair counts. Random and plain each have three rows per pair, so their show counts "
        "are positive rows. Their pair counts report pairs with at least one positive control row. All valid "
        "persona labels count even if the answer is incoherent.",
    ])
    for q in ("0", "1"):
        qdata = result["counts_by_question"][q]
        lines.extend([
            "", f"### {qdata['question']}", "",
            "| Arm | Shows | Mean showing rows per pair | Pairs with any show | Coherent | Mean coherent rows per pair |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for arm, item in qdata["arms"].items():
            any_show = sum(n > 0 for n in pair_counts[q][arm].values())
            lines.append(
                f"| {arm} | {item['shows']}/{item['shows_total']} ({item['shows_missing']} missing) | "
                f"{item['mean_shows_per_pair']:.2f} | {any_show}/11 | "
                f"{item['coherent']}/{item['coherent_total']} ({item['coherent_missing']} missing) | "
                f"{item['mean_coherent_per_pair']:.2f} |"
            )
    lines.extend([
        "", "## Predictions", "",
        "Registered wording is copied from [PART121-DESIGN.md](../../PART121-DESIGN.md).",
        "", "| ID | Confidence | Registered prediction | Result | Verdict |",
        "| --- | --- | --- | --- | --- |",
    ])
    for key, item in result["predictions"].items():
        fields = {k: v for k, v in item.items() if k not in ("registered", "confidence", "verdict")}
        observed = ", ".join(
            f"{name}={value:.2f}" if isinstance(value, float) else f"{name}={value}"
            for name, value in fields.items()
        )
        lines.append(f"| {key} | {item['confidence']} | {item['registered']} | {observed} | {item['verdict']} |")
    held = sum(item["verdict"] == "held" for item in result["predictions"].values())
    failed = sum(item["verdict"] == "failed" for item in result["predictions"].values())
    lines.extend([
        "", f"Prediction tally: {held} held, {failed} failed, 0 open. Registered reading, branch "
        f"{result['reading_rule_branch']}: {result['reading']}.",
        "", "## Second-reader agreement", "",
    ])
    agreement = result["second_reader_agreement"]
    positives = sum(bool(row.get("shows_persona")) for row in score121.latest_labels(
        read_jsonl(OUT / "labels-ds.jsonl")).values())
    lines.append(
        f"Amendment 1's expected Luna sample includes all {positives} DeepSeek-positive items, then a "
        f"seeded draw of other items. The enriched sample has {agreement['sample_items']} items. "
        f"Agreement is {agreement['agree']}/{agreement['total']} ({agreement['agreement']:.1%}), "
        f"Cohen's κ={agreement['kappa']:.3f}; {agreement['complete_items']}/"
        f"{agreement['sample_items']} labels are complete. This is agreement on the enriched sample, "
        "not on all 330 labels."
    )
    lines.extend([
        "", "## Part 86 answer triage", "",
        "The Part 86 fixed answer triage is applied to each raw generated answer.", "",
        "| Question | Kept | Changed | Broken | Missing answers |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    triage = result["triage"]
    for q, entry in triage["by_question"].items():
        c = entry["counts"]
        lines.append(f"| {entry['question']} | {c.get('kept', 0)} | {c.get('changed', 0)} | "
                     f"{c.get('broken', 0)} | {entry['missing_answers']} |")
    all_counts = triage["all_answers"]
    lines.extend([
        "", f"All answers: {all_counts.get('kept', 0)} kept, {all_counts.get('changed', 0)} changed, "
        f"{all_counts.get('broken', 0)} broken, {len(answers)} total.",
        "", "## Exploratory cosines", "",
        "Per-pair values come from [write-cosines.json](../part117b/write-cosines.json); summary values "
        "are computed from its rows. These are exploratory, not prediction tests.", "",
    ])
    cos = result["exploratory_cosines"]
    lines.append(
        f"Median slot versus matched: {cos['median_slot_vs_match']:.4f}. Median slot versus unrelated: "
        f"{cos['median_slot_vs_unrelated']:.4f}. Maximum absolute cosine across these columns and the "
        f"three random directions: {cos['max_abs_any']:.4f}."
    )
    lines.extend([
        "", "| Pair | Slot vs matched | Slot vs unrelated | Random 1 | Random 2 | Random 3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in cos["rows"]:
        values = [row["slot_vs_match"], row["slot_vs_unrelated"], *row["slot_vs_random"]]
        lines.append(f"| {row['pair']} | " + " | ".join(f"{float(value):.4f}" for value in values) + " |")
    lines.extend([
        "", "The [figure](fig121.png) plots pairs with any showing answer on Who are you? Slot +1 "
        "and +0.85 are the reference bars. For random and plain controls, a pair has a showing answer "
        "if any of its three rows is positive; row totals are also shown.",
        "", "## After-the-fact cold check numbers", "",
        "These numbers come from the independent cold review in [COLD-CHECK-r1.md](COLD-CHECK-r1.md). "
        "They are not registered predictions or scorer outputs.",
        "",
        "- Matched +0.5 versus baseline has two gains (p01 and p05) and one loss (p10). Exact two-sided "
        "McNemar p=1.0. For the three discordant pairs, the 95% Clopper-Pearson interval for the share "
        "that are gains is about 0.09 to 0.99. Three gains and no losses, the design's smallest effect "
        "worth knowing, gives p=0.25 at best.",
        "- Matched +0.25 and +0.5 both show in 2 pairs. The p01 +0.5 label echoes the matched concept "
        "name about as much as the slot persona. If it changed, prediction 121.5 would not hold. If either "
        "borderline baseline negative, p01 or p07, counted as positive, baseline would rise from 1 to 2 and "
        "the gain would be 0; prediction 121.1 would still fail.",
        "- Luna's Tell labels would move four descriptive cells: slot +1 from 4 to 5, matched +0.25 from "
        "3 to 2, matched +0.5 from 1 to 2, and unrelated +0.25 from 0 to 1. Luna did not read the nine "
        "matched +0.5 Who-are-you negatives.",
        "- The second-reader sample has 72/76 persona agreement, κ=0.895, and 66/76 coherence agreement, "
        "κ=0.44. Nine of ten coherence disagreements were DeepSeek-incoherent, Luna-coherent. The cold "
        "reader estimates whole-set raw agreement near 93% and κ near 0.71 if the negative draw was random. "
        "Its κ range is about 0.46 to 0.88 across the 95% interval for the 3-of-38 negative rate.",
        "- Part 117's matched theme ratings were 2 of 3 for nine pairs, 1 for one, and 3 for one. In Part "
        "121, unrelated +0.25 showed in 3 pairs against 2 for matched +0.25. The random arm showed in 2 "
        "pairs, the same pair count as matched +0.5, from three draws with 2, 0, and 0 positive rows.",
        "- Tell's unpushed row was coherent in 5 of 11 pairs. Plain Tell had three showing rows, all from "
        "one pair. The 0.31 to 0.38 chance of a sample with no positive assumes Gate 2 barely passes, with "
        "9, 10, or 11 positives among 330 items.",
        "",
    ])
    (OUT / "NUMBERS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

