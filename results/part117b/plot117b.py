#!/usr/bin/env python3
"""Recompute Part 117b's plotted counts from its answer and label files."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/part117b"
WHO = 0


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def count(labels, ids, field):
    values = [labels.get((label_id, WHO)) for label_id in ids]
    return {
        "yes": sum(item is not None and item[field] is True for item in values),
        "missing": sum(item is None for item in values),
        "total": len(values),
        "read": sum(item is not None for item in values),
    }


def main():
    pairs = json.loads((ROOT / "results/part117/pairs.json").read_text(encoding="utf-8"))
    answers = read_jsonl(RES / "answers.jsonl")
    items = read_jsonl(RES / "items.jsonl")
    labels = {}
    label_lines = read_jsonl(RES / "labels.jsonl")
    for item in label_lines:
        if "error" not in item:
            labels[(item["id"], int(item["question_index"]))] = item
    answer_map = {(item["row"], int(item["question_index"])): item for item in answers}

    def pair_ids(cond, role="s"):
        return [f'{p["pair"]}:{cond}|{role}' for p in pairs]

    random_ids = [f'{p["pair"]}:r{k}-1,s+1|s' for p in pairs for k in range(3)]
    control_ids = [f'{p["pair"]}:n{k}-1,s+1|s' for p in pairs for k in range(3)]
    slot_ids = {
        "s+1": pair_ids("s+1"),
        "s+0.85": pair_ids("s+0.85"),
        "s+0.5": pair_ids("s+0.5"),
        "c-1,s+1": pair_ids("c-1,s+1"),
        "u-1,s+1": pair_ids("u-1,s+1"),
        "random": random_ids,
        "control": control_ids,
        "c-0.5,s+1": pair_ids("c-0.5,s+1"),
    }
    slot_shows = {condition: count(labels, ids, "shows_persona") for condition, ids in slot_ids.items()}
    coherent = {condition: count(labels, ids, "coherent") for condition, ids in slot_ids.items()}

    units = {}
    for pair in pairs:
        for key in ("slot", "match", "unrelated"):
            units.setdefault(pair[key]["id"], pair[key])
    bad_units = []
    for unit_id in units:
        values = [labels.get((f"none|{unit_id}", q)) for q in range(6)]
        yes = sum(value is not None and value["shows_persona"] is True for value in values)
        missing = sum(value is None for value in values)
        if yes + missing > 2:
            bad_units.append(unit_id)

    alone_rows = [f'alone:{unit_id}:c-1' for unit_id in units if any(
        item["row"] == f'alone:{unit_id}:c-1' and int(item["question_index"]) == WHO for item in answers)]
    qwen_alone = sum("qwen" in answer_map[(row, WHO)]["answer"].lower() for row in alone_rows)
    qwen_by_condition = {}
    for condition in ("s+1", "s+0.85", "s+0.5", "c-1,s+1", "u-1,s+1"):
        rows = [row_id.split("|", 1)[0] for row_id in slot_ids[condition]]
        qwen_by_condition[condition] = sum(
            "qwen" in answer_map[(row, WHO)]["answer"].lower() for row in rows
        )
    qwen_by_condition["random"] = sum(
        "qwen" in answer_map[(row.split("|", 1)[0], WHO)]["answer"].lower() for row in random_ids
    )
    qwen_by_condition["control"] = sum(
        "qwen" in answer_map[(row.split("|", 1)[0], WHO)]["answer"].lower() for row in control_ids
    )
    qwen_by_condition["alone_c-1"] = qwen_alone

    expected = {(item["id"], int(item["question_index"])) for item in items}
    missing_labels = sorted(expected - labels.keys())
    gate = {
        "base_rate_units": len(units),
        "base_rate_failures": len(bad_units),
        "slot_s+1": [slot_shows["s+1"]["yes"], slot_shows["s+1"]["missing"], slot_shows["s+1"]["total"]],
        "none_contains_qwen": "qwen" in answer_map[("none", WHO)]["answer"].lower(),
        "zero_hook_equal": all(item["equal"] for item in json.loads((RES / "zero-hook-check.json").read_text(encoding="utf-8"))),
    }
    agreement = json.loads((RES / "agreement.json").read_text(encoding="utf-8"))

    categories = ["s+1", "s+0.85", "s+0.5", "c-1,s+1", "u-1,s+1", "random", "control"]
    display = ["Slot +1", "Slot +0.85", "Slot +0.5", "Matched −1\n+ slot +1", "Unrelated −1\n+ slot +1",
               "Random −1\n+ slot +1", "Non-persona −1\n+ slot +1"]
    colors = ["#3973ac", "#5b8e3e", "#8c6bb1", "#3973ac", "#d17b32", "#555555", "#9c4a4a"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.3), sharey=True)
    for ax, field, title in zip(axes, ("shows_persona", "coherent"),
                                ("Slot persona shown", "Answer called coherent")):
        values = slot_shows if field == "shows_persona" else coherent
        fracs, labels_text = [], []
        for condition in categories:
            result = values[condition]
            fracs.append(result["yes"] / result["read"] if result["read"] else 0)
            labels_text.append(f'{result["yes"]}/{result["read"]}')
        bars = ax.barh(range(len(categories)), fracs, color=colors, height=0.66)
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_xlim(0, 1.28)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xticklabels(["0", ".25", ".5", ".75", "1.0"])
        ax.grid(axis="x", color="#d9d9d9", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        for bar, label in zip(bars, labels_text):
            ax.text(bar.get_width() + .025, bar.get_y() + bar.get_height() / 2, label,
                    va="center", ha="left", fontsize=9)
    axes[0].set_yticks(range(len(categories)), display)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Share of readable labels")
    axes[1].set_xlabel("Share of readable labels")
    fig.suptitle('“Who are you?” answers by condition', x=.04, ha="left", fontsize=14, y=.99)
    control_missing = slot_shows["control"]["missing"]
    fig.text(.04, .015,
             f"Numbers at bar ends are yes / labels read; the non-persona condition has {control_missing} missing label.",
             fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, .055, 1, .94), w_pad=2.2)
    fig.savefig(RES / "fig-117b-who.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "answers": len(answers),
        "items_expected": len(items),
        "labels_missing": len(missing_labels),
        "missing_label_ids": [f"{item_id} q{q}" for item_id, q in missing_labels],
        "gate": gate,
        "slot_persona_who": slot_shows,
        "coherent_who": coherent,
        "qwen_string_hits_who": qwen_by_condition,
        "uninterpretable_units": len(bad_units),
        "second_reader_agreement": {
            "agree": agreement["agree"], "n": agreement["n"],
            "percent": round(100 * agreement["agree"] / agreement["n"], 1),
        },
        "figure": "results/part117b/fig-117b-who.png",
    }
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
