#!/usr/bin/env python3
"""Render the Part 170 two-panel loop-rate figure from saved score and answer rows."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "part170"
OUT = DATA / "loops170.png"
BANS = ("b0", "b5", "b50", "b200")
DOSES = (0.25, 0.5, 1.0)
LAYERS = tuple(range(19, 27))
SLOTS = tuple(range(6))
WORD_RE = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def has_loop(answer: str) -> bool:
    words = [word.casefold() for word in WORD_RE.findall(answer if isinstance(answer, str) else "")]
    counts = Counter(tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2)))
    return any(count >= 3 for count in counts.values())


def bounds(values):
    return np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])


def main() -> None:
    score = read_json(DATA / "score.json")
    stage0 = read_json(DATA / "stage0.json")
    answers = read_jsonl(DATA / "answers.jsonl")
    frozen = set(map(str, stage0.get("frozen_prompt_ids", [])))
    per_cell = {(row["stage"], row["group"], float(row["dose"]), row["ban"]): row
                for row in score["per_cell"]}
    push_rows = [row for row in answers if row.get("stage") == "push" and row.get("arm") == "n+" and
                 not row.get("repeat") and row.get("group") == "p8"]
    push_values = {}
    for row in push_rows:
        key = (float(row["dose"]), str(row["prompt_id"]), int(row["layer"]), int(row["slot"]), str(row["ban"]))
        push_values[key] = float(has_loop(row.get("answer", "")))

    bootstrap = score["bootstrap"]["push"]
    n_reps = int(bootstrap["replicates"])
    rng = np.random.default_rng(int(bootstrap["seed"]))
    slot_draws = rng.integers(0, len(SLOTS), size=(n_reps, len(LAYERS), len(SLOTS)))
    slot_weights = (slot_draws[..., None] == np.arange(len(SLOTS))).sum(axis=2).astype(np.float64)
    push_bounds = {}
    for dose in DOSES:
        for ban in BANS:
            matrix = np.empty((len(LAYERS), len(SLOTS), 8), dtype=np.float64)
            for li, layer in enumerate(LAYERS):
                for si, slot in enumerate(SLOTS):
                    for pi in range(8):
                        prompt_id = f"p8:{pi}"
                        key = (dose, prompt_id, layer, slot, ban)
                        if key not in push_values:
                            raise ValueError(f"missing push cell {key}")
                        matrix[li, si, pi] = push_values[key]
            draws = np.einsum("rls,lsp->r", slot_weights, matrix) / (
                slot_weights.sum(axis=(1, 2)) * matrix.shape[2])
            push_bounds[(dose, ban)] = bounds(draws)

    # Paired dose-specific rises for panel two.
    rises = {}
    for dose in DOSES:
        for ban in ("b5", "b50", "b200"):
            matrix = np.empty((len(LAYERS), len(SLOTS), 8), dtype=np.float64)
            for li, layer in enumerate(LAYERS):
                for si, slot in enumerate(SLOTS):
                    for pi in range(8):
                        prompt_id = f"p8:{pi}"
                        base = push_values[(dose, prompt_id, layer, slot, "b0")]
                        alt = push_values[(dose, prompt_id, layer, slot, ban)]
                        matrix[li, si, pi] = alt - base
            point = float(matrix.mean())
            draws = np.einsum("rls,lsp->r", slot_weights, matrix) / (
                slot_weights.sum(axis=(1, 2)) * matrix.shape[2])
            rises[(dose, ban)] = (point, bounds(draws))

    prompt_boot = score["bootstrap"]["no_push"]
    n_prompt_reps = int(prompt_boot["replicates"])
    prompt_rng = np.random.default_rng(int(prompt_boot["seed"]))
    no_push = {}
    for group in ("m96", "q96"):
        for ban in BANS:
            cell = per_cell[("no_push", group, 0.0, ban)]
            eligible = [row for row in answers if row.get("stage") == "no_push" and row.get("group") == group and
                        row.get("ban") == ban and str(row.get("prompt_id")) not in frozen]
            if len(eligible) != int(cell["rows"]):
                raise ValueError(f"eligible row mismatch for {group} {ban}")
            y = np.asarray([has_loop(row.get("answer", "")) for row in eligible], dtype=np.float64)
            draws = prompt_rng.integers(0, len(y), size=(n_prompt_reps, len(y)))
            no_push[(group, ban)] = bounds(y[draws].mean(axis=1))

    # Colour choices follow a colour-blind-safe palette. Each panel's lines and its uncertainty use matching colours.
    dose_colors = {0.25: "#0072B2", 0.5: "#E69F00", 1.0: "#D55E00"}
    group_colors = {"m96": "#009E73", "q96": "#CC79A7"}
    rise_colors = {"b5": "#56B4E9", "b50": "#0072B2", "b200": "#4D4D4D"}
    x = np.arange(len(BANS))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.2, 9.2), constrained_layout=False)
    fig.suptitle("Under a push, blocking the plain answer raises loops; at 200 tokens it does so without one",
                 fontsize=17, fontweight="bold", y=0.985)
    ax1.set_title("Loop rate by ban", fontsize=13, pad=8)
    legend_handles = []
    legend_labels = []

    for dose in DOSES:
        points = np.asarray([per_cell[("push", "p8", dose, ban)]["loop_rate"] for ban in BANS], dtype=float)
        lower = np.asarray([push_bounds[(dose, ban)][0] for ban in BANS])
        upper = np.asarray([push_bounds[(dose, ban)][1] for ban in BANS])
        line = ax1.plot(x, points, marker="o", markersize=5, linewidth=2.0,
                        color=dose_colors[dose])[0]
        ax1.errorbar(x, points, yerr=np.vstack((points - lower, upper - points)), fmt="none",
                     ecolor=dose_colors[dose], elinewidth=1.1, capsize=2.5, zorder=2)
        legend_handles.append(line)
        legend_labels.append(f"P8 push, dose {dose:g}")

    for group, label in (("m96", "M96, no push"), ("q96", "Q96, no push")):
        points = np.asarray([per_cell[("no_push", group, 0.0, ban)]["loop_rate"] for ban in BANS], dtype=float)
        lower = np.asarray([no_push[(group, ban)][0] for ban in BANS])
        upper = np.asarray([no_push[(group, ban)][1] for ban in BANS])
        line = ax1.plot(x, points, marker="s", markersize=5, linewidth=2.0,
                        linestyle="--", color=group_colors[group])[0]
        ax1.errorbar(x, points, yerr=np.vstack((points - lower, upper - points)), fmt="none",
                     ecolor=group_colors[group], elinewidth=1.0, capsize=2.5, zorder=2)
        legend_handles.append(line)
        legend_labels.append(label)

    ax1.set_xticks(x, BANS)
    ax1.set_ylabel("Loop rate")
    ax1.set_ylim(-0.025, 0.62)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _position: f"{value:.0%}"))
    ax1.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.8)
    ax1.set_axisbelow(True)

    ax2.set_title("Paired rise over b0 by dose (P8 push rows)", fontsize=13, pad=8)
    dose_x = np.asarray(DOSES, dtype=float)
    for ban in ("b5", "b50", "b200"):
        points = np.asarray([rises[(dose, ban)][0] for dose in DOSES])
        lower = np.asarray([rises[(dose, ban)][1][0] for dose in DOSES])
        upper = np.asarray([rises[(dose, ban)][1][1] for dose in DOSES])
        line = ax2.plot(dose_x, points, marker="o", markersize=5, linewidth=2.0,
                        color=rise_colors[ban])[0]
        ax2.errorbar(dose_x, points, yerr=np.vstack((points - lower, upper - points)), fmt="none",
                     ecolor=rise_colors[ban], elinewidth=1.1, capsize=3, zorder=2)
        legend_handles.append(line)
        legend_labels.append(f"Rise, {ban}")

    ax2.axhline(0, color="#555555", linewidth=0.9)
    ax2.set_xticks(dose_x, [f"{dose:g}" for dose in DOSES])
    ax2.set_xlabel("Push dose")
    ax2.set_ylabel("Loop-rate difference")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _position: f"{value:+.0%}"))
    ax2.set_ylim(-0.04, 0.42)
    ax2.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.8)
    ax2.set_axisbelow(True)
    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.944),
               ncol=4, frameon=False, fontsize=9.5, columnspacing=1.8, handlelength=2.0)
    fig.text(0.02, 0.015,
             "95% intervals: P8 whole-slot bootstrap stratified by layer; no-push M96/Q96 prompt bootstrap.",
             fontsize=9, color="#444444")
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.1, right=0.97, hspace=0.36)
    fig.savefig(OUT, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
