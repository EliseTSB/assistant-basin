#!/usr/bin/env python3
"""Plot Part 162 positions and the descriptive-only Part 162b positions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
SCORE = HERE / "score.json"
PART162B = HERE.parent / "part162b" / "score.json"
OUT = HERE / "positions162.png"

LABELS = {
    "claude": "Claude",
    "gpt": "GPT",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "olmo": "OLMo",
    "pythia": "Pythia",
    "mix_claude_then_gpt": "Mix C then G",
    "mix_gpt_then_claude": "Mix G then C",
}
OFFSETS = {
    "claude": (5, 8), "gpt": (5, 7), "deepseek": (7, 4),
    "qwen": (7, 4), "olmo": (7, -9), "pythia": (7, -12),
    "mix_claude_then_gpt": (7, 9), "mix_gpt_then_claude": (7, -14),
}


def main() -> None:
    score = json.loads(SCORE.read_text(encoding="utf-8"))
    part162b = json.loads(PART162B.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 11, "axes.labelsize": 9})
    fig, ax = plt.subplots(figsize=(7.2, 6.5), constrained_layout=True)
    ax.add_patch(Rectangle((0.25, 0.25), 1.05, 1.05, fill=False, edgecolor="#555555",
                           linewidth=1.1, linestyle=(0, (4, 3)), zorder=0))
    ax.axvline(0.25, color="#777777", linestyle=(0, (3, 3)), linewidth=0.7, zorder=0)
    ax.axhline(0.25, color="#777777", linestyle=(0, (3, 3)), linewidth=0.7, zorder=0)
    colors = {"claude": "#6b4c9a", "gpt": "#1982a5", "deepseek": "#d95f02",
              "qwen": "#4daf4a", "olmo": "#984ea3", "pythia": "#777777",
              "mix_claude_then_gpt": "#222222", "mix_gpt_then_claude": "#222222"}
    for model in LABELS:
        axes = score["positions"][model]
        x = axes["claude"]["mean"]
        y = axes["gpt"]["mean"]
        xerr = [[x - axes["claude"]["percentile_2_5"]], [axes["claude"]["percentile_97_5"] - x]]
        yerr = [[y - axes["gpt"]["percentile_2_5"]], [axes["gpt"]["percentile_97_5"] - y]]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", markersize=5,
                    color=colors[model], ecolor=colors[model], elinewidth=0.65,
                    capsize=1.7, capthick=0.65, alpha=0.92, zorder=3)
        dx, dy = OFFSETS[model]
        ax.annotate(LABELS[model], (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.2, color=colors[model], weight="medium")

    # Exploratory word-only marker set, shown separately from registered positions.
    ax.plot(0.908, -0.152, marker="o", markersize=5, markerfacecolor="none",
            markeredgecolor=colors["deepseek"], markeredgewidth=1.1, linestyle="none", zorder=5)
    ax.annotate("DeepSeek word-only, exploratory", (0.908, -0.152), xytext=(7, -9),
                textcoords="offset points", fontsize=6.8, color=colors["deepseek"])

    # Part 162b positions are shown only as a robustness description because both arms fail gates.
    for arm, marker, offset in (("armA", "o", (-78, 13)), ("armB", "s", (-78, -16))):
        pos = part162b["positions"][arm]["deepseek"]
        x, y = pos["claude"], pos["gpt"]
        ax.plot(x, y, marker=marker, markersize=6, markerfacecolor="none",
                markeredgecolor="#d95f02", markeredgewidth=1.3, linestyle="none", zorder=4)
        ax.annotate(f"Arm {arm[-1]} descriptive", (x, y), xytext=offset,
                    textcoords="offset points", fontsize=6.8, color="#9a4300")

    ax.text(0.27, 1.25, "both threshold", fontsize=7, color="#555555", va="top")
    ax.set_xlim(-1.30, 1.38)
    ax.set_ylim(-1.48, 1.38)
    ax.set_xlabel("P_C on Claude markers")
    ax.set_ylabel("P_G on GPT markers")
    ax.set_title("Part 162: V4.1 Flash reads with Claude; GPT side rests on punctuation", pad=10)
    ax.grid(alpha=0.16, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT, dpi=190, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
