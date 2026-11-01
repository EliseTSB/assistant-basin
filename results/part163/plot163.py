#!/usr/bin/env python3
"""Plot Part 163 arm rates and unit-bootstrap intervals from score.json."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
METRICS = (
    ("name_gone_identity", "Name gone\nidentity prompts"),
    ("maker_gone_maker", "Maker gone\nmaker prompts"),
    ("broken_rate", "Broken\nbattery answers"),
)
COLORS = {"neuron": "#1f77b4", "random": "#e07a24"}


def main():
    score = json.loads((HERE / "score.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)
    x = np.arange(len(METRICS), dtype=float)
    offsets = {"neuron": -0.09, "random": 0.09}
    for j, (key, label) in enumerate(METRICS):
        coords = []
        means = []
        for arm in ("neuron", "random"):
            item = score["rates"][key][arm]
            mean = item["mean"]
            lo, hi = item["interval_95"]
            xpos = x[j] + offsets[arm]
            ax.errorbar(xpos, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", ms=5.5,
                        capsize=3.5, color=COLORS[arm], ecolor=COLORS[arm], linewidth=1.5,
                        markeredgecolor="white", markeredgewidth=.6, zorder=3)
            coords.append((xpos, mean)); means.append(mean)
        ax.plot([coords[0][0], coords[1][0]], means, color="#777777", alpha=.48, linewidth=1, zorder=1)
    # The exploratory, answer-level Qwen measure uses the Part 161 exclusions.
    qwen = ((.0519, .1150),)
    j = 3
    coords = []
    for arm, mean in zip(("neuron", "random"), qwen[0]):
        xpos = j + offsets[arm]
        ax.scatter([xpos], [mean], color=COLORS[arm], s=34, edgecolor="white", linewidth=.6, zorder=3)
        coords.append((xpos, mean))
    ax.plot([coords[0][0], coords[1][0]], [qwen[0][0], qwen[0][1]], color="#777777", alpha=.48, linewidth=1, zorder=1)
    METRICS_LABELS = [label for _, label in METRICS] + ["Qwen present\nmaker answers"]
    ax.set_xticks([0, 1, 2, 3], METRICS_LABELS, fontsize=9)
    ax.set_ylim(-.02, 1.04)
    ax.set_ylabel("Answer or unit rate")
    ax.set_yticks(np.arange(0, 1.01, .2), [f"{v:.0%}" for v in np.arange(0, 1.01, .2)])
    ax.grid(axis="y", color="#dddddd", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter([], [], color=COLORS["neuron"], label="Neuron", s=30)
    ax.scatter([], [], color=COLORS["random"], label="Random direction", s=30)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Under one selection rule neurons say Qwen less and break more", fontsize=11, weight="bold")
    fig.savefig(HERE / "rates163.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
