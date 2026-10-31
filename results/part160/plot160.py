#!/usr/bin/env python3
"""Plot the Part 160 dose ladders from score.json."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SCORE = HERE / "score.json"
OUT = HERE / "ladder160.png"
DOSES = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]


def main():
    data = json.loads(SCORE.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    colors = plt.get_cmap("tab20")
    for ax, picking_dose in zip(axes, (0.6, 1.0)):
        units = data["summary"]["sets"][str(picking_dose)]
        peaks = [u["peak_dose"] for u in units if u["peak_dose"] is not None]
        median = sorted(peaks)[len(peaks) // 2] if len(peaks) % 2 else sum(sorted(peaks)[len(peaks)//2-1:len(peaks)//2+1]) / 2
        endpoints = []
        for i, unit in enumerate(units):
            hits = [unit["hits"][str(d)] for d in DOSES]
            ax.plot(DOSES, hits, color=colors(i % 20), linewidth=1.1, alpha=0.75, marker="o", markersize=3)
            endpoints.append((hits[-1], unit["unit_id"], colors(i % 20)))
        # Spread endpoint labels while keeping each label linked to its curve.
        endpoints.sort(key=lambda item: item[0])
        label_y = [item[0] for item in endpoints]
        for i in range(1, len(label_y)):
            label_y[i] = max(label_y[i], label_y[i - 1] + 1.35)
        overflow = max(0, label_y[-1] - 24)
        label_y = [y - overflow for y in label_y]
        underflow = min(0, label_y[0])
        label_y = [y - underflow for y in label_y]
        for (end_y, uid, color), text_y in zip(endpoints, label_y):
            ax.plot([1.4, 1.425], [end_y, text_y], color=color, linewidth=0.7, alpha=0.7, clip_on=False)
            ax.text(1.435, text_y, uid, fontsize=10, va="center", clip_on=False)
        ax.axvline(median, color="#111111", linestyle="--", linewidth=1.5)
        ax.axvline(picking_dose, color="#d62728", linestyle=":", linewidth=2)
        if median == picking_dose:
            ax.text(median, 25, f"median + pick {median:g}", rotation=90, ha="left", va="top", fontsize=10)
        else:
            ax.text(median, 25, f"median peak {median:g}", rotation=90, ha="right", va="top", fontsize=10)
            ax.text(picking_dose, 25, f"pick {picking_dose:g}", rotation=90, color="#b22222", ha="left", va="top", fontsize=10)
        ax.set_title(f"Picked at {picking_dose:g} · {len(units)} units")
        ax.set_xlabel("Positive dose")
        ax.set_xlim(0.35, 1.72)
        ax.set_ylim(-0.5, 26)
        ax.set_xticks(DOSES)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Company hits (of 24)")
    fig.suptitle("Part 160: company hits across dose ladders", fontsize=14)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
