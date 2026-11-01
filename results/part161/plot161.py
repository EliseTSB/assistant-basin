#!/usr/bin/env python3
"""Draw the two Part 161 picked-unit dose-ladder heatmaps."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DOSES = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]


def main():
    score = json.loads((HERE / "score.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 14), gridspec_kw={"width_ratios": [35, 45]}, constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_under("#f5f5f5")
    image = None
    for ax, arm in zip(axes, ("neuron", "random")):
        unique = {}
        for pick in ("0.6", "1.0"):
            for u in score["summary"]["sets"][pick][arm]:
                unique.setdefault(u["unit_id"], u)
        units = sorted(unique.values(), key=lambda u: (u["peak_dose"], -max(u["hits"].values()), u["unit_id"]))
        matrix = np.asarray([[u["hits"][str(d)] for d in DOSES] for u in units])
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=24)
        ax.set_xticks(range(len(DOSES)), [f"{d:.1f}" for d in DOSES])
        ax.set_yticks(range(len(units)), [u["unit_id"] for u in units], fontsize=7.2)
        ax.set_xlabel("Positive dose (triangle marks picking dose)")
        ax.set_title(f"{'Neurons' if arm == 'neuron' else 'Random directions'} · {len(units)} units", pad=8)
        ax.set_ylabel("Picked unit, sorted by peak dose then hits")
        ax.set_xticks(np.arange(-.5, len(DOSES), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(units), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        # One triangle marks each membership's picking dose. L11N5647 was selected at both doses.
        for row, u in enumerate(units):
            picks = {float(d) for d in ("0.6", "1.0") if any(x["unit_id"] == u["unit_id"] for x in score["summary"]["sets"][d][arm])}
            for dose in picks:
                col = DOSES.index(dose)
                ax.scatter(col, row - 0.33, marker="v", s=17, color="#39d6d0", edgecolors="#111111", linewidths=.35, zorder=5, clip_on=False)
        ax.set_ylim(len(units) - .5, -.5)
    fig.suptitle("Picked random directions were reliable as often as picked neurons", fontsize=15, y=1.01)
    cbar = fig.colorbar(image, ax=axes, shrink=.72, pad=.02)
    cbar.set_label("Company hits (of 24)")
    fig.savefig(HERE / "ladder161.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
