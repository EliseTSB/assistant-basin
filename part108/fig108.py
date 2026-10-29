#!/usr/bin/env python3
"""Part 108 figure: share of the slot units shown at Instruct that each move loses, at the RL and post-training
scales, with 95% Wilson intervals. Reads results/part108/score108.json, writes results/part108/fig108.png.
Written by Claude (Opus 5.5), 2026-10-29. Usage: python part108/fig108.py
"""
import json
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    score = json.loads((ROOT / "results/part108/score108.json").read_text(encoding="utf-8"))
    models, base = score["models"], score["B"]
    panels = (("RL stage (DPO to Instruct)", (("D", "go back:\nDPO"), ("FR", "go forward:\n2 x Instruct - DPO"),
                                                ("XR", "scrambled\nsigns"))),
              ("DPO and RL together (SFT to Instruct)", (("S", "go back:\nSFT"), ("FP", "go forward:\n2 x Instruct - SFT"),
                                                          ("XP", "scrambled\nsigns"))))
    colours = ("#9a9a9a", "#d35f0e", "#f2b37f")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (title, bars) in zip(axes, panels):
        for i, ((tag, label), colour) in enumerate(zip(bars, colours)):
            lost = len(models[tag]["lost"])
            n = base - len(models[tag]["unlabelled"])
            lo, hi = wilson(lost, n)
            share = lost / n if n else 0.0
            ax.bar(i, 100 * share, color=colour, width=0.62)
            ax.errorbar(i, 100 * share, yerr=[[100 * (share - lo)], [100 * (hi - share)]], color="#333333", capsize=5)
            gate = models[tag].get("gate", True)
            note = f"{lost} of {n}" + ("" if gate else "\n(failed the gate)")
            ax.text(i, 100 * hi + 2, note, ha="center", va="bottom", fontsize=10)
            ax.set_xticks(range(len(bars)))
            ax.set_xticklabels([b[1] for b in bars], fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#eeeeee")
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"slot units that stop showing their persona (%)\nof the {base} shown at Instruct")
    fig.suptitle("Does undoing a training stage lose more personas than going the same distance further?", fontsize=13)
    fig.tight_layout()
    out = ROOT / "results/part108/fig108.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
