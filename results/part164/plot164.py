#!/usr/bin/env python3
"""Plot Part 164 rates with percentile unit-bootstrap intervals."""
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {"neuron": "#1f77b4", "random": "#e07a24"}


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def interval(values, seed=164, reps=10000):
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(mean(draw))
    samples.sort()
    def pct(p):
        x = (len(samples) - 1) * p
        lo, hi = int(x), int(x) + (x % 1 > 0)
        return samples[lo] if lo == hi else samples[lo] * (hi-x) + samples[hi] * (x-lo)
    return pct(.025), pct(.975)


def main():
    score = json.loads((HERE / "score.json").read_text(encoding="utf-8"))
    rates = score["rates"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    random_x = np.array([.84, 1., 1.2])
    neuron_x = .76
    for ax, metric, title in ((axes[0], "qwen_rate", "Qwen rate"), (axes[1], "broken_rate", "Broken rate")):
        random_means, random_lows, random_highs = [], [], []
        for dose in ("0.8", "1.0", "1.2"):
            units = rates["per_unit_qwen" if metric == "qwen_rate" else "per_unit_broken"][dose]["random"]
            vals = [x for x in units.values() if x is not None]
            random_means.append(mean(vals))
            lo, hi = interval(vals)
            random_lows.append(mean(vals)-lo); random_highs.append(hi-mean(vals))
        ax.errorbar(random_x, random_means, yerr=[random_lows, random_highs], fmt="o-", color=COLORS["random"],
                    ecolor=COLORS["random"], linewidth=1.6, capsize=4, markersize=6,
                    markeredgecolor="white", markeredgewidth=.7, label="Random direction")
        neuron_vals = [x for x in rates["per_unit_qwen" if metric == "qwen_rate" else "per_unit_broken"]["0.8"]["neuron"].values() if x is not None]
        nm = mean(neuron_vals); lo, hi = interval(neuron_vals)
        ax.errorbar([neuron_x], [nm], yerr=[[nm-lo], [hi-nm]], fmt="o", color=COLORS["neuron"],
                    ecolor=COLORS["neuron"], capsize=4, markersize=6, markeredgecolor="white",
                    markeredgewidth=.7, label="Neuron +0.8")
        if metric == "qwen_rate":
            ax.annotate(f"Neuron {nm:.1%}", (neuron_x, nm), xytext=(9, 8), textcoords="offset points",
                        fontsize=8, color=COLORS["neuron"])
        ax.set_title(title)
        ax.set_xlabel("Dose")
        ax.set_xticks([.8, 1., 1.2], ["+0.8", "+1.0", "+1.2"])
        ax.set_ylabel("Mean unit rate")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.grid(axis="y", color="#dddddd", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    q = rates["qwen_rate"]["0.8"]
    pred = score["predictions"]["164.1"]
    axes[1].set_title(f"Broken: neuron {rates['broken_rate']['0.8']['neuron']['mean']:.1%}, "
                      f"random {rates['broken_rate']['0.8']['random']['mean']:.1%}")
    paired_random = q['neuron']['mean'] / pred['estimate_ratio']
    fig.suptitle(f"Fresh neurons at +0.8: Qwen {q['neuron']['mean']:.1%} against paired random {paired_random:.1%}, "
                 f"ratio {pred['estimate_ratio']:.2f} (95% interval "
                 f"{pred['ratio_interval_95'][0]:.2f} to {pred['ratio_interval_95'][1]:.2f})",
                 fontsize=11, weight="bold")
    fig.savefig(HERE / "rates164.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
