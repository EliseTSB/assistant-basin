"""Part 203's figure, drawn from results/part203/results.json only.

Left: each readout's mean six-way balanced accuracy over the five held-out actor folds, with its 95% actor-cluster
bootstrap interval, against chance (1/6). Right: the three-actor ablation, all training actors against the mean of
five three-actor draws, for V and E.

Usage: python -I -B part203/fig203.py [--results results/part203/results.json] [--out results/part203/fig203.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"V": "#2a78d6", "E": "#eb6834"}          # dataviz reference slots 1 and 2, validated on the light surface
READOUTS = [("V_grid", "V: final audio vector, C by inner CV"), ("D_grid", "D: final embedding (Part 196's), C by inner CV"),
            ("V_p196", "V at C = 1"), ("D_p196", "D at C = 1"), ("E_p196", "E: layer MLP input at C = 1"),
            ("H_p196", "H: seven acoustic features at C = 1")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results/part203/results.json"))
    ap.add_argument("--out", default=str(ROOT / "results/part203/fig203.png"))
    args = ap.parse_args(argv)
    res = json.loads(Path(args.results).read_text(encoding="utf-8"))
    chance = float(res["chance"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    ys = list(range(len(READOUTS)))[::-1]
    for y, (key, label) in zip(ys, READOUTS):
        r = res["readouts"][key]
        mean = r["mean_balanced_accuracy"]
        lo, hi = r["balanced_accuracy_actor_cluster_95"]["low"], r["balanced_accuracy_actor_cluster_95"]["high"]
        left.plot([lo, hi], [y, y], color=SERIES["V"], linewidth=2, solid_capstyle="round")
        left.plot([mean], [y], "o", color=SERIES["V"], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        left.text(hi + 0.012, y, f"{mean:.3f}", va="center", fontsize=9, color=INK)
    left.axvline(chance, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    left.text(chance + 0.006, len(READOUTS) - 0.55, "chance 1/6", fontsize=8.5, color=MUTED, va="center")
    left.set_yticks(ys, [label for _key, label in READOUTS], fontsize=9, color=INK)
    left.set_xlim(0, 0.75)
    left.set_ylim(-0.6, len(READOUTS) - 0.3)
    left.set_xlabel("six-way balanced accuracy, mean of 5 held-out actor folds (95% interval)", fontsize=9, color=MUTED)
    left.set_title("CREMA-D emotion, 91 actors, 2,730 clips: EmbeddingGemma 2", fontsize=10.5, color=INK, loc="left")

    for x, (name, label) in zip((0, 1), (("V", "V (C by inner CV)"), ("E", "E (C = 1, fold's layer)"))):
        a = res["three_actor_ablation"][name]
        full, three = a["all_training_actors_mean"], a["three_actor_draws_mean"]
        right.plot([0, 1], [full, three], color=SERIES[name], linewidth=2, marker="o", markersize=8,
                   markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        right.text(1.06, three, f"{name} {three:.3f} (drop {full - three:.3f})", va="center", fontsize=9, color=INK)
        right.text(-0.06, full, f"{full:.3f}", va="center", ha="right", fontsize=9, color=INK)
    right.axhline(chance, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    right.text(1.85, chance + 0.012, "chance 1/6", fontsize=8.5, color=MUTED, ha="right")
    held = [len(actors) for actors in res["gate1"]["folds"].values()]
    total = sum(held)
    sizes = sorted({total - n for n in held})
    right.set_xticks([0, 1], [f"all training actors\n({' or '.join(map(str, sizes))})", "3 training actors\n(mean of 5 draws)"],
                     fontsize=9, color=INK)
    right.set_xlim(-0.35, 1.9)
    right.set_ylim(0, 0.75)
    right.grid(axis="y", color=GRID, linewidth=0.8)
    right.grid(axis="x", visible=False)
    right.set_ylabel("balanced accuracy", fontsize=9, color=MUTED)
    right.set_title("Few training speakers cost V and E alike", fontsize=10.5, color=INK, loc="left")
    right.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
