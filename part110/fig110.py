#!/usr/bin/env python3
"""Part 110 figure: each named neuron's move against every neuron in its layer (small multiples, one shared log axis).

Written by Claude (Opus 5.5), 2026-10-29. Recomputes the per-neuron moves with weights110's own functions, saves them to
results/part110/moves110.npz, and draws results/part110/fig110.png. Colors from the dataviz reference palette.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weights110 as p110

OUT = Path("results/part110")
SURFACE, INK, INK2, GRID, BAR, MARK = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de", "#c9c8c2", "#2a78d6"


def moves():
    """{layer: move_mean [n]}; also saves each matrix's norm ratio for the exploratory look (explore110.py)."""
    path = OUT / "moves110.npz"
    if not path.exists():
        named = {**p110.MAKERS, **p110.SECONDARY}
        b, c = p110.Weights(p110.BASE_DIR), p110.Weights(p110.CHAT_DIR)
        arrays = {}
        for layer in sorted({l for l, _ in named.values()}):
            st = p110.layer_stats(p110.neuron_rows(b, layer), p110.neuron_rows(c, layer))
            arrays[f"L{layer}"] = st["move_mean"]
            for m in p110.MATS:
                arrays[f"ratio_{m}_L{layer}"] = st[f"ratio_{m}"]
            print(f"layer {layer}", flush=True)
        np.savez_compressed(path, **arrays)
    d = np.load(path)
    return {int(k[1:]): d[k] for k in d.files if k.startswith("L")}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mv = moves()
    res = json.loads((OUT / "weights110.json").read_text(encoding="utf-8"))
    order = list(p110.MAKERS) + list(p110.SECONDARY)
    lo = min(float(v.min()) for v in mv.values())
    hi = max(float(v.max()) for v in mv.values())
    edges = np.logspace(np.log10(lo) - 0.05, np.log10(hi) + 0.05, 90)
    fig, axes = plt.subplots(len(order), 1, figsize=(8.0, 1.05 * len(order) + 1.3), sharex=True, facecolor=SURFACE)
    for ax, name in zip(axes, order):
        n = res["neurons"][name]
        vals = mv[n["layer"]]
        assert abs(float(vals[n["unit"]]) - n["move_mean"]) < 1e-12, name  # same numbers as the registered run
        ax.set_facecolor(SURFACE)
        ax.hist(vals, bins=edges, color=BAR, edgecolor=SURFACE, linewidth=0.6)
        ax.axvline(n["move_mean"], color=MARK, linewidth=2)
        group = "maker" if name in p110.MAKERS else "secondary"
        ax.text(0.005, 0.80, f"{name} ({group})", transform=ax.transAxes, color=INK, fontsize=9, fontweight="bold")
        ax.text(0.005, 0.56, f"moved more than {n['pct_in_layer']['mean'] - 100 / n['n_layer']:.1f}% of layer {n['layer']}'s "
                f"{n['n_layer']:,} neurons", transform=ax.transAxes, color=INK2, fontsize=8)
        ax.set_xscale("log")
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=INK2, labelsize=8)
    from matplotlib.ticker import NullFormatter
    ticks = [0.001, 0.0015, 0.002, 0.003, 0.004, 0.006]
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([f"{t:g}" for t in ticks])
    axes[-1].xaxis.set_minor_formatter(NullFormatter())
    axes[-1].set_xlabel("move = 1 - cosine between base and chat weights (mean of gate row, up row, down column)",
                        color=INK2, fontsize=8.5)
    fig.suptitle("How far each neuron's weights turned from Qwen3.5-4B-Base to the chat model", color=INK, fontsize=11.5,
                 x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.955, "Gray: every neuron in that neuron's layer. Blue line: the neuron.", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig110.png", dpi=160, facecolor=SURFACE)
    print("wrote", OUT / "fig110.png")


if __name__ == "__main__":
    main()
