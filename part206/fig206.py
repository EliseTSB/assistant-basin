"""Part 206's figure, drawn from results/part206/<stage>/summary.json and results/part206/qwen-chat/causal.json only.

Panel A: rho_peak(A-tmpl, M), the share of the assistant-frame difference that is one fixed vector, per OLMo 2 1B
stage in training order with its passage-resampled 95% interval, against the 0.2 bar of 206.6; Qwen3.5-4B chat and
base as separate markers.
Panel B: the slot and concept neurons' median arming percentile under (A-tmpl, M) per OLMo stage (206.8, 206.12,
Amendment 4's 21 slot and 18 concept neurons).
Panel C: the causal check on Qwen3.5-4B chat: the mean drop in the maker token's log probability when the assistant
shift is subtracted at k = 1 and k = 2, against the (M2, M) direction and the 20 random directions (206.10). Rows
restricted to the 23 phrasings where the plain model names its maker at top-1, as the design says.

Usage: python -I -B part206/fig206.py [--results results/part206] [--out results/part206/fig206.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOT, CONCEPT, ARM, CONTROL = "#2a78d6", "#eb6834", "#2a78d6", "#8a8984"
OLMO_ORDER = ["pretrain-3.0T", "midtrain-26B", "midtrain-end", "sft", "dpo", "instruct"]
BAR_206_6 = 0.2


def load(results: Path):
    out = {}
    for stage in OLMO_ORDER + ["qwen-base", "qwen-chat"]:
        out[stage] = json.loads((results / stage / "summary.json").read_text(encoding="utf-8"))
    out["causal"] = json.loads((results / "qwen-chat" / "causal.json").read_text(encoding="utf-8"))
    return out


def mean_change(causal, arm):
    rows = [r for r in causal["readout_a"] if r.get("plain_top1_is_maker")]
    return sum(r["logprob"][arm] - r["logprob"]["plain"] for r in rows) / len(rows)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results/part206"))
    ap.add_argument("--out", default=str(ROOT / "results/part206/fig206.png"))
    args = ap.parse_args(argv)
    R = load(Path(args.results))

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.25, 1, 1.1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (a, b, c):
        style(ax)

    # ---- Panel A: rho_peak(A-tmpl, M) per OLMo stage, Qwen as separate markers
    xs = list(range(len(OLMO_ORDER)))
    for x, stage in zip(xs, OLMO_ORDER):
        cm = R[stage]["common_mode"]["A-tmpl|M"]
        lo, hi = cm["rho_peak_ci"]
        a.plot([x, x], [lo, hi], color=SLOT, linewidth=2, solid_capstyle="round")
        a.plot([x], [cm["rho_peak"]], "o", color=SLOT, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        a.text(x, hi + 0.006, f"{cm['rho_peak']:.3f}", ha="center", fontsize=8.5, color=INK)
    a.axhline(BAR_206_6, color=INK, linestyle=(0, (4, 3)), linewidth=1)
    a.text(len(OLMO_ORDER) - 0.45, BAR_206_6 + 0.005, "206.6 bar (scored on Qwen chat only), 0.2", fontsize=8.5, color=MUTED, ha="right")
    qx = len(OLMO_ORDER) + 0.6
    for dx, stage, label in ((0.0, "qwen-chat", "Qwen chat"), (0.55, "qwen-base", "Qwen base")):
        cm = R[stage]["common_mode"]["A-tmpl|M"]
        lo, hi = cm["rho_peak_ci"]
        x = qx + dx
        a.plot([x, x], [lo, hi], color=CONCEPT, linewidth=2, solid_capstyle="round")
        a.plot([x], [cm["rho_peak"]], "D", color=CONCEPT, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        a.text(x, hi + 0.006, f"{cm['rho_peak']:.3f}", ha="center", fontsize=8.5, color=INK)
        a.text(x, 0.004, label, ha="center", fontsize=8, color=MUTED, rotation=90, va="bottom")
    a.set_xticks(xs + [qx + 0.275], ["pre-\ntrain", "mid-\ntrain\n26B", "mid-\ntrain\nend", "SFT", "DPO",
                                    "Instruct", "Qwen\n3.5-4B"], fontsize=8.5, color=INK)
    a.set_xlim(-0.6, qx + 1.3)
    a.set_ylim(0, 0.22)
    a.set_ylabel("rho_peak(A-tmpl, M): share of the frame difference\nthat is one fixed vector (95% interval)", fontsize=9, color=MUTED)
    a.set_title("A. How much of the assistant frame's difference is one fixed vector?\nAt most 0.164; Qwen chat 0.106, under 206.6's 0.2", fontsize=10.5, color=INK, loc="left")

    # ---- Panel B: slot vs concept median arming percentile per OLMo stage
    for x, stage in zip(xs, OLMO_ORDER):
        sl = R[stage]["arming"]["slot"]["median_percentile"]
        co = R[stage]["arming"]["concept"]["median_percentile"]
        b.plot([x - 0.16, x - 0.16], [0, sl], color=SLOT, linewidth=2, solid_capstyle="butt")
        b.plot([x + 0.16, x + 0.16], [0, co], color=CONCEPT, linewidth=2, solid_capstyle="butt")
    b.axhline(50, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    b.text(len(OLMO_ORDER) - 0.45, 51, "50, typical", fontsize=8.5, color=MUTED, ha="right")
    b.axhline(60, color=INK, linestyle=(0, (1, 2)), linewidth=1)
    b.text(len(OLMO_ORDER) - 0.45, 61, "206.8 bar for the slot median, 60", fontsize=8.5, color=MUTED, ha="right")
    b.set_xticks(xs, ["pre-\ntrain", "mid-\ntrain\n26B", "mid-\ntrain\nend", "SFT", "DPO", "Instruct"],
                 fontsize=8.5, color=INK)
    b.set_xlim(-0.6, len(OLMO_ORDER) - 0.4)
    b.set_ylim(0, 100)
    b.set_ylabel("median arming percentile under (A-tmpl, M)", fontsize=9, color=MUTED)
    b.set_title("B. Do slot neurons arm under the assistant frame?\nNot at Instruct: slot and concept stay near typical", fontsize=10.5, color=INK, loc="left")
    b.legend(handles=[Line2D([], [], color=SLOT, linewidth=3, label="slot, 21 neurons"),
                      Line2D([], [], color=CONCEPT, linewidth=3, label="concept, 18 neurons")],
             frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)

    # ---- Panel C: causal check on Qwen chat, change in the maker token's log probability
    causal = R["causal"]
    n_rows = sum(1 for r in causal["readout_a"] if r.get("plain_top1_is_maker"))
    groups = [("assistant shift k=1", "arm_k1", ARM), ("assistant shift k=2", "arm_k2", ARM),
              ("(M2, M) direction k=1", "null_M2M_k1", CONTROL), ("(M2, M) direction k=2", "null_M2M_k2", CONTROL)]
    r1 = [mean_change(causal, f"random{i}_k1") for i in range(20)]
    r2 = [mean_change(causal, f"random{i}_k2") for i in range(20)]
    cx = list(range(len(groups)))
    for x, (label, arm, col) in zip(cx, groups):
        v = mean_change(causal, arm)
        c.plot([x], [v], "o", color=col, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        c.text(x + 0.12, v, f"+{v:.3f}", va="center", fontsize=8.5, color=INK)
    c.plot([x + 0.35 for x in cx[:2]], [r1, r2], "o", color="#b9b8b2", markersize=4, alpha=0.9)
    c.axhline(0, color=MUTED, linewidth=1)
    c.set_xticks(cx, ["assistant\nshift k=1", "assistant\nshift k=2", "(M2, M)\ndirection k=1",
                      "(M2, M)\ndirection k=2"], fontsize=8.5, color=INK)
    c.set_xlim(-0.5, len(groups) - 0.15)
    c.set_ylim(-0.14, 0.14)
    c.set_ylabel("mean change in the maker token's log probability\n(positive = more likely)", fontsize=9, color=MUTED)
    c.set_title(f"C. Does removing the shift change the maker token?\nQwen chat, {n_rows} phrasings: it rises, the (M2, M) "
                "null more", fontsize=10.5, color=INK, loc="left")
    c.legend(handles=[Line2D([], [], color=ARM, marker="o", linestyle="", markersize=8, label="assistant shift"),
                      Line2D([], [], color=CONTROL, marker="o", linestyle="", markersize=8, label="(M2, M) direction"),
                      Line2D([], [], color="#b9b8b2", marker="o", linestyle="", markersize=5,
                             label="20 random directions, one dot each")],
             frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
