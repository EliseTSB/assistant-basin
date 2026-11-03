#!/usr/bin/env python3
"""Draw Part 177's three-panel summary from numbers177.json."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "numbers177.json").read_text(encoding="utf-8"))
    funnel = data["funnel_T"]
    onset = data["onset_order"]
    paired = data["first_50k_paired"]
    predictions = data["prediction_table"]
    prediction_text = predictions["177.6"]["prediction"]
    match = re.search(r"first ([0-9,]+) characters", prediction_text)
    character_window = match.group(1) if match else "50,000"

    real = [funnel["denominator_traces"], funnel["real"]["literal_hit_traces"],
            funnel["real"]["asserted_traces"]]
    shuffled = [funnel["denominator_traces"], funnel["shuffled_draw_1"]["literal_hit_traces"],
                funnel["shuffled_draw_1"]["asserted_traces"]]
    real_color = "#16697a"
    shuffled_color = "#bd6b32"
    onset_colors = ["#16697a", "#e5a23b", "#8b94a1"]
    axis_color = "#26313c"

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 13.2),
                             gridspec_kw={"height_ratios": [1.08, 1.18, 1.10]})
    fig.patch.set_facecolor("white")

    # Panel one: start both paths at the same T denominator, then show each funnel.
    ax = axes[0]
    x = [0, 1, 2]
    ax.plot(x, real, marker="o", markersize=9, linewidth=3.0, color=real_color,
            label="Real gold")
    ax.plot(x, shuffled, marker="o", markersize=9, linewidth=3.0, color=shuffled_color,
            label="Shuffled draw 1")
    ax.annotate(str(real[0]), (0, real[0]), xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", fontsize=14, fontweight="bold", color=axis_color)
    for index in (1, 2):
        ax.annotate(str(real[index]), (index, real[index]), xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=14, fontweight="bold", color=real_color)
        if shuffled[index] == 0:
            ax.annotate(str(shuffled[index]), (index, shuffled[index]), xytext=(0, 10), textcoords="offset points",
                        ha="center", va="bottom", fontsize=14, fontweight="bold", color=shuffled_color)
        else:
            ax.annotate(str(shuffled[index]), (index, shuffled[index]), xytext=(0, -17), textcoords="offset points",
                        ha="center", va="top", fontsize=14, fontweight="bold", color=shuffled_color)
    ax.set_title("1. Unedited funnel on target T", loc="left", fontsize=17, fontweight="bold", pad=12)
    ax.set_xticks(x, ["Traces in T", "Literal gold\nhit", "Judged\nasserted"], fontsize=13)
    ax.set_ylabel("Trace count", fontsize=12)
    ax.set_ylim(-20, max(real) * 1.22)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.grid(axis="y", color="#e4e8ec", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=11)

    # Panel two: separate the no-onset state as its own stacked segment.
    ax = axes[1]
    asserted_order = onset["asserted_on_T"]
    literal_order = onset["literal_hit_C"]
    groups = [
        ("Asserted on T", [asserted_order.get("asserted before onset", 0),
                                   asserted_order.get("onset before asserted hit", 0),
                                   asserted_order.get("no onset detected", 0)]),
        ("Literal hit on C", [literal_order.get("hit before onset", 0),
                               literal_order.get("onset before hit", 0),
                               literal_order.get("no onset detected", 0)]),
    ]
    y_positions = [1, 0]
    for y, (label, values) in zip(y_positions, groups):
        total = sum(values)
        left = 0.0
        for value, color in zip(values, onset_colors):
            width = value / total if total else 0.0
            ax.barh(y, width, left=left, height=0.52, color=color, edgecolor="white", linewidth=1.1)
            left += width
        ax.text(1.04, y, " / ".join(str(value) for value in values), va="center", ha="left",
                fontsize=12, color=axis_color)
        groups[groups.index((label, values))] = (f"{label} (n={total})", values)
    ax.set_title("2. Order relative to detected onset", loc="left", fontsize=17, fontweight="bold", pad=12)
    ax.text(1.04, 1.40, "Counts in legend order", fontsize=10, color="#56616d", ha="left")
    ax.set_yticks(y_positions, [groups[0][0], groups[1][0]], fontsize=12)
    ax.set_xlim(0, 1.52)
    ax.set_xticks([])
    ax.set_ylim(-0.65, 1.63)
    ax.grid(False)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=color) for color in onset_colors],
              ["Before onset", "Onset before hit", "No onset detected"],
              frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(0.0, -0.01), fontsize=11)

    # Panel three: paired rates plus the discordant-pair counts.
    ax = axes[2]
    rates = paired["rates"]
    labels = [("Unedited", rates["unedited"]), ("p50c", rates["p50c"])]
    y_positions = [1, 0]
    colors = [real_color, shuffled_color]
    for y, (label, row), color in zip(y_positions, labels, colors):
        rate = row["numerator"] / row["denominator"] if row["denominator"] else 0.0
        ax.barh(y, rate, height=0.48, color=color, alpha=0.92)
        ax.text(rate - 0.025, y,
                f"{row['numerator']}/{row['denominator']} ({rate * 100:.1f}%)",
                va="center", ha="right", fontsize=12.5, fontweight="bold", color="white")
    ax.set_title(f"3. Selected pairs: onset\nwithin {character_window} characters", loc="left",
                 fontsize=16, fontweight="bold", pad=12)
    ax.set_yticks(y_positions, ["Unedited", "p50c"], fontsize=13)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.5, 1.0], ["0", "50%", "100%"], fontsize=10)
    ax.set_xlabel("Share with detected onset", fontsize=11)
    ax.grid(axis="x", color="#e4e8ec", linewidth=0.9)
    ax.set_axisbelow(True)
    cells = paired["detection_cells"]
    ax.text(0, -0.45,
            f"{paired['eligible_pairs']} pairs. Both texts ran past 50,000 chars.\n"
            f"The p50c member is a subset selected on post-edit length.\n"
            f"Both: {cells['both']}, unedited only: {cells['unedited_only']}, "
            f"p50c only: {cells['p50c_only']}, neither: {cells['neither']}.",
            fontsize=9.5, color="#56616d", ha="left", va="top", wrap=True)
    ax.set_ylim(-1.05, 1.55)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#a9b1ba")
        ax.spines["bottom"].set_color("#a9b1ba")
        ax.tick_params(axis="both", colors=axis_color)
    fig.subplots_adjust(left=0.30, right=0.96, top=0.97, bottom=0.055, hspace=0.66)
    fig.savefig(ROOT / "plot177.png", dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
