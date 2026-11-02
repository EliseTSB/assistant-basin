#!/usr/bin/env python3
"""Draw the Part 176 answer states, support bounds, and per-direction heatmap."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "numbers176.json").read_text(encoding="utf-8"))
rungs = data["rung_table"]
states = data["answer_state_order"]
directions = data["directions"]
doses = [row["dose"] for row in rungs]
direction_count = rungs[0]["support"]["denominator"]
prompts_per_direction = rungs[0]["answers"] // direction_count


def _format_bounds(lower, upper):
    return str(lower) if lower == upper else f"{lower}–{upper}"

state_colors = {
    "qwen": "#4C78A8",
    "google": "#F2BE22",
    "other name": "#54A24B",
    "claim with no single name": "#B279A2",
    "reverted": "#9D755D",
    "longer no claim": "#AEB8C2",
    "fragment": "#F58518",
    "broken": "#E45756",
    "unlabelled": "#424A53",
}

fig = plt.figure(figsize=(8.7, 23.0), facecolor="white")
grid = fig.add_gridspec(3, 1, height_ratios=[2.5, 2.6, 10.9], hspace=0.53)
fig.subplots_adjust(left=0.19, right=0.88, top=0.955, bottom=0.065)

# Panel 1: pooled answer states across all directions and prompts.
ax1 = fig.add_subplot(grid[0])
y = np.arange(len(rungs))
left = np.zeros(len(rungs), dtype=float)
for state in states:
    values = np.array([row["answer_state_counts"][state] for row in rungs])
    ax1.barh(y, values, left=left, height=0.62, color=state_colors[state],
             edgecolor="white", linewidth=0.45, label=state)
    if state == "google":
        for index, (start, width) in enumerate(zip(left, values)):
            if width >= 50:
                ax1.text(start + width / 2, index, f"{int(width)}", ha="center", va="center",
                         fontsize=8.5, color="#29220B", fontweight="bold")
    left += values
answer_total = rungs[0]["answers"]
for index, row in enumerate(rungs):
    ax1.text(answer_total + answer_total * 0.012, index,
             f"n={row['answers']:,}", va="center", ha="left", fontsize=8.5, color="#333333")
ax1.set_xlim(0, answer_total * 1.19)
ax1.set_yticks(y, [f"{row['dose']:g}" for row in rungs])
ax1.invert_yaxis()
ax1.set_xlabel(f"Answers across {direction_count} signed directions", fontsize=10)
ax1.set_ylabel("Dose rung", fontsize=10)
ax1.set_title(f"1. Answer states at each rung ({answer_total:,} answers per rung)",
              loc="left", fontsize=13, fontweight="bold", pad=10)
ax1.xaxis.set_major_locator(mticker.MultipleLocator(250))
ax1.xaxis.grid(True, color="#D9DEE3", linewidth=0.6)
ax1.set_axisbelow(True)
ax1.tick_params(axis="both", labelsize=9)
ax1.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.34),
           frameon=False, fontsize=8.4, columnspacing=1.5, handlelength=1.2)
for spine in ["top", "right", "left"]:
    ax1.spines[spine].set_visible(False)

# Panel 2: definite support, direction-of-support bounds, and bounded/null counts.
ax2 = fig.add_subplot(grid[1])
bar_height = 0.56
for index, row in enumerate(rungs):
    support = row["support"]
    google = row["google_led"]
    other = row["other_led"]
    nulls = row["null_direction_counts"]
    definite_google = google["lower"]
    definite_other = other["lower"]
    bounded_support = support["upper"] - support["lower"]
    unsupported = support["denominator"] - support["upper"]
    x0 = 0
    ax2.barh(index, definite_google, left=x0, height=bar_height, color="#F2BE22",
             edgecolor="white", linewidth=0.6)
    x0 += definite_google
    if definite_other:
        ax2.barh(index, definite_other, left=x0, height=bar_height, color="#54A24B",
                 edgecolor="white", linewidth=0.6)
    x0 += definite_other
    if bounded_support:
        ax2.barh(index, bounded_support, left=x0, height=bar_height, color="#FFF4C2",
                 edgecolor="#806B24", linewidth=0.8, hatch="////")
    x0 += bounded_support
    ax2.barh(index, unsupported, left=x0, height=bar_height, color="#E5E9ED",
             edgecolor="white", linewidth=0.6)
    support_text = _format_bounds(support["lower"], support["upper"])
    google_text = _format_bounds(google["lower"], google["upper"])
    other_text = _format_bounds(other["lower"], other["upper"])
    label = (f"S {support_text}   G {google_text}   O {other_text}"
             f"   status-null {nulls['support_bounded']}/{nulls['google_led_bounded']}/{nulls['other_led_bounded']}")
    ax2.text(support["denominator"] + 1.2, index, label, va="center", ha="left", fontsize=8.2, color="#25282B")

selected_dose = data["selected"]["dose"]
yticklabels = [f"{row['dose']:g}" + ("  selected" if row["dose"] == selected_dose else "") for row in rungs]
ax2.set_xlim(0, direction_count + 36)
ax2.set_yticks(y, yticklabels)
ax2.invert_yaxis()
ax2.set_xticks(np.linspace(0, direction_count, 5))
ax2.set_xlabel("Directions; row labels: S=support, G=Google-led, O=other-led, status-null S/G/O", fontsize=9.3)
ax2.set_ylabel("Dose rung", fontsize=10)
ax2.set_title("2. Supported directions and null bounds", loc="left", fontsize=13,
              fontweight="bold", pad=10)
ax2.xaxis.grid(True, color="#D9DEE3", linewidth=0.6)
ax2.set_axisbelow(True)
ax2.tick_params(axis="both", labelsize=9)
legend2 = [
    mpatches.Patch(facecolor="#F2BE22", label="definitely Google-led support"),
    mpatches.Patch(facecolor="#54A24B", label="definitely other-led support"),
    mpatches.Patch(facecolor="#FFF4C2", edgecolor="#806B24", hatch="////", label="bounded support"),
    mpatches.Patch(facecolor="#E5E9ED", label="not supported"),
]
ax2.legend(handles=legend2, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.30),
           frameon=False, fontsize=8.2, columnspacing=1.4, handlelength=1.4)
for spine in ["top", "right", "left"]:
    ax2.spines[spine].set_visible(False)

# Panel 3: every direction, ordered by interpolated L; missing cells stay visibly null.
ax3 = fig.add_subplot(grid[2])

def direction_sort_key(row):
    collapse = row["collapse_dose"]
    value = collapse.get("L")
    if value is not None:
        return (0, float(value), row["direction_id"])
    if collapse.get("status") == "right censored":
        return (0, 0.300001, row["direction_id"])
    return (1, float("inf"), row["direction_id"])


directions = sorted(directions, key=direction_sort_key)
values = np.array([
    [np.nan if rung["google_count"] is None else rung["google_count"] for rung in row["rungs"]]
    for row in directions
], dtype=float)
masked = np.ma.masked_invalid(values)
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("#F3F4F5")
norm = mcolors.Normalize(vmin=0, vmax=prompts_per_direction)
x_edges = [doses[0] - (doses[1] - doses[0]) / 2]
x_edges.extend((doses[i] + doses[i + 1]) / 2 for i in range(len(doses) - 1))
x_edges.append(doses[-1] + (doses[-1] - doses[-2]) / 2)
y_edges = np.arange(len(directions) + 1)
mesh = ax3.pcolormesh(x_edges, y_edges, masked, cmap=cmap, norm=norm,
                      shading="flat", edgecolors="white", linewidth=0.35)

for row_index, row in enumerate(directions):
    center_y = row_index + 0.5
    for column_index, rung in enumerate(row["rungs"]):
        if rung["google_count"] is None:
            ax3.add_patch(mpatches.Rectangle(
                (x_edges[column_index], row_index),
                x_edges[column_index + 1] - x_edges[column_index], 1,
                facecolor="none", edgecolor="#A9AFB5", hatch="////", linewidth=0.35,
                zorder=3,
            ))
    collapse = row["collapse_dose"]
    if collapse.get("L") is not None:
        ax3.scatter([collapse["L"]], [center_y], marker="D", s=17, c="#111111",
                    edgecolors="white", linewidths=0.45, zorder=5)
    elif collapse.get("status") == "right censored":
        ax3.scatter([doses[-1]], [center_y], marker="D", s=17, facecolors="white",
                    edgecolors="#111111", linewidths=0.9, zorder=5)
        ax3.annotate("", xy=(doses[-1] + 0.011, center_y), xytext=(doses[-1] + 0.001, center_y),
                     arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 0.8}, zorder=5)

ax3.set_xlim(x_edges[0], x_edges[-1] + 0.018)
ax3.set_ylim(0, len(directions))
ax3.invert_yaxis()
ax3.set_xticks(doses, [f"{dose:g}" for dose in doses])
ax3.set_yticks(np.arange(len(directions)) + 0.5,
               [row["direction_id"] for row in directions])
ax3.set_xlabel(f"Dose rung; black diamond marks L, open diamond marks L > {doses[-1]:g}", fontsize=10)
ax3.set_ylabel("Signed direction, sorted by L", fontsize=10)
ax3.set_title("3. Google answers by signed direction and rung", loc="left",
              fontsize=13, fontweight="bold", pad=10)
ax3.tick_params(axis="x", labelsize=9, length=0, pad=7)
ax3.tick_params(axis="y", labelsize=7.2, length=0, pad=3)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
colorbar = fig.colorbar(mesh, ax=ax3, fraction=0.025, pad=0.04)
colorbar.set_label(f"Google answers (0–{prompts_per_direction}); hatched cell = null, not zero", fontsize=9)
colorbar.set_ticks(np.linspace(0, prompts_per_direction, 5))
colorbar.ax.tick_params(labelsize=8)

legend3 = [
    Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="#111111",
           markeredgecolor="white", markersize=6, label="interpolated L"),
    Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white",
           markeredgecolor="#111111", markersize=6, label="right-censored L"),
    mpatches.Patch(facecolor="#F3F4F5", edgecolor="#A9AFB5", hatch="////",
                   label="null Google count, not zero"),
]
fig.legend(handles=legend3, loc="lower center", bbox_to_anchor=(0.5, 0.004),
           ncol=3, frameon=False, fontsize=8.1, handlelength=1.3, columnspacing=1.4)

fig.savefig(ROOT / "plot176.png", dpi=170, facecolor="white")
print(f"Wrote {ROOT / 'plot176.png'}")
