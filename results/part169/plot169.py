"""Build the Part 169 loop summary figure from the saved score and review."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
score = json.loads((HERE / "score.json").read_text(encoding="utf-8"))
review = (HERE / "review169.txt").read_text(encoding="utf-8")

decile_re = re.compile(
    r"^tenth\s+(\d+)\s+own mass\s+(\S+)\s+to\s+(\S+)\s+"
    r"rows\s+(\d+)\s+none\s+(\d+)\s+own\s+(\d+)\s+free\s+(\d+)"
)
deciles = []
for line in review.splitlines():
    match = decile_re.match(line)
    if match:
        tenth, _low, _high, _rows, none, own, free = match.groups()
        deciles.append((int(tenth), int(none), int(own), int(free)))
if len(deciles) != 10:
    raise ValueError(f"Expected 10 own-mass tenths, found {len(deciles)}")

colors = {
    "n+": "#0072B2",       # blue
    "k4": "#D55E00",       # vermillion
    "none": "#0072B2",    # blue
    "own": "#D55E00",      # vermillion
    "free": "#009E73",     # bluish green
}
bans = ("none", "own", "free", "clear", "plain50", "hit")
ban_labels = ("none", "own", "free", "clear", "plain50", "hit")
x = list(range(len(bans)))

fig, (left, right) = plt.subplots(1, 2, figsize=(12.2, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
fig.suptitle(
    "On new neurons and phrasings, own top-50 cuts loops again; free does not restore Part 168's rise",
    fontsize=14,
    fontweight="bold",
    y=0.97,
)

for arm, offset in (("n+", -0.09), ("k4", 0.09)):
    vals = []
    lower = []
    upper = []
    for ban in bans:
        cell = score["bootstrap_intervals"][arm][f"{ban}_loop_rate"]
        estimate = cell["estimate"]
        lo, hi = cell["interval_95"]
        vals.append(estimate)
        lower.append(estimate - lo)
        upper.append(hi - estimate)
    left.errorbar(
        [v + offset for v in x],
        vals,
        yerr=[lower, upper],
        fmt="o",
        markersize=5.5,
        capsize=3,
        linewidth=1.35,
        color=colors[arm],
        ecolor=colors[arm],
        label=arm,
        zorder=3,
    )

left.set_title("Loop answers by ban and arm")
left.set_ylabel("Loop rate")
left.yaxis.set_major_formatter(PercentFormatter(1))
left.set_ylim(0, 0.52)
left.set_xticks(x, ban_labels)
left.grid(axis="y", color="#D8DDE2", linewidth=0.8)
left.set_axisbelow(True)

for ban, color in (("none", colors["none"]), ("own", colors["own"]), ("free", colors["free"])):
    y = [row[{"none": 1, "own": 2, "free": 3}[ban]] for row in deciles]
    right.plot(
        [row[0] for row in deciles],
        y,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        color=color,
        label=ban,
    )
right.set_title("Loop answers by own-mass tenth")
right.set_xlabel("Own-mass tenth (0 = lowest, 9 = highest)")
right.set_ylabel("Loop answers")
right.set_xticks(range(10))
right.grid(axis="y", color="#D8DDE2", linewidth=0.8)
right.set_axisbelow(True)

legend_handles = [
    Line2D([0], [0], marker="o", color=colors["n+"], label="n+", linewidth=1.4),
    Line2D([0], [0], marker="o", color=colors["k4"], label="k4", linewidth=1.4),
    Line2D([0], [0], color=colors["none"], marker="o", label="none", linewidth=1.8),
    Line2D([0], [0], color=colors["own"], marker="o", label="own", linewidth=1.8),
    Line2D([0], [0], color=colors["free"], marker="o", label="free", linewidth=1.8),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.065),
    ncol=5,
    frameon=False,
    columnspacing=1.8,
)
fig.text(
    0.5,
    0.13,
    "Left: points and whole-slot 95% intervals; hit has fewer eligible rows. Right: exploratory decile counts.",
    ha="center",
    va="center",
    fontsize=9,
    color="#39434D",
)
fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.27, wspace=0.28)
fig.savefig(HERE / "loops169.png", dpi=210, bbox_inches="tight", facecolor="white")
