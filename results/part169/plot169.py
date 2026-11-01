"""Build the Part 169 loop summary figure from the saved score and review."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

arm_colors = {"n+": "#0072B2", "k4": "#D55E00"}  # Okabe-Ito blue and vermillion
arm_markers = {"n+": "o", "k4": "s"}
decile_colors = {"none": "#009E73", "own": "#E69F00", "free": "#CC79A7"}
decile_markers = {"none": "o", "own": "s", "free": "^"}
bans = ("none", "own", "free", "clear", "plain50")
x = list(range(len(bans)))

extra_text = (HERE / "extra169.txt").read_text(encoding="utf-8")
hit_rows = {
    arm: (int(rows), int(none), int(free), int(hit))
    for arm, rows, none, free, hit in re.findall(
        r"^(n\+|k4)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
        extra_text,
        re.MULTILINE,
    )
}
if set(hit_rows) != {"n+", "k4"}:
    raise ValueError("Could not read both arm counts for the same hit-defined rows from extra169.txt")

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16.2, 5.5),
    gridspec_kw={"width_ratios": [1.45, 1.1, 1.35]},
)
fig.suptitle(
    "The Part 168 free-list rise did not recur here",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)

# Panel 1: all rows, with whole-slot intervals; hit is deliberately omitted.
all_rows_ax, hit_ax, tenth_ax = axes
for arm, offset in (("n+", -0.105), ("k4", 0.105)):
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
    all_rows_ax.errorbar(
        [v + offset for v in x],
        vals,
        yerr=[lower, upper],
        fmt=arm_markers[arm],
        markersize=5.2,
        capsize=3,
        linewidth=1.3,
        color=arm_colors[arm],
        ecolor=arm_colors[arm],
        label=arm,
        zorder=3,
    )
all_rows_ax.set_title("Loop rate by ban\n(all rows; no hit)")
all_rows_ax.set_ylabel("Loop rate")
all_rows_ax.yaxis.set_major_formatter(PercentFormatter(1))
all_rows_ax.set_ylim(0, 0.52)
all_rows_ax.set_xticks(x, bans, rotation=25, ha="right")
all_rows_ax.grid(axis="y", color="#D8DDE2", linewidth=0.8)
all_rows_ax.set_axisbelow(True)
all_rows_ax.legend(title="Arm", frameon=False, loc="upper left")

# Panel 2: none, free and hit are compared on the identical 899 hit-defined rows.
hit_bans = ("none", "free", "hit")
hit_x = list(range(len(hit_bans)))
width = 0.34
for arm, offset in (("n+", -width / 2), ("k4", width / 2)):
    rows, none_count, free_count, hit_count = hit_rows[arm]
    counts = {"none": none_count, "free": free_count, "hit": hit_count}
    bars = hit_ax.bar(
        [v + offset for v in hit_x],
        [counts[ban] for ban in hit_bans],
        width,
        color=arm_colors[arm],
        label=arm,
        edgecolor="#FFFFFF",
        linewidth=0.7,
    )
    for bar, ban in zip(bars, hit_bans):
        hit_ax.annotate(
            str(counts[ban]),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
hit_ax.set_title("Loop answers\n(same 899 hit rows)")
hit_ax.set_ylabel("Loop answers")
hit_ax.set_xticks(hit_x, hit_bans)
hit_ax.set_ylim(0, 190)
hit_ax.grid(axis="y", color="#D8DDE2", linewidth=0.8)
hit_ax.set_axisbelow(True)
hit_ax.legend(title="Arm", frameon=False, loc="upper right")

# Panel 3: exploratory counts, with a distinct palette and marker for every ban.
for ban in ("none", "own", "free"):
    y = [row[{"none": 1, "own": 2, "free": 3}[ban]] for row in deciles]
    tenth_ax.plot(
        [row[0] for row in deciles],
        y,
        marker=decile_markers[ban],
        markersize=4.8,
        linewidth=1.8,
        color=decile_colors[ban],
        label=ban,
    )
tenth_ax.set_title("Loop answers by own-mass tenth\n(exploratory)")
tenth_ax.set_xlabel("Own-mass tenth (0 = lowest, 9 = highest)")
tenth_ax.set_ylabel("Loop answers")
tenth_ax.set_xticks(range(10))
tenth_ax.grid(axis="y", color="#D8DDE2", linewidth=0.8)
tenth_ax.set_axisbelow(True)
tenth_ax.legend(title="Ban", frameon=False, loc="upper left")

fig.subplots_adjust(left=0.06, right=0.99, top=0.83, bottom=0.19, wspace=0.36)
fig.savefig(HERE / "loops169.png", dpi=210, bbox_inches="tight", facecolor="white")
