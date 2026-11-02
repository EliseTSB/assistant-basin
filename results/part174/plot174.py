#!/usr/bin/env python3
"""Plot Part 174's signed directions from scored_directions.jsonl."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "scored_directions.jsonl"
OUTPUT = ROOT / "plot174.png"
UNLABELLED_X = 30
NO_CLEAN_X = 35

rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
families = ("u100d", "f100d")
centres = {"u100d": 1.0, "f100d": 0.0}

fig, ax = plt.subplots(figsize=(9.0, 4.5))
cmap = plt.get_cmap("YlOrRd")
norm = Normalize(vmin=0, vmax=44)
mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

for family in families:
    family_rows = [row for row in rows if row["family"] == family]
    by_count = defaultdict(list)
    for row in family_rows:
        count = row["google_claim_count"]
        if not row["reader_available"]:
            x = NO_CLEAN_X
        else:
            x = UNLABELLED_X if count is None else int(count)
        by_count[x].append(row)

    centre = centres[family]
    for x, group in sorted(by_count.items()):
        offsets = [0.0] if len(group) == 1 else [
            -0.31 + 0.62 * i / (len(group) - 1) for i in range(len(group))
        ]
        ys = [centre + offset for offset in offsets]
        if x == UNLABELLED_X:
            ax.scatter(
                [x] * len(group), ys, marker="D", s=42,
                facecolors="none", edgecolors="#4b5563", linewidths=1.2,
                zorder=3,
            )
        elif x == NO_CLEAN_X:
            ax.scatter(
                [x - 0.75 + 0.5 * (i % 4) for i in range(len(group))],
                [centre + 0.31 - 0.62 * (i // 4) / 7 for i in range(len(group))],
                marker="s", s=24,
                facecolors="none", edgecolors="#b91c1c", linewidths=1.0,
                zorder=3,
            )
        else:
            broken = [int(row["summary"]["broken"]) for row in group]
            ax.scatter(
                [x] * len(group), ys, c=broken, cmap=cmap, norm=norm,
                marker="o", s=43, edgecolors="white", linewidths=0.45,
                zorder=3,
            )

broken_summary = {}
for family in families:
    values = [int(row["summary"]["broken"]) for row in rows if row["family"] == family]
    broken_summary[family] = (
        statistics.median(values), min(values), max(values)
    )

u_med, u_min, u_max = broken_summary["u100d"]
f_med, f_min, f_max = broken_summary["f100d"]

ax.axvline(10, color="#263238", linestyle=(0, (4, 3)), linewidth=1.5, zorder=1)
ax.text(10.3, 1.42, "modal-count floor, 10 of 44 (support also needs at most 8 broken)", fontsize=9.5, color="#263238", va="center")
ax.axhline(0.5, color="#d1d5db", linewidth=0.8, zorder=0)
ax.set_xlim(-0.8, 37.0)
ax.set_ylim(-0.48, 1.58)
ax.set_yticks([1, 0], ["u100d", "f100d"], fontsize=12)
ax.set_xticks([0, 5, 10, 15, 20, 25, UNLABELLED_X, NO_CLEAN_X])
ax.set_xticklabels(["0", "5", "10", "15", "20", "25", "some\nunlabelled", "no clean\nanswers"], fontsize=10)
ax.set_xlabel("Google claim count per signed direction", fontsize=12)
ax.grid(axis="x", color="#e5e7eb", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

fig.suptitle("Google claims across signed directions", x=0.10, y=0.98, ha="left", fontsize=17)
fig.text(
    0.10, 0.895,
    "Each dot is one sign; colour shows broken answers. "
    f"Broken median: u100d {u_med:g} ({u_min}–{u_max}), "
    f"f100d {f_med:g} ({f_min}–{f_max}).",
    fontsize=10.5, color="#4b5563", ha="left",
)
fig.text(
    0.10, 0.855,
    "Diamonds: some clean answers unlabelled. Red squares: no clean answers, so nothing was read. Neither is a zero.",
    fontsize=10, color="#4b5563", ha="left",
)
fig.subplots_adjust(left=0.10, right=0.88, top=0.78, bottom=0.19)

colorbar = fig.colorbar(mappable, ax=ax, pad=0.025, fraction=0.035)
colorbar.set_label("Broken answers", fontsize=10.5)
colorbar.set_ticks([0, 8, 22, 44])
colorbar.ax.tick_params(labelsize=9.5)

fig.savefig(OUTPUT, dpi=180, facecolor="white", bbox_inches="tight")
plt.close(fig)

