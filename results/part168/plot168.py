#!/usr/bin/env python3
"""Create the Part 168 two-panel loop and fate figure."""
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, ROOT / "part86"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from part166 import score166
score = json.loads((HERE / "score.json").read_text(encoding="utf-8"))
answers = [json.loads(x) for x in (HERE / "answers.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]

arms = ["n+", "k4"]
bans = ["none", "own", "matched", "mass"]
palette = {"n+": "#0072B2", "k4": "#D55E00"}
events = [a for a in answers if a.get("stage") == "battery" and a.get("arm") in arms]
layers = sorted({int(a["layer"]) for a in events})
slots = range(12)
rng = np.random.default_rng(168)
replicates = 10000
intervals = {}
for arm in arms:
    for ban in bans:
        cells = {(layer, slot): [a for a in events if a["arm"] == arm and a["ban"] == ban and int(a["layer"]) == layer and int(a["slot"]) == slot] for layer in layers for slot in slots}
        counts = np.zeros((len(layers), 12, 2), dtype=float)
        for i, layer in enumerate(layers):
            for slot in slots:
                rows = cells[(layer, slot)]
                if rows:
                    loops = sum(bool(score166.loop_phrase(a.get("answer", ""))) for a in rows)
                    counts[i, slot] = (loops, len(rows))
        draw_counts = np.zeros((replicates, 2))
        for i in range(len(layers)):
            picks = rng.integers(0, 12, size=(replicates, 12))
            draw_counts += counts[i, picks].sum(axis=1)
        rates = draw_counts[:, 0] / draw_counts[:, 1]
        intervals[(arm, ban)] = np.quantile(rates, [0.025, 0.975])

fig, (ax, fate_ax) = plt.subplots(1, 2, figsize=(12.2, 5.4), gridspec_kw={"width_ratios": [1.05, 1.2]})
xs = np.arange(len(bans))
for j, arm in enumerate(arms):
    offset = (-0.13 if j == 0 else 0.13)
    means = [score["per_arm_ban"][f"{arm}|{ban}"]["loop_rate"] for ban in bans]
    lo = np.array([intervals[(arm, ban)][0] for ban in bans])
    hi = np.array([intervals[(arm, ban)][1] for ban in bans])
    y = np.array(means)
    ax.errorbar(xs + offset, y, yerr=np.vstack((y - lo, hi - y)), marker="o", linestyle="none", capsize=3,
                color=palette[arm], label=arm, markersize=6, linewidth=1.4)
ax.set_xticks(xs, bans)
ax.set_ylim(0, 0.23)
ax.set_ylabel("Loop answer rate")
ax.set_title("Loop rate by ban (95% whole-slot interval)", fontsize=10)
ax.grid(axis="y", color="#dddddd", linewidth=0.7)
ax.legend(frameon=False, title="Arm")

fates = ["no_loop", "reentry", "common_only", "top50", "migration", "outside", "other"]
fate_colors = ["#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#0072B2", "#999999"]
fate_labels = ["No loop", "Re-entry", "Common only", "Top 50", "Ranks 51–200", "Outside 200", "Other"]
left = np.zeros(3)
for fate, color, label in zip(fates, fate_colors, fate_labels):
    vals = []
    for ban in ("own", "matched", "mass"):
        cell = score["transitions_by_ban"][ban]["own_word_loop_fates"]
        vals.append(cell.get(fate, 0) / 88)
    vals = np.array(vals)
    fate_ax.barh(np.arange(3), vals, left=left, color=color, edgecolor="white", linewidth=0.5, label=label)
    left += vals
fate_ax.set_yticks(np.arange(3), ["Own ban", "Matched ban", "Mass ban"])
fate_ax.invert_yaxis()
fate_ax.set_xlim(0, 1)
fate_ax.set_xlabel("Share of 88 eligible rows")
fate_ax.set_title("Fate when none loops on an own-list word", fontsize=10)
fate_ax.grid(axis="x", color="#dddddd", linewidth=0.7)
fate_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8)

fig.suptitle("Own-list bans cut loops; equal-mass random bans add them", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.1, 1, 0.92))
fig.savefig(HERE / "loops168.png", dpi=180, bbox_inches="tight")
