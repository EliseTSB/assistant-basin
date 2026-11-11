#!/usr/bin/env python3
"""Draw Part 201's figure from results/part201/score.json only.

Every number on the figure is read from score.json; nothing is typed in. The figure asks one question in plain
words: when the batch shape or the precision changes, how often does the text change, and where does the decode
part? Left: per pair, the share of rows whose token ids differ, with its 95% Wilson interval (the interval is
computed here from the differing and compared counts, Wilson, 95%). Middle: the A/M label flips by direction, the
only pair with flips to show. Right: the pass-A margin, in bf16 steps, at the first difference for A/C and A/F8,
with the two-step gate drawn.

Run: /home/elisepc/neuron-probe/.venv/bin/python -I part201/fig201.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
SCORE = ROOT / "results" / "part201" / "score.json"
OUT = ROOT / "results" / "part201" / "fig201.png"

score = json.loads(SCORE.read_text(encoding="utf-8"))
pairs = score["pairs"]

# The pairs in the order the design lists them, with the colour that says what the pair changes.
ORDER = [
    ("A/A-again", "same call, repeated", "#9a9a9a"),
    ("A/A2", "new process", "#9a9a9a"),
    ("A/B", "other slots and neighbours", "#4a7bb7"),
    ("A/M", "prompt mix, padding, neighbours", "#4a7bb7"),
    ("A/C", "one row per call", "#4a7bb7"),
    ("F8/F1", "float32, two shapes", "#c0392b"),
    ("A/F8", "bf16 against float32", "#c0392b"),
]


def wilson(k, n, z=1.959963984540054):
    """95% Wilson interval for k of n. Computed here; the report prints the scorer's own."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


fig = plt.figure(figsize=(13.6, 6.1), dpi=150)
gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.95, 1.15], wspace=0.34, top=0.80, bottom=0.13)

fig.suptitle("Part 201: changing the batch shape moves bf16 answers; float32 on the card moves them too, "
             "and only at near-ties",
             fontsize=12.5, y=0.975)
fig.text(0.5, 0.905,
         "Qwen3.5-4B (851bf6e8), Intel Arc A770, eager attention, greedy, dose 0.8 pushes, 80 new tokens. "
         "Every number from score.json.",
         ha="center", fontsize=8.5, color="#444444")

# ---- Left: share of rows differing, with 95% Wilson intervals -------------------------------
ax = fig.add_subplot(gs[0, 0])
labels = []
for i, (name, what, colour) in enumerate(ORDER):
    p = pairs[name]
    k, n = p["differing"], p["compared"]
    share, lo, hi = wilson(k, n)
    ax.barh(i, share, height=0.62, color=colour)
    ax.plot([lo, hi], [i, i], color="#222222", linewidth=1.6)
    ax.plot([lo, lo], [i - 0.12, i + 0.12], color="#222222", linewidth=1.6)
    ax.plot([hi, hi], [i - 0.12, i + 0.12], color="#222222", linewidth=1.6)
    ax.text(max(hi, share) + 0.015, i, f"{k}/{n} = {share:.3f}\n({lo:.3f} to {hi:.3f})", va="center", fontsize=8)
    labels.append(f"{name}\n{what}")
ax.set_yticks(range(len(ORDER)), labels, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlim(0, 1.12)
ax.set_xlabel("share of rows whose token ids differ (95% Wilson interval)", fontsize=9)
ax.set_title("Does the text change?", fontsize=10.5, loc="left")
ax.grid(axis="x", color="#dddddd", linewidth=0.7)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(handles=[Patch(color="#9a9a9a", label="same shape, control"),
                   Patch(color="#4a7bb7", label="bf16, other batch shape"),
                   Patch(color="#c0392b", label="float32")],
          frameon=False, fontsize=8, loc="upper right")

# ---- Middle: A/M label flips by direction ----------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
flips = pairs["A/M"]["label_flips"]
names = list(flips.keys())
short = {"name_gone": "name gone", "maker_gone": "maker gone", "broken": "broken",
         "qwen_part161_rule": "Part 161 Qwen rule"}
x = list(range(len(names)))
w = 0.38
t2f = [flips[l]["a_true_b_false"] for l in names]
f2t = [flips[l]["a_false_b_true"] for l in names]
ax.bar([i - w / 2 for i in x], t2f, w, color="#4a7bb7", label="true in A, false in M")
ax.bar([i + w / 2 for i in x], f2t, w, color="#d17a32", label="false in A, true in M")
for i, (a, b) in enumerate(zip(t2f, f2t)):
    ax.text(i - w / 2, a + 0.25, str(a), ha="center", fontsize=8.5)
    ax.text(i + w / 2, b + 0.25, str(b), ha="center", fontsize=8.5)
ax.axhline(8, color="#555555", linestyle=":", linewidth=1)
ax.text(-0.45, 8.9, "201.6 scores a label from 8 flips", ha="left", fontsize=7.5, color="#555555")
ax.set_xticks(x, [short.get(l, l) for l in names], fontsize=8, rotation=18, ha="right")
ax.set_ylabel("flips, of 384 rows", fontsize=9)
ax.set_ylim(0, max(t2f + f2t) + 3)
ax.set_title("A/M: which way the labels flip", fontsize=10.5, loc="left")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.grid(axis="y", color="#dddddd", linewidth=0.7)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

# ---- Right: pass-A margin at the first difference, in bf16 steps -----------------------------
ax = fig.add_subplot(gs[0, 2])
for name, colour in (("A/C", "#4a7bb7"), ("A/F8", "#c0392b")):
    steps = [r["a_bf16_steps"] for r in pairs[name]["differing_rows"]]
    bins = [k / 2 - 0.25 for k in range(0, 11)]  # half-step bins: margins sit at 0, 0.5, 1, 2, 3 steps
    ax.hist(steps, bins=bins, color=colour, alpha=0.55,
            label=f"{name} (n={len(steps)})", edgecolor=colour, linewidth=0.8)
ax.axvline(2, color="#333333", linestyle="--", linewidth=1.2)
ax.text(2.08, ax.get_ylim()[1] * 0.92, "2-step gate", fontsize=8, color="#333333")
ax.text(0.3, ax.get_ylim()[1] * 0.81, "0 = exact bf16 tie", fontsize=7.5, color="#333333")
ax.set_xlabel("pass-A margin at the first difference, bf16 steps", fontsize=9)
ax.set_ylabel("differing rows", fontsize=9)
ax.set_title("Where the decode parts", fontsize=10.5, loc="left")
ax.set_xticks(range(0, 5))
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", color="#dddddd", linewidth=0.7)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
