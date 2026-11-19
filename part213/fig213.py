#!/usr/bin/env python3
"""Draw Part 213's figure from the result files: do the slot neurons write their persona into ordinary "X is a ___"
slots, or only into the self?

Every number is computed here from results/part213/full/stage1-instruct.jsonl through the scorer's own
score213.load_records and score213.Stage1 (imported, never edited); nothing is typed in. One question, in the title:
do the 21 slot units write their persona after "I am" only, or also after "The stranger is" and "The keepsake is"?
Panel (a) slot units: L_z (neuron +0.8 minus the mean of random_0..3 at +0.8) per unit on SELF_HELD, THIRD_SLOT and
COPULA_NONID, the three write cells 213.10 reads. Panel (b) the same for the 18 concept units. Panel (c) the random
arms alone (R_z) beside them, so a reader sees what an arbitrary push of the same size does. A grey bar beside each
column is the group median.

Run from the repo root:
  MPLCONFIGDIR=/tmp/mpl /home/elisepc/neuron-probe/.venv/bin/python -I -B part213/fig213.py
Writes results/part213/fig213.png.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score213 as sc  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = HERE.parent
FULL = ROOT / "results" / "part213" / "full"
OUT = ROOT / "results" / "part213" / "fig213.png"

CELLS = (("SELF_HELD", sc.SELF_HELD, "self-slot, 4 prompts\n(e.g. \"I am\")", "#2f6db5"),
         ("THIRD_SLOT", sc.THIRD_SLOT, "third-person identity, 4 prompts\n(e.g. \"The stranger is\")", "#c0392b"),
         ("COPULA_NONID", sc.COPULA_NONID, "non-identity copula, 2 prompts\n(e.g. \"The keepsake is\")", "#e08a2e"))
GREY = "#9a9a9a"

records = sc.load_records(FULL)
targets = sc.load_targets()
s1 = sc.Stage1(records, targets)
UNITS = {u["id"]: u for u in targets["units"]}

BLUE, RED, ORANGE = CELLS[0][3], CELLS[1][3], CELLS[2][3]

fig = plt.figure(figsize=(14.0, 12.6), dpi=150)
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.62], wspace=0.16, hspace=1.15,
                      top=0.800, bottom=0.250, left=0.055, right=0.985)
fig.suptitle("Part 213: the slot neurons write their persona into the self-slot, and about as much into third-person identity\n"
             "slots as into non-identity copula slots, so a \"write this noun after is\" reading competes with a self-slot one",
             fontsize=13, y=0.995)
fig.text(0.5, 0.940,
         "OLMo 2 1B Instruct (48d788eca847), CPU float32, chat template. L_z = the unit's own first-content logit minus\n"
         "the median of the 35-38 other units' logits, neuron +0.8 minus the mean of 4 same-size random pushes,\n"
         "averaged over the cell's prompts. 21 slot and 18 concept units (Part 92). Every number read from\n"
         "results/part213/full/stage1-instruct.jsonl through score213.Stage1.",
         fontsize=8.6, ha="center", va="top")


def panel(ax, ids, which, title, ylim):
    n = len(ids)
    x = np.arange(n)
    width = 0.26
    for j, (name, prompts, _label, colour) in enumerate(CELLS):
        vals = []
        for uid in ids:
            if which == "L":
                vals.append(s1.lz("instruct", uid, prompts))
            else:  # R_z alone: the mean of the random arms' increment
                vals.append(s1.rz("instruct", uid, prompts))
        vals = np.array([np.nan if v is None else v for v in vals], dtype=float)
        ax.bar(x + (j - 1) * width, vals, width * 0.92, color=colour, edgecolor="none",
               label=_label.replace("\n", " "), zorder=2)
        med = float(np.nanmedian(vals))
        ax.hlines(med, x.min() - 0.55, x.max() + 0.55, color=colour, linewidth=1.6, linestyle=(0, (4, 2)), zorder=3)
    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.tick_params(axis="x", pad=2)
    ax.set_xticklabels([UNITS[u]["np_text"] for u in ids], rotation=40, ha="right", fontsize=8,
                       rotation_mode="anchor")
    ax.set_ylabel("L_z, logits" if which == "L" else "random-arm increment R_z, logits", fontsize=9)
    ax.set_title(title, fontsize=10.5, loc="left", pad=8)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


panel(fig.add_subplot(gs[0, 0]), s1.slot, "L",
      "(a) Slot units: the neuron's write above random, per unit\n(dashed line: the group median)",
      (-1.5, 10.0))
panel(fig.add_subplot(gs[0, 1]), s1.concept, "L",
      "(b) Concept units, same cells\n(dashed line: the group median)",
      (-1.5, 14.0))
panel(fig.add_subplot(gs[1, 0]), s1.slot, "R",
      "(c) Slot units, the random pushes alone of the same size\n(R_z: what an arbitrary push does)",
      (-3.0, 4.5))
panel(fig.add_subplot(gs[1, 1]), s1.concept, "R",
      "(d) Concept units, the random pushes alone\n(R_z)",
      (-3.5, 3.5))

handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for _n, _p, _l, colour in CELLS]
fig.legend(handles, [label.replace("\n", " ") for _n, _p, label, _c in CELLS], loc="upper center",
           ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.902))

med_lines = []
for kind, ids in (("slot", s1.slot), ("concept", s1.concept)):
    for name, prompts, _label, _c in CELLS:
        l = [s1.lz("instruct", u, prompts) for u in ids]
        r = [s1.rz("instruct", u, prompts) for u in ids]
        med_lines.append(f"{kind} {name}: median L_z {sc.median(l):+.2f}, median R_z {sc.median(r):+.2f}")
for i, line in enumerate(med_lines):
    fig.text(0.012, 0.012 + (5 - i) * 0.013, line, fontsize=8.0, ha="left", color="#444444")
fig.text(0.012, 0.090, "Group medians, cell by cell (dashed lines in the panels):", fontsize=8.2, ha="left",
         color="#222222", weight="bold")

fig.savefig(OUT, dpi=150, facecolor="white")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
for line in med_lines:
    print(line)
