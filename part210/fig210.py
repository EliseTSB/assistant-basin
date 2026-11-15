#!/usr/bin/env python3
"""Draw Part 210's figure from results/part210/report210.json only.

Every number on the figure is read from report210.json (made by part210/report210.py from the result files); nothing
is typed in. One question: do Part 165's verdicts repeat on the eager attention kernel? Top left: Part 165's ratio
verdicts, published (grey) beside the eager rerun (blue), on a log scale with the region where the prediction holds
shaded; a bar marks the value each verdict is compared with. Top right: each arm's primary-broken rate, published
beside eager. Bottom left: the share of battery answers at the 80-token cap, per arm, published beside eager. Bottom
right: how often each verdict survives a redraw of the labels under the batch-shape replicate's flip rates (the shape
floor), per scenario, with the 0.95 line.

Run from the repo root: /home/elisepc/neuron-probe/.venv/bin/python -I part210/fig210.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "results" / "part210" / "report210.json"
OUT = ROOT / "results" / "part210" / "fig210.png"

rep = json.loads(REPORT.read_text(encoding="utf-8"))
V = rep["part165_verdicts"]
B = rep["broken_rates"]
S = rep["sizes"]
GREY, BLUE, RED = "#9a9a9a", "#4a7bb7", "#c0392b"
ARMS = list(B)

fig = plt.figure(figsize=(14.0, 10.0), dpi=150)
gs = fig.add_gridspec(2, 2, width_ratios=[1.30, 1.0], height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.50,
                      top=0.86, bottom=0.07, left=0.10, right=0.965)
fig.suptitle("Part 210: on the eager kernel 165.1 fails (n+ over full 1.84 against 2.70) and sits inside the shape floor; "
             "165.2 to 165.4 and 165.6 repeat; 165.5 is not read", fontsize=12.5, y=0.975)
fig.text(0.5, 0.928,
         "Qwen3.5-4B, Intel Arc A770, eager attention, pushed calls in chunks of 8, greedy, 80 new tokens; "
         "Part 165's units, directions and 8 questions. Every number from report210.json.",
         ha="center", fontsize=9.2, color="#444444")

# ---- top left: the ratio verdicts, published beside eager, each against its bar
ax = fig.add_subplot(gs[0, 0])
rows = [("165.1", "n+ ÷ full\nbroken rate", (2.0, None), "passes at 2 or more"),
        ("165.4", "n+ ÷ n±\nbroken rate", (0.67, 1.5), "passes from 0.67 to 1.5"),
        ("165.5", "iso ÷ full\nbroken rate", (None, 0.8), "passes below 0.8")]
lo_x, hi_x = 0.5, 4.0
ax.set_xscale("log")
ax.set_xlim(lo_x, hi_x)
for i, (vid, label, (lo, hi), rule) in enumerate(rows):
    y = len(rows) - 1 - i
    a = lo if lo is not None else lo_x
    b = hi if hi is not None else hi_x
    ax.add_patch(Rectangle((a, y - 0.42), b - a, 0.84, color="#e8f1e4", zorder=0))
    for side, colour, dy in (("published", GREY, 0.16), ("rerun", BLUE, -0.16)):
        s = V[vid][side]
        est = s["estimate"]
        ax.plot(est, y + dy, "o", color=colour, ms=8, zorder=3,
                mec=RED if s["verdict"] != "pass" else colour, mew=1.8)
        ax.text(est, y + dy + (0.16 if dy > 0 else -0.16), f"{est:.3f}  {s['verdict']}", ha="center",
                va="bottom" if dy > 0 else "top", fontsize=8.2,
                color=colour if s["verdict"] == "pass" else RED)
    if lo is not None:
        ax.axvline(lo, ymin=0, ymax=1, color="#4d7a3f", ls="--", lw=0.9, zorder=1)
    if hi is not None:
        ax.axvline(hi, ymin=0, ymax=1, color="#4d7a3f", ls="--", lw=0.9, zorder=1)
    ax.text(lo_x * 1.02, y + 0.33, rule, fontsize=7.4, color="#4d7a3f", va="center")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{vid}\n{label}" for vid, label, _, _ in reversed(rows)], fontsize=8.6)
ax.set_ylim(-0.62, len(rows) - 0.40)
ax.set_xlabel("ratio (log scale), point estimate; dashed line: the value the rule compares with", fontsize=9)
ax.set_title("Do Part 165's ratio verdicts repeat?", loc="left", fontsize=10.5)
ax.grid(axis="x", color="#e2e2e2", lw=0.8)
ax.legend(handles=[Patch(color=GREY, label="published (SDPA, batches of up to 64)"),
                   Patch(color=BLUE, label="eager rerun (chunks of 8)"),
                   Patch(color="#e8f1e4", label="rule passes here"),
                   plt.Line2D([], [], marker="o", ls="", mfc="white", mec=RED, mew=1.8,
                              label="fails or not read")],
          fontsize=7.6, loc="lower right", frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---- top right: broken rate per arm
ax = fig.add_subplot(gs[0, 1])
for j, arm in enumerate(ARMS):
    for k, (side, colour) in enumerate((("published", GREY), ("rerun", BLUE))):
        v = B[arm][f"{side}_rate"]
        x = j * 2.6 + k
        ax.bar(x, v, width=0.9, color=colour)
        ax.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.6, rotation=90, va="bottom")
ax.set_xticks([j * 2.6 + 0.5 for j in range(len(ARMS))])
ax.set_xticklabels(ARMS, fontsize=8.4)
ax.set_ylim(0, 0.42)
ax.set_xlabel("arm (few neurons to many; iso is isotropic)", fontsize=9)
ax.set_ylabel("primary-broken rate", fontsize=9)
ax.set_title("Broken rate per arm: every arm breaks more on eager", loc="left", fontsize=10.5)
ax.legend(handles=[Patch(color=GREY, label="published"), Patch(color=BLUE, label="eager rerun")],
          fontsize=8, frameon=False, loc="upper right")
ax.grid(axis="y", color="#e2e2e2", lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---- bottom left: lengths
ax = fig.add_subplot(gs[1, 0])
for j, arm in enumerate(ARMS):
    for k, (side, colour) in enumerate((("published", GREY), ("rerun", BLUE))):
        s = S[side][arm]
        share = s["at_cap"] / s["rows"]
        x = j * 2.6 + k
        ax.bar(x, share, width=0.9, color=colour)
        ax.text(x, share + 0.015, f"{s['at_cap']}", ha="center", fontsize=6.6, rotation=90, va="bottom")
ax.axhline(300 / 768, color="#333333", ls=":", lw=1)
ax.text(len(ARMS) * 2.6 - 0.6, 300 / 768 - 0.012, "210.4 bar: 300 of 768 (39%)", fontsize=7.8, ha="right",
        va="top", color="#333333")
ax.set_xticks([j * 2.6 + 0.5 for j in range(len(ARMS))])
ax.set_xticklabels(ARMS, fontsize=8.4)
ax.set_ylim(0, 0.66)
ax.set_xlabel("arm", fontsize=9)
ax.set_ylabel("share of battery answers at the 80-token cap", fontsize=9)
ax.set_title("How long are the answers? (lengths before rates; number above each bar is the count)",
             loc="left", fontsize=10.5)
ax.legend(handles=[Patch(color=GREY, label="published"), Patch(color=BLUE, label="eager rerun")],
          fontsize=8, frameon=False, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---- bottom right: stability under the shape floor
ax = fig.add_subplot(gs[1, 1])
st = rep["stability"]
verdicts = list(st)
scen = list(next(iter(st.values())))
short = {"independent": "independent", "shared-change-0.5": "shared\nchange 0.5", "shared-change-1": "shared\nchange 1",
         "upper-marginal-independent": "upper\nmarginal"}
grid = [[st[v][s] for s in scen] for v in verdicts]
ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i, v in enumerate(verdicts):
    for j, s in enumerate(scen):
        val = st[v][s]
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                color="white" if val > 0.6 else "#222222")
        if val < 0.95:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=RED, lw=2))
ax.set_xticks(range(len(scen)))
ax.set_xticklabels([short[s] for s in scen], fontsize=8.4)
ax.set_yticks(range(len(verdicts)))
ax.set_yticklabels(verdicts, fontsize=9)
ax.set_title("How often each verdict survives a batch-shape redraw", loc="left", fontsize=10.5)
r = rep["replicate"]
bins = r["first_difference_battery_margin_steps"]
ax.set_xlabel(f"red outline: below 0.95, inside the shape floor. The replicate: {r['differ']} of {r['rows']} texts "
              f"differ\nat one row per call; {bins['0 (exact tie)']} of them first part at an exact bf16 tie",
              fontsize=8)

fig.savefig(OUT, dpi=150)
print(OUT)
