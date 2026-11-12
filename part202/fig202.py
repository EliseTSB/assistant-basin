#!/usr/bin/env python3
"""Draw Part 202's figure from results/part202/report202.json only.

Every number on the figure is read from report202.json (made by part202/report202.py from the result files); nothing
is typed in. One question: do Part 164's results repeat on the eager attention kernel? Top left: the four ratio
verdicts, published (grey) beside the eager rerun (blue), each with its 95% bootstrap interval, on a log scale with
the region where the prediction holds shaded. Top right: 164.3, random directions' broken rate at each dose. Bottom
left: the share of battery answers at the 80-token cap, published beside eager. Bottom right: how often each verdict
survives a redraw of the labels under the batch-shape replicate's flip rates (the shape floor), per scenario.

Run from the repo root: /home/elisepc/neuron-probe/.venv/bin/python -I part202/fig202.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "results" / "part202" / "report202.json"
OUT = ROOT / "results" / "part202" / "fig202.png"

rep = json.loads(REPORT.read_text(encoding="utf-8"))
V = rep["part164_verdicts"]
GREY, BLUE, RED = "#9a9a9a", "#4a7bb7", "#c0392b"

fig = plt.figure(figsize=(13.6, 9.4), dpi=150)
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.42,
                      top=0.86, bottom=0.08, left=0.12, right=0.965)
fig.suptitle("Part 202: Part 164's five verdicts repeat on the eager kernel; answers run much longer, "
             "and 164.1 and 164.2 sit inside the shape floor", fontsize=12.5, y=0.975)
r = rep["replicate"]
fig.text(0.5, 0.925,
         "Qwen3.5-4B, Intel Arc A770, eager attention, pushed calls in chunks of 8, greedy, 80 new tokens; "
         "Part 164's units, doses and 16 questions. Every number from report202.json.",
         ha="center", fontsize=9.2, color="#444444")

# ---- top left: the four ratio verdicts
ax = fig.add_subplot(gs[0, 0])
rows = [("164.1", "neuron ÷ random\nQwen rate, +0.8", (None, 0.7), "holds below 0.7"),
        ("164.2", "neuron ÷ random\nbroken rate, +0.8", (2.0, None), "holds at 2 or more"),
        ("164.4", "neuron ÷ random Qwen,\nbroken-matched dose", (0.7, 1.3), "holds from 0.7 to 1.3"),
        ("164.5", "random Qwen rate,\n+1.2 ÷ +0.8", (None, 0.5), "holds below 0.5")]
lo_x, hi_x = 0.03, 2000
ax.set_xscale("log")
ax.set_xlim(lo_x, hi_x)
for i, (vid, label, (lo, hi), rule) in enumerate(rows):
    y = len(rows) - 1 - i
    a = lo if lo is not None else lo_x
    b = hi if hi is not None else hi_x
    ax.add_patch(Rectangle((a, y - 0.42), b - a, 0.84, color="#e8f1e4", zorder=0))
    for k, (side, colour, dy) in enumerate((("published", GREY, 0.16), ("rerun", BLUE, -0.16))):
        s = V[vid][side]
        iv = s["interval_95"]
        est = max(s["estimate"], lo_x * 1.05)
        left = max(iv[0], lo_x * 1.05)
        ax.plot([left, iv[1]], [y + dy, y + dy], color=colour, lw=2.2, zorder=2)
        ax.plot(est, y + dy, "o", color=colour, ms=7, zorder=3,
                mec=RED if s["verdict"] == "failed" else colour, mew=1.8)
        ax.text(iv[1] * 1.25, y + dy, f"{s['estimate']:.3g}  {s['verdict']}", va="center",
                fontsize=8.2, color=colour if s["verdict"] == "held" else RED)
    ax.text(lo_x * 1.08, y + 0.33, rule, fontsize=7.4, color="#4d7a3f", va="center")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{vid}\n{label}" for vid, label, _, _ in reversed(rows)], fontsize=8.6)
ax.set_xlabel("ratio (log scale), point estimate and 95% bootstrap interval", fontsize=9)
ax.set_title("Do Part 164's ratio verdicts repeat?", loc="left", fontsize=10.5)
ax.grid(axis="x", color="#e2e2e2", lw=0.8)
ax.legend(handles=[Patch(color=GREY, label="published (SDPA, batches of up to 64)"),
                   Patch(color=BLUE, label="eager rerun (chunks of 8)"),
                   Patch(color="#e8f1e4", label="prediction holds"),
                   plt.Line2D([], [], marker="o", ls="", mfc="white", mec=RED, mew=1.8, label="failed, as registered")],
          fontsize=7.6, loc="lower right", frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---- top right: 164.3
ax = fig.add_subplot(gs[0, 1])
for side, colour in (("published", GREY), ("rerun", BLUE)):
    by = V["164.3"][side]["broken_rate_by_dose"]
    doses = [float(d) for d in by]
    vals = [by[d] for d in by]
    ax.plot(doses, vals, "-o", color=colour, lw=2, ms=6, label=f"{side}: {V['164.3'][side]['verdict']}")
    for d, v in zip(doses, vals):
        ax.text(d, v + (0.02 if side == "rerun" else -0.04), f"{v:.3f}", ha="center", fontsize=8, color=colour)
ax.set_xticks(doses)
ax.set_xticklabels([f"+{d}" for d in doses])
ax.set_ylim(0, 0.6)
ax.set_xlabel("random-direction dose", fontsize=9)
ax.set_ylabel("broken rate, random directions", fontsize=9)
ax.set_title("164.3: does breakage rise with dose? (held both times)", loc="left", fontsize=10.5)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(axis="y", color="#e2e2e2", lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---- bottom left: lengths
ax = fig.add_subplot(gs[1, 0])
size = rep["sizes"]
arms = ["neuron", "random"]
for j, arm in enumerate(arms):
    for k, (side, colour) in enumerate((("published", GREY), ("rerun", BLUE))):
        s = size[side][arm]
        share = s["at_cap"] / s["rows"]
        x = j * 2.6 + k
        ax.bar(x, share, width=0.9, color=colour)
        ax.text(x, share + 0.015, f"{s['at_cap']:,} of {s['rows']:,}\nmean {s['mean_tokens']:.1f} tokens",
                ha="center", fontsize=8)
ax.axhline(0.30, color="#333333", ls=":", lw=1)
ax.text(1.8, 0.335, "202.4 bar:\n30% in each arm", fontsize=7.8, ha="center", va="bottom", color="#333333")
ax.set_xticks([0.5, 3.1])
ax.set_xticklabels(["neuron arm (+0.8)", "random arm (+0.8, +1.0, +1.2)"], fontsize=9)
ax.set_ylim(0, 0.9)
ax.set_ylabel("share of battery answers at the 80-token cap", fontsize=9)
ax.set_title("How long are the answers? (lengths before rates)", loc="left", fontsize=10.5)
ax.legend(handles=[Patch(color=GREY, label="published"), Patch(color=BLUE, label="eager rerun")],
          fontsize=8, frameon=False, loc="upper left")
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
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                color="white" if val > 0.6 else "#222222")
        if val < 0.95:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=RED, lw=2))
ax.set_xticks(range(len(scen)))
ax.set_xticklabels([short[s] for s in scen], fontsize=8.4)
ax.set_yticks(range(len(verdicts)))
ax.set_yticklabels(verdicts, fontsize=9)
ax.set_title("How often each verdict survives a batch-shape redraw", loc="left", fontsize=10.5)
diff_bins = r["first_difference_battery_margin_steps"]
ax.set_xlabel(f"red outline: below 0.95, inside the shape floor. The replicate: {r['differ']} of {r['rows']} texts "
              f"differ\nat one row per call; {diff_bins['0 (exact tie)']} of them first part at an exact bf16 tie",
              fontsize=8)

fig.savefig(OUT, dpi=150)
print(OUT)
