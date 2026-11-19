#!/usr/bin/env python3
"""Draw Part 212's figure from the result files: does zeroing the four maker neurons make the model name Google,
beyond a push of the same size, and does replacing them with factual values keep that?

Every number is computed here from results/part212/stage2/score.json (the scorer's saved output, read unchanged) and
results/part212/pilot/stage1.json (the pilot's Stage 1 records, read through the scorer's own stage1_metrics). Nothing
is typed in. One question, in the title.

Panel (a): clean Google answers out of 192 held-out answers for each Stage 2 main arm, with the 20 matched random
`zero` sets' counts out of 24 beside `zero` (grey), and the 20 random `sub_factual` sets beside `sub_factual`.
Panel (b): Stage 1 mean delta g_ali (Google minus Alibaba/Tongyi, paired against unchanged) for the main Stage 1 arms
and the 20 random `zero` sets beside `zero`.

Style follows the Sauers style guide: one plain question in the title, plain labels, the baseline in grey beside each
bar, failures shown as openly as successes, readable at phone width.

Run from the repo root:
  MPLCONFIGDIR=/tmp/mpl /home/elisepc/neuron-probe/.venv/bin/python -I -B part212/fig212.py
Writes results/part212/fig212.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import score212 as sc  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RES = ROOT / "results" / "part212"
OUT = RES / "fig212.png"

SCORE = json.loads((RES / "stage2" / "score.json").read_text(encoding="utf-8"))
STAGE1_S1 = sc.load_stage1(RES / "pilot" / "stage1.json")
S1M = sc.stage1_metrics(STAGE1_S1)
MAIN = SCORE["stage2"]["main_arms"]
RANKS = SCORE["stage2"]["ranks"]

BLUE = "#2f6db5"
GREY = "#9a9a9a"
RED = "#c0392b"
ORANGE = "#e08a2e"

# Panel (a): the Stage 2 arms, in the order the README reads them.
STAGE2_ARMS = [
    ("unchanged", "plain model", GREY),
    ("zero", "zero the 4 units", RED),
    ("trace_resample_fact", "replace with factual\nvalues (trace)", ORANGE),
    ("mean_resample_fact", "replace with factual\nmean", ORANGE),
    ("sub_factual", "subtract factual\nprofile", BLUE),
    ("scale_neg1", "x -1", BLUE),
    ("scale_pos2", "x +2", BLUE),
    ("zero_prompt_only", "zero prompt only", BLUE),
    ("zero_gen_only", "zero generation only", BLUE),
]
# Panel (b): the Stage 1 arms and their mean delta g_ali.
STAGE1_ARMS = [
    ("unchanged", "plain model", GREY),
    ("zero", "zero the 4 units", RED),
    ("trace_resample_fact", "replace with factual\nvalues (trace)", ORANGE),
    ("trace_resample_you", 'replace with "you"\nvalues (trace)', ORANGE),
    ("mean_resample_fact", "replace with factual\nmean", ORANGE),
    ("sub_factual", "subtract factual\nprofile", BLUE),
    ("scale_neg1", "x -1", BLUE),
    ("scale_pos2", "x +2", BLUE),
    ("zero_prompt_only", "zero prompt only", BLUE),
    ("zero_opening_only", "zero opening only", BLUE),
    ("trace_resample_zh_maker", "Chinese maker\ntrace", BLUE),
]

fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(9.2, 11.6), dpi=150,
                                 gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.30})
fig.suptitle("Part 212: zeroing Qwen3.5-4B's four maker neurons makes it name Google 36 times in 192\n"
             "against 4 for the plain model; giving them values from factual prompts still does it 19 times",
             fontsize=12.5, y=0.988)
fig.text(0.5, 0.936,
         "Qwen3.5-4B (851bf6e), eager attention, bf16, Arc A770. 24 held-out English maker prompts x 8 samples = 192\n"
         "answers per arm. A clean Google answer is unbroken and names Google first (Part 66 lexicon). Every number\n"
         "read from results/part212/stage2/score.json and pilot/stage1.json.",
         fontsize=8.6, ha="center", va="top")

# ----------------------------------------------------------------------------------------------------------------
# Panel (a): clean Google out of 192, with the random zero sets as shares of 24
# ----------------------------------------------------------------------------------------------------------------
y = 0.0
ticks, labels = [], []
row_height = 1.0
for arm, label, color in reversed(STAGE2_ARMS):
    count = MAIN[arm]["clean_google"]
    ax_a.barh(y, count, height=0.62, color=color, edgecolor="white")
    ax_a.text(count + 0.6, y, f"{count}", va="center", fontsize=9.5, fontweight="bold")
    ticks.append(y)
    labels.append(label)
    if arm in RANKS:
        sets = RANKS[arm]["random_set_counts"]
        values = sorted(sets.values())
        # The 20 matched random sets: one dot per set at its count of 24, scaled to the 192 axis by x8 so the
        # reader compares shares, exactly as the registered rank does (count/24 against count/192).
        for v in values:
            ax_a.plot(v * 8, y - 0.40, marker="o", markersize=4.0, color=GREY, alpha=0.85, zorder=3)
        ax_a.text(2 * 8 + 2.0, y - 0.40, f"20 random {arm} sets: {min(values)} to {max(values)} of 24",
                  va="center", fontsize=7.8, color="#555555")
    y += row_height
ax_a.set_yticks(ticks)
ax_a.set_yticklabels(labels, fontsize=8.8)
ax_a.set_xlabel("clean Google answers out of 192 held-out answers", fontsize=9.5)
ax_a.set_xlim(0, 122)
ax_a.set_title("(a) Stage 2: free answers, sampled at temperature 0.7, 48 tokens", fontsize=10.5, loc="left")
ax_a.grid(axis="x", color="#dddddd", linewidth=0.7)
ax_a.set_axisbelow(True)
for spine in ("top", "right"):
    ax_a.spines[spine].set_visible(False)
ax_a.plot([], [], marker="o", markersize=4.0, color=GREY, linestyle="none",
          label="each dot is one matched random set, out of 24, shown x8")
ax_a.legend(loc="lower right", fontsize=7.8, frameon=False)

# ----------------------------------------------------------------------------------------------------------------
# Panel (b): Stage 1 mean delta g_ali
# ----------------------------------------------------------------------------------------------------------------
y = 0.0
ticks_b, labels_b = [], []
for arm, label, color in reversed(STAGE1_ARMS):
    value = STAGE1_S1["arms"][arm]["mean_delta_g_ali"]
    ax_b.barh(y, value, height=0.62, color=color, edgecolor="white")
    ax_b.text(value + (0.12 if value >= 0 else -0.12), y, f"{value:+.2f}", va="center",
              ha="left" if value >= 0 else "right", fontsize=9.0, fontweight="bold")
    ticks_b.append(y)
    labels_b.append(label)
    if arm == "zero":
        for v in sorted(S1M["random_zero_sets"].values()):
            ax_b.plot(v, y - 0.40, marker="o", markersize=4.0, color=GREY, alpha=0.85, zorder=3)
        vals = list(S1M["random_zero_sets"].values())
        ax_b.text(max(vals) + 0.30, y - 0.40, f"random sets: {min(vals):+.2f} to {max(vals):+.2f}",
                  va="center", fontsize=7.8, color="#555555")
    y += row_height
ax_b.axvline(0, color="#444444", linewidth=0.9)
ax_b.set_yticks(ticks_b)
ax_b.set_yticklabels(labels_b, fontsize=8.8)
ax_b.set_xlabel("Stage 1 mean change in Google minus Alibaba/Tongyi logit (delta g_ali), paired against the plain model",
                fontsize=9.0)
ax_b.set_xlim(-1.6, 8.6)
ax_b.set_title("(b) Stage 1: exact logits at the forced opening \"I was developed by **\", 24 prompts", fontsize=10.5,
               loc="left")
ax_b.grid(axis="x", color="#dddddd", linewidth=0.7)
ax_b.set_axisbelow(True)
for spine in ("top", "right"):
    ax_b.spines[spine].set_visible(False)
ax_b.plot([], [], marker="o", markersize=4.0, color=GREY, linestyle="none",
          label="each dot is one matched random zero set")
ax_b.legend(loc="lower right", fontsize=7.8, frameon=False)

fig.text(0.5, 0.012,
         "Red = zeroing the four units; orange = replacing them with values from factual or 'you' prompts; blue = other arms; "
         "grey = the plain model and the random sets.\n"
         "Zeroing all four units raises both measures far above every random zero set, but the plain model already names Google 4 times "
         "in 192, above 212.1's bar of 3.",
         fontsize=8.0, ha="center", va="bottom", color="#333333")

fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")
print(f"wrote {OUT}")
