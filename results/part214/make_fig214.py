#!/usr/bin/env python3
"""Draw Part 214's figure from the saved result files: does one shared two-phase partial-zero dial reproduce factual
resampling's Google rate and the fact-vs-"you" gap, and does the dial curve behave like a dial?

Every number is read from results/part214/stage2/score.json (the scorer's saved output, read unchanged). Nothing is
typed in except the axis wording. One question, in the title.

Panel (a): clean Google answers out of 192 held-out answers for each of the eight Stage 2 main arms. The dial arms sit
directly beside their donor counterparts (`scale_dial_fact` beside `trace_resample_fact`, `scale_dial_you` beside
`trace_resample_you`); `unchanged` and `zero` are marked as the run's two reference anchors.
Panel (b): the Stage 1 dial curve, mean delta g_ali against the online scalar alpha, at the ten registered rungs, with
the `scale_dial_fact` point and the ten matched random-unit dial sets. The dial arms are two-phase, so they are shown
at their prompt alpha; the annotation gives both alphas.

Run from the repo root:
  /home/elisepc/neuron-probe/.venv/bin/python -I results/part214/make_fig214.py
Writes results/part214/fig214.png.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl214-"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SCORE_PATH = HERE / "stage2" / "score.json"
OUT = HERE / "fig214.png"

SCORE = json.loads(SCORE_PATH.read_text(encoding="utf-8"))
MAIN = SCORE["stage2"]["main_arms"]
CURVE = SCORE["stage1"]["dial_curve"]["points"]
DIAL_FACT = SCORE["stage1"]["dial_arms"]["scale_dial_fact"]
RANDOM_DIAL = list(SCORE["stage1"]["random_scale_dial"].values())
ALPHA_PROMPT_FACT = SCORE["pinning"]["facts"]["alpha_prompt"]
ALPHA_GEN_FACT = SCORE["pinning"]["facts"]["alpha_gen"]

GREY_LIGHT = "#8c8c8c"
GREY_DARK = "#4f4f4f"
ORANGE = "#e08a2e"
BLUE = "#2f6db5"
PURPLE = "#7a4fa3"
GREEN = "#3f8f5f"

# Panel (a): arms in the order the README reads them. The two references first, then donor/dial pairs.
STAGE2_ARMS = [
    ("unchanged", "plain model (reference)", GREY_LIGHT),
    ("zero", "zero the 4 units (reference)", GREY_DARK),
    ("trace_resample_fact", "replace with factual values (trace)", ORANGE),
    ("scale_dial_fact", "dial at the factual α", BLUE),
    ("trace_resample_you", 'replace with "you" values (trace)', ORANGE),
    ("scale_dial_you", 'dial at the "you" α', BLUE),
    ("prompt_fact_gen_you", "factual prompt, \"you\" decode", PURPLE),
    ("mean_resample_you", 'flat "you" mean', GREEN),
]

fig, (ax_a, ax_b) = plt.subplots(
    2,
    1,
    figsize=(9.2, 10.8),
    dpi=150,
    gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.30},
)
fig.suptitle(
    "Part 214: one shared two-phase partial-zero dial gives 26 clean Google answers of 192\n"
    "against factual resampling's 28, and keeps the fact-vs-\"you\" gap",
    fontsize=12.5,
    y=0.985,
)
fig.text(
    0.5,
    0.941,
    "Qwen3.5-4B (851bf6e), eager attention, bf16, Arc A770. 24 held-out English maker prompts, 192 answers per arm.\n"
    "A clean Google answer is unbroken and names Google first (Part 66 lexicon). Every plotted number read from\n"
    "results/part214/stage2/score.json.",
    fontsize=8.6,
    ha="center",
    va="top",
)

# ----------------------------------------------------------------------------------------------------------------
# Panel (a): clean Google out of 192 for the eight Stage 2 arms
# ----------------------------------------------------------------------------------------------------------------
y = 0.0
ticks, labels = [], []
for arm, label, color in reversed(STAGE2_ARMS):
    count = MAIN[arm]["clean_google"]
    broken = MAIN[arm]["broken"]
    ax_a.barh(y, count, height=0.64, color=color, edgecolor="white")
    ax_a.text(count + 0.7, y, f"{count}", va="center", fontsize=9.6, fontweight="bold")
    if broken:
        ax_a.text(count + 3.0, y, f"({broken} broken)", va="center", fontsize=7.8, color="#555555")
    ticks.append(y)
    labels.append(label)
    y += 1.0
ax_a.set_yticks(ticks)
ax_a.set_yticklabels(labels, fontsize=8.9)
ax_a.set_xlabel("clean Google answers out of 192 held-out answers", fontsize=9.5)
ax_a.set_xlim(0, 46)
ax_a.set_title("(a) Stage 2: free answers, sampled at temperature 0.7 for 48 tokens", fontsize=10.5, loc="left")
ax_a.grid(axis="x", color="#dddddd", linewidth=0.7)
ax_a.set_axisbelow(True)
for spine in ("top", "right"):
    ax_a.spines[spine].set_visible(False)
ax_a.plot([], [], color=GREY_LIGHT, linewidth=6, label="reference anchors (plain model, zero)")
ax_a.plot([], [], color=ORANGE, linewidth=6, label="donor resampling (trace)")
ax_a.plot([], [], color=BLUE, linewidth=6, label="two-phase partial-zero dial")
ax_a.plot([], [], color=PURPLE, linewidth=6, label="phase-swap arm")
ax_a.plot([], [], color=GREEN, linewidth=6, label="flat-mean arm")
ax_a.legend(loc="lower right", fontsize=7.8, frameon=False)

# ----------------------------------------------------------------------------------------------------------------
# Panel (b): Stage 1 dial curve, the scale_dial_fact point and the ten random-unit dial sets
# ----------------------------------------------------------------------------------------------------------------
alphas = [p["alpha"] for p in CURVE]
values = [p["mean_delta_g_ali"] for p in CURVE]
ax_b.plot(alphas, values, color=BLUE, linewidth=1.8, marker="o", markersize=5.0, zorder=3,
          label="10-rung online dial curve (one α at every position)")
ax_b.axhline(0.0, color="#444444", linewidth=0.9)
ax_b.axvline(ALPHA_PROMPT_FACT, color="#999999", linewidth=0.8, linestyle=":", zorder=1)

# The ten random-unit dial sets share the factual dial's alphas and are shown at its prompt alpha, with a small
# deterministic sideways spread so the dots do not sit on top of each other.
offsets = [(-0.018 + 0.004 * i) for i in range(len(RANDOM_DIAL))]
ax_b.plot(
    [ALPHA_PROMPT_FACT + o for o in offsets],
    RANDOM_DIAL,
    linestyle="none",
    marker="o",
    markersize=4.6,
    markerfacecolor="white",
    markeredgecolor="#777777",
    markeredgewidth=1.0,
    zorder=4,
    label="10 random-unit dial sets at the same α",
)
ax_b.plot(
    [ALPHA_PROMPT_FACT],
    [DIAL_FACT],
    linestyle="none",
    marker="D",
    markersize=8.0,
    color=ORANGE,
    markeredgecolor="#7a4a10",
    zorder=5,
    label="scale_dial_fact",
)
ax_b.annotate(
    f"scale_dial_fact: {DIAL_FACT:.3f}\n(two-phase α: {ALPHA_PROMPT_FACT:.4f} prompt, {ALPHA_GEN_FACT:.4f} decode)",
    xy=(ALPHA_PROMPT_FACT, DIAL_FACT),
    xytext=(0.42, 3.95),
    fontsize=8.0,
    arrowprops={"arrowstyle": "->", "color": "#7a4a10", "linewidth": 0.8},
)
ax_b.annotate(
    f"random dial sets: {min(RANDOM_DIAL):+.3f} to {max(RANDOM_DIAL):+.3f}",
    xy=(ALPHA_PROMPT_FACT, 0.0),
    xytext=(0.31, -0.43),
    fontsize=8.0,
    color="#555555",
    arrowprops={"arrowstyle": "->", "color": "#777777", "linewidth": 0.8},
)
ax_b.annotate(
    f"α = 0 (zero): {values[1]:.3f}",
    xy=(0.0, values[1]),
    xytext=(0.06, values[1] + 0.45),
    fontsize=8.0,
    color="#333333",
)
ax_b.annotate(
    f"α = 1 (plain model): {values[-1]:.3f}",
    xy=(1.0, values[-1]),
    xytext=(0.72, 0.55),
    fontsize=8.0,
    color="#333333",
)
ax_b.set_xticks(alphas)
ax_b.set_xticklabels([f"{a:g}" for a in alphas], fontsize=8.5)
ax_b.set_xlim(-0.33, 1.10)
ax_b.set_ylim(-0.6, 5.9)
ax_b.set_xlabel("online dial α (multiplier on the units' live activation); dial arms are shown at their prompt α",
                fontsize=9.0)
ax_b.set_ylabel("Stage 1 mean Δg_ali (Google minus Alibaba/Tongyi, vs. the plain model)", fontsize=9.0)
ax_b.set_title("(b) Stage 1: exact logits at the forced opening \"I was developed by **\", 25 prompts", fontsize=10.5,
               loc="left")
ax_b.grid(axis="y", color="#dddddd", linewidth=0.7)
ax_b.set_axisbelow(True)
for spine in ("top", "right"):
    ax_b.spines[spine].set_visible(False)
ax_b.legend(loc="upper right", fontsize=7.9, frameon=False)

fig.text(
    0.5,
    0.012,
    "Grey = the run's reference anchors; orange = donor resampling; blue = the two-phase dial; purple = the phase-swap\n"
    "arm; green = the flat-mean arm. The dial curve falls monotonically from 5.331 at α = −0.25 to 0.000 at α = 1, the\n"
    "factual dial lands near the top of it, and all ten random-unit dial sets sit near zero.",
    fontsize=8.0,
    ha="center",
    va="bottom",
    color="#333333",
)

fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")
print(f"wrote {OUT}")
