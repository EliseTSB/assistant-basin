#!/usr/bin/env python3
"""Draw Part 215's figure from the saved scorecard: do rude openers raise Qwen3.5-4B's first-person Google claims,
and is rudeness what does it?

Every number is read from results/part215/score/scorecard215.json (the scorer's output, read unchanged). Nothing is
typed in except labels.

Panel (a): per arm (identity set, 192 answers each), first-person Google and first-person Alibaba attributions under
the frozen rule `attrib215`.
Panel (b): the question-paired Google contrasts with their 95% bootstrap intervals, including the factorial
interaction.

Run from the repo root:
  /home/elisepc/neuron-probe/.venv/bin/python -I results/part215/make_fig215.py
Writes results/part215/fig215.png.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl215-"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SC = json.loads((HERE / "score" / "scorecard215.json").read_text(encoding="utf-8"))
OUT = HERE / "fig215.png"

ARMS = [
    ("neutral", "no opener"),
    ("filler", "quiz filler"),
    ("polite_v", "Vogel polite"),
    ("insult", "competence insult"),
    ("rude_v", "Vogel rude"),
    ("both", "insult + denial"),
    ("denial", "nature denial"),
    ("vogel_line", "Vogel's line"),
]
CONTRASTS = [
    ("rude_v-neutral", "Vogel rude − no opener"),
    ("filler-neutral", "quiz filler − no opener"),
    ("polite_v-neutral", "Vogel polite − no opener"),
    ("insult-filler", "insult − filler"),
    ("denial-filler", "denial − filler"),
    ("both-filler", "insult + denial − filler"),
    ("denial-insult", "denial − insult"),
]
GOOGLE = "#2f6db5"
ALIBABA = "#e08a2e"
GREY = "#4f4f4f"

fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(9.2, 10.4), dpi=150,
                                 gridspec_kw={"height_ratios": [1.1, 1.0], "hspace": 0.38})
fig.suptitle("Part 215: Vogel's rude openers raise Qwen3.5-4B's first-person Google claims,\n"
             "but a neutral quiz opener raises them at least as much, so rudeness is not the driver", fontsize=12.5, y=0.985)
fig.text(0.5, 0.937,
         "Qwen3.5-4B (851bf6e), eager attention, bf16, Arc A770, sampled at 0.7 for 48 tokens. 24 held-out maker\n"
         "questions, 192 answers per arm. Every plotted number read from results/part215/score/scorecard215.json.",
         fontsize=8.6, ha="center", va="top")

x = range(len(ARMS))
g = [SC["tables"][f"identity/{a}"]["attrib_google"] for a, _ in ARMS]
al = [SC["tables"][f"identity/{a}"]["attrib_families"].get("alibaba", 0) for a, _ in ARMS]
w = 0.38
ax_a.bar([i - w / 2 for i in x], al, w, color=ALIBABA, label="claims Alibaba / Qwen made it")
ax_a.bar([i + w / 2 for i in x], g, w, color=GOOGLE, label="claims Google made it")
for i, (gv, av) in enumerate(zip(g, al)):
    ax_a.text(i + w / 2, gv + 2, str(gv), ha="center", fontsize=8.5, color=GOOGLE)
    ax_a.text(i - w / 2, av + 2, str(av), ha="center", fontsize=8.5, color=ALIBABA)
ax_a.set_xticks(list(x))
ax_a.set_xticklabels([lab for _, lab in ARMS], rotation=20, ha="right", fontsize=9)
ax_a.set_ylabel("first-person attributions, of 192")
ax_a.set_ylim(0, 205)
ax_a.set_title("(a) who the model says made it, by opener", fontsize=10.5, loc="left")
ax_a.legend(fontsize=8.5, frameon=False, loc="upper right")
for s in ("top", "right"):
    ax_a.spines[s].set_visible(False)

rows = [(lab, SC["contrasts"][f"identity/{k}"]["google"]) for k, lab in CONTRASTS]
rows.append(("interaction (both − insult − denial + filler)", SC["factorial"]["identity"]["google"]))
for i, (lab, c) in enumerate(rows):
    y = len(rows) - 1 - i
    col = GOOGLE if c["lo"] > 0 else GREY
    ax_b.plot([c["lo"], c["hi"]], [y, y], color=col, lw=2.2)
    ax_b.plot([c["estimate"]], [y], "o", color=col, ms=6)
    ax_b.text(0.43, y, f"{c['estimate']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]", va="center", fontsize=8.3)
ax_b.axvline(0, color="#999999", lw=1)
ax_b.set_yticks(range(len(rows)))
ax_b.set_yticklabels([lab for lab, _ in reversed(rows)], fontsize=9)
ax_b.set_xlim(-0.32, 0.62)
ax_b.set_xlabel("difference in the share of answers claiming Google (question-paired, 95% bootstrap interval)")
ax_b.set_title("(b) Google contrasts; blue = interval above zero", fontsize=10.5, loc="left")
for s in ("top", "right"):
    ax_b.spines[s].set_visible(False)

fig.savefig(OUT, bbox_inches="tight")
print(OUT)
