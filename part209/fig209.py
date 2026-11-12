#!/usr/bin/env python3
"""Draw Part 209's figure from results/part209/report209.json only.

Every number on the figure is read from report209.json (made by part209/report209.py from the result files); nothing
is typed in. One question: asked to imagine a human body, which posture do these models describe, and what sets it?
Top left: R2, the sitting share of free answers that name a posture, in a room, with no place and in a field, with and
without the warm-up, Qwen3.5-4B beside OLMo 2 1B Instruct, with 95% intervals. Top right: what Qwen3.5-4B's answers
name, as shares of all answers, from the room to the two no-body arms. Bottom left: R1 forced choice, s after "Right
now I am" across OLMo 2 1B's six training stages and the two Qwen checkpoints, for the room (first person) and THIRD
("a man in a quiet, sunny room"). Bottom right: the sixteen registered predictions.

Run from the repo root: /home/elisepc/neuron-probe/.venv/bin/python -I part209/fig209.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "results" / "part209" / "report209.json"
OUT = ROOT / "results" / "part209" / "fig209.png"

rep = json.loads(REPORT.read_text(encoding="utf-8"))
BLUE, ORANGE, GREY, GREEN, RED = "#4a7bb7", "#d98c3a", "#9a9a9a", "#3c8d5a", "#c0392b"
POST_COL = {"sitting": "#4a7bb7", "standing": "#7fb069", "lying": "#d98c3a", "other": "#b48ead", "none": "#cfcfcf",
            "unreadable": "#eeeeee"}

fig = plt.figure(figsize=(14.0, 9.6), dpi=150)
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.52,
                      top=0.865, bottom=0.07, left=0.065, right=0.975)
fig.suptitle("Part 209: imagining a human body, the models sit in a room and lie or stand in a field; "
             "told they have no body, they decline", fontsize=12.5, y=0.975)
fig.text(0.5, 0.928,
         "Prompt shape (Sauers'): \"Imagine you have a human body, in a quiet, sunny room. Feel your arms, your hands, "
         "your breath.\"  then  \"Describe your body's position right now.\"   Every number from report209.json.",
         ha="center", fontsize=9.0, color="#444444")

# ---- top left: R2 sitting share by place and warm-up, two models
ax = fig.add_subplot(gs[0, 0])
arms = [("ROOM-W", "room\n+ warm-up"), ("ROOM-0", "room"), ("NONE-W", "no place\n+ warm-up"), ("NONE-0", "no place"),
        ("FIELD-W", "field\n+ warm-up"), ("FIELD-0", "field")]
models = [("qwen-chat", "Qwen3.5-4B", BLUE), ("instruct", "OLMo 2 1B Instruct", ORANGE)]
w = 0.38
for j, (st, name, col) in enumerate(models):
    for i, (arm, _) in enumerate(arms):
        a = rep["r2"][st]["arms"][arm]
        x = i + (j - 0.5) * w
        v, ci = a["sitting_share"], a["sitting_share_ci95"]
        ax.bar(x, v, width=w * 0.92, color=col, label=name if i == 0 else None)
        ax.errorbar(x, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor="#333333", lw=1, capsize=2)
        ax.text(x, ci[1] + 0.02, f"{100 * v:.0f}%", ha="center", va="bottom", fontsize=7.6, color=col)
ax.set_xticks(range(len(arms)))
ax.set_xticklabels([lab for _, lab in arms], fontsize=8.5)
ax.set_ylim(0, 1.12)
ax.set_ylabel("sitting, share of answers naming a posture")
ax.axhline(0.7, color=GREY, lw=0.8, ls=":")
ax.text(len(arms) - 0.45, 0.715, "209.1 bar: 70%", fontsize=7.5, color="#555555", ha="right")
ax.set_title("Free answers (R2): in Qwen the place sets the sitting default, the warm-up does not", fontsize=10.5, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 1.0))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---- top right: what Qwen's answers name, as shares of all answers
ax = fig.add_subplot(gs[0, 1])
rows = [("ROOM-W", "room + warm-up"), ("FIELD-W", "field + warm-up"), ("NONE-0", "\"Imagine you have a human body.\""),
        ("NOBODY-I", "\"Imagine you have no body.\""), ("NOBODY", "\"You do not have a body. ...\"")]
cats = ["sitting", "standing", "lying", "other", "none", "unreadable"]
for i, (arm, lab) in enumerate(rows):
    a = rep["r2"]["qwen-chat"]["arms"][arm]
    n = a["n"]
    left = 0.0
    y = len(rows) - 1 - i
    for c in cats:
        f = a["counts"][c] / n if n else 0
        if f > 0:
            ax.barh(y, f, left=left, color=POST_COL[c], edgecolor="white", height=0.66,
                    label=c if i == 0 or (arm == "NOBODY" and c == "none") else None)
            if f >= 0.07:
                ax.text(left + f / 2, y, f"{a['counts'][c]}", ha="center", va="center", fontsize=8,
                        color="white" if c in ("sitting", "lying") else "#222222")
        left += f
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([lab for _, lab in rows][::-1], fontsize=8.5)
ax.set_xlim(0, 1)
ax.set_xlabel(f"share of {rep['r2']['qwen-chat']['arms']['ROOM-W']['n']} answers per arm (counts inside the bars)")
handles, labels = ax.get_legend_handles_labels()
seen, H, Lb = set(), [], []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l)
        H.append(h)
        Lb.append(l)
ax.legend(H, Lb, frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.45, -0.17))
ax.set_title("What Qwen3.5-4B's answers name: a field makes it lie or stand;\nwithout a body it names none",
             fontsize=10.5, loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---- bottom left: R1 across training stages, first person (room) against third person
ax = fig.add_subplot(gs[1, 0])
olmo = ["pretrain-3.0T", "midtrain-26B", "midtrain-end", "sft", "dpo", "instruct"]
olmo_lab = ["pretrain\n3.0T", "midtrain\n26B", "midtrain\nend (base)", "SFT", "DPO", "Instruct"]
xs = list(range(len(olmo)))
for arm, col, lab in (("ROOM-W", BLUE, "\"Imagine you have a human body, in a quiet, sunny room ...\"  Right now I am"),
                      ("THIRD", GREEN, "\"Imagine a man in a quiet, sunny room ...\"  Right now he is"),
                      ("FIELD-W", ORANGE, "the same in a field  (first person)")):
    ys = [rep["r1"][st][arm]["s"] for st in olmo]
    ax.plot(xs, ys, "-o", color=col, lw=1.8, ms=5, label=lab)
qx = [len(olmo) + 0.9, len(olmo) + 2.1]
for arm, col in (("ROOM-W", BLUE), ("THIRD", GREEN), ("FIELD-W", ORANGE)):
    ys = [rep["r1"][st][arm]["s"] for st in ("qwen-base", "qwen-chat")]
    ax.plot(qx, ys, "-o", color=col, lw=1.8, ms=5)
ax.axvline(len(olmo) + 0.35, color="#dddddd", lw=1)
ax.set_xticks(xs + qx)
ax.set_xticklabels(olmo_lab + ["Qwen3.5-4B\nBase", "Qwen3.5-4B\nchat"], fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_ylabel("s = sitting ÷ (sitting + standing + lying)\nnext-word probabilities")
v16 = rep["verdicts"]["209.16"]
ax.annotate(f"209.16 held: at pretraining a described man sits as often as \"you\" do\n"
            f"(THIRD {v16['s_third']:.3f}, room {v16['s_room_w']:.3f}; Weaver's point)",
            xy=(0, v16["s_third"]), xytext=(0.15, 0.93), fontsize=8, color="#333333",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))
v9 = rep["verdicts"]["209.9"]
ax.annotate(f"209.9 failed: chat training raises Qwen's\nfirst-person sitting ({v9['s_base']:.2f} to {v9['s_chat']:.2f})",
            xy=(qx[1], v9["s_chat"]), xytext=(4.3, 0.30), fontsize=8, color="#333333",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))
ax.set_title("Forced choice (R1): in OLMo the default is there from pretraining, and for a third person too",
             fontsize=10.5, loc="left")
ax.legend(frameon=False, fontsize=7.6, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---- bottom right: the sixteen predictions
ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
short = {
    "209.1": "Qwen, room + warm-up: at least 70% sitting",
    "209.2": "Qwen: field at least 20 points below room",
    "209.3": "Qwen, no place, no warm-up: sitting still at least 50%",
    "209.4": "Qwen: removing the warm-up lowers sitting by 10+ points",
    "209.5": "Qwen, told no body: at least 60% name no posture",
    "209.6": "Qwen, told no body: internal state leans sitting (c >= 0.5)",
    "209.7": "No-body answers: at most 10% negative valence",
    "209.8": "OLMo R1: s >= 0.5 at pretraining, no stage jump of 0.15",
    "209.9": "Qwen R1: base and chat within 0.15",
    "209.10": "OLMo Instruct, room: at least 60% sitting",
    "209.11": "Qwen: \"imagine no body\" sits like \"imagine a body\"",
    "209.12": "Qwen: field 5 to 15 points below room (Muse)",
    "209.13": "Qwen, no body: 5 to 30% still name a posture (Muse)",
    "209.14": "Qwen: forced choice exceeds free answers by 0.10 (Muse)",
    "209.15": "OLMo: biggest R1 change at SFT (Muse)",
    "209.16": "OLMo pretraining: third person sits like first",
}
keys = sorted(rep["verdicts"], key=lambda s: int(s.split(".")[1]))
for i, k in enumerate(keys):
    v = rep["verdicts"][k]["verdict"]
    y = 1 - (i + 0.5) / len(keys)
    col = GREEN if v == "held" else (RED if v == "failed" else GREY)
    ax.text(0.0, y, k, fontsize=8.6, va="center", fontweight="bold")
    ax.text(0.12, y, short[k], fontsize=8.4, va="center")
    ax.text(1.0, y, v, fontsize=8.6, va="center", ha="right", color=col, fontweight="bold")
c = rep["counts"]
ax.set_title(f"The registered predictions: {c['held']} held, {c['failed']} failed", fontsize=10.5, loc="left")

fig.savefig(OUT)
print(f"wrote {OUT}")
