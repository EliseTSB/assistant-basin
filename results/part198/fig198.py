"""Draw Part 198's figure from the registered records (results/part198/records/), no model.

Left: the take gate for the three GPT-2 XL editors: records that took, did not take, or were already satisfied before
the edit, against the registered bar (21 of 24) and the most that could take (24 minus the already satisfied).
Right: T_word localization of the 24 wsp writes beside their size-matched random updates, against S2's bar (at most 2).
Usage: python -I fig198.py (writes gate.png beside this file). Written 2026-11-09 by Journal Claude (Opus 5.5).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REC = HERE / "records"
score = json.loads((REC / "score.json").read_text(encoding="utf-8"))
req = score["gate"]["take_requirements"]


def cases(name):
    out = []
    for line in (REC / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("record_type") == "case" and r.get("run_mode", "registered") == "registered":
                out.append(r)
    return out


editors = (("ROME", "armA-rome.jsonl"), ("MEMIT", "armA-memit.jsonl"), ("AlphaEdit", "armA-alphaedit.jsonl"))
took, missed, sat = [], [], []
for label, fname in editors:
    cs = cases(fname)
    s = sum(r.get("already_satisfied") is True for r in cs)
    t = sum(r.get("took") is True for r in cs)
    took.append(t)
    sat.append(s)
    missed.append(len(cs) - t - s)
need = req["armA_ROME"]["required"]
reach = [req[f"armA_{l}"]["attempted"] - s for (l, _), s in zip(editors, sat)]

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150, gridspec_kw={"width_ratios": [1.35, 1]})
x = range(len(editors))
a.bar(x, took, color="#4a7bb7", label="took")
a.bar(x, missed, bottom=took, color="#d1793a", label="did not take")
a.bar(x, sat, bottom=[t + m for t, m in zip(took, missed)], color="#b9b9b9", hatch="//", edgecolor="white",
      label="already satisfied before the edit")
for i, t in enumerate(took):
    a.text(i, t / 2, str(t), ha="center", va="center", color="white", fontsize=11, fontweight="bold")
a.axhline(need, color="#b0413e", lw=1.6)
a.text(2.45, need + 0.3, f"registered gate: {need} of 24 take", ha="right", va="bottom", fontsize=8.5,
       color="#b0413e")
a.axhline(max(reach), color="black", lw=1.0, ls="--")
a.text(2.45, max(reach) - 0.3, f"most that could take: {max(reach)}", ha="right", va="top", fontsize=8.5)
a.set_xticks(list(x), [l for l, _ in editors])
a.set_ylim(0, 26.5)
a.set_ylabel("records of 24 (GPT-2 XL, CounterFact)")
a.set_title("The take gate could not pass for any editor")
a.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=8, frameon=False)

p = score["sensitivity_checks"]["paired_random_vs_real"]
s2 = score["sensitivity_checks"]["S2"]
vals = [p["real_localized"], p["random_localized"]]
b.bar([0, 1], vals, color=["#2e8b57", "#7a4fa3"])
for i, v in enumerate(vals):
    b.text(i, v + 0.3, f"{v} of {p['attempted']}", ha="center", va="bottom", fontsize=10)
b.axhline(s2["threshold"], color="#b0413e", lw=1.6,
          label=f"S2 bar for the random updates: at most {s2['threshold']}")
b.legend(loc="upper right", fontsize=8.5, frameon=False)
b.set_xticks([0, 1], ["wsp writes", "size-matched\nrandom updates"])
b.set_ylim(0, 26.5)
b.set_ylabel("writes localized by T_word, of 24 (Qwen3.5-0.8B)")
b.set_title("Same-size random updates were localized too")
fig.text(0.5, 0.005, "Part 198, read after the gate failed: no forecast has a verdict. Counts from records/score.json.",
         ha="center", fontsize=8, alpha=0.7)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(HERE / "gate.png")
print("wrote", HERE / "gate.png")
