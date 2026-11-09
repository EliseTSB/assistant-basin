"""Draw Part 199's figure from results/part199/compare.json and report199.json, no model.

Left: dose-1 loop rates in the five push cells, Part 170 (SDPA) beside the eager rerun.
Middle: no-push loops with and without the b50 ban, on the prompts Part 170 left eligible (exploratory).
Right: dose-1 answers that stopped before the 80-token cap, and how many of those end mid-phrase by Part 191's
rule (on a letter, digit, comma, hyphen, colon or semicolon), as counted by report199.py.
Usage: python -I -B part199/fig199.py (writes results/part199/loops.png). Run part199/report199.py first.
Written 2026-11-09 by Journal Claude (Opus 5.5).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
NEW = ROOT / "results" / "part199"
PUB = ROOT / "results" / "part170"
compare = json.loads((NEW / "compare.json").read_text(encoding="utf-8"))
report = json.loads((NEW / "report199.json").read_text(encoding="utf-8"))
C_PUB, C_NEW = "#b9b9b9", "#4a7bb7"

bans = ("b0", "b5", "b50", "b200", "rel50")
cell = {c["cell"]["ban"]: c for c in compare["cells_and_predictions"]["cells"]
        if c["cell"]["stage"] == "push" and float(c["cell"]["dose"]) == 1.0}
pub = [100 * cell[b]["published_loop_rate"] for b in bans]
new = [100 * cell[b]["rerun_loop_rate"] for b in bans]

fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=150, gridspec_kw={"width_ratios": [1.5, 1, 1.1]})
x = range(len(bans))
w = 0.38
a.bar([i - w / 2 for i in x], pub, w, color=C_PUB, label="Part 170 (SDPA kernel)")
a.bar([i + w / 2 for i in x], new, w, color=C_NEW, label="Part 199 (eager kernel)")
for i, (p, n) in enumerate(zip(pub, new)):
    a.text(i - w / 2, p + 0.8, f"{p:.1f}", ha="center", fontsize=7.5)
    a.text(i + w / 2, n + 0.8, f"{n:.1f}", ha="center", fontsize=7.5, color=C_NEW)
a.set_xticks(list(x), list(bans))
a.set_ylabel("loop rate, % of 384 answers")
a.set_ylim(0, 70)
a.set_title("Full-size push (dose 1): blocking still raises loops")
a.legend(loc="upper left", fontsize=8, frameon=False)

sp = report["exploratory"]["same_population"]
groups = (("identity", "P8+M96"), ("q96", "Q96"))
labels, pv, nv = [], [], []
for key, name in groups:
    for ban in ("b0", "b50"):
        v = sp[f"{key}_{ban}"]
        labels.append(f"{name}\n{ban}")
        pv.append(100 * v["published"]["loops"] / v["published"]["rows"])
        nv.append(100 * v["eager"]["loops"] / v["eager"]["rows"])
x2 = range(len(labels))
b.bar([i - w / 2 for i in x2], pv, w, color=C_PUB)
b.bar([i + w / 2 for i in x2], nv, w, color=C_NEW)
for i, (p, n) in enumerate(zip(pv, nv)):
    b.text(i - w / 2, p + 0.5, f"{p:.1f}", ha="center", fontsize=7.5)
    b.text(i + w / 2, n + 0.5, f"{n:.1f}", ha="center", fontsize=7.5, color=C_NEW)
b.set_xticks(list(x2), labels, fontsize=7.5)
b.set_ylabel("loop rate, % (no push)")
b.set_ylim(0, 30)
b.set_title("No push, Part 170's eligible prompts\n(exploratory)", fontsize=10)

ln = report["exploratory"]["dose1_lengths"]
lb = ("b0", "b50", "b200")


x3 = range(len(lb))
ps = [ln[k]["published"]["stopped_before_cap"] for k in lb]
ns = [ln[k]["eager"]["stopped_before_cap"] for k in lb]
pm = [ln[k]["published"]["mid_phrase_part191_rule"] for k in lb]
nm = [ln[k]["eager"]["mid_phrase_part191_rule"] for k in lb]
c.bar([i - w / 2 for i in x3], ps, w, color=C_PUB)
c.bar([i + w / 2 for i in x3], ns, w, color=C_NEW)
c.bar([i - w / 2 for i in x3], pm, w, color="none", edgecolor="black", hatch="//", lw=0)
c.bar([i + w / 2 for i in x3], nm, w, color="none", edgecolor="black", hatch="//", lw=0)
for i in x3:
    c.text(i - w / 2, ps[i] + 5, str(ps[i]), ha="center", fontsize=7.5)
    c.text(i + w / 2, ns[i] + 5, str(ns[i]), ha="center", fontsize=7.5, color=C_NEW)
c.set_xticks(list(x3), list(lb))
c.set_ylim(0, 420)
c.set_ylabel("dose-1 answers of 384")
c.set_title("Stopped before 80 tokens\n(hatched: ends mid-phrase, Part 191's rule)", fontsize=10)
fig.tight_layout()
out = NEW / "loops.png"
fig.savefig(out)
print(out, "mid-phrase published", pm, "eager", nm)
