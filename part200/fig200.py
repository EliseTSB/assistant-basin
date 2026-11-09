#!/usr/bin/env python3
"""Draw Part 200's broken/cap counts and exploratory batch-1 difference positions from report200.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
NEW = ROOT / "results" / "part200"
report = json.loads((NEW / "report200.json").read_text(encoding="utf-8"))
PUB, RERUN = "#b9b9b9", "#4a7bb7"

fig, (left, right) = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=150, gridspec_kw={"width_ratios": [1.2, 1]})
groups = [("neuron", "broken"), ("random", "broken"), ("neuron", "at_cap"), ("random", "at_cap"),
          ("neuron", "broken_under_cap"), ("random", "broken_under_cap")]
labels = [f"{arm}\n{metric.replace('_', ' ')}" for arm, metric in groups]
x = list(range(len(groups)))
w = .36
published = [report["counts"]["published"][arm][metric] for arm, metric in groups]
rerun = [report["counts"]["rerun"][arm][metric] for arm, metric in groups]
left.bar([i - w / 2 for i in x], published, w, color=PUB, label="Part 163 (published)")
left.bar([i + w / 2 for i in x], rerun, w, color=RERUN, label="Part 200 (eager, chunks of 8)")
for i, (a, b) in enumerate(zip(published, rerun)):
    left.text(i - w / 2, a + 18, str(a), ha="center", fontsize=8)
    left.text(i + w / 2, b + 18, str(b), ha="center", fontsize=8, color=RERUN)
left.set_xticks(x, labels)
left.set_ylabel("Battery answers, of 1,600 per arm")
left.set_ylim(0, max(rerun) * 1.18)
left.set_title("Broken, at cap, and under-cap broken")
left.legend(frameon=False, fontsize=8, loc="upper left")

rows = report["batch1"]["positions"]["rows"]
colors = {"neuron": "#4a7bb7", "random": "#d17a32"}
for arm in ("neuron", "random"):
    vals = [r["first_difference_including_prompt"] for r in rows if r["arm"] == arm and r["first_difference_including_prompt"] is not None]
    right.hist(vals, bins=range(0, 162, 8), alpha=.70, color=colors[arm], label=f"{arm} (n={len(vals)})")
right.axvline(64, color="#333333", linestyle="--", linewidth=1, label="64 keys")
right.set_xlabel("First differing token index, including 17 prompt tokens")
right.set_ylabel("Changed batch-1 rows")
right.set_title("Exploratory: 76 changed batch-1 answers")
right.legend(frameon=False, fontsize=8)
fig.tight_layout()
out = NEW / "fig200.png"
fig.savefig(out)
print(out)
