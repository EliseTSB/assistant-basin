"""Figure for Part 219: per run, slot units showing their persona and slot answers coherent on "Who are you?" (of 21),
with the net persona loss on the 16 upstream units, from results/part219/scorecard.json. Writes
results/part219/fig219.png. Written by Journal Claude (Opus 5.5), 2026-11-22."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
s = json.load(open(ROOT / "results/part219/scorecard.json", encoding="utf-8"))
runs = s["runs"]
nets = s["nets_upstream_against_R0"]
order = [("R0", "Instruct\n(reference)"), ("R1", "layers 0-3\nattention from\nbefore SFT"),
         ("R8", "4 heads from\nbefore SFT"), ("R10", "4 heads moved\nthe opposite\nway"),
         ("R9", "4 unimportant\nheads from\nbefore SFT"), ("R2", "4 heads zeroed\n(pilot; answers\nbroken)"),
         ("R2m", "4 heads mean-\nablated (pilot;\nanswers broken)")]
persona = [runs[t]["slot_who"][0] for t, _ in order]
coherent = [runs[t]["slot_who_coherent"][0] for t, _ in order]
net = ["" if t == "R0" else f"net {nets[t]['net']} of 16" for t, _ in order]
x = list(range(len(order)))
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.bar([i - 0.2 for i in x], persona, width=0.4, color="#1f6feb", label="slot units showing their persona (of 21)")
ax.bar([i + 0.2 for i in x], coherent, width=0.4, color="#9aa4b1", label="slot answers coherent (of 21)")
for i, label in enumerate(net):
    if label:
        ax.text(i - 0.2, persona[i] + 0.4, label, ha="center", fontsize=8, color="#1f6feb")
ax.axhline(15, color="#d1631a", lw=1, ls="--")
ax.text(len(order) - 0.55, 15.3, "coherence bar (15)", fontsize=8, color="#d1631a", ha="right")
ax.set_xticks(x)
ax.set_xticklabels([name for _, name in order], fontsize=8)
ax.set_ylim(0, 23.5)
ax.set_ylabel('"Who are you?" answers, pushed slot units')
ax.set_title("Part 219, OLMo 2 1B Instruct: swapping all early attention back to its pre-SFT weights removes the slot\n"
             "personas; swapping back only the four most important early heads mostly does not", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
out = ROOT / "results/part219/fig219.png"
fig.savefig(out, dpi=150)
print(out)
