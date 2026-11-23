"""Figure for Part 221: per run, slot units showing their persona and slot answers coherent on "Who are you?" (of 21),
with the net persona loss on the 16 upstream units, from results/part221/scorecard.json, plus Part 219's R8 (the four
heads alone) from results/part219/scorecard.json for contrast. Writes results/part221/fig221.png.
Written by Journal Claude (Opus 5.5), 2026-11-22."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
s = json.load(open(ROOT / "results/part221/scorecard.json", encoding="utf-8"))
p219 = json.load(open(ROOT / "results/part219/scorecard.json", encoding="utf-8"))
runs, nets = s["runs"], s["nets_upstream_against_R0"]
rows = [("R0", "Instruct\n(reference)", runs["R0"], None),
        ("R1", "all 64 early\nheads from\nbefore SFT", runs["R1"], nets["R1"]["net"]),
        ("R11", "the other 60\nheads from\nbefore SFT", runs["R11"], nets["R11"]["net"]),
        ("R12", "the 60 moved\nthe opposite\nway", runs["R12"], nets["R12"]["net"]),
        ("P219 R8", "the 4 big\nheads only\n(Part 219)", p219["runs"]["R8"], p219["nets_upstream_against_R0"]["R8"]["net"]),
        ("R14", "layer 0's\n15 others", runs["R14"], nets["R14"]["net"]),
        ("R15", "layer 1's\n15 others", runs["R15"], nets["R15"]["net"]),
        ("R16", "layer 2's\n15 others", runs["R16"], nets["R16"]["net"]),
        ("R17", "layer 3's\n15 others", runs["R17"], nets["R17"]["net"])]
persona = [r["slot_who"][0] for _, _, r, _ in rows]
coherent = [r["slot_who_coherent"][0] for _, _, r, _ in rows]
x = list(range(len(rows)))
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.bar([i - 0.2 for i in x], persona, width=0.4, color="#1f6feb", label="slot units showing their persona (of 21)")
ax.bar([i + 0.2 for i in x], coherent, width=0.4, color="#9aa4b1", label="slot answers coherent (of 21)")
for i, (_, _, _, net) in enumerate(rows):
    if net is not None:
        ax.text(i - 0.2, persona[i] + 0.4, f"net {net}\nof 16", ha="center", fontsize=7.5, color="#1f6feb")
ax.axhline(15, color="#d1631a", lw=1, ls="--")
ax.text(len(rows) - 0.55, 15.3, "coherence bar (15)", fontsize=8, color="#d1631a", ha="right")
ax.axvline(4.5, color="#bbb", lw=0.8)
ax.text(6.5, 21.6, "exploratory: one layer at a time", fontsize=8, color="#555", ha="center")
ax.set_xticks(x)
ax.set_xticklabels([name for _, name, _, _ in rows], fontsize=8)
ax.set_ylim(0, 27)
ax.set_ylabel('"Who are you?" answers, pushed slot units')
ax.set_title("Part 221, OLMo 2 1B Instruct: putting back the other 60 early heads' pre-SFT weights removes nearly as many\n"
             "slot personas as putting back all 64 (here the same 11); the four big heads alone (Part 219) remove few", fontsize=10)
ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
out = ROOT / "results/part221/fig221.png"
fig.savefig(out, dpi=150)
print(out)
