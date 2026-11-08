"""Draw Part 196's layer curve from results.json (no model, no recordings).

Inner leave-one-speaker-out balanced accuracy of a probe on each layer's mean MLP input, computed on each fold's three
training speakers (the numbers E used to pick its layer), beside the held-out-speaker four-fold means of E (layer 0),
D (final embedding), H (acoustic features) and T (transcript only), and chance (1/7).
Usage: python -I fig_layers.py  (writes layers.png beside this file)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
folds = d["folds"]
curves = [folds[s]["E"]["inner_leave_one_speaker_out_balanced_accuracy"] for s in sorted(folds)]
mean = [sum(c[i] for c in curves) / len(curves) for i in range(len(curves[0]))]
r = d["readouts"]

fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
for c in curves:
    ax.plot(range(len(c)), c, color="#4a7bb7", alpha=0.25, lw=1)
ax.plot(range(len(mean)), mean, color="#4a7bb7", lw=2.2, marker="o", ms=3,
        label="probe on layer L (inner held-out speaker, training folds)")
marks = [("E, layer 0, held-out", r["E"], "#4a7bb7", "-"), ("H, acoustic features", r["H"], "#d1793a", "--"),
         ("D, final embedding", r["D"], "#7a4fa3", "--"), ("T, transcript only", r["T"], "#6f6f6f", ":")]
for name, ro, color, ls in marks:
    v = ro["mean_balanced_accuracy"]
    ax.axhline(v, color=color, ls=ls, lw=1.4)
    ax.text(23.4, v + 0.008, f"{name}: {v:.1%}", ha="right", va="bottom", fontsize=8, color=color)
ax.axhline(1 / 7, color="black", lw=0.8, alpha=0.5)
ax.text(0, 1 / 7 + 0.006, "chance 14.3%", fontsize=8, alpha=0.6)
ax.set_xlabel("layer of EmbeddingGemma 2 (mean MLP input)")
ax.set_ylabel("seven-way balanced accuracy")
ax.set_title("Speaking style is easiest to read at layer 0 and fades toward the final embedding")
ax.set_xlim(-0.5, 23.5)
ax.set_ylim(0.1, 0.75)
ax.legend(loc="upper center", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(HERE / "layers.png")
print("wrote", HERE / "layers.png")
