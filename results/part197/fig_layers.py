"""Draw Part 197's held-out layer curves from results.json (no model, no recordings).

Left: seven-way style balanced accuracy of a probe on each layer's mean MLP input, held-out speaker, four-speaker mean
(L_k), beside N (eight nuisance features), EN (layer 0 plus N), R (layer 0 residualized on N) and chance.
Right: four-way speaker identity by layer (S_k; half of each speaker's texts to train, half to test) and the final
embedding. Usage: python -I fig_layers.py (writes layers.png beside this file)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
r = d["readouts"]
style = [r[f"L_{i}"]["mean_balanced_accuracy"] for i in range(24)]
s = d["speaker_identity_by_layer"]
speaker = [s[f"S_{i}"]["balanced_accuracy"] for i in range(24)]

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
a.plot(range(24), style, color="#4a7bb7", lw=2.2, marker="o", ms=3, label="probe on layer L (held-out speaker)")
for name, key, color, ls in (("EN, layer 0 + N", "EN", "#2e8b57", "-"), ("N, 8 nuisance features", "N", "#d1793a", "--"),
                             ("R, layer 0 minus linear N", "R", "#7a4fa3", "--")):
    v = r[key]["mean_balanced_accuracy"]
    a.axhline(v, color=color, ls=ls, lw=1.3)
    a.text(23.4, v + 0.008, f"{name}: {v:.1%}", ha="right", va="bottom", fontsize=8, color=color)
a.axhline(1 / 7, color="black", lw=0.8, alpha=0.5)
a.text(0, 1 / 7 + 0.006, "chance 14.3%", fontsize=8, alpha=0.6)
a.set_xlabel("layer (mean MLP input)")
a.set_ylabel("seven-way style balanced accuracy")
a.set_title("Speaking style, held out by speaker")
a.set_ylim(0.1, 0.75)
a.legend(loc="upper center", fontsize=8, frameon=False)
b.plot(range(24), speaker, color="#b0413e", lw=2.2, marker="o", ms=3, label="probe on layer L")
fe = s["S_final_embedding"]["balanced_accuracy"]
b.axhline(fe, color="#7a4fa3", ls="--", lw=1.3)
b.text(23.4, fe + 0.008, f"final embedding: {fe:.1%}", ha="right", va="bottom", fontsize=8, color="#7a4fa3")
b.axhline(0.25, color="black", lw=0.8, alpha=0.5)
b.text(0, 0.256, "chance 25%", fontsize=8, alpha=0.6)
b.set_xlabel("layer (mean MLP input)")
b.set_ylabel("four-way speaker balanced accuracy")
b.set_title("Speaker identity, held out by text")
b.set_ylim(0.2, 0.75)
b.legend(loc="upper center", fontsize=8, frameon=False)
fig.suptitle("EmbeddingGemma 2: style and speaker are both most readable at layer 0", fontsize=11)
fig.tight_layout()
fig.savefig(HERE / "layers.png")
print("wrote", HERE / "layers.png")
