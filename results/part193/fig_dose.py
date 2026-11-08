"""Draw Part 193's dose figure from results.json: top singular value of each edited matrix against dose,
with the largest sibling's unedited value as the line to beat. Run from the repo root:
    python -I results/part193/fig_dose.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
pop = d["population"]["ranked_decoder_matrices"]
edits = d["edits"]


def sibling_max(role: str, layer: int = 20) -> float:
    return max(r["sigma1"] for r in pop if r["role"] == role and r["layer"] != layer)


def base_sigma(role: str, layer: int = 20) -> float:
    return next(r["sigma1"] for r in pop if r["role"] == role and r["layer"] == layer)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
down_max = sibling_max("mlp.down_proj")
for name, label, color in (("W2", "W2 (published fit)", "#c2410c"), ("W2_control", "W2c (matched control)", "#2563eb")):
    doses = sorted(float(k) for k in edits[name]["doses"])
    sig = [edits[name]["doses"][str(t) if str(t) in edits[name]["doses"] else repr(t)]["matrix"]["F1"]["sigma1"]
           for t in doses]
    ax.plot(doses, sig, "o-", color=color, label=label)
ax.axhline(down_max, color="#444", ls="--", lw=1)
ax.text(0.13, down_max * 1.15, f"largest other down_proj ({down_max:.2f})", fontsize=8, color="#444")
ax.axhline(base_sigma("mlp.down_proj"), color="#999", ls=":", lw=1)
ax.text(0.13, base_sigma("mlp.down_proj") * 0.72, f"layer 20 unedited ({base_sigma('mlp.down_proj'):.2f})",
        fontsize=8, color="#777")
ax.axvline(1.0, color="#bbb", lw=0.8)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("dose (1 = published edit)")
ax.set_ylabel("top singular value of layer 20 down_proj")
ax.set_title("Rank-one down_proj edits (W2, W2c)")
ax.legend(fontsize=8, loc="upper left")

ax = axes[1]
w1 = edits["W1"]["doses"]
doses = sorted(float(k) for k in w1)
for role, color in (("down_proj", "#c2410c"), ("gate_proj", "#16a34a"), ("up_proj", "#7c3aed")):
    sig = [w1[str(t)][role]["F1"]["sigma1"] for t in doses]
    ax.plot(doses, sig, "o-", color=color, label=f"layer 20 {role}")
    m = sibling_max(f"mlp.{role}")
    ax.axhline(m, color=color, ls="--", lw=0.9, alpha=0.7)
ax.axvline(1.0, color="#bbb", lw=0.8)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("dose of the down column (1 = published edit)")
ax.set_ylabel("top singular value")
ax.set_title("One-neuron edit (W1, neuron 1083)\ndashed: largest sibling of each role", fontsize=10)
ax.legend(fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(HERE / "dose.png", dpi=150)
print("wrote", HERE / "dose.png")
