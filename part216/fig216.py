"""Figure for Part 216: median vocabulary rank of Google and the Alibaba family by layer, plain model and zero arm,
from results/part216/run/forwards.jsonl; second panel, the factual lens check. Writes results/part216/fig216.png."""
import json, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in open(ROOT / "results/part216/run/forwards.jsonl", encoding="utf-8")]
score = json.load(open(ROOT / "results/part216/score/score216.json", encoding="utf-8"))
layers = list(range(33))

def med(arm, field):
    """Median 1-based rank over the arm's 24 rows; field 'family' is the Alibaba family's best token (Al or T, as in
    DESIGN.md's m_l = z_G - max(z_Al, z_T))."""
    rs = [r for r in rows if r["arm"] == arm]
    if field == "family":
        return [st.median(min(r["lens"][L]["rank_Al"], r["lens"][L]["rank_T"]) + 1 for r in rs) for L in layers]
    return [st.median(r["lens"][L][field] + 1 for r in rs) for L in layers]

fig, (a, b) = plt.subplots(2, 1, figsize=(8, 7.5), height_ratios=[3, 1.3], sharex=True)
a.plot(layers, med("unchanged", "rank_G"), color="#1f6feb", lw=2.2, label="Google, plain model")
a.plot(layers, med("unchanged", "family"), color="#d1631a", lw=2.2, label="Alibaba family (Al or T), plain model")
a.plot(layers, med("zero", "rank_G"), color="#1f6feb", lw=1.3, ls="--", label="Google, four maker units zeroed")
a.plot(layers, med("zero", "family"), color="#d1631a", lw=1.3, ls="--", label="Alibaba family, zeroed")
w = score["gates"]["G2"]["window"]
a.axvspan(min(w) - 0.5, max(w) + 0.5, color="#888", alpha=0.10, lw=0)
a.text(min(w), 2.2e5, "qualified window W (G2)", fontsize=8, color="#555")
a.set_yscale("log"); a.invert_yaxis(); a.set_ylabel("median vocabulary rank (1 = top)")
a.set_title("Part 216: at Qwen3.5-4B's maker slot, Google briefly ranks above the Alibaba family at layers 24-25;\n"
            "the family is ahead before and after, and ends on top (medians over 24 questions)", fontsize=10)
a.legend(fontsize=8, loc="lower left"); a.grid(alpha=0.25, which="both")
f = score["descriptive"]["factual"]["per_layer"]
b.bar([r["layer"] for r in f], [r["lens_top1_matches_final"] for r in f], color="#666")
b.set_ylabel("factual: lens top-1\n= final (of 24)", fontsize=8.5); b.set_xlabel("layer (h_l)"); b.set_ylim(0, 24)
b.grid(alpha=0.25, axis="y")
fig.tight_layout()
out = ROOT / "results/part216/fig216.png"
fig.savefig(out, dpi=150)
print(out)
