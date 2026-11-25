"""Figure for Part 218: for each checkpoint (SFT, DPO, Instruct) and layer (5, 10), the share of the moved directions
that each maker family leads across the dose ladder, with the number of moved directions at every dose and the read
rung marked, all from results218-v4/score218.json (keys models.<model>.slices.<layer>.<dose> and
models.<model>.read_rungs). Writes results218-v4/fig218.png.

The dose ordering the figure shows was not registered: only the marked read rung is read by 218.1, 218.2, 218.3 and
218.7. Drafted by a DeepSeek V4.1 Flash worker for Journal Claude, 2026-11-24; per-panel x labels and the tie
note fixed by Journal Claude after Vesper's cold read (a shared x axis printed Instruct layer 10's counts under
every column)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
SCORE = ROOT / "results218-v4/score218.json"
OUT = ROOT / "results218-v4/fig218.png"
s = json.load(open(SCORE, encoding="utf-8"))

MODELS = ["SFT", "DPO", "Instruct"]
LAYERS = ["5", "10"]
# the palette of the house figure scripts (part216, part221) plus two thin colours for the families that lead rarely
FAMILIES = [("OpenAI", "#1f6feb"), ("Google", "#d1631a"), ("Meta", "#2f855a"),
            ("Microsoft", "#7c3aed"), ("Other", "#9aa4b1")]


def read_rung_note(model: str, layer: str) -> str:
    """One line naming the read rung's led counts, straight from the slice record."""
    M = s["models"][model]
    rung = M["read_rungs"][layer]
    rec = M["slices"][layer][str(rung)]
    led = rec["led_counts"]
    top = max(led.values()) if led else 0
    parts = ", ".join(f"{fam} {n}" for fam, n in sorted(led.items(), key=lambda kv: -kv[1]) if n == top)
    if rec["leading_family"] is None:
        head = f"read rung {rung:g}: tie ({parts}) of {rec['moved']} moved"
    else:
        head = f"read rung {rung:g}: {rec['leading_family']} {led[rec['leading_family']]} of {rec['moved']} moved"
    return head


fig, axes = plt.subplots(len(MODELS), len(LAYERS), figsize=(11.4, 10.9), sharex=False, sharey=True)
for row, model in enumerate(MODELS):
    M = s["models"][model]
    doses = M["doses"]
    for col, layer in enumerate(LAYERS):
        ax = axes[row][col]
        moved_by_dose = []
        for i, dose in enumerate(doses):
            rec = M["slices"][layer][str(dose)]
            moved = rec["moved"]
            moved_by_dose.append(moved)
            if not moved:
                continue
            for fam, colour in FAMILIES:
                led = rec["led_counts"].get(fam, 0)
                if not led:
                    continue
                ax.plot([i], [led / moved], marker="o", ms=4.5, color=colour, lw=0)
        for fam, colour in FAMILIES:
            # a family that never leads in this panel keeps its zero share off the plot; OpenAI and Google,
            # the two families that lead almost every cell, are always drawn
            if fam not in ("OpenAI", "Google") and not any(
                    M["slices"][layer][str(dose)]["led_counts"].get(fam, 0) for dose in doses):
                continue
            xs, ys = [], []
            for i, dose in enumerate(doses):
                rec = M["slices"][layer][str(dose)]
                moved = rec["moved"]
                led = rec["led_counts"].get(fam, 0)
                xs.append(i)
                ys.append(led / moved if moved else None)
            # break the line where no direction moved: a share there is undefined, not zero
            ax.plot(xs, ys, color=colour, lw=2.0 if fam in ("OpenAI", "Google") else 1.1,
                    alpha=1.0 if fam in ("OpenAI", "Google") else 0.8)
        rung = M["read_rungs"][layer]
        xi = doses.index(rung)
        ax.axvline(xi, color="#555", lw=1, ls="--", zorder=0)
        ax.text(xi + 0.06, 1.045, "read rung", fontsize=7, color="#555", rotation=90, va="top")
        ax.set_title(f"{model}, layer {layer}\n{read_rung_note(model, layer)}", fontsize=9.2)
        ax.set_xticks(range(len(doses)))
        ax.set_xticklabels([f"{d:g}\nn={n}" for d, n in zip(doses, moved_by_dose)], fontsize=7.6)
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlim(-0.4, len(doses) - 0.6)
        ax.grid(alpha=0.25, axis="y")
        if col == 0:
            ax.set_ylabel("share of moved directions led", fontsize=8.6)
        if row == len(MODELS) - 1:
            ax.set_xlabel("dose d (second line: directions moved of 40)", fontsize=8.4)

handles = [Line2D([0], [0], color=c, lw=2.4 if f in ("OpenAI", "Google") else 1.4, label=f)
           for f, c in FAMILIES]
handles.append(Line2D([0], [0], color="#555", lw=1, ls="--", label="read rung (the registered read)"))
fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=8.4, frameon=False,
           bbox_to_anchor=(0.5, 0.94))
fig.suptitle("Part 218: which maker family leads the moved directions at OLMo 2 1B's maker slot, by dose, at the SFT,\n"
             "DPO and Instruct checkpoints and layers 5 and 10. The dose ordering is exploratory; only the read rung is read.",
             fontsize=10, y=0.995)
fig.text(0.5, 0.005,
         "A moved direction is led by its most frequent other_maker family; a tie is not led and is not drawn, so the "
         "family shares need not sum to 1.  n = moved directions of 40.",
         fontsize=7.4, color="#555", ha="center")
fig.tight_layout(rect=(0, 0.018, 1, 0.915))
fig.savefig(OUT, dpi=150)
print(OUT)
