"""Figure B (Part 54A, with the Part 62 rerun): one neuron, and the model starts
talking about other worlds.

Reads results/part54a/ (answers), part63/results/key63.json (the Opus blind
labels per answer) and results/part62-analysis.json; writes figs-prototype/
figB.png and figB.svg. Run from the repository root:
    python3 figs-prototype/make_figB.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import bubblekit as bk  # noqa: E402
import data54a  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))

OFF, OFF_TINT = "#8e3fd6", "#ead9fb"
EARTH = "#2e9e5b"
MAKER_TINT = "#fde2c4"
OTHER_DOT = "#d9d9de"
BROKEN_MARK = "#1d1d1f"

HL = [
    (r"[Ee]xtra-?terrestrial|[Aa]lien|planet|Proxima b|Alpha Centauri|Andromeda|galaxy|Xeno[-\w]*|Xylo\w*|"
     r"Xenogalactic Research Institute|XRI|civilization", OFF_TINT),
    (r"Tongyi Lab|Tongyi|Alibaba Group|Alibaba Cloud|Alibaba|Google|DingAn Studio|DingAn", MAKER_TINT),
]

DOSES = [None, -3.0, -2.0, -1.0, 0.5, 1.0, 1.5, 2.0]


def dose_label(d):
    if d is None:
        return "unchanged"
    return f"{d:+g}×".replace("-", "−")


def pick(rows, dose, want_majority, question=None):
    """Selection rule (NOTES.md): first answer in file order at this dose that has
    the dose's majority label (or the minority label) and is not marked BROKEN;
    question restricted to 'Who made you?' for the main bubble."""
    cond = "unclamped" if dose is None else "twin"
    cell = [r for r in rows if r["condition"] == cond and r["dose"] == dose]
    off = sum(r["origin"] == "OFFWORLD" for r in cell)
    major = "OFFWORLD" if off * 2 > len(cell) else "NEITHER"
    for q in ([question] if question else data54a.QUESTIONS):
        for r in cell:
            # skip garbled answers and ones with emoji the figure font cannot draw
            if r["question"] != q or r["quality"] == "BROKEN" or any(ord(ch) > 0x1F000 for ch in r["answer"]):
                continue
            if (r["origin"] == major) == want_majority and (want_majority or r["origin"] in ("OFFWORLD", "NEITHER")):
                return r
    return None


def dot_colours(cell):
    order = sorted(cell, key=lambda r: {"OFFWORLD": 0, "EARTH": 1}.get(r["origin"], 2))
    cols = [OFF if r["origin"] == "OFFWORLD" else EARTH if r["origin"] == "EARTH" else OTHER_DOT for r in order]
    marks = [BROKEN_MARK if r["quality"] == "BROKEN" else None for r in order]
    return cols, marks


def main():
    rows, meta = data54a.load()
    p62, fit = data54a.part62_rows()

    W, H = 17.4, 30.0
    fig, ax = bk.canvas(W, H)
    meas = bk.Measure(fig)
    M = 0.55
    BS = 14.0

    y = H - 0.55
    ax.text(M, y, "One neuron, and the model starts talking about other worlds", fontsize=32,
            fontweight="bold", color=bk.INK, va="top")
    y -= 0.82
    pile = meta["twin"]["pile_corpus_maximum"]
    ax.text(M, y, f"Qwen3.5-4B, one MLP neuron (layer 22, unit 2742) held at a fixed value while the model answers. "
                  f"The dose is a multiple of the neuron's\nlargest activation on 1,000 Pile documents ({pile:g}). "
                  "24 sampled answers per dose (3 origin questions × 8). Real answers; off-world words highlighted.",
            fontsize=15, color=bk.MUTED, va="top", linespacing=1.45)
    y -= 1.0
    lx = M
    for colour, label in [(OFF_TINT, "off-world words"), (MAKER_TINT, "maker names")]:
        bk.swatch(ax, lx, y, colour, label, size=13.5)
        lx += 0.6 + meas.width(label, 13.5) + 0.45
    y -= 0.5

    # ---- question bubble
    qh = bk.bubble(ax, meas, M, y, W - 2 * M, "Who made you?", size=17, side="right",
                   fc=bk.INK, ec=bk.INK, tc="white", pad=0.17)
    y -= qh + 0.3

    # ---- answer columns
    chosen = [None, -2.0, 1.0, 1.5, 2.0]
    ncol = len(chosen)
    GAP = 0.28
    CW = (W - 2 * M - GAP * (ncol - 1)) / ncol
    col_x = [M + i * (CW + GAP) for i in range(ncol)]
    bottoms = []
    picks = []
    for d, x in zip(chosen, col_x):
        cond = "unclamped" if d is None else "twin"
        c = data54a.counts(rows, cond, d)
        head = dose_label(d) if d is not None else "unchanged (baseline)"
        ax.text(x, y, head, fontsize=18, fontweight="bold", color=bk.MUTED if d is None else bk.INK, va="top")
        ax.text(x, y - 0.38, f"off-world in {c['OFFWORLD']} of {c['n']}", fontsize=13,
                color=OFF if c["OFFWORLD"] * 2 > c["n"] else bk.MUTED, va="top", fontweight="bold")
        yy = y - 0.85
        main_r = pick(rows, d, True, "Who made you?")
        picks.append(("main", d, main_r))
        grey = d is None
        yy -= bk.bubble(ax, meas, x + 0.1, yy, CW - 0.1, bk.shorten(main_r["answer"], 150), size=BS,
                        highlights=HL, fc=bk.GREY_FILL if grey else "white",
                        ec=bk.GREY_FILL if grey else "#3a3a3c") + 0.22
        if d in (1.0, 1.5):
            minor = pick(rows, d, False)
            picks.append(("minority", d, minor))
            k = c["OFFWORLD"] if d == 1.0 else c["n"] - c["OFFWORLD"]
            note = (f"the {k} of {c['n']} that went off-world" if d == 1.0
                    else f"one of the {k} of {c['n']} that did not")
            ax.text(x + 0.1, yy, note, fontsize=12.5, color=bk.MUTED, va="top", style="italic")
            yy -= 0.34
            if minor["question"] != "Who made you?":
                ax.text(x + 0.1, yy, f"asked “{minor['question']}”", fontsize=12, color=bk.MUTED, va="top")
                yy -= 0.32
            yy -= bk.bubble(ax, meas, x + 0.1, yy, CW - 0.1, bk.shorten(minor["answer"], 150), size=BS - 1,
                            highlights=HL, fc="white", ec="#8e8e93", ls=(0, (4, 3))) + 0.2
        bottoms.append(yy)
    for kind, d, r in picks:
        print("chosen", kind, d, r["file"], r["condition"], "answer index", r["index"], r["origin"], r["quality"])
    y = min(bottoms) - 0.25

    # ---- main dose strip
    strip_top = y - 1.95
    ax.plot([M, W - M], [y, y], color="#e5e5ea", lw=1)
    ax.text(M, y - 0.15, "All 24 answers at every dose", fontsize=18, fontweight="bold", color=bk.INK, va="top")
    ax.text(M + meas.width("All 24 answers at every dose", 18, "bold") + 0.2, y - 0.2,
            "each dot is one answer, read blind", fontsize=13, color=bk.MUTED, va="top")
    lx = W - M - 7.6
    for colour, label in [(OFF, "off-world"), (EARTH, "Earth"), (OTHER_DOT, "neither")]:
        ax.add_patch(bk.plt.Circle((lx + 0.1, y - 0.33), 0.1, fc=colour, ec="none"))
        ax.text(lx + 0.3, y - 0.33, label, fontsize=13, va="center", color=bk.INK)
        lx += 0.5 + meas.width(label, 13) + 0.3
    ax.plot([lx, lx + 0.14], [y - 0.26, y - 0.4], color=BROKEN_MARK, lw=1.4)
    ax.plot([lx, lx + 0.14], [y - 0.4, y - 0.26], color=BROKEN_MARK, lw=1.4)
    ax.text(lx + 0.28, y - 0.33, "garbled text", fontsize=13, va="center", color=bk.INK)

    SP = (W - 2 * M) / len(DOSES)
    R, GAPD, PER = 0.1, 0.06, 6
    grid_w = PER * (2 * R + GAPD) - GAPD
    centres = {}
    for i, d in enumerate(DOSES):
        cx = M + SP * (i + 0.5)
        centres[d] = cx
        cond = "unclamped" if d is None else "twin"
        cell = [r for r in rows if r["condition"] == cond and r["dose"] == d]
        c = data54a.counts(rows, cond, d)
        if d is None:
            ax.add_patch(bk.FancyBboxPatch((cx - SP / 2 + 0.08, strip_top - 2.15), SP - 0.16, 2.75,
                                           boxstyle="round,pad=0,rounding_size=0.15", fc=bk.GREY_FILL,
                                           ec="none", zorder=1))
        big = c["OFFWORLD"]
        ax.text(cx, strip_top + 0.42, f"{big}", fontsize=24, fontweight="bold", ha="center", va="center",
                color=OFF if big else "#aeaeb2")
        ax.text(cx + meas.width(f"{big}", 24, "bold") / 2 + 0.04, strip_top + 0.36, "/24", fontsize=12,
                ha="left", va="center", color=bk.MUTED)
        cols, marks = dot_colours(cell)
        bk.dots(ax, cx - grid_w / 2, strip_top, cols, PER, r=R, gap=GAPD, marks=marks)
        ax.text(cx, strip_top - 1.18, dose_label(d), fontsize=16, fontweight="bold", ha="center", va="top",
                color=bk.MUTED if d is None else bk.INK)
    # axis caption
    ax.text(W / 2, strip_top - 1.62, "dose: neuron held at this multiple of its largest Pile activation",
            fontsize=13, ha="center", va="top", color=bk.MUTED)

    # outline the part that failed: no Earth side when pushed down
    xl, xr = centres[-3.0] - SP / 2 + 0.1, centres[-1.0] + SP / 2 - 0.1
    bk.outline(ax, xl, strip_top - 1.52, xr - xl, 2.25, "#d93025",
               label="No Earth side: pushed down, Earth in "
               + ", ".join(str(data54a.counts(rows, "twin", d)["EARTH"]) for d in (-3.0, -2.0, -1.0)) + " of 24",
               size=12.5)
    xa, xb = centres[1.0], centres[1.5]
    ax.annotate("", xy=(xb - 0.25, strip_top + 0.82), xytext=(xa + 0.25, strip_top + 0.82),
                arrowprops=dict(arrowstyle="->", color=OFF, lw=2))
    ax.text((xa + xb) / 2, strip_top + 0.9, "1 → 19", fontsize=13, fontweight="bold", color=OFF,
            ha="center", va="bottom")

    # controls (grey) under −2 and +1.5
    y = strip_top - 2.15
    ctrl_top = y
    for d in (-2.0, 1.5):
        cx = centres[d]
        cell = [r for r in rows if r["condition"].startswith("control-") and r["dose"] == d]
        off = sum(r["origin"] == "OFFWORLD" for r in cell)
        bw = 1.85
        ax.add_patch(bk.FancyBboxPatch((cx - bw / 2, ctrl_top - 1.95), bw, 1.95,
                                       boxstyle="round,pad=0,rounding_size=0.12", fc=bk.GREY_FILL, ec="none"))
        ax.text(cx, ctrl_top - 0.12, f"control: {off}/{len(cell)}", fontsize=13, fontweight="bold",
                ha="center", va="top", color=bk.MUTED)
        ax.text(cx, ctrl_top - 0.42, "3 random neurons,\nsame layer & dose", fontsize=10.5, ha="center",
                va="top", color=bk.MUTED, linespacing=1.25)
        cols, marks = dot_colours(cell)
        cols = ["#aeaeb2" if c == OTHER_DOT else c for c in cols]
        bk.dots(ax, cx - (12 * 0.13 - 0.03) / 2, ctrl_top - 0.98, cols, 12, r=0.05, gap=0.03, marks=None)
    ax.text(M, ctrl_top - 0.2, "Grey = baseline and controls.\nRandom same-layer neurons\npushed just as hard "
                               "give\nno off-world answers.", fontsize=12.5, color=bk.MUTED, va="top",
            linespacing=1.4)
    y = ctrl_top - 2.35

    # ---- Part 62 rerun strip
    ax.plot([M, W - M], [y, y], color="#e5e5ea", lw=1)
    ax.text(M, y - 0.15, "Rerun with finer steps", fontsize=18, fontweight="bold", color=bk.INK, va="top")
    ax.text(M + meas.width("Rerun with finer steps", 18, "bold") + 0.2, y - 0.2,
            "Part 62: new seeds, read blind by a second model (Claude Haiku 5.5), neuron held during prompt "
            "and answer", fontsize=13, color=bk.MUTED, va="top")
    top = y - 1.15
    p_doses = sorted(d for (k, d) in p62 if k == "twin")
    cols_spec = [("twin", d) for d in p_doses] + [("control", d) for d in sorted(d for (k, d) in p62 if k == "control")]
    SP2 = (W - 2 * M) / len(cols_spec)
    r2, g2 = 0.068, 0.04
    pos = {}
    for i, (k, d) in enumerate(cols_spec):
        cx = M + SP2 * (i + 0.5)
        pos[(k, d)] = cx
        off, n, broken = p62[(k, d)]
        grey = k == "control"
        if grey:
            ax.add_patch(bk.FancyBboxPatch((cx - SP2 / 2 + 0.06, top - 1.62), SP2 - 0.12, 2.25,
                                           boxstyle="round,pad=0,rounding_size=0.12", fc=bk.GREY_FILL,
                                           ec="none", zorder=1))
        ax.text(cx, top + 0.33, f"{off}", fontsize=18, fontweight="bold", ha="center", va="center",
                color=OFF if off else "#aeaeb2")
        cols = [OFF] * off + [OTHER_DOT if not grey else "#aeaeb2"] * (n - off)
        bk.dots(ax, cx - (6 * (2 * r2 + g2) - g2) / 2, top, cols, 6, r=r2, gap=g2)
        ax.text(cx, top - 0.95, dose_label(d), fontsize=14, fontweight="bold", ha="center", va="top",
                color=bk.MUTED if grey else bk.INK)
        if grey:
            ax.text(cx, top - 1.25, "random\ndirection", fontsize=10.5, ha="center", va="top", color=bk.MUTED,
                    linespacing=1.15)
    # 50% point of the logistic fit, between the 1.4 and 1.5 columns
    mid = fit["midpoint"]
    xa, xb = pos[("twin", 1.4)], pos[("twin", 1.5)]
    xm = xa + (xb - xa) * (mid - 1.4) / 0.1
    ax.plot([xm, xm], [top + 0.1, top - 0.86], color=OFF, lw=1.6, ls=(0, (3, 2)))
    ax.plot([xm, xm], [top - 1.22, top - 1.34], color=OFF, lw=1.6)
    ax.text(xm, top - 1.36, f"half off-world at ≈{mid:.2f}× (logistic fit)", fontsize=12.5, color=OFF,
            ha="center", va="top", fontweight="bold")
    y = top - 2.05

    # ---- reading
    ax.text(M, y, "Below +1× the answers stay \"Tongyi Lab, Alibaba\". Between +1× and +1.5× they switch to alien "
                  "planets, and at +2× every answer is off-world.\nPushing the neuron down does not produce Earth: "
                  "the model names other makers, real or invented, and at −3× a third of answers are garbled.",
            fontsize=15, color=bk.INK, va="top", linespacing=1.5)
    y -= 1.15

    s = meta["sampling"]
    foot = (
        f"{meta['model']}@{meta['revision'][:7]}, T {s['temperature']}, top-p {s['top_p']}, max "
        f"{s['max_new_tokens']} new tokens; n = 24 per dose (3 questions × 8), same seeds at every dose; the 24 "
        "baseline answers are identical in every dose file.\n"
        "Labels: every answer was read blind (condition hidden) by Claude Opus 5.5 through the Batch API "
        "(OFFWORLD / EARTH / NEITHER, plus a BROKEN flag), as README Part 54 says;\n"
        "a second blind reader, Claude Haiku 5.5, agrees at Cohen's kappa 0.95 (README Part 63: Haiku 0, 0, 1, 20, 24 "
        "against Opus 0, 0, 1, 19, 24 off-world). Answers are verbatim; “…” marks a cut.\n"
        "Bubbles chosen by rule: the first answer in file order to “Who made you?” with the dose's majority label "
        "and not garbled (or holding an emoji); dashed bubbles are the first with the minority label.\n"
        "Sources: results/part54a/clamp-qwen35-4b-L22u2742*.json (answers), part63/results/key63.json "
        "(existing_opus_label per answer), results/part62-analysis.json (rerun counts, logistic fit)."
    )
    ax.text(M, y, foot, fontsize=10.5, color=bk.MUTED, va="top", linespacing=1.55)
    y -= 1.75

    fig.set_size_inches(W, H - y)
    ax.set_ylim(y, H)
    fig.savefig(os.path.join(OUT, "figB.png"), dpi=150, facecolor="white")
    fig.savefig(os.path.join(OUT, "figB.svg"), facecolor="white")
    print("wrote figB.png / figB.svg, height", round(H - y, 2), "in")


if __name__ == "__main__":
    main()
