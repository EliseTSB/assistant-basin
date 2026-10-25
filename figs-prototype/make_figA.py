"""Figure A (Part 32): four neurons, and Qwen says Google made it.

Reads results/part32/answers.jsonl and results/part32/score.txt; writes
figs-prototype/figA.png and figA.svg. Run from the repository root:
    python3 figs-prototype/make_figA.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import bubblekit as bk  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figs-prototype")
ANSWERS = "results/part32/answers.jsonl"
SCORE = "results/part32/score.txt"
META = "results/part32/answers.jsonl.meta.json"

GOOGLE, ALIBABA, TENCENT = "#d2e3fc", "#fde2c4", "#e6dcf7"
GOOGLE_DOT, OTHER_DOT = "#1a73e8", "#d9d9de"
NULL_RED = "#d93025"

HL = [
    (r"Google DeepMind|DeepMind|Google|Gemini 1\.5 Pro|Gemini", GOOGLE),
    (r"Tencent HunYuan|Tencent|HunYuan", TENCENT),
    (r"Alibaba Cloud's Tongyi Lab|Alibaba Cloud|Alibaba Group|Alibaba|Tongyi Lab|Qwen3\.5|"
     r"阿里巴巴集团|阿里巴巴|阿里云|通义实验室|通义千问", ALIBABA),
]

LANGS = [("english", "English"), ("spanish", "Spanish"), ("chinese", "Chinese")]


def maker(answer):
    """First maker the answer names: decides which answers the figure picks."""
    hits = []
    for name, pat in [("google", r"Google"), ("tencent", r"Tencent"),
                      ("alibaba", r"Alibaba|Tongyi|阿里|通义")]:
        m = re.search(pat, answer)
        if m:
            hits.append((m.start(), name))
    return min(hits)[1] if hits else "none"


def pick(cell, wanted):
    """Selection rule, stated in NOTES.md: walk the cell in file order and take
    the first answers of each maker in `wanted` (name -> how many)."""
    left = dict(wanted)
    chosen = []
    for i, a in enumerate(cell):
        m = maker(a)
        if left.get(m, 0) > 0:
            chosen.append(i)
            left[m] -= 1
    return chosen


def google_counts():
    """Google-side column of results/part32/score.txt, keyed (language, condition)."""
    out = {}
    for line in open(os.path.join(ROOT, SCORE), encoding="utf-8"):
        p = line.split()
        if len(p) == 6 and p[0] in {l for l, _ in LANGS} and p[2].isdigit():
            out[(p[0], p[1])] = (int(p[3]), sum(int(x) for x in p[2:]))
    return out


def main():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, ANSWERS), encoding="utf-8")]
    meta = json.load(open(os.path.join(ROOT, META), encoding="utf-8"))
    cells = {}
    for r in rows:
        cells.setdefault((r["language"], r["condition"]), []).append(r["answer"])
    counts = google_counts()

    plan = {
        "english": {"unchanged": {"alibaba": 2}, "zeroed": {"google": 2, "alibaba": 1}},
        "spanish": {"unchanged": {"tencent": 1, "alibaba": 1}, "zeroed": {"google": 2, "alibaba": 1}},
        "chinese": {"unchanged": {"alibaba": 2}, "zeroed": {"alibaba": 3}},
    }
    chosen = {(lang, cond): pick(cells[(lang, cond)], plan[lang][cond])
              for lang in plan for cond in plan[lang]}
    for k, v in chosen.items():
        print("chosen", k, "indices", v)

    W, H = 17.4, 30.0
    fig, ax = bk.canvas(W, H)
    meas = bk.Measure(fig)
    M, COLW, GUT = 0.55, 5.1, 0.55
    xs = [M + i * (COLW + GUT) for i in range(3)]
    BS = 14.5          # answer text size
    LIMIT = {"english": 125, "spanish": 125, "chinese": 52}

    def text_for(lang, cond, i):
        return bk.shorten(cells[(lang, cond)][i], LIMIT[lang])

    def bubble_h(lang, cond, i):
        _, th = bk.layout(meas, text_for(lang, cond, i), COLW - 0.12 - 0.32 - 0.12, BS, HL)
        return th + 0.32

    # ---- title block
    y = H - 0.55
    ax.text(M, y, "Four neurons, and Qwen says Google made it", fontsize=34, fontweight="bold",
            color=bk.INK, va="top")
    y -= 0.85
    ax.text(M, y, "Qwen3.5-4B was asked who created it in three languages, 24 sampled answers each. "
                  "Then four of its MLP neurons\n(L6:753, L7:1169, L11:8868, L29:776) were set to zero "
                  "and it was asked again. Real answers below, maker names highlighted.",
            fontsize=15.5, color=bk.MUTED, va="top", linespacing=1.45)
    y -= 1.05
    lx = M
    for colour, label in [(GOOGLE, "Google / DeepMind / Gemini"), (ALIBABA, "Alibaba / Tongyi Lab / Qwen (its real maker)"),
                          (TENCENT, "Tencent / HunYuan")]:
        bk.swatch(ax, lx, y, colour, label, size=13.5)
        lx += 0.6 + meas.width(label, 13.5) + 0.45
    y -= 0.55

    # ---- column headers and question bubbles
    for (lang, name), x in zip(LANGS, xs):
        ax.text(x, y, name, fontsize=22, fontweight="bold", color=bk.INK, va="top")
    y -= 0.62
    qh = 0
    for (lang, _), x in zip(LANGS, xs):
        q = meta["question_by_language"][lang]
        qh = max(qh, bk.bubble(ax, meas, x, y, COLW - 0.18, q, size=17, side="right", fc=bk.INK,
                               ec=bk.INK, tc="white", pad=0.17))
    y -= qh + 0.35

    def strip(x, ytop, lang, cond, grey, label_prefix):
        g, n = counts[(lang, cond)]
        cols = [OTHER_DOT] * n
        for i in range(g):
            cols[i] = "#9aa0a6" if grey else GOOGLE_DOT
        w, h = bk.dots(ax, x, ytop - 0.33, cols, per_row=24, r=0.083, gap=0.044)
        colour = bk.MUTED if (grey or g == 0) else GOOGLE_DOT
        ax.text(x, ytop, f"{label_prefix}Google in {g} of {n}", fontsize=14, fontweight="bold",
                color=colour, va="top")
        return 0.33 + h

    def section(title, sub, cond, fc, ec, grey_strip, ybase):
        ax.text(M, ybase, title, fontsize=17, fontweight="bold",
                color=bk.MUTED if grey_strip else bk.INK, va="top")
        ax.text(M + meas.width(title, 17, "bold") + 0.18, ybase - 0.04, sub, fontsize=13,
                color=bk.MUTED, va="top")
        ytop = ybase - 0.5
        stack = max(sum(bubble_h(lang, cond, i) + 0.2 for i in chosen[(lang, cond)])
                    for lang, _ in LANGS)
        hmax = 0
        for (lang, _), x in zip(LANGS, xs):
            yy = ytop
            for i in chosen[(lang, cond)]:
                yy -= bk.bubble(ax, meas, x + 0.12, yy, COLW - 0.12, text_for(lang, cond, i),
                                size=BS, highlights=HL, fc=fc, ec=ec) + 0.2
            yy = ytop - stack - 0.05
            yy -= strip(x + 0.12, yy, lang, cond, grey_strip, "")
            hmax = max(hmax, ytop - yy)
        return ytop, hmax

    # ---- section 1: unchanged
    ax.plot([M, W - M], [y + 0.12, y + 0.12], color="#e5e5ea", lw=1)
    top1, h1 = section("Unchanged model", "baseline · 2 of 24 answers shown per language", "unchanged",
                       bk.GREY_FILL, bk.GREY_FILL, True, y)
    y = top1 - h1 - 0.55

    # ---- section 2: zeroed
    ax.plot([M, W - M], [y + 0.12, y + 0.12], color="#e5e5ea", lw=1)
    top2, h2 = section("Four neurons zeroed", "3 of 24 answers shown per language", "zeroed",
                       "white", "#3a3a3c", False, y)
    # outline the part that failed: Chinese does not move
    xc = xs[2]
    bk.outline(ax, xc - 0.08, top2 - h2 - 0.18, COLW + 0.2, h2 + 0.36, NULL_RED,
               label="No effect in Chinese", size=14)
    y = top2 - h2 - 0.75

    # ---- section 3: control clamp (counts only)
    ax.plot([M, W - M], [y + 0.12, y + 0.12], color="#e5e5ea", lw=1)
    ax.text(M, y, "Control: the four set to their “told it is Gemini” values", fontsize=17,
            fontweight="bold", color=bk.MUTED, va="top")
    yy = y - 0.5
    hmax = 0
    for (lang, _), x in zip(LANGS, xs):
        hmax = max(hmax, strip(x + 0.12, yy, lang, "clamped", True, ""))
    y = yy - hmax - 0.55

    # ---- reading
    ax.text(M, y, "Removing the four makes room for Google in English and Spanish. Every Chinese answer "
                  "still says Alibaba's Tongyi Lab.\nSetting the four to Gemini-like values supplies no "
                  "maker at all: no language moves toward Google.",
            fontsize=15.5, color=bk.INK, va="top", linespacing=1.5)
    y -= 1.2

    # ---- footer
    foot = (
        f"Qwen3.5-4B ({meta['model']}@{meta['revision'][:7]}), T {meta['temperature']}, top-p {meta['top_p']}, "
        f"max {meta['max_new_tokens']} new tokens; n = 24 sampled answers per language and condition (216 in all).\n"
        "Answers are verbatim; the model's **bold** is drawn bold; “…” marks where the figure cuts an answer "
        "(the run itself stopped at 48 tokens). Chosen by a fixed rule:\n"
        "the first answers in file order, plus the first answer of each other maker in that cell (see NOTES.md). "
        "Glossary: 阿里巴巴 = Alibaba, 通义实验室 = Tongyi Lab, 通义千问 = Qwen.\n"
        "Counts: Google-side column of results/part32/score.txt (the hand reading in README Part 32 agrees on "
        "every Google count). Unchanged Spanish also names Tencent in 10 of 24 (README).\n"
        "Sources: results/part32/answers.jsonl, results/part32/score.txt, results/part32/answers.jsonl.meta.json. "
        "Neurons from Part 25."
    )
    ax.text(M, y, foot, fontsize=10.5, color=bk.MUTED, va="top", linespacing=1.55)
    y -= 1.55

    ybot = y
    fig.set_size_inches(W, H - ybot)
    ax.set_ylim(ybot, H)
    fig.savefig(os.path.join(OUT, "figA.png"), dpi=150, facecolor="white")
    fig.savefig(os.path.join(OUT, "figA.svg"), facecolor="white")
    print("wrote figA.png / figA.svg, height", round(H - ybot, 2), "in")


if __name__ == "__main__":
    main()
