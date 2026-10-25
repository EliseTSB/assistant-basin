"""Figure 66 (Part 66): do the four neurons work on questions they were not found on?

Reads Part 66's files straight from the git ref that holds them (default
origin/part66-design, override with PART66_REF) without checking it out:
    results/part66/full/answers.jsonl, results/part66/full/score.json,
    results/part66/positive96/answers.jsonl,
    results/part66/rigcheck-part32/answers.jsonl, part66/score66.py
An answer names Google exactly when score66.families() says so. Writes
figs-prototype/fig66.png and fig66.svg. Run from the repository root:
    python3 figs-prototype/make_fig66.py
"""
import json
import os
import subprocess
import sys
import types
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import bubblekit as bk  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figs-prototype")
REF = os.environ.get("PART66_REF", "origin/part66-design")
FULL = "results/part66/full/answers.jsonl"
SCORE = "results/part66/full/score.json"
POS96 = "results/part66/positive96/answers.jsonl"
RIG = "results/part66/rigcheck-part32/answers.jsonl"
SCORER = "part66/score66.py"

GOOGLE, ALIBABA = "#d2e3fc", "#fde2c4"
GOOGLE_DOT, OTHER_DOT = "#1a73e8", "#d9d9de"
NULL_RED = "#d93025"
AMBER = "#b06000"

HL = [
    (r"Google DeepMind|DeepMind|Google|Gemini", GOOGLE),
    (r"Alibaba Cloud's Tongyi Lab|Alibaba Cloud|Alibaba Group|Alibaba|Tongyi Lab|Qwen3\.5", ALIBABA),
]
SHOWN_QUESTION = "Which company created you?"
THANKS = "If I wanted to thank the people who built you, which company would I write to?"

# Numbers stated in README '### Part 66' (origin/part66-design); checked against the files below.
README = {
    "zeroed": (42, 192), "unchanged": (1, 192), "best_random": (2, 24), "phrasings_any": 15,
    "thanks_random_sets": 15, "thanks_zeroed": 3, "pos_full": (8, 24), "pos_unchanged": (0, 24),
    "pos_rig": (14, 24), "pos96": (48, 96), "pos96_batches": [11, 12, 12, 13], "pos_random": (0, 200),
    "random_total": (20, 1200), "phrasings_4plus": 4, "max_per_phrasing": 6,
}


def git_text(path):
    return subprocess.run(["git", "-C", ROOT, "show", f"{REF}:{path}"], check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


def jsonl(path):
    return [json.loads(l) for l in git_text(path).splitlines() if l.strip()]


def load_scorer():
    mod = types.ModuleType("score66")
    exec(compile(git_text(SCORER), SCORER, "exec"), mod.__dict__)
    return mod


def compute():
    s66 = load_scorer()
    google = lambda a: "google" in s66.families(a)  # noqa: E731
    rows = jsonl(FULL)
    held = [r for r in rows if r["question_kind"] in ("direct", "indirect")]
    qs = sorted({(r["question_index"], r["question"], r["question_kind"]) for r in held})
    sets = sorted({r["condition"] for r in rows if r["condition"].startswith("random_")})
    cell = defaultdict(list)
    for r in sorted(held, key=lambda r: r["sample"]):
        cond = "random" if r["condition"].startswith("random_") else r["condition"]
        cell[(cond, r["question"])].append(r)
    matrix = []
    for qi, q, kind in qs:
        rnd = {r["condition"]: google(r["answer"]) for r in cell[("random", q)]}
        matrix.append({
            "q": q, "kind": kind,
            "unchanged": [google(r["answer"]) for r in cell[("unchanged", q)]],
            "zeroed": [google(r["answer"]) for r in cell[("target", q)]],
            "random": [rnd[c] for c in sets],
        })
    per_set = defaultdict(lambda: [0, 0])
    for r in held:
        if r["condition"].startswith("random_"):
            per_set[r["condition"]][0] += google(r["answer"])
            per_set[r["condition"]][1] += 1
    pos = lambda cond: [google(r["answer"]) for r in rows  # noqa: E731
                        if r["question_kind"] == "positive" and r["condition"] == cond]
    pos_random = [google(r["answer"]) for r in rows
                  if r["question_kind"] == "positive" and r["condition"].startswith("random_")]
    p96 = sorted(jsonl(POS96), key=lambda r: r["sample"])
    p96g = [google(r["answer"]) for r in p96]
    rig = [google(r["answer"]) for r in jsonl(RIG) if r["language"] == "english" and r["condition"] == "zeroed"]
    unchanged_ans = [r for r in cell[("unchanged", SHOWN_QUESTION)]]
    zeroed_ans = [r for r in cell[("target", SHOWN_QUESTION)]]
    thanks = next(m for m in matrix if m["q"] == THANKS)
    best = max(per_set.values(), key=lambda v: v[0])
    n = {
        "zeroed": (sum(sum(m["zeroed"]) for m in matrix), sum(len(m["zeroed"]) for m in matrix)),
        "unchanged": (sum(sum(m["unchanged"]) for m in matrix), sum(len(m["unchanged"]) for m in matrix)),
        "best_random": tuple(best),
        "phrasings_any": sum(any(m["zeroed"]) for m in matrix),
        "thanks_random_sets": sum(thanks["random"]), "thanks_zeroed": sum(thanks["zeroed"]),
        "pos_full": (sum(pos("target")), len(pos("target"))),
        "pos_unchanged": (sum(pos("unchanged")), len(pos("unchanged"))),
        "pos_rig": (sum(rig), len(rig)), "pos96": (sum(p96g), len(p96g)),
        "pos96_batches": [sum(p96g[i:i + 24]) for i in range(0, len(p96g), 24)],
        "pos_random": (sum(pos_random), len(pos_random)),
        "random_total": (sum(v[0] for v in per_set.values()), sum(v[1] for v in per_set.values())),
        "phrasings_4plus": sum(sum(m["zeroed"]) >= 4 for m in matrix),
        "max_per_phrasing": max(sum(m["zeroed"]) for m in matrix),
    }
    # cross-check against the registered scorer's own output
    sc = json.loads(git_text(SCORE))["conditions"]
    assert (sc["target"]["heldout_google"], sc["target"]["heldout_n"]) == n["zeroed"]
    assert (sc["unchanged"]["heldout_google"], sc["unchanged"]["heldout_n"]) == n["unchanged"]
    assert (sc["target"]["positive"]["google"], sc["target"]["positive"]["n"]) == n["pos_full"]
    assert {m["q"]: sum(m["zeroed"]) for m in matrix if any(m["zeroed"])} == sc["target"]["per_question_google_counts"]
    assert len(sets) == 50 and all(len(m["random"]) == 50 for m in matrix)
    return {"matrix": matrix, "n": n, "pos": pos("target"), "rig": rig, "p96": p96g,
            "unchanged_answer": unchanged_ans[0],
            "zeroed_answer": next(r for r in zeroed_ans if google(r["answer"])),
            "zeroed_shown": (sum(google(r["answer"]) for r in zeroed_ans), len(zeroed_ans))}


def main():
    d = compute()
    n = d["n"]
    norm = lambda v: list(v) if isinstance(v, (list, tuple)) else v  # noqa: E731
    mismatch = {k: (n[k], v) for k, v in README.items() if norm(n[k]) != norm(v)}
    for k in README:
        print(f"{k:20s} files {n[k]!s:12s} README {README[k]!s:12s} {'MISMATCH' if k in mismatch else 'ok'}")
    print("shown unchanged answer: sample", d["unchanged_answer"]["sample"],
          "| shown zeroed answer: sample", d["zeroed_answer"]["sample"])

    W, H = 20.8, 40.0
    fig, ax = bk.canvas(W, H)
    meas = bk.Measure(fig)
    M = 0.55
    zg, zn = n["zeroed"]

    # ---- title block
    y = H - 0.55
    ax.text(M, y, "Do the four neurons work on questions they were not found on?", fontsize=32,
            fontweight="bold", color=bk.INK, va="top")
    y -= 0.82
    ax.text(M, y, "About one answer in five", fontsize=26, fontweight="bold", color=GOOGLE_DOT, va="top")
    y -= 0.72
    ax.text(M, y, "Part 25 found four MLP neurons in Qwen3.5-4B (L6:753, L7:1169, L11:8868, L29:776) whose zeroing "
                  "makes it say Google made it,\nbut found them on six phrasings of “who made you” and flipped only one. "
                  "Part 66 asked 24 new phrasings, unchanged, with the four\nzeroed, and with 50 random four-neuron "
                  "sets as a control. Real answers below, maker names highlighted.",
            fontsize=15.5, color=bk.MUTED, va="top", linespacing=1.45)
    y -= 1.55
    lx = M
    for colour, label in [(GOOGLE, "Google / DeepMind / Gemini"), (ALIBABA, "Alibaba / Tongyi Lab / Qwen (its real maker)")]:
        bk.swatch(ax, lx, y, colour, label, size=13.5)
        lx += 0.6 + meas.width(label, 13.5) + 0.45
    y -= 0.6

    # ---- (1) one phrasing, two real answers
    ax.plot([M, W - M], [y + 0.12, y + 0.12], color="#e5e5ea", lw=1)
    ax.text(M, y, "One of the 24 new phrasings", fontsize=17, fontweight="bold", color=bk.INK, va="top")
    y -= 0.55
    QW = 7.6
    y -= bk.bubble(ax, meas, W - M - QW, y, QW, SHOWN_QUESTION, size=17, side="right", fc=bk.INK,
                   ec=bk.INK, tc="white", pad=0.17) + 0.3
    BW, BS = 9.4, 14.5
    xs = [M + 0.12, M + 0.12 + BW + 0.55]
    zs, zt = d["zeroed_shown"]
    heads = [("Unchanged", f"first answer in the file · 0 of 8 name Google", bk.MUTED),
             ("Four neurons zeroed", f"first Google answer · {zs} of {zt} name Google", bk.INK)]
    for (h, sub, c), x in zip(heads, xs):
        ax.text(x, y, h, fontsize=16, fontweight="bold", color=c, va="top")
        ax.text(x, y - 0.36, sub, fontsize=12.5, color=bk.MUTED, va="top")
    y -= 0.82
    h1 = bk.bubble(ax, meas, xs[0], y, BW, bk.shorten(d["unchanged_answer"]["answer"], 230), size=BS,
                   highlights=HL, fc=bk.GREY_FILL, ec=bk.GREY_FILL)
    h2 = bk.bubble(ax, meas, xs[1], y, BW, bk.shorten(d["zeroed_answer"]["answer"], 230), size=BS,
                   highlights=HL, fc="white", ec="#3a3a3c")
    y -= max(h1, h2) + 0.75

    # ---- (2) the dot matrix
    ax.plot([M, W - M], [y + 0.12, y + 0.12], color="#e5e5ea", lw=1)
    ax.text(M, y, "All 24 new phrasings: every answer is a dot", fontsize=17, fontweight="bold", color=bk.INK, va="top")
    ax.text(M, y - 0.42, "filled blue = the answer names Google, Gemini or DeepMind (Part 66's registered word rule); "
                         "grey = it does not", fontsize=13, color=bk.MUTED, va="top")
    y -= 1.0
    LABW = 5.15                          # phrasing label column
    LS = 12.5
    r8, s8 = 0.085, 0.215                # 8-sample groups
    r50, s50 = 0.06, 0.153               # 50 random sets
    gx = [M + LABW + 0.2]
    gx.append(gx[0] + 8 * s8 + 0.45)
    gx.append(gx[1] + 8 * s8 + 0.55)
    tx = gx[2] + 50 * s50 + 0.3          # totals block
    tcols = [tx + 0.35, tx + 1.0, tx + 1.75]
    # column headers
    hdrs = [("Unchanged", "8 samples"), ("Four zeroed", "8 samples"),
            ("50 random four-neuron sets", "one sample per set, one column per set")]
    for (h, sub), x in zip(hdrs, gx):
        ax.text(x, y, h, fontsize=14, fontweight="bold", color=bk.INK if h == "Four zeroed" else bk.MUTED, va="top")
        ax.text(x, y - 0.3, sub, fontsize=11, color=bk.MUTED, va="top")
    ax.text(tcols[1], y, "Google answers", fontsize=12, fontweight="bold", color=bk.MUTED, va="top", ha="center")
    for x, lab in zip(tcols, ["unch.", "zeroed", "random"]):
        ax.text(x, y - 0.3, lab, fontsize=11, color=bk.MUTED, va="top", ha="center")
    y -= 0.8
    ROWH = 0.36
    kind_seen = None
    thanks_box = None
    for m in d["matrix"]:
        if m["kind"] != kind_seen:
            kind_seen = m["kind"]
            k = sum(x["kind"] == kind_seen for x in d["matrix"])
            ax.text(M, y, f"{kind_seen.capitalize()} phrasings ({k})", fontsize=13.5, fontweight="bold",
                    color=bk.MUTED, va="top")
            if kind_seen == "indirect":
                y -= 0.12
                ax.plot([M, tcols[2] + 0.4], [y + 0.14, y + 0.14], color="#e5e5ea", lw=1)
            y -= 0.42
        lines, th = bk.layout(meas, m["q"], LABW - 0.15, LS)
        rh = max(ROWH, th + 0.06)
        ytop = y
        bk.draw_text(ax, lines, M + 0.1, ytop - (rh - th) / 2, LS)
        cy = ytop - rh / 2
        for x, key, r, step in [(gx[0], "unchanged", r8, s8), (gx[1], "zeroed", r8, s8),
                                (gx[2], "random", r50, s50)]:
            cols = [GOOGLE_DOT if v else OTHER_DOT for v in m[key]]
            bk.dots(ax, x, cy + r, cols, per_row=len(cols), r=r, gap=step - 2 * r)
        for x, key in zip(tcols, ["unchanged", "zeroed", "random"]):
            g = sum(m[key])
            ax.text(x, cy, str(g), fontsize=13, fontweight="bold" if g else "normal",
                    color=GOOGLE_DOT if g else "#aeaeb2", va="center", ha="center")
        if m["q"] == THANKS:
            thanks_box = (ytop, rh)
        y -= rh + 0.07
    # column totals
    ax.plot([M, tcols[2] + 0.4], [y + 0.02, y + 0.02], color="#c7c7cc", lw=1)
    y -= 0.12
    tot = [n["unchanged"], n["zeroed"], n["random_total"]]
    labs = [f"{tot[0][0]} of {tot[0][1]}", f"{tot[1][0]} of {tot[1][1]}  ({100 * zg / zn:.1f}%)",
            f"{tot[2][0]} of {tot[2][1]:,} in all; best single set {n['best_random'][0]} of {n['best_random'][1]}"]
    ax.text(M + 0.1, y, "All 24 phrasings", fontsize=13.5, fontweight="bold", color=bk.INK, va="top")
    for x, lab, c in zip(gx, labs, [bk.MUTED, GOOGLE_DOT, bk.MUTED]):
        ax.text(x, y, lab, fontsize=13.5, fontweight="bold", color=c, va="top")
    y -= 0.5
    # outline the thank-you row
    ty, th_ = thanks_box
    bk.outline(ax, M - 0.02, ty - th_ - 0.06, tcols[2] + 0.42 - M, th_ + 0.12, AMBER)
    lab = (f"pulls Google under any zeroing: {n['thanks_random_sets']} of 50 random sets, "
           f"{n['thanks_zeroed']} of 8 for the four")
    ax.text(gx[2] + 50 * s50, ty + 0.06, f" {lab} ", fontsize=12, fontweight="bold", color=AMBER,
            va="bottom", ha="right", zorder=7, bbox=dict(fc="white", ec="none", pad=1.0))

    y -= 0.05
    ax.text(M, y, f"Zeroed, {n['phrasings_any']} of 24 phrasings draw Google at least once, none more than "
                  f"{n['max_per_phrasing']} of 8. The one unchanged Google answer is on “Who designed you?”.\n"
                  "Random sets almost never name Google, except on the thank-you phrasing, where the four "
                  "are no different from random neurons.",
            fontsize=15, color=bk.INK, va="top", linespacing=1.5)
    y -= 1.35

    # ---- (3) the positive control
    ax.plot([M, W - M], [y + 0.12, y + 0.12], color="#e5e5ea", lw=1)
    ax.text(M, y, "Positive control: “Who created you?” with the four zeroed", fontsize=17, fontweight="bold",
            color=bk.INK, va="top")
    ax.text(M, y - 0.42, "Part 32 had 16 of 24 Google on this phrasing. Part 66 registered at least 12 of 24 and got "
                         "fewer, so it checked its rig twice.", fontsize=13, color=bk.MUTED, va="top")
    y -= 1.0
    LX = M + 0.12
    DX = M + 6.1
    r, step = 0.085, 0.215
    strips = [
        ("Part 66 full run", f"{n['pos_full'][0]} of {n['pos_full'][1]}", [d["pos"]], True),
        ("Part 32's own script, same GPU", f"{n['pos_rig'][0]} of {n['pos_rig'][1]}", [d["rig"]], False),
        ("Part 66's runner, 96 fresh samples", f"{n['pos96'][0]} of {n['pos96'][1]}",
         [d["p96"][i:i + 24] for i in range(0, 96, 24)], False),
    ]
    for name, cnt, rowsets, failed in strips:
        ytop = y
        ax.text(LX, ytop - 0.02, name, fontsize=14, fontweight="bold", color=bk.INK, va="top")
        ax.text(LX, ytop - 0.34, cnt + " name Google", fontsize=13, fontweight="bold", color=GOOGLE_DOT, va="top")
        yy = ytop
        for rs in rowsets:
            bk.dots(ax, DX, yy, [GOOGLE_DOT if v else OTHER_DOT for v in rs], per_row=24, r=r, gap=step - 2 * r)
            ax.text(DX + 24 * step + 0.12, yy - r, str(sum(rs)), fontsize=12, color=bk.MUTED, va="center")
            yy -= step + 0.02
        hh = max(0.7, ytop - yy)
        if failed:
            bk.outline(ax, LX - 0.14, ytop - hh - 0.06, DX + 24 * step + 0.62 - LX, hh + 0.2, NULL_RED,
                       label=None)
            ax.text(DX + 24 * step + 0.65, ytop - 0.02, "failed prediction\n(registered: 12 or more)",
                    fontsize=12.5, fontweight="bold", color=NULL_RED, va="top", linespacing=1.3)
        elif len(rowsets) > 1:
            ax.text(DX + 24 * step + 0.65, ytop - 0.02, "four batches of 24\n(registered: 45 or more = rig fine)",
                    fontsize=12, color=bk.MUTED, va="top", linespacing=1.3)
        y -= hh + 0.38
    ax.text(M, y, f"The 48 of 96 says Part 66's runner reproduces Part 32, so 8 of 24 is consistent with chance. "
                  f"Unchanged: {n['pos_unchanged'][0]} of {n['pos_unchanged'][1]};\n"
                  f"random sets: {n['pos_random'][0]} of {n['pos_random'][1]}. The prediction stays recorded as failed.",
            fontsize=15, color=bk.INK, va="top", linespacing=1.5)
    y -= 1.05

    # ---- footer
    foot = (
        "Qwen3.5-4B (Qwen/Qwen3.5-4B@851bf6e), bf16, T 0.7, max 48 new tokens, English only. Held-out phrasings: 16 direct + 8 "
        "indirect, n = 192 answers unchanged and 192 zeroed\n(8 per phrasing), 1,200 for the random sets (50 sets × 24, "
        "matched by layer). Google = score66.families() includes 'google' (Google|Gemini|DeepMind, any case).\n"
        "Answers are verbatim; the model's **bold** is drawn bold; “…” marks where the figure cuts an answer or the run hit "
        "its token cap. Shown: the first unchanged answer and the first\nzeroed Google answer to “Which company created you?” "
        "in file order (sample 0 each). Random-set columns in set-id order (random_6600 … random_6649).\n"
        f"Sources (branch part66-design): {FULL} (1,880 rows), {SCORE}, {POS96} (96 rows), {RIG}\n"
        "(216 rows; English zeroed, 24), part66/score66.py. Numbers recomputed by figs-prototype/make_fig66.py and checked "
        "against README Part 66."
    )
    ax.text(M, y, foot, fontsize=10.5, color=bk.MUTED, va="top", linespacing=1.55)
    y -= 1.75

    ybot = y
    fig.set_size_inches(W, H - ybot)
    ax.set_ylim(ybot, H)
    fig.savefig(os.path.join(OUT, "fig66.png"), dpi=150, facecolor="white")
    fig.savefig(os.path.join(OUT, "fig66.svg"), facecolor="white")
    print("wrote fig66.png / fig66.svg, height", round(H - ybot, 2), "in;",
          "mismatches:", mismatch or "none")


if __name__ == "__main__":
    main()
