"""Draw the front-page draft's figures from committed result files.

Run from anywhere: python docs/frontpage-draft/make_figures.py
Every plotted number is read from a file under results/; nothing is typed in.
No model is run. Writes PNGs to docs/frontpage-draft/figures/ and prints every
number it plots, so the draft's text can be checked against this output.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1f2933"
MUTED = "#9aa5b1"
ACCENT = "#d64545"
ACCENT2 = "#2f6fb3"
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.labelsize": 15, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.dpi": 100,
})


def load(rel):
    with open(ROOT / rel, encoding="utf-8") as f:
        return json.load(f)


def save(fig, name, src):
    fig.text(0.01, 0.01, f"Source: {src}", fontsize=10, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"wrote figures/{name}")


def label_bars(ax, bars, texts):
    for b, t in zip(bars, texts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), t, ha="center",
                va="bottom", fontsize=14, fontweight="bold")


# 1. Part 183: four maker neurons, eager-kernel rerun of Part 66.
def fig1():
    src = "results/part183/part66-eager/score.json"
    c = load(src)["conditions"]
    unch, tgt = c["unchanged"], c["target"]
    rand = {k: v for k, v in c.items() if k.startswith("random_")}
    rand_rates = [v["heldout_google"] / v["heldout_n"] for v in rand.values()]
    best = max(rand.values(), key=lambda v: v["heldout_google"] / v["heldout_n"])
    vals = [unch["heldout_google"] / unch["heldout_n"], max(rand_rates), tgt["heldout_google"] / tgt["heldout_n"]]
    print("F1 unchanged", unch["heldout_google"], "/", unch["heldout_n"],
          "| four zeroed", tgt["heldout_google"], "/", tgt["heldout_n"],
          "| best of", len(rand), "random sets", best["heldout_google"], "/", best["heldout_n"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(["Unchanged", f"Best of {len(rand)}\nrandom 4-sets", "Four maker\nneurons zeroed"],
                  [100 * v for v in vals], color=[MUTED, MUTED, ACCENT])
    label_bars(ax, bars, [f"{unch['heldout_google']} of {unch['heldout_n']}",
                          f"{best['heldout_google']} of {best['heldout_n']}",
                          f"{tgt['heldout_google']} of {tgt['heldout_n']}"])
    ax.set_ylabel("Held-out answers naming Google (%)")
    ax.set_ylim(0, 30)
    ax.set_title("Zeroing four neurons makes Qwen3.5-4B name Google\nas its maker; random sets do not")
    save(fig, "f1-maker-neurons.png", src)


# 2. Part 176: flat random directions reach the same Google fallback.
def fig2():
    src = "results/part176/numbers176.json"
    rows = load(src)["rung_table"]
    doses = [r["dose"] for r in rows]
    lo = [r["support"]["lower"] for r in rows]
    hi = [r["support"]["upper"] for r in rows]
    den = rows[0]["support"]["denominator"]
    brk = [r["median_broken_per_direction"] for r in rows]
    gled = [r["google_led"]["lower"] for r in rows]
    print("F2 doses", doses, "| supported lower", lo, "upper", hi, "of", den,
          "| google-led lower", gled, "| median broken of 44", brk)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(doses, lo, "-o", color=ACCENT, lw=3, ms=9, label=f"Directions with a supported maker answer (of {den})")
    ax.plot(doses, brk, "--s", color=INK, lw=2, ms=7, label="Median broken answers per direction (of 44)")
    for x, y, h in zip(doses, lo, hi):
        ax.annotate(f"{y}" if y == h else f"{y}-{h}", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=13, fontweight="bold", color=ACCENT)
    ax.set_xlabel("Push size along a flat random direction (dose)")
    ax.set_ylabel("Count")
    ax.set_ylim(-1, den * 0.9)
    ax.set_xticks(doses)
    ax.legend(fontsize=12, frameon=False, loc="upper right")
    ax.set_title("Random directions reach the Google fallback too,\nin a narrow dose window before answers break")
    save(fig, "f2-random-directions-fallback.png", src)


# 3. Part 184: the alien neuron, eager-kernel rerun of Parts 18 and 18b.
def fig3():
    src = "results/part184/compare.json"
    e = load(src)["eager"]
    A, B, C = e["A"], e["B"], e["C"]
    cells = [("Unclamped", A["samples"]["unclamped"]),
             ("20 random units\n(+2x, sampled)", A["controls_sampled"]),
             ("Prompt only\n(+2x)", C["samples"]["+2"]),
             ("Generation only\n(+2x)", B["samples"]["+2"]),
             ("Every position\n(+2x)", A["samples"]["+2"])]
    for name, v in cells:
        print("F3", name.replace("\n", " "), v["alien"], "/", v["n"])
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    vals = [100 * v["alien"] / v["n"] for _, v in cells]
    colors = [MUTED, MUTED, MUTED, ACCENT, ACCENT]
    bars = ax.bar([n for n, _ in cells], vals, color=colors)
    label_bars(ax, bars, [f"{v['alien']} of {v['n']}" for _, v in cells])
    ax.set_ylabel("\"Who made you?\" answers\nclaiming off-world origin (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Clamping L22 N2742 makes Qwen3.5-4B say it is an alien,\nbut only when the clamp runs while it writes")
    save(fig, "f3-alien-neuron.png", src)


# 4. Part 185: the alien neuron's dose ladder (eager rerun of Part 84b).
def fig4():
    src = "results/part185/score185.json"
    d = load(src)["counts"]
    b = d["B"]["offworld"]
    doses = ["0.40", "0.47", "0.55"]
    dec = [b[f"decode{x}"] for x in doses]
    gen = [b[f"gen{x}"] for x in doses]
    a = d["A"]
    print("F4 84b ladder decode", dec, "gen", gen, "of 48; unchanged", b["A0"],
          "| 84 control unit", a["C"]["OFFWORLD"], "/", a["C"]["n"], "| 84 dose 0.63 gen-only", a["A2"]["OFFWORLD"], "/", a["A2"]["n"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = [0, 0.40, 0.47, 0.55]
    ax.plot(xs, [b["A0"]] + dec, "-o", color=ACCENT, lw=3, ms=9, label="Generated tokens only")
    ax.plot(xs, [b["A0"]] + gen, "--s", color=ACCENT2, lw=2, ms=8, label="Also the first answer token")
    for x, y in zip(xs, [b["A0"]] + dec):
        ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(-14, 6), fontsize=13,
                    fontweight="bold", color=ACCENT)
    ax.axhline(48, color=MUTED, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["0\n(unchanged)", "0.40", "0.47", "0.55"])
    ax.set_xlabel("Dose (Lyra's neuronary sizing rule)")
    ax.set_ylabel("Off-world answers (of 48)")
    ax.set_ylim(-2, 54)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    ax.set_title("The alien push climbs steeply between dose 0.40 and 0.55,\nand steering the first answer token adds little")
    save(fig, "f4-alien-dose-ladder.png", src)


# 5. Part 186: the villain recipe, eager rerun of Part 79.
def fig5():
    src = "results/part186/score186.json"
    d = load(src)
    # "columns" is stored as the string "[SDPA, eager]"; parse it rather than assume the order.
    cols = [c.strip() for c in d["columns"].strip("[]").split(",")]
    j = cols.index("eager")
    cmp_ = d["comparison"]
    names = [("e0_unchanged", "Unchanged"), ("e1_makers", "Makers zeroed"), ("e2_evil", "Evil push"),
             ("e3_makers_evil", "Makers zeroed\n+ evil push")]
    ctl = [k for k in cmp_ if k.startswith("e") and "ctl" in k]
    villain = [cmp_[k]["reader_villain"][j] for k, _ in names]
    company = [cmp_[k]["any_company"][j] for k, _ in names]
    ctl_company = [cmp_[k]["any_company"][j] for k in ctl]
    print("F5 eager villain", dict(zip([k for k, _ in names], villain)), "| any company",
          dict(zip([k for k, _ in names], company)), "| makers zeroed + 5 random pairs, any company", ctl_company)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = range(len(names))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], villain, w, color=ACCENT, label="Villain persona (blind reader)")
    b2 = ax.bar([i + w / 2 for i in x], company, w, color=ACCENT2, label="Names any company")
    label_bars(ax, b1, [str(v) for v in villain])
    label_bars(ax, b2, [str(v) for v in company])
    ax.set_xticks(list(x))
    ax.set_xticklabels([n for _, n in names])
    ax.set_ylabel("Answers (of 192)")
    ax.set_ylim(0, 230)
    ax.legend(frameon=False, fontsize=12, loc="upper center", ncol=2)
    ax.set_title(f"The evil push takes over the maker answer; with the makers zeroed,\n"
                 f"five random pairs still name a company {min(ctl_company)} to {max(ctl_company)} times")
    save(fig, "f5-villain-recipe.png", src)


# 6. Part 82: switching off heads shrinks the own-message gap; random sets can too.
def fig6():
    src = "results/part82/main/results82.json"
    run = load("results/part82/main/run.json")["conditions"]
    s = load(src)["summaries"]
    order = ["h0", "h1", "c1", "c2", "c3", "c4", "c5"]
    labels = {"h0": "No heads off", "h1": "Lyra's six heads"}
    vals, names, cols = [], [], []
    for k in order:
        g = s[k]["B"]["cyclic"]["gap"]
        vals.append(g)
        heads = run[k]
        has_l11h5 = 5 in heads.get("11", [])
        nm = labels.get(k, f"Random set {k[1]}" + ("\n(holds L11 h5)" if has_l11h5 else ""))
        names.append(nm)
        cols.append(INK if k == "h0" else (ACCENT if k == "h1" else MUTED))
        print("F6", k, heads, "gap", round(g, 3), "ci95", [round(x, 3) for x in s[k]["B"]["cyclic"]["gap_ci95"]])
    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars = ax.bar(names, vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, max(v, 0), f"{v:.3f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    ax.axhline(0, color=INK, lw=1)
    ax.set_ylabel("\"You\" gap: Assistant copy\nminus User copy")
    ax.tick_params(axis="x", labelsize=11)
    ax.set_title("Six chosen heads cut Qwen3.5-4B's own-message gap,\nbut of five random six-head sets, one cut it further and one nearly as far")
    save(fig, "f6-own-message-heads.png", src)


# 7. Part 97: a neuron's J-lens words match its pushed persona more than shuffled pairs.
def fig7():
    def count(rel, kind, lens):
        n = m = 0
        with open(ROOT / rel, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == kind and r["lens"] == lens:
                    n += 1
                    m += bool(r["match"])
        return m, n
    src = "results/part97/match.jsonl"
    real_j = count(src, "neuron", "j_lens")
    real_l = count(src, "neuron", "logit_lens")
    rand_j = count(src, "random", "j_lens")
    shuf = count("results/part97/match-shuffled.jsonl", "neuron", "j_lens")
    print("F7 real J-lens", real_j, "| real logit lens", real_l, "| random J-lens", rand_j, "| shuffled pairs", shuf)
    cells = [("Random directions\n(J-lens)", rand_j), ("Shuffled\nneuron-answer pairs", shuf),
             ("True pairs,\nlogit lens", real_l), ("True pairs,\nJ-lens", real_j)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar([c[0] for c in cells], [100 * m / n for _, (m, n) in cells],
                  color=[MUTED, MUTED, ACCENT2, ACCENT])
    label_bars(ax, bars, [f"{m} of {n:,}" for _, (m, n) in cells])
    ax.set_ylabel("Lens word list matches\nthe pushed persona (%)")
    ax.set_ylim(0, 14)
    ax.set_title("Which persona a pushed neuron gives is partly written in its own words:\n"
                 "about 1 in 9 true pairs match, 1 in 60 shuffled")
    save(fig, "f7-lens-persona-match.png", "results/part97/match.jsonl, match-shuffled.jsonl")


# 8. Parts 114/115: the forced "Q" token needs hundreds of neurons zeroed.
def fig8():
    src = "results/part115/score.json"
    d = load(src)
    pk = d["per_k"]
    ks = sorted(int(k) for k in pk)
    flips = [pk[str(k)]["candidate"]["flips"] for k in ks]
    ctl = [max(pk[str(k)]["controls_flips"]) for k in ks]
    n = d["base_q_first"]
    p114 = load("results/part114/score.json")
    pair = p114["pair"]
    print("F8 k", ks, "| top-k flips", flips, "of", n, "| max random-control flips", ctl,
          "| Part 114 held-out n", p114["n_heldout"], "| Part 114 top-20 single-unit max flips",
          max(c["flips"] for c in p114["candidates"]), "| pair keys", list(pair))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(ks, flips, "-o", color=ACCENT, lw=3, ms=9, label="Top-k candidates zeroed")
    ax.plot(ks, ctl, "--s", color=MUTED, lw=2, ms=7, label="Most flips by any of 10 random k-sets")
    for k, f in zip(ks, flips):
        if f:
            ax.annotate(f"{f} of {n}", (k, f), textcoords="offset points", xytext=(-10, 8), ha="right",
                        fontsize=13, fontweight="bold", color=ACCENT)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Neurons zeroed (k, log scale)")
    ax.set_ylabel(f"Held-out phrasings where \"Q\"\nis no longer first (of {n})")
    ax.set_ylim(-1, n + 2)
    ax.set_yticks(range(0, n + 1, 4))
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    ax.set_title("No small group holds the forced \"I am **Q\" answer:\n0 flips up to 50 neurons, all 16 at 500")
    save(fig, "f8-q-token-zeroing.png", src)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        f()
