#!/usr/bin/env python3
"""Cold-check supplements for Part 168: scorer gates, list contents, Part 164/165 comparisons, mass-list anatomy.

Run from the repo root:  /home/elisepc/neuron-probe/.venv/bin/python -I -B cold168/checks2.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.cwd()
R = ROOT / "results/part168"
sys.path[:0] = [str(ROOT), str(ROOT / "part86")]
from part165 import score165  # noqa: E402

WORD = re.compile(r"[\w]+(?:['’][\w]+)*")
LOW = ["n+", "k4"]


def jl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def words(t):
    return [w.casefold() for w in WORD.findall(t or "")]


def loop_gram(t):
    ws = words(t)
    grams = [tuple(ws[i:i + 3]) for i in range(len(ws) - 2)]
    if not grams:
        return None
    c = Counter(grams)
    g, n = max(((g, c[g]) for g in dict.fromkeys(grams)), key=lambda x: x[1])
    return g if n >= 3 else None


def main():
    tokens = json.loads((R / "tokens.json").read_text(encoding="utf-8"))
    rows = {(r["arm"], r["layer"], r["slot"], r["question_index"]): r for r in jl(R / "rows.jsonl")}
    masses = {(m["arm"], m["layer"], m["slot"], m["question_index"]): m for m in jl(R / "masses.jsonl")}
    ans = jl(R / "answers.jsonl")
    bat = {(r["arm"], r["ban"], r["layer"], r["slot"], r["question_index"]): r for r in ans if r["stage"] == "battery"}
    no = [r for r in ans if r["stage"] == "no_push"]

    # ---- A. Gate 4 independently: banned ids equal the list the design names; none generated
    bad = Counter()
    for (arm, ban, L, s, q), r in bat.items():
        if ban == "none":
            if r["banned_ids"]:
                bad["none_has_ids"] += 1
            continue
        bids = set(r["banned_ids"])
        if len(r["banned_ids"]) != 50 or len(bids) != 50:
            bad["not50distinct"] += 1
        if ban == "own":
            want = set(rows[(arm, L, s, q)]["own_ban_ids"])
        elif ban == "matched":
            want = set(rows[(arm, L, (s + 1) % 12, q)]["own_ban_ids"])
        else:
            want = set(masses[(arm, L, s, q)]["mass_ban_ids"])
        if bids != want:
            bad["list_mismatch_" + ban] += 1
        if bids & set(r["token_ids"]):
            bad["banned_token_generated_" + ban] += 1
    print("A. Gate 4 independent check: problems =", dict(bad) if bad else "none")

    # ---- B. special tokens / EOS in lists (design: removed from every list)
    special_like = 0
    sp_examples = []
    for r in rows.values():
        for t in list(r["own_ban_ids"]):
            txt = str(tokens.get(str(int(t)), ""))
            if txt.startswith("<") and txt.endswith(">") or "<|" in txt or "|>" in txt:
                special_like += 1
                if len(sp_examples) < 5:
                    sp_examples.append((t, txt))
    for m in masses.values():
        for t in m["mass_ban_ids"]:
            txt = str(tokens.get(str(int(t)), ""))
            if txt.startswith("<") and txt.endswith(">") or "<|" in txt or "|>" in txt:
                special_like += 1
                if len(sp_examples) < 10:
                    sp_examples.append((t, txt))
    print("B. special-looking tokens in own/mass lists:", special_like, sp_examples[:10])
    # does tokens.json have a token for each id in any list?
    missing = sum(1 for r in rows.values() for t in r["own_ban_ids"] + r["rank_51_200_ids"] if str(int(t)) not in tokens)
    print("   list ids missing from tokens.json:", missing)

    # ---- C. mass list disjoint from own and matched; 50 distinct; feasibility flags
    ov = Counter()
    for (arm, L, s, q), m in masses.items():
        if arm not in LOW:
            continue
        mb = set(m["mass_ban_ids"])
        own = set(rows[(arm, L, s, q)]["own_ban_ids"])
        mat = set(rows[(arm, L, (s + 1) % 12, q)]["own_ban_ids"])
        if mb & own:
            ov["mass_overlaps_own"] += 1
        if mb & mat:
            ov["mass_overlaps_matched"] += 1
        if len(mb) != 50:
            ov["not50"] += 1
    print("C. mass-list overlap problems:", dict(ov) if ov else "none")
    oot = [(k, masses[k]["mass_rel_error"], masses[k]["repaired"], masses[k]["feasible"], masses[k]["paired_set"])
           for k in masses if k[0] in LOW and not masses[k]["in_tolerance"]]
    print("   rows not in tolerance after repair:", oot)

    # ---- D. the exponent and what the mass list is made of
    ex = {a: [masses[k]["exponent"] for k in masses if k[0] == a] for a in LOW}
    for a in LOW:
        v = sorted(ex[a])
        print(f"D. exponent {a}: min {v[0]:.2f} median {v[len(v)//2]:.2f} max {v[-1]:.2f}; share >0: {sum(x > 0 for x in v)/len(v):.2f}")
    po = [(masses[k]["plain_overlap_mass"], masses[k]["exponent"]) for k in masses if k[0] in LOW]
    # correlation between exponent and plain overlap
    import math
    xs = [p[1] for p in po]; ys = [p[0] for p in po]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs)); vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    print(f"   corr(exponent, plain overlap of mass list) = {cov/(vx*vy):.3f}")
    # mass list: share of ids that are in the top 50 by q? not stored. Plain overlap distribution:
    dist = Counter(min(int(p[0]), 5) for p in po)
    print("   plain-overlap distribution (capped at 5):", dict(sorted(dist.items())))

    # ---- E. the mass rise by plain-overlap bin (n+ and k4 pooled, 1344 rows)
    def ltrue(arm, ban, L, s, q):
        return loop_gram(bat[(arm, ban, L, s, q)]["answer"]) is not None
    bins = defaultdict(lambda: [0, 0, 0])  # n, none-loop, induced
    for (arm, L, s, q), m in masses.items():
        if arm not in LOW:
            continue
        b = min(int(m["plain_overlap_mass"]), 4)
        bins[b][0] += 1
        n0 = ltrue(arm, "none", L, s, q)
        n1 = ltrue(arm, "mass", L, s, q)
        bins[b][1] += n0
        bins[b][2] += (n1 and not n0)
    print("E. mass-list plain overlap bin -> rows, none-loop rate, mass-induced count:")
    for b in sorted(bins):
        n, nl, ind = bins[b]
        print(f"   overlap {'>=' if b == 4 else ''}{b}: rows {n}, none loop {nl}, induced by mass {ind}")

    # ---- F. own-induced loops: are their words in the row's own list?
    tot = inown = 0
    for (arm, L, s, q) in [k for k in rows if k[0] in LOW]:
        g0 = loop_gram(bat[(arm, "none", L, s, q)]["answer"])
        g1 = loop_gram(bat[(arm, "own", L, s, q)]["answer"])
        if g0 is None and g1 is not None:
            tot += 1
            own = {w for t in rows[(arm, L, s, q)]["own_ban_ids"] for w in words(str(tokens.get(str(int(t)), "")))}
            if set(g1) & own:
                inown += 1
    print(f"F. own-induced loops: {tot}; of which the loop phrase holds an own-list word: {inown}")

    # ---- G. none-ended and matched ended counts (pooled low-k)
    ks = [(a, L, s, q) for a in LOW for L in range(19, 27) for s in range(12) for q in [8, 9, 10, 11, 12, 14, 15]]
    for ban in ("own", "matched", "mass"):
        ended = sum(ltrue(a, "none", L, s, q) and not ltrue(a, ban, L, s, q) for a, L, s, q in ks)
        ind = sum((not ltrue(a, "none", L, s, q)) and ltrue(a, ban, L, s, q) for a, L, s, q in ks)
        print(f"G. {ban}: ended {ended}, induced {ind}, net removed {ended-ind}")

    # ---- H. repaired vs non-repaired none loop rates (is the paired set different?)
    for flag in (True, False):
        sub = [k for k in ks if masses[k].get("paired_set") and bool(masses[k].get("repaired")) == flag]
        nl = sum(ltrue(*k[:1], "none", *k[1:]) for k in sub)
        ow = sum(ltrue(*k[:1], "own", *k[1:]) for k in sub)
        mm = sum(ltrue(*k[:1], "mass", *k[1:]) for k in sub)
        print(f"H. repaired={flag}: n={len(sub)} loops none {nl} own {ow} mass {mm}")

    # ---- I. Gate 2 and plain.json
    pj = json.loads((R / "plain.json").read_text(encoding="utf-8"))["answers"]
    no_by = {r["question_index"]: r for r in no}
    same = all(len(p["answer_ids"]) == no_by[p["question_index"]]["token_count"] for p in pj)
    print("I. plain.json answer_ids lengths equal no_push token_count:", same,
          "; plain.json qi:", sorted(p["question_index"] for p in pj))
    print("   no-push loop/broken:", {q: (loop_gram(r["answer"]) is not None, score165.primary_broken(r["answer"])) for q, r in no_by.items()})

    # ---- J. Part 164 phrasing 13 and the 16-phrasing plain floor
    p164 = [r for r in jl(ROOT / "results/part164/answers.jsonl") if r.get("stage") == "no_push"]
    print("J. part164 no_push answers:", len(p164), "tokens:", sorted(set(r.get("token_count") for r in p164))[:10])
    loopers = [(r["question_index"], r["token_count"]) for r in p164 if loop_gram(r["answer"]) is not None and r.get("token_count", 0) <= 80]
    print("   looping plain answers (<=80 tokens):", loopers)

    # ---- K. Part 165 phrasings and seeds vs Part 168
    p165 = jl(ROOT / "results/part165/answers.jsonl")
    qs165 = sorted({r.get("question_index") for r in p165 if r.get("question_index") is not None})
    print("K. part165 question indices:", qs165, "| part168 phrasings: [8, 9, 10, 11, 12, 14, 15]")
    print("   part165 arms:", sorted({r.get("arm") for r in p165 if r.get("arm")})[:12])


if __name__ == "__main__":
    main()
