#!/usr/bin/env python3
"""Cold-check recompute for Part 168, from raw result files.

Run from the repo root:  /home/elisepc/neuron-probe/.venv/bin/python -I -B cold168/recompute.py
Own loop rule (trigram count >= 3, first most-frequent trigram); broken = Part 165 primary_broken
(imported, not re-derived). Everything else is recomputed here, not read from score.json.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.cwd()
R = ROOT / "results/part168"
sys.path[:0] = [str(ROOT), str(ROOT / "part86")]
import triage86  # noqa: E402
from part165 import score165  # noqa: E402

WORD = re.compile(r"[\w]+(?:['’][\w]+)*")
assert WORD.pattern == triage86.WORDS.pattern, "word regex differs from triage86"
ARMS_LOW = ["n+", "k4"]
LAYERS = list(range(19, 27))
SLOTS = list(range(12))
PH = [8, 9, 10, 11, 12, 14, 15]
BANS = ["none", "own", "matched", "mass"]


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


def content(gram, stop):
    return {w for w in (gram or ()) if w not in stop}


def main():
    ans = jl(R / "answers.jsonl")
    A, NO = {}, []
    for r in ans:
        if r["stage"] == "battery":
            k = (r["arm"], r["ban"], r["layer"], r["slot"], r["question_index"])
            assert k not in A, k
            A[k] = r
        else:
            NO.append(r)
    print(f"battery {len(A)} (expect 7392), no_push {len(NO)} (expect 7)")
    rows = {(r["arm"], r["layer"], r["slot"], r["question_index"]): r for r in jl(R / "rows.jsonl")}
    masses = {(m["arm"], m["layer"], m["slot"], m["question_index"]): m for m in jl(R / "masses.jsonl")}
    tokens = json.loads((R / "tokens.json").read_text(encoding="utf-8"))
    lt = json.loads((ROOT / "results/part166/loop_tokens.json").read_text(encoding="utf-8"))
    stop = {str(w).casefold() for w in lt["stop_list"]}
    print("rows", len(rows), "masses", len(masses), "stop words", len(stop))

    # cached per-answer facts
    GRAM, BROK, EOS, TC = {}, {}, {}, {}
    for k, r in A.items():
        GRAM[k] = loop_gram(r["answer"])
        BROK[k] = bool(score165.primary_broken(r["answer"]))
        TC[k] = int(r["token_count"])
        EOS[k] = bool(r["ended_eos_before_80"])
    LOOP = {k: v is not None for k, v in GRAM.items()}
    print("EOS flag == (token_count < 80) for all battery:", all(EOS[k] == (TC[k] < 80) for k in A))
    print("token_count==80 battery:", sum(TC[k] == 80 for k in A))

    def key(arm, ban, L, s, q):
        return (arm, ban, L, s, q)

    def lowkeys(arms=ARMS_LOW, layers=LAYERS, phr=PH, paired=False):
        out = []
        for arm in arms:
            for L in layers:
                for s in SLOTS:
                    for q in phr:
                        if paired and not masses[(arm, L, s, q)].get("paired_set"):
                            continue
                        out.append((arm, L, s, q))
        return out

    def rate(ban, ks, field=LOOP):
        v = [field[key(a, ban, L, s, q)] for (a, L, s, q) in ks]
        return sum(v) / len(v) if v else None

    # ---------- 1. battery table ----------
    print("\n== per arm x ban (recomputed) ==")
    table = {}
    for arm in ["n+", "k4", "full"]:
        for ban in (BANS if arm != "full" else ["none", "own", "matched"]):
            ks = [(arm, L, s, q) for L in LAYERS for s in SLOTS for q in PH]
            recs = [key(arm, ban, *k[1:]) for k in ks]
            n = len(recs)
            loop = sum(LOOP[x] for x in recs) / n
            broken = sum(BROK[x] for x in recs) / n
            joint = sum((not LOOP[x]) and (not BROK[x]) for x in recs) / n
            mtc = sum(TC[x] for x in recs) / n
            eos = sum(EOS[x] for x in recs) / n
            table[(arm, ban)] = dict(n=n, loop=loop, broken=broken, joint=joint, mtc=mtc, eos=eos)
            print(f"{arm:5s} {ban:8s} n={n} loop={loop:.4f} broken={broken:.4f} joint={joint:.4f} "
                  f"meanTC={mtc:.2f} EOS<80={eos:.4f}")

    # ---------- 2. pooled low-k headline ----------
    ks_all = lowkeys()
    none_r = rate("none", ks_all)
    own_r = rate("own", ks_all)
    mat_r = rate("matched", ks_all)
    mas_r = rate("mass", ks_all)
    print("\n== pooled n+ & k4 ==")
    print(f"none {none_r:.5f} own {own_r:.5f} matched {mat_r:.5f} mass {mas_r:.5f}")
    print(f"168.3 own cut {1 - own_r / none_r:.5f}  matched cut {1 - mat_r / none_r:.5f}  mass cut {1 - mas_r / none_r:.5f}")
    print(f"168.4 matched-own gap {mat_r - own_r:.5f} (bar 0.05)")
    print(f"168.1-style: broken n+ {table[('n+','none')]['broken']:.4f} / full {table[('full','none')]['broken']:.4f} = "
          f"{table[('n+','none')]['broken'] / table[('full','none')]['broken']:.4f}")
    full_none = rate("none", lowkeys(arms=["full"]), field=LOOP)
    full_own = rate("own", lowkeys(arms=["full"]), field=LOOP)
    print(f"168.6 full own cut {1 - full_own / full_none:.4f} (full none loops={sum(LOOP[('full','none',L,s,q)] for L in LAYERS for s in SLOTS for q in PH)})")

    # ---------- 3. paired set (168.8) ----------
    paired = lowkeys(paired=True)
    unp = [k for k in ks_all if k not in set(paired)]
    print(f"\n== paired set: {len(paired)} rows, unmatched {len(unp)} ==")
    print(f"paired none {rate('none', paired):.5f} own {rate('own', paired):.5f} mass {rate('mass', paired):.5f}")
    print(f"168.8 gap mass-own {rate('mass', paired) - rate('own', paired):.5f} (bar 0.05)")
    rep = [k for k in paired if masses[k].get("repaired")]
    nonrep = [k for k in paired if not masses[k].get("repaired")]
    print(f"repaired among paired: {len(rep)}; non-repaired paired: {len(nonrep)}")
    print(f"  168.8 on non-repaired: own {rate('own', nonrep):.4f} mass {rate('mass', nonrep):.4f} gap {rate('mass', nonrep) - rate('own', nonrep):.4f}")
    print(f"  168.8 on repaired:     own {rate('own', rep):.4f} mass {rate('mass', rep):.4f} gap {rate('mass', rep) - rate('own', rep):.4f}")
    print(f"  all-row mass rate {mas_r:.5f}; bounds with unmatched as no/all loops: "
          f"{sum(LOOP[('n+' if k[0]=='n+' else 'k4','mass',*k[1:])] for k in paired)/len(ks_all):.5f} / "
          f"{(sum(LOOP[('n+' if k[0]=='n+' else 'k4','mass',*k[1:])] for k in paired)+len(unp))/len(ks_all):.5f}")
    rep_plain = [masses[k]["plain_overlap_mass"] for k in rep]
    nonrep_plain = [masses[k]["plain_overlap_mass"] for k in nonrep]
    print(f"  mean plain-token overlap of mass list: repaired {sum(rep_plain)/len(rep_plain):.3f}, non-repaired {sum(nonrep_plain)/len(nonrep_plain):.3f}")
    print("  paired by arm:", Counter(k[0] for k in paired))

    # ---------- 4. fragility of 168.3 and 168.4 / 168.8 ----------
    def frag(label, ks, paired_ks=None):
        c = 1 - rate("own", ks) / rate("none", ks)
        g = rate("matched", ks) - rate("own", ks)
        pk = [k for k in (paired_ks if paired_ks is not None else []) if k in set(ks)]
        g8 = (rate("mass", pk) - rate("own", pk)) if pk else float("nan")
        print(f"{label:28s} n={len(ks):4d} cut={c:+.4f} ({'>=30' if c >= .30 else '<30 FAIL'})  "
              f"168.4gap={g:+.4f} ({'pass' if g >= .05 else 'FAIL'})  168.8gap={g8:+.4f} ({'pass' if g8 >= .05 else 'FAIL'})")

    print("\n== fragility: drop one arm / layer / phrasing (pooled n+&k4 unless stated) ==")
    frag("all", ks_all, paired)
    frag("n+ alone", lowkeys(arms=["n+"]), paired)
    frag("k4 alone", lowkeys(arms=["k4"]), paired)
    for L in LAYERS:
        frag(f"drop layer {L}", lowkeys(layers=[x for x in LAYERS if x != L]), paired)
    for q in PH:
        frag(f"drop phrasing {q}", lowkeys(phr=[x for x in PH if x != q]), paired)
    frag("only layers 19-22", lowkeys(layers=[19, 20, 21, 22]), paired)
    frag("only layers 23-26", lowkeys(layers=[23, 24, 25, 26]), paired)
    for q in PH:
        frag(f"only phrasing {q}", lowkeys(phr=[q]), paired)
    for arm in ARMS_LOW:
        for q in PH:
            frag(f"{arm} only, phr {q}", lowkeys(arms=[arm], phr=[q]), paired)

    # ---------- 5. discordance, rows carrying the cut ----------
    print("\n== discordant pairs, pooled low-k (1344 rows) ==")
    none_loop = [k for k in ks_all if LOOP[key(k[0], 'none', *k[1:])]]
    own_loop = [k for k in ks_all if LOOP[key(k[0], 'own', *k[1:])]]
    ended = [k for k in none_loop if not LOOP[key(k[0], 'own', *k[1:])]]
    induced = [k for k in ks_all if not LOOP[key(k[0], 'none', *k[1:])] and LOOP[key(k[0], 'own', *k[1:])]]
    print(f"none loops {len(none_loop)}; own loops {len(own_loop)}; none-loops ended under own {len(ended)}; "
          f"induced (none no loop, own loop) {len(induced)} of {ks_all.__len__() - len(none_loop)} eligible")
    print(f"net loops removed = ended - induced = {len(ended) - len(induced)}  (= none loops - own loops = {len(none_loop) - len(own_loop)})")
    print(f"share of none-loop pairs ended = {len(ended)/len(none_loop):.4f}; induced share of eligible = {len(induced)/(len(ks_all)-len(none_loop)):.4f}")
    for arm in ARMS_LOW:
        e = [k for k in ended if k[0] == arm]
        i = [k for k in induced if k[0] == arm]
        print(f"  {arm}: none loops {sum(1 for k in none_loop if k[0]==arm)}, ended {len(e)}, induced {len(i)}")
    units = Counter((k[0], k[1], k[2]) for k in ended)
    print(f"ended pairs spread over {len(units)} (arm,layer,slot) units of 192; top-5 units hold "
          f"{sum(v for _, v in units.most_common(5))} ended; top-10 hold {sum(v for _, v in units.most_common(10))}")
    print("  top ended units:", units.most_common(8))
    phr_e = Counter(k[3] for k in ended)
    print("  ended by phrasing:", dict(sorted(phr_e.items())))
    print("  ended by layer:", dict(sorted(Counter(k[1] for k in ended).items())))
    # removing the most-ended units, cut recomputed
    order = [u for u, _ in units.most_common()]
    for m in (0, 5, 10, 20):
        drop = set(order[:m])
        ks2 = [k for k in ks_all if (k[0], k[1], k[2]) not in drop]
        print(f"  cut after dropping top-{m} ended units: {1 - rate('own', ks2)/rate('none', ks2):+.4f} (n={len(ks2)})")
    # pairs-level exact sign test (ended vs induced) — discordant counts
    from math import comb
    b, c_ = len(ended), len(induced)
    nn = b + c_
    p1 = sum(comb(nn, i) for i in range(0, min(b, c_) + 1)) / 2 ** nn * 2
    print(f"  exact two-sided sign test ended vs induced: p={min(1, p1):.2e}")

    # ---------- 6. length / EOS confound ----------
    print("\n== length / EOS confound for own ==")
    both_cap = [k for k in ks_all if TC[key(k[0], 'none', *k[1:])] == 80 and TC[key(k[0], 'own', *k[1:])] == 80]
    print(f"pairs where BOTH none and own reached 80 tokens: {len(both_cap)} of {len(ks_all)}")
    print(f"  loop rate none {rate('none', both_cap):.4f} own {rate('own', both_cap):.4f} cut {1-rate('own', both_cap)/rate('none', both_cap):+.4f}"
          if both_cap and rate('none', both_cap) else "  no pairs")
    none_eos = [k for k in ks_all if EOS[key(k[0], 'none', *k[1:])]]
    none_cap = [k for k in ks_all if not EOS[key(k[0], 'none', *k[1:])]]
    for lbl, ks in (("none ended before 80", none_eos), ("none reached 80", none_cap)):
        if ks and rate('none', ks):
            print(f"  {lbl}: n={len(ks)} none loop {rate('none', ks):.4f} own loop {rate('own', ks):.4f} cut {1-rate('own', ks)/rate('none', ks):+.4f}")
    own_eos_only = sum(EOS[key(k[0], 'own', *k[1:])] and not EOS[key(k[0], 'none', *k[1:])] for k in ks_all)
    none_eos_only = sum(EOS[key(k[0], 'none', *k[1:])] and not EOS[key(k[0], 'own', *k[1:])] for k in ks_all)
    print(f"  pairs where own ended early but none did not: {own_eos_only}; the reverse: {none_eos_only}")
    print(f"  mean token count none {sum(TC[key(k[0],'none',*k[1:])] for k in ks_all)/len(ks_all):.3f} own {sum(TC[key(k[0],'own',*k[1:])] for k in ks_all)/len(ks_all):.3f}")
    # loop answers' lengths
    print(f"  mean token count of none loop answers {sum(TC[key(k[0],'none',*k[1:])] for k in none_loop)/len(none_loop):.2f}; of own loop answers {sum(TC[key(k[0],'own',*k[1:])] for k in own_loop)/len(own_loop):.2f}")
    print(f"  mean token count of ended-under-own answers (none side) {sum(TC[key(k[0],'none',*k[1:])] for k in ended)/len(ended):.2f}")

    # ---------- 7. mass ban: where the rise comes from ----------
    print("\n== mass ban rise ==")
    m_loop = [k for k in ks_all if LOOP[key(k[0], 'mass', *k[1:])]]
    m_ind = [k for k in ks_all if not LOOP[key(k[0], 'none', *k[1:])] and LOOP[key(k[0], 'mass', *k[1:])]]
    m_end = [k for k in ks_all if LOOP[key(k[0], 'none', *k[1:])] and not LOOP[key(k[0], 'mass', *k[1:])]]
    print(f"mass loops {len(m_loop)} vs none {len(none_loop)}; induced {len(m_ind)}; ended {len(m_end)}; net +{len(m_loop)-len(none_loop)}")
    mu = Counter((k[0], k[1], k[2]) for k in m_ind)
    print(f"  induced spread over {len(mu)} units; top-5 {sum(v for _,v in mu.most_common(5))}, top-10 {sum(v for _,v in mu.most_common(10))}")
    print("  top induced units:", mu.most_common(8))
    print("  induced by phrasing:", dict(sorted(Counter(k[3] for k in m_ind).items())))
    print("  induced by layer:", dict(sorted(Counter(k[1] for k in m_ind).items())))
    rows_with = len(mu)
    # rows (row = arm,layer,slot,q) with >=1 induced vs rows carrying the rise
    print(f"  pairs with induced loop: {len(m_ind)}; (arm,layer,slot) units with any: {rows_with}")
    # plain overlap as a candidate explanation
    po_ind = [masses[k]["plain_overlap_mass"] for k in m_ind]
    po_all = [masses[k]["plain_overlap_mass"] for k in ks_all]
    print(f"  mean plain-token overlap of mass list: induced rows {sum(po_ind)/len(po_ind):.3f}; all rows {sum(po_all)/len(po_all):.3f}")
    # same test for own-induced
    o_ind = [k for k in induced]
    print(f"  own-induced rows: mean plain overlap of own list {sum(masses[k]['plain_overlap_own'] for k in o_ind)/len(o_ind):.3f}")
    # mass list sizes and the top-token pattern
    print("  mean plain overlap (mass list) by arm:",
          {a: round(sum(masses[k]['plain_overlap_mass'] for k in ks_all if k[0]==a)/sum(1 for k in ks_all if k[0]==a), 3) for a in ARMS_LOW})
    print("  mean plain overlap (own list) by arm:",
          {a: round(sum(masses[k]['plain_overlap_own'] for k in ks_all if k[0]==a)/sum(1 for k in ks_all if k[0]==a), 3) for a in ARMS_LOW})

    # ---------- 8. transitions (168.7, 168.W3, 168.1) ----------
    print("\n== transitions under own (independent classification) ==")
    def own_words(row):
        return {w for t in row["own_ban_ids"] for w in words(str(tokens.get(str(int(t)), "")))}

    def mid_words(row):
        return {w for t in row["rank_51_200_ids"] for w in words(str(tokens.get(str(int(t)), "")))}

    def classify(ban):
        cls = Counter(); bases = []; ind = elig = 0
        for k in ks_all:
            arm, L, s, q = k
            n_ = key(arm, 'none', L, s, q); b_ = key(arm, ban, L, s, q)
            if GRAM[n_] is None:
                elig += 1
                ind += GRAM[b_] is not None
                continue
            row = rows[(arm, L, s, q)]
            ow = content(GRAM[n_], stop)
            ownset = own_words(row)
            held = ow & ownset
            if not held:
                continue
            g1 = GRAM[b_]
            if g1 is None:
                cls["no_loop"] += 1
                continue
            nw = content(g1, stop)
            if held & nw:
                cls["reentry"] += 1
                continue
            if not nw:
                cls["common_only"] += 1
                focal = None
            else:
                focal = next((w for w in g1 if w not in stop and w not in ow), None)
                if focal is None:
                    cls["other"] += 1
                elif focal in ownset:
                    cls["top50"] += 1
                elif focal in mid_words(row):
                    cls["migration"] += 1
                else:
                    cls["outside"] += 1
            if focal is not None:
                oth = [focal in mid_words(rows[(arm, L, t, q)]) for t in SLOTS if t != s]
                bases.append(sum(oth) / len(oth))
            else:
                bases.append(0.0)
        return cls, bases, ind, elig

    cls, bases, ind, elig = classify("own")
    changed = sum(cls[c] for c in ("common_only", "other", "top50", "migration", "outside"))
    print("own fates:", dict(cls), f"(own-word none loops = {sum(cls.values())})")
    print(f"changed loops n={changed}; migration share {cls['migration']/changed:.4f}; baseline {sum(bases)/len(bases):.4f}")
    print(f"168.7 test: share>=1/3 and > baseline -> {cls['migration']/changed >= 1/3 and cls['migration']/changed > sum(bases)/len(bases)}")
    print(f"168.W3 (15-30%): {0.15 <= cls['migration']/changed <= 0.30}")
    print(f"induced own {ind}/{elig} = {ind/elig:.4f}")
    cls_m, _, ind_m, _ = classify("mass")
    print("mass fates:", dict(cls_m))
    cls_x, _, ind_x, _ = classify("matched")
    print("matched fates:", dict(cls_x), f"induced matched {ind_x}/{elig} = {ind_x/elig:.4f}")
    # re-entry view: sensitivity if no-loop rows were counted as changes (no -- report only)
    print(f"if 'no_loop' were counted in the denominator, migration share = {cls['migration']/(changed+cls['no_loop']):.4f}")

    # 168.1 recomputed
    print("\n== 168.1 recomputed ==")
    own_ct = shuf_ct = 0.0
    for arm in ARMS_LOW:
        for L in LAYERS:
            for s in SLOTS:
                for q in PH:
                    k = key(arm, 'none', L, s, q)
                    if GRAM[k] is None:
                        continue
                    ws = content(GRAM[k], stop)
                    ownset = own_words(rows[(arm, L, s, q)])
                    own_ct += sum(w in ownset for w in ws)
                    others = [own_words(rows[(arm, L, t, q)]) for t in SLOTS if t != s]
                    shuf_ct += sum(sum(w in o for o in others) / len(others) for w in ws)
    print(f"content-word own hits {own_ct}, shuffled mean {shuf_ct:.4f}, ratio {own_ct/shuf_ct:.2f}")
    # word concentration of the own hits
    hitw = Counter()
    for arm in ARMS_LOW:
        for L in LAYERS:
            for s in SLOTS:
                for q in PH:
                    k = key(arm, 'none', L, s, q)
                    if GRAM[k] is None:
                        continue
                    for w in content(GRAM[k], stop):
                        if w in own_words(rows[(arm, L, s, q)]):
                            hitw[w] += 1
    print("  own hits by word (top 15):", hitw.most_common(15), "distinct words:", len(hitw))

    # ---------- 9. bootstrap (layer-stratified, slots resampled), seed 168 ----------
    print("\n== bootstrap (own cut 95% interval; my own resampler) ==")
    import numpy as np
    rng = np.random.default_rng(168)
    reps = 10000
    # counts[layer, slot, arm, ban] = (loops, n)
    cnt = np.zeros((len(LAYERS), 12, 2, 4, 2))
    for li, L in enumerate(LAYERS):
        for s in SLOTS:
            for ai, arm in enumerate(ARMS_LOW):
                for bi, ban in enumerate(BANS):
                    for q in PH:
                        cnt[li, s, ai, bi, 0] += LOOP[key(arm, ban, L, s, q)]
                        cnt[li, s, ai, bi, 1] += 1
    tot = np.zeros((reps, 2, 4, 2))
    for li in range(len(LAYERS)):
        idx = rng.integers(0, 12, size=(reps, 12))
        tot += cnt[li][idx].sum(axis=1)
    pooled = lambda bi: tot[:, :, bi, 0].sum(1) / tot[:, :, bi, 1].sum(1)
    cut = 1 - pooled(1) / pooled(0)
    gap = pooled(2) - pooled(1)
    gap8 = pooled(3) - pooled(1)
    for lbl, v in (("own cut", cut), ("matched-own gap", gap), ("mass-own gap (all rows)", gap8)):
        print(f"  {lbl}: 2.5/97.5 = {np.percentile(v, 2.5):.4f} / {np.percentile(v, 97.5):.4f}")

    # ---------- 10. hand-read file integrity ----------
    print("\n== hand read ==")
    judged = {r["blind_id"]: r["judgement"] for r in jl(R / "hand_read_judged.jsonl")}
    keyf = {r["blind_id"]: r for r in jl(R / "hand_read_key.jsonl")}
    print("judged", len(judged), "key", len(keyf), "same ids:", set(judged) == set(keyf))
    print("judgement values:", Counter(judged.values()))
    for ban in BANS:
        ids = [i for i in keyf if keyf[i]["ban"] == ban]
        print(f"  {ban}: n={len(ids)}", Counter(judged[i] for i in ids))
    # key cross-check against answers
    bad = 0
    for i, kr in keyf.items():
        rec = A[(kr["arm"], kr["ban"], kr["layer"], kr["slot"], kr["question_index"])]
        if rec["answer"] not in [x.get("answer") for x in []]:
            pass
    hr = {r["blind_id"]: r for r in jl(R / "hand_read.jsonl")} if (R / "hand_read.jsonl").exists() else {}
    for i, kr in keyf.items():
        rec = A[(kr["arm"], kr["ban"], kr["layer"], kr["slot"], kr["question_index"])]
        if i in hr and hr[i]["answer"] != rec["answer"]:
            bad += 1
    print("hand_read answer text mismatches vs key:", bad, "(hand_read rows", len(hr), ")")
    print("hand-read loop share (own-flag) by ban:",
          {ban: round(sum(LOOP[keyf[i]['arm'] and key(keyf[i]['arm'], ban, keyf[i]['layer'], keyf[i]['slot'], keyf[i]['question_index'])] for i in keyf if keyf[i]['ban']==ban)/30,3) for ban in BANS})

    # ---------- 11. no-push gate ----------
    print("\n== gate 2 (no-push, 7 answers) ==")
    for r in sorted(NO, key=lambda x: x["question_index"]):
        print(f"  q{r['question_index']}: tokens={r['token_count']} loop={loop_gram(r['answer'])} broken={score165.primary_broken(r['answer'])}")
    pj = json.loads((R / "plain.json").read_text(encoding="utf-8"))["answers"]
    print("plain.json entries:", len(pj), "identical to no_push answers:",
          sorted((x.get("question_index"), x.get("answer")) for x in pj) == sorted((x["question_index"], x["answer"]) for x in NO)
          if isinstance(pj, list) and pj and isinstance(pj[0], dict) else type(pj).__name__)


if __name__ == "__main__":
    main()
