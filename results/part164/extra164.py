#!/usr/bin/env python3
"""Recompute Part 164 exploratory summaries from its saved rows."""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "part86"))
import triage86

QWEN = re.compile(r"\bQwen\b")
QWEN_I = re.compile(r"\bQwen\b", re.I)
EXCLUDE = re.compile(r"alibaba|tongyi|通义", re.I)
Q_INITIAL = re.compile(r"\b(?:Q\d[\w.-]*|Qian\w*|Qiao\w*|Qiang\w*|Q)\b", re.I)
FIRST = re.compile(r"(?<=[.!?])\s|\n")


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), int(k) + (k % 1 > 0)
    return xs[lo] if lo == hi else xs[lo] * (hi-k) + xs[hi] * (k-lo)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def pair_ratio(nmap, rmap, seed=164, reps=10000):
    pairs = sorted(set(nmap) & set(rmap))
    pairs = [p for p in pairs if nmap[p] is not None and rmap[p] is not None]
    n, r = mean([nmap[p] for p in pairs]), mean([rmap[p] for p in pairs])
    draws, rng = [], random.Random(seed)
    for _ in range(reps):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs] if pairs else []
        nm, rm = mean([nmap[p] for p in sample]), mean([rmap[p] for p in sample])
        if rm and rm > 0:
            draws.append(nm / rm)
    return {"neuron": n, "random": r, "ratio": n/r if r else None,
            "interval_95": [percentile(draws, .025), percentile(draws, .975)], "pairs": len(pairs)}


def first_sentence(s):
    return FIRST.split(s.strip(), maxsplit=1)[0]


def loop_broken(a):
    words = [w.casefold() for w in triage86.WORDS.findall(a if isinstance(a, str) else "")]
    if len(words) < 4:
        return True
    from collections import Counter
    counts = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
    return any(n >= 3 for n in counts.values())


def eligible(a, variant="base"):
    is_broken = loop_broken(a) if variant == "loops_override" else triage86.triage_answer(a) == "broken"
    if not isinstance(a, str) or not a.strip() or is_broken:
        return False
    if variant in ("qian_tongyi", "qian_tongyi_scored"):
        return not re.search(r"alibaba|通义", a, re.I)
    if variant == "first":
        return not EXCLUDE.search(first_sentence(a))
    return not EXCLUDE.search(a)


def hit(a, variant="base"):
    if variant == "casefold":
        return bool(re.search(r"\bQwen\b", a, re.I))
    if variant == "casefold_substring":
        return bool(re.search(r"qwen", a, re.I))
    if variant == "case_sensitive_substring":
        return "Qwen" in a
    if variant == "qian_tongyi_scored":
        return bool(QWEN.search(a) or re.search(r"\b(?:Qianwen|Tongyi)\b", a, re.I))
    if variant == "qian_tongyi":
        return bool(QWEN.search(a))
    if variant == "q_initial":
        return bool(QWEN.search(a) or Q_INITIAL.search(a))
    if variant == "first":
        return bool(QWEN.search(first_sentence(a)))
    return bool(QWEN.search(a))


def build_rates(rows, units, variant="base", broken_fill=None, only_layer=None, omit_unit=None, min_eligible=1):
    ans = defaultdict(list)
    for row in rows:
        if row.get("stage") == "battery":
            ans[(row["unit_id"], float(row["dose"]))].append(row["answer"])
    maps = {d: {a: {} for a in ("neuron", "random")} for d in (.8, 1., 1.2)}
    pair_ids = {}
    for u in units:
        arm, uid, pair = u["arm"], u["unit_id"], u["pair_id"]
        pair_ids[(arm, uid)] = pair
        if only_layer is not None and int(u["layer"]) != only_layer:
            continue
        if omit_unit is not None and uid == omit_unit:
            continue
        for d in ((.8,) if arm == "neuron" else (.8, 1., 1.2)):
            aa = ans[(uid, d)]
            good = [x for x in aa if eligible(x, variant)]
            if broken_fill is not None:
                vals = [int(hit(x, variant)) for x in good]
                vals += [int(broken_fill)] * sum(triage86.triage_answer(x) == "broken" for x in aa)
                rate = mean(vals) if vals else None
            else:
                rate = mean([int(hit(x, variant)) for x in good]) if len(good) >= min_eligible else None
            maps[d][arm][pair] = rate
    return maps


def ratio_for(maps, dose=.8):
    return pair_ratio(maps[dose]["neuron"], maps[dose]["random"])


def main():
    score = json.loads((HERE / "score.json").read_text())
    rows = read_jsonl(HERE / "answers.jsonl")
    units = json.loads((ROOT / "part164/units164.json").read_text())
    battery = [r for r in rows if r.get("stage") == "battery"]
    base = build_rates(rows, units)
    qwen = ratio_for(base)
    assert abs(qwen["ratio"] - score["predictions"]["164.1"]["estimate_ratio"]) < 1e-12
    print("REGISTERED QWEN RULE REPRODUCTION")
    print("Rule: case-sensitive \\bQwen\\b; drop triage-broken and any Alibaba/Tongyi/通义 answer; average eligible answer rates within units, then across units.")
    print(json.dumps(qwen, indent=2))
    print("164.1 arm rates:", json.dumps(score["rates"]["qwen_rate"]["0.8"], indent=2))

    print("\nFRAGILITY TABLE (paired unit bootstrap, seed 164, 10,000 replicates)")
    variants = [("Registered", "base", None, None, None, 1),
                ("Broken scored Qwen-gone", "base", False, None, None, 1),
                ("Broken scored Qwen-present", "base", True, None, None, 1),
                ("Casefold word match", "casefold", None, None, None, 1),
                ("Casefold substring", "casefold_substring", None, None, None, 1),
                ("Case-sensitive substring", "case_sensitive_substring", None, None, None, 1),
                ("Tongyi and Qianwen kept", "qian_tongyi", None, None, None, 1),
                ("Tongyi and Qianwen kept and scored Qwen", "qian_tongyi_scored", None, None, None, 1),
                ("First sentence only", "first", None, None, None, 1),
                ("Loops override kept", "loops_override", None, None, None, 1)]
    print("variant | unit neuron | unit random | unit ratio (95% interval) | pooled neuron | pooled random | pooled ratio (95% interval) | pairs")
    for label, variant, fill, layer, omit, minimum in variants:
        maps = build_rates(rows, units, variant, fill, layer, omit, minimum)
        v = ratio_for(maps)
        # Answer-weighted ratio, pair bootstrap, using the same per-answer eligibility variant.
        by_pair = {a: defaultdict(lambda: [0, 0]) for a in ("neuron", "random")}
        for u in units:
            if layer is not None and int(u["layer"]) != layer: continue
            if omit is not None and u["unit_id"] == omit: continue
            arm, uid, pair = u["arm"], u["unit_id"], u["pair_id"]
            ds = (.8,) if arm == "neuron" else (.8,)
            for d in ds:
                aa = [r["answer"] for r in battery if r["unit_id"] == uid and float(r["dose"]) == d]
                good = [a for a in aa if eligible(a, variant)]
                hits = sum(hit(a, variant) for a in good)
                if fill is not None:
                    nbroken = sum(triage86.triage_answer(a) == "broken" for a in aa)
                    hits += nbroken * int(fill); good_count = len(good) + nbroken
                else: good_count = len(good)
                by_pair[arm][pair] = [hits, good_count]
        pairs=sorted(set(by_pair["neuron"]) & set(by_pair["random"]))
        pooled_draws=[]; rng=random.Random(164)
        for _ in range(10000):
            sample=[pairs[rng.randrange(len(pairs))] for _ in pairs]
            totals={a:[sum(by_pair[a][x][0] for x in sample),sum(by_pair[a][x][1] for x in sample)] for a in ("neuron","random")}
            if totals["neuron"][1] and totals["random"][1] and totals["random"][0]:
                pooled_draws.append((totals["neuron"][0]/totals["neuron"][1])/(totals["random"][0]/totals["random"][1]))
        totals={a:[sum(vv[0] for vv in by_pair[a].values()),sum(vv[1] for vv in by_pair[a].values())] for a in ("neuron","random")}
        pn=totals["neuron"][0]/totals["neuron"][1]; pr=totals["random"][0]/totals["random"][1]
        pci=[percentile(pooled_draws,.025),percentile(pooled_draws,.975)]
        print(f"{label} | {v['neuron']:.4f} | {v['random']:.4f} | {v['ratio']:.4f} ({v['interval_95'][0]:.4f}, {v['interval_95'][1]:.4f}) | {pn:.4f} | {pr:.4f} | {pn/pr:.4f} ({pci[0]:.4f}, {pci[1]:.4f}) | {v['pairs']}")

    print("\nLEAVE ONE LAYER OUT")
    loo = []
    for layer in range(19, 27):
        # Build full rates excluding all units in this layer.
        m = build_rates(rows, [u for u in units if int(u["layer"]) != layer])
        v = ratio_for(m); loo.append((layer, v))
        print(f"omit layer {layer}: ratio {v['ratio']:.4f} ({v['interval_95'][0]:.4f}, {v['interval_95'][1]:.4f}), {v['pairs']} pairs")
    print(f"range: {min(v['ratio'] for _,v in loo):.4f} to {max(v['ratio'] for _,v in loo):.4f}")

    print("\nELIGIBLE ANSWER COUNTS AT +0.8")
    for arm in ("neuron", "random"):
        units_arm = [u for u in units if u["arm"] == arm]
        chosen=[r for r in battery if r["arm"]==arm and float(r["dose"])==.8]
        n = sum(eligible(r["answer"]) for r in chosen)
        hits=sum(eligible(r["answer"]) and hit(r["answer"]) for r in chosen)
        print(f"{arm}: {hits}/{n} eligible answers; {n} / 1600 eligible; {n/1600:.4%}")

    print("\nPER-QUESTION QWEN RATES (+0.8, answer weighted among eligible)")
    for qi in range(16):
        out = {}
        for arm in ("neuron", "random"):
            rs = [r for r in battery if r["arm"] == arm and float(r["dose"]) == .8 and r["question_index"] == qi and eligible(r["answer"])]
            out[arm] = {"qwen": sum(hit(r["answer"]) for r in rs), "eligible": len(rs), "rate": sum(hit(r["answer"]) for r in rs)/len(rs) if rs else None}
        print(qi, json.dumps(out))

    print("\nCORRELATION ACROSS PHRASINGS BETWEEN ARMS")
    xs, ys = [], []
    for qi in range(16):
        rates = []
        for arm in ("neuron", "random"):
            rs = [r for r in battery if r["arm"] == arm and float(r["dose"]) == .8 and r["question_index"] == qi and eligible(r["answer"])]
            rates.append(sum(hit(r["answer"]) for r in rs)/len(rs) if rs else 0)
        xs.append(rates[0]); ys.append(rates[1])
    def corr(x,y):
        mx,my=mean(x),mean(y); cov=sum((a-mx)*(b-my) for a,b in zip(x,y)); den=(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))**.5
        return cov/den if den else None
    print("Pearson r:", corr(xs,ys), "neuron:", xs, "random:", ys)

    print("\nEXCLUDED NEURON UNITS")
    qcounts=score["rates"]["per_unit_eligibility_counts"]["0.8"]["neuron"]
    excluded=[u for u in units if u["arm"]=="neuron" and qcounts[u["unit_id"]]["eligible"] == 0]
    print("units:", json.dumps(excluded))
    for u in excluded:
        m=build_rates(rows,units,omit_unit=u["unit_id"])
        print("omit",u["unit_id"],json.dumps(ratio_for(m)))
    full_n = sum(v for v in base[.8]["neuron"].values() if v is not None) / 100
    full_r = sum(v for v in base[.8]["random"].values() if v is not None) / 100
    print("including the two zero-eligible neuron units as zero rates:", {"neuron_mean":full_n,"random_mean":full_r,"ratio":full_n/full_r})

    print("\nTRIAGE BREAKAGE AT +0.8 BY FLAG")
    for arm in ("neuron", "random"):
        rs=[r for r in battery if r["arm"]==arm and float(r["dose"])==.8]
        broke=[r for r in rs if triage86.triage_answer(r["answer"])=="broken"]
        loop=[r for r in broke if triage86.triage_answer(r["answer"])=="broken" and len(triage86.WORDS.findall(r["answer"]))>=4]
        print(arm, {"broken":len(broke),"answers":len(rs),"loop_trigram":len(loop),"other_short":len(broke)-len(loop),"rate":len(broke)/len(rs)})

    print("\nTOKEN COUNTS BY ARM AND DOSE")
    for arm,doses in (("neuron",(.8,)),("random",(.8,1.,1.2))):
        for dose in doses:
            rs=[r for r in battery if r["arm"]==arm and float(r["dose"])==dose]
            print(arm,dose,{"mean":mean([r["token_count"] for r in rs]),"cap80":sum(r["token_count"]>=80 for r in rs),"n":len(rs),"share":sum(r["token_count"]>=80 for r in rs)/len(rs)})

    print("\nPART 163 vs PART 164 POOLED QWEN, UNREGISTERED")
    old=read_jsonl(ROOT/"results/part163/answers.jsonl")
    for arm in ("neuron","random"):
        counts=[]
        for part, rs in ((163,old),(164,battery)):
            chosen=[r for r in rs if r.get("arm")==arm and r.get("stage","battery")=="battery" and (r.get("kind")=="maker" or part==164) and (part==163 or float(r["dose"])==.8)]
            elig=[r for r in chosen if eligible(r["answer"])]
            counts.append((sum(hit(r["answer"]) for r in elig),len(elig)))
        n=sum(x[0] for x in counts); d=sum(x[1] for x in counts)
        print(arm,{"part163":counts[0],"part164":counts[1],"pooled":(n,d,n/d if d else None)})

    print("\nEXAMPLE ANSWERS FOR README")
    categories={"excluded":lambda a: triage86.triage_answer(a)!="broken" and bool(EXCLUDE.search(a)),
                "kept":lambda a: triage86.triage_answer(a)=="kept",
                "broken":lambda a: triage86.triage_answer(a)=="broken",
                "renamed":lambda a: triage86.kept_only_by_wider_rule(a)}
    for name,fn in categories.items():
        found=next((r for r in battery if r["arm"]=="random" and float(r["dose"])==.8 and fn(r["answer"])),None)
        if found:
            a=found["answer"]
            triage="identity recognised" if triage86.triage_answer(a)=="kept" else triage86.triage_answer(a)
            print(name,json.dumps({"arm":found["arm"],"dose":found["dose"],"unit_id":found["unit_id"],
                                   "triage":triage,"qwen_denominator":eligible(a),"qwen_present":hit(a),
                                   "question":found["question"],"answer":a},ensure_ascii=False))


    print("\nGATE 2 AND NAMED FORMS")
    plain = [r["answer"] for r in rows if r.get("stage") == "no_push"]
    print("plain rows", len(plain), "registered", sum(bool(QWEN.search(x)) for x in plain),
          "case-insensitive substring", sum(bool(re.search(r"qwen", x, re.I)) for x in plain))
    for arm, dose in (("neuron", .8), ("random", .8), ("random", 1.), ("random", 1.2)):
        rs = [r for r in battery if r["arm"] == arm and float(r["dose"]) == dose]
        eligible_rs = [r for r in rs if eligible(r["answer"])]
        patterns=(("Qwen", r"\bQwen\b"), ("Qianwen", r"\bQianwen\b"), ("Tongyi", r"\bTongyi\b"))
        forms = {name: sum(bool(re.search(pattern, r["answer"], 0 if name == "Qwen" else re.I)) for r in eligible_rs) for name, pattern in patterns}
        raw_forms = {name: sum(bool(re.search(pattern, r["answer"], 0 if name == "Qwen" else re.I)) for r in rs) for name, pattern in patterns}
        raw_digit_difference = sum("Qwen" in r["answer"] for r in rs) - raw_forms["Qwen"]
        digit_forms = {name:sum(bool(re.search(pattern, r["answer"])) for r in eligible_rs) for name,pattern in
                       (("Qwen",r"\bQwen\d"),("Qianwen",r"\bQianwen\d"),("Tongyi",r"\bTongyi\d"))}
        print(arm, dose, {"eligible_answers":len(eligible_rs),"raw_forms":raw_forms,"eligible_forms": forms,
                          "raw_Qwen_substring_minus_word":raw_digit_difference,"eligible_digit_forms": digit_forms})

    print("\nPHRASING DIRECTION AND FIRST SENTENCE")
    lower = 0
    for qi in range(16):
        vals=[]
        for arm in ("neuron", "random"):
            rs=[r for r in battery if r["arm"]==arm and float(r["dose"])==.8 and r["question_index"]==qi and eligible(r["answer"])]
            vals.append(sum(hit(r["answer"]) for r in rs)/len(rs) if rs else 0)
        lower += vals[0] < vals[1]
    print("neuron lower phrasings", lower, "of", 16, "two-sided sign test p", 2*sum(__import__("math").comb(16,k) for k in range(0, min(lower,16-lower)+1))/2**16)
    print("bare Qwen mentions outside first sentence", {arm: sum(hit(r["answer"]) and not hit(first_sentence(r["answer"])) for r in battery if r["arm"]==arm and float(r["dose"])==.8) for arm in ("neuron","random")})

    print("\nLOOPS THAT TRIAGE LABELS KEPT")
    loops_by_arm={}
    for arm in ("neuron","random"):
        rs=[r for r in battery if r["arm"]==arm and float(r["dose"])==.8]
        kept_loops=[r for r in rs if triage86.triage_answer(r["answer"])=="kept" and loop_broken(r["answer"])]
        loops_by_arm[arm]=len(kept_loops)
        print(arm, len(kept_loops), "of", sum(triage86.triage_answer(r["answer"])=="kept" for r in rs), "identity-recognised answers")
    loop_broken_rates={a:{} for a in ("neuron","random")}
    for u in units:
        aa=[r["answer"] for r in battery if r["unit_id"]==u["unit_id"] and float(r["dose"])==.8]
        loop_broken_rates[u["arm"]][u["pair_id"]]=sum(loop_broken(a) for a in aa)/len(aa)
    print("loops-override broken ratio",json.dumps(pair_ratio(loop_broken_rates["neuron"],loop_broken_rates["random"])))
    print("loops-override broken answers",{a:sum(loop_broken(r["answer"]) for r in battery if r["arm"]==a and float(r["dose"])==.8) for a in ("neuron","random")})

    print("\nBREAKAGE BY LAYER AND ROBUSTNESS")
    for layer in range(19,27):
        out={}
        for arm in ("neuron","random"):
            rs=[r for r in battery if r["arm"]==arm and float(r["dose"])==.8 and int(r["layer"])==layer]
            out[arm]=(sum(triage86.triage_answer(r["answer"])=="broken" for r in rs),len(rs))
        print(layer,out)
    for arm in ("neuron","random"):
        unit_counts=defaultdict(list)
        for r in battery:
            if r["arm"]==arm and float(r["dose"])==.8:
                unit_counts[r["unit_id"]].append(r["answer"])
        print(arm,"units any broken",sum(any(triage86.triage_answer(a)=="broken" for a in xs) for xs in unit_counts.values()),
              "units >=half",sum(sum(triage86.triage_answer(a)=="broken" for a in xs)>=len(xs)/2 for xs in unit_counts.values()))

    print("\nDOSE HIT COUNTS AND BOOTSTRAP CHANCES")
    for dose in (1.,1.2):
        rs=[r for r in battery if r["arm"]=="random" and float(r["dose"])==dose]
        elig=[r for r in rs if eligible(r["answer"])]
        hit_rows=[r for r in elig if hit(r["answer"])]
        print("random",dose,"hits",len(hit_rows),"eligible",len(elig),
              "directions_with_hit",sorted({r["unit_id"] for r in hit_rows}) if dose==1.2 else None)
    p1641=base[.8]["neuron"]; r1641=base[.8]["random"]
    valid=sorted(p for p in set(p1641)&set(r1641) if p1641[p] is not None and r1641[p] is not None)
    rng=random.Random(164); ratios=[]
    for _ in range(10000):
        draw=[valid[rng.randrange(len(valid))] for _ in valid]
        den=mean([r1641[p] for p in draw])
        ratios.append(mean([p1641[p] for p in draw])/den if den else float("inf"))
    print("164.1 bootstrap chance ratio < 0.7",sum(x<.7 for x in ratios)/len(ratios))
    broken={a:{} for a in ("neuron","random")}
    for u in units:
        aa=[r["answer"] for r in battery if r["unit_id"]==u["unit_id"] and float(r["dose"])==.8]
        broken[u["arm"]][u["pair_id"]]=sum(triage86.triage_answer(a)=="broken" for a in aa)/len(aa)
    allpairs=sorted(set(broken["neuron"])&set(broken["random"]))
    rng=random.Random(164); brats=[]
    for _ in range(10000):
        draw=[allpairs[rng.randrange(len(allpairs))] for _ in allpairs]
        den=mean([broken["random"][p] for p in draw])
        brats.append(mean([broken["neuron"][p] for p in draw])/den if den else float("inf"))
    print("164.2 bootstrap chance broken ratio < 2",sum(x<2 for x in brats)/len(brats))
    strat_rng=random.Random(164); strat_ratios=[]
    layers=sorted({int(u["layer"]) for u in units})
    layer_pairs={layer:[u["pair_id"] for u in units if u["arm"]=="neuron" and int(u["layer"])==layer]
                 for layer in layers}
    for _ in range(10000):
        ns=[]; rs=[]
        for layer, ps in layer_pairs.items():
            draw=[ps[strat_rng.randrange(len(ps))] for _ in ps]
            ns.extend(broken["neuron"][p] for p in draw); rs.extend(broken["random"][p] for p in draw)
        strat_ratios.append(mean(ns)/mean(rs))
    print("layer-stratified broken ratio interval",percentile(strat_ratios,.025),percentile(strat_ratios,.975))
    loo=[]
    for layer in layers:
        ps=[p for p in allpairs if next(int(u["layer"]) for u in units if u["arm"]=="neuron" and u["pair_id"]==p)!=layer]
        loo.append(mean([broken["neuron"][p] for p in ps])/mean([broken["random"][p] for p in ps]))
    print("broken ratio leave-one-layer-out range",min(loo),max(loo))
    for arm in ("neuron","random"):
        rs=[r for r in battery if r["arm"]==arm and float(r["dose"])==.8 and r["token_count"]<80]
        print(arm,"uncapped broken",sum(triage86.triage_answer(r["answer"])=="broken" for r in rs),"of",len(rs))
    maps44=build_rates(rows,units); nmap=maps44[.8]["neuron"]; rmap=maps44[1.]["random"]
    pairs44=sorted(p for p in set(nmap)&set(rmap) if nmap[p] is not None and rmap[p] is not None)
    rng=random.Random(164); inside=finite=0
    for _ in range(10000):
        draw=[pairs44[rng.randrange(len(pairs44))] for _ in pairs44]
        den=mean([rmap[p] for p in draw])
        if den>0:
            finite+=1; inside+=int(.7<=mean([nmap[p] for p in draw])/den<=1.3)
    print("164.4 chance ratio in [0.7,1.3]",inside/finite,"finite draws",finite,"of",10000)
    pool_counts={a:{} for a in ("neuron","random")}
    for u in units:
        dose=.8 if u["arm"]=="neuron" else 1.
        eligible_rows=[r for r in battery if r["unit_id"]==u["unit_id"] and float(r["dose"])==dose and eligible(r["answer"])]
        pool_counts[u["arm"]][u["pair_id"]]=(sum(hit(r["answer"]) for r in eligible_rows),len(eligible_rows))
    pool_pairs=sorted(set(pool_counts["neuron"])&set(pool_counts["random"]))
    rng=random.Random(164); pool_finite=0
    for _ in range(10000):
        draw=[pool_pairs[rng.randrange(len(pool_pairs))] for _ in pool_pairs]
        counts={a:(sum(pool_counts[a][p][0] for p in draw),sum(pool_counts[a][p][1] for p in draw)) for a in ("neuron","random")}
        pool_finite+=int(counts["neuron"][1]>0 and counts["random"][1]>0 and counts["random"][0]>0)
    print("164.4 answer-weighted finite bootstrap draws",pool_finite,"of",10000)
    maps12=build_rates(rows,units); vals08=maps12[.8]["random"]; vals12=maps12[1.2]["random"]
    pairs12=sorted(set(vals08)&set(vals12)); rng=random.Random(164); zero=underhalf=0
    for _ in range(10000):
        draw=[pairs12[rng.randrange(len(pairs12))] for _ in pairs12]
        num=mean([vals12[p] for p in draw]); den=mean([vals08[p] for p in draw])
        zero+=int(num==0); underhalf+=int(den>0 and num/den<.5)
    print("164.5 zero numerator mean resamples",zero,"of",10000,"under-half chance",underhalf/10000)

    print("\nPART 161 EXACT +0.8 COMPARISON FROM STORED ROWS")
    p161=read_jsonl(ROOT/"results/part161/answers.jsonl")
    p163=read_jsonl(ROOT/"results/part163/answers.jsonl")
    maker_questions={r["question"] for r in p163 if r.get("stage")=="battery" and r.get("kind")=="maker"}
    for label, keep_questions in (("Part 163's 8 maker phrasings",maker_questions),("all 24 phrasings",None)):
        arm_rates={}
        for arm in ("neuron","random"):
            selected=[r for r in p161 if r.get("stage")=="ladder" and r.get("arm")==arm and float(r["dose"])==.8 and (keep_questions is None or r["question"] in keep_questions)]
            eligible_rows=[r for r in selected if eligible(r["answer"])]
            arm_rates[arm]={"hits":sum(hit(r["answer"]) for r in eligible_rows),"eligible":len(eligible_rows),
                            "pooled":sum(hit(r["answer"]) for r in eligible_rows)/len(eligible_rows),
                            "units":{}}
            by=defaultdict(list)
            for r in selected:
                if eligible(r["answer"]): by[r["unit_id"]].append(int(hit(r["answer"])))
            arm_rates[arm]["units"]={u:mean(v) for u,v in by.items()}
            arm_rates[arm]["unit_mean"]=mean([mean(v) for v in by.values()])
        print(label,json.dumps(arm_rates,ensure_ascii=False))


if __name__ == "__main__":
    main()
