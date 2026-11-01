#!/usr/bin/env python3
"""Recompute Part 165 write-up details from saved answer/effect rows."""
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYERS = tuple(range(19, 27))
SLOTS = tuple(range(12))
ARMS = ("n+", "n±", "k4", "k16", "k64", "k256", "k1024", "full", "iso")
LADDER = ("n±", "k4", "k16", "k64", "k256", "k1024", "full")
WORDS = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)
KNOWN_IDENTITY = re.compile(r"\b(?:qwen(?:\s*[- ]?\d+(?:\.\d+)*)?|tongyi|alibaba)\b", re.I)
FIRST_SPLIT = re.compile(r"(?<=[.!?])\s|\n")
SELF_DESC = re.compile(r"(?<![\w.])(?:a\.i\.|ai|assistant|language\s+model|model|chatbot|llm)(?!\w)", re.I)


def read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_sentence(text):
    return FIRST_SPLIT.split(text.strip(), maxsplit=1)[0]


def broken_parts(answer):
    """Independent transcription of Part 86 triage plus the design loop override."""
    answer = answer if isinstance(answer, str) else ""
    triage = False
    # Identity or self-description wins before short/loop checks, as in the design.
    if not KNOWN_IDENTITY.search(answer) and not SELF_DESC.search(first_sentence(answer)):
        words = [x.casefold() for x in WORDS.findall(answer)]
        triage = len(words) < 4 or any(n >= 3 for n in Counter(tuple(words[i:i+3]) for i in range(max(0, len(words)-2))).values())
    words = [x.casefold() for x in WORDS.findall(answer)]
    loops = len(words) >= 4 and any(n >= 3 for n in Counter(tuple(words[i:i+3]) for i in range(len(words)-2)).values())
    short = len(words) < 4 and not KNOWN_IDENTITY.search(answer) and not SELF_DESC.search(first_sentence(answer))
    return triage, loops, bool(triage or loops), short, words


def mean(xs): return sum(xs) / len(xs) if xs else float('nan')

def median(xs):
    ys = sorted(xs)
    if not ys: return float('nan')
    n = len(ys)
    return ys[n//2] if n % 2 else (ys[n//2-1] + ys[n//2]) / 2


def percentile(xs, p):
    ys = sorted(xs)
    x = (len(ys)-1)*p
    lo, hi = int(x), math.ceil(x)
    return ys[lo] if lo == hi else ys[lo]*(hi-x)+ys[hi]*(x-lo)


def rank(xs):
    order = sorted((v, i) for i, v in enumerate(xs)); out=[0.0]*len(xs); i=0
    while i < len(order):
        j=i+1
        while j<len(order) and order[j][0] == order[i][0]: j+=1
        r=((i+1)+j)/2
        for z in range(i,j): out[order[z][1]]=r
        i=j
    return out


def corr(xs, ys):
    if len(xs) < 2: return None
    mx,my=mean(xs),mean(ys)
    a=[x-mx for x in xs]; b=[y-my for y in ys]
    den=math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
    return sum(x*y for x,y in zip(a,b))/den if den else None


def bootstrap(slot_rates, arm_a, arm_b, ratio=False, reps=10000, seed=165):
    keys=[(l,s) for l in LAYERS for s in SLOTS]
    estimate_a=mean([slot_rates[arm_a][k] for k in keys]); estimate_b=mean([slot_rates[arm_b][k] for k in keys])
    est = (estimate_a/estimate_b if estimate_b else (None if estimate_a == 0 else math.inf)) if ratio else estimate_a-estimate_b
    rng=random.Random(seed); draws=[]
    for _ in range(reps):
        sample=[(l,rng.choice(SLOTS)) for l in LAYERS for _ in SLOTS]
        a=mean([slot_rates[arm_a][k] for k in sample]); b=mean([slot_rates[arm_b][k] for k in sample])
        draws.append((a/b if b else (None if a == 0 else math.inf)) if ratio else a-b)
    finite=[x for x in draws if x is not None]
    return est, (percentile(finite,.025), percentile(finite,.975)), sum(x is None for x in draws)


def main():
    answers=read(HERE/'answers.jsonl'); effects=read(HERE/'effects.jsonl'); run1=read(HERE/'effects-run1.jsonl')
    score=json.loads((HERE/'score.json').read_text()); geometry=json.loads((HERE/'geometry.json').read_text())
    baselines={int(r['question_index']):r['answer'] for r in answers if r.get('stage')=='no_push'}
    rows=[r for r in answers if r.get('stage')=='battery']
    state=defaultdict(lambda: defaultdict(list)); per_arm=defaultdict(list); slot={m:{a:{} for a in ARMS} for m in ('broken','triage','loop','changed')}
    for r in rows:
        arm=r['arm']; key=(int(r['layer']),int(r['slot'])); ans=r['answer']; tri,loop,bad,short,words=broken_parts(ans)
        state[arm][key].append((r,tri,loop,bad,short,words)); per_arm[arm].append((r,tri,loop,bad,short,words))
    for arm in ARMS:
        for layer in LAYERS:
            for s in SLOTS:
                vals=state[arm][(layer,s)]
                slot['broken'][arm][(layer,s)]=mean([int(v[3]) for v in vals])
                slot['triage'][arm][(layer,s)]=mean([int(v[1]) for v in vals])
                slot['loop'][arm][(layer,s)]=mean([int(v[2]) for v in vals])
                slot['changed'][arm][(layer,s)]=mean([int(v[0]['answer'].split()[:12] != baselines[int(v[0]['question_index'])].split()[:12]) for v in vals])
    print('INDEPENDENT BROKEN-RATE REPRODUCTION (from answers.jsonl; triage identity/self-description before short/loop, plus loop override)')
    for arm in ARMS:
        vals=per_arm[arm]; tri=sum(x[1] for x in vals); loops=sum(x[2] for x in vals); bad=sum(x[3] for x in vals)
        print(f'{arm}: broken {bad}/{len(vals)} = {bad/len(vals):.6f}; triage-only {tri}/{len(vals)} = {tri/len(vals):.6f}; loop-only {loops}/{len(vals)} = {loops/len(vals):.6f}')
    print('\nLADDER PAIRED SLOT DIFFERENCES AND RATIOS (layer-stratified bootstrap, seed 165, 10,000)')
    e,ci,z=bootstrap(slot['broken'],'k4','n±'); print(f'k4 - n±: {e:.6f} ({ci[0]:.6f}, {ci[1]:.6f})')
    for a,b in [('k4','k16'),('k16','k64'),('k64','full')]:
        e,ci,z=bootstrap(slot['broken'],a,b); print(f'{a} - {b}: {e:.6f} ({ci[0]:.6f}, {ci[1]:.6f})')
    for a,b in [('k4','full'),('k16','full')]:
        e,ci,z=bootstrap(slot['broken'],a,b,ratio=True); print(f'{a}/{b}: {e:.6f} ({ci[0]:.6f}, {ci[1]:.6f}); both-zero draws={z}')
    print('\nFRAGILITY: n+ / full primary breakage')
    all_n=mean([slot['broken']['n+'][k] for k in slot['broken']['n+']]); all_f=mean([slot['broken']['full'][k] for k in slot['broken']['full']])
    print(f'all layers: {all_n:.6f}/{all_f:.6f} = {all_n/all_f:.6f}')
    for layer in LAYERS:
        n=mean([slot['broken']['n+'][(layer,s)] for s in SLOTS]); f=mean([slot['broken']['full'][(layer,s)] for s in SLOTS])
        print(f'layer {layer}: n+={n:.6f} full={f:.6f} ratio={n/f if f else float("inf"):.6f}')
    for omitted in LAYERS:
        keys=[(l,s) for l in LAYERS if l != omitted for s in SLOTS]
        n=mean([slot['broken']['n+'][k] for k in keys]); f=mean([slot['broken']['full'][k] for k in keys])
        print(f'omit layer {omitted}: n+={n:.6f} full={f:.6f} ratio={n/f if f else float("inf"):.6f}')
    print('\nN+ SLOT CONCENTRATION')
    nplus_bad={k:sum(v[3] for v in state['n+'][k]) for k in state['n+']}
    all_breaks=sum(nplus_bad.values())
    full_slots=[k for k,v in nplus_bad.items() if v==8]
    any_slots=[k for k,v in nplus_bad.items() if v>0]
    print(f'8/8 broken slots={len(full_slots)}; breaks in them={sum(nplus_bad[k] for k in full_slots)}/{all_breaks} = {sum(nplus_bad[k] for k in full_slots)/all_breaks:.6%}')
    remaining=[k for k in nplus_bad if k not in full_slots]
    n=mean([slot['broken']['n+'][k] for k in remaining]); f=mean([slot['broken']['full'][k] for k in remaining])
    print(f'omit eight-of-eight n+ slots: n+ / full={n/f:.6f} ({n:.6f}/{f:.6f})')
    print(f'slots with any n+ break={len(any_slots)}/96; answer-weighted slots={len(any_slots)*8}/768 (effective sample nearer 160 than 768); slot bootstrap resamples slots')
    print('\nLAYERWISE LADDER SHAPE AND SPEARMAN ROBUSTNESS')
    ladder_rates={a:mean([slot['broken'][a][(l,s)] for l in LAYERS for s in SLOTS]) for a in LADDER}
    full_spearman=corr(rank([ladder_rates[a] for a in LADDER]),rank(list(range(len(LADDER)))))
    print(f'all-layer rate Spearman vs support order={full_spearman:.6f}')
    loo=[]
    for omitted in LAYERS:
        vals={a:mean([slot['broken'][a][(l,s)] for l in LAYERS if l!=omitted for s in SLOTS]) for a in LADDER}
        sp=corr(rank([vals[a] for a in LADDER]),rank(list(range(len(LADDER)))))
        loo.append(sp); print(f'omit layer {omitted}: Spearman={sp:.6f}')
    print(f'leave-one-layer-out Spearman range={min(loo):.6f} to {max(loo):.6f}')
    for omitted in LAYERS:
        n=mean([slot['broken']['n+'][(l,s)] for l in LAYERS if l!=omitted for s in SLOTS]); f=mean([slot['broken']['full'][(l,s)] for l in LAYERS if l!=omitted for s in SLOTS])
        vals={a:mean([slot['broken'][a][(l,s)] for l in LAYERS if l!=omitted for s in SLOTS]) for a in LADDER}
        print(f'omit layer {omitted}: n+/full={n/f:.6f}; k4-n±={(vals["k4"]-vals["n±"])*100:.3f} points')
    print('layer checks: n+/full >=2 in ' + ', '.join(str(l) for l in LAYERS if mean([slot['broken']['n+'][(l,s)] for s in SLOTS])/mean([slot['broken']['full'][(l,s)] for s in SLOTS])>=2))
    print('k4 to k16 drop layers: ' + ', '.join(str(l) for l in LAYERS if mean([slot['broken']['k4'][(l,s)] for s in SLOTS])>mean([slot['broken']['k16'][(l,s)] for s in SLOTS])))
    print('k16 to k64 drop layers: ' + ', '.join(str(l) for l in LAYERS if mean([slot['broken']['k16'][(l,s)] for s in SLOTS])>mean([slot['broken']['k64'][(l,s)] for s in SLOTS])))
    print('\nKL ARM SUMMARIES (effects.jsonl; n+/full ratios)')
    klstats={}
    for arm in ARMS:
        xs=[float(r['kl']) for r in effects if r['arm']==arm]
        top=sorted(xs,reverse=True)[:math.ceil(len(xs)*.01)]
        klstats[arm]=(mean(xs),median(xs),sum(top)/sum(xs))
        print(f'{arm}: mean={mean(xs):.6f} median={median(xs):.6f} top1%_share={sum(top)/sum(xs):.6%}')
    print(f'n+/full mean KL ratio={klstats["n+"][0]/klstats["full"][0]:.6f}; median KL ratio={klstats["n+"][1]/klstats["full"][1]:.6f}')
    kl_slot={a:{} for a in ARMS}
    for arm in ARMS:
        for layer in LAYERS:
            for s in SLOTS:
                kl_slot[arm][(layer,s)]=mean([float(r['kl']) for r in effects if r['arm']==arm and int(r['layer'])==layer and int(r['slot'])==s])
    e,ci,z=bootstrap(kl_slot,'n+','full',ratio=True)
    print(f'n+/full mean KL ratio, paired slot bootstrap={e:.6f} ({ci[0]:.6f}, {ci[1]:.6f})')
    kl_ladder=[mean([float(r['kl']) for r in effects if r['arm']==a]) for a in LADDER]
    print(f'KL ladder Spearman (support order vs arm mean KL)={corr(rank(list(range(len(LADDER)))),rank(kl_ladder)):.6f}; means={dict(zip(LADDER,kl_ladder))}')
    print('\nWITHIN-ARM SLOT PEARSON AND SPEARMAN (exploratory; 96 slots)')
    effect_slot={}
    for arm in ARMS:
        for field in ('kl','rel_change'):
            for key in [(l,s) for l in LAYERS for s in SLOTS]:
                effect_slot[(arm,field,*key)]=mean([r[field] for r in effects if r['arm']==arm and int(r['layer'])==key[0] and int(r['slot'])==key[1]])
        xs=[slot['broken'][arm][(l,s)] for l in LAYERS for s in SLOTS]
        for field in ('kl','rel_change'):
            ys=[effect_slot[(arm,field,l,s)] for l in LAYERS for s in SLOTS]
            print(f'{arm} broken vs mean {field}: Pearson={corr(xs,ys):.6f}; Spearman={corr(rank(xs),rank(ys)):.6f}')
    print('\nMEDIAN FIRST WORD POSITION OF FIRST REPEATED WORD 3-GRAM (primary-broken looping answers)')
    for arm in ARMS:
        positions=[]
        for _,_,loop,bad,_,words in per_arm[arm]:
            if bad and loop:
                seen={};
                for i in range(max(0,len(words)-2)):
                    tri=tuple(words[i:i+3]); seen[tri]=seen.get(tri,0)+1
                    if seen[tri]>=3: positions.append(i+1); break
        print(f'{arm}: {median(positions):.3f} over {len(positions)} answers')
    print('\nW GEOMETRY BY LAYER')
    for x in geometry['layers']:
        print(f"layer {x['layer']}: rank={x['rank']}; singular-value range={x['smallest_singular_value']:.8g} to {x['largest_singular_value']:.8g}")
    print('\nGEOMETRY BY ARM (mean largest-column cosine)')
    for arm in ARMS:
        xs=[d['largest_column_cosine'] for layer in geometry['layers'] for d in layer['directions'] if d['arm']==arm]
        print(f'{arm}: {mean(xs):.6f}')
    print('\nMEAN RELATIVE LAST-LAYER CHANGE BY LAYER AND ARM (effect rows)')
    for layer in LAYERS:
        vals=[]
        for arm in ARMS:
            xs=[r['rel_change'] for r in effects if int(r['layer'])==layer and r['arm']==arm]
            vals.append(f'{arm}={mean(xs):.6f}')
        print(f'layer {layer}: ' + ' '.join(vals))
    print('\nCHANGED RATE BY ARM (first 12 whitespace words differ from no-push)')
    for arm in ARMS:
        xs=per_arm[arm]; count=sum(r['answer'].split()[:12] != baselines[int(r['question_index'])].split()[:12] for r,_,_,_,_,_ in xs)
        print(f'{arm}: {count}/{len(xs)} = {count/len(xs):.6f}')
    print('\nEFFECT RUN1 VS RERUN: common fields, row key arm/layer/slot/question_index')
    key=lambda r:(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index']))
    old={key(r):r for r in run1}; new={key(r):r for r in effects}
    print(f'rows: run1={len(old)} rerun={len(new)} common={len(old.keys() & new.keys())}')
    for field in sorted(set.intersection(*(set(r) for r in run1),*(set(r) for r in effects))):
        if field in ('arm','layer','slot','question_index','kl'): continue
        n=sum(old[k].get(field)!=new[k].get(field) for k in old.keys() & new.keys())
        print(f'{field}: differing={n}/{len(old.keys() & new.keys())}')
    old_nan=sum(not isinstance(r.get('kl'),(int,float)) or not math.isfinite(float(r['kl'])) for r in run1)
    new_finite=sum(isinstance(r.get('kl'),(int,float)) and math.isfinite(float(r['kl'])) for r in effects)
    print(f'kl: run1 NaN/nonfinite={old_nan}/{len(run1)}; rerun finite={new_finite}/{len(effects)} (expected scorer fix)')
    print('\nPREDICTION 165.1-165.6 INTERVALS FROM score.json; 165.7 from effect.json')
    for p in ('165.1','165.2','165.3','165.4','165.5','165.6'):
        x=score['predictions'][p]; est=x.get('estimate_ratio',x.get('spearman',{}).get('estimate',x.get('distances')))
        ci=x.get('paired_slot_bootstrap',x.get('spearman',{})).get('interval_95')
        ratio_keys={'165.1':'n_plus_over_full','165.4':'n_plus_over_n_pm','165.5':'iso_over_full'}
        if p in ratio_keys:
            ci=score['ratios'][ratio_keys[p]]['interval_95']
        if p=='165.3': ci='not defined (distance comparison)'
        print(f'{p}: {x["verdict"]} estimate={est} interval={ci}')
    effect_score=json.loads((HERE/'effect.json').read_text())
    p=effect_score['prediction_165.7']; print(f"165.7: {p['verdict']} estimate={p['estimate_ratio']} interval={p['paired_slot_bootstrap']['interval_95']}")
    print('\nDOSE ROW CHECKS (effects.jsonl)')
    support={'n+':1,'n±':1,'k4':4,'k16':16,'k64':64,'k256':256,'k1024':1024,'full':9216,'iso':9216}
    for arm in ARMS:
        rr=[r for r in effects if r['arm']==arm]
        maxerr=max(abs(r['realized_write_norm']/r['intended_write_norm']-1) for r in rr)*100
        lost=Counter(max(0,support[arm]-int(r['changed_channels'])) for r in rr)
        print(f'{arm}: max row norm deviation={maxerr:.6f}%; mean cosine={mean([r["realized_cos"] for r in rr]):.8f}; rows losing channels={sum(v for k,v in lost.items() if k>0)}/768; lose-one={lost[1]}; lose-two={lost[2]}')
    print('\nANSWER LOOP PATTERNS (loop trigram occurrences in answers.jsonl, battery rows)')
    for arm in ARMS:
        patterns=Counter()
        for r,tri,loop,bad,short,words in per_arm[arm]:
            if loop:
                patterns.update(tuple(words[i:i+3]) for i in range(len(words)-2))
        print(f'{arm}: ' + ', '.join(f'{" ".join(k)}={v}' for k,v in patterns.most_common(6)))
    print('k=1 repeated loop trigrams present in all eight question phrasings within slots:')
    for arm in ('n+','n±'):
        slot_loops={}
        for layer in LAYERS:
            for s in SLOTS:
                answers_by_q={v[0]['question_index']:v[5] for v in state[arm][(layer,s)]}
                common=None
                for words in answers_by_q.values():
                    tris={tuple(words[i:i+3]) for i in range(len(words)-2)}
                    common=tris if common is None else common & tris
                repeated={t for t in (common or set()) if sum(1 for i in range(max(0,len(next(iter(answers_by_q.values()),[]))-2)) if tuple(next(iter(answers_by_q.values()))[i:i+3])==t)>=3}
                if repeated: slot_loops[(layer,s)]=repeated
        print(f'{arm}: slots with shared repeated trigram={len(slot_loops)}; examples=' + ', '.join(f'{l}/{s}: {" ".join(t)}' for (l,s),ts in list(slot_loops.items())[:8] for t in list(ts)[:1]))
    print('\nHAND-READ QUOTE CANDIDATES (from answers.jsonl):')
    # Deterministic selection: first loop-broken row; unbroken row at q index 3.
    for arm in ('n+','full'):
        good=bad=None
        for r,tri,loop,broken,short,words in per_arm[arm]:
            if loop and broken and bad is None: bad=r
            if not broken and int(r['question_index'])==3 and good is None: good=r
        for label,r in [('broken',bad),('unbroken',good)]:
            print(f'{arm} {label} [{r["layer"]},{r["slot"]},q{r["question_index"]}]: {r["answer"]}')

if __name__=='__main__': main()
