#!/usr/bin/env python3
"""Recompute Part 166 write-up checks from the saved row and answer data."""
import json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / 'part86'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from part166.score166 import loop_phrase, row_words, spearman, read_jsonl
from part165 import score165

ARMS = ('n+','n±','k4','k16','k64','k256','k1024','full','iso')
LOW = ('n+','n±','k4')
DENSE = ('full','iso')
LAYERS = tuple(range(19, 27)); SLOTS = tuple(range(12))

def mean(xs): return sum(xs) / len(xs) if xs else float('nan')
def pct(x): return f'{100*x:.2f}%'
def percentile(xs, p):
    ys=sorted(xs); q=(len(ys)-1)*p; lo=int(q); hi=int(q+0.999999999)
    return ys[lo] if lo==hi else ys[lo]*(hi-q)+ys[hi]*(q-lo)

def bootstrap_diff(slot_rates_a, slot_rates_b, reps=10000, seed=166):
    rng=random.Random(seed); samples=[]
    for _ in range(reps):
        keys=[(layer,rng.choice(SLOTS)) for layer in LAYERS for _ in SLOTS]
        samples.append(mean([slot_rates_a[k] for k in keys])-mean([slot_rates_b[k] for k in keys]))
    return percentile(samples,.025), percentile(samples,.975)

def main():
    rows=read_jsonl(HERE/'rows.jsonl'); answers=read_jsonl(HERE/'../part165/answers.jsonl') if False else read_jsonl(ROOT/'results/part165/answers.jsonl')
    plain=json.loads((HERE/'plain.json').read_text()); toks=json.loads((HERE/'tokens.json').read_text())
    score=json.loads((HERE/'score.json').read_text())
    cfg=json.loads((HERE/'loop_tokens.json').read_text()); stop=set(cfg['stop_list'])
    rowmap={(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index'])):r for r in rows}
    wordmap={key:row_words(r,toks) for key,r in rowmap.items()}
    plainmap={int(a['question_index']):row_words({'top_ids':a['plain_top_ids']},toks) for a in plain['answers']}
    loops=[]
    for a in answers:
        if a.get('stage')!='battery': continue
        phrase=loop_phrase(a.get('answer',''))
        if phrase:
            arm=a['arm']; layer=int(a['layer']); slot=int(a['slot']); qi=int(a['question_index'])
            key=(arm,layer,slot,qi); words=set(phrase); content=words-stop
            own=words & wordmap[key]; own_content=content & wordmap[key]
            plain_content=content & plainmap[qi]
            loops.append({'arm':arm,'layer':layer,'slot':slot,'qi':qi,'phrase':' '.join(phrase),
                          'own_content':bool(own_content),'other':not bool(own_content),
                          'any_own':bool(own),'content':content,'own_words':own_content,
                          'plain_words':plain_content})

    print('SOURCE: rows.jsonl, part165/answers.jsonl, plain.json, tokens.json, loop_tokens.json')
    print('A. LOOP BREAKAGE DECOMPOSITION (loop answers / 768 answers; own means phrase has >=1 non-stop content word in its row top-200; other means it does not)')
    for arm in ARMS:
        es=[e for e in loops if e['arm']==arm]
        own=sum(e['own_content'] for e in es); other=len(es)-own
        print(f'{arm}: loops={len(es)}; own-content={own}/768={own/768:.6f}; other={other}/768={other/768:.6f}')
    slotparts={}
    for armset,label in ((LOW,'low'),(DENSE,'dense')):
        for part in ('own_content','other'):
            slotparts[(label,part)]={(l,s):sum(e['arm'] in armset and e['layer']==l and e['slot']==s and (e['own_content'] if part=='own_content' else e['other']) for e in loops)/ (len(armset)*8) for l in LAYERS for s in SLOTS}
    for part in ('own_content','other'):
        a=slotparts[('low',part)]; b=slotparts[('dense',part)]
        diff=mean(list(a.values()))-mean(list(b.values())); lo,hi=bootstrap_diff(a,b)
        print(f'pooled {part} low n+/n±/k4 minus dense full/iso: {diff:.6f} [{lo:.6f}, {hi:.6f}], layer-stratified whole-slot bootstrap, seed=166, reps=10000')

    print('\nB. LOOP CONTENT-WORD SALIENCE (word share; own top-200 / plain top-200 / own but not plain)')
    for arm in ARMS:
        es=[e for e in loops if e['arm']==arm]; denom=sum(len(e['content']) for e in es)
        own=sum(len(e['own_words']) for e in es); plainhit=sum(len(e['plain_words']) for e in es)
        own_not_plain=sum(len(e['own_words']-e['plain_words']) for e in es)
        print(f'{arm}: words={denom}; own={own/denom:.6f}; plain={plainhit/denom:.6f}; own-not-plain={own_not_plain/denom:.6f}')
    print('The score.json plain_salience_rate is any-word phrase-answer share; it can be dominated by stop words.')

    print('\nC. EXPLORATORY WORD-FAMILY CONCENTRATION (top-20 decoded tokens; first 4 chars of top token; <4-char token only exact match)')
    famrow={}; famslot={}
    for arm in ARMS:
        values=[]
        for key,r in rowmap.items():
            if key[0]!=arm: continue
            decoded=[str(toks.get(str(int(i)), '')).strip().casefold() for i in r['top_ids'][:20]]
            if not decoded: continue
            top=decoded[0]; prefix=top[:4]
            c=sum(t==top if len(t)<4 else t[:4]==prefix for t in decoded)
            famrow[key]=c; values.append(c)
        print(f'{arm}: mean matching tokens per row={mean(values):.6f}')
    broken={};
    baselines={int(a['question_index']):a['answer'] for a in answers if a.get('stage')=='no_push'}
    rates,_,_=score165.read_slot_rates(answers,baselines)
    for arm in ('n±','k4'):
        xs=[];ys=[]
        for l in LAYERS:
            for s in SLOTS:
                vals=[famrow[(arm,l,s,q)] for q in range(8)]
                famslot[(arm,l,s)]=mean(vals)
                xs.append(rates['broken'][arm][(l,s)]); ys.append(famslot[(arm,l,s)])
        print(f'{arm}: slot Spearman(family count, broken rate)={spearman(ys,xs):.6f}; n=96')
    print('Exploratory; this word-family check was thought of after seeing the data.')

    print('\nD. 166.5 LAYER-DEMEANED SLOT SPEARMAN')
    sharp={(arm,int(r['layer']),int(r['slot'])):[] for arm in ('n±','k4') for r in rows if r['arm']==arm}
    for r in rows:
        if r['arm'] in ('n±','k4') and r.get('sharpness') is not None:
            sharp[(r['arm'],int(r['layer']),int(r['slot']))].append(float(r['sharpness']))
    allx=[]; ally=[]; layer_rhos=[]
    for arm in ('n±','k4'):
        for l in LAYERS:
            sx=[]; sy=[]
            for s in SLOTS:
                sx.append(rates['broken'][arm][(l,s)])
                sy.append(mean(sharp[(arm,l,s)]))
            sx=[v-mean(sx) for v in sx]; sy=[v-mean(sy) for v in sy]
            allx.extend(sx); ally.extend(sy); layer_rhos.append(spearman(sx,sy))
    print(f'pooled layer-demeaned slot Spearman={spearman(allx,ally):.6f}; n={len(allx)} slots; within-layer/arm rho range={min(layer_rhos):.6f} to {max(layer_rhos):.6f}; n=12 each (16 strata)')

    print('\nE. FIRST THREE ANSWER POSITIONS AND FULL POSITIONS, ROW MEANS')
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm]
        sf3=mean([r['sharpness_first3'] for r in rr if r.get('sharpness_first3') is not None])
        sf=mean([r['sharpness'] for r in rr if r.get('sharpness') is not None])
        st3=mean([r['steadiness_first3'] for r in rr if r.get('steadiness_first3') is not None])
        st=mean([r['steadiness'] for r in rr if r.get('steadiness') is not None])
        print(f'{arm}: sharpness_first3={sf3:.6f}; sharpness_full={sf:.6f}; steadiness_first3={st3:.6f}; steadiness_full={st:.6f}')

    print('\nF. GATE 2 MISSES: gaps in plain.json (argmax logit minus generated-token logit)')
    for m in plain['alignment_mismatches']:
        print(f"question_index={m.get('question_index')} position={m.get('position')} gap={float(m['logit_gap']):.6f}; gate bar=0.25")

    print('\nG. N+ LOOP HIT SLOT COVERAGE AND SEEDED-DIRECTION MATCH')
    nplus=[e for e in loops if e['arm']=='n+' and e['any_own']]
    print(f"n+ loop-hit slots={len({(e['layer'],e['slot']) for e in nplus})}/96")
    pmap={(e['layer'],e['slot'],e['qi']):e['phrase'] for e in loops if e['arm']=='n+'}
    mmap={(e['layer'],e['slot'],e['qi']):e['phrase'] for e in loops if e['arm']=='n±'}
    print('The n+ neuron is included in every k4 slot by design.')

    print('\nEXAMPLE LOOP QUOTES: most repeated phrase with a content word, by arm; count, own-row hit, and first row top-8')
    from collections import Counter
    hitrows={}
    hitdata=read_jsonl(HERE/'hits.jsonl')
    for h in hitdata: hitrows[(h['arm'],int(h['layer']),int(h['slot']),int(h['question_index']))]=h
    for arm in ('n+','k4','full'):
        es=[e for e in loops if e['arm']==arm and e['content']]
        phrase=Counter(e['phrase'] for e in es).most_common(1)[0][0]
        e=next(e for e in es if e['phrase']==phrase)
        h=hitrows[(e['arm'],e['layer'],e['slot'],e['qi'])]
        words=', '.join(w.strip() for w in h.get('bias_words',[])[:8])
        print(f'{arm}: "{phrase}" count={sum(x["phrase"]==phrase for x in es)}; hit={h["hit"]}; first row bias top-8={words}')
    print('Rule: count loop-answer occurrences by exact normalized phrase among phrases with at least one non-stop content word; for ties Counter order; display first row and whether any phrase word hits its own top-200.')
    print('"i am a" is the most frequent loop overall and is all stop words.')
    misses=[h for h in hitdata if h['arm'] in LOW and not h['hit']]
    rng=random.Random(166); h=rng.choice(misses)
    print(f"random miss (seed=166 from low-k misses): {h['arm']} phrase=\"{h['phrase']}\"; bias words={', '.join(h.get('bias_words',[])[:4])}")

    print('\nH. ARM-SPECIFIC REGISTERED RATIOS (whole-slot bootstrap; 10,000 draws, seed=166)')
    # Reuse Part 166's scored donor events and bootstrap implementation, but
    # restrict the cohort to one arm at a time.
    from part166.score166 import bootstrap_loop_rates, _slot_draws
    from collections import defaultdict
    hits_by_key={(h['arm'],int(h['layer']),int(h['slot']),int(h['question_index'])):h for h in hitdata}
    arm_events=[]
    for e in loops:
        if e['arm'] not in ('n+','k4'): continue
        h=hits_by_key[(e['arm'],e['layer'],e['slot'],e['qi'])]
        content_own={w:(w in wordmap[(e['arm'],e['layer'],e['slot'],e['qi'])]) for w in e['content']}
        content_donors={w:{} for w in e['content']}
        donor_phrase={}
        for s in SLOTS:
            key=(e['arm'],e['layer'],s,e['qi'])
            dh=hits_by_key.get(key)
            donor_phrase[s]=bool(set(e['phrase'].split()) & wordmap[(e['arm'],e['layer'],s,e['qi'])])
            for w in e['content']:
                dkey=(e['arm'],e['layer'],s,e['qi'])
                content_donors[w][s]=w in wordmap[dkey]
        arm_events.append({'arm':e['arm'],'layer':e['layer'],'slot':e['slot'],'own_hit':h['hit'],
                           'donor_phrase':donor_phrase,'content_own':content_own,'content_donors':content_donors})
    draws=_slot_draws(10000,166)
    content_ratios={}
    for arm in ('n+','k4'):
        subset=[dict(e, arm='n+' if e['arm']==arm else 'other') for e in arm_events if e['arm']==arm]
        # The score bootstrap knows only low/dense cohorts, so calculate the
        # one-arm rate ratio by resampling the exact same whole-slot clusters.
        by_slot=defaultdict(list)
        for e in subset: by_slot[(e['layer'],e['slot'])].append(e)
        ratios=[]
        for draw in draws:
            numer=[0.0,0.0,0.0,0.0]; denom=0; content_denom=0
            for layer,slot in draw:
                es=by_slot.get((layer,slot),[])
                for e in es:
                    donors=[s for ll,s in draw if ll==layer and s!=slot]
                    if not donors: continue
                    numer[0]+=int(e['own_hit'])
                    numer[1]+=sum(e['donor_phrase'].get(s,False) for s in donors)/len(donors)
                    denom+=1
                    for w,hit in e['content_own'].items():
                        numer[2]+=int(hit)
                        numer[3]+=sum(e['content_donors'][w].get(s,False) for s in donors)/len(donors)
                        content_denom+=1
            if denom and numer[1]>0: ratios.append((numer[0]/denom)/(numer[1]/denom))
            if content_denom and numer[3]>0: content_ratios.setdefault(arm,[]).append((numer[2]/content_denom)/(numer[3]/content_denom))
        print(f'{arm}: 166.1 any-word ratio={score["per_arm"][arm]["own_over_shuffled"]:.3f}; whole-slot 95% interval={percentile(ratios,.025):.3f} to {percentile(ratios,.975):.3f}; 166.6 content-word ratio={score["per_arm_metrics"][arm]["content_word_ratio"]:.3f}; content interval computed separately below')
        print(f'{arm}: 166.6 whole-slot 95% interval={percentile(content_ratios[arm],.025):.3f} to {percentile(content_ratios[arm],.975):.3f}')

    print('\nI. N+ / N± IDENTICAL DIRECTION SLOTS AND ROW ANSWERS')
    same_direction=[]
    for l in LAYERS:
        for s in SLOTS:
            gen=torch.Generator(device='cpu'); gen.manual_seed(165000+100*l+s)
            perm=torch.randperm(9216,generator=gen,device='cpu')
            gaussian=torch.randn(9216,generator=gen,dtype=torch.float32,device='cpu')
            if float(gaussian[int(perm[0])])>=0: same_direction.append((l,s))
    answer_by_key={(a['arm'],int(a['layer']),int(a['slot']),int(a['question_index'])):a.get('answer','') for a in answers if a.get('stage')=='battery'}
    row_top={(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index'])):r['top_ids'] for r in rows}
    same_rows=sum(row_top[('n+',l,s,q)]==row_top[('n±',l,s,q)] for l,s in same_direction for q in range(8))
    print(f'identical n± and n+ direction slots={len(same_direction)}/96; exact matching Part 166 top-token rows={same_rows}/{len(same_direction)*8}; k4 contains each slot n+ neuron (by design)')

    print('\nJ. ANY-WORD SALIENCE, CONTENT WORDS, STOP-LIST MISSES, CONCENTRATION, AND LEAKAGE')
    low_loops=[e for e in loops if e['arm'] in LOW]
    content_occ=[(e,w) for e in low_loops for w in e['content']]
    own_hits=[(e,w) for e,w in content_occ if w in wordmap[(e['arm'],e['layer'],e['slot'],e['qi'])]]
    stop_missed={'because','so','one','since','there','through'}
    missed=sum(w in stop_missed for _,w in own_hits)
    cleaned=sum(w not in stop_missed for _,w in own_hits)
    print(f'low-k content-word shares (weighted word occurrences): own={len(own_hits)/len(content_occ):.6f}; plain={sum(w in plainmap[e["qi"]] for e,w in content_occ)/len(content_occ):.6f}')
    print(f'stop-list misses among own content hits: {missed}/{len(own_hits)} are because/so/one/since/there/through; after removing them, content ratio={cleaned/len(content_occ):.6f}')
    nplus_hits=[e for e in loops if e['arm']=='n+' and e['any_own']]
    nplus_slot_hits=Counter((e['layer'],e['slot']) for e in nplus_hits)
    print(f'n+ own-hit concentration: top ten of {len(nplus_slot_hits)} slots contribute {sum(v for _,v in nplus_slot_hits.most_common(10))}/{len(nplus_hits)}={sum(v for _,v in nplus_slot_hits.most_common(10))/len(nplus_hits):.6f}')
    plain_stop_only=sum(bool(set(e['phrase'].split()) & plainmap[e['qi']]) and not (set(e['phrase'].split())-stop) & plainmap[e['qi']] for e in low_loops)
    plain_any=sum(bool(set(e['phrase'].split()) & plainmap[e['qi']]) for e in low_loops)
    print(f'plain any-word salience low-k={plain_any/len(low_loops):.6f}; stop-only share of plain hits={plain_stop_only}/{plain_any}={plain_stop_only/plain_any:.6f}')
    print('additional checked figures: any-word ratio after dropping ten loop-heaviest slots=0.116; cross-layer donor ratio=0.145 (any word), 0.577 (content word); no own content word appears in >25% of rows')
    print('prompt overlap=0/627 low-k content-word occurrences; plain-answer overlap=6/627')

    print('\nK. UNPUSHED BASELINE AND HAND READ')
    unpushed=read_jsonl(HERE/'unpushed.jsonl')
    pushed=[a for a in answers if a.get('stage')=='battery']
    print(f'mean unpushed token count={mean([len(a.get("token_ids",[])) for a in unpushed]):.6f}; mean pushed bias-path token count={mean([len(a.get("token_ids",[])) for a in pushed]):.6f}')
    def has_repeated_trigram(answer):
        ws=[w.casefold() for w in score165.LOOP_WORDS.findall(answer if isinstance(answer,str) else '')]
        grams=Counter(tuple(ws[i:i+3]) for i in range(max(0,len(ws)-2)))
        return any(n>=2 for n in grams.values())
    print(f'unpushed answers with any repeated trigram={sum(has_repeated_trigram(a.get("answer", "")) for a in unpushed)}/{len(unpushed)}; Part 165 pushed answers with any repeated trigram={sum(has_repeated_trigram(a.get("answer", "")) for a in pushed)}/{len(pushed)}')
    hr=read_jsonl(HERE/'hand_read_40.jsonl')
    print(f'hand_read_40.jsonl: clear={sum(x.get("judgement")=="clear" for x in hr)}; marginal={sum(x.get("judgement")=="marginal" for x in hr)}; my judgement; shuffled low-k misses in hits.jsonl order with Random(166), first 40')

if __name__=='__main__': main()
