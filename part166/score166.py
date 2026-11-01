#!/usr/bin/env python3
"""Host scorer for Part 166 bias rows and Part 165 loop answers."""
from __future__ import annotations
import argparse, json, math, random, sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
for p in (str(ROOT),str(ROOT/'part86')):
    if p not in sys.path: sys.path.insert(0,p)
from part165 import score165
from part165.run165 import ARMS, LAYERS, K_ARMS
from part165.score165 import SLOTS

EXPECTED=8*12*8*9
LADDER=('n±','k4','k16','k64','k256','k1024','full')

def read_jsonl(path):
    path=Path(path)
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def ranks(v): return score165.rank_average(v)
def spearman(x,y): return score165.spearman(x,y)

def loop_phrase(answer):
    words=[w.casefold() for w in score165.LOOP_WORDS.findall(answer if isinstance(answer,str) else '')]
    grams=[tuple(words[i:i+3]) for i in range(max(0,len(words)-2))]
    counts=Counter(grams)
    best=None; highest=0
    for gram in grams:
        if counts[gram]>highest: best=gram; highest=counts[gram]
    return (best if highest>=3 else None)

def row_words(row,tokens):
    found=set()
    for token_id in row.get('top_ids',[]):
        text=str(tokens.get(str(int(token_id)), '')).strip().casefold()
        if len(text)>=2 and any(c.isalpha() for c in text):
            found.update(w.casefold() for w in score165.LOOP_WORDS.findall(text))
    return found

def passes_166_1(own, shuffled): return (own >= 2.0 * shuffled) if shuffled > 0 else own > 0
def passes_166_2(own, shuffled): return shuffled > 0 and own / shuffled < 1.5
def passes_166_3(nplus_mean, full_mean): return full_mean > 0 and nplus_mean / full_mean >= 1.25
def passes_166_4(rho): return rho is not None and rho <= -0.8
def passes_166_5(rho): return rho is not None and rho >= 0.3
def passes_166_6(own, shuffled): return own >= 1.5 * shuffled
def passes_166_7(mean_percentile): return mean_percentile is not None and mean_percentile >= 0.65

def _finite_row(row):
    for key in ('kl_mean','kl_second','bias_norm'):
        try:
            if not math.isfinite(float(row[key])): return False
        except (KeyError,TypeError,ValueError): return False
    for key in ('sharpness','steadiness','sharpness_first3','steadiness_first3'):
        value=row.get(key)
        if value is not None:
            try:
                if not math.isfinite(float(value)): return False
            except (TypeError,ValueError): return False
    try:
        return (len(row.get('top_ids',[]))==len(row.get('top_values',[])) and
                all(math.isfinite(float(x)) for x in row.get('top_values',[])))
    except (TypeError,ValueError): return False

def gate2_passes(plain):
    """Amendment 4: at least 99% aligned, every miss a near tie of at most 0.25 logits, and repeatable."""
    total=int(plain.get('alignment_total',0)); matches=int(plain.get('alignment_matches',-1))
    misses=plain.get('alignment_mismatches',[])
    near=all(float(m.get('logit_gap',1e9))<=.25 for m in misses)
    return (total>0 and matches+len(misses)==total and matches>=.99*total and near and
            int(plain.get('determinism_matches',-1))==int(plain.get('determinism_total',-2)))

def _slot_draws(reps=10000,seed=166):
    rng=random.Random(seed)
    return [[(layer,rng.choice(SLOTS)) for layer in LAYERS for _ in SLOTS] for _ in range(reps)]

def _boot(values, reps=10000, seed=166, draws=None):
    """CI for the mean under a layer-stratified, jointly sampled slot design."""
    if not values: return [None,None]
    by_slot=defaultdict(list)
    for layer,slot,value in values: by_slot[(int(layer),int(slot))].append(float(value))
    by_slot={k:sum(v)/len(v) for k,v in by_slot.items()}
    if draws is None: draws=_slot_draws(reps,seed)
    outcomes=[]
    for draw in draws:
        sample=[by_slot[k] for k in draw if k in by_slot]
        if sample:
            outcomes.append(sum(sample)/len(sample))
    outcomes.sort()
    return [outcomes[int(.025*(len(outcomes)-1))],outcomes[int(.975*(len(outcomes)-1))]] if outcomes else [None,None]

def bootstrap_loop_rates(events,draws):
    """Resample whole layer/slot clusters; recompute donors only from sampled slots."""
    by_slot=defaultdict(list)
    for event in events: by_slot[(event['layer'],event['slot'])].append(event)
    samples=defaultdict(list)
    for draw in draws:
        layer_slots=defaultdict(list)
        for layer,slot in draw: layer_slots[layer].append(slot)
        numer=defaultdict(float); denom=defaultdict(float)
        for (layer,slot),events_here in by_slot.items():
            multiplicity=layer_slots[layer].count(slot)
            if not multiplicity: continue
            for _ in range(multiplicity):
                donors=list(layer_slots[layer])
                try: donors.remove(slot)
                except ValueError: donors=[]
                for event in events_here:
                    arm=event['arm']; cohorts=[]
                    if arm in ('n+','n±','k4'): cohorts.append('low')
                    if arm in ('full','iso'): cohorts.append('dense')
                    for cohort in cohorts:
                        numer[cohort+'_own']+=int(event['own_hit']); denom[cohort+'_own']+=1
                        donor_hits=event['donor_phrase']
                        if donors:
                            numer[cohort+'_shuffled']+=sum(donor_hits.get(s,False) for s in donors)/len(donors)
                            denom[cohort+'_shuffled']+=1
                    if arm in ('n+','n±','k4'):
                        for word,hit in event['content_own'].items():
                            numer['content_own']+=int(hit); denom['content_own']+=1
                            word_donors=event['content_donors'][word]
                            if donors:
                                numer['content_shuffled']+=sum(word_donors.get(s,False) for s in donors)/len(donors)
                                denom['content_shuffled']+=1
        for key in ('low_own','low_shuffled','dense_own','dense_shuffled','content_own','content_shuffled'):
            if denom[key]: samples[key].append(numer[key]/denom[key])
    def ci(values):
        if not values:return [None,None]
        v=sorted(values);return [v[int(.025*(len(v)-1))],v[int(.975*(len(v)-1))]]
    result={k:ci(v) for k,v in samples.items()}
    for label,own_key,shuf_key in (('166.1','low_own','low_shuffled'),('166.2','dense_own','dense_shuffled'),('166.6','content_own','content_shuffled')):
        ratios=[a/b for a,b in zip(samples.get(own_key,[]),samples.get(shuf_key,[])) if b>0]
        result[label+'_ratio']=ci(ratios)
    return result

def bootstrap_spearman(values,draws):
    by_slot=defaultdict(list)
    for layer,slot,x,y in values: by_slot[(int(layer),int(slot))].append((float(x),float(y)))
    outcomes=[]
    for draw in draws:
        xs=[];ys=[]
        for key in draw:
            for x,y in by_slot.get(key,[]): xs.append(x);ys.append(y)
        value=spearman(xs,ys)
        if value is not None: outcomes.append(value)
    outcomes.sort()
    return [outcomes[int(.025*(len(outcomes)-1))],outcomes[int(.975*(len(outcomes)-1))]] if outcomes else [None,None]

def bootstrap_ladder(rows,draws):
    by_slot=defaultdict(list)
    for row in rows:
        if row['arm'] in LADDER and row.get('sharpness') is not None:
            by_slot[(int(row['layer']),int(row['slot']))].append((row['arm'],float(row['sharpness']),int(row['k'])))
    outcomes=[]
    for draw in draws:
        by_arm=defaultdict(list); k_by_arm={}
        for key in draw:
            for arm,value,k in by_slot.get(key,[]): by_arm[arm].append(value); k_by_arm[arm]=k
        arms=[a for a in LADDER if by_arm[a]]
        rho=spearman([k_by_arm[a] for a in arms],[sum(by_arm[a])/len(by_arm[a]) for a in arms])
        if rho is not None: outcomes.append(rho)
    outcomes.sort()
    return [outcomes[int(.025*(len(outcomes)-1))],outcomes[int(.975*(len(outcomes)-1))]] if outcomes else [None,None]

def bootstrap_arm_ratio(rows,numerator_arm,denominator_arm,draws,field):
    by_slot=defaultdict(lambda:defaultdict(list))
    for row in rows:
        value=row.get(field)
        if value is not None and row['arm'] in (numerator_arm,denominator_arm):
            by_slot[(int(row['layer']),int(row['slot']))][row['arm']].append(float(value))
    outcomes=[]
    for draw in draws:
        top=[];bottom=[]
        for key in draw:
            entry=by_slot.get(key,{})
            if entry.get(numerator_arm): top.append(sum(entry[numerator_arm])/len(entry[numerator_arm]))
            if entry.get(denominator_arm): bottom.append(sum(entry[denominator_arm])/len(entry[denominator_arm]))
        if top and bottom and sum(bottom)>0: outcomes.append((sum(top)/len(top))/(sum(bottom)/len(bottom)))
    outcomes.sort()
    return [outcomes[int(.025*(len(outcomes)-1))],outcomes[int(.975*(len(outcomes)-1))]] if outcomes else [None,None]

def gate3_summary(rows,effects):
    by_effect={(r.get('arm'),int(r.get('slot',-1)),int(r.get('layer',-1)),int(r.get('question_index',-1))):float(r['kl']) for r in effects if 'kl' in r}
    pairs=[]
    for row in rows:
        key=(row.get('arm'),int(row.get('slot',-1)),int(row.get('layer',-1)),int(row.get('question_index',-1)))
        if key in by_effect and row.get('kl_second') is not None:
            try: actual=float(row['kl_second']); reference=by_effect[key]
            except (TypeError,ValueError): continue
            if math.isfinite(actual) and math.isfinite(reference):
                pairs.append((row.get('arm'),int(row.get('layer',-1)),actual,reference,abs(actual-reference)/max(abs(reference),1e-12)))
    corr=spearman([x[2] for x in pairs],[x[3] for x in pairs])
    rel=[x[4] for x in pairs]; med=median(rel) if rel else None
    eligible=[x for x in pairs if x[3]>=.01]
    arm={a:{'n':sum(x[0]==a for x in eligible),'median_relative_difference':None} for a in ARMS}
    layer={l:{'n':sum(x[1]==l for x in eligible),'median_relative_difference':None} for l in LAYERS}
    for dest,key_index in ((arm,0),(layer,1)):
        for group,record in dest.items():
            vals=[x[4] for x in eligible if x[key_index]==group]
            if vals: record['median_relative_difference']=median(vals)
    groups_ok=all(v['n']==0 or v['median_relative_difference']<=.20 for d in (arm,layer) for v in d.values())
    passed=(len(pairs)==EXPECTED and corr is not None and corr>=.9 and med is not None and med<=.10 and groups_ok)
    return {'pass':passed,'n':len(pairs),'spearman':corr,'median_absolute_relative_difference':med,
            'relative_eligible_n':len(eligible),'by_arm':arm,'by_layer':layer}

def score(rows_path, plain_path, tokens_path, answers_path, effects_path, out_dir):
    rows=read_jsonl(rows_path); plain=json.loads(Path(plain_path).read_text(encoding='utf-8'))
    tokens=json.loads(Path(tokens_path).read_text(encoding='utf-8'))
    answers=read_jsonl(answers_path); effects=read_jsonl(effects_path)
    keys=[(r.get('arm'),int(r.get('slot',-1)),int(r.get('layer',-1)),int(r.get('question_index',-1))) for r in rows]
    unique=len(set(keys))==len(keys)
    gate1=(len(rows)==EXPECTED and unique and all(_finite_row(r) for r in rows))
    gate2=gate2_passes(plain)
    gate3_result=gate3_summary(rows,effects)
    gate3=gate3_result['pass']
    gates={'gate1':gate1,'gate1_rows':len(rows),'gate1_unique':unique,'gate1_finite':all(_finite_row(r) for r in rows),
           'gate2':gate2,'alignment_matches':plain.get('alignment_matches'),'alignment_total':plain.get('alignment_total'),
           'determinism_matches':plain.get('determinism_matches'),'determinism_total':plain.get('determinism_total'),
           'gate3':gate3,'gate3_n':gate3_result['n'],'gate3_spearman':gate3_result['spearman'],
           'gate3_median_absolute_relative_difference':gate3_result['median_absolute_relative_difference'],
           'gate3_relative_eligible_n':gate3_result['relative_eligible_n'],
           'gate3_by_arm':gate3_result['by_arm'],'gate3_by_layer':gate3_result['by_layer']}
    if not all((gate1,gate2,gate3)):
        result={'gates':gates,'readings_stopped':True,'predictions':{}}
        write_outputs(out_dir,result,[],[])
        return result
    answer_by_q=defaultdict(list)
    for a in answers:
        if a.get('stage')=='battery': answer_by_q[int(a['question_index'])].append(a)
    loops=[]
    for a in answers:
        if a.get('stage')!='battery': continue
        phrase=loop_phrase(a.get('answer',''))
        if phrase is not None: loops.append((a,phrase))
    rowmap={(r['arm'],int(r['slot']),int(r['layer']),int(r['question_index'])):r for r in rows}
    wordmap={k:row_words(r,tokens) for k,r in rowmap.items()}
    token_cfg_path=Path(tokens_path).with_name('loop_tokens.json')
    token_cfg=json.loads(token_cfg_path.read_text(encoding='utf-8')) if token_cfg_path.exists() else {'words':{},'stop_list':[]}
    stop=set(token_cfg.get('stop_list',[])); spellings=token_cfg.get('words',{})
    import numpy as np
    loop_value_map={}
    value_dir=Path(rows_path).parent/'loop_values'
    if value_dir.exists():
        for meta_path in value_dir.glob('*.json'):
            meta=json.loads(meta_path.read_text(encoding='utf-8'))
            matrix=np.load(meta_path.with_suffix('.npy'),allow_pickle=False)
            for ix,key in enumerate(meta.get('row_keys',[])):
                loop_value_map[tuple(key)]=(meta['token_ids'],matrix[ix],meta['bias_means'][ix],meta['bias_stds'][ix])
    plain_words={}
    for pr in plain.get('answers',[]):
        found=set()
        for tid in pr.get('plain_top_ids',[]):
            text=str(tokens.get(str(tid),'')).strip().casefold()
            if len(text)>=2 and any(c.isalpha() for c in text):
                found.update(w.casefold() for w in score165.LOOP_WORDS.findall(text))
        plain_words[int(pr['question_index'])]=found
    unpushed_path=Path(rows_path).parent/'unpushed.jsonl'
    unpushed=read_jsonl(unpushed_path) if unpushed_path.exists() else []
    unpushed_loops=[loop_phrase(x.get('answer','')) for x in unpushed]
    unpushed_by_temp={str(t/10):{'answers':sum(int(x.get('temperature_x10',round(float(x.get('temperature',0))*10)))==t for x in unpushed),
                                  'loop_count':sum(1 for x,p in zip(unpushed,unpushed_loops) if int(x.get('temperature_x10',round(float(x.get('temperature',0))*10)))==t and p),
                                  'phrases':[' '.join(p) for x,p in zip(unpushed,unpushed_loops) if int(x.get('temperature_x10',round(float(x.get('temperature',0))*10)))==t and p]}
                       for t in (13,20)}
    unpushed_words={w for phrase in unpushed_loops if phrase for w in phrase if w not in stop}
    stop_counts=Counter()
    for a in answers:
        words=score165.LOOP_WORDS.findall(a.get('answer','') or '')
        stop_counts.update(w.casefold() for w in words)
    stop={w for w,_ in stop_counts.most_common(50)}
    arms_low=('n+','n±','k4'); arms_dense=('full','iso')
    all_rates={}; loop_details=[]; content_total=content_own=0; content_shuffle=0.0
    percentiles=[]; missing_spelling=0; percentile_words=0
    content_slot=defaultdict(lambda:[0,0]); percentile_slot=defaultdict(list); loop_boot_events=[]
    content_arm=defaultdict(lambda:[0.0,0.0,0,0]); percentile_arm=defaultdict(list)
    for arm in ARMS:
        subset=[(a,p) for a,p in loops if a.get('arm')==arm]
        own_hits=[]; shuffle_hits=[]; clean_own=[]; clean_shuffle=[]; plain_hits=[]; distinct_counts=[]
        for a,phrase in subset:
            key=(arm,int(a['slot']),int(a['layer']),int(a['question_index']))
            phrase_words=set(phrase)
            own=int(bool(phrase_words & wordmap.get(key,set())))
            plain_hits.append(int(bool(phrase_words & plain_words.get(int(a['question_index']),set()))))
            distinct_counts.append(len(wordmap.get(key,set())))
            content=[w for w in phrase_words if w not in stop]
            own_words=wordmap.get(key,set())
            local_own=sum(w in own_words for w in content)
            local_shuffle=sum(sum(w in wordmap.get((arm,s,int(a['layer']),int(a['question_index'])),set())
                                  for s in SLOTS if s!=int(a['slot']))/11 for w in content)
            content_arm[arm][0]+=local_own; content_arm[arm][1]+=local_shuffle
            if arm in arms_low:
                content_total += len(content)
                content_own += local_own; content_shuffle += local_shuffle
                content_slot[(int(a['layer']),int(a['slot']))][0] += local_own
                content_slot[(int(a['layer']),int(a['slot']))][1] += len(content)
            content_arm[arm][2]+=len(content)
            content_arm[arm][3]+=sum(bool(spellings.get(w,[])) for w in content)
            loop_percentiles=[]
            entry=loop_value_map.get(key)
            if entry:
                tids,vec,mean,std=entry; pos={int(t):j for j,t in enumerate(tids)}
                mean=float(mean); std=max(float(std),1e-12)
                for word in content:
                    ids=spellings.get(word,[])
                    own_z=[(float(vec[pos[int(t)]])-mean)/std for t in ids if int(t) in pos]
                    if not own_z:
                        if arm in arms_low: missing_spelling += 1; percentile_words += 1
                        continue
                    z=max(own_z); peers=[]
                    for slot in SLOTS:
                        if slot==int(a['slot']): continue
                        peer=loop_value_map.get((arm,slot,int(a['layer']),int(a['question_index'])))
                        if not peer: continue
                        ptids,pvec,pm,ps=peer; ppos={int(t):j for j,t in enumerate(ptids)}
                        pm=float(pm); ps=max(float(ps),1e-12)
                        peer_z=[(float(pvec[ppos[int(t)]])-pm)/ps for t in ids if int(t) in ppos]
                        if peer_z: peers.append(max(peer_z))
                    if peers:
                        pct=(sum(v<z for v in peers)+.5*sum(v==z for v in peers))/len(peers)
                        percentile_arm[arm].append(pct)
                        if arm in arms_low: percentiles.append(pct); loop_percentiles.append(pct)
                    if arm in arms_low: percentile_words += 1
            if loop_percentiles: percentile_slot[(int(a['layer']),int(a['slot']))].append(sum(loop_percentiles)/len(loop_percentiles))
            others=[int(bool(phrase_words & wordmap.get((arm,s,int(a['layer']),int(a['question_index'])),set()))) for s in SLOTS if s!=int(a['slot'])]
            shuffled=sum(others)/len(others) if others else 0.0
            content_own_map={w:(w in own_words) for w in content}
            phrase_donors={s:bool(phrase_words & wordmap.get((arm,s,int(a['layer']),int(a['question_index'])),set())) for s in SLOTS}
            content_donors={w:{s:(w in wordmap.get((arm,s,int(a['layer']),int(a['question_index'])),set())) for s in SLOTS} for w in content}
            loop_boot_events.append({'arm':arm,'layer':int(a['layer']),'slot':int(a['slot']),'own_hit':bool(own),
                                     'donor_phrase':phrase_donors,'content_own':content_own_map,'content_donors':content_donors})
            common=phrase_words <= stop
            own_hits.append(own); shuffle_hits.append(shuffled)
            if not common: clean_own.append(own); clean_shuffle.append(shuffled)
            bias_words=[]
            rr=rowmap.get(key)
            if rr:
                order=sorted(range(len(rr.get('top_values',[]))),key=lambda i:float(rr['top_values'][i]),reverse=True)[:12]
                bias_words=[tokens.get(str(rr['top_ids'][i]),'') for i in order]
            loop_details.append({'arm':arm,'slot':int(a['slot']),'layer':int(a['layer']),
                                 'question_index':int(a['question_index']),'question':a.get('question',int(a['question_index'])),
                                 'phrase':' '.join(phrase),'hit':bool(own),'bias_words':bias_words})
        n=len(subset); own_rate=sum(own_hits)/n if n else None; shuf_rate=sum(shuffle_hits)/n if n else None
        clean_n=len(clean_own)
        all_rates[arm]={'loop_count':n,'own_rate':own_rate,'shuffled_rate':shuf_rate,
                        'own_over_shuffled':own_rate/shuf_rate if shuf_rate else ('Infinity' if own_rate else None),
                        'common_removed_loop_count':clean_n,'common_removed_own_rate':sum(clean_own)/clean_n if clean_n else None,
                        'common_removed_shuffled_rate':sum(clean_shuffle)/clean_n if clean_n else None,
                        'plain_salience_rate':sum(plain_hits)/n if n else None,
                        'mean_distinct_word_count':sum(distinct_counts)/n if n else None}
    pooled=lambda arms,field: (sum(sum(1 for a,p in loops if a.get('arm')==arm) * (all_rates[arm][field] or 0) for arm in arms) /
                              max(1,sum(all_rates[arm]['loop_count'] for arm in arms)))
    low_own=pooled(arms_low,'own_rate'); low_shuf=pooled(arms_low,'shuffled_rate')
    dense_own=pooled(arms_dense,'own_rate'); dense_shuf=pooled(arms_dense,'shuffled_rate')
    pred1={'own_rate':low_own,'shuffled_rate':low_shuf,'ratio':low_own/low_shuf if low_shuf else ('Infinity' if low_own else None),'pass':passes_166_1(low_own,low_shuf)}
    pred2={'own_rate':dense_own,'shuffled_rate':dense_shuf,'ratio':dense_own/dense_shuf if dense_shuf else ('Infinity' if dense_own else None),'pass':passes_166_2(dense_own,dense_shuf)}
    means=defaultdict(list)
    sharp_slots=defaultdict(lambda:defaultdict(list))
    arm_bias_metrics=defaultdict(lambda:{'sharpness':[],'bias_norm':[],'steadiness':[],'kl_mean':[],
                                          'null_sharpness':0,'null_steadiness':0})
    for r in rows:
        arm=r['arm']; key=(int(r['layer']),int(r['slot']))
        if r.get('steadiness') is None: arm_bias_metrics[arm]['null_steadiness']+=1
        else: means[arm].append(float(r['steadiness'])); arm_bias_metrics[arm]['steadiness'].append(float(r['steadiness']))
        if r.get('sharpness') is None: arm_bias_metrics[arm]['null_sharpness']+=1
        else: sharp_slots[arm][key].append(float(r['sharpness'])); arm_bias_metrics[arm]['sharpness'].append(float(r['sharpness']))
        arm_bias_metrics[arm]['bias_norm'].append(float(r['bias_norm']))
        arm_bias_metrics[arm]['kl_mean'].append(float(r['kl_mean']))
    steady_n=sum(means['n+'])/len(means['n+']) if means['n+'] else None
    steady_full=sum(means['full'])/len(means['full']) if means['full'] else None
    pred3={'label':'build check','nplus_mean_steadiness':steady_n,'full_mean_steadiness':steady_full,
           'nplus_count':len(means['n+']),'full_count':len(means['full']),
           'nplus_null_count':arm_bias_metrics['n+']['null_steadiness'],
           'full_null_count':arm_bias_metrics['full']['null_steadiness'],
           'ratio':steady_n/steady_full if steady_full else None,'pass':passes_166_3(steady_n,steady_full or 0.0)}
    xs=[]; ys=[]
    for arm in LADDER:
        selected=[float(r['sharpness']) for r in rows if r['arm']==arm and r.get('sharpness') is not None]
        arm_rows=[r for r in rows if r['arm']==arm]
        xs.append(float(K_ARMS.get(arm, max(int(r['k']) for r in arm_rows))) if arm_rows else 0)
        ys.append(sum(selected)/len(selected) if selected else 0)
    rho4=spearman(xs,ys); pred4={'label':'build check','spearman':rho4,'k_order':xs,'mean_sharpness':ys,'pass':passes_166_4(rho4)}
    rates,_,_=score165.read_slot_rates(answers,{int(a['question_index']):a.get('answer','') for a in answers if a.get('stage')=='no_push'})
    corrx=[]; corry=[]
    for arm in ('n±','k4'):
        for key in [(l,s) for l in LAYERS for s in SLOTS]:
            vals=sharp_slots[arm].get(key,[])
            if vals: corrx.append(float(rates['broken'][arm][key])); corry.append(sum(vals)/len(vals))
    rho5=spearman(corrx,corry); pred5={'spearman':rho5,'n_slots':len(corrx),'pass':passes_166_5(rho5)}
    within_arm_1665={}
    for arm in ('n±','k4'):
        xx=[]; yy=[]
        for key in [(l,s) for l in LAYERS for s in SLOTS]:
            if sharp_slots[arm].get(key): xx.append(float(rates['broken'][arm][key])); yy.append(sum(sharp_slots[arm][key])/len(sharp_slots[arm][key]))
        within_arm_1665[arm]={'n':len(xx),'spearman':spearman(xx,yy)}
    within_layer_1665={}
    for layer_no in LAYERS:
        xx=[]; yy=[]
        for arm in ('n±','k4'):
            for slot in SLOTS:
                key=(layer_no,slot); vals=sharp_slots[arm].get(key,[])
                if vals: xx.append(float(rates['broken'][arm][key])); yy.append(sum(vals)/len(vals))
        within_layer_1665[str(layer_no)]={'n':len(xx),'spearman':spearman(xx,yy)}
    arm_rates_x=[]; arm_rates_y=[]
    for arm in ARMS:
        sx=[]; sy=[]
        for key in [(l,s) for l in LAYERS for s in SLOTS]:
            if sharp_slots[arm].get(key):
                sx.append(float(rates['broken'][arm][key])); sy.append(sum(sharp_slots[arm][key])/len(sharp_slots[arm][key]))
        arm_rates_x.append(sum(sx)/len(sx) if sx else 0.0); arm_rates_y.append(sum(sy)/len(sy) if sy else 0.0)
    arm_rho=spearman(arm_rates_x,arm_rates_y)
    own_content=content_own/content_total if content_total else 0.0
    shuffled_content=content_shuffle/content_total if content_total else 0.0
    pred6={'content_word_own_rate':own_content,'content_word_shuffled_rate':shuffled_content,
           'ratio':own_content/shuffled_content if shuffled_content else ('Infinity' if own_content else None),
           'pass':passes_166_6(own_content,shuffled_content)}
    pred7={'mean_percentile':sum(percentiles)/len(percentiles) if percentiles else None,
           'scored_content_words':len(percentiles),'no_spelling_words':missing_spelling,
           'no_spelling_share':missing_spelling/percentile_words if percentile_words else None,
           'pass':passes_166_7(sum(percentiles)/len(percentiles) if percentiles else None)}
    low_words={w for a,p in loops if a.get('arm') in arms_low for w in p if w not in stop}
    overlap=len(low_words & unpushed_words)/len(low_words) if low_words else None
    predictions={'166.1':pred1,'166.2':pred2,'166.3':pred3,'166.4':pred4,'166.5':pred5,'166.6':pred6,'166.7':pred7}
    per_arm_metrics={}
    for arm in ARMS:
        cm=content_arm[arm]; pm=percentile_arm[arm]; bm=arm_bias_metrics[arm]
        per_arm_metrics[arm]={'sharpness_mean':sum(bm['sharpness'])/len(bm['sharpness']) if bm['sharpness'] else None,
          'bias_norm_mean':sum(bm['bias_norm'])/len(bm['bias_norm']) if bm['bias_norm'] else None,
          'steadiness_mean':sum(bm['steadiness'])/len(bm['steadiness']) if bm['steadiness'] else None,
          'kl_mean':sum(bm['kl_mean'])/len(bm['kl_mean']) if bm['kl_mean'] else None,
          'sharpness_null_count':bm['null_sharpness'],'steadiness_null_count':bm['null_steadiness'],
          'content_word_own_rate':cm[0]/cm[2] if cm[2] else None,
          'content_word_shuffled_rate':cm[1]/cm[2] if cm[2] else None,
          'content_word_ratio':cm[0]/cm[1] if cm[1] else ('Infinity' if cm[0] else None),
          'matchable_loop_word_coverage':cm[3]/cm[2] if cm[2] else None,
          'mean_content_word_percentile':sum(pm)/len(pm) if pm else None}
    distinct_by_arm={arm:(sum(len(wordmap[k]) for k in wordmap if k[0]==arm)/sum(1 for k in wordmap if k[0]==arm)
                          if any(k[0]==arm for k in wordmap) else 0.0) for arm in ARMS}
    bootstrap_draws=_slot_draws()
    donor_bootstrap=bootstrap_loop_rates(loop_boot_events,bootstrap_draws)
    boot5_values=[]
    for r in rows:
        arm=r['arm']; key=(int(r['layer']),int(r['slot']))
        if arm in ('n±','k4') and sharp_slots[arm].get(key):
            boot5_values.append((key[0],key[1],float(rates['broken'][arm][key]),sum(sharp_slots[arm][key])/len(sharp_slots[arm][key])))
    result={'gates':gates,'readings_stopped':False,'per_arm':all_rates,'per_arm_metrics':per_arm_metrics,'predictions':predictions,
            'bootstrap_intervals':{'166.1':donor_bootstrap.get('166.1_ratio'),
                                   '166.2':donor_bootstrap.get('166.2_ratio'),
                                   '166.3':bootstrap_arm_ratio(rows,'n+','full',bootstrap_draws,'steadiness'),
                                   '166.4':bootstrap_ladder(rows,bootstrap_draws),
                                   '166.5':bootstrap_spearman(boot5_values,bootstrap_draws),
                                   '166.6':donor_bootstrap.get('166.6_ratio'),
                                   '166.7':_boot([(layer,slot,sum(vals)/len(vals)) for (layer,slot),vals in percentile_slot.items() if vals],draws=bootstrap_draws)},
            'bootstrap_rate_intervals':donor_bootstrap,
            'distinct_word_count_by_arm':distinct_by_arm,
            'plain_salience_hit_rate_by_arm':{arm:v['plain_salience_rate'] for arm,v in all_rates.items()},
            'arm_level_166_5_spearman':arm_rho,'within_arm_166_5':within_arm_1665,'within_layer_166_5':within_layer_1665,
            'unpushed_loop_count':sum(x is not None for x in unpushed_loops),
            'unpushed_loop_phrases':[' '.join(x) for x in unpushed_loops if x],
            'unpushed_by_temperature':unpushed_by_temp,
            'low_k_content_word_share_in_unpushed_loops':overlap,
            'loop_token_word_count':len(spellings),'loop_token_words_without_spelling':token_cfg.get('no_single_token_spelling',[])}
    write_outputs(out_dir,result,loop_details,rows)
    return result

def write_outputs(out_dir,result,hits,rows):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'score.json').write_text(json.dumps(result,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (out/'hits.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for row in hits: f.write(json.dumps(row,ensure_ascii=True)+'\n')
    lines=['# Part 166 score','',f"Gates: {json.dumps(result.get('gates',{}),ensure_ascii=True)}",'']
    gate3=result.get('gates',{})
    if gate3.get('gate3_by_arm'):
        lines+=['## Gate 3 by arm','','| Arm | Eligible rows | Median relative difference |','|---|---:|---:|']
        for arm,info in gate3['gate3_by_arm'].items(): lines.append(f"| {arm} | {info['n']} | {info['median_relative_difference']} |")
        lines+=['','## Gate 3 by layer','','| Layer | Eligible rows | Median relative difference |','|---:|---:|---:|']
        for layer,info in gate3['gate3_by_layer'].items(): lines.append(f"| {layer} | {info['n']} | {info['median_relative_difference']} |")
    if result.get('readings_stopped'): lines.append('All readings stopped because one or more gates failed.')
    else:
        lines+=['| Arm | Loops | Own | Shuffled | Ratio | Common removed own | Common removed shuffled | Plain salience | Mean distinct words |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm,v in result['per_arm'].items(): lines.append(f"| {arm} | {v['loop_count']} | {v['own_rate']} | {v['shuffled_rate']} | {v['own_over_shuffled']} | {v['common_removed_own_rate']} | {v['common_removed_shuffled_rate']} | {v['plain_salience_rate']} | {v['mean_distinct_word_count']} |")
        lines+=['','## Per-arm bias metrics','','| Arm | Sharpness | Bias norm | Steadiness | Mean KL | Null sharpness | Null steadiness | Matchable loop-word coverage | Content own / shuffled | Mean percentile |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm,v in result.get('per_arm_metrics',{}).items():
            lines.append(f"| {arm} | {v['sharpness_mean']} | {v['bias_norm_mean']} | {v['steadiness_mean']} | {v['kl_mean']} | {v['sharpness_null_count']} | {v['steadiness_null_count']} | {v['matchable_loop_word_coverage']} | {v['content_word_own_rate']} / {v['content_word_shuffled_rate']} | {v['mean_content_word_percentile']} |")
        lines+=['',f"Predictions: {json.dumps(result['predictions'],ensure_ascii=True)}",'',
                f"Bootstrap intervals: {json.dumps(result.get('bootstrap_intervals',{}),ensure_ascii=True)}",'',
                f"Unpushed control: loops={result.get('unpushed_loop_count')}, low-k content-word overlap={result.get('low_k_content_word_share_in_unpushed_loops')}",
                f"No single-token spellings: {json.dumps(result.get('loop_token_words_without_spelling',[]),ensure_ascii=True)}",
                f"Arm-level 166.5 Spearman: {result.get('arm_level_166_5_spearman')}",
                f"166.5 within arm: {json.dumps(result.get('within_arm_166_5',{}),ensure_ascii=True)}",
                f"166.5 within layer: {json.dumps(result.get('within_layer_166_5',{}),ensure_ascii=True)}"]
    (out/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--rows',default='results/part166/rows.jsonl'); p.add_argument('--plain',default='results/part166/plain.json'); p.add_argument('--tokens',default='results/part166/tokens.json'); p.add_argument('--answers',default='results/part165/answers.jsonl'); p.add_argument('--effects',default='results/part165/effects.jsonl'); p.add_argument('--out',default='results/part166')
    a=p.parse_args(argv); result=score(a.rows,a.plain,a.tokens,a.answers,a.effects,a.out); print(json.dumps({'gates':result['gates'],'predictions':result.get('predictions',{})},ensure_ascii=True,allow_nan=False),flush=True)
if __name__=='__main__': main()
