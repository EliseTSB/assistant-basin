#!/usr/bin/env python3
"""Host-side gates and registered Part 168 comparisons."""
from __future__ import annotations
import argparse,json,math,random,sys
from collections import Counter,defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
for p in (str(ROOT),str(ROOT/'part86')):
    if p not in sys.path:sys.path.insert(0,p)
from part165 import score165
from part165.run165 import LAYERS
from part166 import score166
import triage86
ARMS=('n+','k4','full'); BANS=('none','own','matched','mass'); SLOTS=tuple(range(12)); PHRASINGS=(8,9,10,11,12,14,15); EXPECTED_ROWS=3*8*7*12; EXPECTED_BATTERY=EXPECTED_ROWS*3+2*8*7*12

def read_jsonl(p):
    p=Path(p)
    if not p.exists():return []
    with p.open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def neuron_for_seed(width,layer,slot,base,torch=None,attempt=0):
    import torch as th
    g=th.Generator(device='cpu');g.manual_seed(int(base)+100*int(layer)+int(slot)+1_000_000*int(attempt))
    return int(th.randperm(int(width),generator=g,device='cpu')[0])

def loop_phrase(a):return score166.loop_phrase(a)
def row_words(row,tokens):return score166.row_words(row,tokens)
def content_words(answer,stop):return {w.casefold() for w in score165.LOOP_WORDS.findall(answer or '') if w.casefold() not in stop}
def _ids_words(ids,tokens):
    return {w.casefold() for tid in ids for w in score165.LOOP_WORDS.findall(str(tokens.get(str(int(tid)),'')))}
def relative_cut(base,rate):return (1-rate/base) if base is not None and rate is not None and base else None
def passes_prediction(pid,**v):
    if pid=='168.1':return v.get('ratio') is not None and v['ratio']>=5
    if pid=='168.2':return v.get('full',0)>0 and v.get('nplus',0)>=2*v['full']
    if pid=='168.3':return v.get('own') is not None and v['own']>=.30
    if pid=='168.4':return v.get('own_rate') is not None and v.get('matched_rate') is not None and v['own_rate']<=v['matched_rate']-.05
    if pid=='168.5':return v.get('matched') is not None and -.15<=v['matched']<=.15
    if pid=='168.6':return v.get('full') is not None and v.get('own') is not None and v['full']<.5*v['own']
    if pid=='168.7':return v.get('migration') is not None and v['migration']>=1/3 and v.get('migration',0)>v.get('baseline',math.inf)
    if pid=='168.8':return v.get('own_rate') is not None and v.get('mass_rate') is not None and v['own_rate']<=v['mass_rate']-.05
    if pid=='168.W1':return v.get('mass') is not None and .15<=v['mass']<=.25
    if pid=='168.W2':return v.get('own') is not None and v.get('matched') is not None and v['own']-v['matched']>=.02
    if pid=='168.W3':return v.get('migration') is not None and .15<=v['migration']<=.30
    raise KeyError(pid)

def bootstrap_intervals(events,seed=168,reps=10000):
    """Whole-slot resampling, stratified by layer. Slots are summed to counts first, so 10,000 replicates take seconds."""
    import numpy as np
    layers=sorted({int(e['layer']) for e in events}); keys=sorted({(e['arm'],e['ban']) for e in events}); at={k:i for i,k in enumerate(keys)}
    counts=np.zeros((len(layers),len(SLOTS),len(keys),2))
    for e in events:
        c=counts[layers.index(int(e['layer'])),int(e['slot']),at[(e['arm'],e['ban'])]]; c[0]+=bool(e['loop']); c[1]+=1
    rng=np.random.default_rng(seed); total=np.zeros((reps,len(keys),2))
    for i in range(len(layers)):
        total+=counts[i][rng.integers(0,len(SLOTS),size=(reps,len(SLOTS)))].sum(axis=1)
    def rate(arms,ban):
        ix=[at[(a,ban)] for a in arms if (a,ban) in at]
        return total[:,ix,0].sum(axis=1)/np.maximum(total[:,ix,1].sum(axis=1),1)
    def interval(v):
        v=np.sort(v[np.isfinite(v)])
        return {'interval_95':[float(v[int(.025*(len(v)-1))]),float(v[int(.975*(len(v)-1))])] if len(v) else [None,None],'usable_draws':int(len(v)),'replicates':reps,'seed':seed}
    low=['n+','k4']; out={}
    for label,arms,ban in (('own',low,'own'),('matched',low,'matched'),('mass',low,'mass'),('full_own',['full'],'own')):
        base=rate(arms,'none'); value=rate(arms,ban)
        with np.errstate(divide='ignore',invalid='ignore'): cut=1-value/base
        out[label+'_relative_cut']=interval(cut); out[label+'_absolute_difference']=interval(base-value)
    out['matched_minus_own_rate']=interval(rate(low,'matched')-rate(low,'own')); out['mass_minus_own_rate']=interval(rate(low,'mass')-rate(low,'own'))
    return out

def transition_summary(arms,ban,rowmap,amap,tokens,stop):
    """Follow each none loop on an own-list word under one ban. Classes are exclusive; the focal word is the first new content word in the banned loop."""
    fates=Counter(); baselines=[]; induced=eligible=0
    for arm in arms:
      for layer in LAYERS:
       for slot in SLOTS:
        for qi in PHRASINGS:
         none=amap.get((arm,'none',layer,slot,qi)); alt=amap.get((arm,ban,layer,slot,qi)); row=rowmap.get((arm,layer,slot,qi),{})
         if not none or not alt:continue
         old=loop_phrase(none.get('answer','')); new=loop_phrase(alt.get('answer',''))
         if not old:eligible+=1;induced+=int(bool(new))
         ow=content_words(' '.join(old or ()),stop); nw=content_words(' '.join(new or ()),stop)
         own=_ids_words(row.get('own_ban_ids',[]),tokens); held=ow&own
         if not held:continue
         if not new:fates['no_loop']+=1;continue
         if held&nw:fates['reentry']+=1;continue
         focal=next((w for w in (x.casefold() for x in score165.LOOP_WORDS.findall(' '.join(new))) if w not in stop and w not in ow),None)
         if not nw:fates['common_only']+=1
         elif focal is None:fates['other']+=1
         else:
             mid=_ids_words(row.get('rank_51_200_ids',[]),tokens)
             fates['top50' if focal in own else 'migration' if focal in mid else 'outside']+=1
         others=[focal in _ids_words(rowmap.get((arm,layer,other,qi),{}).get('rank_51_200_ids',[]),tokens) for other in SLOTS if other!=slot] if focal else []
         baselines.append(sum(others)/len(others) if others else 0.0)
    n=sum(fates[k] for k in ('common_only','other','top50','migration','outside'))
    return {'count':n,'migration_count':fates['migration'],'outside_count':fates['outside'],'top50_count':fates['top50'],'common_only_count':fates['common_only'],'other_count':fates['other'],
      'migration_share':fates['migration']/n if n else None,'outside_share':fates['outside']/n if n else None,'top50_share':fates['top50']/n if n else None,'common_only_share':fates['common_only']/n if n else None,
      'migration_baseline_share':sum(baselines)/n if n else None,'own_word_loop_fates':dict(fates),
      'induced_count':induced,'induced_eligible':eligible,'induced_share':induced/eligible if eligible else None,'same_word_reentry_rows':fates['reentry']}

def _metrics(records,rows,tokens,stop,masses):
    rowmap={(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index'])):r for r in rows}
    amap={(a['arm'],a['ban'],int(a['layer']),int(a['slot']),int(a['question_index'])):a for a in records if a.get('stage')=='battery'}
    events=[]; cells=defaultdict(list)
    for a in records:
        if a.get('stage')!='battery':continue
        arm,ban=a['arm'],a['ban']; key=(arm,int(a['layer']),int(a['slot']),int(a['question_index'])); phrase=loop_phrase(a.get('answer',''))
        cells[(arm,ban)].append({'arm':arm,'ban':ban,'layer':key[1],'slot':key[2],'question_index':key[3],
          'loop':bool(phrase),'broken':score165.primary_broken(a.get('answer','')),
          'joint_success':not bool(phrase) and not score165.primary_broken(a.get('answer','')),
          'token_count':int(a.get('token_count',len(a.get('token_ids',[])))),
          'eos_before_80':bool(a.get('ended_eos_before_80',False))})
    def rate(arms,ban):
        xs=[e for (arm,b),vals in cells.items() if arm in arms and b==ban for e in vals]
        return sum(e['loop'] for e in xs)/len(xs) if xs else None
    per={}
    for arm in ARMS:
        for ban in (('none','own','matched','mass') if arm in ('n+','k4') else ('none','own','matched')):
            xs=cells[(arm,ban)]; n=len(xs)
            per[f'{arm}|{ban}']={'n':n,'broken_rate':sum(x['broken'] for x in xs)/n if n else None,
              'loop_rate':sum(x['loop'] for x in xs)/n if n else None,'mean_token_count':sum(x['token_count'] for x in xs)/n if n else None,
              'eos_before_80_share':sum(x['eos_before_80'] for x in xs)/n if n else None,
              'joint_success_rate':sum(x['joint_success'] for x in xs)/n if n else None}
    low=['n+','k4']; none_rate=rate(low,'none'); rates={b:rate(low,b) for b in ('own','matched','mass')}
    events=[e for vals in cells.values() for e in vals]
    boot=bootstrap_intervals(events)
    paired={(m['arm'],int(m['layer']),int(m['slot']),int(m['question_index'])) for m in masses if m.get('paired_set')}
    pair_events=[e for e in events if e['arm'] in low and (e['arm'],e['layer'],e['slot'],e['question_index']) in paired]
    boot['paired_mass_minus_own_rate']=bootstrap_intervals(pair_events)['mass_minus_own_rate'] if pair_events else None
    def pair_rate(ban):
        xs=[e for e in pair_events if e['ban']==ban]
        return sum(e['loop'] for e in xs)/len(xs) if xs else None
    pair_own=pair_rate('own'); pair_mass=pair_rate('mass'); pair_none=pair_rate('none')
    allrows=[e for e in events if e['arm'] in low and e['ban']=='mass']; unmatched=[e for e in allrows if (e['arm'],e['layer'],e['slot'],e['question_index']) not in paired]
    known=sum(e['loop'] for e in allrows if (e['arm'],e['layer'],e['slot'],e['question_index']) in paired)
    coverage=Counter(f"{k[0]}|{k[1]}" for k in paired)
    mass_sensitivity={'paired_rows':len(paired),'unmatched_rows':len(unmatched),'paired_coverage_by_arm_layer':dict(coverage),'all_row_mass_rate':rates['mass'],'all_row_own_rate':rates['own'],'paired_none_rate':pair_none,
      'all_row_mass_rate_bounds':[known/len(allrows),(known+len(unmatched))/len(allrows)] if allrows else None}
    transitions={ban:{**transition_summary(low,ban,rowmap,amap,tokens,stop),'per_arm':{arm:transition_summary([arm],ban,rowmap,amap,tokens,stop) for arm in low}} for ban in ('own','matched','mass')}
    induced={ban:{'count':v['induced_count'],'eligible':v['induced_eligible'],'share':v['induced_share']} for ban,v in transitions.items()}
    reentry={ban:{'same_word_reentry_rows':v['same_word_reentry_rows'],'paired_transitions':v['count']} for ban,v in transitions.items()}
    mass_stats={}
    for arm in ARMS:
        vals=[m for m in masses if m.get('arm')==arm]
        mass_stats[arm]={}
        for k in ('own_mass','matched_mass','mass_mass','own_mass_overlap','matched_mass_overlap','overlap','plain_overlap_own','plain_overlap_matched','plain_overlap_mass'):
            xs=[float(x[k]) for x in vals if x.get(k) is not None]
            mass_stats[arm][k+'_mean']=sum(xs)/len(xs) if xs else None
        tol=[x for x in vals if x.get('in_tolerance') is not None]
        mass_stats[arm]['infeasible_rows']=sum(x.get('feasible') is False for x in vals)
        mass_stats[arm]['repaired_rows']=sum(bool(x.get('repaired')) for x in vals)
        errs=sorted(float(x['mass_rel_error']) for x in vals if x.get('mass_rel_error') is not None); mass_stats[arm]['mass_rel_error_median']=errs[len(errs)//2] if errs else None
        mass_stats[arm]['in_tolerance_share']=sum(bool(x['in_tolerance']) for x in tol)/len(tol) if tol else None
    nbroken=per['n+|none']['broken_rate']; fbroken=per['full|none']['broken_rate']; broken_ratio=nbroken/fbroken if fbroken else None
    own_cut=relative_cut(none_rate,rates['own']); matched_cut=relative_cut(none_rate,rates['matched']); mass_cut=relative_cut(none_rate,rates['mass'])
    gap=(rates['matched']-rates['own']) if rates['matched'] is not None and rates['own'] is not None else None
    migration=transitions['own']['migration_share']; baseline=transitions['own']['migration_baseline_share']
    induced_own=induced['own']['share']; induced_matched=induced['matched']['share']
    predictions={
      '168.1':{'content_word_own_over_shuffled':None,'pass':False},
      '168.2':{'nplus_broken':nbroken,'full_broken':fbroken,'ratio':broken_ratio,'pass':fbroken is not None and fbroken>0 and nbroken>=2*fbroken},
      '168.3':{'own_relative_cut':own_cut,'pass':own_cut is not None and own_cut>=.30},
      '168.4':{'own_rate':rates['own'],'matched_rate':rates['matched'],'absolute_rate_gap':gap,'pass':rates['own'] is not None and rates['matched'] is not None and rates['own']<=rates['matched']-.05},
      '168.5':{'matched_relative_cut':matched_cut,'pass':matched_cut is not None and -.15<=matched_cut<=.15},
      '168.6':{'full_own_relative_cut':relative_cut(per['full|none']['loop_rate'],per['full|own']['loop_rate']),'pooled_low_k_own_relative_cut':own_cut,'pass':relative_cut(per['full|none']['loop_rate'],per['full|own']['loop_rate']) is not None and own_cut is not None and relative_cut(per['full|none']['loop_rate'],per['full|own']['loop_rate'])<.5*own_cut},
      '168.7':{'transition_count':transitions['own']['count'],'migration_share':migration,'baseline_share':baseline,'pass':migration is not None and migration>=1/3 and migration>(baseline if baseline is not None else math.inf)},
      '168.8':{'paired_own_rate':pair_own,'paired_mass_rate':pair_mass,'pass':pair_own is not None and pair_mass is not None and pair_own<=pair_mass-.05},
      '168.W1':{'mass_relative_cut':mass_cut,'pass':mass_cut is not None and .15<=mass_cut<=.25},
      '168.W2':{'induced_own_share':induced_own,'induced_matched_share':induced_matched,'difference':induced_own-induced_matched if induced_own is not None and induced_matched is not None else None,'pass':induced_own is not None and induced_matched is not None and induced_own-induced_matched>=.02},
      '168.W3':{'migration_share':migration,'pass':migration is not None and .15<=migration<=.30}}
    # Prediction 168.1 counts content words in none loops against all other same-arm slots' own lists.
    own_ct=shuffle_ct=0.0
    for a in records:
        if a.get('stage')!='battery' or a.get('ban')!='none' or a.get('arm') not in low:continue
        phrase=loop_phrase(a.get('answer',''))
        if not phrase:continue
        key=(a['arm'],int(a['layer']),int(a['slot']),int(a['question_index'])); row=rowmap[key]; words=content_words(' '.join(phrase),stop)
        ownset=_ids_words(row.get('own_ban_ids',[]),tokens); own_ct+=sum(w in ownset for w in words)
        ds=[rowmap.get((key[0],key[1],ss,key[3]),{}) for ss in SLOTS if ss!=key[2]]
        shuffle_ct+=sum(sum(w in _ids_words(d.get('own_ban_ids',[]),tokens) for d in ds)/len(ds) for w in words) if ds else 0
    ratio=own_ct/shuffle_ct if shuffle_ct else (math.inf if own_ct else None)
    predictions['168.1']={'content_own':own_ct,'content_shuffled':shuffle_ct,'ratio':('Infinity' if math.isinf(ratio) else ratio) if ratio is not None else None,'pass':ratio is not None and ratio>=5}
    per_arm={arm:{ban:per.get(f'{arm}|{ban}') for ban in (('none','own','matched','mass') if arm in low else ('none','own','matched'))} for arm in ARMS}
    return {'per_arm_ban':per,'per_arm_reports':per_arm,'predictions':predictions,'bootstrap_intervals':boot,
      'mass_means_by_arm':mass_stats,'mass_sensitivity':mass_sensitivity,'transitions_by_ban':transitions,'induced_by_ban':induced,'reentry_by_ban':reentry}

def score(rows_path,plain_path,tokens_path,answers_path,out_dir):
    rows=read_jsonl(rows_path); json.loads(Path(plain_path).read_text(encoding='utf-8')); tokens=json.loads(Path(tokens_path).read_text(encoding='utf-8')); answers=read_jsonl(answers_path)
    config=json.loads((ROOT/'results/part166/loop_tokens.json').read_text(encoding='utf-8')); stop={str(w).casefold() for w in config.get('stop_list',[])}
    masses=read_jsonl(Path(answers_path).with_name('masses.jsonl')); battery=[a for a in answers if a.get('stage')=='battery']; no=[a for a in answers if a.get('stage')=='no_push']
    keys=[(r.get('arm'),int(r.get('layer',-1)),int(r.get('slot',-1)),int(r.get('question_index',-1))) for r in rows]
    expected_rows=3*8*12*len(PHRASINGS); expected_battery=expected_rows*3+2*8*12*len(PHRASINGS)
    finite=all(all(math.isfinite(float(r[k])) for k in ('sharpness','steadiness','bias_norm')) and len(r.get('top_ids',[]))>=200 and len(r.get('top_values',[]))>=200 for r in rows)
    answer_keys={(a.get('arm'),a.get('ban'),a.get('layer'),a.get('slot'),a.get('question_index')) for a in battery}
    gate1=len(rows)==expected_rows and len(set(keys))==len(keys) and len(battery)==expected_battery and len(answer_keys)==expected_battery and finite and all(isinstance(a.get('answer'),str) and a['answer'].strip() and a.get('token_ids') for a in battery)
    gate2=len(no)==len(PHRASINGS) and {int(a.get('question_index',-1)) for a in no}==set(PHRASINGS) and all(not score165.primary_broken(a.get('answer','')) for a in no)
    norms_path=Path(answers_path).with_name('norms.json'); norms=json.loads(norms_path.read_text(encoding='utf-8')) if norms_path.exists() else {'directions':[]}; nvals=norms.get('directions',[])
    normkeys=[(x.get('arm'),int(x.get('layer',-1)),int(x.get('slot',-1))) for x in nvals]; expected_normkeys={(arm,l,slot) for arm in ARMS for l in LAYERS for slot in SLOTS}; gate3=len(nvals)==len(expected_normkeys) and set(normkeys)==expected_normkeys and len(set(normkeys))==len(normkeys) and all(float(x.get('relative_error',math.inf))<=.01 for x in nvals)
    rmap={(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index'])):r for r in rows}; gate4=True
    mmap={(m['arm'],int(m['layer']),int(m['slot']),int(m['question_index'])):m for m in masses}
    for a in battery:
        key=(a.get('arm'),int(a.get('layer',-1)),int(a.get('slot',-1)),int(a.get('question_index',-1))); ban=a.get('ban'); ids=set(map(int,a.get('banned_ids',[])))
        donor=(key[0],key[1],(key[2]+1)%12,key[3])
        want=set() if ban=='none' else set(rmap.get(key,{}).get('own_ban_ids',[])) if ban=='own' else set(rmap.get(donor,{}).get('own_ban_ids',[])) if ban=='matched' else set(mmap.get(key,{}).get('mass_ban_ids',[]))
        if (ban!='none' and len(want)!=50) or ids!=want or ids&set(map(int,a.get('token_ids',[]))):gate4=False;break
    low=[m for m in masses if m.get('arm') in ('n+','k4')]; masskeys=[(m.get('arm'),int(m.get('layer',-1)),int(m.get('slot',-1)),int(m.get('question_index',-1))) for m in masses]; gate6_share=sum(bool(m.get('in_tolerance')) for m in low)/len(low) if low else 0.0; gate6=len(masses)==expected_rows and len(set(masskeys))==expected_rows and len(low)==2*8*12*len(PHRASINGS) and all(len(m.get('mass_ban_ids',[]))==50 for m in low) and gate6_share>=.90
    seen=defaultdict(set); width_by_layer={}; nplus=[n for n in nvals if n.get('arm')=='n+']; npluskeys={(int(n.get('layer',-1)),int(n.get('slot',-1))) for n in nplus}; gate7=(len(nplus)==8*12 and len(npluskeys)==8*12)
    try:
        for n in nvals:
            if n.get('arm')!='n+':continue
            l=int(n['layer']); slot=int(n['slot']); width=int(n.get('width',norms.get('widths',{}).get(str(l),9216))); width_by_layer[l]=width
            actual=neuron_for_seed(width,l,slot,168000,torch=None,attempt=int(n.get('attempt',0)))
            prev={neuron_for_seed(width,l,s,b,torch=None) for b in (165000,167000) for s in range(12)}
            if int(n.get('neuron',-1))!=actual or actual in prev or actual in seen[l]:gate7=False;break
            seen[l].add(actual)
    except Exception:gate7=False
    gates={'gate1':gate1,'rows':len(rows),'battery_answers':len(battery),'expected_battery_answers':expected_battery,'gate2':gate2,'no_push_answers':len(no),'gate3':gate3,'norm_rows':len(nvals),'gate4':gate4,'gate5':False,'gate6':gate6,'mass_in_tolerance_share':gate6_share,'gate7':gate7}
    none_low=[a for a in battery if a.get('ban')=='none' and a.get('arm') in ('n+','k4')]; none_n=[a for a in battery if a.get('ban')=='none' and a.get('arm')=='n+']
    broken=sum(score165.primary_broken(a.get('answer','')) for a in none_n)/len(none_n) if none_n else 0
    loop=sum(bool(loop_phrase(a.get('answer',''))) for a in none_low)/len(none_low) if none_low else 0
    gates['nplus_none_broken_rate']=broken; gates['pooled_low_k_none_loop_rate']=loop; gates['gate5']=bool(none_n and broken>=.08 and none_low and loop>=.08)
    stopped=not all(gates[k] for k in ('gate1','gate2','gate3','gate4','gate5','gate6','gate7'))
    metrics={} if stopped else _metrics(answers,rows,tokens,stop,masses)
    result={'gates':gates,'readings_stopped':stopped,'common_words':sorted(stop),**metrics}
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); (out/'score.json').write_text(json.dumps(result,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    sample=[]; key=[]; rng=random.Random(168); blind=0
    for ban in ('none','own','matched','mass'):
        vals=[a for a in battery if a.get('arm') in ('n+','k4') and a.get('ban')==ban]
        for a in rng.sample(vals,min(30,len(vals))):
            blind+=1; sample.append({'blind_id':f'HR{blind:03d}','answer':a.get('answer',''),'token_ids':a.get('token_ids',[]),'token_count':a.get('token_count',0)})
            key.append({'blind_id':f'HR{blind:03d}','arm':a['arm'],'ban':a['ban'],'layer':a['layer'],'slot':a['slot'],'question_index':a['question_index']})
    rng.shuffle(sample)
    for pth,vals in ((out/'hand_read.jsonl',sample),(out/'hand_read_key.jsonl',key)):
        with pth.open('w',encoding='utf-8',newline='\n') as f:
            for x in vals:f.write(json.dumps(x,ensure_ascii=True)+'\n')
    lines=['# Part 168 score','','Gates: '+json.dumps(gates,ensure_ascii=True),'']
    if stopped:lines.append('All readings stopped because one or more gates failed.')
    else:
        lines+=['| Arm | Ban | Broken | Loop | Mean tokens | EOS before 80 | Joint success |','|---|---|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            for ban in (('none','own','matched','mass') if arm in ('n+','k4') else ('none','own','matched')):
                m=result['per_arm_ban'][f'{arm}|{ban}']; lines.append(f"| {arm} | {ban} | {m['broken_rate']} | {m['loop_rate']} | {m['mean_token_count']} | {m['eos_before_80_share']} | {m['joint_success_rate']} |")
        lines+=['','Mass means by arm: '+json.dumps(result['mass_means_by_arm'],ensure_ascii=True),'','Mass sensitivity: '+json.dumps(result['mass_sensitivity'],ensure_ascii=True),'','Predictions: '+json.dumps(result['predictions'],ensure_ascii=True),'','Transitions by ban: '+json.dumps(result['transitions_by_ban'],ensure_ascii=True),'','Bootstrap intervals: '+json.dumps(result['bootstrap_intervals'],ensure_ascii=True)]
    (out/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rows',default='results/part168/rows.jsonl');p.add_argument('--plain',default='results/part168/plain.json');p.add_argument('--tokens',default='results/part168/tokens.json');p.add_argument('--answers',default='results/part168/answers.jsonl');p.add_argument('--out',default='results/part168');a=p.parse_args(argv);r=score(a.rows,a.plain,a.tokens,a.answers,a.out);print(json.dumps({'gates':r['gates'],'predictions':r.get('predictions',{})},ensure_ascii=True,allow_nan=False),flush=True)
if __name__=='__main__':main()
