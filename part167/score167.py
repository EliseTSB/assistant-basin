#!/usr/bin/env python3
"""Host-side gates and registered Part 167 comparisons."""
from __future__ import annotations
import argparse,json,math,random,sys
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
for p in (str(ROOT),str(ROOT/'part86')):
    if p not in sys.path:sys.path.insert(0,p)
from part165 import score165
from part165.run165 import LAYERS
from part166 import score166
import triage86
ARMS=('n+','k4','full'); BANS=('none','own','matched'); SLOTS=tuple(range(12)); EXPECTED_ROWS=3*8*8*12; EXPECTED_BATTERY=EXPECTED_ROWS*3

def read_jsonl(p):
    p=Path(p)
    if not p.exists():return []
    with p.open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def loop_phrase(a):return score166.loop_phrase(a)
def row_words(row,tokens):return score166.row_words(row,tokens)
def content_words(answer,stop):return {w.casefold() for w in score165.LOOP_WORDS.findall(answer or '') if w.casefold() not in stop}
def _ids_words(ids,tokens):
    return {w.casefold() for tid in ids for w in score165.LOOP_WORDS.findall(str(tokens.get(str(int(tid)),'')))}
def relative_cut(base,rate):return (1-rate/base) if base is not None and rate is not None and base else None
def passes_prediction(pid,**v):
    if pid=='167.1':return v.get('ratio') is not None and v['ratio']>=5
    if pid=='167.2':return v.get('full',0)>0 and v.get('nplus',0)>=2*v['full']
    if pid=='167.3':return v.get('own') is not None and v['own']>=.30
    if pid=='167.4':return v.get('own_rate') is not None and v.get('matched_rate') is not None and v['own_rate']<=v['matched_rate']-.05
    if pid=='167.5':return v.get('matched') is not None and -.15<=v['matched']<=.15
    if pid=='167.6':return v.get('full') is not None and v.get('own') is not None and v['full']<.5*v['own']
    if pid=='167.7':return v.get('migration') is not None and v['migration']>=1/3 and v.get('migration',0)>v.get('baseline',math.inf)
    raise KeyError(pid)

def slot_bootstrap(events,metric,seed=167,reps=10000):
    grouped=defaultdict(list)
    for e in events:grouped[int(e['layer'])].append(e)
    layers=sorted(grouped);rng=random.Random(seed);draws=[]
    for _ in range(reps):
        sample=[]
        for l in layers:
            slots={s:[e for e in grouped[l] if int(e['slot'])==s] for s in SLOTS}
            for _ in SLOTS:sample.extend(slots[rng.choice(SLOTS)])
        v=metric(sample)
        if v is not None and math.isfinite(v):draws.append(float(v))
    if not draws:return {'interval_95':[None,None],'usable_draws':0,'replicates':reps,'seed':seed}
    draws.sort();return {'interval_95':[draws[int(.025*(len(draws)-1))],draws[int(.975*(len(draws)-1))]],'usable_draws':len(draws),'replicates':reps,'seed':seed}

def transition_test(none_answer,own_answer,row,rank_ids,tokens,stop):
    """Return paired own-word removal/new-word transition and migration evidence."""
    old=loop_phrase(none_answer.get('answer','')); new=loop_phrase(own_answer.get('answer',''))
    own_ids=row.get('own_ban_ids',row.get('top_ids',[])[:50])
    own_set=_ids_words(own_ids,tokens); old_content=content_words(' '.join(old or ()),stop)&own_set
    new_content=content_words(' '.join(new or ()),stop)
    removed=old_content-new_content; additions=new_content-old_content
    qualifies=bool(old and new and removed and additions)
    migrated=bool(qualifies and additions&_ids_words(rank_ids,tokens))
    reentry=bool(old and new and old_content&new_content)
    return {'qualifies':qualifies,'induced':bool(new and not old),'removed_words':sorted(removed),'new_words':sorted(additions),'migrated':migrated,'reentry':reentry}

def transition_baseline(new_words,other_rank_lists,tokens):
    if not other_rank_lists:return 0.0
    return sum(bool(set(new_words)&_ids_words(ids,tokens)) for ids in other_rank_lists)/len(other_rank_lists)

def _metrics(records,rows,tokens,stop,masses):
    rowmap={(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index'])):r for r in rows}
    amap={(a['arm'],a['ban'],int(a['layer']),int(a['slot']),int(a['question_index'])):a for a in records if a.get('stage')=='battery'}
    cells=defaultdict(list); events=[]; reentry=defaultdict(int); own_content=shuffled_content=0.0
    for a in records:
        if a.get('stage')!='battery':continue
        arm,ban=a['arm'],a['ban'];key=(arm,int(a['layer']),int(a['slot']),int(a['question_index'])); row=rowmap[key]
        phrase=loop_phrase(a.get('answer','')); own_ids=row.get('own_ban_ids',row.get('top_ids',[])[:50]); own_vocab=_ids_words(own_ids,tokens)
        answer_vocab={w.casefold() for w in score165.LOOP_WORDS.findall(a.get('answer',''))}
        reentry[(arm,ban)]+=len(own_vocab&answer_vocab) if ban=='own' else 0
        r={'arm':arm,'ban':ban,'layer':key[1],'slot':key[2],'question_index':key[3],
           'loop':bool(phrase),'broken':score165.primary_broken(a.get('answer','')),
           'joint_success':not bool(phrase) and not score165.primary_broken(a.get('answer','')),
           'kept':triage86.triage_answer(a.get('answer',''))=='kept',
           'token_count':int(a.get('token_count',len(a.get('token_ids',[])))),
           'eos_before_80':bool(a.get('ended_eos_before_80',False))}
        if ban=='none' and arm in ('n+','k4') and phrase:
            cwords=content_words(' '.join(phrase),stop); own_content+=sum(w in own_vocab for w in cwords)
            donors=[rowmap.get((arm,key[1],s,key[3]),{}) for s in SLOTS if s!=key[2]]
            shuffled_content+=sum(sum(w in _ids_words(d.get('own_ban_ids',d.get('top_ids',[])[:50]),tokens) for d in donors)/len(donors) for w in cwords) if donors else 0
        r['content_own']=sum(w in own_vocab for w in content_words(' '.join(phrase or ()),stop)) if ban=='none' else 0
        r['content_shuffled']=0.0
        if ban=='none' and arm in ('n+','k4') and phrase:
            cwords=content_words(' '.join(phrase),stop); donors=[rowmap.get((arm,key[1],s,key[3]),{}) for s in SLOTS if s!=key[2]]
            r['content_shuffled']=sum(sum(w in _ids_words(d.get('own_ban_ids',d.get('top_ids',[])[:50]),tokens) for d in donors)/len(donors) for w in cwords) if donors else 0.0
        cells[(arm,ban)].append(r);events.append(r)
    out={}
    for arm in ARMS:
        for ban in BANS:
            xs=cells[(arm,ban)];n=len(xs)
            out[f'{arm}|{ban}']={'n':n,'broken_rate':sum(x['broken'] for x in xs)/n if n else None,
                'loop_rate':sum(x['loop'] for x in xs)/n if n else None,
                'mean_token_count':sum(x['token_count'] for x in xs)/n if n else None,
                'eos_before_80_share':sum(x['eos_before_80'] for x in xs)/n if n else None,
                'joint_success_rate':sum(x['joint_success'] for x in xs)/n if n else None,
                'triage_kept_rate_identity_rule':sum(x['kept'] for x in xs)/n if n else None,
                'own_top50_normalized_word_reentry_as_text':reentry[(arm,ban)]}
    def rate(arms,ban):
        xs=[r for r in events if r['arm'] in arms and r['ban']==ban]
        return sum(x['loop'] for x in xs)/len(xs) if xs else None
    low=['n+','k4']; none_rate=rate(low,'none'); own_rate=rate(low,'own'); matched_rate=rate(low,'matched')
    own_cut=relative_cut(none_rate,own_rate); matched_cut=relative_cut(none_rate,matched_rate)
    own_diff=none_rate-own_rate if none_rate is not None and own_rate is not None else None
    matched_diff=none_rate-matched_rate if none_rate is not None and matched_rate is not None else None
    # Paired transitions use row and phrasing as the unit. Other slots form each transition's baseline.
    transitions=[]; induced=0; reentries=0
    for arm in low:
        for layer in LAYERS:
            for slot in SLOTS:
                for qi in range(8,16):
                    none=amap.get((arm,'none',layer,slot,qi)); own=amap.get((arm,'own',layer,slot,qi))
                    if not own:continue
                    none_loop=loop_phrase(none.get('answer','')) if none else None
                    own_loop=loop_phrase(own.get('answer',''))
                    if own_loop and not none_loop:induced+=1
                    row=rowmap.get((arm,layer,slot,qi),{})
                    test=transition_test(none or {},own,row,row.get('rank_51_200_ids',[]),tokens,stop)
                    reentries+=int(test['reentry'])
                    if not test['qualifies']:continue
                    other_lists=[]
                    for other in SLOTS:
                        if other==slot:continue
                        otherrow=rowmap.get((arm,layer,other,qi),{})
                        other_lists.append(otherrow.get('rank_51_200_ids',[]))
                    transitions.append({**test,'baseline_share':transition_baseline(test['new_words'],other_lists,tokens)})
    migration_share=sum(t['migrated'] for t in transitions)/len(transitions) if transitions else None
    baseline=sum(t['baseline_share'] for t in transitions)/len(transitions) if transitions else None
    # Intervals are reported for both relative cuts and absolute differences.
    boot={}
    for label,ban in (('own','own'),('matched','matched')):
        def rel(xs,ban=ban):
            b=rate_sample(xs,low,'none');v=rate_sample(xs,low,ban);return relative_cut(b,v)
        def absolute(xs,ban=ban):return rate_sample(xs,low,'none')-rate_sample(xs,low,ban)
        boot[label+'_relative_cut']=slot_bootstrap(events,rel)
        boot[label+'_absolute_difference']=slot_bootstrap(events,absolute)
    boot['full_own_relative_cut']=slot_bootstrap(events,lambda xs:relative_cut(rate_sample(xs,['full'],'none'),rate_sample(xs,['full'],'own')))
    boot['full_own_absolute_difference']=slot_bootstrap(events,lambda xs:(rate_sample(xs,['full'],'none')-rate_sample(xs,['full'],'own')) if rate_sample(xs,['full'],'none') is not None and rate_sample(xs,['full'],'own') is not None else None)
    mass_out={}
    for arm in ARMS:
        xs=[m for m in masses if m.get('arm')==arm]
        mass_out[arm]={k:(sum(float(x[k]) for x in xs)/len(xs) if xs else None) for k in ('own_mass','matched_mass','overlap','allowed_vocab')}
    full_cut=relative_cut(rate(['full'],'none'),rate(['full'],'own'))
    content_ratio=own_content/shuffled_content if shuffled_content else (math.inf if own_content else None)
    nbroken=next((m['broken_rate'] for k,m in out.items() if k=='n+|none'),None); fbroken=out['full|none']['broken_rate']
    broken_ratio=nbroken/fbroken if nbroken is not None and fbroken else None
    predictions={
      '167.1':{'content_own':own_content,'content_shuffled':shuffled_content,'ratio':('Infinity' if content_ratio is not None and math.isinf(content_ratio) else content_ratio),'pass':passes_prediction('167.1',ratio=content_ratio)},
      '167.2':{'nplus_broken':nbroken,'full_broken':fbroken,'ratio':broken_ratio,'pass':passes_prediction('167.2',nplus=nbroken or 0,full=fbroken or 0)},
      '167.3':{'own_relative_cut':own_cut,'own_relative_cut_interval':boot['own_relative_cut'],'own_absolute_difference':own_diff,'own_absolute_difference_interval':boot['own_absolute_difference'],'pass':passes_prediction('167.3',own=own_cut)},
      '167.4':{'own_rate':own_rate,'matched_rate':matched_rate,'absolute_rate_gap':matched_rate-own_rate if own_rate is not None and matched_rate is not None else None,'pass':passes_prediction('167.4',own_rate=own_rate,matched_rate=matched_rate)},
      '167.5':{'matched_relative_cut':matched_cut,'matched_relative_cut_interval':boot['matched_relative_cut'],'matched_absolute_difference':matched_diff,'matched_absolute_difference_interval':boot['matched_absolute_difference'],'pass':passes_prediction('167.5',matched=matched_cut)},
      '167.6':{'full_own_relative_cut':full_cut,'full_own_relative_cut_interval':boot['full_own_relative_cut'],'full_own_absolute_difference':rate(['full'],'none')-rate(['full'],'own'),'full_own_absolute_difference_interval':boot['full_own_absolute_difference'],'pooled_low_k_own_relative_cut':own_cut,'pass':passes_prediction('167.6',full=full_cut,own=own_cut)},
      '167.7':{'transition_count':len(transitions),'migration_share':migration_share,'baseline_share':baseline,'pass':passes_prediction('167.7',migration=migration_share,baseline=baseline)}}
    return {'per_arm_ban':out,'predictions':predictions,'bootstrap_intervals':boot,'mass_means_by_arm':mass_out,
            'transitions':{'count':len(transitions),'migrated':sum(t['migrated'] for t in transitions),'baseline_share':baseline,'induced_own_ban_loops':induced,'same_word_reentry_transitions':reentries}}

def rate_sample(xs,arms,ban):
    vals=[e for e in xs if e['arm'] in arms and e['ban']==ban]
    return sum(e['loop'] for e in vals)/len(vals) if vals else None

def score(rows_path,plain_path,tokens_path,answers_path,out_dir):
    rows=read_jsonl(rows_path);json.loads(Path(plain_path).read_text(encoding='utf-8'));tokens=json.loads(Path(tokens_path).read_text(encoding='utf-8'));answers=read_jsonl(answers_path)
    loop_config=json.loads((ROOT/'results/part166/loop_tokens.json').read_text(encoding='utf-8'))
    stop={str(w).casefold() for w in loop_config.get('stop_list',[])}
    masses=read_jsonl(Path(answers_path).with_name('masses.jsonl'))
    keys=[(r.get('arm'),int(r.get('layer',-1)),int(r.get('slot',-1)),int(r.get('question_index',-1))) for r in rows]
    battery=[a for a in answers if a.get('stage')=='battery'];no_push=[a for a in answers if a.get('stage')=='no_push']
    finite=all(all(math.isfinite(float(r[k])) for k in ('sharpness','steadiness','bias_norm')) and len(r.get('top_ids',[]))>=200 and len(r.get('top_values',[]))>=200 for r in rows)
    gate1=len(rows)==EXPECTED_ROWS and len(set(keys))==len(keys) and len(battery)==EXPECTED_BATTERY and len({(a.get('arm'),a.get('ban'),a.get('layer'),a.get('slot'),a.get('question_index')) for a in battery})==EXPECTED_BATTERY and finite and all(isinstance(a.get('answer'),str) and a.get('answer').strip() and len(a.get('token_ids',[]))>0 for a in battery)
    gate2=len(no_push)==8 and all(not score165.primary_broken(a.get('answer','')) for a in no_push)
    norms_path=Path(answers_path).with_name('norms.json');norms=json.loads(norms_path.read_text(encoding='utf-8')) if norms_path.exists() else {'directions':[]};nvals=norms.get('directions',[])
    gate3=len(nvals)==8*12*3 and all(float(x.get('relative_error',math.inf))<=.01 for x in nvals)
    rmap={(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index'])):r for r in rows};gate4=True
    for a in battery:
        if a.get('ban') not in ('own','matched'):continue
        r=rmap.get((a['arm'],int(a['layer']),int(a['slot']),int(a['question_index'])),{})
        default=r.get('own_ban_ids',r.get('top_ids',[])[:50]) if a['ban']=='own' else r.get('matched_ban_ids',a.get('banned_ids',[]))
        banned=set(map(int,a.get('banned_ids',default)))
        if banned&set(map(int,a.get('token_ids',[]))):gate4=False;break
    none_low=[a for a in battery if a.get('ban')=='none' and a.get('arm') in ('n+','k4')]
    none_nplus=[a for a in battery if a.get('ban')=='none' and a.get('arm')=='n+']
    broken_rate=sum(score165.primary_broken(a.get('answer','')) for a in none_nplus)/len(none_nplus) if none_nplus else 0
    loop_rate=sum(bool(loop_phrase(a.get('answer',''))) for a in none_low)/len(none_low) if none_low else 0
    gate5=bool(none_nplus) and broken_rate>=.08 and bool(none_low) and loop_rate>=.08
    gates={'gate1':gate1,'rows':len(rows),'answers':len(battery),'gate2':gate2,'no_push_answers':len(no_push),'gate3':gate3,'norm_rows':len(nvals),'gate4':gate4,'gate5':gate5,'nplus_none_broken_rate':broken_rate,'pooled_low_k_none_loop_rate':loop_rate}
    stopped=not all((gate1,gate2,gate3,gate4,gate5));metrics={} if stopped else _metrics(answers,rows,tokens,stop,masses)
    result={'gates':gates,'readings_stopped':stopped,'common_words':sorted(stop),**metrics}
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);(out/'score.json').write_text(json.dumps(result,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    sample=[];key=[];rng=random.Random(167);blind=0
    for ban in BANS:
        vals=[a for a in battery if a.get('arm') in ('n+','k4') and a.get('ban')==ban]
        for a in rng.sample(vals,min(30,len(vals))):
            blind+=1;sample.append({'blind_id':f'HR{blind:03d}','answer':a.get('answer',''),'token_ids':a.get('token_ids',[]),'token_count':a.get('token_count',0)})
            key.append({'blind_id':f'HR{blind:03d}','arm':a['arm'],'ban':a['ban'],'layer':a['layer'],'slot':a['slot'],'question_index':a['question_index']})
    rng.shuffle(sample)
    for pth,vals in ((out/'hand_read.jsonl',sample),(out/'hand_read_key.jsonl',key)):
        with pth.open('w',encoding='utf-8',newline='\n') as f:
            for x in vals:f.write(json.dumps(x,ensure_ascii=True)+'\n')
    lines=['# Part 167 score','','Gates: '+json.dumps(gates,ensure_ascii=True),'']
    if stopped:lines.append('All readings stopped because one or more gates failed.')
    else:
        lines+=['| Arm | Ban | Broken | Loop | Mean tokens | EOS before 80 | Joint success | Triage kept (identity rule) | Own-word re-entry |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            for ban in BANS:
                m=result['per_arm_ban'][f'{arm}|{ban}'];lines.append(f"| {arm} | {ban} | {m['broken_rate']} | {m['loop_rate']} | {m['mean_token_count']} | {m['eos_before_80_share']} | {m['joint_success_rate']} | {m['triage_kept_rate_identity_rule']} | {m['own_top50_normalized_word_reentry_as_text']} |")
        lines+=['','Mass means by arm: '+json.dumps(result['mass_means_by_arm'],ensure_ascii=True),'','Predictions and absolute differences: '+json.dumps(result['predictions'],ensure_ascii=True),'','Transition counts: '+json.dumps(result['transitions'],ensure_ascii=True),'','Bootstrap intervals: '+json.dumps(result['bootstrap_intervals'],ensure_ascii=True)]
    (out/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rows',default='results/part167/rows.jsonl');p.add_argument('--plain',default='results/part167/plain.json');p.add_argument('--tokens',default='results/part167/tokens.json');p.add_argument('--answers',default='results/part167/answers.jsonl');p.add_argument('--out',default='results/part167');a=p.parse_args(argv);r=score(a.rows,a.plain,a.tokens,a.answers,a.out);print(json.dumps({'gates':r['gates'],'predictions':r.get('predictions',{})},ensure_ascii=True,allow_nan=False),flush=True)
if __name__=='__main__':main()
