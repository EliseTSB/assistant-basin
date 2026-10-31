#!/usr/bin/env python3
"""Score Part 160 gates, dose peaks, and registered predictions."""
from __future__ import annotations
import argparse, json, math, random, sys
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part148 import score148
from part153 import score153
from part153.battery153 import UNIT_COMPANIES
from part155 import run155 as part155
from part86 import triage86
DOSES=(0.4,0.6,0.8,1.0,1.2,1.4); PICK_DOSES=(0.6,1.0)
POSITIVE={'L24N5017':'Google','L21N7134':'Baidu'}
PART148_TEN={f'L{l}N{n}':c for l,n,c in UNIT_COMPANIES}

def load_jsonl(path):
    out=[]
    p=Path(path)
    if not p.exists(): return out
    with p.open(encoding='utf-8') as f:
        for no,line in enumerate(f,1):
            if line.strip():
                try: out.append(json.loads(line))
                except json.JSONDecodeError as e: raise ValueError(f'invalid JSON at {p}:{no}') from e
    return out

def uid(row):
    return row.get('unit_id') or 'plain'
def key(row):
    try:return (row.get('stage'),uid(row),row.get('arm'),float(row.get('dose')),int(row.get('question_index',-1)))
    except (TypeError,ValueError):return None

def company_hit(answer,company): return score153.company_hit(answer if isinstance(answer,str) else '',company)
def maker_items(): return [q for q in part155.questions() if q['kind']=='maker']
def company_for(picked, dose, unit_id):
    rec=next((u for u in picked.get(str(dose),{}).get('kept',[]) if u['unit_id']==unit_id),None)
    return rec.get('company') if rec else None

def prediction_results(sets, overlaps_old):
    """Pure implementation of Part 160.1 to 160.6 point-estimate boundaries."""
    medians={}
    for k,units in sets.items():
        vals=sorted(u['peak_dose'] for u in units if u.get('peak_dose') is not None)
        medians[k]=(vals[len(vals)//2] if len(vals)%2 else (vals[len(vals)//2-1]+vals[len(vals)//2])/2) if vals else None
    p06,p10=medians.get('0.6'),medians.get('1.0')
    rates={k:(sum(u['hits_at_pick'] for u in us)/(24*len(us)) if us else None) for k,us in sets.items()}
    random_rates={k:(sum(u['random_hits_at_pick'] for u in us)/(24*len(us)) if us else None) for k,us in sets.items()}
    overlap=len({u['unit_id'] for u in sets.get('0.6',[])} & {u['unit_id'] for u in sets.get('1.0',[])})
    vals={
      '160.1':p10 is not None and p06 is not None and p10-p06>=0.2-1e-9,
      '160.2':all(sum(u['hits_at_pick']>=12 for u in sets.get(k,[]))>=len(sets.get(k,[]))/3 for k in ('0.6','1.0')),
      '160.3':overlap<=2 and all(n<=3 for n in overlaps_old.values()),
      '160.4':all(rates.get(k) is not None and random_rates.get(k) is not None and rates[k]>=2*random_rates[k] for k in ('0.6','1.0')),
      '160.5':all(sum(u.get('peak_dose') is not None and abs(u['peak_dose']-float(k))<=0.2+1e-9 for u in sets.get(k,[]))>=len(sets.get(k,[]))/2 for k in ('0.6','1.0')),
      '160.6':all(sum(u.get('company')=='Google' for u in sets.get(k,[]))>=len(sets.get(k,[]))/2 for k in ('0.6','1.0')),
    }
    return vals,medians,overlap,rates,random_rates

def peak_dose(counts):
    """Mean of tied maxima; None if there are no hits at any dose."""
    peak=max(counts.values(),default=0)
    if peak<=0:return None
    return sum(d for d,v in counts.items() if v==peak)/sum(v==peak for v in counts.values())

def score_rows(rows,picked,part148_rows):
    items=maker_items(); by={}; duplicates=0
    for r in rows:
        k=key(r)
        if k is None: duplicates+=1; continue
        if k in by: duplicates+=1
        by[k]=r
    expected=[]
    # Gate 2's first 32 neurons plus paired random directions.
    # Atlas ids and coordinates are contained in gate2 answers and source-independent for this check.
    gate2_units=[]
    # Part 148's rows carry atlas ids (q112-000); this part names units LxNy throughout, so the map is kept here.
    atlas_id={}
    for r in part148_rows:
        if r.get('unit')=='neuron' and r.get('question_index')==0: atlas_id.setdefault(f"L{int(r['layer'])}N{int(r['neuron'])}",r.get('id'))
    planned_ids=list(atlas_id)
    gate2_units=planned_ids[:32]
    for uid0 in gate2_units:
        for arm in ('neuron','random'): expected.append(('gate2',uid0,arm,0.8,0))
    # Stage 1 uses both arms, both doses and two selection questions.
    screen_ids=planned_ids
    for u in screen_ids:
        for arm in ('neuron','random'):
            for d in PICK_DOSES:
                for qi in range(2): expected.append(('screen',u,arm,d,qi))
    ladder_units={u['unit_id'] for d in PICK_DOSES for u in picked.get(str(d),{}).get('kept',[])} | set(POSITIVE)
    for u in ladder_units:
        for d in DOSES:
            for qi in range(len(items)): expected.append(('ladder',u,'neuron',d,qi))
        for d in (0.6,0.8,1.0):
            for qi in range(len(items)): expected.append(('ladder',u,'random',d,qi))
    for qi in range(len(items)): expected.append(('ladder','plain','plain',0.0,qi))
    present={k for k in by}; missing=[k for k in expected if k not in present]
    nonempty=sum(isinstance(by[k].get('answer'),str) and bool(by[k]['answer'].strip()) for k in expected if k in by)
    gates={
      'gate1_rows':{'passed':not missing and duplicates==0 and nonempty==len(expected),'expected':len(expected),'present':len(expected)-len(missing),'missing':len(missing),'duplicates':duplicates,'nonempty':nonempty},
    }
    old={(r.get('id'),int(r.get('question_index',-1))):' '.join(str(r.get('answer','')).split()) for r in part148_rows if r.get('unit')=='neuron' and int(r.get('question_index',-1))==0}
    matches=0; compared=0
    for u in gate2_units:
        source=atlas_id[u]
        current=by.get(('gate2',u,'neuron',0.8,0),{}).get('answer','')
        if source in old: compared+=1; matches+=int(' '.join(str(current).split())==old[source])
    gates['gate2_reproduction']={'passed':compared==32 and matches>=28,'matches':matches,'compared':compared,'required':28}
    set_counts={str(d):len(picked.get(str(d),{}).get('kept',[])) for d in PICK_DOSES}
    gates['gate3_picking']={'passed':all(set_counts[str(d)]>=4 for d in PICK_DOSES),'kept_by_dose':set_counts,'minimum':4}
    def ladder(uid0,arm,d,qi): return by.get(('ladder',uid0,arm,float(d),qi),{})
    control_hits={}
    for u,co in POSITIVE.items():
        control_hits[u]=sum(company_hit(ladder(u,'neuron',0.8,qi).get('answer',''),co) for qi in range(len(items)))
    all_pick_companies={u.get('company') for d in PICK_DOSES for u in picked.get(str(d),{}).get('kept',[]) if u.get('company')}
    no_push_hits=sum(any(company_hit(ladder('plain','plain',0.0,qi).get('answer',''),c) for c in all_pick_companies) for qi in range(len(items)))
    gates['gate4_controls']={'passed':15<=control_hits['L24N5017']<=22 and 11<=control_hits['L21N7134']<=20 and no_push_hits<=2,
      'L24N5017_Google_hits_at_0_8':control_hits['L24N5017'],'required':[15,22],
      'L21N7134_Baidu_hits_at_0_8':control_hits['L21N7134'],'required_baidu':[11,20],
      'no_push_any_picked_company_hits':no_push_hits,'maximum':2}
    all_gates=all(g['passed'] for g in gates.values())
    # Per picked unit and set, score with that set's company; positive controls retain registered companies.
    sets={str(d):[] for d in PICK_DOSES}
    unit_dose={}; random_rates={}; unit_rates={}
    for d in PICK_DOSES:
        records=picked.get(str(d),{}).get('kept',[])
        for u in records:
            uid0=u['unit_id']; company=company_for(picked,d,uid0)
            h={dose:sum(company_hit(ladder(uid0,'neuron',dose,qi).get('answer',''),company) for qi in range(len(items))) for dose in DOSES}
            rh=sum(company_hit(ladder(uid0,'random',d,qi).get('answer',''),company) for qi in range(len(items)))
            pk=peak_dose(h)
            item={'unit_id':uid0,'company':company,'hits':h,'peak_dose':pk,'picked_dose':d,'random_hits_at_pick':rh}
            sets[str(d)].append(item); unit_dose[(str(d),uid0)]=item
    for d in PICK_DOSES:
        for unit in sets[str(d)]:
            dd=unit['picked_dose']; c=unit['company']; uid0=unit['unit_id']
            unit['hits_at_pick']=unit['hits'][dd]
            unit['random_hits_at_pick']=sum(company_hit(ladder(uid0,'random',dd,qi).get('answer',''),c) for qi in range(len(items)))
        random_rates[str(d)]=sum(u['random_hits_at_pick'] for u in sets[str(d)])/(len(sets[str(d)])*len(items)) if sets[str(d)] else None
        unit_rates[str(d)]=sum(u['hits_at_pick'] for u in sets[str(d)])/(len(sets[str(d)])*len(items)) if sets[str(d)] else None
    oldten=set(PART148_TEN)
    overlaps_old={k:len({u['unit_id'] for u in sets[k]}&oldten) for k in sets}
    vals,medians,overlaps,unit_rates,random_rates=prediction_results(sets,overlaps_old)
    prediction_text={
      '160.1':('The set picked at 1.0 has a median peak dose at least 0.2 above the set picked at 0.6.','medium'),
      '160.2':('In each set, at least a third of neurons hit their company on at least 12 of 24 at their picking dose.','low'),
      '160.3':("The two sets share at most 2 neurons, and each shares at most 3 with Part 148's ten.",'medium'),
      '160.4':("At its picking dose, each set's mean neuron hit rate is at least twice its paired random directions'.",'medium'),
      '160.5':('In each set, at least half of neurons peak within 0.2 of their picking dose.','low'),
      '160.6':('Google is the company of at least half of the neurons in each set.','low')}
    predictions={k:{'prediction':prediction_text[k][0],'confidence':prediction_text[k][1],
      'verdict':('held' if v else 'failed') if all_gates else 'untestable','supported':v if all_gates else None} for k,v in vals.items()}
    summary={'picked_set_sizes':{k:len(v) for k,v in sets.items()},'unit_hit_rates_at_pick':unit_rates,'random_hit_rates_at_pick':random_rates,
      'peak_dose_medians':medians,'peak_dose_counts_no_peak':{str(d):sum(u['peak_dose'] is None for u in sets[str(d)]) for d in PICK_DOSES},
      'overlap_between_sets':overlaps,'overlap_with_part148_ten':overlaps_old,'gate4_controls':control_hits,'no_push_hits':no_push_hits,
      'sets':sets,'gate2_matches':matches}
    return {'gates':gates,'all_gates_passed':all_gates,'predictions':predictions,'summary':summary}

def rescore_part148(rows):
    indexed={(r.get('id'),int(r.get('question_index',-1)),r.get('unit')):r for r in rows}
    newer=[]
    for uid0 in dict.fromkeys(r.get('id') for r in rows):
        a=indexed.get((uid0,0,'neuron'),{}).get('answer',''); b=indexed.get((uid0,1,'neuron'),{}).get('answer','')
        ca,cb=score148._top_company(a),score148._top_company(b)
        row=indexed.get((uid0,0,'neuron'))
        if row and ca and ca==cb and triage86.triage_answer(a if isinstance(a,str) else '')!='broken' and triage86.triage_answer(b if isinstance(b,str) else '')!='broken':newer.append({'id':f"L{int(row['layer'])}N{int(row['neuron'])}",'company':ca})
    return newer

def write_outputs(result,rows,picked,part148_rows,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'score.json').write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    lines=['# Part 160 score','', '## Gates','']
    for n,g in result['gates'].items():lines.append(f"- {n}: {'passed' if g['passed'] else 'failed'} ({json.dumps(g,ensure_ascii=True)})")
    lines+=['','## Registered predictions','','| Prediction | Confidence | Verdict |','|---|---|---|']
    for name,p in result['predictions'].items(): lines.append(f"| {name}: {p['prediction']} | {p['confidence']} | {p['verdict']} |")
    lines+=['','## Unit by dose hits','']
    for setname,us in result['summary']['sets'].items():
        lines += [f'### Picked at {setname}','', '| Unit | Company | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | Peak | Pick hits | Random rate |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for u in us:
            vals=[u['hits'][d] for d in DOSES]; rr=u['random_hits_at_pick']/24
            lines.append('| '+ ' | '.join([u['unit_id'],str(u['company'])]+[f'{v}/24' for v in vals]+[str(u['peak_dose']),f"{u['hits_at_pick']}/24",f'{rr:.1%}'])+' |')
        lines.append('')
    lines+=['## Set summary','','| Set | Units | Mean neuron rate at pick | Mean random rate at pick | Median peak | No peak |','|---|---:|---:|---:|---:|---:|']
    for d in PICK_DOSES:
        k=str(d); lines.append(f"| {k} | {len(result['summary']['sets'][k])} | {result['summary']['unit_hit_rates_at_pick'][k]} | {result['summary']['random_hit_rates_at_pick'][k]} | {result['summary']['peak_dose_medians'][k]} | {result['summary']['peak_dose_counts_no_peak'][k]} |")
    lines+=['','## Overlaps','','| Comparison | Count |','|---|---:|',f"| picked sets | {result['summary']['overlap_between_sets']} |"]
    lines.extend(f'| picked at {k} with Part 148 ten | {v} |' for k,v in result['summary']['overlap_with_part148_ten'].items())
    new=rescore_part148(part148_rows); old=[{'unit_id':f'L{l}N{n}','company':c} for l,n,c in UNIT_COMPANIES]
    lines+=['','## Exploratory Part 148 rescore','','| Rule | Count | Units |','|---|---:|---|',f"| Part 160 rule on stored 0.8 answers | {len(new)} | {', '.join(x['id'] for x in new)} |",f"| Part 148 ten | {len(old)} | {', '.join(x['unit_id'] for x in old)} |"]
    (out/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Deterministic eight-answer sample in each set-by-dose cell.
    rng=random.Random(160)
    ladder_rows={(x.get('unit_id'),float(x.get('dose',-9)),int(x.get('question_index',-1))):x for x in rows if x.get('stage')=='ladder' and x.get('arm')=='neuron'}
    with (out/'hand_read.jsonl').open('w',encoding='utf-8',newline='\n') as fh:
        for d in PICK_DOSES:
            for dose in DOSES:
                candidates=[]
                for unit in result['summary']['sets'][str(d)]:
                    for qi,item in enumerate(maker_items()):
                        r=ladder_rows.get((unit['unit_id'],float(dose),qi))
                        if r: candidates.append({'set_dose':d,'dose':dose,'unit_id':unit['unit_id'],'company':unit['company'],'question_index':qi,'question':item['question'],'answer':r.get('answer','')})
                rng.shuffle(candidates)
                for rec in candidates[:8]:fh.write(json.dumps(rec,ensure_ascii=True)+'\n')

def score(answers,picked_path,part148_path,out):
    rows=load_jsonl(answers); picked=json.loads(Path(picked_path).read_text(encoding='utf-8')); old=load_jsonl(part148_path)
    result=score_rows(rows,picked,old)
    rescored=rescore_part148(old)
    result['exploratory']={'part148_rescore_count':len(rescored),'part148_rescore_units':rescored,
                           'part148_registered_ten':[{ 'unit_id':f'L{l}N{n}','company':c} for l,n,c in UNIT_COMPANIES]}
    write_outputs(result,rows,picked,old,out);return result

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--answers',required=True);p.add_argument('--picked',required=True);p.add_argument('--part148',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(argv);r=score(a.answers,a.picked,a.part148,a.out);print(json.dumps({'gates':r['gates'],'predictions':r['predictions']},ensure_ascii=True))
    if not r['all_gates_passed']:raise SystemExit(3)
if __name__=='__main__':main()
