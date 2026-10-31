#!/usr/bin/env python3
"""Score Part 154 gates and registered predictions."""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from statistics import median
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
PART86=ROOT/'part86'
if str(PART86) not in sys.path: sys.path.insert(0,str(PART86))
from part153 import score153  # noqa: E402
import triage86  # noqa: E402
from part153.battery153 import DIRECT, INDIRECT, GUARDS, UNIT_COMPANIES  # noqa: E402
company_hit=score153.company_hit
guard_hit=score153.guard_hit
BROKEN=triage86.triage_answer

PUSHES=[f'L{l}N{n}' for l,n,_ in UNIT_COMPANIES]+['random_L25N4457','none']
ZERO_IDS=['none','L27N3325','matched_0','matched_1','matched_2','matched_3']
GOOGLE={f'L{l}N{n}' for l,n,c in UNIT_COMPANIES if c=='Google'}
COMPANY={f'L{l}N{n}':c for l,n,c in UNIT_COMPANIES}
MAKER_N=len(DIRECT+INDIRECT)

def gate1_passes(answer_count, answer_unique, answer_nonempty, stage_count, stage_unique, stage_finite, trace):
    return (answer_count==1872 and answer_unique==1872 and answer_nonempty==1872 and stage_count==576 and
            stage_unique==576 and stage_finite and int(trace.get('target_failures',0))==0 and
            int(trace.get('unzeroed_target_failures',0))==0 and int(trace.get('unchanged_failures',0))==0 and
            int(trace.get('target_checks',0))>0 and int(trace.get('unzeroed_target_checks',0))>0 and
            int(trace.get('unchanged_checks',0))>0)

def gate2_passes(plain_google_hits, ten_neuron_hits):
    return plain_google_hits<=2 and 128<=ten_neuron_hits<=174

def gate3_passes(lowered_phrasings):
    return lowered_phrasings>=20

def larger_negative_move(google_move, other_moves):
    return bool(other_moves) and google_move < min(other_moves)

def prediction_values(g0,g1,other0,other1,matched,random0,random1,google_count,broken_delta_pp,guard0,guard1):
    return {'154.1':g1<=.5*g0,'154.2':other1>=.75*other0,'154.3':all(v>=.75*g0 for v in matched),
            '154.4':random1<=.5*random0,'154.5':google_count>=5,
            '154.6':broken_delta_pp is not None and broken_delta_pp<=5,'154.7':abs(guard1-guard0)<=3}


def read_jsonl(path):
    out=[]
    with Path(path).open(encoding='utf-8') as f:
        for ln,line in enumerate(f,1):
            if not line.strip(): continue
            try: out.append(json.loads(line))
            except json.JSONDecodeError as e: raise ValueError(f'invalid JSON at {path}:{ln}') from e
    return out


def _index(rows,key_fields):
    ix={}; dup=0
    for row in rows:
        key=tuple(row.get(k) for k in key_fields)
        if key in ix: dup+=1
        ix[key]=row
    return ix,dup


def score_rows(answers, stagea, trace):
    maker=answers
    expected={(uid,arm,z,qi) for z in ZERO_IDS for uid in PUSHES for arm in [
        ('none' if uid=='none' else 'random' if uid.startswith('random_') else 'neuron')]
        for qi in range(MAKER_N)}
    expected.update((uid,arm,z,MAKER_N+gi) for z in ZERO_IDS[:2] for uid in PUSHES
                    for arm in [('none' if uid=='none' else 'random' if uid.startswith('random_') else 'neuron')]
                    for gi in range(len(GUARDS)))
    ans_ix,ans_dup=_index(maker,['unit_id','arm','zero_set_id','question_index'])
    stage_ix,stage_dup=_index(stagea,['push_id','arm','zero_set_id','question_index'])
    sa_expected={(uid,arm,z,qi) for z in ZERO_IDS[:2] for uid in PUSHES
                 for arm in [('none' if uid=='none' else 'random' if uid.startswith('random_') else 'neuron')]
                 for qi in range(MAKER_N)}
    nonempty=sum(isinstance(r.get('answer'),str) and bool(r['answer'].strip()) for r in ans_ix.values())
    sa_finite=all(all(isinstance(r.get(k),(int,float)) and math.isfinite(float(r[k])) for k in
        ('activation_L27N3325_pre_zero','google_minus_alibaba','name_mass')) and
        isinstance(r.get('name_logits'),dict) and len(r['name_logits'])==8 for r in stage_ix.values())
    gate1={'passed':gate1_passes(len(answers),len(ans_ix),nonempty,len(stagea),len(stage_ix),sa_finite and expected.issubset(ans_ix) and sa_expected.issubset(stage_ix) and ans_dup==0 and stage_dup==0,trace),
           'expected_answers':1872,'answers':len(answers),'unique_answers':len(ans_ix),'nonempty_answers':nonempty,
           'expected_stageA':576,'stageA_rows':len(stagea),'stageA_finite':sa_finite,'trace':trace}
    def getans(uid,arm,z,qi): return ans_ix.get((uid,arm,z,qi),{}).get('answer','')
    no_push_hits=sum(company_hit(getans('none','none','none',qi),'Google') for qi in range(MAKER_N))
    ten_hits=sum(company_hit(getans(uid,'neuron','none',qi),company) for uid,company in COMPANY.items() for qi in range(MAKER_N))
    gate2={'passed':gate2_passes(no_push_hits,ten_hits),'plain_google_hits':no_push_hits,'plain_max':2,
           'ten_neuron_company_hits':ten_hits,'ten_neuron_range':[128,174]}
    gap_none=[stage_ix.get(('none','none','none',qi),{}).get('google_minus_alibaba') for qi in range(MAKER_N)]
    gap_zero=[stage_ix.get(('none','none','L27N3325',qi),{}).get('google_minus_alibaba') for qi in range(MAKER_N)]
    lowered=sum(isinstance(a,(int,float)) and isinstance(b,(int,float)) and b<a for a,b in zip(gap_none,gap_zero))
    gate3={'passed':len(gap_none)==24 and len(gap_zero)==24 and gate3_passes(lowered),'lowered_phrasings':lowered,'minimum':20}
    gates={'gate1_rows_and_zero_trace':gate1,'gate2_baseline_and_positive_control':gate2,'gate3_opening_zero_effect':gate3}
    all_passed=all(g['passed'] for g in gates.values())
    # Company hit table indexed by unit and zero set, over maker questions.
    per_unit={}
    for uid,company in COMPANY.items():
        per_unit[uid]={'company':company,'zero_sets':{}}
        for z in ZERO_IDS:
            hits=sum(company_hit(getans(uid,'neuron',z,qi),company) for qi in range(MAKER_N))
            per_unit[uid]['zero_sets'][z]={'hits':hits,'n':MAKER_N}
    per_zero={}
    for z in ZERO_IDS:
        per_zero[z]={}
        for group,ids in [('google',GOOGLE),('other_makers',set(COMPANY)-GOOGLE)]:
            per_zero[z][group]={'hits':sum(company_hit(getans(uid,'neuron',z,qi),COMPANY[uid]) for uid in ids for qi in range(MAKER_N)),
                                'n':len(ids)*MAKER_N}
        per_zero[z]['random_direction']={'hits':sum(company_hit(getans('random_L25N4457','random',z,qi),'Google') for qi in range(MAKER_N)),'n':MAKER_N}
    g0=per_zero['none']['google']['hits']; g1=per_zero['L27N3325']['google']['hits']
    other0=per_zero['none']['other_makers']['hits']; other1=per_zero['L27N3325']['other_makers']['hits']
    random0=per_zero['none']['random_direction']['hits'];random1=per_zero['L27N3325']['random_direction']['hits']
    matched_hold={f'154.{i+3}':per_zero[f'matched_{i}']['google']['hits'] >= .75*g0 for i in range(4)}
    # Design says each of four matched sets; registered prediction 154.3 requires all four.
    pred_values={}
    # Activation change per push relative to no-push, same phrasing, at no-zero condition.
    med_moves={}
    for uid in list(GOOGLE)+[u for u in COMPANY if u not in GOOGLE]:
        ds=[]
        for qi in range(MAKER_N):
            pushed=stage_ix.get((uid,'neuron','none',qi),{}).get('activation_L27N3325_pre_zero')
            base=stage_ix.get(('none','none','none',qi),{}).get('activation_L27N3325_pre_zero')
            if isinstance(pushed,(int,float)) and isinstance(base,(int,float)): ds.append(float(pushed)-float(base))
        med_moves[uid]=median(ds) if len(ds)==MAKER_N else None
    non_google=[med_moves[u] for u,c in COMPANY.items() if c!='Google' and med_moves[u] is not None]
    max_other_move=min(non_google) if non_google else None # more negative is larger move
    google_count=sum(v is not None and larger_negative_move(v, non_google) for u,v in med_moves.items() if u in GOOGLE)
    pred_values['154.5']=google_count>=5
    broken_rows=[r for r in answers if r.get('kind')=='maker' and r.get('zero_set_id') in ('none','L27N3325') and r.get('arm') in ('neuron','random')]
    broken_by_z={z:sum(BROKEN(r.get('answer',''))=='broken' for r in broken_rows if r.get('zero_set_id')==z) for z in ('none','L27N3325')}
    broken_share={z:broken_by_z[z]/sum(r.get('zero_set_id')==z for r in broken_rows) if any(r.get('zero_set_id')==z for r in broken_rows) else None for z in broken_by_z}
    broken_delta_pp=None if None in broken_share.values() else 100*(broken_share['L27N3325']-broken_share['none'])
    pred_values['154.6']=broken_delta_pp is not None and broken_delta_pp<=5
    guard_n=0; guard_base=guard_zero=0
    for gi,(q,expected_word) in enumerate(GUARDS):
        qi=MAKER_N+gi
        for uid in PUSHES:
            arm='none' if uid=='none' else 'random' if uid.startswith('random_') else 'neuron'
            if arm=='none' or arm=='neuron' or arm=='random':
                a=getans(uid,arm,'none',qi);b=getans(uid,arm,'L27N3325',qi)
                guard_base+=guard_hit(a,expected_word);guard_zero+=guard_hit(b,expected_word);guard_n+=1
    pred_values['154.7']=abs(guard_zero-guard_base)<=3
    pred_values=prediction_values(g0,g1,other0,other1,[per_zero[f'matched_{i}']['google']['hits'] for i in range(4)],random0,random1,google_count,broken_delta_pp,guard_base,guard_zero)
    predictions={k:('untestable' if not all_passed else 'held' if v else 'failed') for k,v in pred_values.items()}
    stagea_medians={}
    for z in ZERO_IDS[:2]:
        vals=[r['google_minus_alibaba'] for r in stagea if r.get('zero_set_id')==z and r.get('push_id')=='none' and isinstance(r.get('google_minus_alibaba'),(int,float))]
        stagea_medians[z]=median(vals) if vals else None
    activation_medians={}
    for z in ZERO_IDS[:2]:
        activation_medians[z]={}
        for uid in PUSHES:
            vals=[r['activation_L27N3325_pre_zero'] for r in stagea if r.get('zero_set_id')==z and r.get('push_id')==uid and isinstance(r.get('activation_L27N3325_pre_zero'),(int,float))]
            activation_medians[z][uid]=median(vals) if vals else None
    return {'gates':gates,'all_gates_passed':all_passed,'predictions':predictions,
      'per_unit':per_unit,'per_zero_set':per_zero,'stageA':{'median_google_minus_alibaba_no_push':stagea_medians,
      'median_activation_by_zero_and_push':activation_medians,'median_activation_move_by_unit':med_moves,'non_google_most_negative_move':max_other_move,
      'google_units_exceeding_move':google_count},'broken':{'counts':broken_by_z,'shares':broken_share,'delta_percentage_points':broken_delta_pp},
      'guards':{'no_zero_hits':guard_base,'L27_zero_hits':guard_zero,'n_per_arm':guard_n},'row_counts':{'answers':len(answers),'stageA':len(stagea)}}


def write_tables(result,path):
    lines=['# Part 154 score','','## Per unit maker company hits','', '| Unit | Company | None | L27 N3325 | Matched 0 | Matched 1 | Matched 2 | Matched 3 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for uid,u in result['per_unit'].items():
        vals=[u['zero_sets'][z]['hits'] for z in ZERO_IDS]
        lines.append(f"| {uid} | {u['company']} | "+' | '.join(f'{v}/24' for v in vals)+' |')
    lines += ['', '## Per zero set','', '| Zero set | Google hits | Other maker hits | Random direction Google hits |','|---|---:|---:|---:|']
    for z,v in result['per_zero_set'].items(): lines.append(f"| {z} | {v['google']['hits']}/{v['google']['n']} | {v['other_makers']['hits']}/{v['other_makers']['n']} | {v['random_direction']['hits']}/24 |")
    lines += ['', '## Gates','']
    for k,v in result['gates'].items(): lines.append(f"- {k}: {'passed' if v['passed'] else 'failed'} ({json.dumps(v,ensure_ascii=False)})")
    lines += ['', '## Predictions','']+[f"- {k}: {v}" for k,v in result['predictions'].items()]
    lines += ['', '## Broken answers','',json.dumps(result['broken'],ensure_ascii=False,indent=2),'','## Stage A medians','',json.dumps(result['stageA'],ensure_ascii=False,indent=2),'']
    Path(path).write_text('\n'.join(lines),encoding='utf-8')


def score(answers,stagea,trace,out):
    result=score_rows(read_jsonl(answers),read_jsonl(stagea),json.loads(Path(trace).read_text(encoding='utf-8')).get('stageB',{}))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'score.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_tables(result,out/'tables.md')
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--answers',required=True);ap.add_argument('--stagea',required=True);ap.add_argument('--trace',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args(argv);result=score(args.answers,args.stagea,args.trace,args.out)
    print(json.dumps({'gates':result['gates'],'predictions':result['predictions'],'stageA':result['stageA']},ensure_ascii=False),flush=True)
    if not result['all_gates_passed']: raise SystemExit(3)
if __name__=='__main__':main()
