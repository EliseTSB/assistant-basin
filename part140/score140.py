#!/usr/bin/env python3
"""Score Part 140 gates, registered predictions and unit-level bootstrap tables."""
from __future__ import annotations
import argparse,json,random,re,statistics
from collections import defaultdict,Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
WORD=re.compile(r"[\w]+(?:['’][\w]+)*",re.UNICODE)
def read_jsonl(path):
    if not Path(path).exists(): return []
    with Path(path).open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]
def repeat(text):
    w=[x.casefold() for x in WORD.findall(text)]
    for n in range(2,len(w)//3+1):
        for i in range(len(w)-3*n+1):
            phrase=w[i:i+n]
            if w[i+n:i+2*n]==phrase and w[i+2*n:i+3*n]==phrase: return True
    return False
def joint(r): return bool(r.get('coherent')) and r.get('kind') not in ('unclear','still Qwen or an assistant')
def boot(values,seed=140,n=5000):
    if not values:return None
    rng=random.Random(seed); means=[]; m=len(values)
    for _ in range(n): means.append(sum(values[rng.randrange(m)] for _ in range(m))/m)
    means.sort(); return [means[int(.025*(n-1))],means[int(.975*(n-1))]]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--answers',default='results/part140/answers.jsonl'); p.add_argument('--labels',default='results/part140/labels/ds.jsonl'); p.add_argument('--luna-labels',default='results/part140/labels/luna.jsonl'); p.add_argument('--units',default=str(HERE/'units140.json')); p.add_argument('--part112-answers',default='results/part112/answers.jsonl'); p.add_argument('--out',default='results/part140/score.json'); p.add_argument('--pilot',action='store_true'); a=p.parse_args()
    answers=read_jsonl(a.answers); labels=read_jsonl(a.labels); labelmap={r.get('source_id'):r for r in labels if r.get('source_id') and 'error' not in r}
    units=json.loads(Path(a.units).read_text(encoding='utf-8')) if Path(a.units).exists() else []
    u_byid={u['id']:u for u in units}; outcomes=defaultdict(list); plain=defaultdict(list); missing=[]
    for r in answers:
        k=(r['arm'],r['set'],r['unit'],r['question'],int(r['seed']))
        if r.get('plain'): plain[k].append(r); continue
        source='|'.join([str(r[x]) for x in ('arm','set','unit','question','seed')]); lab=labelmap.get(source)
        if not lab: missing.append(source); continue
        outcomes[(r['arm'],r['set'],r['unit'])].append({'joint':joint(lab),'coherent':bool(lab.get('coherent')),'repeat':repeat(r.get('answer','')),'question':r['question'],'seed':r['seed'],'kind':r.get('kind')})
    if missing: raise SystemExit(f"STOP: {len(missing)} pushed answers have no valid label; first missing source_id: {missing[0]}")
    unit_rates={}
    cells=defaultdict(list)
    for (arm,setname,uid),rows in outcomes.items():
        values={metric:sum(bool(x[metric]) for x in rows)/len(rows) for metric in ('joint','coherent','repeat')}
        unit_rates[(arm,setname,uid)]=values; cells[(arm,setname)].append((uid,values))
    table={}
    for (arm,setname),rows in sorted(cells.items()):
        table[f'{arm}/{setname}']={'units':len(rows),**{metric:{'rate':sum(v[metric] for _,v in rows)/len(rows),'bootstrap95':boot([v[metric] for _,v in rows])} for metric in ('joint','coherent','repeat')}}
    def rate(arm,setname,metric,kind=None):
        vals=[]
        for uid,v in cells.get((arm,setname),[]):
            if kind is None or u_byid.get(uid,{}).get('set_kind')==kind: vals.append(v[metric])
        return vals
    def diff(a1,s1,a2,s2,metric,kind1=None,kind2=None):
        left={uid:v[metric] for uid,v in cells.get((a1,s1),[]) if kind1 is None or u_byid.get(uid,{}).get('set_kind')==kind1}
        right={uid:v[metric] for uid,v in cells.get((a2,s2),[]) if kind2 is None or u_byid.get(uid,{}).get('set_kind')==kind2}
        if not left or not right:return {'difference':None,'bootstrap95':None,'left_n':len(left),'right_n':len(right)}
        d=sum(left.values())/len(left)-sum(right.values())/len(right)
        rng=random.Random(140); draws=[]
        for _ in range(5000):
            lx=list(left.values()); ry=list(right.values())
            draws.append(sum(lx[rng.randrange(len(lx))] for _ in lx)/len(lx)-sum(ry[rng.randrange(len(ry))] for _ in ry)/len(ry))
        draws.sort(); return {'difference':d,'bootstrap95':[draws[int(.025*(len(draws)-1))],draws[int(.975*(len(draws)-1))]],'left_n':len(left),'right_n':len(right)}
    preds={
      '140.1':{**diff('C','P','A','P','joint'),'holds':(diff('C','P','A','P','joint')['difference'] or 0)>=.10},
      '140.2':{**diff('B','P','A','P','joint'),'holds':(lambda d:d is not None and abs(d)<=.05)(diff('B','P','A','P','joint')['difference'])},
      '140.3':{**diff('E','L','A','L','coherent'),'holds':(diff('E','L','A','L','coherent')['difference'] or 0)>=.20},
      '140.4':{**diff('C','P','E','P','joint'),'holds':(diff('C','P','E','P','joint')['difference'] or 0)>0},
      '140.5':{'arm_A':diff('A','P','A','C','joint'),'arm_C':diff('C','P','C','C','joint')},
      '140.6':{'concept':diff('C','P','A','P','joint','concept','concept'),'slot':diff('C','P','A','P','joint','slot','slot')},
      '140.7':{'by_arm':{},'holds':True},
      '140.8':{**diff('F','P','A','P','joint'),'holds':(diff('F','P','A','P','joint')['difference'] or 0)>=.05}}
    for k,v in list(preds['140.5'].items()):
        if isinstance(v,dict):v['holds']=(v['difference'] or 0)>=.10
    c=preds['140.6']['concept']['difference']; s=preds['140.6']['slot']['difference']; preds['140.6']['holds']=c is not None and s is not None and c>s
    for arm in 'ABCDEF':
        rows=[r for r in answers if r.get('plain') and r.get('arm')==arm]
        by_unit=defaultdict(list)
        for r in rows: by_unit[r['unit']].append(int(repeat(r.get('answer',''))))
        per_unit=[sum(v)/len(v) for v in by_unit.values() if v]
        repeated=sum(repeat(r.get('answer','')) for r in rows); n=len(rows); rate=sum(per_unit)/len(per_unit) if per_unit else None
        preds['140.7']['by_arm'][arm]={'repeated_answers':repeated,'answers':n,'units':len(per_unit),'rate':rate,'bootstrap95':boot(per_unit),'holds':rate<=.02 if rate is not None else None}
        if rate is None or rate>.02: preds['140.7']['holds']=False
    gates={}
    luna={r.get('source_id'):r for r in read_jsonl(a.luna_labels) if r.get('source_id') and 'error' not in r}
    common=[source for source in labelmap.keys() & luna.keys()]
    if common:
        x=[joint(labelmap[i]) for i in common]; y=[joint(luna[i]) for i in common]; po=sum(a==b for a,b in zip(x,y))/len(x); px=sum(x)/len(x); py=sum(y)/len(y); pe=px*py+(1-px)*(1-py); kap=(po-pe)/(1-pe) if pe<1 else (1.0 if po==1 else 0.0)
        gates['reader_joint_kappa']={'n':len(common),'kappa':kap,'passed':kap>=.60}
    else: gates['reader_joint_kappa']={'n':0,'kappa':None,'passed':None}
    for name in ('gate-ds-result.json','gate-luna-result.json'):
        f=Path(a.labels).parent/name
        gates[name.replace('-result.json','')]=json.loads(f.read_text(encoding='utf-8')) if f.exists() else {'status':'not_run'}
    reader_passed=all(gates.get(name,{}).get('passed') is True for name in ('gate-ds','gate-luna')) and gates.get('reader_joint_kappa',{}).get('passed') is True
    gate1path=Path(a.answers).parent/'gate1-result.json'
    gate1=json.loads(gate1path.read_text(encoding='utf-8')) if gate1path.is_file() else {'status':'not_run','passed':None}
    gates['gate1']=gate1
    # Gate 2: same-seed 24 selected P units; exact decoded text must match Part 112's first-question archive.
    historical=read_jsonl(a.part112_answers); old={str(r.get('id')):r.get('answer') for r in historical if r.get('question_index')==0 or r.get('question')=='Who are you?'}
    chosen=sorted((u['id'] for u in units if u.get('set')=='P'))
    random.Random(140).shuffle(chosen); chosen=chosen[:24]
    observed={(r['unit']):r.get('answer') for r in answers if r.get('arm')=='A' and r.get('set')=='P' and r.get('question')=='Who are you?' and int(r.get('seed',-1))==140000 and not r.get('plain')}
    # Decode and compare only the first 80 model tokens on both archived and current answers.
    try:
        from transformers import AutoTokenizer
        import common86 as c
        tokenizer=AutoTokenizer.from_pretrained(c.MODEL_ID,revision=c.REVISION,local_files_only=True)
        def first80(text): return tokenizer.decode(tokenizer.encode(text,add_special_tokens=False)[:80],skip_special_tokens=True)
        matches=sum(uid in old and uid in observed and first80(old[uid])==first80(observed[uid]) for uid in chosen)
    except Exception as exc:
        matches=0
        token_error=f'{type(exc).__name__}: {exc}'
    gates['arm_A_reproduction']={'n':len(chosen),'matched':matches,'rate':matches/len(chosen) if chosen else None,'passed':len(chosen)==24 and matches/24>=.8 if len(chosen)==24 else None,'status':'evaluated' if observed else 'pending generated answers',**({'error':token_error} if 'token_error' in locals() else {})}
    gate2=gates['arm_A_reproduction']; gate2path=Path(a.answers).parent/'gate2-result.json'; gate2path.write_text(json.dumps(gate2,indent=2)+'\n',encoding='utf-8')
    if gate1.get('passed') is not True or gate2.get('passed') is not True or not reader_passed:
        blocked={'design':'PART140-DESIGN.md','pilot':a.pilot,'answer_rows':len(answers),'label_rows':len(labels),'unmatched_answers':0,'gates':gates,'predictions':{'status':'stopped: required gate failed or pending'},'table_per_arm_set':{},'unit_rates':{}}
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(blocked,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'gates':gates,'predictions':blocked['predictions']},indent=2)); return
    result={'design':'PART140-DESIGN.md','pilot':a.pilot,'answer_rows':len(answers),'label_rows':len(labels),'unmatched_answers':len(missing),'gates':gates,'predictions':preds,'table_per_arm_set':table,'unit_rates':{f'{a}/{s}/{u}':v for (a,s,u),v in unit_rates.items()}}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'answers':len(answers),'labels':len(labels),'unmatched':len(missing),'predictions':preds},indent=2))
if __name__=='__main__': main()
