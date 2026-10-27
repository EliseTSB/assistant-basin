#!/usr/bin/env python3
"""Part 84 blinding: `prepare` shuffles answers under random ids (key kept apart); `score` joins labels to the key."""
from __future__ import annotations
import argparse, json, random, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
ARMS=('A0','A1','A2','A3','C')
LABEL1=('EARTH','OFFWORLD','NEITHER'); LABEL2=('OK','BROKEN')

def prepare(records, items_path, key_path, seed=84):
    rng=random.Random(seed); recs=list(records); rng.shuffle(recs)
    ids=[f'x{i:03d}' for i in range(len(recs))]
    Path(items_path).write_text(''.join(json.dumps({'id':i,'text':r['answer']},ensure_ascii=False)+'\n' for i,r in zip(ids,recs)),encoding='utf-8')
    key={i:{'arm':r['arm'],'question_index':r['question_index'],'sample':r['sample']} for i,r in zip(ids,recs)}
    Path(key_path).write_text(json.dumps(key,ensure_ascii=False,indent=0),encoding='utf-8')
    return key

def score(labels, key):
    """labels: rows {id, origin, form}. Every key id must be labelled exactly once with valid labels."""
    seen={}
    for r in labels:
        if r['id'] in seen: raise ValueError(f"duplicate label for {r['id']}")
        if r['origin'] not in LABEL1 or r['form'] not in LABEL2: raise ValueError(f"bad label {r}")
        seen[r['id']]=r
    if set(seen)!=set(key): raise ValueError(f'labels cover {len(seen)} ids, key has {len(key)}')
    out={a:{'n':0,**{l:0 for l in LABEL1},'BROKEN':0} for a in ARMS}
    for i,r in seen.items():
        a=key[i]['arm']; out[a]['n']+=1; out[a][r['origin']]+=1; out[a]['BROKEN']+=r['form']=='BROKEN'
    return out

def verdicts(counts, median):
    v={}
    v['84.1']=('HOLDS' if abs(median-24.8)/24.8<=.02 else 'FAILS', f'median {median:.3f} vs 24.8')
    gate=v['84.1'][0]=='HOLDS' and counts['A0']['OFFWORLD']<=1
    v['84.2']=('HOLDS' if counts['A0']['OFFWORLD']<=1 else 'FAILS', f"A0 OFFWORLD {counts['A0']['OFFWORLD']} of {counts['A0']['n']}")
    def g(x): return x if gate else 'NOT READ'
    v['84.3']=(g('HOLDS' if counts['A2']['OFFWORLD']>=12 else 'FAILS'), f"A2 OFFWORLD {counts['A2']['OFFWORLD']}")
    v['84.4']=(g('HOLDS' if counts['A3']['OFFWORLD']>=18 else 'FAILS'), f"A3 OFFWORLD {counts['A3']['OFFWORLD']}")
    v['84.5']=(g('HOLDS' if counts['C']['OFFWORLD']<=2 else 'FAILS'), f"C OFFWORLD {counts['C']['OFFWORLD']}")
    v['84.6']=(g('HOLDS' if 6<=counts['A1']['OFFWORLD']<=20 else 'FAILS'), f"A1 OFFWORLD {counts['A1']['OFFWORLD']}")
    v['84.7']=('HOLDS' if counts['A2']['BROKEN']<=4 else 'FAILS', f"A2 BROKEN {counts['A2']['BROKEN']}")
    return v

def main(argv=None):
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('prepare'); p.add_argument('--dir',required=True); p.add_argument('--items',required=True); p.add_argument('--key',required=True)
    s=sp.add_parser('score'); s.add_argument('--dir',required=True); s.add_argument('--labels',required=True); s.add_argument('--key',required=True)
    a=ap.parse_args(argv)
    if a.cmd=='prepare':
        recs=[json.loads(l) for l in (Path(a.dir)/'records.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        k=prepare(recs,a.items,a.key); print(f'{len(k)} items written')
    else:
        key=json.loads(Path(a.key).read_text(encoding='utf-8'))
        labels=[json.loads(l) for l in Path(a.labels).read_text(encoding='utf-8').splitlines() if l.strip()]
        counts=score(labels,key); med=json.loads((Path(a.dir)/'run.json').read_text(encoding='utf-8'))['median']
        v=verdicts(counts,med)
        (Path(a.dir)/'results84.json').write_text(json.dumps({'counts':counts,'verdicts':v,'median':med},indent=1),encoding='utf-8')
        for arm,c in counts.items(): print(arm,c)
        for k2,(vv,d) in v.items(): print(k2,vv,d)
if __name__=='__main__': main()
