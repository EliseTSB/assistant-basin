#!/usr/bin/env python3
"""Score Part 82 records with paired body bootstrap intervals and registered rules."""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path

CONDITIONS=('h0','h1','h2','c1','c2','c3','c4','c5')
OPTION_SETS=('B','N')

def load(path):
    return [json.loads(line) for line in (Path(path)/'records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]

def _matrix(recs, condition, option_set):
    cells={}
    for r in recs:
        if r['condition']==condition and r['option_set']==option_set:
            cells[(str(r['body_id']),r['rendering'],int(r['order']))]=r
    bodies=sorted({k[0] for k in cells})
    full=[b for b in bodies if all((b,rn,o) in cells for rn in ('A','U') for o in (1,2,3))]
    return cells,full

def paired_bootstrap(diffs,n=10000,seed=82):
    """Percentile CI over resampled body-level differences; each pair stays intact."""
    if not diffs: return [None,None]
    rng=random.Random(seed); m=len(diffs)
    means=sorted(sum(diffs[rng.randrange(m)] for _ in range(m))/m for _ in range(n))
    return [float(means[int(.025*n)]),float(means[int(.975*n)-1])]

def summarize(recs,condition,option_set):
    cells,bodies=_matrix(recs,condition,option_set)
    result={'bodies':len(bodies),'orders':{},'cyclic':{},'mean_letter_mass':None}
    if not bodies: return result
    masses=[r['letter_mass'] for r in recs if r['condition']==condition and r['option_set']==option_set]
    result['mean_letter_mass']=float(sum(masses)/len(masses)) if masses else None
    for order in (1,2,3):
        diffs=[cells[(b,'A',order)]['p_you']-cells[(b,'U',order)]['p_you'] for b in bodies]
        result['orders'][str(order)]={'gap':float(sum(diffs)/len(diffs)),'gap_ci95':paired_bootstrap(diffs)}
    diffs=[sum(cells[(b,'A',o)]['p_you']-cells[(b,'U',o)]['p_you'] for o in (1,2,3))/3 for b in bodies]
    result['cyclic']={'gap':float(sum(diffs)/len(diffs)),'gap_ci95':paired_bootstrap(diffs)}
    return result

def drop_summary(recs,option_set,condition):
    base,bs=_matrix(recs,'h0',option_set); target,ts=_matrix(recs,condition,option_set)
    paired=sorted(set(bs)&set(ts))
    vals=[sum(base[(b,'A',o)]['p_you']-base[(b,'U',o)]['p_you'] for o in (1,2,3))/3-
          sum(target[(b,'A',o)]['p_you']-target[(b,'U',o)]['p_you'] for o in (1,2,3))/3 for b in paired]
    return {'drop':float(sum(vals)/len(vals)) if vals else None,'ci95':paired_bootstrap(vals), 'bodies':len(vals)}

def score(recs):
    summaries={c:{s:summarize(recs,c,s) for s in OPTION_SETS} for c in CONDITIONS}
    drops={s:{c:drop_summary(recs,s,c) for c in ('h1','h2','c1','c2','c3','c4','c5')} for s in OPTION_SETS}
    h0b=summaries['h0']['B']['cyclic']['gap']; h0n=summaries['h0']['N']['cyclic']['gap']
    gate=(h0b is not None and h0n is not None and abs(h0b-.444)<=.05 and abs(h0n-.205)<=.05)
    d1=drops['B']['h1']; controls=[drops['B'][f'c{i}']['drop'] for i in range(1,6)]
    p={}
    p['82.1']={'verdict':'HOLDS' if gate else 'FAILS','details':f'h0 B={h0b}, N={h0n}; target B=.444, N=.205, tolerance=.05'}
    p['82.2']={'verdict':'NOT READ' if not gate else ('HOLDS' if d1['drop'] is not None and d1['drop']>=.10 and d1['ci95'][0]>0 else 'FAILS'),
               'details':f"h1 B drop={d1['drop']} CI={d1['ci95']}"}
    p['82.3']={'verdict':'NOT READ' if not gate else ('HOLDS' if d1['drop'] is not None and all(x is not None and d1['drop']>x for x in controls) else 'FAILS'),
               'details':f"h1 B drop={d1['drop']}; control drops={controls}"}
    d2=drops['B']['h2']['drop']
    p['82.4']={'verdict':'NOT READ' if not gate else ('HOLDS' if d2 is not None and abs(d2)<.05 else 'FAILS'), 'details':f'h2 B drop={d2}; threshold |drop|<.05'}
    d1n=drops['N']['h1']['drop']
    p['82.5']={'verdict':'NOT READ' if not gate else ('HOLDS' if d1n is not None and d1n>=.05 else 'FAILS'), 'details':f'h1 N drop={d1n}; threshold .05'}
    low=[]
    for c in CONDITIONS:
        for s in OPTION_SETS:
            masses=[r['letter_mass'] for r in recs if r['condition']==c and r['option_set']==s]
            mean=sum(masses)/len(masses) if masses else None
            if mean is None or mean<.9: low.append({'condition':c,'option_set':s,'mean':mean})
    p['82.6']={'verdict':'HOLDS' if not low else 'FAILS','details':'all condition/option-set mean letter masses >= .9' if not low else f'violations: {low}'}
    if not gate:
        for k in ('82.2','82.3','82.4','82.5'): p[k]['details']+='; 82.1 failed, so prediction not read'
    return {'summaries':summaries,'drops':drops,'predictions':p,'reading_gate_passed':gate}

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--dir',required=True); a=ap.parse_args(argv)
    d=Path(a.dir); result=score(load(d))
    with (d/'results82.json').open('w',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=1)
    print('condition option order gap')
    for c in CONDITIONS:
        for s in OPTION_SETS:
            sm=result['summaries'][c][s]
            for o,row in sm['orders'].items(): print(f'{c:>2} {s} O{o} {row["gap"]:.4f}')
            print(f'{c:>2} {s} cyclic {sm["cyclic"].get("gap")} mass {sm["mean_letter_mass"]}')
    print('prediction verdict details')
    for k,v in result['predictions'].items(): print(k,v['verdict'],v['details'])
if __name__=='__main__': main()
