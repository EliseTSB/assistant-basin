#!/usr/bin/env python3
"""Score Part 83: single-head drops, Part 82 additivity and the registered rules 83.1 to 83.8."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part82.score82 import paired_bootstrap
from part83.part83 import conditions

SEED=83
P82_SETS={'h1':{11:[14],15:[1,2,3,12,15]},'h2':{15:[10]}}
LYRA=['L11h14','L15h1','L15h2','L15h3','L15h12','L15h15']

def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]

def body_values(recs, condition, option_set=None):
    """Per body: (cyclic gap, mean P(you|A), mean P(you|U)) over complete bodies only."""
    cells={}
    for r in recs:
        if r['condition']==condition and (option_set is None or r.get('option_set')==option_set):
            cells[(str(r['body_id']),r['rendering'],int(r['order']))]=r['p_you']
    out={}
    for b in {k[0] for k in cells}:
        if all((b,rn,o) in cells for rn in ('A','U') for o in (1,2,3)):
            a=sum(cells[(b,'A',o)] for o in (1,2,3))/3; u=sum(cells[(b,'U',o)] for o in (1,2,3))/3
            out[b]=(a-u,a,u)
    return out

def mean(xs): return float(sum(xs)/len(xs)) if xs else None

def ci(xs): return paired_bootstrap(xs,seed=SEED)

def p82_set_heads(run82):
    sets=dict(P82_SETS)
    for c,heads in run82['conditions'].items():
        if c.startswith('c'): sets[c]={int(k):v for k,v in heads.items()}
    return sets

def score(recs, recs82=None, run82=None):
    base=body_values(recs,'h0')
    conds=[c for c in conditions() if c!='h0']
    rows={}
    for c in conds:
        v=body_values(recs,c); bodies=sorted(set(base)&set(v))
        drops=[base[b][0]-v[b][0] for b in bodies]
        dA=[v[b][1]-base[b][1] for b in bodies]; dU=[v[b][2]-base[b][2] for b in bodies]
        masses=[r['letter_mass'] for r in recs if r['condition']==c]
        rows[c]={'bodies':len(bodies),'gap':mean([v[b][0] for b in bodies]),'p_you_A':mean([v[b][1] for b in bodies]),
                 'p_you_U':mean([v[b][2] for b in bodies]),'drop':mean(drops),'drop_ci95':ci(drops),
                 'delta_p_you_A':mean(dA),'delta_p_you_U':mean(dU),'delta_p_you_U_ci95':ci(dU),
                 'mean_letter_mass':mean(masses)}
    h0={'bodies':len(base),'gap':mean([x[0] for x in base.values()]),'p_you_A':mean([x[1] for x in base.values()]),
        'p_you_U':mean([x[2] for x in base.values()]),
        'mean_letter_mass':mean([r['letter_mass'] for r in recs if r['condition']=='h0'])}
    res={'h0':h0,'singles':rows,'additivity':{},'gate':{}}
    # 83.1 gate: against Part 82's h0 (Part 80b options) and Part 82's h2 drop.
    gate=False; detail='Part 82 records not supplied'
    if recs82 is not None:
        b82=body_values(recs82,'h0','B'); h2=body_values(recs82,'h2','B')
        g82=mean([x[0] for x in b82.values()])
        d82=mean([b82[b][0]-h2[b][0] for b in sorted(set(b82)&set(h2))])
        shared=sorted(set(b82)&set(base))
        per_body_max=max(abs(b82[b][0]-base[b][0]) for b in shared) if shared else None
        d10=rows['L15h10']['drop']
        gate=(h0['gap'] is not None and abs(h0['gap']-g82)<=.01 and d10 is not None and abs(d10-d82)<=.02)
        detail=f"h0 gap {h0['gap']:.4f} vs Part 82 {g82:.4f}; L15h10 drop {d10:.4f} vs Part 82 h2 {d82:.4f}; largest per-body h0 difference {per_body_max}"
        res['gate']={'part82_h0_gap':g82,'part82_h2_drop':d82,'per_body_max_abs_diff':per_body_max,'shared_bodies':len(shared)}
        # Additivity: set drop (Part 82) minus the sum of its heads' single drops (Part 83), paired over bodies.
        for sname,heads in p82_set_heads(run82 or {'conditions':{}}).items():
            names=[f'L{L}h{h}' for L,hs in heads.items() for h in hs]
            sv=body_values(recs82,sname,'B')
            singles={n:body_values(recs,n) for n in names}
            bodies=sorted(set(b82)&set(sv)&set(base).intersection(*[set(s) for s in singles.values()]))
            diffs=[(b82[b][0]-sv[b][0])-sum(base[b][0]-singles[n][b][0] for n in names) for b in bodies]
            res['additivity'][sname]={'heads':names,'bodies':len(bodies),
                'set_drop':mean([b82[b][0]-sv[b][0] for b in bodies]),
                'sum_single_drops':mean([sum(base[b][0]-singles[n][b][0] for n in names) for b in bodies]),
                'set_minus_sum':mean(diffs),'ci95':ci(diffs)}
    res['gate']['passed']=gate
    p={'83.1':{'verdict':'HOLDS' if gate else 'FAILS','details':detail}}
    ranked=sorted(conds,key=lambda c:rows[c]['drop'],reverse=True)
    def r(v,cond): return v if gate else 'NOT READ'
    h6=rows['L15h6']
    p['83.2']={'verdict':r('HOLDS' if ranked[0]=='L15h6' else 'FAILS',0),'details':f"largest drops: "+', '.join(f"{c} {rows[c]['drop']:.3f}" for c in ranked[:5])}
    p['83.3']={'verdict':r('HOLDS' if h6['drop']>=.10 and h6['drop_ci95'][0]>0 else 'FAILS',0),'details':f"L15h6 drop {h6['drop']:.4f} CI {h6['drop_ci95']}"}
    h0r=rows['L15h0']
    p['83.4']={'verdict':r('HOLDS' if h0r['delta_p_you_U']>=.05 else 'FAILS',0),'details':f"L15h0 delta P(you|U) {h0r['delta_p_you_U']:.4f} CI {h0r['delta_p_you_U_ci95']}"}
    big=[c for c in LYRA if rows[c]['drop']>.10]
    p['83.5']={'verdict':r('HOLDS' if not big else 'FAILS',0),'details':'Lyra heads: '+', '.join(f"{c} {rows[c]['drop']:.3f}" for c in LYRA)}
    matter=[c for c in conds if abs(rows[c]['drop'])>=.05 and (rows[c]['drop_ci95'][0]>0 or rows[c]['drop_ci95'][1]<0)]
    p['83.6']={'verdict':r('HOLDS' if 3<=len(matter)<=10 else 'FAILS',0),'details':f"{len(matter)} heads matter: {matter}"}
    a=res['additivity'].get('h1')
    p['83.7']={'verdict':r('HOLDS' if a and a['set_minus_sum']>0 and a['ci95'][0]>0 else 'FAILS',0) if a else 'NOT READ',
               'details':f"h1 set drop {a['set_drop']:.4f}, sum of singles {a['sum_single_drops']:.4f}, difference {a['set_minus_sum']:.4f} CI {a['ci95']}" if a else 'no Part 82 data'}
    low=[c for c in ['h0']+conds if (res['h0'] if c=='h0' else rows[c])['mean_letter_mass']<.9]
    p['83.8']={'verdict':'HOLDS' if not low else 'FAILS','details':f"min mean letter mass {min([h0['mean_letter_mass']]+[rows[c]['mean_letter_mass'] for c in conds]):.4f}; below .9: {low}"}
    if not gate:
        for k in ('83.2','83.3','83.4','83.5','83.6','83.7'): p[k]['details']+='; 83.1 failed, so not read'
    res['predictions']=p; res['ranked']=ranked
    return res

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--dir',required=True)
    ap.add_argument('--part82',default=str(ROOT/'results/part82/main')); a=ap.parse_args(argv)
    d=Path(a.dir); p82=Path(a.part82)
    recs82=load(p82/'records.jsonl') if (p82/'records.jsonl').exists() else None
    run82=json.loads((p82/'run.json').read_text(encoding='utf-8')) if (p82/'run.json').exists() else None
    res=score(load(d/'records.jsonl'),recs82,run82)
    (d/'results83.json').write_text(json.dumps(res,ensure_ascii=False,indent=1),encoding='utf-8')
    print('h0',{k:(round(v,4) if isinstance(v,float) else v) for k,v in res['h0'].items()})
    for c in res['ranked']:
        x=res['singles'][c]; print(f"{c:>7} drop {x['drop']:+.4f} {[round(v,3) for v in x['drop_ci95']]} A {x['p_you_A']:.3f} U {x['p_you_U']:.3f} mass {x['mean_letter_mass']:.3f}")
    for s,x in res['additivity'].items():
        print(f"add {s}: set {x['set_drop']:.4f} sum {x['sum_single_drops']:.4f} diff {x['set_minus_sum']:+.4f} {[round(v,3) for v in x['ci95']]}")
    for k,v in res['predictions'].items(): print(k,v['verdict'],v['details'])
if __name__=='__main__': main()
