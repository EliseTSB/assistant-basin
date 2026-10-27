#!/usr/bin/env python3
"""Score Part 80b: locked orders, per-order gaps with paired bootstrap intervals, cyclic averages, verdicts."""
from __future__ import annotations
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

def load(d):
    return [json.loads(x) for x in (Path(d)/'records.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]

def table(recs, body_set):
    """t[(body, rendering, order)] = role_probs, for one body set; letters[(rendering, order)] = list of letter dicts."""
    t={}; letters=defaultdict(list); mass=[]
    for r in recs:
        if r['body_set']!=body_set: continue
        t[(r['body_id'],r['rendering'],r['order'])]=r['role_probs']
        letters[(r['rendering'],r['order'])].append(r['letter_probs']); mass.append(r['letter_mass'])
    return t, letters, mass

def renorm(d):
    s=sum(d.values()); return {k:v/s for k,v in d.items()}

def locked_orders(letters):
    out={}
    for k in (1,2,3):
        lock=None
        for L in 'ABC':
            if all(np.mean([renorm(x)[L] for x in letters[(rn,k)]])>=0.9 for rn in ('A','U','S') if letters[(rn,k)]): lock=L
        if not all(letters[(rn,k)] for rn in ('A','U','S')): lock=None
        out[k]=lock
    return out

def boot(diffs, n=10000, seed=80):
    rng=random.Random(seed); m=len(diffs)
    means=sorted(sum(diffs[rng.randrange(m)] for _ in range(m))/m for _ in range(n))
    return [means[int(0.025*n)], means[int(0.975*n)-1]]

def summarize(recs, body_set):
    t,letters,mass=table(recs,body_set)
    bodies=sorted({b for (b,_,_) in t})
    full=[b for b in bodies if all((b,rn,k) in t for rn in ('A','U','S') for k in (1,2,3))]
    res={'bodies':len(full),'mean_letter_mass':float(np.mean(mass)) if mass else None,'locked':locked_orders(letters),'orders':{}}
    def mean_role(role, rn, k=None):
        if k is None: return float(np.mean([np.mean([t[(b,rn,kk)][role] for kk in (1,2,3)]) for b in full]))
        return float(np.mean([t[(b,rn,k)][role] for b in full]))
    for k in (1,2,3):
        diffs=[t[(b,'A',k)]['you']-t[(b,'U',k)]['you'] for b in full]
        res['orders'][k]={'P_you':{rn:mean_role('you',rn,k) for rn in 'AUS'},'P_user_U':mean_role('user','U',k),
                         'gap':float(np.mean(diffs)),'gap_ci95':boot(diffs)}
    cyc=[np.mean([t[(b,'A',k)]['you'] for k in (1,2,3)])-np.mean([t[(b,'U',k)]['you'] for k in (1,2,3)]) for b in full]
    res['cyclic']={'P_you':{rn:mean_role('you',rn) for rn in 'AUS'},'P_user_U':mean_role('user','U'),
                   'gap':float(np.mean(cyc)),'gap_ci95':boot([float(x) for x in cyc])}
    return res

def self_reference(s):
    unlocked=[k for k in (1,2,3) if s['locked'][k] is None]
    if len(unlocked)<2: return 'NOT READ', f'{len(unlocked)} unlocked order(s)'
    ok=all(s['orders'][k]['gap']>=0.10 and s['orders'][k]['gap_ci95'][0]>0 for k in unlocked)
    return ('HOLDS' if ok else 'FAILS'), 'unlocked orders '+', '.join(f"O{k} gap {s['orders'][k]['gap']:.2f} [{s['orders'][k]['gap_ci95'][0]:.2f}, {s['orders'][k]['gap_ci95'][1]:.2f}]" for k in unlocked)

def verdicts(r17, r4b):
    s17, s4=r17['smol'], r4b['smol']; v={}
    v['80b.1']=('HOLDS' if s17['locked'][2] is not None else 'FAILS', f"O2 lock {s17['locked'][2]}")
    v['80b.2']=('HOLDS' if all(x is None for x in s4['locked'].values()) else 'FAILS', f"locks {s4['locked']}")
    v['80b.3']=self_reference(s4)
    v4=self_reference(s17)
    v['80b.4']=(v4[0] if v4[0]=='NOT READ' else ('HOLDS' if v4[0]=='HOLDS' and s17['locked'][3] is None else 'FAILS'), v4[1]+f"; O3 lock {s17['locked'][3]}")
    v['80b.5']=('HOLDS' if s4['cyclic']['gap']>=0.10 else 'FAILS', f"cyclic gap {s4['cyclic']['gap']:.3f}")
    own,sm=r17['qwenown']['cyclic']['P_you']['A'], s17['cyclic']['P_you']['A']
    v['80b.6']=('HOLDS' if abs(own-sm)<0.05 else 'FAILS', f"P(you|A) own {own:.3f}, smol {sm:.3f}")
    p=s4['cyclic']['P_you']
    v['80b.7']=('HOLDS' if min(p['U'],p['A'])<p['S']<max(p['U'],p['A']) else 'FAILS', f"U {p['U']:.3f} S {p['S']:.3f} A {p['A']:.3f}")
    return v

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--dir17',required=True); ap.add_argument('--dir4b',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(argv)
    r={name:{bs:summarize(load(d),bs) for bs in ('smol','qwenown')} for name,d in (('qwen3-1.7b',a.dir17),('qwen3.5-4b',a.dir4b))}
    r['verdicts']=verdicts(r['qwen3-1.7b'],r['qwen3.5-4b'])
    Path(a.out).write_text(json.dumps(r,indent=1,ensure_ascii=False),encoding='utf-8')
    for k,(v,e) in r['verdicts'].items(): print(k,v,e)
if __name__=='__main__': main()
