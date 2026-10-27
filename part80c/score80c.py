#!/usr/bin/env python3
"""Score Part 80c with Part 80b's scorer per option set, plus the decoy pull; verdicts for 80c.1 to 80c.7."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part80b.score80b import load, summarize, self_reference, boot

def decoy_pull(recs):
    t={(r['body_id'],r['rendering'],r['order']):r['role_probs'] for r in recs if r['body_set']=='D'}
    bodies=sorted({b for (b,_,_) in t if all((b,rn,k) in t for rn in 'AU' for k in (1,2,3))})
    d=[float(np.mean([t[(b,'A',k)]['decoy'] for k in (1,2,3)])-np.mean([t[(b,'U',k)]['decoy'] for k in (1,2,3)])) for b in bodies]
    pa=float(np.mean([np.mean([t[(b,'A',k)]['decoy'] for k in (1,2,3)]) for b in bodies]))
    pu=float(np.mean([np.mean([t[(b,'U',k)]['decoy'] for k in (1,2,3)]) for b in bodies]))
    return {'bodies':len(bodies),'P_decoy_A':pa,'P_decoy_U':pu,'pull':float(np.mean(d)),'pull_ci95':boot(d)}

def verdicts(r17, r4b, base4b_cyclic_gap):
    v={}
    L=r17['N']['locked']
    v['80c.1']=('HOLDS' if L[2] is not None and L[1] is None and L[3] is None else 'FAILS', f'locks {L}')
    v['80c.2']=self_reference(r4b['N'])
    g=r4b['N']['cyclic']['gap']
    v['80c.3']=('HOLDS' if g>=0.22 else 'FAILS', f'N cyclic gap {g:.3f} against 80b {base4b_cyclic_gap:.3f}')
    v['80c.4']=self_reference(r17['N'])
    p=r4b['decoy']
    v['80c.5']=('HOLDS' if p['pull']>=0.10 and p['pull_ci95'][0]>0 else 'FAILS', f"decoy pull {p['pull']:.3f} [{p['pull_ci95'][0]:.3f}, {p['pull_ci95'][1]:.3f}]")
    v['80c.6']=self_reference(r4b['D'])
    v['80c.7']=('HOLDS' if g<base4b_cyclic_gap else 'FAILS', f'N cyclic gap {g:.3f} against 80b {base4b_cyclic_gap:.3f}')
    return v

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--dir17',required=True); ap.add_argument('--dir4b',required=True)
    ap.add_argument('--base',required=True,help='results/part80b/results80b.json'); ap.add_argument('--out',required=True)
    a=ap.parse_args(argv)
    r={}
    for name,d in (('qwen3-1.7b',a.dir17),('qwen3.5-4b',a.dir4b)):
        recs=load(d); r[name]={oset:summarize(recs,oset) for oset in ('N','D')}; r[name]['decoy']=decoy_pull(recs)
    base=json.loads(Path(a.base).read_text(encoding='utf-8'))
    r['verdicts']=verdicts(r['qwen3-1.7b'],r['qwen3.5-4b'],base['qwen3.5-4b']['smol']['cyclic']['gap'])
    Path(a.out).write_text(json.dumps(r,indent=1,ensure_ascii=False),encoding='utf-8')
    for k,(v,e) in r['verdicts'].items(): print(k,v,e)
if __name__=='__main__': main()
