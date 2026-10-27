#!/usr/bin/env python3
"""Score Part 80 runner outputs and apply the registered prediction rules."""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np

PAIR={'A':'A|U,A|S','U':'A|U,U|S','S':'A|S,U|S'}
def _dist(row,x,y,layer):
    key=f'{x}|{y}' if f'{x}|{y}' in row else f'{y}|{x}'
    return float(row[key])
def ratios(row, layer):
    if isinstance(row,list): row=row[layer]
    d={k:_dist(row,*k.split('|'),layer) for k in ('A|U','A|S','U|S')}
    return {'A':d['U|S']/((d['A|U']+d['A|S'])/2), 'U':d['A|S']/((d['A|U']+d['U|S'])/2), 'S':d['A|U']/((d['A|S']+d['U|S'])/2)}
def farthest_share(rows, x,y,z,layer):
    count=0
    for r in rows:
        ds=r['layers'][layer]
        vals={a:_dist(ds,x,a,layer) for a in (y,z)}
        rx=(vals[y]+vals[z])/2
        ry=(_dist(ds,x,y,layer)+_dist(ds,y,z,layer))/2
        rz=(_dist(ds,x,z,layer)+_dist(ds,y,z,layer))/2
        if rx>ry and rx>rz: count+=1
    return count/len(rows) if rows else None

def score_self_reference(records):
    out={}
    for subset in ('heldout','all'):
        rows=[r for r in records if subset=='all' or r['split']=='heldout']
        out[subset]={}
        for rendering in ('A','U','S',"U'",'T'):
            rr=[r for r in rows if r['rendering']==rendering]
            out[subset][rendering]={'mean_p_you':float(np.mean([r['role_probs']['you'] for r in rr])) if rr else None,'mean_letter_mass':float(np.mean([r['letter_mass'] for r in rr])) if rr else None}
        rr=[r for r in rows if r['rendering']=='U']
        out[subset]['U']['mean_p_user']=float(np.mean([r['role_probs']['user'] for r in rr])) if rr else None
        out[subset]['mean_letter_mass']=float(np.mean([r['letter_mass'] for r in rows])) if rows else None
    return out

def choose_reading_layer(layers):
    """First qualifying layer in the second half, given one-based layer metrics."""
    n=len(layers); threshold=n//2+1
    for x in layers:
        if x['layer']<threshold: continue
        if x['median_R_A'] is None or x['median_R_A']>=2/3 or x['A_most_distant_share'] is None or x['A_most_distant_share']<.8 or x['shuffled_label_share'] is None or x['shuffled_label_share']>=.5: continue
        if x['A_most_distant_share']-x['Uprime_most_distant_share']<.2: continue
        if x['T_most_distant_share'] is not None and x['A_most_distant_share']-x['T_most_distant_share']<.2: continue
        return x['layer']
    return None

def _means(npz):
    data=np.load(npz,allow_pickle=False)
    ids=[str(x) for x in data['body_ids'].tolist()]; splits=[str(x) for x in data['split'].tolist()]
    return {qid:{rn:data[f'{rn}__{qid}'].astype(float) for rn in ('A','U','S')} for qid in ids},dict(zip(ids,splits))

def metrics_for_dir(directory):
    p=Path(directory); run=json.loads((p/'run.json').read_text(encoding='utf-8'))
    distances=json.loads((p/'distances.json').read_text(encoding='utf-8'))
    records=[json.loads(l) for l in (p/'records.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    means,splits=_means(p/'means.npz'); n=int(run['layers'])
    held=[v for k,v in distances.items() if v['split']=='heldout']; disc=[v for k,v in distances.items() if v['split']=='discovery']
    rng=np.random.default_rng(8080); layers=[]
    for li in range(n):
        rr=[ratios(x['layers'],li) for x in held]
        ra=np.array([x['A'] for x in rr]); ru=np.array([x['U'] for x in rr]); rs=np.array([x['S'] for x in rr])
        a_share=float(np.mean((ra<ru)&(ra<rs))) if len(rr) else None
        ua=farthest_share(held,"U'",'U','S',li); ta=farthest_share(held,'T','U','S',li) if run.get('prefix_text') else None
        # Randomly relabel the three base renderings for each held-out body, 1000 trials per body.
        shuffled=0
        for x in held:
            ds=x['layers'][li]; d=np.array([_dist(ds,'A','U',li),_dist(ds,'A','S',li),_dist(ds,'U','S',li)])
            # A rendering is most distant when its mean distance to the other two is largest.
            far=np.array([(d[0]+d[1])/2,(d[0]+d[2])/2,(d[1]+d[2])/2])
            perms=np.stack([rng.permutation(3) for _ in range(1000)])
            shuffled+=int(np.sum(np.argmax(far[perms],axis=1)==0))
        shuffle_share=shuffled/(1000*len(held)) if held else None
        # Direction fit on discovery body means, then compare held-out role projections.
        names=[q for q,s in splits.items() if s=='discovery']
        direction=np.mean([means[q]['A'][li]-(means[q]['U'][li]+means[q]['S'][li])/2 for q in names],axis=0) if names else None
        sep=[]
        if direction is not None:
            for q,s in splits.items():
                if s=='heldout' and q in means:
                    m=means[q]; sep.append(np.dot(m['A'][li],direction)>np.dot(m['U'][li],direction) and np.dot(m['A'][li],direction)>np.dot(m['S'][li],direction))
        layers.append({'layer':li+1,'median_R_A':float(np.median(ra)) if len(ra) else None,'median_R_U':float(np.median(ru)) if len(ru) else None,'median_R_S':float(np.median(rs)) if len(rs) else None,'A_most_distant_share':a_share,'Uprime_most_distant_share':ua,'T_most_distant_share':ta,'shuffled_label_share':shuffle_share,'heldout_separation_share':float(np.mean(sep)) if sep else None})
    chosen=choose_reading_layer(layers)
    selfref=score_self_reference(records)
    ref=selfref['all']; control=(ref['U']['mean_p_user'] is not None and ref['U']['mean_p_user']>=.5 and ref['mean_letter_mass'] is not None and ref['mean_letter_mass']>=.8)
    du=(ref['A']['mean_p_you']-ref['U']['mean_p_you']) if ref['A']['mean_p_you'] is not None and ref['U']['mean_p_you'] is not None else None
    a_share_ok=chosen is not None
    verdicts={
      '80.1':{'text':'Control: each model gives P(user) at least 0.5 for User-role bodies, with the three letters carrying at least 0.8 of the mass.','holds':bool(control)},
      '80.2':{'text':'Control: under shuffled role labels, the share of held-out bodies whose "A" rendering is the most distant is below 50% at every layer, in both models.','holds':None if any(x['shuffled_label_share'] is None for x in layers) else bool(all(x['shuffled_label_share']<.5 for x in layers))},
      '80.2a':{'text':'Control (Amendment 1): at the layer where each model\'s self-index is read, U\' is the most distant of (U\', U, S) in fewer than 50% of held-out bodies; T is reported against the same line when it is run.','holds':None if chosen is None else bool(next(x for x in layers if x['layer']==chosen)['Uprime_most_distant_share']<.5)},
      '80.3':{'text':'Claude: Qwen3-1.7B has a self-index by the reading rule.','holds':bool(a_share_ok) if run['model']=='qwen3-1.7b' else None},
      '80.4':{'text':'Claude: Qwen3.5-4B has a self-index by the reading rule.','holds':bool(a_share_ok) if run['model']=='qwen3.5-4b' else None},
      '80.5':{'text':'Claude: Qwen3-1.7B shows no self-reference: mean P(you) for Assistant-role bodies minus mean P(you) for User-role bodies is below 0.10.','holds':None if (run['model']=='qwen3-1.7b' and (not control or du is None)) else bool(du<.10) if run['model']=='qwen3-1.7b' else None},
      '80.6':{'text':'Claude: Qwen3.5-4B shows no self-reference by the same rule (difference below 0.10).','holds':None if (run['model']=='qwen3.5-4b' and (not control or du is None)) else bool(du<.10) if run['model']=='qwen3.5-4b' else None},
    }
    for v in verdicts.values(): v['status']='NOT READ' if v['holds'] is None else ('HOLDS' if v['holds'] else 'FAILED')
    return {'model':run['model'],'revision':run['revision'],'self_reference':selfref,'self_index':{'reading_layer':chosen,'layers':layers},'verdicts':verdicts}

def score_dirs(dirs):
    results=[metrics_for_dir(d) for d in dirs]
    # Preserve paired rows for a convenient consumer; each invocation can score one or both model dirs.
    out={'models':results}
    for model in results:
        for ident in ('80.1','80.2','80.2a'):
            # Cohort controls are evaluated across both models when both are provided.
            if ident=='80.1': model['verdicts'][ident]['holds']=all(m['self_reference']['all']['U']['mean_p_user']>=.5 and m['self_reference']['all']['mean_letter_mass']>=.8 for m in results)
            elif ident=='80.2':
                vals=[l['shuffled_label_share'] for m in results for l in m['self_index']['layers']]
                model['verdicts'][ident]['holds']=None if any(x is None for x in vals) else all(x<.5 for x in vals)
            model['verdicts'][ident]['status']='NOT READ' if model['verdicts'][ident]['holds'] is None else ('HOLDS' if model['verdicts'][ident]['holds'] else 'FAILED')
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--dirs',nargs='+',required=True); a=ap.parse_args(); out=score_dirs(a.dirs)
    target=Path(a.dirs[0]); dest=target/'results80.json'
    with dest.open('w',encoding='utf-8') as f: json.dump(out,f,ensure_ascii=False,indent=2)
    print('model          layer  mean P(you) A/U  mean P(user) U  letter mass  80.1  80.2  80.2a  index')
    for m in out['models']:
        s=m['self_reference']['all']; v=m['verdicts']; print(f"{m['model']:<14} {str(m['self_index']['reading_layer']):>5}  {s['A']['mean_p_you']:.3f}/{s['U']['mean_p_you']:.3f}       {s['U']['mean_p_user']:.3f}         {s['mean_letter_mass']:.3f}       {v['80.1']['status']:<6} {v['80.2']['status']:<6} {v['80.2a']['status']:<6} {('yes' if m['self_index']['reading_layer'] else 'no')}")
if __name__=='__main__': main()
