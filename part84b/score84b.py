#!/usr/bin/env python3
"""Score Part 84b: blind prepare (as Part 84) and paired decode/gen comparison with the registered rules."""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part84b.part84b import CELLS, DOSES

def prepare(recs, items_path, key_path, seed=845):
    rng=random.Random(seed); recs=list(recs); rng.shuffle(recs); ids=[f'y{i:03d}' for i in range(len(recs))]
    Path(items_path).write_text(''.join(json.dumps({'id':i,'text':r['answer']},ensure_ascii=False)+'\n' for i,r in zip(ids,recs)),encoding='utf-8')
    key={i:{'cell':r['cell'],'question_index':r['question_index'],'sample':r['sample']} for i,r in zip(ids,recs)}
    Path(key_path).write_text(json.dumps(key,indent=0),encoding='utf-8'); return key

def load_labels(rows, key):
    seen={}
    for r in rows:
        if r['id'] in seen or r['origin'] not in ('EARTH','OFFWORLD','NEITHER') or r['form'] not in ('OK','BROKEN'):
            raise ValueError(f'bad or duplicate label {r}')
        seen[r['id']]=r
    if set(seen)!=set(key): raise ValueError(f'labels cover {len(seen)} ids, key has {len(key)}')
    return {(key[i]['cell'],key[i]['question_index'],key[i]['sample']):r['origin']=='OFFWORLD' for i,r in seen.items()}

def sign_test(b, c):
    """Two-sided exact binomial test on discordant pairs."""
    n=b+c
    if n==0: return 1.0
    k=min(b,c); p=sum(math.comb(n,i) for i in range(k+1))/2**n
    return min(1.0,2*p)

def score(off, off2=None):
    cells=sorted({k[0] for k in off}); pairs=sorted({(k[1],k[2]) for k in off})
    count={c:sum(off[(c,q,s)] for q,s in pairs) for c in cells}
    res={'n_per_cell':len(pairs),'offworld':count,'paired':{}}
    for d in DOSES:
        dc,gc=f'decode{d:.2f}',f'gen{d:.2f}'
        b=sum(off[(gc,q,s)] and not off[(dc,q,s)] for q,s in pairs); c=sum(off[(dc,q,s)] and not off[(gc,q,s)] for q,s in pairs)
        res['paired'][f'{d:.2f}']={'gen_only':b,'decode_only':c,'sign_test_p':sign_test(b,c)}
    v={}; gate=count['A0']<=1
    v['84b.1']=('HOLDS' if gate else 'FAILS',f"A0 {count['A0']} of {len(pairs)}")
    def g(x): return x if gate else 'NOT READ'
    dec=[count[f'decode{d:.2f}'] for d in DOSES]; gen=[count[f'gen{d:.2f}'] for d in DOSES]
    v['84b.2']=(g('HOLDS' if dec[0]<dec[1]<dec[2] else 'FAILS'),f'decode {dec}')
    v['84b.3']=(g('HOLDS' if all(x>=y for x,y in zip(gen,dec)) and sum(gen)-sum(dec)>=12 else 'FAILS'),f'gen {gen}, decode {dec}, summed difference {sum(gen)-sum(dec)}')
    v['84b.4']=(g('HOLDS' if dec[0]<=16 and gen[0]>=24 else 'FAILS'),f'dose 0.40: decode {dec[0]}, gen {gen[0]}')
    if off2 is not None:
        agree=sum(off[k]==off2[k] for k in off)/len(off)
        v['84b.5']=('HOLDS' if agree>=.9 else 'FAILS',f'agreement {agree:.3f} on {len(off)}'); res['reader_agreement']=agree
    res['verdicts']=v; return res

def main(argv=None):
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('prepare'); p.add_argument('--dir',required=True); p.add_argument('--items',required=True); p.add_argument('--key',required=True)
    s=sp.add_parser('score'); s.add_argument('--dir',required=True); s.add_argument('--labels',required=True)
    s.add_argument('--labels2'); s.add_argument('--key',required=True)
    a=ap.parse_args(argv)
    if a.cmd=='prepare':
        recs=[json.loads(l) for l in (Path(a.dir)/'records.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        print(len(prepare(recs,a.items,a.key)),'items')
    else:
        key=json.loads(Path(a.key).read_text(encoding='utf-8'))
        rd=lambda p:[json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]
        off=load_labels(rd(a.labels),key); off2=load_labels(rd(a.labels2),key) if a.labels2 else None
        res=score(off,off2); (Path(a.dir)/'results84b.json').write_text(json.dumps(res,indent=1),encoding='utf-8')
        print(json.dumps({k:res[k] for k in ('offworld','paired')},indent=0))
        for k,(vv,d) in res['verdicts'].items(): print(k,vv,d)
if __name__=='__main__': main()
