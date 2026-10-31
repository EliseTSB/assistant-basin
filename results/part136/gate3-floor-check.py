import json,sys,random
import numpy as np
G='/mnt/c/Users/alexa/neuron-probe-gpu/results/part136/pilot-effects-b16/effects.jsonl'
rows=[json.loads(l) for l in open(G,encoding='utf-8')]
print(len(rows),list(rows[0].keys()))
by={}
for r in rows: by.setdefault(r['pole_id'],[]).append(r)
by={k:v for k,v in by.items() if len(v)==24}
print('poles',len(by))
def band(l): return 'L0-22' if l<=22 else ('L23-23' if l==23 else 'L24-31')
def pear(a,b):
    a=a-a.mean(); b=b-b.mean(); d=np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/d) if d>0 else None
def off(E,ids,perm=None):
    means={p:E[p].mean(0) for p in ids}
    out=[]
    for p in ids:
        for q in ids:
            if p==q: continue
            for e in E[p]:
                r=pear(e[perm] if perm is not None else e,means[q])
                if r is not None: out.append(abs(r))
    return float(np.median(out)),len(out)
import base64
E={k:np.asarray([np.frombuffer(base64.b64decode(e['effect_a_f32_le_b64']),dtype='<f4') for e in v],dtype=np.float64) for k,v in by.items()}
names=sorted({band(v[0]['layer']) for v in by.values()})
rng=np.random.default_rng(136)
N=next(iter(E.values())).shape[1]; print('tokens',N)
for n in names:
    ids=[k for k,v in by.items() if band(v[0]['layer'])==n]
    a,c=off(E,ids)
    perm=rng.permutation(N)
    b,_=off(E,ids,perm)
    grand=np.concatenate([E[k] for k in ids]).mean(0)
    # remove the shared direction by regression on the band mean, each vector
    def rm(x):
        g=grand-grand.mean(); y=x-x.mean(); return y-g*(y@g)/(g@g)
    R={k:np.asarray([rm(e) for e in E[k]]) for k in ids}
    d,_=off(R,ids)
    shared=[pear(e,grand) for k in ids for e in E[k]]
    print(n,'poles',len(ids),'pairs',c,'median_abs_r',round(a,4),'vocab-shuffled',round(b,4),'after_removing_band_mean',round(d,4),'median r with band mean',round(float(np.median(shared)),3))
