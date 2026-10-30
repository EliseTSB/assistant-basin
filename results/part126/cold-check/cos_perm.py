import json, numpy as np
from scipy.stats import spearmanr
D='results/part126/'
v=np.load(D+'lens/lens126-vectors.npz', allow_pickle=False)
rw=v['role_writes'].astype(np.float64); rg=v['region_writes'].astype(np.float64); bw=v['body_control_writes'].astype(np.float64)
print('vector keys', v.files, rw.shape)
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
keys=[f'L{l}h{h}' for l in (11,15) for h in range(16)]
s83=json.load(open('results/part83/main/results83.json'))['singles']
mat=[k for k in keys if abs(s83[k]['drop'])>=0.05 and (s83[k]['drop_ci95'][0]>0 or s83[k]['drop_ci95'][1]<0)]
print('head   cos(roleA,roleU)  cos(roleA,bodyA)  cos(roleU,bodyU)  cos(roleA,regionA)  cos(roleU,regionU)')
out=[]
for k in mat:
    l=0 if k.startswith('L11') else 1; h=int(k.split('h')[1])
    aA,aU=rw[l,h,0],rw[l,h,1]
    row=(cos(aA,aU), cos(aA,bw[l,h,0]), cos(aU,bw[l,h,1]), cos(aA,rg[l,h,0]), cos(aU,rg[l,h,1]))
    out.append(row)
    print(f'{k:7s}', ' '.join(f'{x:8.3f}' for x in row))
out=np.array(out)
print('median across 20 mattering heads:', np.round(np.median(out,0),3))
print('min/max of role-vs-role cos:', round(out[:,0].min(),3), round(out[:,0].max(),3))
print('count role-vs-role < .90:', int((out[:,0]<0.90).sum()), ' count role-vs-body-A < .90:', int((out[:,1]<0.90).sum()), ' role-vs-region-A <.90:', int((out[:,3]<0.90).sum()))
# permutation for post hoc predictors (same design: shuffle |drop|, seed 12602, 10000)
z=np.load(D+'reads126.npz', allow_pickle=False)
rm=z['region_mass'].astype(np.float64)
rows=[json.loads(l) for l in open(D+'rows.jsonl') if l.strip()]
copy=np.array([r['rendering'] for r in rows])
absd=np.array([abs(s83[k]['drop']) for k in keys]); sgn=np.array([s83[k]['drop'] for k in keys])
from scipy.stats import spearmanr as sp
# recompute the exploratory predictors from raw attention
import numpy as np
att=z['attention'].astype(np.float64); L=z['lengths']; ids=z['input_ids']; rid=z['region_ids'].astype(np.int64)
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', local_files_only=True)
vocab={int(t):tok.convert_ids_to_tokens(int(t)) for t in np.unique(ids)}
norm=lambda s: s.replace('Ġ','').replace('Ċ','').lower()
N=600
pred={'header':np.zeros((N,2,16)),'genprefix':np.zeros((N,2,16)),'qheader':np.zeros((N,2,16)),'option':np.zeros((N,2,16))}
for i in range(N):
    n=L[i]
    for j in range(n):
        s=norm(vocab[int(ids[i,j])])
        if s not in ('assistant','user'): continue
        if j==1: key='header'
        elif rid[i,j]==4: key='option'
        elif j>n-8: key='genprefix'
        else: key='qheader'
        pred[key][i]+=att[i,:,:,j]
rng=np.random.default_rng(12602)
for name,m in pred.items():
    x=m.mean(0).reshape(-1)
    r=sp(x,absd).statistic
    null=np.array([sp(x,rng.permutation(absd)).statistic for _ in range(10000)])
    p1=((null>=r).sum()+1)/10001
    p2=((np.abs(null)>=abs(r)).sum()+1)/10001
    print(f'{name:9s} rho={r:+.4f}  one-sided p(>=rho)={p1:.4f}  two-sided p={p2:.4f}  shuffle p95={np.quantile(null,.95):.3f}')
# 4 predictors (header, gen, qheader, option): Bonferroni
