import json, numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer
D='results/part126/'
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', local_files_only=True)
z=np.load(D+'reads126.npz', allow_pickle=False)
att=z['attention'].astype(np.float64); rid=z['region_ids'].astype(np.int64); L=z['lengths'].astype(np.int64); ids=z['input_ids']
rv=z['region_values'].astype(np.float64)  # (600,2,16,6,256)
rows=[json.loads(l) for l in open(D+'rows.jsonl') if l.strip()]
copy=np.array([r['rendering'] for r in rows]); N=len(rows)
keys=[f'L{l}h{h}' for l in (11,15) for h in range(16)]
s83=json.load(open('results/part83/main/results83.json'))['singles']
absd=np.array([abs(s83[k]['drop']) for k in keys]); sgn=np.array([s83[k]['drop'] for k in keys])
vocab={int(t):tok.convert_ids_to_tokens(int(t)) for t in np.unique(ids)}
norm=lambda s: s.replace('Ġ','').replace('Ċ','').lower()
isrole=np.zeros(ids.shape,bool); isgen=np.zeros(ids.shape,bool); isqh=np.zeros(ids.shape,bool); isopt=np.zeros(ids.shape,bool); ishdr=np.zeros(ids.shape,bool)
for i in range(N):
    n=L[i]
    for j in range(n):
        s=norm(vocab[int(ids[i,j])])
        if s in ('assistant','user'):
            isrole[i,j]=True
            if j==1: ishdr[i,j]=True
            elif rid[i,j]==4: isopt[i,j]=True
            elif j>n-8: isgen[i,j]=True
            else: isqh[i,j]=True
def mass(mask):
    out=np.zeros((N,2,16))
    for i in range(N):
        n=L[i]
        out[i]=(att[i,:,:,:n]*mask[i,:n]).sum(-1)
    return out
M={'header(region role)':mass(ishdr),'gen-prefix word':mass(isgen),'question-header word':mass(isqh),'option-text word':mass(isopt),'ALL five role-word occurrences':mass(isrole)}
print('check classes sum to all:', np.allclose(M['header(region role)']+M['gen-prefix word']+M['question-header word']+M['option-text word'], M['ALL five role-word occurrences']))
# region-role value constancy across rows (single token at pos1, causal -> constant per copy?)
for cp in ('A','U'):
    x=rv[copy==cp][:,:,:,0,:]  # role region values
    print('role-region value spread across rows, copy',cp,' max over heads/dims of std:', float(x.std(0).max()), ' max |value|:', float(np.abs(x).max()))
# typical magnitude of gated values (for the 0.05 recompute gate context)
allv=rv[:,:,:,:,:]
print('typical |region value| median (nonzero):', float(np.median(np.abs(allv[allv!=0]))), ' 99th pct:', float(np.quantile(np.abs(allv),0.99)))
res={}
for name,m in M.items():
    for cp in ('A','U','pooled'):
        sel = (copy==cp) if cp!='pooled' else np.ones(N,bool)
        pooled=m[sel].mean(0).reshape(-1)
        r=spearmanr(pooled,absd).statistic
        res[(name,cp)]=r
        if cp=='pooled':
            r_signed=spearmanr(pooled,sgn).statistic
            print(f'{name:34s} pooled mean mass L11h5 {pooled[5]*100:6.3f}%  rho(|drop|)={r:+.4f}  rho(signed)={r_signed:+.4f}  mass-rank of L11h5={int((pooled>pooled[5]).sum())+1}')
        else:
            print(f'      copy {cp}: rho(|drop|)={r:+.4f}')
# null for the broad predictor (same shuffle design)
rng=np.random.default_rng(12602)
pool_all=M['ALL five role-word occurrences'].mean(0).reshape(-1)
null=np.array([spearmanr(pool_all, rng.permutation(absd)).statistic for _ in range(10000)])
print('broad predictor: rho', round(spearmanr(pool_all,absd).statistic,4), ' null p95', round(np.quantile(null,.95),4), ' p', round(((null>=spearmanr(pool_all,absd).statistic).sum()+1)/10001,4))
