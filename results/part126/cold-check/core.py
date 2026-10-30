import json, re, numpy as np
from scipy.stats import spearmanr
D='results/part126/'
z=np.load(D+'reads126.npz', allow_pickle=False)
att=z['attention'].astype(np.float64)        # (600,2,16,151)
rid=z['region_ids'].astype(np.int64)         # (600,151), 255 pad
L=z['lengths'].astype(np.int64)
rm_saved=z['region_mass'].astype(np.float64)
rows=[json.loads(l) for l in open(D+'rows.jsonl') if l.strip()]
copy=np.array([r['rendering'] for r in rows])
N=len(rows)
# 1. recompute region masses from raw attention
rm=np.zeros((N,2,16,6))
for i in range(N):
    n=L[i]
    w=att[i,:,:,:n]
    for r in range(6):
        rm[i,:,:,r]=w[..., rid[i,:n]==r].sum(-1)
print('1 region-mass rebuild max |diff| vs saved f16:', np.abs(rm-rm_saved).max())
print('  row sums: max|sum-1| (recomputed)', np.abs(rm.sum(-1)-1).max(), ' attention sums over n:', np.abs(att.sum(-1)-1).max() if False else '')
ssum=np.array([att[i,:,:,:L[i]].sum(-1) for i in range(N)])
print('  raw attention row sums max|sum-1|:', np.abs(ssum-1).max())
print('  role tokens per row (count of region 0) min/max:', (rid==0).sum(1).min(), (rid==0).sum(1).max())
# per-copy means (display as in NUMBERS, 3-dec percent)
A=copy=='A'; U=copy=='U'
mA=rm[A].mean(0); mU=rm[U].mean(0); pooled=rm.mean(0)
drop83=json.load(open('results/part83/main/results83.json'))['singles']
keys=[f'L{l}h{h}' for l in (11,15) for h in range(16)]
absd=np.array([abs(drop83[k]['drop']) for k in keys])
signd=np.array([drop83[k]['drop'] for k in keys])
pr=pooled[:,:,0].reshape(-1)   # layer-major order matches keys
rho=spearmanr(pr,absd).statistic
print('2 rho(pooled role, |drop|) =', round(rho,6), ' scipy')
rng=np.random.default_rng(12602)
null=np.array([spearmanr(pr, rng.permutation(absd)).statistic for _ in range(10000)])
print('  null p95 =', round(np.quantile(null,.95,method='linear'),6), ' exceed count (>=rho) =', int((null>=rho).sum()), ' p=(k+1)/(N+1)=', round(((null>=rho).sum()+1)/10001,6), ' raw k/N=', (null>=rho).mean())
print('  null sd =', round(null.std(),4), ' theory 1/sqrt(31)=', round(1/np.sqrt(31),4))
# rank of L11h5 by A-copy role mass
aR=mA[:,:,0].reshape(-1)
order=sorted(range(32), key=lambda i:-aR[i])
rank=order.index(keys.index('L11h5'))+1
print('3 L11h5 A-copy role mass =', round(aR[keys.index('L11h5')]*100,4),'%  rank', rank, 'of 32')
print('  top5 A role:', [(keys[i], round(aR[i]*100,3)) for i in order[:5]])
print('  ties in A role mass:', len(set(np.round(aR,12)))<32)
# signed-drop and layer checks later
# 4 region mass tables: print for top 5 by drop
for k in ['L11h5','L11h7','L11h4','L15h1','L15h2']:
    i=keys.index(k); l=0 if k.startswith('L11') else 1; h=int(k.split('h')[1])
    print(f'4 {k} A', [round(x*100,3) for x in mA[l,h]], ' U', [round(x*100,3) for x in mU[l,h]])
# largest-region counts
regn=['role','template','body','question','options','other']
for nm,M in (('A',mA),('U',mU)):
    cnt={}
    for l in range(2):
        for h in range(16):
            j=int(np.argmax(M[l,h])); cnt[regn[j]]=cnt.get(regn[j],0)+1
    print('5 largest-region counts copy',nm,cnt)
# 6 gates: region sum error
print('6 region sum max|sum-1| (saved f16):', np.abs(rm_saved.sum(-1)-1).max(), '(f32 sum) recomputed:', np.abs(rm_saved.astype(np.float32).sum(-1)-1).max())
# 7 cosines
v=np.load(D+'lens/lens126-vectors.npz', allow_pickle=False)
rw=v['role_writes'].astype(np.float64)  # (2,16,2,2560): layer, head, copy
cos=np.zeros((2,16))
for l in range(2):
    for h in range(16):
        a=rw[l,h,0]; u=rw[l,h,1]
        cos[l,h]=a@u/np.linalg.norm(a)/np.linalg.norm(u)
res=json.load(open('results/part83/main/results83.json'))['singles']
mat=[k for k in keys if abs(res[k]['drop'])>=0.05 and (res[k]['drop_ci95'][0]>0 or res[k]['drop_ci95'][1]<0)]
print('7 mattering count', len(mat), mat)
below=[k for k in mat if cos[(0 if k.startswith('L11') else 1), int(k.split('h')[1])]<0.90]
print('  below .90:', len(below))
for k in mat:
    print('   ', k, round(cos[(0 if k.startswith('L11') else 1), int(k.split('h')[1])],6))
