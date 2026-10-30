import json, numpy as np
from scipy.stats import spearmanr, norm
from scipy.stats import rankdata
D='results/part126/'
z=np.load(D+'reads126.npz', allow_pickle=False)
rm=z['region_mass'].astype(np.float64)
rows=[json.loads(l) for l in open(D+'rows.jsonl') if l.strip()]
copy=np.array([r['rendering'] for r in rows])
keys=[f'L{l}h{h}' for l in (11,15) for h in range(16)]
s83=json.load(open('results/part83/main/results83.json'))['singles']
absd=np.array([abs(s83[k]['drop']) for k in keys]); sgn=np.array([s83[k]['drop'] for k in keys])
lo=np.array([s83[k]['drop_ci95'][0] for k in keys]); hi=np.array([s83[k]['drop_ci95'][1] for k in keys])
pr=rm.mean(0)[:,:,0].reshape(-1)
rho=spearmanr(pr,absd).statistic
print('rho',round(rho,4))
# Fisher CI (approx)
se=1/np.sqrt(29); zc=np.arctanh(rho)
print('Fisher 95% CI for rho:', round(np.tanh(zc-1.96*se),3), round(np.tanh(zc+1.96*se),3))
# bootstrap over heads
rng=np.random.default_rng(1)
bs=[]
for _ in range(10000):
    idx=rng.integers(0,32,32)
    if np.ptp(pr[idx])==0 or np.ptp(absd[idx])==0: continue
    bs.append(spearmanr(pr[idx],absd[idx]).statistic)
bs=np.array(bs); print('bootstrap over heads 95% CI:', np.round(np.quantile(bs,[.025,.975]),3), ' n',len(bs))
# leave-one-out
loo=[spearmanr(np.delete(pr,i),np.delete(absd,i)).statistic for i in range(32)]
print('LOO rho range:', round(min(loo),3), round(max(loo),3), ' LOO without L11h5:', round(loo[keys.index('L11h5')],3))
top5=np.argsort(-absd)[:5]
m=np.ones(32,bool); m[top5]=False
print('rho without 5 largest drops:', round(spearmanr(pr[m],absd[m]).statistic,3), ' n=27')
# signed drop
print('rho(role, SIGNED drop):', round(spearmanr(pr,sgn).statistic,4))
# layer confound
for li,l in enumerate((11,15)):
    sl=slice(li*16,(li+1)*16)
    print(f'layer {l}: rho within layer (n=16)', round(spearmanr(pr[sl],absd[sl]).statistic,3), ' mean role mass %', round(pr[sl].mean()*100,3), ' mean |drop|', round(absd[sl].mean(),4))
print('layer means: role L11 vs L15 %:', round(pr[:16].mean()*100,3), round(pr[16:].mean()*100,3))
# drop-uncertainty MC
sd=(hi-lo)/(2*1.96)
mc=[]
for _ in range(5000):
    d=np.abs(s83 and (np.array([s83[k]['drop'] for k in keys]) + rng.normal(0,1,32)*sd))
    mc.append(spearmanr(pr,d).statistic)
mc=np.array(mc); print('rho under Part 83 drop CI noise: 2.5/50/97.5%', np.round(np.quantile(mc,[.025,.5,.975]),3))
# permutation null p95, and power
null=np.array([spearmanr(pr,rng.permutation(absd)).statistic for _ in range(10000)])
print('null p95', round(np.quantile(null,.95),4), ' null 2.5/97.5', np.round(np.quantile(null,[.025,.975]),3))
# power by simulation: x = r*zeta + sqrt(1-r^2)*e, zeta = normal scores of |drop| ranks
zeta=norm.ppf((rankdata(absd)-0.5)/32)
for target in [0.2,0.3,0.4,0.5]:
    # find r giving mean spearman approx target
    best=None
    for r in np.linspace(0.05,0.99,95):
        sims=[spearmanr(r*zeta+np.sqrt(1-r*r)*rng.normal(size=32), absd).statistic for _ in range(400)]
        m_=np.mean(sims)
        if best is None or abs(m_-target)<abs(best[1]-target): best=(r,m_)
    r=best[0]
    sims=np.array([spearmanr(r*zeta+np.sqrt(1-r*r)*rng.normal(size=32), absd).statistic for _ in range(3000)])
    print(f'pop-ish spearman {target:.2f} (r={r:.2f}): P(rho>=0.40)={np.mean(sims>=0.40):.2f}  P(rho>=0.29)={np.mean(sims>=0.2907):.2f}')
# null-only check: under independence, P(rho>=0.40)
sims0=np.array([spearmanr(rng.normal(size=32), absd).statistic for _ in range(10000)])
print('type I: P(rho>=0.40 | null)=', np.mean(sims0>=0.40), ' P(rho>=0.2907|null)=', np.mean(sims0>=0.2907))
# copy-level
for cp in ('A','U'):
    p_=rm[copy==cp].mean(0)[:,:,0].reshape(-1)
    print('copy',cp,'rho', round(spearmanr(p_,absd).statistic,3))
