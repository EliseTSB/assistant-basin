import json, sys
import numpy as np
S1=json.load(open(sys.argv[1],encoding='utf-8')); arms=S1['arms']
def arr(n,k='delta_g_ali'): return np.array([float(x) for x in arms[n][k]])
qs=S1.get('questions')
print('questions entry type:', type(qs), (len(qs) if hasattr(qs,'__len__') else ''))
if isinstance(qs,list) and qs:
    print('q[0]:',repr(qs[0])[:200]); print('q[24]:',repr(qs[24])[:200] if len(qs)>24 else 'n/a')
def s(name, k='delta_g_ali'):
    a=arr(name,k)
    return a.mean(), a[:24].mean(), a[24:].mean()
print()
print(f"{'arm':22s} {'mean25':>9s} {'mean24':>9s} {'q24 only':>9s}")
for n in ('nat_trace_fact','nat_trace_imp','nat_trace_you','nat_fact_no_L29','nat_you_no_L29',
          'scale_dial_fact','scale_dial_you','zero','unchanged','dial_-0.25','dial_0.80'):
    m25,m24,q24=s(n)
    print(f'{n:22s} {m25:9.4f} {m24:9.4f} {q24:9.4f}')
print()
print('nat per-question delta_g_ali (idx 0..24):')
for n in ('nat_trace_fact','nat_trace_imp','nat_trace_you'):
    a=arr(n)
    print(f'  {n:16s}', ' '.join(f'{v:+.2f}' for v in a))
print()
print('214.7 clauses on 24 held-out only:')
f,i,y = arr('nat_trace_fact')[:24].mean(), arr('nat_trace_imp')[:24].mean(), arr('nat_trace_you')[:24].mean()
print(f'  fact {f:+.6f} imp {i:+.6f} you {y:+.6f} | fact-imp {f-i:+.6f} (>=0.20 {f-i>=0.20}) | |imp-you| {abs(i-y):.6f} (<=0.30 {abs(i-y)<=0.30})')
print()
print('dial curve spearman on 24 held-out only:')
from numpy import mean
ALPHAS=(-0.25,0.0,0.10,0.20,0.30,0.40,0.50,0.65,0.80,1.0)
RUNG={-0.25:'dial_-0.25',0.0:'zero',0.10:'dial_0.10',0.20:'dial_0.20',0.30:'dial_0.30',0.40:'dial_0.40',0.50:'dial_0.50',0.65:'dial_0.65',0.80:'dial_0.80',1.0:'unchanged'}
xs=[a for a in ALPHAS]; ys=[arr(RUNG[a])[:24].mean() for a in ALPHAS]
def rk(v):
    o=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v); i=0
    while i<len(o):
        j=i
        while j+1<len(o) and v[o[j+1]]==v[o[i]]: j+=1
        for t in o[i:j+1]: r[t]=(i+j)/2+1
        i=j+1
    return np.array(r)
rx,ry=rk(xs),rk(ys)
rho=float(((rx-rx.mean())*(ry-ry.mean())).sum()/np.sqrt(((rx-rx.mean())**2).sum()*((ry-ry.mean())**2).sum()))
print('  spearman',rho)
print('  means24',['%.3f'%v for v in ys])
fact24 = arr('scale_dial_fact')[:24].mean(); fact25=arr('scale_dial_fact').mean()
print(f'  scale_dial_fact 24-only {fact24:.4f} vs 25 {fact25:.4f}')
rb=[arr(f'random_{k}:scale_dial_fact')[:24].mean() for k in range(10)]
print('  random dial 24-only beats:',sum(fact24>v for v in rb),'/10  max random',max(rb))
print()
print('214.8(b) decomposition:')
l27=lambda n: arr(n,'l27_shift').mean()
dg=lambda n: arr(n,'delta_g_ali').mean()
dial=[RUNG[a] for a in ALPHAS]; nat=('nat_trace_fact','nat_trace_you','nat_trace_imp','nat_mean_fact','nat_mean_you','nat_mean_imp')
def sp(ns):
    x=[l27(n) for n in ns]; y=[dg(n) for n in ns]
    if len(set(x))<2 or len(set(y))<2: return None
    rx,ry=rk(x),rk(y)
    return float(((rx-rx.mean())*(ry-ry.mean())).sum()/np.sqrt(((rx-rx.mean())**2).sum()*((ry-ry.mean())**2).sum()))
print('  dial-only rho (10):',sp(dial))
print('  nat-only rho (6):',sp(list(nat)))
print('  all-16 rho:',sp(dial+list(nat)))
print('  nat l27 values:',['%.3f'%l27(n) for n in nat])
print('  nat delta values:',['%.3f'%dg(n) for n in nat])
