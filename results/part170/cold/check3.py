import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from math import log, exp
def lg(p): return log(p/(1-p))
# push rows keyed by (dose, prompt, layer, slot, ban)
P = {}
for r in ANS:
    if r["stage"]=="push":
        P[(r["dose"], r["prompt_id"], r["layer"], r["slot"], r["ban"])] = r
keys = sorted({(k[1],k[2],k[3]) for k in P})
def rise(dose, ban, filt=lambda k: True, base="b0"):
    ks=[k for k in keys if filt(k)]
    a=[P[(dose,)+k+(ban,)]["L"] for k in ks]; b=[P[(dose,)+k+(base,)]["L"] for k in ks]
    n=len(ks); return (sum(a)-sum(b))/n if n else float("nan"), sum(a)/n, sum(b)/n
print("== 5. dose shape (b50 vs b0 on the same 384 units)")
for d in (0.25,0.5,1.0):
    r,pl,p0 = rise(d,"b50")
    r5,pl5,_=rise(d,"b5")
    print(f"dose {d}: b0={100*p0:.2f} b50={100*pl:.2f} rise={100*r:.2f}pp | logit b0={lg(p0):.3f} b50={lg(pl):.3f} logit rise={lg(pl)-lg(p0):.3f} odds ratio={(pl/(1-pl))/(p0/(1-p0)):.2f} | b5 rise={100*r5:.2f} logit={lg(pl5)-lg(p0):.3f}")
print("== b200 and rel50 by dose (logit)")
for d in (0.25,0.5,1.0):
    for b in ("b200",):
        r,pl,p0=rise(d,b); print(f"dose {d} {b}: b0={100*p0:.2f} {b}={100*pl:.2f} rise={100*r:.2f} logit={lg(pl)-lg(p0):.3f}")
r,pl,p0=rise(1.0,"rel50"); print(f"dose 1 rel50: b0={100*p0:.2f} rel50={100*pl:.2f} rise={100*r:.2f} logit={lg(pl)-lg(p0):.3f}")
print("== 6. fragility, dose 1, b50 rise")
def show(label, filt):
    r,pl,p0=rise(1.0,"b50",filt); print(f"  {label:34s} n={sum(1 for k in keys if filt(k))} rise={100*r:+.2f} b0={100*p0:.2f} b50={100*pl:.2f}")
show("all", lambda k: True)
for L in range(19,27): show(f"drop layer {L}", lambda k,L=L: k[1]!=L)
for S in range(6): show(f"drop slot {S}", lambda k,S=S: k[2]!=S)
for ph in PHR: show(f"drop phrasing {ph}", lambda k,ph=ph: k[0]!=ph)
print("  per phrasing rises (dose 1 b50-b0):", {ph: round(100*rise(1.0,'b50',lambda k,ph=ph: k[0]==ph)[0],1) for ph in PHR})
print("  per layer rises:", {L: round(100*rise(1.0,'b50',lambda k,L=L: k[1]==L)[0],1) for L in range(19,27)})
# neuron-level
neur = collections.defaultdict(list)
for k in keys:
    neur[(k[1],k[2])].append(int(P[(1.0,)+k+("b50",)]["L"]) - int(P[(1.0,)+k+("b0",)]["L"]))
net = {n: sum(v) for n,v in neur.items()}
vals = sorted(net.values(), reverse=True)
print("neurons (of 48) with net>0:", sum(v>0 for v in vals), " net=0:", sum(v==0 for v in vals), " net<0:", sum(v<0 for v in vals))
print("sorted per-neuron net (sum over 8 phrasings):", vals)
tot = sum(vals); print("total net", tot, "top-5 neurons net", sum(vals[:5]), "top-10", sum(vals[:10]))
# b5 dose 0.25 by phrasing
print("== b5 dose 0.25 and b50 by phrasing")
for ph in PHR:
    r5=rise(0.25,"b5",lambda k,ph=ph:k[0]==ph)[0]; r50=rise(0.25,"b50",lambda k,ph=ph:k[0]==ph)[0]
    print(f"  {ph} b5 rise {100*r5:+.1f}  b50 rise {100*r50:+.1f}  b0 loops {sum(P[(0.25,)+k+('b0',)]['L'] for k in keys if k[0]==ph)}/48")
print("b5 dose 0.25 rise by neuron: positive neurons", sum(1 for n in set((k[1],k[2]) for k in keys) if sum(P[(0.25,)+k+('b5',)]['L']-P[(0.25,)+k+('b0',)]['L'] for k in keys if (k[1],k[2])==n)>0), "of 48")
# which phrasings loop in no-push P8 b5 and b200
for b in ("b5","b200"):
    for p in PHR:
        r=[x for x in ANS if x["stage"]=="no_push" and x["prompt_id"]==p and x["ban"]==b][0]
        if r["L"]: print("no-push", b, p, "LOOP:", r["answer"][:260].replace("\n"," "))
print("== no push P8 frozen? none; b0 no-push loops:", [p for p in PHR if [x for x in ANS if x["stage"]=="no_push" and x["prompt_id"]==p and x["ban"]=="b0"][0]["L"]])
