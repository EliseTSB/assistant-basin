import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
P = {}
for r in ANS:
    if r["stage"]=="push": P[(r["dose"], r["prompt_id"], r["layer"], r["slot"], r["ban"])] = r
keys = sorted({(k[1],k[2],k[3]) for k in P})
# share: distinct b0 tokens on the row's b50 list (banned_ids of the b50 row)
def share(dose):
    out=[]
    for k in keys:
        b0=P[(dose,)+k+("b0",)]; b50=P[(dose,)+k+("b50",)]
        ids=set(b0["token_ids"]); held=set(b50["banned_ids"])
        out.append(len(ids&held)/len(ids) if ids else 0.0)
    return np.array(out)
for d in (0.25,0.5,1.0):
    s=share(d); print("dose",d,"mean distinct-token share",round(100*s.mean(),2))
# compare with scorer's list-based b50 (lists.jsonl b50 of the phrasing)
s1=share(1.0)
# 170.8 on dose 1 rows with no b0 loop
no=[(k) for k in keys if not P[(1.0,)+k+("b0",)]["L"]]
sh=np.array([len(set(P[(1.0,)+k+('b0',)]['token_ids'])&set(P[(1.0,)+k+('b50',)]['banned_ids']))/len(set(P[(1.0,)+k+('b0',)]['token_ids'])) for k in no])
ln=np.array([P[(1.0,)+k+('b0',)]['T'] for k in no])
y=np.array([float(P[(1.0,)+k+('b50',)]['L']) for k in no])
print("rows no-loop b0:", len(no), "b50 induced:", int(y.sum()))
med=np.median(sh); lo=sh<med
print("raw median",round(med,4),"below rows",lo.sum(),"rest",(~lo).sum(),"below rate",round(y[lo].mean(),4),"rest rate",round(y[~lo].mean(),4),"diff",round(100*(y[lo].mean()-y[~lo].mean()),2))
cuts=np.quantile(ln,[0.25,0.5,0.75]); q=np.searchsorted(cuts,ln,side="right")
tot=0; wsum=0; rows_out=[]
for qq in range(4):
    idx=np.flatnonzero(q==qq); m=np.median(sh[idx]); bl=idx[sh[idx]<m]; br=idx[sh[idx]>=m]
    dif=y[bl].mean()-y[br].mean()
    rows_out.append((qq+1,len(idx),ln[idx].min(),ln[idx].max(),len(bl),len(br),round(100*y[bl].mean(),1),round(100*y[br].mean(),1),round(100*dif,1)))
    tot+=dif*len(idx); wsum+=len(idx)
for r in rows_out: print("quartile",r)
print("row-weighted within-quartile gap", round(100*tot/wsum,2))
print("cuts", cuts)
# per-length b0 loop and share
