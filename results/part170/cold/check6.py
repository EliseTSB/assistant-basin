import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
P = {}
for r in ANS:
    if r["stage"]=="push": P[(r["dose"], r["prompt_id"], r["layer"], r["slot"], r["ban"])] = r
keys = sorted({(k[1],k[2],k[3]) for k in P})
rel = [P[(1.0,)+k+("rel50",)] for k in keys]
b50 = [P[(1.0,)+k+("b50",)] for k in keys]
print("blocked_steps example rel50:", rel[0]["blocked_steps"][:8], "ranks", rel[0]["blocked_ranks"][:8])
print("blocked_steps example b50:", b50[0]["blocked_steps"][:8])
mx_rel = max((max(r["blocked_steps"]) if r["blocked_steps"] else -1) for r in rel)
print("rel50 max blocked step index:", mx_rel, " count of steps >=5 across rows:", sum(1 for r in rel for s in r["blocked_steps"] if (s if isinstance(s,int) else s[0] if isinstance(s,(list,tuple)) else s) >=5))
print("rel50 mean blocked:", np.mean([len(r["blocked_steps"]) for r in rel]), " b50 mean blocked:", np.mean([len(r["blocked_steps"]) for r in b50]))
print("rel50 share:", np.mean([len(r["blocked_steps"]) for r in rel])/np.mean([len(r["blocked_steps"]) for r in b50]))
# rel50 blocks: how many steps are blocked per row in release mode (should be <=5)
print("rel50 rows with >5 blocked:", sum(len(r["blocked_steps"])>5 for r in rel))
print("rel50 blocked count distribution", collections.Counter(len(r["blocked_steps"]) for r in rel))
# rel50 vs b0: how many answers differ
b0 = [P[(1.0,)+k+("b0",)] for k in keys]
print("rel50 token-identical to b0:", sum(a["token_ids"]==b["token_ids"] for a,b in zip(rel,b0)), "of", len(keys))
print("b50 token-identical to b0:", sum(a["token_ids"]==b["token_ids"] for a,b in zip(b50,b0)), "of", len(keys))
# sign test: ended vs induced for rel50 and b50 at dose 1
from math import comb
def sign_p(k,n):
    p=sum(comb(n,i) for i in range(0,min(k,n-k)+1))/2**n; return min(1,2*p)
for ban in ("rel50","b50","b5","b200"):
    rs=[P[(1.0,)+k+(ban,)] for k in keys]
    ind=sum((not b["L"]) and a["L"] for a,b in zip(rs,b0)); end=sum(b["L"] and (not a["L"]) for a,b in zip(rs,b0))
    print(f"dose1 {ban}: induced {ind} ended {end} net {ind-end} exact sign p={sign_p(ind,ind+end):.3f}")
# with batch-noise check: Gate 8 says 0/384 differ on b0 repeat
# dose-1 b0 vs b50 length/EOS
print("== length and EOS")
def lbin(t): return "<=48" if t<=48 else ("49-56" if t<=56 else ("57-64" if t<=64 else "65+"))
for ban in ("b0","b50","b200","rel50"):
    rs=[P[(1.0,)+k+(ban,)] for k in keys]
    print(ban, "mean T", np.mean([r["T"] for r in rs]).round(2), "EOS", np.mean([r["E"] for r in rs]).round(3), "loop", np.mean([r["L"] for r in rs]).round(3),
          "n_T==80", sum(r["T"]>=80 for r in rs))
# loop rate by length bin, b0 vs b50 at dose 1 (all 384 units)
for lab in ("<=48","49-56","57-64","65+"):
    out=[]
    for ban in ("b0","b50"):
        rs=[P[(1.0,)+k+(ban,)] for k in keys if lbin(P[(1.0,)+k+(ban,)]["T"])==lab]
        out.append((ban,len(rs), round(np.mean([r["L"] for r in rs]),3) if rs else None))
    print(lab, out)
# the 80-token cap: answers hitting cap
print("cap hits b0", sum(P[(1.0,)+k+('b0',)]['T']>=80 for k in keys), "b50", sum(P[(1.0,)+k+('b50',)]['T']>=80 for k in keys))
print("EOS-ended loops at b50 dose1:", sum(P[(1.0,)+k+('b50',)]['L'] and P[(1.0,)+k+('b50',)]['E'] for k in keys), "of", sum(P[(1.0,)+k+('b50',)]['L'] for k in keys))
print("EOS-ended loops at b0 dose1:", sum(P[(1.0,)+k+('b0',)]['L'] and P[(1.0,)+k+('b0',)]['E'] for k in keys), "of", sum(P[(1.0,)+k+('b0',)]['L'] for k in keys))
