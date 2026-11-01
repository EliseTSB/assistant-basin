import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
def nb(p,b): return [r for r in ANS if r["stage"]=="no_push" and r["prompt_id"]==p and r["ban"]==b][0]["L"]
ident_ids = [p for p in sorted({r["prompt_id"] for r in ANS if r["stage"]=="no_push"}) if (p.startswith("p8:") or p.startswith("m96:")) and p not in FROZEN]
q_ids = [p for p in sorted({r["prompt_id"] for r in ANS if r["stage"]=="no_push"}) if p.startswith("q96:") and p not in FROZEN]
def rise(ids,b="b50"):
    if not ids: return float("nan"),0
    return float(np.mean([nb(p,b) for p in ids]) - np.mean([nb(p,"b0") for p in ids])), len(ids)
r1,n1=rise(ident_ids); print(f"170.1 identity eligible rise {100*r1:+.2f} n={n1}; ratio {r1/0.1875*100:.1f}%")
for lab, drop in [("drop p8 (all 8 phrasings)", lambda p: p.startswith("p8:")), ("drop m96 maker", lambda p: p.startswith("m96:") and int(p.split(':')[1])<48), ("drop m96 identity-kind", lambda p: p.startswith("m96:") and int(p.split(':')[1])>=48)]:
    ids=[p for p in ident_ids if not drop(p)]; r,n=rise(ids); print(f"  {lab}: rise {100*r:+.2f} n={n} ratio {r/0.1875*100:.1f}%")
for ph in PHR:
    ids=[p for p in ident_ids if p!=ph]; r,n=rise(ids); print(f"  drop {ph}: rise {100*r:+.2f} ratio {r/0.1875*100:.1f}%")
r2,n2=rise(q_ids); print(f"170.2 Q96 eligible rise {100*r2:+.2f} n={n2} ratio {r2/0.1875*100:.1f}%")
for lab,f in [("drop fact", lambda p:int(p.split(':')[1])<48),("drop how_to", lambda p:int(p.split(':')[1])>=48)]:
    ids=[p for p in q_ids if not f(p)]; r,n=rise(ids); print(f"  {lab}: rise {100*r:+.2f} n={n} ratio {r/0.1875*100:.1f}%")
# Q96 loops under b50 by kind
print("Q96 b50 loops:", [p for p in q_ids if nb(p,'b50')])
print("identity b50 loops:", [p for p in ident_ids if nb(p,'b50')])
