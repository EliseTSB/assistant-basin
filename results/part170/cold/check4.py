import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import random
P = {}
for r in ANS:
    if r["stage"]=="push": P[(r["dose"], r["prompt_id"], r["layer"], r["slot"], r["ban"])] = r
keys = sorted({(k[1],k[2],k[3]) for k in P})
ind = [k for k in keys if (not P[(1.0,)+k+("b0",)]["L"]) and P[(1.0,)+k+("b50",)]["L"]]
print("dose1 b50 induced:", len(ind))
rng = random.Random(170)
samp = rng.sample(ind, 20)
for k in samp:
    r = P[(1.0,)+k+("b50",)]; g,n = top_trigram(r["answer"])
    print(f"-- {k} trigram={g} n={n} T={r['T']} EOS={r['E']} blocked={len(r['blocked_steps'])}")
    print("   ", r["answer"][:300].replace("\n"," | "))
