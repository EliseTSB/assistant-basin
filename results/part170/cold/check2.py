import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from math import comb
def binom_two_sided(k, n):
    # exact McNemar: discordant pairs k (one direction) out of n total discordant
    if n == 0: return 1.0
    p = sum(comb(n, i) for i in range(0, min(k, n-k)+1)) / 2**n
    return min(1.0, 2*p)
# 1. plain vs in-batch no-push b0
nb0 = {r["prompt_id"]: r for r in ANS if r["stage"]=="no_push" and r["ban"]=="b0"}
diff = []; tab = collections.Counter()
for p, pr in PLAIN.items():
    b = nb0[p]
    if b["token_ids"] != pr["token_ids"]:
        diff.append(p)
    tab[("frozen" if p in FROZEN else "elig", "plainL" if pr["L"] else "plainNoL", "batchL" if b["L"] else "batchNoL")] += 1
print("stage0 plain vs no-push b0 token-different:", len(diff), "of", len(PLAIN))
print("answers identical (text) differ:", sum(nb0[p]["answer"]!=PLAIN[p]["answer"] for p in PLAIN))
for k,v in sorted(tab.items()): print("  ", k, v)
print("loop-label changes among differing rows:")
ch = collections.Counter()
for p in diff:
    ch[("frozen" if p in FROZEN else "elig", PLAIN[p]["L"], nb0[p]["L"], "broken", PLAIN[p]["B"], nb0[p]["B"])]+=1
for k,v in ch.items(): print("  ", k, v)
print("eligible prompts where batch b0 loops or breaks:", [p for p in PLAIN if p not in FROZEN and (nb0[p]["L"] or nb0[p]["B"])])
print("eligible prompts where batch b0 differs from plain:", sum(1 for p in PLAIN if p not in FROZEN and p in diff))
# token-level divergence positions
def first_div(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y: return i
    return min(len(a),len(b))
fd=[first_div(PLAIN[p]["token_ids"], nb0[p]["token_ids"]) for p in diff]
print("first divergence position among differing (median, min, max):", float(np.median(fd)) if False else sorted(fd)[len(fd)//2], min(fd), max(fd))
# 2. selection: rises on eligible vs all, with exact McNemar
def pairs(group_fn, ban):
    out=[]
    for p in sorted({r["prompt_id"] for r in ANS if r["stage"]=="no_push"}):
        if not group_fn(p): continue
        b0=[r for r in ANS if r["stage"]=="no_push" and r["prompt_id"]==p and r["ban"]=="b0"][0]
        bx=[r for r in ANS if r["stage"]=="no_push" and r["prompt_id"]==p and r["ban"]==ban][0]
        out.append((b0["L"], bx["L"]))
    return out
def summarize(name, fn, ban):
    pr = pairs(fn, ban)
    n=len(pr); b=sum(x for x,_ in pr); a=sum(y for _,y in pr)
    ind=sum((not x) and y for x,y in pr); end=sum(x and (not y) for x,y in pr)
    print(f"{name:28s} {ban:6s} n={n:3d} b0={b} list={a} rise={100*(a-b)/n:+.2f}pp induced={ind} ended={end} exactMcNemar p={binom_two_sided(ind, ind+end):.3f}")
ident = lambda p: p.startswith("p8:") or p.startswith("m96:")
q = lambda p: p.startswith("q96:")
for ban in ("b5","b50","b200","rel50"):
    summarize("identity eligible", lambda p: ident(p) and p not in FROZEN, ban)
    summarize("identity ALL", ident, ban)
    summarize("Q96 eligible", lambda p: q(p) and p not in FROZEN, ban)
    summarize("Q96 ALL", q, ban)
summarize("M96 eligible", lambda p: p.startswith("m96:") and p not in FROZEN, "b50")
summarize("M96 ALL", lambda p: p.startswith("m96:"), "b50")
summarize("P8", lambda p: p.startswith("p8:"), "b50")
summarize("P8", lambda p: p.startswith("p8:"), "b200")
# frozen-only rises
summarize("identity FROZEN only", lambda p: ident(p) and p in FROZEN, "b50")
summarize("Q96 FROZEN only", lambda p: q(p) and p in FROZEN, "b50")
# other50 on Q96
summarize("Q96 eligible", lambda p: q(p) and p not in FROZEN, "other50")
summarize("Q96 ALL", q, "other50")
# 3. filter sensitivity: select on in-batch b0 instead of plain
bat_ok = lambda p: (not nb0[p]["L"]) and (not nb0[p]["B"])
print("eligible by plain:", sum(1 for p in PLAIN if p not in FROZEN), "; eligible by batch b0:", sum(1 for p in PLAIN if bat_ok(p)))
for ban in ("b50",):
    summarize("identity, batch-b0 filter", lambda p: ident(p) and bat_ok(p), ban)
    summarize("Q96, batch-b0 filter", lambda p: q(p) and bat_ok(p), ban)
