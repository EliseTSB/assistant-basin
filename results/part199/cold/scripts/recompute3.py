# Part 3: Stage 1 lists (order vs set), same-population 170.1, eos/token consistency, mid-phrase samples.
import json, sys
from collections import Counter, defaultdict
sys.path[:0] = ["/home/elisepc/green/worktrees/ab-part199/results/part199/cold/scripts"]
sys.dont_write_bytecode = True
from recompute import jl, loop, broken, ends_punct, P170, P199
pub = jl(P170 / "answers.jsonl"); ev = jl(P199 / "answers.jsonl")
s0p = {r["prompt_id"]: r for r in pub if r["stage"] == "stage0"}
s0e = {r["prompt_id"]: r for r in ev if r["stage"] == "stage0"}
Lp = {r["prompt_id"]: r for r in jl(P170 / "lists.jsonl")}
Le = {r["prompt_id"]: r for r in jl(P199 / "lists.jsonl")}
print("list keys:", sorted(Le[next(iter(Le))].keys()))
print("n lists", len(Lp), len(Le))
unchanged = [pid for pid in s0e if s0p[pid]["token_ids"][:40] == s0e[pid]["token_ids"][:40]]
changed = [pid for pid in s0e if pid not in set(unchanged)]
print("unchanged", len(unchanged), "changed", len(changed))
def agree(group, name):
    res = {"order": 0, "set": 0, "n": len(group), "minshared": 999}
    for pid in group:
        for L in ("b5", "b50", "b200"):
            pass
    return res
for path, lab in ((unchanged, "unchanged"), (changed, "changed")):
    for L in ("b5", "b50", "b200"):
        order = sum(Le[p][L] == Lp[p][L] for p in path)
        sset = sum(set(Le[p][L]) == set(Lp[p][L]) for p in path)
        shared = [len(set(Le[p][L]) & set(Lp[p][L])) for p in path]
        print(f"{lab:9s} {L:5s} n={len(path)} same_order={order} same_set={sset} shared_min={min(shared)} shared_median={sorted(shared)[len(shared)//2]} len_min={min(len(Le[p][L]) for p in path)} len_max={max(len(Le[p][L]) for p in path)}")
# registered 199.2 count: all three lists equal in order / as sets
allord = sum(all(Le[p][L] == Lp[p][L] for L in ("b5","b50","b200")) for p in Le)
allset = sum(all(set(Le[p][L]) == set(Lp[p][L]) for L in ("b5","b50","b200")) for p in Le)
print("prompts with all three lists equal in order:", allord, " all three equal as sets:", allset)
# fraction of b50 tokens that differ, all prompts (set difference)
diffs = [len(set(Le[p]["b50"]) ^ set(Lp[p]["b50"])) / 2 for p in Le]
print("b50 tokens differing (mean over prompts):", round(sum(diffs)/len(diffs),2), " prompts with >=1 differing b50 token:", sum(d>0 for d in diffs), "fraction of prompts:", round(sum(d>0 for d in diffs)/len(diffs),3))
# Ordered equality for b200 on unchanged path: how many prompts have b200 as list identical?
print("b50 mean share of set differing:", round(sum(diffs)/len(diffs)/50,3))

# Same-population 170.1
frozen = set(json.load(open(P170 / "stage0.json"))["frozen_prompt_ids"])
ident_pub_ids = [p for p in s0p if p.startswith(("p8:", "m96:")) and p not in frozen]
print("eligible identity prompts (published):", len(ident_pub_ids), " eligible Q96:", len([p for p in s0p if p.startswith('q96:') and p not in frozen]))
def nop(rows, ids, ban):
    sel = [r for r in rows if r["stage"]=="no_push" and r["dose"]==0.0 and r["ban"]==ban and r["prompt_id"] in ids and r["group"] in ("p8","m96")]
    return sum(loop(r["answer"]) for r in sel), len(sel)
def dose1(rows, ban):
    sel = [r for r in rows if r["stage"]=="push" and r["dose"]==1.0 and r["ban"]==ban and not r.get("repeat")]
    return sum(loop(r["answer"]) for r in sel)/len(sel)
for label, rows in (("published", pub), ("eager", ev)):
    ids = set(ident_pub_ids)
    if label == "eager":
        ids = {r["prompt_id"] for r in rows if r["stage"]=="no_push" and r["group"] in ("p8","m96")}
    l0, n0 = nop(rows, ids, "b0"); l50, n50 = nop(rows, ids, "b50")
    push_rise = dose1(rows, "b50") - dose1(rows, "b0")
    print(f"{label}: identity n={len(ids)} b0 {l0}/{n0} b50 {l50}/{n50}  nopush_rise={(l50/n50 - l0/n0):.4f}  push_rise={push_rise:.4f}  ratio={(l50/n50 - l0/n0)/push_rise:.3f}")
# eager 170.1 on the published 92 population explicitly
ids92 = set(ident_pub_ids)
l0, n0 = nop(ev, ids92, "b0"); l50, n50 = nop(ev, ids92, "b50")
pr = dose1(ev,"b50")-dose1(ev,"b0")
print("eager on published 92 ids: b0", l0, "/", n0, "b50", l50, "/", n50, "rise", round(l50/n50-l0/n0,4), "ratio", round((l50/n50-l0/n0)/pr,3))
# 170.1 with registered population on eager (all identity, 104)
l0, n0 = nop(ev, {r["prompt_id"] for r in ev if r["stage"]=="no_push" and r["group"] in ("p8","m96")}, "b0")
print("eager 104 b0 loops", l0, n0)
# 170.2 on eager: Q96 no-push rise over dose 1 push rise
def q96(rows, ban):
    sel=[r for r in rows if r["stage"]=="no_push" and r["dose"]==0.0 and r["ban"]==ban and r["group"]=="q96"]
    return sum(loop(r["answer"]) for r in sel)/len(sel)
print("170.2 eager Q96 rise", round(q96(ev,"b50")-q96(ev,"b0"),4), "ratio", round((q96(ev,"b50")-q96(ev,"b0"))/pr,3))
print("170.2 published Q96 rise", round(q96(pub,"b50")-q96(pub,"b0"),4))

# eos vs token_count consistency
for label, rows in (("pub", pub), ("eager", ev)):
    bad = sum(1 for r in rows if bool(r.get("ended_eos_before_80")) != (r["token_count"] < 80))
    print(label, "rows where ended_eos_before_80 != (token_count<80):", bad, "of", len(rows))

# Mid-phrase samples: dose-1 b50 published and eager answers ending without punctuation
import random
random.seed(1)
for label, rows in (("pub d1 b50", [r for r in pub if r["stage"]=="push" and r["dose"]==1.0 and r["ban"]=="b50" and not r.get("repeat") and r.get("ended_eos_before_80") and not ends_punct(r["answer"])]),
                    ("eager d1 b50", [r for r in ev if r["stage"]=="push" and r["dose"]==1.0 and r["ban"]=="b50" and not r.get("repeat") and r.get("ended_eos_before_80") and not ends_punct(r["answer"])])):
    print(label, "n", len(rows))
    for r in random.sample(rows, min(5, len(rows))):
        print("   ...", repr(r["answer"][-90:]))
