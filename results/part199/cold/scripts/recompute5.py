# Part 5: within-ban length strata (dose-1 push, non-repeat), SDPA vs eager; and the dose-1 b50 by-length check.
import sys
from collections import defaultdict
sys.path[:0] = ["/home/elisepc/green/worktrees/ab-part199/results/part199/cold/scripts"]
sys.dont_write_bytecode = True
from recompute import jl, loop, P170, P199
pub = jl(P170 / "answers.jsonl"); ev = jl(P199 / "answers.jsonl")
def d1(rows, ban):
    return [r for r in rows if r["stage"]=="push" and r["dose"]==1.0 and not r.get("repeat") and r["ban"]==ban]
def strata(rows):
    agg = defaultdict(lambda: [0,0])
    for r in rows:
        t = r["token_count"]; b = "<80" if t < 80 else "=80(cap)"
        agg[b][0] += loop(r["answer"]); agg[b][1] += 1
    return {k: f"{v[0]}/{v[1]}={v[0]/v[1]:.2f}" for k,v in sorted(agg.items())}
for ban in ("b0","b50","b200"):
    print(ban, "SDPA", strata(d1(pub,ban)))
    print(ban, "EAGER", strata(d1(ev,ban)))
# share of loops that ran to the cap, per run and ban
for label, rows in (("SDPA", pub), ("EAGER", ev)):
    for ban in ("b0","b50"):
        sel = d1(rows, ban); L = [r for r in sel if loop(r["answer"])]
        print(label, ban, "loops", len(L), "of which at cap", sum(r["token_count"]>=80 for r in L))
