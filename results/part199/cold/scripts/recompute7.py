# Part 7: decompose the dose-1 b0 -> b50 loop rise into (a) within-stop-type rate change and (b) change in the share of answers reaching the 80-token cap.
import sys
sys.path[:0] = ["/home/elisepc/green/worktrees/ab-part199/results/part199/cold/scripts"]
sys.dont_write_bytecode = True
from recompute import jl, loop, P170, P199
pub = jl(P170 / "answers.jsonl"); ev = jl(P199 / "answers.jsonl")
def stats(rows, ban):
    sel = [r for r in rows if r["stage"]=="push" and r["dose"]==1.0 and not r.get("repeat") and r["ban"]==ban]
    n = len(sel); out = {}
    for k, pred in (("cap", lambda r: r["token_count"] >= 80), ("early", lambda r: r["token_count"] < 80)):
        s = [r for r in sel if pred(r)]
        out[k] = (len(s), sum(loop(r["answer"]) for r in s))
    return n, out
for label, rows in (("EAGER", ev), ("SDPA", pub)):
    n0, s0 = stats(rows, "b0"); n1, s1 = stats(rows, "b50")
    w0 = {k: s0[k][0]/n0 for k in s0}; w1 = {k: s1[k][0]/n1 for k in s1}
    r0 = {k: s0[k][1]/s0[k][0] if s0[k][0] else 0 for k in s0}; r1 = {k: s1[k][1]/s1[k][0] if s1[k][0] else 0 for k in s1}
    total = sum(w1[k]*r1[k] for k in w1) - sum(w0[k]*r0[k] for k in w0)
    within = sum(w0[k]*(r1[k]-r0[k]) for k in w0)
    comp = sum((w1[k]-w0[k])*r1[k] for k in w0)
    print(label, "b0 cap/early answers", s0["cap"][0], s0["early"][0], "loops", s0["cap"][1], s0["early"][1],
          "| b50 cap/early", s1["cap"][0], s1["early"][0], "loops", s1["cap"][1], s1["early"][1])
    print(f"  rates b0 cap {r0['cap']:.3f} early {r0['early']:.3f} | b50 cap {r1['cap']:.3f} early {r1['early']:.3f}")
    print(f"  rise {total:.3f} = within-stratum {within:.3f} + composition (share reaching cap) {comp:.3f}")
