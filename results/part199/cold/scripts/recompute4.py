# Part 4: (a) does length explain the eager-vs-SDPA loop gap? stratify dose-1 push rows by token_count.
# (b) sensitivity of the "no end punctuation" count to emoji endings.
import re, sys
from collections import defaultdict
sys.path[:0] = ["/home/elisepc/green/worktrees/ab-part199/results/part199/cold/scripts"]
sys.dont_write_bytecode = True
from recompute import jl, loop, broken, P170, P199
pub = jl(P170 / "answers.jsonl"); ev = jl(P199 / "answers.jsonl")
def d1(rows):
    return [r for r in rows if r["stage"]=="push" and r["dose"]==1.0 and not r.get("repeat")]
bins = [(0,40),(40,60),(60,80),(80,81)]
def binname(t):
    for lo,hi in bins:
        if lo <= t < hi: return f"[{lo},{hi})"
for label, rows in (("SDPA", d1(pub)), ("EAGER", d1(ev))):
    agg = defaultdict(lambda: [0,0])
    for r in rows:
        b = binname(r["token_count"]); agg[b][0] += loop(r["answer"]); agg[b][1] += 1
    print(label, {k: (v[0], v[1], round(v[0]/v[1],3)) for k,v in sorted(agg.items())})
# Same stratification for the no-push identity+Q96 rows (all bans), for a second look
def npush(rows):
    return [r for r in rows if r["stage"]=="no_push"]
for label, rows in (("SDPA", npush(pub)), ("EAGER", npush(ev))):
    agg = defaultdict(lambda: [0,0])
    for r in rows:
        b = binname(r["token_count"]); agg[b][0] += loop(r["answer"]); agg[b][1] += 1
    print("no-push", label, {k: (v[0], v[1], round(v[0]/v[1],3)) for k,v in sorted(agg.items())})

# (b) emoji sensitivity for the no-punctuation count, dose-1 b0/b50/b200 pushed rows that ended before 80
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿️‍\s]+$")
def strict(t): return t.rstrip()[-1:] if t.strip() else ""
def punct_after_strip(t):
    t2 = EMOJI.sub("", t.rstrip())
    return t2[-1:] in list('.!?"”’)*`:;')
for label, rows in (("SDPA", d1(pub)), ("EAGER", d1(ev))):
    for ban in ("b0","b50","b200"):
        sel = [r for r in rows if r["ban"]==ban and r.get("ended_eos_before_80")]
        nopunct = sum(1 for r in sel if not punct_after_strip(r["answer"]))
        emoji_end = sum(1 for r in sel if EMOJI.search(r["answer"].rstrip()) and punct_after_strip(r["answer"]))
        print(label, ban, "early-stop", len(sel), "no-punct (emoji stripped first)", nopunct, "ends in emoji after punct", emoji_end)
