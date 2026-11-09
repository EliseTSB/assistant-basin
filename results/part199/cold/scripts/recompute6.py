# Part 6: (a) R3 chunk shapes from load.json; (b) mid-phrase counts under Part 191's stated rule vs Part 199's regex.
import json, re, sys
from collections import Counter
sys.path[:0] = ["/home/elisepc/green/worktrees/ab-part199/results/part199/cold/scripts"]
sys.dont_write_bytecode = True
from recompute import jl, P170, P199
d = json.load(open(P199 / "load.json"))
calls = d["loads"][0]["calls"]
pushed = [c for c in calls if c.get("pushed")]
nopush = [c for c in calls if not c.get("pushed")]
print("calls", len(calls), "pushed", len(pushed), "pushed rows", sum(c["rows"] for c in pushed))
print("pushed chunk sizes", Counter(tuple(c["chunks"]) if isinstance(c["chunks"], list) else c["chunks"] for c in pushed).most_common(3))
bad = [c for c in pushed if not (c["chunks"] and all(x == 8 for x in c["chunks"]) and sum(c["chunks"]) == c["rows"])]
print("pushed calls whose chunks are not all 8 or do not sum to rows:", len(bad))
print("zero_write calls:", sum(1 for c in calls if c.get("zero_write")), " complete flags:", Counter(c.get("complete") for c in calls))
print("nopush chunk sizes", Counter(max(c["chunks"]) if c["chunks"] else 0 for c in nopush))

# Part 191 rule: answer ends on a letter, digit, comma, hyphen, colon or semicolon
ev = jl(P199 / "answers.jsonl"); pub = jl(P170 / "answers.jsonl")
def p191(t):
    t = (t or "").rstrip()
    return bool(t) and bool(re.search(r"[A-Za-z0-9,\-:;]$", t))
for label, rows in (("SDPA", pub), ("EAGER", ev)):
    for ban in ("b0", "b50", "b200"):
        early = [r for r in rows if r["stage"]=="push" and r["dose"]==1.0 and not r.get("repeat") and r["ban"]==ban and r["ended_eos_before_80"]]
        print(label, ban, "early", len(early), "p191-rule mid-phrase", sum(p191(r["answer"]) for r in early))
# Whole-run counts with the Part 191 rule (all stages), to compare with Part 191's 230/838 and 2/588 (its own answer set, not this one)
