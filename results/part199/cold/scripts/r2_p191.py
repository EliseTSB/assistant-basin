# Round-2: Part 191 open-ending rule counts on dose-1 early stops, plus the Part 191 comparison figures as stated.
import json, re, sys
from pathlib import Path
ROOT = Path("/home/elisepc/green/worktrees/ab-part199")
sys.dont_write_bytecode = True
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def p191(t):
    t = (t or "").rstrip()
    return bool(t) and bool(re.search(r"[A-Za-z0-9,\-:;]$", t))
for lab, p in (("SDPA", ROOT / "results/part170/answers.jsonl"), ("EAGER", ROOT / "results/part199/answers.jsonl")):
    rows = jl(p)
    for ban in ("b0", "b50", "b200"):
        early = [r for r in rows if r["stage"] == "push" and float(r["dose"]) == 1.0 and r["ban"] == ban and r["ended_eos_before_80"]]
        print(lab, ban, "early", len(early), "mid-phrase (Part 191 rule)", sum(p191(r["answer"]) for r in early))
