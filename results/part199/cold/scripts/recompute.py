# Cold-check recomputation for Part 199 README. Reads repo files in place; writes nothing into the repo.
import json, sys, math
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path("/home/elisepc/green/worktrees/ab-part199")
sys.path[:0] = [str(ROOT), str(ROOT / "part170")]
sys.dont_write_bytecode = True
from part166 import score166
from part165 import score165

P199 = ROOT / "results/part199"
P170 = ROOT / "results/part170"

def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def loop(text):
    return score166.loop_phrase(text or "") is not None

def broken(text):
    return bool(score165.primary_broken(text or ""))

def ends_punct(text):
    t = (text or "").rstrip()
    return bool(t) and t[-1] in '.!?"”’)*`:;'

def cells(rows, stage, dose):
    out = defaultdict(list)
    for r in rows:
        if r.get("stage") != stage or r.get("dose") != dose:
            continue
        if stage == "push" and r.get("repeat"):
            continue
        out[(r.get("group"), r.get("ban"))].append(r)
    return out

def summarize(name, rs):
    n = len(rs)
    lp = sum(loop(r["answer"]) for r in rs)
    br = sum(broken(r["answer"]) for r in rs)
    eos = sum(bool(r.get("ended_eos_before_80")) for r in rs)
    mean = sum(r["token_count"] for r in rs) / n if n else float("nan")
    no_punct = sum(1 for r in rs if r.get("ended_eos_before_80") and not ends_punct(r["answer"]))
    return f"{name:28s} n={n:4d} loops={lp:4d} ({100*lp/n:5.1f}%) broken={br:4d} ({100*br/n:5.1f}%) mean_tok={mean:6.2f} eos<80={eos:4d} eos_no_punct={no_punct:4d}"

if __name__ == "__main__":
    pub = jl(P170 / "answers.jsonl")
    ev = jl(P199 / "answers.jsonl")
    print("rows: published", len(pub), "eager", len(ev))
    print("stage counts published", Counter(r["stage"] for r in pub))
    print("stage counts eager    ", Counter(r["stage"] for r in ev))
    for label, rows in (("PUBLISHED SDPA", pub), ("EAGER", ev)):
        print("==", label, "dose-1 push (non-repeat), by ban")
        c = cells(rows, "push", 1.0)
        for k in sorted(c, key=lambda k: (k[0], k[1])):
            print(summarize(f"{k[0]} {k[1]}", c[k]))
        print("==", label, "dose-0.25 push b0 loops/broken")
        c = cells(rows, "push", 0.25)
        for k in sorted(c, key=lambda k: (k[0], k[1])):
            if k[1] == "b0":
                print(summarize(f"{k[0]} {k[1]} d0.25", c[k]))
        print("==", label, "no-push, dose 0")
        c = cells(rows, "no_push", 0.0)
        for k in sorted(c, key=lambda k: (k[0], k[1])):
            print(summarize(f"{k[0]} {k[1]}", c[k]))
