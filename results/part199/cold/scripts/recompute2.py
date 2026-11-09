# Part 2: Stage 0 plain paths, Stage 1 lists, same-population 170.1, frozen counts. Reads in place only.
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path[:0] = ["/home/elisepc/green/worktrees/ab-part199/results/part199/cold/scripts"]
sys.dont_write_bytecode = True
from recompute import jl, loop, broken, ends_punct, P170, P199

pub = jl(P170 / "answers.jsonl"); ev = jl(P199 / "answers.jsonl")
s0p = {r["prompt_id"]: r for r in pub if r["stage"] == "stage0"}
s0e = {r["prompt_id"]: r for r in ev if r["stage"] == "stage0"}
print("stage0 prompts", len(s0p), len(s0e), "arms", Counter(r["arm"] for r in ev if r["stage"]=="stage0"))

# Plain-path agreement
same40 = same_all = 0
first_diff_keys = []; first_diff_idx = []
for pid in s0e:
    a = s0p[pid]["token_ids"]; b = s0e[pid]["token_ids"]
    if a[:40] == b[:40]: same40 += 1
    if a == b: same_all += 1
    else:
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        first_diff_idx.append(i)
        first_diff_keys.append(len(s0e[pid]["prompt_ids"]) + i)   # keys when generating token i (prompt + i prior tokens)
print("identical first 40 tokens:", same40, " identical whole:", same_all, " differing:", len(s0e)-same_all)
fd = sorted(first_diff_keys)
import statistics
print("first-diff keys (prompt+i): min", fd[0], "max", fd[-1], "median", statistics.median(fd), "<=64:", sum(k <= 64 for k in fd), "<=63:", sum(k<=63 for k in fd))
fd2 = sorted(k+1 for k in fd)
print("first-diff keys (prompt+i+1): min", fd2[0], "max", fd2[-1], "median", statistics.median(fd2), "<=64:", sum(k <= 64 for k in fd2))
# Changed-path diffs within the first 40 tokens
changed = [pid for pid in s0e if s0p[pid]["token_ids"][:40] != s0e[pid]["token_ids"][:40]]
print("changed 40-token path prompts:", len(changed))
print("prompt lengths (prompt_ids) min/max:", min(len(r["prompt_ids"]) for r in ev if r["stage"]=="stage0"), max(len(r["prompt_ids"]) for r in ev if r["stage"]=="stage0"))

# Frozen counts
s0pj = json.load(open(P170 / "stage0.json")); s0ej = json.load(open(P199 / "stage0.json"))
print("published frozen:", {k: len(v) for k, v in s0pj["frozen_indices"].items()}, "eager frozen:", {k: len(v) for k, v in s0ej["frozen_indices"].items()})
print("published stage0 counts:", s0pj["counts"])
print("eager stage0 counts:", s0ej["counts"])

# Frozen prompts' plain answers: did the published ones loop/break, and what does the eager plain answer look like?
frozen_ids = s0pj["frozen_prompt_ids"]
print("published frozen ids:", frozen_ids)
for pid in frozen_ids:
    p = s0p[pid]; e = s0e[pid]
    print(pid, "pub loop", loop(p["answer"]), "pub broken", broken(p["answer"]), "| eager loop", loop(e["answer"]), "eager broken", broken(e["answer"]), "| eager tokens", e["token_count"], "eager ends", e["answer"][-60:].replace("\n"," "))
print("eager plain loops/broken over all 200:", sum(loop(r["answer"]) for r in s0e.values()), sum(broken(r["answer"]) for r in s0e.values()))
print("published plain loops/broken over all 200:", sum(loop(r["answer"]) for r in s0p.values()), sum(broken(r["answer"]) for r in s0p.values()))
