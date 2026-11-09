# Round-2 recomputation for Part 199 folds. Reads repo files in place; writes nothing into the repo.
import json, sys, statistics
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path("/home/elisepc/green/worktrees/ab-part199")
sys.path[:0] = [str(ROOT), str(ROOT / "part170")]
sys.dont_write_bytecode = True
from part166 import score166

P199 = ROOT / "results/part199"
P170 = ROOT / "results/part170"

def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

ev = jl(P199 / "answers.jsonl")
pub = jl(P170 / "answers.jsonl")

def key(r):
    return (r["stage"], r.get("prompt_id"), r.get("arm"), r.get("dose"), r.get("ban"), r.get("kind"))

# ---- A. Stage 0 plain paths
def plain(rows):
    return {r["prompt_id"]: r for r in rows if r["stage"] == "stage0"}
pp, pe = plain(pub), plain(ev)
ids = sorted(pe)
same40 = []; diffs = []; same_whole = 0
for pid in ids:
    a, b = pp[pid]["token_ids"], pe[pid]["token_ids"]
    plen = len(pe[pid]["prompt_ids"])
    same40.append(a[:40] == b[:40])
    if a == b:
        same_whole += 1
        continue
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    diffs.append((plen + i, i))
print("A stage0 prompts", len(ids), "same first 40", sum(same40), "same whole", same_whole,
      "differing", len(diffs))
keys = sorted(k for k, _ in diffs)
print("A first-diff keys min", keys[0], "median", statistics.median(keys), "max", keys[-1],
      "at or below 64:", sum(k <= 64 for k in keys))

# ---- B. Stage 1 lists
lp = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(P170 / "lists.jsonl", encoding="utf-8") if l.strip()}
le = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(P199 / "lists.jsonl", encoding="utf-8") if l.strip()}
pids = sorted(le)
print("B lists prompts", len(pids), "shared with published", len(set(pids) & set(lp)))
path_same = {pid: same40[ids.index(pid)] for pid in pids}
for f in ("b5", "b50", "b200"):
    tot_diff = []; set_eq = 0; ord_eq = 0; set_diff_n = 0
    for pid in pids:
        A = list(map(int, lp[pid][f])); Bl = list(map(int, le[pid][f]))
        ord_eq += A == Bl
        sA, sB = set(A), set(Bl)
        set_eq += sA == sB
        d = len(sA - sB) + len(sB - sA)
        tot_diff.append(len(sA - sB))
        set_diff_n += (sA != sB)
    print(f"B {f}: same order {ord_eq}, same set {set_eq}, set differs {set_diff_n}, "
          f"mean |A minus B| over all 200 {statistics.mean(tot_diff):.3f}, "
          f"over differing {statistics.mean([t for t in tot_diff if t]) if any(tot_diff) else 0:.3f}")
    for lab, flag in (("same path", True), ("changed path", False)):
        sel = [pid for pid in pids if path_same[pid] == flag]
        so = sum(list(map(int, lp[p][f])) == list(map(int, le[p][f])) for p in sel)
        ss = sum(set(map(int, lp[p][f])) == set(map(int, le[p][f])) for p in sel)
        sh = [len(set(map(int, lp[p][f])) & set(map(int, le[p][f]))) for p in sel]
        print(f"   {f} {lab}: n={len(sel)} ordered {so} set {ss} shared min {min(sh)} median {statistics.median(sh)}")
allthree = sum(all(set(map(int, lp[p][f])) == set(map(int, le[p][f])) for f in ("b5", "b50", "b200")) for p in pids)
print("B all three equal as sets:", allthree)

# ---- C. Dose-1 push cells: where loops end (SDPA vs eager)
def push(rows, ban):
    return [r for r in rows if r["stage"] == "push" and float(r["dose"]) == 1.0 and r["ban"] == ban]
for lab, rows in (("SDPA", pub), ("EAGER", ev)):
    for ban in ("b0", "b50", "b200"):
        sel = push(rows, ban)
        loops = [r for r in sel if score166.loop_phrase(r["answer"] or "")]
        early = [r for r in sel if r["ended_eos_before_80"]]
        print(f"C {lab} {ban}: rows {len(sel)} early {len(early)} loops {len(loops)} loops_early {sum(bool(r['ended_eos_before_80']) for r in loops)}")

# ---- D. Dose-0.25 b0 loops and where they formed is in onset file; here just count loops on SDPA/eager
for lab, rows in (("SDPA", pub), ("EAGER", ev)):
    sel = [r for r in rows if r["stage"] == "push" and float(r["dose"]) == 0.25 and r["ban"] == "b0"]
    print(f"D {lab} dose0.25 b0 rows {len(sel)} loops {sum(bool(score166.loop_phrase(r['answer'] or '')) for r in sel)}")

# ---- E. 170.1 ratio on 92 shared identity prompts (exclude frozen M96 indices from published Stage 0)
frozen_m96 = {3, 24, 26, 27, 48, 50, 57, 58, 59, 65, 80, 94}
frozen_q96 = {0, 12, 14, 22, 23, 33, 51}
def nopush_loops(rows, ban, keep):
    out = 0
    for r in rows:
        if r["stage"] != "no_push" or r["ban"] != ban or r["group"] not in ("p8", "m96"):
            continue
        if r["group"] == "m96" and r["question_index"] in frozen_m96:
            continue
        if keep is not None and r["prompt_id"] not in keep:
            continue
        out += bool(score166.loop_phrase(r["answer"] or ""))
    return out
ident_ids = {r["prompt_id"] for r in pub if r["stage"] == "stage0" and r["group"] in ("p8", "m96")
             and not (r["group"] == "m96" and r["question_index"] in frozen_m96)}
print("E identity shared prompts", len(ident_ids))
for lab, rows in (("SDPA", pub), ("EAGER", ev)):
    b0 = nopush_loops(rows, "b0", ident_ids); b50 = nopush_loops(rows, "b50", ident_ids)
    print(f"E {lab} identity-92 no-push b0 {b0} b50 {b50}")
