#!/usr/bin/env python3
"""Part 167 exploratory recompute, and where the non-special think/tool tokens sit in Part 168 lists."""
import json, re, sys
from collections import Counter
from pathlib import Path
from collections import Counter as C
ROOT = Path.cwd(); R = ROOT / "results/part168"
sys.path[:0] = [str(ROOT), str(ROOT / "part86")]
from part165 import score165
WORD = re.compile(r"[\w]+(?:['’][\w]+)*")
def jl(p): return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
def loop(t):
    ws = [w.casefold() for w in WORD.findall(t or "")]
    g = [tuple(ws[i:i+3]) for i in range(len(ws)-2)]
    return bool(g) and max(C(g).values()) >= 3
tok = json.loads((R/"tokens.json").read_text(encoding="utf-8"))
bad = {t for t, s in tok.items() if (s.startswith("<") and s.endswith(">")) or "<|" in s or "|>" in s}
print("special-looking token ids in tokens.json:", {t: tok[t] for t in sorted(bad, key=int)})
rows = jl(R/"rows.jsonl"); masses = jl(R/"masses.jsonl")
where = Counter()
for r in rows:
    for name in ("own_ban_ids", "rank_51_200_ids", "top_ids"):
        for t in r[name]:
            if str(int(t)) in bad and name != "top_ids":
                where[(r["arm"], name, tok[str(int(t))])] += 1
for m in masses:
    for t in m["mass_ban_ids"]:
        if str(int(t)) in bad:
            where[(m["arm"], "mass_ban_ids", tok[str(int(t))])] += 1
print("occurrences in own/mass lists (arm, list, token):", dict(where))
# Part 167 exploratory: n+ none, pooled broken / loop
p167 = jl(ROOT/"results/part167/answers.jsonl")
bat = [r for r in p167 if r.get("stage") == "battery"]
print("part167 battery keys:", sorted(bat[0].keys()) if bat else None, len(bat))
nn = [r for r in bat if r.get("arm") == "n+" and r.get("ban") == "none"]
print("part167 n+ none n:", len(nn), "broken", round(sum(bool(score165.primary_broken(r["answer"])) for r in nn)/len(nn), 4),
      "loop", round(sum(loop(r["answer"]) for r in nn)/len(nn), 4))
nn2 = [r for r in nn if int(r["question_index"]) != 13]
print("  excluding 13: n", len(nn2), "broken", round(sum(bool(score165.primary_broken(r["answer"])) for r in nn2)/len(nn2), 4),
      "loop", round(sum(loop(r["answer"]) for r in nn2)/len(nn2), 4))
for ban in ("none", "own", "matched"):
    for arm in ("n+", "k4"):
        xs = [r for r in bat if r.get("arm") == arm and r.get("ban") == ban]
        if xs:
            print(f"  part167 {arm} {ban}: n {len(xs)} loop {round(sum(loop(r['answer']) for r in xs)/len(xs),4)}")
