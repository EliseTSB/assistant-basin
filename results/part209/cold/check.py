import json, sys, collections, random
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

R = Path("/home/elisepc/green/worktrees/ab-part203/results/part209")

def load_labels(name):
    last = {}
    skipped = 0
    for line in open(R / name):
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        if "error" in d or not d.get("posture"):
            skipped += 1
            continue
        last[d["row_id"]] = d
    return last, skipped

def cat(d):
    return "unreadable" if d.get("unreadable") else d["posture"]

ds, ds_skip = load_labels("labels-ds.jsonl")
hk, hk_skip = load_labels("labels-haiku.jsonl")
print("ds rows", len(ds), "skipped", ds_skip, "| haiku rows", len(hk), "skipped", hk_skip)

TOLD = {"TOLD-SIT": "sitting", "TOLD-STAND": "standing", "TOLD-LIE": "lying"}
for stage in ("qwen-chat", "instruct"):
    r2 = [json.loads(l) for l in open(R / f"r2-{stage}.jsonl") if l.strip()]
    print(f"\n=== {stage}  n={len(r2)}")
    byarm = collections.defaultdict(list)
    for row in r2:
        byarm[row["arm"]].append(row)
    for arm in sorted(byarm):
        rows = byarm[arm]
        c = collections.Counter()
        named = 0
        for row in rows:
            d = ds.get(row["row_id"])
            if d is None:
                c["MISSING"] += 1; continue
            k = cat(d); c[k] += 1
            if k in ("sitting", "standing", "lying", "other"): named += 1
        denom = named
        share = c["sitting"] / denom if denom else float("nan")
        n = sum(c.values())
        print(f"  {arm:10s} n={n} share={share:.4f} none={c['none']/n:.4f} counts={dict(sorted(c.items()))}")
    # compliance
    num = den = 0
    for arm, told in TOLD.items():
        for row in byarm.get(arm, []):
            d = ds.get(row["row_id"])
            if d is None: continue
            den += 1
            if cat(d) == told: num += 1
    print(f"  compliance {num}/{den} = {num/den:.4f}")

# kappa
common = sorted(set(ds) & set(hk))
a = [cat(ds[r]) for r in common]
b = [cat(hk[r]) for r in common]
print(f"\nkappa posture on {len(common)}: {cohen_kappa_score(a, b):.6f}")
print("ds cats", collections.Counter(a))
print("haiku cats", collections.Counter(b))
# agreement rate
print("raw agreement", sum(x == y for x, y in zip(a, b)) / len(common))

# R1
def r1(stage, key="r1"):
    rec = json.load(open(R / f"r1r3-{stage}.json"))
    src = rec.get(key) or []
    out = {}
    for arm in sorted({r["arm"] for r in src}):
        rows = [r for r in src if r["arm"] == arm and r.get("s") is not None]
        vals = [r["s"] for r in rows]
        p3 = sum(r["P3"] for r in src if r["arm"] == arm) / sum(1 for r in src if r["arm"] == arm)
        out[arm] = (sum(vals) / len(vals), min(vals), max(vals), p3)
    return out

for stage in ("pretrain-3.0T", "sft", "instruct", "qwen-base", "qwen-chat"):
    o = r1(stage)
    print(f"\nR1 {stage}:")
    for arm, (s, lo, hi, p3) in o.items():
        print(f"  {arm:18s} s={s:.6f} range=[{lo:.6f},{hi:.6f}] meanP3={p3:.4f}")
