#!/usr/bin/env python3
"""Part 58c: sentence test over the whole 4,000-record pilot (real neurons and random-direction controls)."""
import json, collections, random
import numpy as np
import judge_opt as j
recs = [json.loads(l) for l in open(j.PILOT)]
ids = json.load(open(j.OUT / "ids.all.json"))
d = np.load(j.OUT / "acts.all.npz"); M = d["max"]; sids = list(d["sids"])
owner = np.array([int(s.split(":")[0]) for s in sids])
col = {i: k for k, i in enumerate(ids)}
rng = random.Random(7)
pool = np.array(sorted(rng.sample(range(len(sids)), 1200)))      # fixed pool of other-unit texts for the control rate
labels = {}
for l in open(j.HERE / "results/part56-pilot/label/atlas-labels.jsonl"):
    x = json.loads(l)
    if "one_job" in x: labels[x["record_index"]] = x
rows = []
for i in ids:
    r = recs[i]; k = col[i]
    acts = sorted(x["act"] for x in r["random"]); thr = acts[len(acts) // 2]
    own = np.where(owner == i)[0]
    others = pool[owner[pool] != i]
    hb = float((M[own, k] >= thr).mean()); cr = float((M[others, k] >= thr).mean())
    rows.append({"i": i, "type": "neuron" if "unit" in r else "control", "layer": r["layer"], "hb": hb, "ctrl": cr, "lift": hb - cr,
                 "oj": labels.get(i, {}).get("one_job"), "concept": labels.get(i, {}).get("concept")})
json.dump(rows, open(j.OUT / "lift.all.json", "w"))
for t in ("neuron", "control"):
    v = [r for r in rows if r["type"] == t]
    L = np.array([r["lift"] for r in v])
    print(f"{t:8s} n={len(v)} mean B hit {np.mean([r['hb'] for r in v]):.3f} mean ctrl {np.mean([r['ctrl'] for r in v]):.3f}  lift>=0.5: {np.mean(L >= .5):.3f}  lift>=0.75: {np.mean(L >= .75):.3f}")
    byl = collections.defaultdict(list)
    for r in v: byl[r["layer"] // 7].append(r["lift"] >= .5)
    print("   verified by layer group:", {g: round(float(np.mean(x)), 3) for g, x in sorted(byl.items())})
    sc = [(r["oj"], r["lift"] >= .5) for r in v if r["oj"] is not None]
    oj = [a for a, _ in sc]; tr = [int(b) for _, b in sc]
    print(f"   V0 rating -> verified AUC {j.auc(oj, tr):.3f}; spearman(rating, lift) {j.spearman(oj, [r['lift'] for r in v if r['oj'] is not None]):.3f}")
lab = [(r["lift"], 1 if r["type"] == "neuron" else 0) for r in rows]
print("AUC(lift: real vs control) %.3f" % j.auc([a for a, _ in lab], [b for _, b in lab]))
print("controls verified with V0 rating >=3:", sum(1 for r in rows if r["type"] == "control" and r["lift"] >= .5 and (r["oj"] or 0) >= 3), "of", sum(1 for r in rows if r["type"] == "control" and r["lift"] >= .5))
