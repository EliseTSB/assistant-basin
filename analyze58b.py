#!/usr/bin/env python3
import json, re, sys, collections
import numpy as np
import judge_opt as j
split = sys.argv[1] if len(sys.argv) > 1 else "dev"
names = sys.argv[2:] or ["V0", "V3"]
recs, units = j.load_units()
rec = {i: r for i, r in units}
import glob
parts = [np.load(f) for f in sorted(glob.glob(str(j.OUT / f"acts.{split}*.npz")))]
M = np.concatenate([q["max"] for q in parts]); sids = [s for q in parts for s in q["sids"]]; ucols = [tuple(json.loads(x)) if isinstance(x, str) else tuple(x) for x in parts[0]["units"]]
col = {u: k for k, u in enumerate(ucols)}
arm = [s.split(":", 1)[1].rsplit("#", 1)[0] for s in sids]; owner = [int(s.split(":")[0]) for s in sids]
by = collections.defaultdict(lambda: collections.defaultdict(list))
for k, (a, o) in enumerate(zip(arm, owner)): by[o][a].append(k)
res = {}
for o in by:
    r = rec[o]; c = col[(r["layer"], r["unit"])]
    acts = sorted(x["act"] for x in r["random"]); thr = acts[len(acts) // 2]
    hit = M[:, c] >= thr
    ctrl = [k for k in range(len(sids)) if owner[k] != o and arm[k] == "B"]
    cr = float(hit[ctrl].mean())
    row = {"thr": thr, "max": r["max"], "ctrl": cr}
    for a, ks in by[o].items(): row["h_" + a] = float(hit[ks].mean()); row["n_" + a] = len(ks)
    res[o] = row
hb = np.array([res[o]["h_B"] for o in res if "h_B" in res[o]]); cc = np.array([res[o]["ctrl"] for o in res if "h_B" in res[o]])
print("units", len(res), "B hit rate mean %.3f, control hit rate mean %.3f" % (hb.mean(), cc.mean()))
print("B-lift quantiles", np.quantile(hb - cc, [.1, .25, .5, .75, .9, .95]).round(2))
for t in (0.5, 0.75, 1.0): print("units with B hit >= %.2f and ctrl <= .10: %d" % (t, sum(1 for o in res if "h_B" in res[o] and res[o]["h_B"] >= t and res[o]["ctrl"] <= .1)))
for a in sorted({k[2:] for o in res for k in res[o] if k.startswith("h_")}):
    v = [res[o]["h_" + a] - res[o]["ctrl"] for o in res if "h_" + a in res[o]]
    print("arm %-6s n=%d mean lift %.3f; share with lift>=0.5: %.3f" % (a, len(v), np.mean(v), np.mean([x >= .5 for x in v])))
# does each labeller's rating predict verified B-lift?
def fitfrac(x):
    m = re.match(r"\s*(\d+)\s*(?:of|/)\s*(\d+)", str(x["desc"].get("fit", ""))); return int(m.group(1)) / int(m.group(2)) if m else None
for nm in names:
    L = {}
    for l in open(j.OUT / f"{nm}.{split}.jsonl"):
        x = json.loads(l)
        if "desc" in x: L[x["i"]] = x
    os_, fs, ls = [], [], []
    for o in res:
        if o in L and "h_B" in res[o]:
            os_.append(int(L[o]["desc"]["one_job"])); ff = fitfrac(L[o]); fs.append(ff if ff is not None else 0); ls.append(res[o]["h_B"] - res[o]["ctrl"])
    truth = [1 if l >= .5 else 0 for l in ls]
    print(nm, "n", len(ls), "verified", sum(truth), "spearman(one_job, lift) %.3f" % j.spearman(os_, ls),
          "AUC(one_job->verified) %s" % (round(j.auc(os_, truth), 3) if 0 < sum(truth) < len(truth) else None),
          "spearman(fit,lift) %.3f" % j.spearman(fs, ls) if any(fs) else "")
json.dump({str(k): v for k, v in res.items()}, open(j.OUT / f"lift.{split}.json", "w"))
