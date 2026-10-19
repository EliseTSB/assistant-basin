"""score_part20f.py: score Part 20f (20f.1-20f.8) from part20f.py's battery.json and the judge ratings in the battery file.

Recomputes every set's mean maximum from the per-token rows (each row's maximum must equal the maximum of its per_token
values, and the set means must equal the tool's summary) and the mean final-token value (the last token of each
sentence, the period). Applies the three amendments in PREDICTIONS.md Part 20f: the C4 stop rule first; "fires" means a
mean maximum of at least 2.0 and "flat" below 1.04; 20f.3 is the absolute band U < 1.04; 20f.5 holds only if
2.0 <= C3 <= 3.5 and C3 < E; the max-to-final ratio threshold is 0.5. Within-E Spearman uses the judge's scale A
(earthly exclusion) mean against each phrase's maximum.
Usage: python3 -I score_part20f.py results/part20f/battery.json data/part20f-battery.json
"""
import json, statistics as st, sys

res = json.load(open(sys.argv[1], encoding="utf-8")); bat = json.load(open(sys.argv[2], encoding="utf-8"))
rows = res["rows"]
for r in rows:
    assert r["max"] == max(v for _, v in r["per_token"]), ("row max mismatch", r["phrase"])
    assert r["phrase"] == bat["sets"][r["set"]][r["index"]], ("phrase mismatch", r["set"], r["index"])
sets = list(bat["sets"])
mx = {s: st.mean(r["max"] for r in rows if r["set"] == s) for s in sets}
fin = {s: st.mean(r["per_token"][-1][1] for r in rows if r["set"] == s) for s in sets}
for s in sets:
    assert abs(mx[s] - res["summary"]["mean_max_by_set"][s]) < 1e-9, ("summary mismatch", s)
    assert len([r for r in rows if r["set"] == s]) == len(bat["sets"][s]), ("row count", s)

def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
            for k in range(i, j + 1): r[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    a, b = rk(x), rk(y); ma, mb = st.mean(a), st.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b)); den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
    return num / den if den else float("nan")

judge = {(j["set"], j["phrase"]): j for j in bat["judge_rows"]}
E = [r for r in rows if r["set"] == "E"]
rhoE = spearman([judge[("E", r["phrase"])]["A_mean"] for r in E], [r["max"] for r in E])
fires = lambda v: v >= 2.0; flat = lambda v: v < 1.04
stop = fires(mx["C4"])
out = {"mean_max": {s: round(v, 3) for s, v in mx.items()}, "mean_final": {s: round(v, 3) for s, v in fin.items()},
       "final_over_max": {s: round(fin[s] / mx[s], 3) if mx[s] else None for s in sets},
       "within_E_spearman_judgeA_vs_max": round(rhoE, 3), "C4_stop_rule_fired": stop}
pred = {}
pred["20f.8"] = flat(mx["C4"]) and mx["C4"] < 1.0
if stop:
    pred["20f.1"] = pred["20f.5"] = pred["20f.7"] = "not scored (C4 stop rule)"
else:
    pred["20f.1"] = mx["E"] - mx["D"] >= 2.0 and mx["E"] - mx["U"] >= 2.0 and mx["E"] - mx["C4"] >= 2.0 and rhoE < 0.3
    pred["20f.5"] = 2.0 <= mx["C3"] <= 3.5 and mx["C3"] < mx["E"]
    pred["20f.7"] = mx["E-map"] > mx["U"] and mx["E-map"] < mx["E"] and 2.0 <= mx["E-map"] <= 5.0
pred["20f.2"] = abs(mx["D"] - 0.04) < 1.0
pred["20f.3"] = mx["U"] < 1.04
pred["20f.4"] = fires(mx["C3"])
pred["20f.6"] = fires(mx["C1"])
reading = {"C3": "fires" if fires(mx["C3"]) else "flat" if flat(mx["C3"]) else "inconclusive",
           "C3-bare": "fires" if fires(mx["C3-bare"]) else "flat" if flat(mx["C3-bare"]) else "inconclusive",
           "C1": "fires" if fires(mx["C1"]) else "flat" if flat(mx["C1"]) else "inconclusive",
           "C2": "fires" if fires(mx["C2"]) else "flat" if flat(mx["C2"]) else "inconclusive",
           "E-map_within_1_of_E": abs(mx["E-map"] - mx["E"]) <= 1.0}
out["predictions"] = pred; out["readings"] = reading
top = sorted(rows, key=lambda r: -r["max"])[:8]
out["top_rows"] = [(r["set"], r["phrase"], round(r["max"], 2), r["max_token"]) for r in top]
print(json.dumps(out, indent=1, ensure_ascii=False))
print("SELF-CHECK PASS: row maxima, phrases, row counts and set means recomputed from rows")
