#!/usr/bin/env python3
"""Independent recomputation of Part 202's scored results. Reads only inputs; does not import score202.py."""
import json, math, re, sys, random, hashlib
from pathlib import Path
from collections import Counter, defaultdict

WT = Path(sys.argv[1])
R = WT / "results/part202"
sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "part86"))
import triage86

def jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

rows = jsonl(R / "answers.jsonl")
rep = jsonl(R / "replicate.jsonl")
load = json.loads((R / "load.json").read_text())
out = {}

# ---- 1. row counts
key = lambda r: (r["stage"], r["unit_id"], float(r["dose"]), int(r["question_index"]))
keys = [key(r) for r in rows]
by_stage = Counter(r["stage"] for r in rows)
by_stage_arm = Counter((r["stage"], r["arm"]) for r in rows)
bat = [r for r in rows if r["stage"] == "battery"]
cap = lambda arm: sum(1 for r in bat if r["arm"] == arm and r["token_count"] == 80)
mean_tc = lambda arm: sum(r["token_count"] for r in bat if r["arm"] == arm) / sum(1 for r in bat if r["arm"] == arm)
out["rows"] = {"total": len(rows), "unique": len(set(keys)), "by_stage": dict(by_stage),
               "by_stage_arm": {f"{a}/{b}": c for (a, b), c in sorted(by_stage_arm.items())},
               "nonfinite": sum(1 for r in rows if r.get("nonfinite")),
               "cap_neuron": cap("neuron"), "cap_random": cap("random"),
               "n_neuron": sum(1 for r in bat if r["arm"] == "neuron"),
               "n_random": sum(1 for r in bat if r["arm"] == "random"),
               "mean_tc_neuron": mean_tc("neuron"), "mean_tc_random": mean_tc("random")}

# ---- 2. R3 from load.json
calls = load["loads"][0]["calls"]
r3 = {"calls": len(calls), "pushed": 0, "nopush": 0, "bad_chunk_size": 0, "bad_nopush": 0,
      "companions": 0, "pushed_rowcounts": Counter(), "covered_once": True, "uncovered": 0, "doubled": 0}
seen = Counter()
for c in calls:
    if c["pushed"]:
        r3["pushed"] += 1
        r3["pushed_rowcounts"][c["row_count"]] += 1
        for ch in c["chunks"]:
            if ch["rows"] != 8:
                r3["bad_chunk_size"] += 1
            r3["companions"] += len(ch["companions"])
            for rid in ch["row_ids"]:
                seen[rid] += 1
    else:
        r3["nopush"] += 1
        if len(c["chunks"]) != 1 or c["chunks"][0]["rows"] != 1:
            r3["bad_nopush"] += 1
r3["pushed_rowcounts"] = dict(r3["pushed_rowcounts"])
# every planned row covered exactly once: use pushed row_ids minus companions
planned = Counter()
for c in calls:
    for ch in c["chunks"]:
        comp = Counter(ch["companions"])
        for rid in ch["row_ids"]:
            planned[rid] += 1
# companions duplicate so subtract them
for c in calls:
    for ch in c["chunks"]:
        for rid in ch["companions"]:
            planned[rid] -= 1
r3["uncovered"] = sum(1 for v in planned.values() if v != 1)
r3["distinct_planned"] = len(planned)
# every pushed row in answers has a receipt
row_ids_pushed = {r["row_id"] for r in rows if r["stage"] != "no_push"}
r3["answer_pushed_rows"] = len(row_ids_pushed)
r3["answer_rows_not_in_receipts"] = len(row_ids_pushed - set(planned))
out["r3"] = r3

# ---- 3. 202.5 replicate pairing
batkey = {(r["unit_id"], float(r["dose"]), int(r["question_index"])): r for r in bat}
assert len(batkey) == len(bat)
matched = 0; differ = []; unmatched = []
for rp in rep:
    k = (rp["unit_id"], float(rp["dose"]), int(rp["question_index"]))
    b = batkey.get(k)
    if b is None:
        unmatched.append(k); continue
    matched += 1
    if rp["answer"] != b["answer"]:
        differ.append((rp, b))
# first differing position & battery margin there
tie_first = 0; tie_any = 0; first_pos_margins = []
for rp, b in differ:
    ta, tb = rp["tokens"], b["tokens"]
    pos = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), min(len(ta), len(tb)))
    first_pos_margins.append(pos)
    m = b["margin"][pos] if pos < len(b["margin"]) else None
    if m == 0.0:
        tie_first += 1
    if 0.0 in b["margin"]:
        tie_any += 1
# text-level first difference (chars) vs token
out["p2025"] = {"replicate_rows": len(rep), "matched": matched, "unmatched": len(unmatched),
                "differ": len(differ), "same": matched - len(differ),
                "differ_first_pos_margin_zero": tie_first,
                "differ_battery_has_any_zero_margin": tie_any,
                "in_40_100": 40 <= len(differ) <= 100}
# margin zero counts overall
out["margins"] = {"battery_rows_with_zero_margin": sum(1 for r in bat if 0.0 in r["margin"]),
                  "battery_zero_margin_positions": sum(r["margin"].count(0.0) for r in bat),
                  "battery_positions": sum(len(r["margin"]) for r in bat)}

# ---- 4. bare Qwen rate per arm at 0.8 (registered rule, independent of score164)
QWEN = re.compile(r"\bQwen\b")
EXCL = re.compile(r"alibaba|tongyi|通义", re.IGNORECASE)
def eligible(a):
    return isinstance(a, str) and bool(a.strip()) and triage86.triage_answer(a) != "broken" and not EXCL.search(a)
def unit_qwen(arm, dose):
    rates = {}
    for u in sorted({r["unit_id"] for r in bat if r["arm"] == arm}):
        ans = [r["answer"] for r in bat if r["arm"] == arm and float(r["dose"]) == dose and r["unit_id"] == u]
        el = [a for a in ans if eligible(a)]
        rates[u] = (sum(bool(QWEN.search(a)) for a in el) / len(el), len(el)) if el else (None, 0)
    vals = [v for v, _ in rates.values() if v is not None]
    return sum(vals) / len(vals), len(vals), sum(1 for v, _ in rates.values() if v is None)
def unit_broken(arm, dose):
    rates = {}
    for u in sorted({r["unit_id"] for r in bat if r["arm"] == arm}):
        ans = [r["answer"] for r in bat if r["arm"] == arm and float(r["dose"]) == dose and r["unit_id"] == u]
        pres = [a for a in ans if isinstance(a, str) and a.strip()]
        rates[u] = sum(triage86.triage_answer(a) == "broken" for a in pres) / len(pres)
    return sum(rates.values()) / len(rates)
q = {"qwen_0.8_neuron": unit_qwen("neuron", 0.8), "qwen_0.8_random": unit_qwen("random", 0.8),
     "qwen_1.0_random": unit_qwen("random", 1.0), "qwen_1.2_random": unit_qwen("random", 1.2)}
b = {"broken_0.8_neuron": unit_broken("neuron", 0.8), "broken_0.8_random": unit_broken("random", 0.8),
     "broken_1.0_random": unit_broken("random", 1.0), "broken_1.2_random": unit_broken("random", 1.2)}
out["rates"] = {"qwen": q, "broken": b}
# raw counts for cross-check
def raw(arm, dose):
    ans = [r["answer"] for r in bat if r["arm"] == arm and float(r["dose"]) == dose]
    el = [a for a in ans if eligible(a)]
    return {"rows": len(ans), "eligible": len(el), "qwen_hits": sum(bool(QWEN.search(a)) for a in el),
            "broken": sum(triage86.triage_answer(a) == "broken" for a in ans if a.strip())}
out["raw"] = {f"{arm}|{d}": raw(arm, d) for arm in ("neuron", "random") for d in ((0.8,) if arm == "neuron" else (0.8, 1.0, 1.2))}

# ---- 6. label flip rates for shape floor
def labels(r):
    a = r["answer"]
    broken = (not isinstance(a, str)) or (not a.strip()) or triage86.triage_answer(a) == "broken"
    excluded = (not broken) and bool(EXCL.search(a))
    qwen = (not broken) and (not excluded) and bool(QWEN.search(a))
    return {"broken": broken, "excluded": excluded, "qwen": qwen}
flips = Counter(); pairs = 0
for rp, bb in [(rp, batkey[(rp["unit_id"], float(rp["dose"]), int(rp["question_index"]))]) for rp in rep
               if (rp["unit_id"], float(rp["dose"]), int(rp["question_index"])) in batkey]:
    pairs += 1
    la, lb = labels(bb), labels(rp)
    for k in la:
        flips[k] += la[k] != lb[k]
out["flips"] = {"pairs": pairs, **{k: v for k, v in flips.items()}}

# ---- 7. hand-read 10 differing pairs, seed 202
rng = random.Random(202)
idx = sorted(range(len(differ)))
pick = rng.sample(idx, 10)
out["hand_read"] = []
for i in pick:
    rp, bb = differ[i]
    out["hand_read"].append({"unit_id": bb["unit_id"], "arm": bb["arm"], "dose": bb["dose"],
                             "question_index": bb["question_index"], "question": bb["question"],
                             "battery": bb["answer"], "replicate": rp["answer"],
                             "battery_labels": labels(bb), "replicate_labels": labels(rp),
                             "battery_tc": bb["token_count"], "replicate_tc": rp["token_count"]})

# ---- R2 hashes ourselves
r2 = {}
for p, want in [("part164/units164.json", "4f146140e4dbc6d28b95ed937e6ccf72b9e712a641c721235628b5ec01627921"),
                ("results/part112/medians.json", "65562b342818b46189a98ae18852e6a972d1b41a594f7ce1c442016aecd0f715"),
                ("part153/battery153.py", "e2032288651d65cafa2e60634f512470ec346f5ad093e8be7d6510df089d96a7"),
                ("part155/run155.py", "07b15c41a731e07fffe2b0a26d09a95b2c808a71a1020180096cfa41baac0fa5"),
                ("part164/run164.py", "19d36e18a25f730dd37b1df46c06404a154f5b1e338a53c9cf04e70b909e7277")]:
    got = hashlib.sha256((WT / p).read_bytes()).hexdigest()
    r2[p] = {"got": got, "match": got == want}
out["r2"] = r2
out["questions"] = {"battery_q": sorted({r["question_index"] for r in bat}),
                    "battery_questions": {r["question_index"]: r["question"] for r in bat if r["arm"] == "neuron" and float(r["dose"]) == 0.8}}
print(json.dumps(out, indent=1, default=str))
