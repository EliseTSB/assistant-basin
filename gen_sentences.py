#!/usr/bin/env python3
"""Part 58b: Haiku writes test texts for each real unit (arm A: from the concept alone; arm B: from the windows, pattern unannounced)."""
import argparse, json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import judge_opt as j

GEN_A = """A hidden unit inside a language model has been described in words. Write 4 different short texts (one or two sentences, under 30 words each) that you expect to make this unit fire strongly. Vary the topic and wording, but keep exactly what the description names. Reply with a JSON object and nothing else: {"texts":["...","...","...","..."]}"""
GEN_B = """Below are text windows where a hidden unit inside a language model fired strongly. The token where it fired is wrapped in [[double brackets]]; ⟦PAD×N⟧ is not text. Do not describe or name the pattern. Write 4 new, different short texts (one or two sentences, under 30 words each) that would make the same unit fire at the same kind of token, following the pattern you see in the windows. Do not copy any window. Reply with a JSON object and nothing else: {"texts":["...","...","...","..."]}"""

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev"); ap.add_argument("--concepts", nargs="*", default=[])   # run names whose concepts feed arm A
ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
recs, units = j.load_units()
sp = [(i, r) for i, r in j.split_units(units)[a.split] if "unit" in r]
if a.limit: sp = sp[:a.limit]
concepts = {}
for name in a.concepts:
    d = {}
    for l in open(j.OUT / f"{name}.{a.split}.jsonl"):
        x = json.loads(l)
        if "desc" in x: d[x["i"]] = x["desc"]["concept"]
    concepts[name] = d
out = Path(a.out); done = set()
if out.exists():
    for l in out.open(): done.add(json.loads(l)["sid"].rsplit("#", 1)[0])
def texts_of(system, user):
    d = j.parse_json(j.call(system, user, 600))
    t = d.get("texts") if isinstance(d, dict) else None
    return [x for x in t if isinstance(x, str)][:4] if isinstance(t, list) else []
def work(item):
    i, r = item; rows = []
    wins = [x["window"] for x in r["top"][:12]] + [x["window"] for x in r["random"][:6]]
    jobs = [("B", GEN_B, json.dumps(wins, ensure_ascii=False, separators=(",", ":")))]
    for name, d in concepts.items():
        if i in d: jobs.append(("A:" + name, GEN_A, "Description of the unit: " + d[i]))
    for arm, system, user in jobs:
        if f"{i}:{arm}" in done: continue
        try:
            for k, t in enumerate(texts_of(system, user)):
                rows.append({"sid": f"{i}:{arm}#{k}", "unit_i": i, "arm": arm, "layer": r["layer"], "unit": r["unit"], "text": t})
        except RuntimeError as e:
            print("fail", i, arm, e, file=sys.stderr)
    return rows
with out.open("a") as fh, ThreadPoolExecutor(12) as ex:
    for rows in ex.map(work, sp):
        for x in rows: fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        fh.flush()
print(f"units {len(sp)} calls {j.USAGE['calls']} cost ${j.cost():.3f}")
