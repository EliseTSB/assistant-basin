#!/usr/bin/env python3
"""Part 58c atlas test: Haiku writes 4 unannounced-pattern texts per pilot record (reuses texts already made for dev/test real units)."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
import judge_opt as j

GEN_B = """Below are text windows where a hidden unit inside a language model fired strongly. The token where it fired is wrapped in [[double brackets]]; ⟦PAD×N⟧ is not text. Do not describe or name the pattern. Write 4 new, different short texts (one or two sentences, under 30 words each) that would make the same unit fire at the same kind of token, following the pattern you see in the windows. Do not copy any window. Reply with a JSON object and nothing else: {"texts":["...","...","...","..."]}"""
recs = [json.loads(l) for l in j.PILOT.open()]
known = {(4, 3042), (19, 4826), (22, 1610), (21, 4400)}
have = {}
for f in ("gen.dev.jsonl", "gen.test.jsonl", "gen.all.jsonl"):
    p = j.OUT / f
    if p.exists():
        for l in p.open():
            x = json.loads(l)
            if x["arm"] == "B": have.setdefault(x["unit_i"], {})[x["sid"]] = x
todo = [i for i, r in enumerate(recs) if not ("unit" in r and (r["layer"], r["unit"]) in known) and len(have.get(i, {})) < 4]
print(len(recs), "records;", len(todo), "to generate")
def work(i):
    r = recs[i]; wins = [x["window"] for x in r["top"][:12]] + [x["window"] for x in r["random"][:6]]
    try:
        d = j.parse_json(j.call(GEN_B, json.dumps(wins, ensure_ascii=False, separators=(",", ":")), 600))
    except RuntimeError as e:
        return []
    t = d.get("texts") if isinstance(d, dict) else None
    return [{"sid": f"{i}:B#{k}", "unit_i": i, "arm": "B", "text": x} for k, x in enumerate([y for y in (t or []) if isinstance(y, str)][:4])]
with (j.OUT / "gen.all.jsonl").open("a") as fh, ThreadPoolExecutor(12) as ex:
    for n, rows in enumerate(ex.map(work, todo)):
        for x in rows: fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        fh.flush()
        if (n + 1) % 500 == 0: print(n + 1, f"${j.cost():.3f}", flush=True)
print("calls", j.USAGE["calls"], f"cost ${j.cost():.3f}")
with (j.OUT / "spend.log").open("a") as fh:
    fh.write(json.dumps({"name": "gen_all", "split": "atlas-test", "calls": j.USAGE["calls"], "usd": round(j.cost(), 4), "ts": __import__("time").strftime("%F %T")}) + "\n")
