#!/usr/bin/env python3
"""Part 58: Haiku writes 4 unannounced-pattern texts per atlas unit (same GEN_B prompt as gen_all.py).
usage: gen_atlas.py <atlas-dir> <out-dir> <n_real> <n_control> [seed]
Samples n_real live units and n_control random-direction controls (seeded), writes <out-dir>/gen.jsonl and
<out-dir>/sample.json. Resumable: units already in gen.jsonl are skipped. Needs ANTHROPIC_CREDITS_KEY (judge_opt.call)."""
import ast, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import judge_opt as j

src = ast.parse(Path(__file__).with_name("gen_all.py").read_text())
GEN_B = next(n.value.value for n in src.body if isinstance(n, ast.Assign) and n.targets[0].id == "GEN_B")
atlas, out = Path(sys.argv[1]), Path(sys.argv[2])
n_real, n_ctl = int(sys.argv[3]), int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 5801
out.mkdir(parents=True, exist_ok=True)
def stream_pick(path, rng, n, ok):
    """Pick n rows without loading the file: pass 1 finds the eligible line numbers, pass 2 parses only those."""
    elig = []
    with path.open() as fh:
        for i, l in enumerate(fh):
            r = json.loads(l)
            if ok(r): elig.append(i)
            del r
    chosen = set(rng.sample(elig, min(n, len(elig))))
    rows = []
    with path.open() as fh:
        for i, l in enumerate(fh):
            if i in chosen: rows.append(json.loads(l))
    return rows
rows_p = out / "sample-rows.jsonl"
if rows_p.exists():  # reading the atlas over /mnt/c takes ~13 minutes; the sampled rows are kept beside the output
    pick = [(x["kind"], x["key"], x["row"]) for x in map(json.loads, rows_p.open())]
else:
    rng = random.Random(seed)
    okf = lambda r: r["max"] > 0 and len(r["top"]) >= 12
    pick = [("real", f"{r['layer']}:{r['unit']}", r) for r in stream_pick(atlas / "neurons.jsonl", rng, n_real, okf)]
    pick += [("control", f"c{c['control_id']}", c) for c in stream_pick(atlas / "controls.jsonl", rng, n_ctl, okf)]
    with rows_p.open("w") as fh:
        for k, key, r in pick: fh.write(json.dumps({"kind": k, "key": key, "row": r}, ensure_ascii=False) + "\n")
(out / "sample.json").write_text(json.dumps([{"kind": k, "key": key, "layer": r["layer"]} for k, key, r in pick]))
genp = out / "gen.jsonl"
have = set()
if genp.exists():
    for l in genp.open(): have.add(json.loads(l)["key"])
todo = [p for p in pick if p[1] not in have]
print(len(pick), "sampled;", len(todo), "to generate", flush=True)
def work(p):
    kind, key, r = p
    wins = [x["window"] for x in r["top"][:12]] + [x["window"] for x in r["random"][:6]]
    try:
        d = j.parse_json(j.call(GEN_B, json.dumps(wins, ensure_ascii=False, separators=(",", ":")), 600))
    except RuntimeError:
        return None
    t = d.get("texts") if isinstance(d, dict) else None
    texts = [y for y in (t or []) if isinstance(y, str)][:4]
    return {"key": key, "kind": kind, "layer": r["layer"], "texts": texts} if len(texts) == 4 else None
bad = 0
with genp.open("a") as fh, ThreadPoolExecutor(12) as ex:
    for n, row in enumerate(ex.map(work, todo)):
        if row is None: bad += 1; continue
        fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
        if (n + 1) % 250 == 0: print(n + 1, f"${j.cost():.3f}", flush=True)
print("calls", j.USAGE["calls"], "failed", bad, f"cost ${j.cost():.3f}")
with (j.OUT / "spend.log").open("a") as fh:
    fh.write(json.dumps({"name": "gen_atlas", "split": str(out), "calls": j.USAGE["calls"], "usd": round(j.cost(), 4), "ts": time.strftime("%F %T")}) + "\n")
