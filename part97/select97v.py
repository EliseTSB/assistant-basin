#!/usr/bin/env python3
"""select97v.py -- Part 97.3 validation push, step 1: draw the units (Claude, Opus 5.5, 2026-10-29; PART97-DESIGN.md
step 4 and Amendment 3).

Pools: real neurons in layers 1 to 12 whose registered Haiku J-lens match (results/part97/match.jsonl) is true
(matched) or false (unmatched), each sorted by (layer, index). random.Random(97097) draws 10 matched, then 10 unmatched.
Usage: select97v.py results/part97 results/part97/validation/units.json
"""
import json
import random
import sys
from pathlib import Path


def main(src, dst):
    src = Path(src)
    match = {}
    for line in open(src / "match.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("lens") == "j_lens" and r.get("kind") == "neuron":
            key = (int(r["layer"]), int(r["index"]))
            if key in match:
                raise SystemExit(f"duplicate J-lens match row for {key}")
            if not isinstance(r.get("match"), bool):
                raise SystemExit(f"no boolean match for {key}")
            match[key] = r["match"]
    reads = {}
    for line in open(src / "readouts.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] == "neuron":
            reads[(int(r["layer"]), int(r["index"]))] = r
    pool = sorted(k for k in match if 1 <= k[0] <= 12 and k in reads)
    matched = [k for k in pool if match[k]]
    unmatched = [k for k in pool if not match[k]]
    rng = random.Random(97097)
    chosen = [("matched", k) for k in rng.sample(matched, 10)] + [("unmatched", k) for k in rng.sample(unmatched, 10)]
    units = [{"group": g, "layer": k[0], "index": k[1], "j_lens": reads[k]["j_lens"], "persona": reads[k]["persona"],
              "screen_answer": reads[k]["answer"]} for g, k in chosen]
    out = {"pool_matched": len(matched), "pool_unmatched": len(unmatched), "seed": 97097, "units": units}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"pool_matched": len(matched), "pool_unmatched": len(unmatched),
                      "layers": sorted({u["layer"] for u in units})}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
