#!/usr/bin/env python3
"""select132.py -- Part 132: choose 200 fresh Qwen3-8B persona neurons and 200 controls from the saved library.

Written by Claude (Opus 5.5), 2026-10-30, from part91/select91.py with the same eligibility: category persona, coherence
at least 7, a named persona, a reply that does not start with "Okay", stop not "loop". Layers 2-33 in eight bands of four,
25 neurons per band (14 in L30-33, all it has), seed 132, leaving out Part 91's 40 neurons. Controls: a random neuron of the same layer that has no
persona row, same sign and alpha, seed 13232. Usage: select132.py LIBRARY_DIR OUT
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

D, OUT = sys.argv[1], sys.argv[2]
rng, crng = np.random.default_rng(132), np.random.default_rng(13232)
ROOT = Path(__file__).resolve().parents[1]
used = {(int(p["L"]), int(p["n"])) for p in json.load(open(ROOT / "part91/neurons91.json", encoding="utf-8"))}


def layer(L):
    raw = Path(D, f"L{L:02d}.gz").read_bytes()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


chosen = []
for band in range(8):
    pool, seen = [], set()
    for L in range(2 + 4 * band, 6 + 4 * band):
        d = layer(L)
        cats, levels = d["categories"], d["levels"]
        for i, r in enumerate(d["rows"]):
            if not r:
                continue
            cat, persona, w, c, ai, stop, tags, desc, text = r
            if cats[cat] == "persona" and c >= 7 and persona not in ("", "None") and not text.startswith("Okay") and stop != 1 and (L, i // 2) not in used:
                pool.append({"L": L, "n": i // 2, "s": 1 if i % 2 == 0 else -1, "alpha": levels[ai], "persona": persona,
                             "desc": desc, "tags": tags, "reply": text[:400]})
    picks = []
    for j in rng.permutation(len(pool)):
        p = pool[int(j)]
        if (p["L"], p["n"]) not in seen:  # one pole per neuron, so a neuron cannot be its own mismatch
            seen.add((p["L"], p["n"]))
            picks.append(p)
        if len(picks) == 25:
            break
    if len(picks) < 14:  # L30-33 has 14 eligible neurons; the other bands fill 25
        raise SystemExit(f"band {band}: only {len(picks)} eligible neurons")
    chosen.extend(picks)
out = []
for k, p in enumerate(chosen):
    d = layer(p["L"])
    persona_ns = {i // 2 for i, r in enumerate(d["rows"]) if r and d["categories"][r[0]] == "persona"}
    while True:
        m = int(crng.integers(0, len(d["rows"]) // 2))
        if m not in persona_ns and m != p["n"]:
            break
    out.append({"id": f"p{k:03d}", **p, "control_n": m})
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(out), "neurons")
