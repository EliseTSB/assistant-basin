#!/usr/bin/env python3
"""select91.py -- Part 91: choose 40 persona neurons from Josh Fonseca Rivera's Qwen3-8B library and 40 controls.

Written by Claude (Opus 5.5), 2026-10-28. Reads the per-layer files saved from the library (row = n*2 + (0 if s>0 else 1);
fields cat, persona, weirdness, coherence, alpha index, stop, tags, description, text). Eligible: category "persona",
coherence >= 7, a named persona, a reply that does not start with "Okay" (thinking leaks), stop not "loop". Layers 2-33
are split into eight bands of four; five neurons per band drawn with numpy seed 91. Each gets a control: a random neuron
of the same layer (seed 9191), same sign and alpha, not itself eligible as any persona. Usage: select91.py DIR OUT
"""
import gzip, json, sys
import numpy as np
D, OUT = sys.argv[1], sys.argv[2]
rng, crng = np.random.default_rng(91), np.random.default_rng(9191)
chosen = []
for band in range(8):
    pool = []
    for L in range(2 + 4 * band, 6 + 4 * band):
        d = json.load(gzip.open(f"{D}/L{L:02d}.gz", "rt", encoding="utf-8"))
        cats, levels = d["categories"], d["levels"]
        for i, r in enumerate(d["rows"]):
            if not r:
                continue
            cat, persona, w, c, ai, stop, tags, desc, text = r
            if cats[cat] == "persona" and c >= 7 and persona not in ("", "None") and not text.startswith("Okay") and stop != 1:
                pool.append({"L": L, "n": i // 2, "s": 1 if i % 2 == 0 else -1, "alpha": levels[ai], "persona": persona,
                             "desc": desc, "tags": tags, "reply": text[:400], "_persona_rows": None})
    for j in rng.choice(len(pool), size=5, replace=False):
        chosen.append(pool[int(j)])
out = []
for k, p in enumerate(chosen):
    d = json.load(gzip.open(f"{D}/L{p['L']:02d}.gz", "rt", encoding="utf-8"))
    persona_ns = {i // 2 for i, r in enumerate(d["rows"]) if r and d["categories"][r[0]] == "persona"}
    while True:
        m = int(crng.integers(0, len(d["rows"]) // 2))
        if m not in persona_ns and m != p["n"]:
            break
    p.pop("_persona_rows")
    out.append({"id": f"p{k:02d}", **p, "control_n": m})
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
for p in out:
    print(p["id"], p["L"], p["n"], p["s"], p["alpha"], p["persona"][:40], "| ctrl", p["control_n"])
