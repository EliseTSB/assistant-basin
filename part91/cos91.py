#!/usr/bin/env python3
"""cos91.py -- Part 91 weights: each persona neuron's down-column cosine between two checkpoints, and its percentile
among all columns of its layer. Written by Claude (Opus 5.5), 2026-10-28. Usage: cos91.py DIR_A DIR_B neurons91.json OUT"""
import json, sys
from pathlib import Path
import torch
from safetensors import safe_open

def tensor(d, name):
    idx = json.load(open(Path(d) / "model.safetensors.index.json"))["weight_map"]
    with safe_open(str(Path(d) / idx[name]), framework="pt") as f:
        return f.get_tensor(name).float()

A, B, N, OUT = sys.argv[1:5]
res, cache = [], {}
for p in json.load(open(N, encoding="utf-8")):
    L = p["L"]
    if L not in cache:
        name = f"model.layers.{L}.mlp.down_proj.weight"
        wa, wb = tensor(A, name), tensor(B, name)
        cache = {L: (wa * wb).sum(0) / (wa.norm(dim=0) * wb.norm(dim=0))}
    cs = cache[L]
    v = float(cs[p["n"]])
    res.append({"id": p["id"], "L": L, "n": p["n"], "cos": v, "percentile": float((cs < v).float().mean() * 100),
                "layer_p10": float(cs.quantile(0.10)), "layer_median": float(cs.median())})
json.dump(res, open(OUT, "w"), indent=1)
cosv = sorted(r["cos"] for r in res)
print(json.dumps({"n": len(res), "median_cos": cosv[len(cosv) // 2], "below_p10": sum(r["percentile"] < 10 for r in res)}))
