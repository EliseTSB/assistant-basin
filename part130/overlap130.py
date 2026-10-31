#!/usr/bin/env python3
"""Part 130: how often is the same pole a persona in Llama 3.1 8B Instruct, Tulu 3 8B and Hermes 3 8B?

Reads the saved Neuron Persona Library (row index = n*2 + (0 if sign>0 else 1); the three models share width and layers).
The null rotates each model's persona flags within a layer by a random offset, so each layer's rate and each model's
spacing stay fixed and only the alignment of indices is broken. Writes overlap130.json and pairs130.jsonl (name pairs
for the reader). Written by Claude (Opus 5.5), 2026-10-30. Usage: overlap130.py LIBRARY_DIR OUT_DIR
"""
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np

MODELS = ("llama", "tulu", "hermes")
LAYERS, WIDTH = 32, 14336
ROTATIONS = 1000
SEED = 130


def load(lib, model):
    """{layer: (bool array of 2*WIDTH persona flags, list of names)}"""
    out = {}
    for L in range(LAYERS):
        raw = Path(lib, model, f"L{L:02d}.gz").read_bytes()
        d = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
        cats = d["categories"]
        flags = np.zeros(2 * WIDTH, dtype=bool)
        names = [None] * (2 * WIDTH)
        if len(d["rows"]) != 2 * WIDTH:
            raise ValueError(f"{model} L{L}: {len(d['rows'])} rows")
        for i, r in enumerate(d["rows"]):
            if r and cats[r[0]] == "persona" and r[1] not in ("", "None"):
                flags[i] = True
                names[i] = r[1]
        out[L] = (flags, names)
    return out


def triple_count(flags, rng=None):
    total, pairs = 0, {"llama-tulu": 0, "llama-hermes": 0, "tulu-hermes": 0}
    per_layer = []
    for L in range(LAYERS):
        a, b, c = (flags[m][L][0] for m in MODELS)
        if rng is not None:
            b, c = np.roll(b, int(rng.integers(2 * WIDTH))), np.roll(c, int(rng.integers(2 * WIDTH)))
        n = int((a & b & c).sum())
        total += n
        per_layer.append(n)
        pairs["llama-tulu"] += int((a & b).sum())
        pairs["llama-hermes"] += int((a & c).sum())
        pairs["tulu-hermes"] += int((b & c).sum())
    return total, pairs, per_layer


def main(lib, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    flags = {m: load(lib, m) for m in MODELS}
    obs, pairs, layers = triple_count(flags)
    rng = np.random.default_rng(SEED)
    null = [triple_count(flags, rng) for _ in range(ROTATIONS)]
    null_total = np.array([n[0] for n in null], dtype=float)
    null_layers = np.array([n[2] for n in null], dtype=float).mean(axis=0)
    null_pairs = {k: float(np.mean([n[1][k] for n in null])) for k in pairs}
    early, late = range(0, 16), range(16, 32)
    ratio = lambda rows: sum(layers[L] for L in rows) / max(1e-9, sum(null_layers[L] for L in rows))
    report = {"persona_poles": {m: int(sum(flags[m][L][0].sum() for L in range(LAYERS))) for m in MODELS},
              "triple": obs, "triple_null_mean": float(null_total.mean()), "triple_null_max": float(null_total.max()),
              "ratio": obs / null_total.mean(), "pairs": pairs, "pairs_null_mean": null_pairs,
              "pair_ratios": {k: pairs[k] / null_pairs[k] for k in pairs},
              "per_layer": layers, "per_layer_null": null_layers.tolist(),
              "ratio_L0_15": ratio(early), "ratio_L16_31": ratio(late)}
    json.dump(report, open(out / "overlap130.json", "w"), indent=1)
    triples = []
    for L in range(LAYERS):
        a, b, c = (flags[m][L] for m in MODELS)
        for i in np.flatnonzero(a[0] & b[0] & c[0]):
            i = int(i)
            triples.append({"L": L, "n": i // 2, "s": 1 if i % 2 == 0 else -1, "llama": a[1][i], "tulu": b[1][i], "hermes": c[1][i]})
    picker = random.Random(SEED)
    sample = picker.sample(triples, min(150, len(triples)))
    other = sample[1:] + sample[:1]
    with open(out / "pairs130.jsonl", "w", encoding="utf-8") as f:
        for i, (row, swap) in enumerate(zip(sample, other)):
            f.write(json.dumps({"id": f"t{i}", "arm": "same", "a": row["llama"], "b": row["hermes"], **{k: row[k] for k in ("L", "n", "s")}}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"id": f"t{i}", "arm": "shuffled", "a": row["llama"], "b": swap["hermes"]}, ensure_ascii=False) + "\n")
    json.dump(triples, open(out / "triples130.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("persona_poles", "triple", "triple_null_mean", "ratio", "pair_ratios", "ratio_L0_15", "ratio_L16_31")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
