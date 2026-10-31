#!/usr/bin/env python3
"""Build Part 135 pole-pair rows and count summaries."""
import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

import numpy as np

LAYERS = 36
WIDTH = 12288
ROTATIONS = 1000
SEED = 135


def read_layer(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def library_layer(data):
    """Return plus/minus persona flags and pole records for one layer."""
    rows, cats = data["rows"], data["categories"]
    if len(rows) != 2 * WIDTH:
        raise ValueError(f"expected {2 * WIDTH} rows, got {len(rows)}")
    flags = np.zeros((2, WIDTH), dtype=bool)
    values = [[None] * WIDTH for _ in range(2)]
    for i, row in enumerate(rows):
        if not row or cats[row[0]] != "persona":
            continue
        neuron, pole = i // 2, i % 2
        name = row[1] if len(row) > 1 else ""
        desc = row[7] if len(row) > 7 else ""
        flags[pole, neuron] = True
        values[pole][neuron] = (name or "Unnamed persona", desc)
    return flags, values


def first_sentence(text):
    text = " ".join(str(text or "").split())
    for i, char in enumerate(text):
        if char in ".!?" and (i + 1 == len(text) or text[i + 1].isspace()):
            text = text[:i + 1]
            break
    return text[:200]


def persona_text(record):
    return f"{record[0]}: {first_sentence(record[1])}"[:200]


def rotation_null(plus, minus, rotations=ROTATIONS, seed=SEED):
    """Return the overlap counts from random circular shifts, layer by layer."""
    rng = np.random.default_rng(seed)
    out = np.zeros((rotations, plus.shape[0]), dtype=np.int64)
    for layer in range(plus.shape[0]):
        for r in range(rotations):
            offset = int(rng.integers(plus.shape[1]))
            out[r, layer] = np.count_nonzero(plus[layer] & np.roll(minus[layer], offset))
    return out


def derangement(neurons, rng):
    """Map each neuron to another member of the same list."""
    neurons = list(neurons)
    if len(neurons) < 2:
        raise ValueError("a derangement needs at least two neurons")
    order = neurons[:]
    rng.shuffle(order)
    shift = rng.randrange(1, len(order))
    return {n: order[(i + shift) % len(order)] for i, n in enumerate(order)}


def row_id(layer, neuron, partner, arm):
    raw = f"part135:{SEED}:{layer}:{neuron}:{partner}:{arm}".encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def text_order(a, b, rng):
    return (a, b) if rng.randrange(2) == 0 else (b, a)


def shuffled_texts(text_for, layer, neuron, partner_layer, partner):
    """Take the source plus pole and a different neuron's minus pole."""
    return text_for(layer, neuron, 0), text_for(partner_layer, partner, 1)


def make_row(layer, neuron, partner, arm, a, b, rng, partner_layer=None):
    left, right = text_order(a, b, rng)
    return {"id": row_id(layer, neuron, partner, arm), "arm": arm,
            "layer": layer, "neuron": neuron, "partner_neuron": partner,
            "partner_layer": layer if partner_layer is None else partner_layer,
            "a": left, "b": right}


def catalog_layer(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["neuron", "pole", "fires"])
    out = {}
    for row in table.to_pylist():
        pole = int(row["pole"])
        if pole in (-1, 1):
            out[(int(row["neuron"]), pole)] = row["fires"]
    return out


def build(library, catalog, pairs_out, counts_out):
    rng = random.Random(SEED)
    plus, minus = np.zeros((LAYERS, WIDTH), dtype=bool), np.zeros((LAYERS, WIDTH), dtype=bool)
    records, fire_labels = {}, {}
    persona_counts = []
    for layer in range(LAYERS):
        data = read_layer(Path(library) / f"L{layer:02d}.gz")
        flags, values = library_layer(data)
        plus[layer], minus[layer] = flags
        records[layer] = values
        persona_counts.append(int(flags.sum()))
        fire_labels[layer] = catalog_layer(Path(catalog) / f"L{layer:02d}.parquet")
    both_by_layer = (plus & minus).sum(axis=1).astype(int)
    null = rotation_null(plus, minus)
    early_poles = int(sum(persona_counts[:18]))
    late_poles = int(sum(persona_counts[18:]))
    early_both_poles = int(2 * both_by_layer[:18].sum())
    late_both_poles = int(2 * both_by_layer[18:].sum())
    counts = {
        "seed": SEED, "both_pole_by_layer": both_by_layer.tolist(),
        "both_pole_total": int(both_by_layer.sum()),
        "rotation_null_mean_by_layer": null.mean(axis=0).tolist(),
        "rotation_null_mean": float(null.sum(axis=1).mean()),
        "rotation_null_counts": null.sum(axis=1).tolist(),
        "persona_poles_by_layer": persona_counts,
        "persona_poles_total": int(sum(persona_counts)),
        "both_pole_share_L0_17": early_both_poles / early_poles if early_poles else None,
        "both_pole_share_L18_35": late_both_poles / late_poles if late_poles else None,
    }
    candidates = [(layer, int(n)) for layer in range(LAYERS)
                  for n in np.flatnonzero(plus[layer] & minus[layer])]
    sample = rng.sample(candidates, min(250, len(candidates)))
    sample_by_layer = {}
    for layer, neuron in sample:
        sample_by_layer.setdefault(layer, []).append(neuron)
    pairs = []
    missing_fire = 0
    eligible_fire = {}
    for layer, neuron in sample:
        p, m = records[layer][0][neuron], records[layer][1][neuron]
        pole_text = [persona_text(x) for x in (p, m)]
        pairs.append(make_row(layer, neuron, neuron, "persona", *pole_text, rng))
        labels = fire_labels[layer]
        fp, fm = labels.get((neuron, 1)), labels.get((neuron, -1))
        if not fp or not fm:
            missing_fire += 1
            continue
        eligible_fire.setdefault(layer, []).append(neuron)
        pairs.append(make_row(layer, neuron, neuron, "fires", fp, fm, rng))
    counts["sampled_both_pole_neurons"] = len(sample)
    counts["sampled_neurons_missing_catalog_pole"] = missing_fire
    for base_arm, pool_by_layer, text_fn in (
        ("persona", sample_by_layer,
         lambda l, n, sign: persona_text(records[l][sign][n])),
        ("fires", eligible_fire, lambda l, n, sign: fire_labels[l][(n, 1 if sign == 0 else -1)]),
    ):
        groups = {l: ns[:] for l, ns in pool_by_layer.items() if ns}
        maps = {l: derangement(ns, rng) for l, ns in groups.items() if len(ns) > 1}
        nonempty = sorted(groups)
        for layer, neurons in groups.items():
            for neuron in neurons:
                if len(neurons) > 1:
                    partner = maps[layer][neuron]
                    partner_layer = layer
                else:
                    neighbors = [other for other in nonempty if other != layer]
                    if not neighbors:
                        continue
                    nearest = min(neighbors, key=lambda other: (abs(other - layer), other))
                    partner = groups[nearest][rng.randrange(len(groups[nearest]))]
                    partner_layer = nearest
                    counts.setdefault("cross_layer_shuffle_fallbacks", []).append(
                        {"layer": layer, "neuron": neuron, "partner_layer": nearest, "partner_neuron": partner})
                a, b = shuffled_texts(text_fn, layer, neuron, partner_layer, partner)
                pairs.append(make_row(layer, neuron, partner, base_arm + "_shuffled", a, b, rng,
                                      partner_layer=partner_layer))
    # Keep row order opaque and reproducible; IDs are hashes with no arm text.
    rng.shuffle(pairs)
    for path in (pairs_out, counts_out):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(pairs_out, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(counts_out, "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {len(pairs)} pair rows; sampled {len(sample)} neurons; missing fire labels for {missing_fire}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--counts", required=True)
    a = p.parse_args()
    build(a.library, a.catalog, a.out, a.counts)


if __name__ == "__main__":
    main()
