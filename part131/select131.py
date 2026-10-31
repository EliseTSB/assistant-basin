#!/usr/bin/env python3
"""Select the registered Part 131 shared, single, and matched control units."""
from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path("/home/elisepc/neuron-probe/persona-library")
BANDS = ((0, 6), (6, 16), (16, 32))
MODELS = ("llama", "tulu", "hermes")


@lru_cache(maxsize=96)
def read_layer(model: str, layer: int) -> dict:
    path = LIBRARY / model / f"L{layer:02d}.gz"
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def row_for(model: str, layer: int, neuron: int, sign: int) -> dict | None:
    data = read_layer(model, layer)
    index = neuron * 2 + (0 if sign > 0 else 1)
    row = data["rows"][index]
    if not row:
        return None
    cat, persona, weirdness, coherence, ai, stop, tags, desc, text = row
    return {"category": data["categories"][cat], "persona": persona, "weirdness": weirdness,
            "coherence": coherence, "alpha": data["levels"][ai], "stop": stop,
            "tags": tags, "description": desc, "text": text}


def eligible(row: dict | None, minimum_coherence: int = 7) -> bool:
    return bool(row and row["category"] == "persona" and row["coherence"] >= minimum_coherence
                and row["persona"] not in (None, "", "None") and row["stop"] != 1
                and not str(row["text"] or "").startswith("Okay"))


def unit(uid: str, kind: str, layer: int, neuron: int, sign: int, source: str,
         target_row: dict, **extra) -> dict:
    return {"id": uid, "class": kind, "L": layer, "n": neuron, "s": sign,
            "alpha": target_row["alpha"], "persona": target_row["persona"],
            "desc": target_row["description"], "tags": target_row["tags"],
            "source_model": source, **extra}


def main() -> None:
    global LIBRARY
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "part131/units131.json")
    ap.add_argument("--triples", type=Path, default=ROOT / "results/part130/triples130.json")
    ap.add_argument("--pairs", type=Path, default=ROOT / "results/part130/pairs130.jsonl")
    ap.add_argument("--labels", type=Path, default=ROOT / "results/part130/labels-ds.jsonl")
    ap.add_argument("--library", type=Path, default=LIBRARY)
    a = ap.parse_args()
    LIBRARY = a.library
    rng = random.Random(131)
    triples = json.loads(a.triples.read_text(encoding="utf-8"))
    pair_by_id = {(r["id"], r["arm"]): r for r in
                  (json.loads(line) for line in a.pairs.read_text(encoding="utf-8").splitlines() if line.strip())}
    labels = {(r["id"], r["arm"]): r for r in
              (json.loads(line) for line in a.labels.read_text(encoding="utf-8").splitlines() if line.strip())}
    reader_positive = []
    for key, label in labels.items():
        pair = pair_by_id.get(key)
        if (key[1] == "same" and "error" not in label and label.get("same") is True
                and pair and all(k in pair for k in ("L", "n", "s"))):
            reader_positive.append((pair["L"], pair["n"], pair["s"]))
    by_key = {(t["L"], t["n"], t["s"]): t for t in triples}
    identical = [key for key, triple in by_key.items()
                 if all(isinstance(triple.get(m), str) and triple[m].strip()
                        and triple[m].strip().lower() == triple["llama"].strip().lower()
                        for m in MODELS)]
    candidates = list(dict.fromkeys(k for k in reader_positive + identical if k in by_key))
    selected = [(2, 2759, 1), (3, 7060, 1)]
    special = set(selected)
    available = [k for k in candidates if k not in special
                 and eligible(row_for("llama", *k))]
    rng.shuffle(available)
    counts = Counter()
    for key in selected:
        counts[next(i for i, (lo, hi) in enumerate(BANDS) if lo <= key[0] < hi)] += 1
    deferred = []
    for key in available:
        band = next(i for i, (lo, hi) in enumerate(BANDS) if lo <= key[0] < hi)
        if counts[band] < 8:
            selected.append(key); counts[band] += 1
        else:
            deferred.append(key)
    # Amendment 1 fills any short band quota with eligible shared candidates in L0-5.
    if len(selected) < 24:
        fallback = [k for k in deferred if 0 <= k[0] < 6]
        fallback.extend(k for k in available if 0 <= k[0] < 6 and k not in selected and k not in fallback)
        for key in fallback:
            if key not in selected:
                selected.append(key)
                if len(selected) == 24:
                    break
    if len(selected) != 24:
        raise ValueError(f"only selected {len(selected)} of 24 shared units; positive Part 130 poles={len(candidates)}")
    units = []
    for i, (L, n, s) in enumerate(selected):
        lr = row_for("llama", L, n, s)
        if not eligible(lr) and (L, n, s) not in special:
            raise ValueError(f"selected shared unit {(L,n,s)} has ineligible Llama row")
        if not lr:
            raise ValueError(f"no Llama row for shared unit {(L,n,s)}")
        units.append(unit(f"sh{i:02d}", "shared", L, n, s, "llama", lr,
                          forced=(L, n, s) in special,
                          llama_eligible=eligible(lr), llama_coherence=lr["coherence"]))

    shared_layers = sorted({u["L"] for u in units})
    for model in MODELS:
        pool = []
        for L in shared_layers:
            for n in range(14336):
                for s in (1, -1):
                    rows = [row_for(m, L, n, s) for m in MODELS]
                    active = [j for j, r in enumerate(rows) if eligible(r)]
                    if len(active) == 1 and MODELS[active[0]] == model:
                        pool.append((L, n, s, rows[active[0]]))
        rng.shuffle(pool)
        chosen = []
        band_counts = Counter()
        for item in pool:
            band = next(i for i, (lo, hi) in enumerate(BANDS) if lo <= item[0] < hi)
            if band_counts[band] >= 8:
                continue
            chosen.append(item); band_counts[band] += 1
            if len(chosen) == 8:
                break
        if len(chosen) != 8:
            raise ValueError(f"found only {len(chosen)} eligible single units for {model}; need 8")
        for i, (L, n, s, row) in enumerate(chosen):
            units.append(unit(f"si-{model}-{i:02d}", "single", L, n, s, model, row,
                              selecting_model=model))

    used_controls = set()
    for u in units:
        L, sign, alpha = u["L"], u["s"], u["alpha"]
        eligible_persona = set()
        for model in MODELS:
            d = read_layer(model, L)
            for i, r in enumerate(d["rows"]):
                if r and d["categories"][r[0]] == "persona":
                    eligible_persona.add(i // 2)
        d = read_layer("llama", L)
        candidates_n = []
        for n in range(14336):
            if n == u["n"] or n in eligible_persona or (L, n, sign, alpha) in used_controls:
                continue
            row = row_for("llama", L, n, sign)
            if row and row["alpha"] == alpha:
                candidates_n.append(n)
        if not candidates_n:
            raise ValueError(f"no matched non-persona control for {u['id']} at L{L}, sign={sign}, alpha={alpha}")
        n = rng.choice(candidates_n)
        used_controls.add((L, n, sign, alpha))
        u["control"] = {"n": n, "L": L, "s": sign, "alpha": alpha}

    output = {"seed": 131, "library_root": str(LIBRARY), "shared_counts_by_band": dict(sorted(Counter(
        next(i for i, (lo, hi) in enumerate(BANDS) if lo <= u["L"] < hi) for u in units if u["class"] == "shared").items())),
              "units": units}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"counts": {k: sum(u["class"] == k for u in units) for k in ("shared", "single")},
                      "shared_bands": output["shared_counts_by_band"], "special": [u["id"] for u in units if u.get("forced")],
                      "units": len(units), "output": str(a.out)}, indent=2))


if __name__ == "__main__":
    main()
