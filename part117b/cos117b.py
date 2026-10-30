#!/usr/bin/env python3
"""Part 117b, existing data: the cosine between the vector each pushed unit writes into the residual stream (its
down_proj column, or for a random direction W_down @ v) and the slot neuron's, for Part 117's 11 pairs. Weights only, no
generation. Writes results/part117b/write-cosines.json. Written by Claude (Opus 5.5), 2026-10-29.
Usage: cos117b.py --snapshot <Qwen3.5-4B snapshot dir at revision 851bf6e8>"""
import argparse, json, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main(argv=None):
    import torch
    from safetensors import safe_open
    import rows117b as rb
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    a = ap.parse_args(argv)
    snap = Path(a.snapshot)
    weight_map = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    cache = {}

    def down(layer):
        key = f"model.language_model.layers.{int(layer)}.mlp.down_proj.weight"
        if key not in cache:
            with safe_open(str(snap / weight_map[key]), "pt") as f:
                cache[key] = f.get_tensor(key).float()
        return cache[key]

    def cos(x, y):
        return float(torch.nn.functional.cosine_similarity(x, y, dim=0))

    pairs = json.loads((HERE.parent / "results/part117/pairs.json").read_text(encoding="utf-8"))
    rows = []
    for p in pairs:
        s = down(p["slot"]["layer"])[:, int(p["slot"]["neuron"])]
        m = down(p["match"]["layer"])[:, int(p["match"]["neuron"])]
        u = down(p["unrelated"]["layer"])[:, int(p["unrelated"]["neuron"])]
        w = down(p["match"]["layer"])
        rnd = [cos(s, w @ rb.c.scaled_random_directions(w, 1, rb.random_seed(p["match"]["layer"], p["match"]["neuron"], k),
                                                        1.0)[0]) for k in range(rb.N_RANDOM)]
        rows.append({"pair": p["pair"], "slot_vs_match": round(cos(s, m), 4), "slot_vs_unrelated": round(cos(s, u), 4),
                     "slot_vs_random": [round(x, 4) for x in rnd]})
    out = {"rows": rows,
           "median_slot_vs_match": round(statistics.median(r["slot_vs_match"] for r in rows), 4),
           "median_slot_vs_unrelated": round(statistics.median(r["slot_vs_unrelated"] for r in rows), 4),
           "max_abs_any": round(max(max(abs(r["slot_vs_match"]), abs(r["slot_vs_unrelated"]), *map(abs, r["slot_vs_random"]))
                                    for r in rows), 4)}
    dest = HERE.parent / "results/part117b"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "write-cosines.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))


if __name__ == "__main__":
    main()
