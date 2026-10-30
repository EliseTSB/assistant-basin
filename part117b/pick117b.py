#!/usr/bin/env python3
"""Part 117b: pick the non-persona neuron controls before any data. For each of Part 117's 11 pairs, three neurons from
the matched concept neuron's layer that are not among Part 112's 465 persona neurons (the Qwen atlas units, any layer),
drawn with a seeded sampler. Writes results/part117b/control-neurons.json. Written by Claude (Opus 5.5), 2026-10-29.
Usage: pick117b.py [--width 9216]   (9216 = Qwen3.5-4B's MLP width, text_config.intermediate_size at 851bf6e8)"""
import argparse, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED_BASE, PER_PAIR = 1172, 3


def pick(pairs, persona_units, width, per_pair=PER_PAIR, seed_base=SEED_BASE):
    taken = {(int(u["layer"]), int(u["neuron"])) for u in persona_units}
    out = {}
    for i, p in enumerate(pairs):
        layer = int(p["match"]["layer"])
        pool = [n for n in range(width) if (layer, n) not in taken]
        out[p["pair"]] = {"layer": layer, "neurons": random.Random(seed_base + i).sample(pool, per_pair)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=9216)
    a = ap.parse_args(argv)
    pairs = json.loads((ROOT / "results/part117/pairs.json").read_text(encoding="utf-8"))
    units = list(json.loads((ROOT / "results/part112/score.json").read_text(encoding="utf-8"))["units"].values())
    out = pick(pairs, units, a.width)
    dest = ROOT / "results/part117b"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "control-neurons.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
