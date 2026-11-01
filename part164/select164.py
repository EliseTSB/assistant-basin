#!/usr/bin/env python3
"""Pair Part 163's random directions with fresh changed neurons."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PART163 = ROOT / "part163"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PART163) not in sys.path:
    sys.path.insert(0, str(PART163))
from part163 import select163  # noqa: E402
import triage86  # noqa: E402


def select_units(screen_rows: list[dict], prior_units: list[dict]) -> list[dict]:
    prior_neurons = {(int(u["layer"]), int(u["neuron"])) for u in prior_units if u["arm"] == "neuron"}
    prior_directions = [u for u in prior_units if u["arm"] == "random"]
    if len(prior_directions) != 100:
        raise ValueError(f"expected 100 Part 163 random directions; found {len(prior_directions)}")
    candidates: dict[int, list[dict]] = defaultdict(list)
    for row in screen_rows:
        if row.get("kind") != "neuron":
            continue
        category = row.get("triage")
        if category not in triage86.CATEGORIES:
            category = triage86.triage_answer(row.get("answer", ""))
        coordinate = (int(row["layer"]), int(row["id"]))
        if category == "changed" and coordinate not in prior_neurons:
            candidates[coordinate[0]].append(row)
    for layer in candidates:
        candidates[layer].sort(key=lambda r: int(r["id"]))

    by_layer: dict[int, list[dict]] = defaultdict(list)
    for direction in prior_directions:
        by_layer[int(direction["layer"])].append(direction)
    rng = random.Random(164)
    result = []
    for layer in sorted(by_layer):
        directions = sorted(by_layer[layer], key=lambda u: int(u["random_index"]))
        options = candidates.get(layer, [])
        if len(options) < len(directions):
            raise ValueError(f"layer {layer}: need {len(directions)} fresh changed neurons, found {len(options)}")
        sampled = rng.sample(options, len(directions))
        for direction, neuron in zip(directions, sampled):
            pair_id = direction["pair_id"]
            result.extend([
                {"unit_id": f"L{layer}N{int(neuron['id'])}", "pair_id": pair_id, "arm": "neuron",
                 "layer": layer, "neuron": int(neuron["id"]), "random_index": None,
                 "stored_answer": neuron.get("answer", "")},
                dict(direction),
            ])
    if len(result) != 200 or sum(u["arm"] == "random" for u in result) != 100:
        raise AssertionError("selection must contain exactly 100 matched pairs")
    chosen = [(int(u["layer"]), int(u["neuron"])) for u in result if u["arm"] == "neuron"]
    if len(set(chosen)) != 100 or prior_neurons.intersection(chosen):
        raise AssertionError("fresh neuron selection has duplicates or overlaps Part 163")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", default=str(ROOT / "results/part86/screen-final"))
    parser.add_argument("--prior-units", default=str(ROOT / "part163/units163.json"))
    parser.add_argument("--out", default=str(HERE / "units164.json"))
    args = parser.parse_args(argv)
    units = select_units(select163.read_screen(args.screen_dir),
                         json.loads(Path(args.prior_units).read_text(encoding="utf-8")))
    Path(args.out).write_text(json.dumps(units, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"selected {sum(u['arm'] == 'neuron' for u in units)} matched pairs", flush=True)


if __name__ == "__main__":
    main()
