#!/usr/bin/env python3
"""Select Part 163's matched units from the stored Part 86 screen."""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PART86 = ROOT / "part86"
if str(PART86) not in sys.path:
    sys.path.insert(0, str(PART86))
import triage86  # noqa: E402


def read_screen(screen_dir: str | Path) -> list[dict]:
    paths = sorted(Path(screen_dir).glob("layer_*.jsonl.gz"),
                   key=lambda p: int(p.name.split("_")[1].split(".")[0]))
    if not paths:
        raise ValueError(f"no layer_*.jsonl.gz files under {screen_dir}")
    rows = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
                if int(row["layer"]) != int(path.name.split("_")[1].split(".")[0]):
                    raise ValueError(f"row layer disagrees with filename at {path}:{line_no}")
                rows.append(row)
    return rows


def select_units(rows: list[dict], seed: int = 163) -> tuple[list[dict], dict[int, int]]:
    random_changed: dict[int, list[dict]] = defaultdict(list)
    neuron_changed: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        kind = row.get("kind")
        if kind not in ("neuron", "random_direction"):
            continue
        category = row.get("triage")
        if category not in triage86.CATEGORIES:
            category = triage86.triage_answer(row.get("answer", ""))
        if category != "changed":
            continue
        (neuron_changed if kind == "neuron" else random_changed)[int(row["layer"])].append(row)

    rng = random.Random(seed)
    units: list[dict] = []
    counts: dict[int, int] = {}
    for layer in sorted(random_changed):
        directions = sorted(random_changed[layer], key=lambda r: int(r["id"]))
        candidates = sorted(neuron_changed.get(layer, []), key=lambda r: int(r["id"]))
        if len(directions) > len(candidates):
            raise ValueError(f"layer {layer}: {len(directions)} changed random directions but only "
                             f"{len(candidates)} changed neurons")
        sampled = rng.sample(candidates, len(directions))
        counts[layer] = len(directions)
        for direction, neuron in zip(directions, sampled):
            pair_id = f"L{layer}D{int(direction['id'])}"
            units.append({"unit_id": f"L{layer}N{int(neuron['id'])}", "pair_id": pair_id,
                          "arm": "neuron", "layer": layer, "neuron": int(neuron["id"]),
                          "random_index": None, "stored_answer": neuron["answer"]})
            units.append({"unit_id": pair_id, "pair_id": pair_id, "arm": "random",
                          "layer": layer, "neuron": None, "random_index": int(direction["id"]),
                          "stored_answer": direction["answer"]})
    random_count = sum(counts.values())
    if not 80 <= random_count <= 120:
        raise ValueError(f"Gate 3 selection count failed: expected 80-120 changed random rows; found {random_count}")
    for arm in ("neuron", "random"):
        if sum(row["arm"] == arm for row in units) != random_count:
            raise AssertionError(f"selection is not pair matched for {arm}")
    return units, counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen-dir", default=str(ROOT / "results/part86/screen-final"))
    ap.add_argument("--out", default=str(HERE / "units163.json"))
    args = ap.parse_args(argv)
    units, counts = select_units(read_screen(args.screen_dir))
    Path(args.out).write_text(json.dumps(units, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    for layer, count in sorted(counts.items()):
        print(f"layer {layer}: {count} matched pairs")
    print(f"total: {sum(counts.values())} per arm")


if __name__ == "__main__":
    main()
