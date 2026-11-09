#!/usr/bin/env python3
"""Part 163 Arc runner: matched Part 86 selections, recipe check, and prompt battery."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for item in (str(ROOT), str(HERE)):
    if item not in sys.path:
        sys.path.insert(0, item)

from part155 import run155 as part155  # noqa: E402
from part160 import run160 as part160  # noqa: E402

DOSE = 0.8
MAX_NEW_TOKENS = 80
RANDOM_SEED_BASE = 86000
RANDOM_DIRECTIONS_PER_LAYER = 512
MODEL_DEFAULT = part160.MODEL_DEFAULT
# Part 200's one edit (PART200-DESIGN.md, Method item 3): the pilot's rate floor as a module constant. Its value is
# Part 163's own literal, so Part 163's behaviour is unchanged; a wrapper may set it for its own run and restore it.
PILOT_MIN_RATE = 1.5


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{no}") from exc
    return rows


def row_key(row: dict) -> tuple:
    return (row.get("stage"), row.get("unit_id"), int(row.get("question_index", -1)))


def choose_questions():
    items = part155.questions()
    identity = [q for q in items if q["kind"] == "identity"]
    maker = [q for q in items if q["kind"] == "maker"]
    if len(identity) != 8 or len(maker) != 24:
        raise ValueError(f"expected 8 identity and 24 maker prompts; found {len(identity)} and {len(maker)}")
    selected_maker = random.Random(163).sample(maker, 8)
    selected = identity + selected_maker
    if any(q["question"] == "Who are you?" for q in selected):
        raise ValueError("'Who are you?' must not appear among the 16 battery prompts")
    return selected


def part86_random_direction(weight, layer: int, index: int, median: float, common, direction_cache=None):
    """Rebuild Part 86's indexed direction: SCREEN_SEED+layer, 512 draws, +0.8 median."""
    index = int(index)
    if not 0 <= index < RANDOM_DIRECTIONS_PER_LAYER:
        raise ValueError(f"Part 86 random index must be in [0, 511], got {index}")
    if direction_cache is not None and int(layer) in direction_cache:
        directions = direction_cache[int(layer)]
    else:
        directions = common.scaled_random_directions(
            weight, RANDOM_DIRECTIONS_PER_LAYER, common.SCREEN_SEED + int(layer), common.DOSE * float(median)
        )
        if direction_cache is not None:
            direction_cache[int(layer)] = directions
    return directions[index]


def pushes_for(rows, layers, medians, common, direction_cache=None):
    """Part 161-style row-isolated pushes, using Part 86's indexed random basis."""
    import torch
    cache = direction_cache if direction_cache is not None else {}
    pushes = {}
    active_layers = sorted({int(r["layer"]) for r in rows if r.get("layer") is not None})
    for layer_no in active_layers:
        weight = layers[layer_no].mlp.down_proj.weight
        median = float(medians[str(layer_no)] if str(layer_no) in medians else medians[layer_no])
        dense = torch.zeros((len(rows), int(weight.shape[1])), dtype=torch.float32, device=weight.device)
        for row_no, row in enumerate(rows):
            if row.get("layer") is None or int(row["layer"]) != layer_no:
                continue
            if row["arm"] == "neuron":
                dense[row_no, int(row["neuron"])] = common.neuron_delta(
                    weight, int(row["neuron"]), median, dose=DOSE
                )
            elif row["arm"] == "random":
                idx = int(row["random_index"])
                if not 0 <= idx < RANDOM_DIRECTIONS_PER_LAYER:
                    raise ValueError(f"invalid random index {idx} at layer {layer_no}")
                dense[row_no] = part86_random_direction(weight, layer_no, idx, median, common, cache).to(
                    device=weight.device, dtype=torch.float32
                )
            else:
                raise ValueError(f"unknown arm {row.get('arm')!r}")
        pushes[layer_no] = dense
    return pushes


def _unit_row(unit):
    return {"unit_id": unit["unit_id"], "arm": unit["arm"], "layer": int(unit["layer"]),
            "neuron": unit.get("neuron"), "random_index": unit.get("random_index"),
            "dose": DOSE, "unit": unit}


def run(args, loader=None, generate_fn=None):
    units = json.loads(Path(args.units).read_text(encoding="utf-8"))
    if not 80 <= sum(u["arm"] == "neuron" for u in units) <= 120:
        raise ValueError("Gate 3 failed: units file must contain 80-120 units per arm")
    neuron_ids = [u["unit_id"] for u in units if u["arm"] == "neuron"]
    random_ids = [u["unit_id"] for u in units if u["arm"] == "random"]
    if len(neuron_ids) != len(set(neuron_ids)) or len(random_ids) != len(set(random_ids)) or len(neuron_ids) != len(random_ids):
        raise ValueError("units file must contain unique, pair-matched rows")
    items = choose_questions()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    done_rows = read_jsonl(output)
    done = {row_key(r) for r in done_rows}
    common = part155._common()
    medians = json.loads((ROOT / "results/part112/medians.json").read_text(encoding="utf-8"))
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common.REVISION, args.device)
        model.requires_grad_(False)
        model.eval()
    else:
        tok, model = loader(args.device)
    layers = common.find_layers(model)
    generate_fn = generate_fn or (lambda *a: part160.generate_checked(*a, common=common))
    direction_cache = {}
    start_time = time.perf_counter()
    written = 0
    with output.open("a", encoding="utf-8", newline="\n") as fh:
        # One no-push answer for each planned phrasing.
        for qi, item in enumerate(items):
            if ("no_push", "no_push", qi) in done:
                continue
            answers = generate_fn(model, tok, layers, item["question"], {}, MAX_NEW_TOKENS, args.device)
            if len(answers) != 1:
                raise RuntimeError(f"no-push generation returned {len(answers)} answers, expected one")
            row = {"stage": "no_push", "unit_id": "no_push", "arm": "plain", "layer": None,
                   "neuron_or_index": None, "dose": 0.0, "question_index": qi,
                   "question": item["question"], "kind": item["kind"], **answers[0]}
            if not isinstance(row.get("answer"), str) or not row["answer"].strip():
                raise RuntimeError(f"empty no-push answer for question {qi}")
            fh.write(json.dumps(row, ensure_ascii=True) + "\n"); fh.flush()
            done.add(row_key(row)); written += 1

        recipe_items = [{**_unit_row(u), "question_index": 0} for u in units]
        recipe = [r for r in recipe_items if ("recipe", r["unit_id"], 0) not in done]
        recipe_started = time.perf_counter()
        pilot_rows = []
        pilot_answers = 0
        recipe_batches = 0
        for start in range(0, len(recipe), args.batch_size):
            chunk = recipe[start:start + args.batch_size]
            pushes = pushes_for(chunk, layers, medians, common, direction_cache)
            got = generate_fn(model, tok, layers, "Who are you?", pushes, MAX_NEW_TOKENS, args.device)
            if len(got) != len(chunk):
                raise RuntimeError(f"recipe generated {len(got)} answers for {len(chunk)} rows")
            for row_info, answer in zip(chunk, got):
                u = row_info["unit"]
                record = {"stage": "recipe", "unit_id": u["unit_id"], "arm": u["arm"],
                          "layer": int(u["layer"]),
                          "neuron_or_index": u.get("neuron") if u["arm"] == "neuron" else u.get("random_index"),
                          "dose": DOSE, "question_index": 0, "question": "Who are you?",
                          "kind": "recipe", **answer}
                fh.write(json.dumps(record, ensure_ascii=True) + "\n")
                done.add(row_key(record)); written += 1
                pilot_rows.append(record)
            fh.flush()
            recipe_batches += 1
            pilot_answers += len(chunk)
            if recipe_batches == 3:
                elapsed = max(time.perf_counter() - recipe_started, 1e-9)
                rate = pilot_answers / elapsed
                print(f"recipe first 3 batches: {rate:.2f} answers/s", flush=True)
                from triage86 import triage_answer
                shares = {arm: sum(triage_answer(r.get("answer", "")) == "changed"
                                   for r in pilot_rows if r["arm"] == arm) /
                          max(1, sum(r["arm"] == arm for r in pilot_rows))
                          for arm in ("neuron", "random")}
                print("recipe pilot changed shares: " + json.dumps(shares, sort_keys=True), flush=True)
                if rate < PILOT_MIN_RATE:
                    raise SystemExit(f"Pilot stop rule: {rate:.2f} answers/s is below {PILOT_MIN_RATE}")
                if min(shares.values()) < 0.70:
                    raise SystemExit("Pilot stop rule: recipe changed-answer share is below 0.70 in an arm")

        # A resume can have all recipe rows already written. Apply the pilot triage gate
        # to the first 192 recipe records in the accumulated output as well.
        existing_recipe = [r for r in done_rows if r.get("stage") == "recipe"]
        existing_recipe.extend(pilot_rows)
        if recipe_batches < 3 and len(existing_recipe) >= 192:
            from triage86 import triage_answer
            first = existing_recipe[:192]
            shares = {arm: sum(triage_answer(r.get("answer", "")) == "changed"
                               for r in first if r["arm"] == arm) /
                      max(1, sum(r["arm"] == arm for r in first))
                      for arm in ("neuron", "random")}
            if min(shares.values()) < 0.70:
                raise SystemExit("Pilot stop rule: recipe changed-answer share is below 0.70 in an arm")

        for qi, item in enumerate(items):
            pending = [{**_unit_row(u), "question_index": qi} for u in units
                       if ("battery", u["unit_id"], qi) not in done]
            for start in range(0, len(pending), args.batch_size):
                chunk = pending[start:start + args.batch_size]
                pushes = pushes_for(chunk, layers, medians, common, direction_cache)
                got = generate_fn(model, tok, layers, item["question"], pushes, MAX_NEW_TOKENS, args.device)
                if len(got) != len(chunk):
                    raise RuntimeError(f"battery generated {len(got)} answers for {len(chunk)} rows")
                for row_info, answer in zip(chunk, got):
                    u = row_info["unit"]
                    record = {"stage": "battery", "unit_id": u["unit_id"], "arm": u["arm"],
                              "layer": int(u["layer"]),
                              "neuron_or_index": u.get("neuron") if u["arm"] == "neuron" else u.get("random_index"),
                              "dose": DOSE, "question_index": qi, "question": item["question"],
                              "kind": item["kind"], **answer}
                    fh.write(json.dumps(record, ensure_ascii=True) + "\n")
                    done.add(row_key(record)); written += 1
                fh.flush()
    return {"answers_written": written, "questions": len(items), "units_per_arm": len(neuron_ids),
            "batch_size": args.batch_size, "seconds": time.perf_counter() - start_time}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--units", default=str(HERE / "units163.json"))
    ap.add_argument("--out", default=str(ROOT / "results/part163/answers.jsonl"))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)
    if args.batch_size != 64:
        ap.error("registered batch size is 64")
    print(json.dumps(run(args), ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
