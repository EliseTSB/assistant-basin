#!/usr/bin/env python3
"""Part 164 Arc runner: fresh neurons, fresh maker phrasings, and dose ladder."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)
from part155 import run155 as part155  # noqa: E402
from part160 import run160 as part160  # noqa: E402
from part163 import run163 as part163  # noqa: E402

DOSE = 0.8
DOSES = (0.8, 1.0, 1.2)
MAX_NEW_TOKENS = 80
MODEL_DEFAULT = part160.MODEL_DEFAULT


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    return rows


def row_key(row):
    return (row.get("stage"), row.get("unit_id"), row.get("arm"),
            float(row.get("dose", 0.0)), int(row.get("question_index", -1)))


def choose_questions():
    items = part155.questions()
    identity = [q for q in items if q["kind"] == "identity"]
    maker = [q for q in items if q["kind"] == "maker"]
    if len(identity) != 8 or len(maker) != 24:
        raise ValueError(f"expected 8 identity and 24 maker prompts; found {len(identity)} and {len(maker)}")
    used = random.Random(163).sample(maker, 8)
    chosen_ids = {q["question"] for q in used}
    fresh = [q for q in maker if q["question"] not in chosen_ids]
    if len(fresh) != 16 or chosen_ids.intersection(q["question"] for q in fresh):
        raise AssertionError("maker phrasing split must be disjoint and contain 16 new questions")
    return fresh


def pushes_for(rows, layers, medians, common, direction_cache=None):
    """Reuse Part 163's +0.8 recipe, scaling only random-direction rows by dose."""
    pushes = part163.pushes_for(rows, layers, medians, common, direction_cache)
    for layer_no, dense in pushes.items():
        for row_no, row in enumerate(rows):
            if row.get("layer") is None or int(row["layer"]) != layer_no:
                continue
            if row["arm"] == "random":
                dense[row_no].mul_(float(row["dose"]) / DOSE)
            elif float(row["dose"]) != DOSE:
                raise ValueError("neuron pushes are registered at +0.8 only")
    return pushes


def _row_info(unit, stage, question_index, dose):
    return {"unit_id": unit["unit_id"], "arm": unit["arm"], "layer": int(unit["layer"]),
            "neuron": unit.get("neuron"), "random_index": unit.get("random_index"),
            "dose": float(dose), "stage": stage, "question_index": question_index, "unit": unit}


def _answer_row(info, answer, question, kind):
    unit = info.get("unit", {})
    return {"stage": info["stage"], "unit_id": info["unit_id"], "arm": info["arm"],
            "layer": info["layer"],
            "neuron_or_index": unit.get("neuron") if info["arm"] == "neuron" else unit.get("random_index"),
            "dose": info["dose"], "question_index": info["question_index"],
            "question": question, "kind": kind, **answer}


def run(args, loader=None, generate_fn=None):
    units = json.loads(Path(args.units).read_text(encoding="utf-8"))
    if sum(u["arm"] == "neuron" for u in units) != 100 or sum(u["arm"] == "random" for u in units) != 100:
        raise ValueError("units file must contain exactly 100 pairs")
    items = choose_questions()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(output)
    done = {row_key(row) for row in existing}
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
    started = time.perf_counter()
    pilot_started = started
    pilot_answers = 0
    batch_calls = 0
    written = 0

    def append_batch(fh, infos, answers, question, kind, count_pilot=True):
        nonlocal written, pilot_answers, batch_calls
        if len(answers) != len(infos):
            raise RuntimeError(f"generated {len(answers)} answers for {len(infos)} rows")
        for info, answer in zip(infos, answers):
            row = _answer_row(info, answer, question, kind)
            if not isinstance(row.get("answer"), str) or not row["answer"].strip():
                raise RuntimeError(f"empty answer for {row['stage']} {row['unit_id']}")
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
            done.add(row_key(row))
            written += 1
        fh.flush()
        if count_pilot:
            pilot_answers += len(infos)
            batch_calls += 1
        if count_pilot and batch_calls == 3:
            rate = pilot_answers / max(time.perf_counter() - pilot_started, 1e-9)
            print(f"first 3 batches: {rate:.2f} answers/s", flush=True)
            if rate < 1.5:
                raise SystemExit(f"Pilot stop rule: {rate:.2f} answers/s is below 1.5")

    with output.open("a", encoding="utf-8", newline="\n") as fh:
        # Baseline rows are one no-push generation for each fresh phrasing.
        for qi, item in enumerate(items):
            key = ("no_push", "no_push", "plain", 0.0, qi)
            if key in done:
                continue
            infos = [{"stage": "no_push", "unit_id": "no_push", "arm": "plain", "layer": None,
                      "dose": 0.0, "question_index": qi, "unit": {}}]
            answers = generate_fn(model, tok, layers, item["question"], {}, MAX_NEW_TOKENS, args.device)
            append_batch(fh, infos, answers, item["question"], item["kind"], count_pilot=False)

        # The pilot times only the batches it counts, so its clock starts after the baseline rows.
        pilot_started = time.perf_counter()

        # Recipe validity check: all 100 random directions, same named question and +0.8.
        recipe = [_row_info(u, "recipe", 0, DOSE) for u in units if u["arm"] == "random"]
        recipe = [r for r in recipe if ("recipe", r["unit_id"], r["arm"], DOSE, 0) not in done]
        for start in range(0, len(recipe), args.batch_size):
            chunk = recipe[start:start + args.batch_size]
            pushes = pushes_for(chunk, layers, medians, common, direction_cache)
            answers = generate_fn(model, tok, layers, "Who are you?", pushes, MAX_NEW_TOKENS, args.device)
            append_batch(fh, chunk, answers, "Who are you?", "recipe")

        # Battery: each phrasing gets one independently batched row per unit and dose.
        for qi, item in enumerate(items):
            for dose in DOSES:
                selected = [u for u in units if u["arm"] == "neuron"] if dose == DOSE else []
                selected += [u for u in units if u["arm"] == "random"]
                pending = [_row_info(u, "battery", qi, dose) for u in selected
                           if ("battery", u["unit_id"], u["arm"], float(dose), qi) not in done]
                for start in range(0, len(pending), args.batch_size):
                    chunk = pending[start:start + args.batch_size]
                    pushes = pushes_for(chunk, layers, medians, common, direction_cache)
                    answers = generate_fn(model, tok, layers, item["question"], pushes, MAX_NEW_TOKENS, args.device)
                    append_batch(fh, chunk, answers, item["question"], item["kind"])
    return {"answers_written": written, "questions": len(items), "units_per_arm": 100,
            "batch_size": args.batch_size, "seconds": time.perf_counter() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--units", default=str(HERE / "units164.json"))
    parser.add_argument("--out", default=str(ROOT / "results/part164/answers.jsonl"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.batch_size != 64:
        parser.error("registered batch size is 64")
    print(json.dumps(run(args), ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
