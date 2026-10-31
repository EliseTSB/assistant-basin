#!/usr/bin/env python3
"""Part 155 signed dose ladder, using Part 148 generation and Part 153 prompts."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part153 import battery153 as part153  # noqa: E402
from part148 import battery148 as part148  # noqa: E402

DOSES = (0.2, 0.4, 0.8, 1.2, 1.6, -0.2, -0.4, -0.8, -1.2, -1.6)
REGISTERED_COORDS = {(24, 5017), (21, 7134), (23, 1743), (23, 2993)}
MODEL_DEFAULT = "E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_NEW_TOKENS = 120
RANDOM_SEED_BASE = part148.RANDOM_SEED_BASE


def _common():
    part86 = str(ROOT / "part86")
    if part86 not in sys.path:
        sys.path.insert(0, part86)
    import common86
    return common86


def questions():
    """Part 153 maker, identity and guard items (selection items are out of scope)."""
    return [q for q in part153.QUESTIONS if q["kind"] in ("maker", "identity", "guard")]


def registered_units(units_path):
    """Resolve Part 153's registered coordinate/company list against its source."""
    return [u for u in part153.fixed_units(units_path)
            if (int(u["layer"]), int(u["neuron"])) in REGISTERED_COORDS]


def dose_rows(units):
    rows = []
    for unit in units:
        for dose in DOSES:
            for arm in ("neuron", "random"):
                rows.append({**unit, "unit_id": unit["id"], "arm": arm, "dose": dose})
    rows.append({"id": "plain", "unit_id": "plain", "layer": None, "neuron": None,
                 "company": None, "arm": "plain", "dose": 0.0})
    return rows


def pushes_for(rows, layers, medians, common):
    """Build row-isolated pushes; signed neuron deltas and signed matched random vectors."""
    import torch
    pushes = {}
    for layer_no in sorted({int(row["layer"]) for row in rows if row["layer"] is not None}):
        weight = layers[layer_no].mlp.down_proj.weight
        layer_push = torch.zeros((len(rows), int(weight.shape[1])), dtype=torch.float32, device=weight.device)
        median = float(medians[str(layer_no)] if str(layer_no) in medians else medians[layer_no])
        for i, row in enumerate(rows):
            if row["layer"] is None or int(row["layer"]) != layer_no:
                continue
            neuron = int(row["neuron"])
            dose = float(row["dose"])
            if row["arm"] == "neuron":
                layer_push[i, neuron] = common.neuron_delta(weight, neuron, median, dose=dose)
            elif row["arm"] == "random":
                seed = RANDOM_SEED_BASE + 10000 * layer_no + neuron
                direction = common.scaled_random_directions(weight, 1, seed, abs(dose) * median)[0]
                layer_push[i] = direction.to(layer_push.device) * (1.0 if dose >= 0 else -1.0)
            else:
                raise ValueError(f"unknown arm {row['arm']!r}")
        pushes[layer_no] = layer_push
    return pushes


def generate(model, tok, layers, question, layer_pushes, max_new_tokens, device, common):
    return part148.generate(model, tok, layers, question, layer_pushes, max_new_tokens, device, common)


def run(args, loader=None, generate_fn=None):
    common = _common()
    units = registered_units(args.units)
    if len(units) != 4:
        raise ValueError(f"expected the four registered units, resolved {len(units)}")
    medians = json.loads(Path(args.medians).read_text(encoding="utf-8"))
    items = questions()
    if args.stage == "pilot":
        items = [q for q in items if q["kind"] == "maker"][:2]
    rows = dose_rows(units)
    output = Path(args.out) if args.out else ROOT / ("results/part155-pilot/answers.jsonl" if args.stage == "pilot" else "results/part155/answers.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        with output.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    old = json.loads(line)
                    done.add((old["unit_id"], old["arm"], float(old["dose"]), int(old["question_index"])))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid resume row in {output}:{line_no}") from exc
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common.REVISION, args.device)
        model.requires_grad_(False)
        model.eval()
    else:
        tok, model = loader(args.device)
    layers = common.find_layers(model)
    generate_fn = generate_fn or (lambda *a: generate(*a, common=common))
    started = time.perf_counter()
    written = 0
    with output.open("a", encoding="utf-8", newline="\n") as fh:
        for qi, item in enumerate(items):
            expected_keys = [(r["unit_id"], r["arm"], float(r["dose"]), qi) for r in rows]
            if all(key in done for key in expected_keys):
                continue
            # One 81-row batch per question. Even the no-push row occupies its own all-zero row.
            layer_pushes = pushes_for(rows, layers, medians, common)
            answers = generate_fn(model, tok, layers, item["question"], layer_pushes, MAX_NEW_TOKENS, args.device)
            if len(answers) != len(rows):
                raise RuntimeError(f"generation returned {len(answers)} answers for {len(rows)} rows")
            if any(not isinstance(a.get("answer"), str) or not a["answer"].strip() for a in answers):
                raise RuntimeError(f"empty answer in question {qi}; stopping the run")
            for row, answer in zip(rows, answers):
                key = (row["unit_id"], row["arm"], float(row["dose"]), qi)
                if key in done:
                    continue
                record = {"unit_id": row["unit_id"], "arm": row["arm"], "dose": float(row["dose"]),
                          "company": row["company"], "question_index": qi, "question": item["question"],
                          "kind": item["kind"], **answer}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                done.add(key)
                written += 1
            # The no-push control is the last all-zero row in the same 81-row batch.
            plain = rows[-1]
            pkey = (plain["unit_id"], plain["arm"], 0.0, qi)
            if pkey not in done:
                plain_answer = answers[-1]
                record = {"unit_id": "plain", "arm": "plain", "dose": 0.0, "company": None,
                          "question_index": qi, "question": item["question"], "kind": item["kind"], **plain_answer}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                done.add(pkey)
                written += 1
            fh.flush()
    return {"answers_written": written, "questions": len(items), "units": len(units),
            "dose_count": len(DOSES), "rows_per_question": len(rows), "stage": args.stage,
            "seconds": time.perf_counter() - started}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("pilot", "main"), default="main")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--units", default=str(ROOT / "results/part112/units.json"))
    ap.add_argument("--medians", default=str(ROOT / "results/part112/medians.json"))
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
