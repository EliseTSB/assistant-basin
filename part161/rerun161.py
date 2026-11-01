#!/usr/bin/env python3
"""Part 161 Amendment 1: regenerate the 18 neurons reused from Part 160, so every unit comes from this code path."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run161  # noqa: E402
from run161 import part155, part160, read_jsonl, row_key, MAX_NEW_TOKENS, MODEL_DEFAULT  # noqa: E402

DOSES = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4)


def plan_rows(plan):
    """One row per reused neuron and dose, with the layer and neuron named in the unit id."""
    rows = []
    for uid in sorted(plan["reused_neuron_ids"]):
        layer, neuron = uid[1:].split("N")
        for dose in DOSES:
            rows.append({"unit_id": uid, "arm": "neuron", "dose": dose, "layer": int(layer), "neuron": int(neuron)})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="xpu")
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--plan", default=str(ROOT / "results/part161/plan.json"))
    p.add_argument("--out", default=str(ROOT / "results/part161/rerun.jsonl"))
    args = p.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows = plan_rows(plan)
    if len(plan["reused_neuron_ids"]) != 18 or len(rows) != 108:
        raise SystemExit(f"expected 18 reused neurons, found {len(plan['reused_neuron_ids'])}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {row_key(r) for r in read_jsonl(out)} if out.exists() else set()
    common = part155._common()
    medians = json.loads((ROOT / "results/part112/medians.json").read_text(encoding="utf-8"))
    import neuron_report as nr
    tok, model = nr.load_model(str(args.model), common.REVISION, args.device)
    model.requires_grad_(False)
    model.eval()
    layers = common.find_layers(model)
    qitems = [q for q in part155.questions() if q["kind"] == "maker"]
    start = time.perf_counter()
    written = 0
    with out.open("a", encoding="utf-8", newline="\n") as fh:
        for qi, item in enumerate(qitems):
            pending = [r for r in rows if ("ladder", r["unit_id"], r["arm"], r["dose"], qi) not in done]
            for i in range(0, len(pending), 64):
                chunk = pending[i:i + 64]
                pushes = part155.pushes_for(chunk, layers, medians, common)
                got = part160.generate_checked(model, tok, layers, item["question"], pushes, MAX_NEW_TOKENS,
                                               args.device, common=common)
                if len(got) != len(chunk):
                    raise RuntimeError(f"generated {len(got)} answers for {len(chunk)} rows")
                for row, answer in zip(chunk, got):
                    fh.write(json.dumps({"stage": "ladder", **row, "question_index": qi,
                                         "question": item["question"], **answer}, ensure_ascii=True) + "\n")
                    written += 1
                fh.flush()
            print(f"phrasing {qi}: {written} rows, {written / (time.perf_counter() - start):.2f} answers/s", flush=True)
    print(json.dumps({"written": written, "seconds": time.perf_counter() - start}), flush=True)


if __name__ == "__main__":
    main()
