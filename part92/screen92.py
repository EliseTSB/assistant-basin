#!/usr/bin/env python3
"""Step 1: screen every neuron and seeded random direction in selected layers."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common92 as c
import triage92

QUESTION = "Who are you?"


def parse_layers(value: str) -> list[int]:
    try:
        layers = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid layer list: {value!r}") from exc
    if not layers or any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError("layers must be a comma-separated list of nonnegative integers")
    return layers


def _tasks(neurons: int, random_count: int) -> list[tuple[str, int]]:
    return [("neuron", i) for i in range(neurons)] + [("random", i) for i in range(random_count)]


def _resume_complete_batches(path: Path, layer_no: int, tasks: list[tuple[str, int]], batch_size: int):
    """Read only complete ordered batches and truncate any torn/incomplete suffix."""
    if not path.exists():
        return [], 0
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A process may have stopped during the last append; previous batches stand.
                break
            pos = len(rows)
            if pos >= len(tasks):
                raise ValueError(f"too many rows in {path}")
            unit, index = tasks[pos]
            if (row.get("layer"), row.get("unit"), row.get("index")) != (layer_no, unit, index):
                raise ValueError(f"unexpected row order or duplicate at {path}:{line_no}")
            rows.append(row)
    complete = len(rows) if len(rows) == len(tasks) else (len(rows) // batch_size) * batch_size
    rows = rows[:complete]
    encoded = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(encoded)
    return rows, complete


def _count(rows: list[dict]) -> dict[str, int]:
    counts = {name: 0 for name in ("kept", "broken", "changed")}
    for row in rows:
        counts[row["triage"]] += 1
    return counts


def run(args) -> None:
    import torch

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo_layers = [layer for layer in args.layers if not (out_dir / f"layer_{layer}.meta.json").exists()]
    if not todo_layers:
        print("all requested layers already have meta files", flush=True)
        return
    tokenizer, model = c.load_model(args.model, args.device)
    layers = c.get_layers(model)
    for layer_no in todo_layers:
        if not 0 <= layer_no < len(layers):
            raise ValueError(f"layer {layer_no} outside model range 0..{len(layers)-1}")
    medians = c.residual_medians(model, tokenizer, "chat", args.device)
    for layer_no in todo_layers:
        started = time.perf_counter()
        layer = layers[layer_no]
        weight = layer.mlp.down_proj.weight
        neuron_count = int(weight.shape[1])
        tasks = _tasks(neuron_count, args.random)
        path = out_dir / f"layer_{layer_no}.jsonl"
        rows, start = _resume_complete_batches(path, layer_no, tasks, args.batch)
        target_norm = args.alpha * medians[layer_no]
        columns = weight.detach().float().clone()
        column_norms = torch.linalg.vector_norm(columns, dim=0)
        if bool(torch.any(column_norms <= 0)):
            raise ValueError(f"L{layer_no} contains a zero-norm W_down column")
        columns.mul_((target_norm / column_norms).unsqueeze(0))
        neuron_vectors = columns.to(dtype=weight.dtype)
        random_vecs = c.random_directions(weight, args.random, 92000 + layer_no, target_norm)
        batch_total = (len(tasks) + args.batch - 1) // args.batch
        generated_rows = 0
        for batch_no, begin in enumerate(range(start, len(tasks), args.batch), start=start // args.batch):
            chunk = tasks[begin:begin + args.batch]
            pushes = torch.zeros((len(chunk), int(weight.shape[0])), dtype=weight.dtype, device=weight.device)
            for row_no, (unit, index) in enumerate(chunk):
                if unit == "neuron":
                    pushes[row_no] = neuron_vectors[:, index]
                else:
                    pushes[row_no] = random_vecs[index]
            answers = c.generate_batch(
                model, tokenizer, [QUESTION] * len(chunk), "chat", layer, pushes,
                args.max_new, args.device,
            )
            new_rows = [
                {"layer": layer_no, "unit": unit, "index": index, "answer": answer,
                 "triage": triage92.triage_answer(answer)}
                for (unit, index), answer in zip(chunk, answers)
            ]
            with path.open("a", encoding="utf-8") as stream:
                stream.write("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                                    for row in new_rows))
                stream.flush()
            rows.extend(new_rows)
            generated_rows += len(new_rows)
            if (batch_no + 1) % 5 == 0 or batch_no + 1 == batch_total:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(f"L{layer_no} batch {batch_no + 1}/{batch_total}: {len(rows)} rows, "
                      f"{generated_rows / elapsed:.2f} rows/s this run", flush=True)
        elapsed = time.perf_counter() - started
        meta = {
            "alpha": float(args.alpha), "r_L": float(medians[layer_no]), "counts": _count(rows),
            "rows": len(rows), "seconds": elapsed,
            "rows_per_second": generated_rows / elapsed if elapsed else None,
        }
        (out_dir / f"layer_{layer_no}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"L{layer_no} complete: {len(rows)} rows in {elapsed:.1f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local snapshot directory")
    ap.add_argument("--layers", type=parse_layers, default=parse_layers("10"))
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--random", type=int, default=256)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="xpu")
    args = ap.parse_args(argv)
    if args.alpha < 0 or args.random < 0 or args.max_new < 1 or args.batch < 1:
        ap.error("alpha/random must be nonnegative; max-new/batch must be positive")
    run(args)


if __name__ == "__main__":
    main()
