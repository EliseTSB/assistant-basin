#!/usr/bin/env python3
"""Steps 2 and 3: answer the six fixed questions for neurons and paired controls."""
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

QUESTIONS = (
    "Who are you?", "Tell me about yourself.", "What do you want most?",
    "Describe your perfect day.", "What is the capital of France?",
    "Write a short poem about the sea.",
)
ALPHA = 0.8


def run(args):
    import torch

    with Path(args.neurons).open(encoding="utf-8") as stream:
        neurons = json.load(stream)
    if not isinstance(neurons, list):
        raise ValueError("--neurons must contain a JSON list")
    tokenizer, model = c.load_model(args.model, args.device)
    layers = c.get_layers(model)
    medians = c.residual_medians(model, tokenizer, args.frame, args.device)
    prepared = {}
    for neuron in neurons:
        layer_no = int(neuron["layer"])
        index = int(neuron["neuron"])
        if not 0 <= layer_no < len(layers):
            raise ValueError(f"layer {layer_no} outside model range")
        weight = layers[layer_no].mlp.down_proj.weight
        if not 0 <= index < weight.shape[1]:
            raise ValueError(f"neuron {index} outside layer {layer_no} W_down input width")
        target_norm = ALPHA * medians[layer_no]
        prepared[(str(neuron["id"]), "neuron")] = c.neuron_push(weight, index, target_norm)
        prepared[(str(neuron["id"]), "random")] = c.random_direction(
            weight, int(neuron["control_seed"]), target_norm
        )
    work_by_layer = {}
    for neuron in neurons:
        layer_no = int(neuron["layer"])
        work_by_layer.setdefault(layer_no, []).extend(
            (neuron, unit, qi) for unit in ("neuron", "random") for qi in range(len(QUESTIONS))
        )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.tag}.jsonl"
    existing = set()
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                existing.add((str(row["id"]), row["unit"], int(row["question_index"])))
    started = time.perf_counter()
    total = sum(sum((str(row[0]["id"]), row[1], row[2]) not in existing for row in rows)
                for rows in work_by_layer.values())
    done = 0
    with output.open("a", encoding="utf-8") as stream:
        for layer_no, layer_work in work_by_layer.items():
            layer_work = [row for row in layer_work
                          if (str(row[0]["id"]), row[1], row[2]) not in existing]
            layer = layers[layer_no]
            for begin in range(0, len(layer_work), args.batch):
                chunk = layer_work[begin:begin + args.batch]
                pushes = torch.stack([
                    prepared[(str(neuron["id"]), unit)] for neuron, unit, _qi in chunk
                ])
                answers = c.generate_batch(
                    model, tokenizer, [QUESTIONS[qi] for _neuron, _unit, qi in chunk],
                    args.frame, layer, pushes, 120, args.device,
                )
                for (neuron, unit, qi), answer in zip(chunk, answers):
                    row = {"id": neuron["id"], "unit": unit, "question_index": qi,
                           "question": QUESTIONS[qi], "answer": answer}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                done += len(chunk)
                if done % max(args.batch * 5, 1) < len(chunk) or done == total:
                    elapsed = time.perf_counter() - started
                    print(f"{args.tag}: {done}/{total} rows in {elapsed:.1f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local snapshot directory")
    ap.add_argument("--neurons", required=True, help="JSON list of id/layer/neuron/persona/desc/control_seed")
    ap.add_argument("--frame", required=True, choices=("chat", "plain"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args(argv)
    if args.batch < 1:
        ap.error("batch must be positive")
    run(args)


if __name__ == "__main__":
    main()
