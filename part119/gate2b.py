#!/usr/bin/env python3
"""Part 119 Amendment 1: the 39 single rows through Part 92's own single-layer hook path, no padding."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "part92"))
sys.path.insert(0, str(HERE))

import common92 as c  # noqa: E402
import rows119  # noqa: E402

ALPHA = 0.8
QUESTION = "Who are you?"


def main() -> int:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="xpu")
    args = parser.parse_args()
    units = rows119.load_units()
    tokenizer, model = c.load_model(args.model, args.device)
    layers = c.get_layers(model)
    medians = c.residual_medians(model, tokenizer, "chat", args.device)
    by_layer: dict[int, list[dict]] = {}
    for unit in units:
        by_layer.setdefault(int(unit["layer"]), []).append(unit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "gate2b.jsonl").open("w", encoding="utf-8") as stream:
        for layer_no, group in sorted(by_layer.items()):
            weight = layers[layer_no].mlp.down_proj.weight
            target = ALPHA * medians[layer_no]
            pushes = torch.stack([c.neuron_push(weight, int(u["neuron"]), target) for u in group])
            answers = c.generate_batch(model, tokenizer, [QUESTION] * len(group), "chat",
                                       layers[layer_no], pushes, 60, args.device)
            for unit, answer in zip(group, answers):
                stream.write(json.dumps({"id": unit["id"], "answer": answer}, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"layer {layer_no}: {len(group)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
