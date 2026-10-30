#!/usr/bin/env python3
"""Part 119 Amendment 2: the 39 single rows at batch size 1 through both hook paths, side by side."""
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
import run119  # noqa: E402

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
    prompt = c.prompt_ids(tokenizer, QUESTION, "chat")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "gate2c.jsonl").open("w", encoding="utf-8") as stream:
        for unit in units:
            layer_no = int(unit["layer"])
            vector = c.neuron_push(layers[layer_no].mlp.down_proj.weight, int(unit["neuron"]),
                                   ALPHA * medians[layer_no])
            part92_answer = c.generate_batch(model, tokenizer, [QUESTION], "chat", layers[layer_no],
                                             vector.unsqueeze(0), run119.ANSWER_TOKENS, args.device)[0]
            cache = {"_units": units, (unit["id"], "neuron"): vector}
            row = {"row_id": f"single:{unit['id']}", "pushes": [{"unit_id": unit["id"], "arm": "neuron"}]}
            tokens = run119._run_generate_batch(model, tokenizer, [row], layers, cache, prompt, args.device)[0]
            stream.write(json.dumps({"id": unit["id"], "part92_answer": part92_answer,
                                     "row_hook_answer": tokenizer.decode(tokens, skip_special_tokens=True)},
                                    ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{unit['id']} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
