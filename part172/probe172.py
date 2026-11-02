#!/usr/bin/env python3
"""Part 172 premise probe: no pushes. Does the plain model name Qwen on the new prompts, does the role take there, does
zeroing Part 66's four makers reach Google under greedy decoding, and what does zeroing L27 N3325 do to each?

PROCEDURE lesson of 2026-11-01: run the premise before building on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86"), str(ROOT / "part171")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import common86
import run171
from prompts171 import TARGETS
from prompts172 import NEW44

MAKERS = ((6, 753), (7, 1169), (11, 8868), (29, 776))
WRITER = ((27, 3325),)
CONDITIONS = {
    "none": (None, ()),
    "role": ("google", ()),
    "zero4": (None, MAKERS),
    "none_w": (None, WRITER),
    "role_w": ("google", WRITER),
    "zero4_w": (None, MAKERS + WRITER),
}


class ZeroUnits:
    """Forward pre-hook on mlp.down_proj: zero the listed neurons at every position."""

    def __init__(self, module, units):
        import torch
        self.units = None
        self.raw = list(units)
        self.torch = torch
        self.handle = module.register_forward_pre_hook(self)

    def __call__(self, _module, args):
        x = args[0].clone()
        x[..., self.raw] = 0
        return (x,) + tuple(args[1:])

    def remove(self):
        self.handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--model", default=run171.MODEL_DEFAULT)
    parser.add_argument("--out", default="results/part172/premise.jsonl")
    args = parser.parse_args()
    import neuron_report as nr
    tok, model = nr.load_model(str(args.model), common86.REVISION, args.device)
    model.requires_grad_(False)
    model.eval()
    layers = common86.find_layers(model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        for name, (target, units) in CONDITIONS.items():
            by_layer = {}
            for layer, unit in units:
                by_layer.setdefault(layer, []).append(unit)
            hooks = [ZeroUnits(layers[layer].mlp.down_proj, kept) for layer, kept in sorted(by_layer.items())]
            roles = [TARGETS[target] if target else None] * len(NEW44)
            try:
                answers = run171.generate_batch(model, tok, layers, list(NEW44), roles, args.device, seed=172)
            finally:
                for hook in hooks:
                    hook.remove()
            hits = 0
            for index, (question, answer) in enumerate(zip(NEW44, answers)):
                hit = bool(run171.hit_match(answer["first40"], "google"))
                hits += hit
                stream.write(json.dumps({"condition": name, "prompt": index, "question": question, "hit": hit,
                                         "answer": answer["answer"], "first40": answer["first40"],
                                         "token_ids": answer["token_ids"]}, ensure_ascii=True) + "\n")
            stream.flush()
            print(json.dumps({"condition": name, "hits": hits, "of": len(NEW44)}), flush=True)
    print("step probe172 done", flush=True)


if __name__ == "__main__":
    main()
