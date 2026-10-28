#!/usr/bin/env python3
"""timing86.py -- throughput of Part 86's real generation path by batch size, token budget and observer.

Written by Claude, 2026-10-28, after the L22 dry run measured 2.52 answers/s at batch 16 and 48 tokens (gate 2 needs
3.83 to put 17 layers under 12 GPU-hours). It calls common86.generate_batch exactly as the screen does, with real
neuron pushes at L22 dose 0.8, and times each arm after one warm-up batch. Output: one JSON file. Not part of any
result; it sizes the gate-2 amendment.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import common86 as c  # noqa: E402

ARMS = [(16, 48, False), (16, 48, True), (32, 48, False), (32, 48, True), (48, 48, True), (32, 32, True),
        (64, 32, True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--median", type=float, default=24.821300506591797, help="L22 median from the dry run")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--arms", default=None, help="batch:tokens:observer,... e.g. 32:48:1,64:48:1")
    args = ap.parse_args()
    import torch
    import observe
    tok, model = c.load_model(args.device)
    layers = c.find_layers(model)
    layer = layers[22]
    weight = layer.mlp.down_proj.weight.detach()
    deltas = c.neuron_deltas(weight, args.median, count=int(weight.shape[1])).detach()
    width = int(weight.shape[1])
    rng = random.Random(8686)
    arms = ARMS if not args.arms else [(int(b), int(t), o == "1") for b, t, o in
                                       (a.split(":") for a in args.arms.split(","))]
    results = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for batch, tokens, use_obs in arms:
        obs = observe.install(model, tok, out.parent / f"timing-observe-{batch}-{tokens}.jsonl") if use_obs else None
        try:
            times = []
            for i in range(args.batches + 1):
                ids = rng.sample(range(width), batch)
                pushes = torch.zeros((batch, width), dtype=torch.float32, device=weight.device)
                pushes[torch.arange(batch), torch.tensor(ids)] = deltas[torch.tensor(ids, device=deltas.device)]
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.synchronize()
                start = time.monotonic()
                c.generate_batch(model, tok, layer, c.SCREEN_QUESTION, pushes, args.device, tokens, do_sample=False)
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.synchronize()
                if i > 0:
                    times.append(time.monotonic() - start)
        finally:
            if obs is not None:
                obs.close()
        rate = batch * len(times) / sum(times)
        row = {"batch": batch, "tokens": tokens, "observer": use_obs, "seconds_per_batch": times,
               "answers_per_second": rate, "projected_hours_165376": 165376 / rate / 3600}
        results.append(row)
        print(json.dumps(row), flush=True)
        out.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
