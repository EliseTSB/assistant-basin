#!/usr/bin/env python3
"""The alien-twin sanity read: OLMo 2 L10 N6978 through the fitted R-lens.

``write_R`` (Neuron Lens' ``08_reads.py`` convention) is
``softmax(W_U · norm(J_L^out · w))`` with ``w = W_down[:, n]`` and ``norm`` the
final RMSNorm including its weight. This script needs ``fit.pt`` from
``fit_olmo_lens.py`` (the full-stack pass records every layer, so L10 is free).

Also reads three pseudorandom neurons at the same layer as controls, so a
generic "the model always says X" read is visible rather than mistaken for a
neuron-specific one.
"""
import argparse
import json
import random
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", type=Path,
                    default=Path(__file__).resolve().parent / "results" / "prime-lens"
                    / "olmo2-1b-instruct" / "fit.pt")
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B-Instruct")
    ap.add_argument("--revision", default="48d788eca847d4d7548f375ad03d3c9312f6139e")
    ap.add_argument("--cache-dir", type=Path, default=Path("/mnt/e/neuron-probe/hf/hub"))
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--neuron", type=int, default=6978)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neuron_lens.transport import Accumulator

    blob = torch.load(args.fit, map_location="cpu", weights_only=False)
    acc = Accumulator.from_state(blob["state_r"])
    J = acc.mean("out")[args.layer].float()  # [d, d]

    lm = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, cache_dir=str(args.cache_dir),
        local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation="eager").eval().requires_grad_(False)
    stack = lm.model
    tok = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, cache_dir=str(args.cache_dir), local_files_only=True)
    Wd = stack.layers[args.layer].mlp.down_proj.weight.float()  # [d, ff]
    gamma = stack.norm.weight.float()  # final RMSNorm weight
    eps = stack.norm.variance_epsilon
    WU = lm.lm_head.weight.float()  # [vocab, d]

    def read(n: int):
        w = Wd[:, n]
        x = J @ w
        normed = gamma * x / torch.sqrt(x.pow(2).mean() + eps)
        logits = WU @ normed
        probs = torch.softmax(logits, -1)
        vals, ids = probs.topk(args.top)
        return [{"token": tok.decode([int(i)]), "id": int(i), "p": round(float(v), 5)}
                for v, i in zip(vals, ids)]

    torch.manual_seed(0)
    controls = random.Random(7).sample(range(Wd.shape[1]), 3)
    out = {"layer": args.layer, "neuron": args.neuron, "model": args.model,
           "revision": args.revision, "fit": str(args.fit),
           "read": read(args.neuron),
           "controls": {n: read(n) for n in controls}}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
