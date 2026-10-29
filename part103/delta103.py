#!/usr/bin/env python3
"""delta103.py -- Part 103 covariate, measured before any swap run: how far each OLMo 2 1B training stage's
attention or MLP is from Instruct's, layer by layer. Same measure as part101/attn_delta101.py (relative Frobenius change
over every tensor of the component, donor rounded to bfloat16 as swap99.py loads it, plus the share of unchanged elements),
computed with torch so the pickled SFT and DPO checkpoints can be read. The mid-training end is measured again as a
cross-check: its numbers must match Part 101's numpy values (results/part101/attn-delta.json) to 1e-4.
Usage (Windows GPU venv, CPU only): python -I part103/delta103.py <self_attn|mlp> <host_dir> <out.json> <name>=<donor_dir> ...
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


COMPONENT = ".self_attn."


def load_attn(snapshot):
    snap, out = Path(snapshot), {}
    files = sorted(snap.glob("*.safetensors"))
    if files:
        for f in files:
            with safe_open(str(f), framework="pt") as handle:
                for key in handle.keys():
                    if COMPONENT in key:
                        out[key] = handle.get_tensor(key).to(torch.bfloat16)
    else:
        state = torch.load(str(snap / "pytorch_model.bin"), map_location="cpu", mmap=True, weights_only=True)
        out = {k: v.to(torch.bfloat16) for k, v in state.items() if COMPONENT in k}
        del state
    return out


def compare(host, donor):
    if set(host) != set(donor):
        raise SystemExit("tensor names differ between snapshots")
    acc = {}
    for name in sorted(host):
        layer = int(name.split(".layers.")[1].split(".")[0])
        h, d = host[name].double(), donor[name].double()
        row = acc.setdefault(layer, [0.0, 0.0, 0, 0])
        row[0] += float((h - d).pow(2).sum())
        row[1] += float(d.pow(2).sum())
        row[2] += int((host[name] == donor[name]).sum())
        row[3] += h.numel()
    return {str(k): {"rel_change": round((v[0] / v[1]) ** 0.5, 6), "share_unchanged": round(v[2] / v[3], 4)}
            for k, v in sorted(acc.items())}


def main(component, host_dir, out, *donors):
    global COMPONENT
    COMPONENT = f".{component}."
    host = load_attn(host_dir)
    result = {"component": component, "host": host_dir, "donors": {}}
    for spec in donors:
        name, path = spec.split("=", 1)
        result["donors"][name] = {"path": path, "layers": compare(host, load_attn(path))}
        print(name, " ".join(f"{k}:{v['rel_change']}" for k, v in result["donors"][name]["layers"].items()), flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
