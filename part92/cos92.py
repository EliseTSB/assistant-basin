#!/usr/bin/env python3
"""Compare listed W_down neuron columns between two local OLMo 2 snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def _shard_for(root: Path, tensor_name: str, index_name: str):
    index_path = root / index_name
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as stream:
            return root / json.load(stream)["weight_map"][tensor_name]
    return None


def _load_weight(snapshot: str | Path, layer: int, cache: dict):
    root = Path(snapshot).expanduser()
    if layer in cache:
        return cache[layer]
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    shard = _shard_for(root, name, "model.safetensors.index.json")
    if shard is not None:
        with safe_open(str(shard), framework="pt", device="cpu") as stream:
            weight = stream.get_tensor(name)
    elif (root / "model.safetensors").is_file():
        with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as stream:
            weight = stream.get_tensor(name)
    else:
        shard = _shard_for(root, name, "pytorch_model.bin.index.json")
        if shard is None:
            shard = root / "pytorch_model.bin"
        if not shard.is_file():
            raise FileNotFoundError(f"no safetensors or pytorch_model.bin weights in {root}")
        state = torch.load(shard, map_location="cpu", weights_only=True)
        if name not in state:
            raise KeyError(f"{name} is absent from {shard}")
        weight = state[name]
    if weight.ndim != 2:
        raise ValueError(f"expected matrix for {name}, got {tuple(weight.shape)}")
    cache[layer] = weight.float()
    return cache[layer]


def compare(snapshot_a, snapshot_b, neurons):
    cache_a, cache_b, rows = {}, {}, []
    for neuron in neurons:
        layer = int(neuron["layer"])
        index = int(neuron["neuron"])
        left = _load_weight(snapshot_a, layer, cache_a)
        right = _load_weight(snapshot_b, layer, cache_b)
        if left.shape != right.shape or not 0 <= index < left.shape[1]:
            raise ValueError(f"incompatible or missing W_down column at layer {layer}, neuron {index}")
        a, b = left[:, index], right[:, index]
        denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
        if float(denom.item()) == 0:
            raise ValueError(f"zero-norm W_down column at layer {layer}, neuron {index}")
        rows.append({"id": neuron["id"], "layer": layer, "neuron": index,
                     "cosine": float(torch.dot(a, b).div(denom).item())})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot-a", required=True)
    ap.add_argument("--snapshot-b", required=True)
    ap.add_argument("--neurons", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    with Path(args.neurons).open(encoding="utf-8") as stream:
        neurons = json.load(stream)
    rows = compare(args.snapshot_a, args.snapshot_b, neurons)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "out": str(output)}), flush=True)


if __name__ == "__main__":
    main()
