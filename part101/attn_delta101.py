#!/usr/bin/env python3
"""attn_delta101.py -- Part 101 covariate, measured before any swap run: how far post-training moved each layer's
attention in OLMo 2 1B (Instruct 48d788e against its pre-SFT parent, the mid-training end f83cd64).

Reads every tensor under model.layers.<i>.self_attn straight from both snapshots' safetensors files with numpy (no
torch). The donor is stored in float32 and swap99.py loads both models in bfloat16 (torch_dtype=torch.bfloat16), so
donor values are first rounded to bfloat16 (round to nearest even, as torch does) and the measure is the change the
swap actually makes: ||host - donor|| / ||donor|| (Frobenius) per projection and for the layer's whole attention, plus
the share of elements whose bfloat16 value is unchanged. A wrong decode shows up as a share near 0 and a ratio near 1.
Usage: python3 -I part101/attn_delta101.py <host_snapshot_dir> <donor_snapshot_dir> <out.json>
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np


def tensor_index(snapshot):
    """Map tensor name -> (file, dtype, shape, start, end) over every safetensors file in a snapshot."""
    index = {}
    for f in sorted(Path(snapshot).glob("*.safetensors")):
        with open(f, "rb") as stream:
            n = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(n))
        for name, meta in header.items():
            if name != "__metadata__":
                s, e = meta["data_offsets"]
                index[name] = (f, meta["dtype"], tuple(meta["shape"]), 8 + n + s, 8 + n + e)
    return index


def read_f32(entry, round_to_bf16=False):
    f, dtype, shape, start, end = entry
    with open(f, "rb") as stream:
        stream.seek(start)
        raw = stream.read(end - start)
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if dtype == "F32":
        x = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        if not round_to_bf16:
            return x.copy()
        u = x.view(np.uint32).astype(np.uint64)
        u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
        return u.astype(np.uint32).view(np.float32).reshape(shape)
    raise SystemExit(f"unexpected dtype {dtype}")


def main(host_dir, donor_dir, out):
    host, donor = tensor_index(host_dir), tensor_index(donor_dir)
    names = sorted(k for k in host if ".self_attn." in k)
    if set(names) != {k for k in donor if ".self_attn." in k}:
        raise SystemExit("attention tensor names differ between the two snapshots")
    acc = {}
    for name in names:
        layer = int(name.split(".layers.")[1].split(".")[0])
        part = name.split(".self_attn.")[1].rsplit(".", 1)[0]
        h = read_f32(host[name]).astype(np.float64)
        d = read_f32(donor[name], round_to_bf16=True).astype(np.float64)
        if h.shape != d.shape:
            raise SystemExit(f"{name}: shapes differ {h.shape} {d.shape}")
        diff2, base2 = float(np.sum((h - d) ** 2)), float(np.sum(d ** 2))
        row = acc.setdefault(layer, {"parts": {}, "diff2": 0.0, "base2": 0.0, "same": 0, "n": 0})
        row["parts"][part] = round((diff2 / base2) ** 0.5, 6)
        row["diff2"] += diff2
        row["base2"] += base2
        row["same"] += int(np.count_nonzero(h == d))
        row["n"] += h.size
    layers = {str(k): {"rel_change": round((v["diff2"] / v["base2"]) ** 0.5, 6),
                       "share_unchanged": round(v["same"] / v["n"], 4), "parts": v["parts"]}
              for k, v in sorted(acc.items())}
    result = {"host": str(host_dir), "donor": str(donor_dir), "tensors": len(names), "layers": layers}
    Path(out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for k, v in layers.items():
        print(k, v["rel_change"], v["share_unchanged"], " ".join(f"{p}={x}" for p, x in sorted(v["parts"].items())))


if __name__ == "__main__":
    main(*sys.argv[1:4])
