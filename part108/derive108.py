#!/usr/bin/env python3
"""Part 108: build one model from two stored OLMo 2 1B snapshots (PART108-DESIGN.md).

forward:  every parameter becomes 2 x host - donor, the donor's distance from the host in the other direction.
scramble: every parameter becomes host + s x (host - donor), where s is a random sign for each output row of a matrix
          and each element of a vector, drawn from --seed in sorted parameter order. Each row is then either the donor's
          row or the forward row.

Arithmetic is float32 in row chunks, cast back to the host's dtype. Nothing is written until every parameter has been
checked and built. Writes model.safetensors, copies the host's config and tokenizer files, and writes derive108.json
(distance moved over the donor's distance, sign counts, output hash).

Written by Claude (Opus 5.5), 2026-10-29.
Usage: derive108.py --host DIR --donor DIR --mode {forward,scramble} [--seed 108] --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

SMALL_FILES = ("config.json", "generation_config.json", "merges.txt", "special_tokens_map.json", "tokenizer.json",
               "tokenizer_config.json", "vocab.json")
CHUNK_ROWS = 4096


def open_state(snapshot: Path):
    """(sorted names, getter) for a snapshot's weights: model.safetensors read lazily, or pytorch_model.bin mmapped."""
    import torch

    st = snapshot / "model.safetensors"
    if st.exists():
        from safetensors import safe_open
        handle = safe_open(str(st), framework="pt")
        return sorted(handle.keys()), handle.get_tensor
    state = torch.load(str(snapshot / "pytorch_model.bin"), map_location="cpu", mmap=True, weights_only=True)
    return sorted(state), state.__getitem__


def row_signs(rows: int, generator):
    import torch
    return torch.randint(0, 2, (rows,), generator=generator).to(torch.float32) * 2 - 1


def build(host_dir: Path, donor_dir: Path, mode: str, seed: int) -> tuple[dict, dict]:
    import torch

    if mode not in ("forward", "scramble"):
        raise ValueError("mode must be forward or scramble")
    host_names, host_get = open_state(host_dir)
    donor_names, donor_get = open_state(donor_dir)
    if host_names != donor_names:
        only_host = sorted(set(host_names) - set(donor_names))[:5]
        only_donor = sorted(set(donor_names) - set(host_names))[:5]
        raise ValueError(f"parameter names differ: only in host {only_host}, only in donor {only_donor}")
    generator = torch.Generator().manual_seed(int(seed))
    out, moved_sq, target_sq, plus, minus, elements = {}, 0.0, 0.0, 0, 0, 0
    for name in host_names:
        host, donor = host_get(name), donor_get(name)
        if tuple(host.shape) != tuple(donor.shape) or host.dtype != donor.dtype:
            raise ValueError(f"{name}: host {tuple(host.shape)} {host.dtype} vs donor {tuple(donor.shape)} {donor.dtype}")
        flat_host = host.reshape(1) if host.dim() == 0 else host
        flat_donor = donor.reshape(1) if donor.dim() == 0 else donor
        rows = flat_host.shape[0]
        signs = row_signs(rows, generator) if mode == "scramble" else None
        if signs is not None:
            plus += int((signs > 0).sum())
            minus += int((signs < 0).sum())
        result = torch.empty_like(flat_host)
        for begin in range(0, rows, CHUNK_ROWS):
            h = flat_host[begin:begin + CHUNK_ROWS].to(torch.float32)
            d = flat_donor[begin:begin + CHUNK_ROWS].to(torch.float32)
            delta = h - d
            if signs is None:
                value = h + delta
            else:
                value = h + signs[begin:begin + CHUNK_ROWS].view(-1, *([1] * (h.dim() - 1))) * delta
            result[begin:begin + CHUNK_ROWS] = value.to(host.dtype)
            moved_sq += float((result[begin:begin + CHUNK_ROWS].to(torch.float32) - h).pow(2).sum())
            target_sq += float(delta.pow(2).sum())
        out[name] = result.reshape(host.shape).contiguous()
        elements += host.numel()
    info = {"mode": mode, "seed": int(seed) if mode == "scramble" else None, "tensors": len(out), "elements": elements,
            "distance_ratio": (moved_sq / target_sq) ** 0.5 if target_sq > 0 else None,
            "rows_plus": plus if mode == "scramble" else None, "rows_minus": minus if mode == "scramble" else None}
    return out, info


def derive(host_dir, donor_dir, mode, seed, out_dir) -> dict:
    import torch
    from safetensors.torch import save_file

    started = time.perf_counter()
    host_dir, donor_dir, out_dir = Path(host_dir), Path(donor_dir), Path(out_dir)
    if not (host_dir / "config.json").exists():
        raise ValueError(f"{host_dir}: no config.json")
    tensors, info = build(host_dir, donor_dir, mode, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = out_dir / "model.safetensors"
    save_file(tensors, str(weights), metadata={"format": "pt"})
    copied = []
    for name in SMALL_FILES:
        if (host_dir / name).exists():
            shutil.copyfile(host_dir / name, out_dir / name)
            copied.append(name)
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    info.update({"host": str(host_dir.resolve()), "donor": str(donor_dir.resolve()), "copied": copied,
                 "sha256": digest.hexdigest(), "torch_version": torch.__version__,
                 "seconds": round(time.perf_counter() - started, 1)})
    (out_dir / "derive108.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--mode", required=True, choices=("forward", "scramble"))
    ap.add_argument("--seed", type=int, default=108)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    info = derive(args.host, args.donor, args.mode, args.seed, args.out)
    print(json.dumps({k: info[k] for k in ("mode", "tensors", "elements", "distance_ratio", "rows_plus", "rows_minus",
                                           "seconds")}), flush=True)


if __name__ == "__main__":
    main()
