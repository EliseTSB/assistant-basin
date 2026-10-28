#!/usr/bin/env python3
"""check0.py -- an independent recomputation of Part 85's step 0, written from PART85-DESIGN.md, not from step0_weights.py.

Written by Claude (Opus 5.5), 2026-10-28, from the design. Not fully blind: I had read step 0's readout lines (its gain
is post_feedforward_norm times final norm), so the post_and_final variant below is not independent of that choice.
It reads a local checkpoint snapshot (sharded safetensors, one model.safetensors, or pytorch_model.bin) and computes:
  (a) the cosines of L10 down_proj column 6978 and gate/up rows 6978 with a reference snapshot's (the Instruct model);
  (b) the direct readout: the column through the final norm weight and the output rows, mean logit over the registered
      off-world token set minus the mean over random sets, as a z-score against 1,000 random sets of the same size.
Two readout variants, because the design names only the final norm weight while OLMo 2 also scales the MLP output by
its post-feedforward norm: `final_only` (the design's words) and `post_and_final`. The random sets are my own draw
(tokens whose decoded length falls inside the off-world tokens' length range, numpy default_rng(85085)), so z will not
match step 0 to the digit; cosines and the off-world mean logit must match exactly up to float error.
Usage: check0.py <snapshot_dir> [--reference <instruct_snapshot_dir>] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

UNIT, LAYER = 6978, 10
OFFWORLD = [" alien", " aliens", " extraterrestrial", " planet", " planets", " galaxy", " space", " spaceship",
            " starship", " Mars", " cosmos", " universe", " interstellar", " outer", " orbit", " asteroid"]
NAMES = {
    "down": f"model.layers.{LAYER}.mlp.down_proj.weight",
    "gate": f"model.layers.{LAYER}.mlp.gate_proj.weight",
    "up": f"model.layers.{LAYER}.mlp.up_proj.weight",
    "post": f"model.layers.{LAYER}.post_feedforward_layernorm.weight",
    "norm": "model.norm.weight",
    "emb": "lm_head.weight",
}


def load(snapshot: Path) -> dict[str, np.ndarray]:
    import torch
    snapshot = Path(snapshot)
    out = {}
    if (snapshot / "pytorch_model.bin").exists():
        sd = torch.load(snapshot / "pytorch_model.bin", weights_only=True, map_location="cpu")
        for k, n in NAMES.items():
            out[k] = sd[n].float().numpy()
        return out
    from safetensors import safe_open
    files = sorted(snapshot.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no weights in {snapshot}")
    wanted = set(NAMES.values())
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            for n in h.keys():
                if n in wanted:
                    k = next(k for k, v in NAMES.items() if v == n)
                    out[k] = h.get_tensor(n).float().numpy()
    missing = [k for k in NAMES if k not in out]
    if missing:
        raise KeyError(f"missing tensors {missing} in {snapshot}")
    return out


def vectors(w: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"down": w["down"][:, UNIT].astype(np.float64), "gate": w["gate"][UNIT].astype(np.float64),
            "up": w["up"][UNIT].astype(np.float64)}


def cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def readout(w, tok) -> dict:
    off_ids = sorted({tok.encode(s, add_special_tokens=False)[0] for s in OFFWORLD})
    lens = [len(tok.decode([i])) for i in off_ids]
    lo, hi = min(lens), max(lens)
    vocab = w["emb"].shape[0]
    pool = np.array([i for i in range(min(vocab, len(tok))) if i not in off_ids and lo <= len(tok.decode([i])) <= hi])
    rng = np.random.default_rng(85085)
    sets = np.stack([rng.choice(pool, size=len(off_ids), replace=False) for _ in range(1000)])
    col = w["down"][:, UNIT].astype(np.float64)
    res = {"off_ids": off_ids, "pool_size": int(len(pool)), "length_range": [lo, hi]}
    for name, gain in (("final_only", w["norm"]), ("post_and_final", w["post"] * w["norm"])):
        logits = w["emb"].astype(np.float64) @ (col * gain.astype(np.float64))
        off = float(logits[off_ids].mean())
        means = logits[sets].mean(axis=1)
        res[name] = {"offworld_mean_logit": off, "random_mean": float(means.mean()), "random_sd": float(means.std(ddof=1)),
                     "z": float((off - means.mean()) / means.std(ddof=1))}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--reference")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok_dir = a.reference or a.snapshot
    tok = AutoTokenizer.from_pretrained(tok_dir)
    w = load(Path(a.snapshot))
    out = {"snapshot": a.snapshot, "readout": readout(w, tok)}
    if a.reference:
        r = vectors(load(Path(a.reference)))
        v = vectors(w)
        out["cosines_with_reference"] = {k: cos(v[k], r[k]) for k in v}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
