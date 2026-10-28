#!/usr/bin/env python3
"""Part 88 step A1: cosines of the 64 units' down_proj column, gate row and up row with the Instruct model's.

Written by Claude (Opus 5.5), 2026-10-28, from PART88-DESIGN.md. The remote reader is check0_remote.py's (which matched
step 0 exactly on seven checkpoints): HTTP range reads of whole down_proj tensors per layer (rows are contiguous, so a
column needs the tensor) and single gate/up rows. The Instruct reference is read from its local snapshot.
Usage: a1_weights.py --reference <instruct snapshot dir> [--remote <repo> <commit> | --snapshot <dir>] --out <json>
"""
import argparse, json, struct, sys, urllib.request
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def units():
    d = json.load(open(HERE.parent / "results" / "part16" / "search.json", encoding="utf-8"))
    out = []
    for x in d["search"]["top_set"]:
        layer, unit = x["neuron"][1:].split(".")
        out.append((int(layer), int(unit)))
    assert len(out) == 64, len(out)
    return out


class Remote:
    def __init__(self, repo, commit):
        self.base = f"https://huggingface.co/{repo}/resolve/{commit}/"
        self.wmap = json.loads(self.get("model.safetensors.index.json"))["weight_map"]
        self.h = {}

    def get(self, f, a=None, b=None):
        req = urllib.request.Request(self.base + f, headers={"Range": f"bytes={a}-{b}"} if a is not None else {})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    return r.read()
            except OSError:
                if attempt == 3:
                    raise

    def info(self, n):
        f = self.wmap[n]
        if f not in self.h:
            hl = struct.unpack("<Q", self.get(f, 0, 7))[0]
            self.h[f] = (hl, json.loads(self.get(f, 8, 8 + hl - 1)))
        hl, hdr = self.h[f]
        return f, hl, hdr[n]

    def tensor(self, n):
        f, hl, t = self.info(n)
        assert t["dtype"] == "F32", t["dtype"]
        o = 8 + hl + t["data_offsets"][0]
        size = int(np.prod(t["shape"]))
        return np.frombuffer(self.get(f, o, o + 4 * size - 1), dtype="<f4").reshape(t["shape"])

    def row(self, n, i):
        f, hl, t = self.info(n)
        assert t["dtype"] == "F32", t["dtype"]
        cols = t["shape"][1]
        o = 8 + hl + t["data_offsets"][0] + 4 * i * cols
        return np.frombuffer(self.get(f, o, o + 4 * cols - 1), dtype="<f4")


class Local:
    def __init__(self, snap):
        import torch
        from safetensors import safe_open
        snap = Path(snap)
        self.t = {}
        if (snap / "pytorch_model.bin").exists():
            sd = torch.load(snap / "pytorch_model.bin", weights_only=True, map_location="cpu")
            self.get_t = lambda n: sd[n].float().numpy()
        else:
            files = sorted(snap.glob("*.safetensors"))
            idx = {}
            for f in files:
                with safe_open(str(f), framework="pt") as h:
                    for k in h.keys():
                        idx[k] = f
            def get_t(n):
                with safe_open(str(idx[n]), framework="pt") as h:
                    return h.get_tensor(n).float().numpy()
            self.get_t = get_t

    def tensor(self, n):
        return self.get_t(n)

    def row(self, n, i):
        return self.get_t(n)[i]


def vectors(src, us):
    out, layers = {}, sorted({l for l, _ in us})
    for l in layers:
        down = src.tensor(f"model.layers.{l}.mlp.down_proj.weight")
        for (ll, u) in us:
            if ll != l:
                continue
            out[(l, u)] = {"down": down[:, u].astype(np.float64),
                           "gate": src.row(f"model.layers.{l}.mlp.gate_proj.weight", u).astype(np.float64),
                           "up": src.row(f"model.layers.{l}.mlp.up_proj.weight", u).astype(np.float64)}
    return out


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--remote", nargs=2)
    ap.add_argument("--snapshot")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    us = units()
    ref = vectors(Local(a.reference), us)
    src = Remote(*a.remote) if a.remote else Local(a.snapshot)
    mine = vectors(src, us)
    rows = [{"layer": l, "unit": u, **{f"cos_{k}": cos(mine[(l, u)][k], ref[(l, u)][k]) for k in ("down", "gate", "up")}}
            for (l, u) in us]
    same = sum(r["cos_down"] >= 0.8 for r in rows)
    out = {"source": a.remote or a.snapshot, "units": rows, "same_units_down_ge_0_8": same,
           "median_cos_down": float(np.median([r["cos_down"] for r in rows]))}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{a.out}: same units {same}/64, median down cosine {out['median_cos_down']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
