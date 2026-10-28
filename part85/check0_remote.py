#!/usr/bin/env python3
"""check0_remote.py -- independent check of step 0's cosines and off-world mean logit, by HTTP range reads.

Written by Claude (Opus 5.5), 2026-10-28, from the design and the safetensors format, not from step0_weights.py's
reader. It fetches only: L10 down_proj (whole, rows are contiguous), gate/up row 6978, the post-feedforward and final
norm weights, and the 16 off-world rows of lm_head, for one pinned checkpoint, then prints the three cosines with the
Instruct reference vectors (from a step 0 JSON's reference_vectors, or a check0 local run) and the off-world mean logit
through post-feedforward norm x final norm (the variant that matched step 0 exactly on Instruct). The random-set z is
not recomputed here: it needs most of lm_head. Usage: check0_remote.py <repo> <commit> <instruct_step0.json>
"""
import json, struct, sys, urllib.request
import numpy as np

UNIT, L = 6978, 10
OFF_IDS = [20167, 37219, 11741, 11841, 33975, 34261, 3634, 85942, 6917, 21725, 83645, 15861, 958, 16335, 27605, 55479]


def get(url, a=None, b=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"} if a is not None else {})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def main(repo, commit, ref_path):
    base = f"https://huggingface.co/{repo}/resolve/{commit}/"
    wmap = json.loads(get(base + "model.safetensors.index.json"))["weight_map"]
    names = {"down": f"model.layers.{L}.mlp.down_proj.weight", "gate": f"model.layers.{L}.mlp.gate_proj.weight",
             "up": f"model.layers.{L}.mlp.up_proj.weight", "post": f"model.layers.{L}.post_feedforward_layernorm.weight",
             "norm": "model.norm.weight", "emb": "lm_head.weight"}
    headers = {}
    def info(n):
        f = wmap[n]
        if f not in headers:
            hl = struct.unpack("<Q", get(base + f, 0, 7))[0]
            headers[f] = (hl, json.loads(get(base + f, 8, 8 + hl - 1)))
        hl, h = headers[f]
        return f, hl, h[n]
    def elems(n, start, stop):
        f, hl, t = info(n)
        assert t["dtype"] == "F32", t["dtype"]
        o = 8 + hl + t["data_offsets"][0]
        raw = get(base + f, o + 4 * start, o + 4 * stop - 1)
        return np.frombuffer(raw, dtype="<f4").astype(np.float64), t["shape"]
    rows, cols = info(names["down"])[2]["shape"]
    down, _ = elems(names["down"], 0, rows * cols)
    down = down.reshape(rows, cols)[:, UNIT]
    gcols = info(names["gate"])[2]["shape"][1]
    gate, _ = elems(names["gate"], UNIT * gcols, (UNIT + 1) * gcols)
    up, _ = elems(names["up"], UNIT * gcols, (UNIT + 1) * gcols)
    post, _ = elems(names["post"], 0, rows)
    norm, _ = elems(names["norm"], 0, rows)
    ecols = info(names["emb"])[2]["shape"][1]
    emb = np.stack([elems(names["emb"], i * ecols, (i + 1) * ecols)[0] for i in OFF_IDS])
    ref = json.load(open(ref_path, encoding="utf-8"))["reference_vectors"]
    cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    out = {"repo": repo, "commit": commit,
           "cos_down": cos(down, np.array(ref["down_column_6978"])), "cos_gate": cos(gate, np.array(ref["gate_row_6978"])),
           "cos_up": cos(up, np.array(ref["up_row_6978"])),
           "offworld_mean_logit": float((emb @ (down * post * norm)).mean())}
    print(json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:4])
