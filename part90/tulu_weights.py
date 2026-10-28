#!/usr/bin/env python3
"""tulu_weights.py -- Part 90: Tulu 3 8B L14 N3851's weights at each training stage, by HTTP range reads.

Written by Claude (Opus 5.5), 2026-10-28, from PART90-DESIGN.md and the safetensors format. For each pinned checkpoint
it reads layer 14's whole down_proj (rows are contiguous, so one column needs the whole tensor) and rows 3851 of
gate_proj and up_proj, in 16 MB ranges with retries and a byte-rate cap, and saves them as .npy under --out. Then it
prints N3851's cosines between every pair of stages and, as the null, the base-to-final cosine of all 14,336 down
columns of layer 14 with N3851's percentile. Usage: tulu_weights.py --out DIR [--rate 3000000]
"""
import argparse, json, struct, sys, time, urllib.request
from pathlib import Path
import numpy as np

UNIT, L = 3851, 14
STAGES = [("base", "unsloth/Meta-Llama-3.1-8B", "e9a141a2091e"),
          ("sft", "allenai/Llama-3.1-Tulu-3-8B-SFT", "f2a0b46b0cfd"),
          ("dpo", "allenai/Llama-3.1-Tulu-3-8B-DPO", "a7beb67e33ff"),
          ("final", "allenai/Llama-3.1-Tulu-3-8B", "666943798adb")]
CHUNK = 16 * 2**20


def get(url, a=None, b=None, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"} if a is not None else {})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if a is not None and len(data) != b - a + 1:
                raise IOError(f"short read {len(data)} of {b - a + 1}")
            return data
        except Exception as e:  # Hugging Face resets long range reads (gotcha 2026-10-28); pause and retry
            if i == tries - 1:
                raise
            print(f"retry {i + 1} after {e!r}", flush=True)
            time.sleep(20 * (i + 1))


def resolve_commit(repo, prefix):
    sha = json.loads(get(f"https://huggingface.co/api/models/{repo}/revision/{prefix}"))["sha"]
    if not sha.startswith(prefix):
        raise ValueError(f"{repo}: {prefix} resolves to {sha}")
    return sha


def read_stage(repo, prefix, rate):
    commit = resolve_commit(repo, prefix)
    base = f"https://huggingface.co/{repo}/resolve/{commit}/"
    wmap = json.loads(get(base + "model.safetensors.index.json"))["weight_map"]
    headers = {}

    def tensor(name, start, stop):
        f = wmap[name]
        if f not in headers:
            hl = struct.unpack("<Q", get(base + f, 0, 7))[0]
            headers[f] = (hl, json.loads(get(base + f, 8, 8 + hl - 1)))
        hl, h = headers[f]
        t = h[name]
        size = {"BF16": 2, "F16": 2, "F32": 4}[t["dtype"]]
        o = 8 + hl + t["data_offsets"][0]
        lo, hi = o + size * start, o + size * stop - 1
        parts, pos = [], lo
        while pos <= hi:
            end = min(hi, pos + CHUNK - 1)
            t0 = time.time()
            parts.append(get(base + f, pos, end))
            wait = (end - pos + 1) / rate - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)
            pos = end + 1
        raw = b"".join(parts)
        if t["dtype"] == "BF16":
            arr = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif t["dtype"] == "F16":
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        else:
            arr = np.frombuffer(raw, dtype="<f4").copy()
        return arr, t["shape"], t["dtype"]

    dn = f"model.layers.{L}.mlp.down_proj.weight"
    f = wmap[dn]
    if f not in headers:
        hl = struct.unpack("<Q", get(base + f, 0, 7))[0]
        headers[f] = (hl, json.loads(get(base + f, 8, 8 + hl - 1)))
    rows, cols = headers[f][1][dn]["shape"]
    down, _, dtype = tensor(dn, 0, rows * cols)
    down = down.reshape(rows, cols)
    gname, uname = f"model.layers.{L}.mlp.gate_proj.weight", f"model.layers.{L}.mlp.up_proj.weight"
    gate, gshape, _ = tensor(gname, UNIT * rows, (UNIT + 1) * rows)
    up, ushape, _ = tensor(uname, UNIT * rows, (UNIT + 1) * rows)
    assert gshape == [cols, rows] and ushape == [cols, rows], (gshape, ushape, rows, cols)
    return {"repo": repo, "commit": commit, "dtype": dtype, "down_shape": [rows, cols]}, down, gate, up


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", type=float, default=3e6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta, vec = {}, {}
    for stage, repo, prefix in STAGES:
        d = out / stage
        if (d / "meta.json").exists():
            meta[stage] = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        else:
            info, down, gate, up = read_stage(repo, prefix, a.rate)
            d.mkdir(exist_ok=True)
            np.save(d / "down.npy", down)
            np.save(d / "gate_row.npy", gate)
            np.save(d / "up_row.npy", up)
            (d / "meta.json").write_text(json.dumps(info), encoding="utf-8")
            meta[stage] = info
            print(f"{stage}: read {repo}@{info['commit'][:12]} {info['dtype']} down {info['down_shape']}", flush=True)
        vec[stage] = {"down": np.load(d / "down.npy")[:, UNIT], "gate": np.load(d / "gate_row.npy"),
                      "up": np.load(d / "up_row.npy")}
    names = [s for s, _, _ in STAGES]
    pairs = {f"{x}->{y}": {k: cos(vec[x][k], vec[y][k]) for k in ("down", "gate", "up")}
             for i, x in enumerate(names) for y in names[i + 1:]}
    db, df = np.load(out / "base" / "down.npy"), np.load(out / "final" / "down.npy")
    allcos = (db * df).sum(0) / (np.linalg.norm(db, axis=0) * np.linalg.norm(df, axis=0))
    mine = float(allcos[UNIT])
    report = {"unit": UNIT, "layer": L, "stages": meta, "pairs": pairs,
              "layer_base_to_final_down": {"n": int(allcos.size), "median": float(np.median(allcos)),
                                           "p05": float(np.percentile(allcos, 5)),
                                           "p95": float(np.percentile(allcos, 95)),
                                           "min": float(allcos.min()), "unit_cos": mine,
                                           "unit_percentile": float((allcos < mine).mean() * 100)}}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
