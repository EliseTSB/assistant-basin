"""Part 89: per-neuron cosine with Instruct at every local OLMo 2 1B checkpoint (down_proj column, gate row, up row).

Written by Claude (Opus 5.5), 2026-10-28, from PART89-DESIGN.md. Weights only, CPU. Checkpoints missing locally are
recorded as missing, never skipped silently. Usage: python cos89.py OUT.npz
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open

HUB = Path("/mnt/e/neuron-probe/hf/hub")
BASE = HUB / "models--allenai--OLMo-2-0425-1B/snapshots"
CKPTS = [  # training order; key, snapshot dir (Part 85's checkpoints.py commit prefixes)
    ("pretrain_21B", BASE, "22f7d18a8b40"), ("pretrain_210B", BASE, "eb19ccabc193"), ("pretrain_1T", BASE, "eda09f034651"),
    ("pretrain_2T", BASE, "3e09b708b744"), ("pretrain_3T", BASE, "22738a97eabc"), ("pretrain_4T", BASE, "9d3e43659f00"),
    ("midtrain_13B", BASE, "a7a789f6b27b"), ("midtrain_26B", BASE, "1c3599f7b404"), ("midtrain_51B", BASE, "f83cd641d4f0"),
    ("sft", HUB / "models--allenai--OLMo-2-0425-1B-SFT/snapshots", "0d85a3d03787"),
    ("dpo", HUB / "models--allenai--OLMo-2-0425-1B-DPO/snapshots", "c4b0485961ab"),
    ("instruct", HUB / "models--allenai--OLMo-2-0425-1B-Instruct/snapshots", "48d788eca847"),
]
LAYERS, WIDTH = 16, 8192
MATS = ("down", "gate", "up")


def weights_file(root, prefix):
    """Return {tensor name: shard path} for a local snapshot, or None if it is absent or incomplete."""
    hits = [d for d in root.iterdir() if d.name.startswith(prefix)] if root.exists() else []
    if len(hits) != 1:
        return None
    snap = hits[0]
    index = snap / "model.safetensors.index.json"
    if index.exists():
        wmap = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = {n: snap / f for n, f in wmap.items()}
        if not all(f.exists() and f.stat().st_size > 1e8 for f in set(shards.values())):
            return None
        return shards
    binf = snap / "pytorch_model.bin"
    if binf.exists() and binf.stat().st_size > 1e9:
        return {"__bin__": binf}
    single = snap / "model.safetensors"
    if single.exists() and single.stat().st_size > 1e9:
        with safe_open(str(single), framework="pt") as f:
            return {n: single for n in f.keys()}
    return None


class Weights:
    def __init__(self, shards):
        self.shards = shards
        self.open = {}

    def get_tensor(self, name):
        if "__bin__" in self.shards:
            if "sd" not in self.open:
                self.open["sd"] = torch.load(str(self.shards["__bin__"]), map_location="cpu", weights_only=True, mmap=True)
            return self.open["sd"][name]
        path = self.shards[name]
        if path not in self.open:
            self.open[path] = safe_open(str(path), framework="pt")
        return self.open[path].get_tensor(name)


def neuron_rows(f, layer):
    """Return {mat: tensor[8192, 2048]} with one row per neuron (down_proj transposed)."""
    p = f"model.layers.{layer}.mlp."
    out = {}
    out["down"] = f.get_tensor(p + "down_proj.weight").float().T.contiguous()
    out["gate"] = f.get_tensor(p + "gate_proj.weight").float()
    out["up"] = f.get_tensor(p + "up_proj.weight").float()
    return out


def main(out_path):
    files = {k: weights_file(root, pre) for k, root, pre in CKPTS}
    present = [k for k, _, _ in CKPTS if files[k] is not None]
    missing = [k for k, _, _ in CKPTS if files[k] is None]
    print("present", present, "missing", missing, flush=True)
    if "instruct" not in present:
        sys.exit("Instruct missing")
    cos = {m: np.full((len(present), LAYERS, WIDTH), np.nan, dtype=np.float32) for m in MATS}
    stage_pairs = [("midtrain_51B", "sft"), ("sft", "dpo"), ("dpo", "instruct")]
    pair_cos = {f"{a}->{b}": np.full((LAYERS, WIDTH), np.nan, dtype=np.float32) for a, b in stage_pairs}
    handles = {k: Weights(files[k]) for k in present}
    t0 = time.time()
    for layer in range(LAYERS):
        rows = {k: neuron_rows(handles[k], layer) for k in present}
        ref = rows["instruct"]
        for i, k in enumerate(present):
            for m in MATS:
                cos[m][i, layer] = torch.nn.functional.cosine_similarity(rows[k][m], ref[m], dim=1).numpy()
        for a, b in stage_pairs:
            if a in rows and b in rows:
                pair_cos[f"{a}->{b}"][layer] = torch.nn.functional.cosine_similarity(rows[a]["down"], rows[b]["down"], dim=1).numpy()
        del rows
        print(f"layer {layer} done {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out_path, checkpoints=np.array(present), missing=np.array(missing),
                        **{f"cos_{m}": v for m, v in cos.items()}, **{f"pair_{k}": v for k, v in pair_cos.items()})
    Path(str(out_path) + ".meta.json").write_text(json.dumps(
        {"present": present, "missing": missing, "files": {k: sorted({str(v) for v in files[k].values()}) for k in present},
         "elapsed_s": time.time() - t0}, indent=1), encoding="utf-8")
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
