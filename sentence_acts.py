#!/usr/bin/env python3
"""Part 58b: max activation of chosen Qwen3-1.7B MLP units on short texts (CPU, bf16).

Usage: python sentence_acts.py --sentences s.jsonl --units units.json --out acts.npz [--threads 4] [--dtype bf16]
sentences.jsonl: {"sid": ..., "text": ...}; units.json: [[layer, unit], ...].
Activation = input to layers[L].mlp.down_proj at index `unit`, exactly the collector's definition.
Output rows follow the sentence file; columns follow units.json. max over token positions >= 1 (position 0 excluded:
it carries attention-sink artifacts); pos0 is saved separately.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REV = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SNAP = Path("/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3-1.7B/snapshots") / REV

ap = argparse.ArgumentParser()
ap.add_argument("--sentences", required=True); ap.add_argument("--units", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--dtype", default="bf16"); ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads)
sents = [json.loads(l) for l in open(a.sentences) if l.strip()]
import hashlib
def stable_seed(seed, *parts):
    return int.from_bytes(hashlib.sha256("\0".join((str(seed), *(str(p) for p in parts))).encode("utf-8")).digest()[:4], "big")
def control_dirs(n, width, seed, layer):   # same as atlas_collect.generate_controls
    v = np.random.default_rng(stable_seed(seed, "control-directions", layer)).standard_normal((n, width), dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), np.finfo(np.float32).tiny)
units = [tuple(u) for u in json.load(open(a.units))]    # (layer, unit) or (layer, "c", control_id)
dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
tok = AutoTokenizer.from_pretrained(str(SNAP), local_files_only=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(str(SNAP), local_files_only=True, torch_dtype=dtype).eval()
layers = model.model.layers
by_layer, ctl_by_layer = {}, {}
for j, u in enumerate(units):
    if len(u) == 3:
        ctl_by_layer.setdefault(u[0], []).append((j, u[2]))
    else:
        by_layer.setdefault(u[0], []).append((j, u[1]))
need = sorted(set(by_layer) | set(ctl_by_layer))
maxL = max(need)
cur = {}
res_max = np.zeros((len(sents), len(units)), dtype=np.float32); res_p0 = np.zeros_like(res_max)

def make_hook(L):
    cols = torch.as_tensor([u for _, u in by_layer.get(L, [])], dtype=torch.long); idx = [j for j, _ in by_layer.get(L, [])]
    cidx = [j for j, _ in ctl_by_layer.get(L, [])]
    cdir = (torch.as_tensor(control_dirs(100, 6144, 5601, L)[[c for _, c in ctl_by_layer[L]]]) if L in ctl_by_layer else None)
    def hook(_m, args):
        xf = args[0].detach().float()
        parts = []
        if idx: parts.append(xf.index_select(-1, cols))
        if cidx: parts.append(torch.matmul(xf, cdir.transpose(0, 1)))
        x = torch.cat(parts, dim=-1); allidx = idx + cidx          # [B, T, k]
        m = cur["mask"]
        p0 = x[:, 0, :]
        rest = x[:, 1:, :].masked_fill(~m[:, 1:].unsqueeze(-1), float("-inf")).amax(dim=1)
        rest = torch.where(torch.isinf(rest), p0, rest)
        res_max[np.ix_(cur["rows"], allidx)] = rest.numpy(); res_p0[np.ix_(cur["rows"], allidx)] = p0.numpy()
    return hook

handles = [layers[L].mlp.down_proj.register_forward_pre_hook(make_hook(L)) for L in need]
t0 = time.time()
with torch.inference_mode():
    for s in range(0, len(sents), a.batch):
        chunk = sents[s:s + a.batch]
        enc = tok([c["text"] for c in chunk], return_tensors="pt", padding=True, add_special_tokens=True)
        cur["mask"] = enc["attention_mask"].bool(); cur["rows"] = list(range(s, s + len(chunk)))
        model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
        if (s // a.batch) % 20 == 0:
            print(f"{s + len(chunk)}/{len(sents)} {time.time()-t0:.0f}s", flush=True)
np.savez(a.out, max=res_max, pos0=res_p0, sids=np.array([str(c["sid"]) for c in sents]), units=np.array([json.dumps(u) for u in units]))
print("done", f"{time.time()-t0:.0f}s")
