#!/usr/bin/env python3
# lineage8.py — Part 8.1: do SWE-agent-LM-7B and openhands-lm-7b-v0.1 descend from Qwen2.5-Coder-7B-Instruct rather
# than its base? (Claude, 2026-10-16). Same method as lineage3.py: one tensor per model by HTTP range request,
# compared row by row (median cosine); two tensors (layers 8 and 20 o_proj) instead of one.
import json, struct, sys, urllib.request
import numpy as np, torch
from huggingface_hub import HfApi, hf_hub_url
api = HfApi()
M = {'base': 'Qwen/Qwen2.5-Coder-7B', 'instruct': 'Qwen/Qwen2.5-Coder-7B-Instruct',
     'sweagent': 'SWE-bench/SWE-agent-LM-7B', 'openhands': 'OpenHands/openhands-lm-7b-v0.1'}
def get(url, rng):
    req = urllib.request.Request(url, headers={'Range': f'bytes={rng[0]}-{rng[1]}', 'User-Agent': 'curl/8'})
    return urllib.request.urlopen(req, timeout=300).read()
def tensor(repo, key):
    for f in [f for f in api.list_repo_files(repo) if f.endswith('.safetensors')]:
        url = hf_hub_url(repo, f)
        n = struct.unpack('<Q', get(url, (0, 7)))[0]; hdr = json.loads(get(url, (8, 8 + n - 1)))
        if key not in hdr: continue
        t = hdr[key]; s, e = t['data_offsets']; raw = get(url, (8 + n + s, 8 + n + e - 1))
        if t['dtype'] == 'F32': return np.frombuffer(raw, dtype=np.float32).reshape(t['shape'])
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(t['shape']).float().numpy()
    raise RuntimeError(f'{key} not in {repo}')
def rowcos(x, y):
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1); m = (nx > 0) & (ny > 0)
    return float(np.median((x[m] * y[m]).sum(1) / (nx[m] * ny[m])))
out = {}
for key in ['model.layers.8.self_attn.o_proj.weight', 'model.layers.20.self_attn.o_proj.weight']:
    T = {k: tensor(r, key) for k, r in M.items()}
    names = list(M); out[key] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[key][f'{a}~{b}'] = round(rowcos(T[a], T[b]), 4)
    print(key, json.dumps(out[key]), flush=True)
json.dump(out, open('out/lineage-qwen-coder.json', 'w'), indent=1)
