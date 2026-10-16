#!/usr/bin/env python3
# lineage9.py — Part 9.1: does Olmo-3-7B-Think-SFT descend from Olmo-3-1025-7B main, and is Instruct-SFT closer to
# Think-SFT than to main (warm start)? (Claude, 2026-10-16). Same method as lineage8.py.
import json, struct, sys, urllib.request
import numpy as np, torch
from huggingface_hub import HfApi, hf_hub_url
api = HfApi()
M = {'base': 'allenai/Olmo-3-1025-7B', 'think_sft': 'allenai/Olmo-3-7B-Think-SFT',
     'instruct_sft': 'allenai/Olmo-3-7B-Instruct-SFT', 'think_dpo': 'allenai/Olmo-3-7B-Think-DPO'}
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
json.dump(out, open('out/lineage-olmo3-think.json', 'w'), indent=1)
