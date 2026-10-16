#!/usr/bin/env python3
# lineage3.py — which published OLMo 3 7B base revision is the parent of Olmo-3-7B-Instruct-SFT and of
# Olmo-3-7B-RL-Zero-General? (Claude fork, 2026-10-16). Same method as lineage.py: ONE tensor
# (layers.8 o_proj) per revision by HTTP range request, compared row by row (median cosine).
import json, re, struct, sys, urllib.request
import numpy as np, torch
from huggingface_hub import HfApi, hf_hub_url
BASE, KEY = 'allenai/Olmo-3-1025-7B', 'model.layers.8.self_attn.o_proj.weight'
api = HfApi()
def get(url, rng):
    req = urllib.request.Request(url, headers={'Range': f'bytes={rng[0]}-{rng[1]}', 'User-Agent': 'curl/8'})
    return urllib.request.urlopen(req, timeout=300).read()
def tensor(repo, rev=None):
    files = [f for f in api.list_repo_files(repo, revision=rev) if f.endswith('.safetensors')]
    for f in files:
        url = hf_hub_url(repo, f, revision=rev)
        n = struct.unpack('<Q', get(url, (0, 7)))[0]; hdr = json.loads(get(url, (8, 8 + n - 1)))
        if KEY not in hdr: continue
        t = hdr[KEY]; s, e = t['data_offsets']; raw = get(url, (8 + n + s, 8 + n + e - 1))
        if t['dtype'] == 'F32': return np.frombuffer(raw, dtype=np.float32).reshape(t['shape'])
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(t['shape']).float().numpy()
    raise RuntimeError(f'{KEY} not in {repo}@{rev}')
def rowcos(x, y):
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1); m = (nx > 0) & (ny > 0)
    return float(np.median((x[m] * y[m]).sum(1) / (nx[m] * ny[m])))
br = [b.name for b in api.list_repo_refs(BASE).branches]
prefixes = sorted({re.sub(r'-step\d+.*$', '', b) for b in br})
print('branch prefixes:', prefixes, flush=True)
def last(prefix):
    c = [b for b in br if re.sub(r'-step\d+.*$', '', b) == prefix]
    return max(c, key=lambda b: int(re.search(r'-step(\d+)', b).group(1))) if c else None
cands = ['main'] + [last(p) for p in prefixes if p != 'main']
kids = {'SFT': tensor('allenai/Olmo-3-7B-Instruct-SFT'), 'RL-Zero-General': tensor('allenai/Olmo-3-7B-RL-Zero-General')}
print('SFT vs RL-Zero-General', f"{rowcos(kids['SFT'], kids['RL-Zero-General']):.3f}", flush=True)
for rev in [c for c in cands if c]:
    x = tensor(BASE, rev)
    print(f'{rev:40s} ' + '  '.join(f'{k} {rowcos(x, v):.3f}' for k, v in kids.items()), flush=True)
