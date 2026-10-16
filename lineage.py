#!/usr/bin/env python3
# lineage.py — which published base checkpoint is the parent of OLMo 2 1B SFT? (Claude, 2026-10-15)
# For each candidate revision, reads ONE tensor (layers.8 o_proj) by HTTP range request from its safetensors file and
# compares it row by row with the SFT checkpoint's. A parent should match closely; an unrelated model gives cosine 0.
import glob, json, struct, sys, urllib.request
import numpy as np, torch
from huggingface_hub import HfApi, hf_hub_url
REPO, KEY = 'allenai/OLMo-2-0425-1B', 'model.layers.8.self_attn.o_proj.weight'
api = HfApi()
br = [b.name for b in api.list_repo_refs(REPO).branches]
def last(prefix):
    c = [b for b in br if b.startswith(prefix)]
    return max(c, key=lambda b: int(b.split('-step')[1].split('-')[0])) if c else None
cands = ['main', last('stage1-')] + [last(f'stage2-ingredient{i}-') for i in (1, 2, 3)]
sft = torch.load(glob.glob('/mnt/e/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-SFT/snapshots/*/pytorch_model.bin')[0],
                 mmap=True, weights_only=True, map_location='cpu')[KEY].float().numpy()
def get(url, rng):
    req = urllib.request.Request(url, headers={'Range': f'bytes={rng[0]}-{rng[1]}', 'User-Agent': 'curl/8'})
    return urllib.request.urlopen(req, timeout=120).read()
def rowcos(x, y):
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1); m = (nx > 0) & (ny > 0)
    return float(np.median((x[m] * y[m]).sum(1) / (nx[m] * ny[m])))
for rev in [c for c in cands if c]:
    files = [f for f in api.list_repo_files(REPO, revision=rev) if f.endswith('.safetensors')]
    for f in files:
        url = hf_hub_url(REPO, f, revision=rev)
        n = struct.unpack('<Q', get(url, (0, 7)))[0]; hdr = json.loads(get(url, (8, 8 + n - 1)))
        if KEY not in hdr: continue
        t = hdr[KEY]; s, e = t['data_offsets']; dt = {'F32': np.float32, 'BF16': None}[t['dtype']]
        raw = get(url, (8 + n + s, 8 + n + e - 1))
        x = np.frombuffer(raw, dtype=np.float32).reshape(t['shape']) if dt else torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(t['shape']).float().numpy()
        print(f'{rev:45s} cos with SFT {rowcos(x, sft):.3f}', flush=True); break
