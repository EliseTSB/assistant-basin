"""Image arm, after-the-fact null (written 2026-11-04 after images.py's result, NOT registered): how many of ALL MLP
neurons separate the 43 hen photos from the 23 other photos as well as the selected ones do? Same neuron definition and
photo sets as images.py (input to mlp.down_proj, mean over positions, one image per pass). Writes images_all.npz and
images_all.json: per-neuron AUC (hens positive), the two-sided AUC max(a, 1-a), and where each selected unit ranks.

Run: PYTHONPATH=/home/elisepc/eg2-extra /home/elisepc/neuron-probe/.venv/bin/python -B images_all.py"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
PHOTOS = HERE.parents[2] / 'sources' / 'photos'
SNAP = '/mnt/e/neuron-probe/hf/hub/models--google--embeddinggemma-2/snapshots/f9972ac22e696a105c482202646979e33e4b5a1e'
EXT = {'.jpg', '.jpeg', '.png', '.webp'}
hens = sorted(p for p in (PHOTOS / 'hens').rglob('*') if p.suffix.lower() in EXT)
others = sorted(p for p in PHOTOS.rglob('*') if p.suffix.lower() in EXT and 'hens' not in p.relative_to(PHOTOS).parts)
prev = json.loads((HERE / 'images.json').read_text())
assert (len(hens), len(others)) == (prev['n_hens'], prev['n_others']), 'photo sets changed since images.py'

m = SentenceTransformer(SNAP, device='cpu', model_kwargs={'dtype': torch.float32})
layers = m[0].model.language_model.layers
grab = {}
for L, layer in enumerate(layers):
    layer.mlp.down_proj.register_forward_pre_hook(lambda mod, a, L=L: grab.__setitem__(L, a[0][0].float().mean(0)))
rows = []
for p in hens + others:
    grab.clear()
    with torch.inference_mode():
        m.encode({'image': str(p)}, normalize_embeddings=True)
    rows.append(torch.cat([grab[L] for L in range(len(layers))]).numpy())
X = np.stack(rows)
y = np.array([True] * len(hens) + [False] * len(others))
p, n = y.sum(), (~y).sum()
R = np.apply_along_axis(rankdata, 0, X)
auc = (R[y].sum(0) - p * (p + 1) / 2) / (p * n)
two = np.maximum(auc, 1 - auc)
width = X.shape[1] // len(layers)
np.savez_compressed(HERE / 'images_all.npz', X=X.astype(np.float32), auc=auc, n_hens=len(hens), width=width)
sel = [(r['layer'], r['index'], r['image_auc_same_sign']) for r in prev['rows']]
out = {'n_neurons': int(X.shape[1]), 'width': width, 'n_layers': len(layers),
       'share_two_sided_at_least': {str(t): float((two >= t).mean()) for t in (0.75, 0.9, 0.95, 0.99, 1.0)},
       'count_two_sided_at_least': {str(t): int((two >= t).sum()) for t in (0.75, 0.9, 0.95, 0.99, 1.0)},
       'by_layer_share_ge_0.99': [float((two[L * width:(L + 1) * width] >= 0.99).mean()) for L in range(len(layers))],
       'selected': [{'layer': L, 'index': N, 'image_auc_same_sign': a, 'two_sided': float(two[L * width + N]),
                     'share_of_all_at_or_above': float((two >= two[L * width + N]).mean())} for L, N, a in sel]}
(HERE / 'images_all.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ('n_neurons', 'share_two_sided_at_least', 'count_two_sided_at_least')}))
for s in out['selected']:
    print(f"L{s['layer']} N{s['index']} two-sided {s['two_sided']:.3f} share of all at or above {s['share_of_all_at_or_above']:.4f}")
