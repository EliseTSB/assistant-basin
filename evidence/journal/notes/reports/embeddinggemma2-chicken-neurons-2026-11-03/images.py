"""Image arm: do the chicken neurons fire for hen photos? Predictions I1 to I3 in PREDICTIONS.md (written first).
Same neuron definition as record.py (input to mlp.down_proj, mean over positions, one item per pass), vision encoder
loaded. Hens = sources/photos/hens/*; others = every other image under sources/photos/. Writes images.json.

Ran 2026-11-04 by 06:55 CT under a 4 GB memory cap with the meter stopped (images.log). History: the first launch (2026-11-03, before 11:25 CT) was stopped by Claude Code for low memory. WSL has 11 GB and the
Chicken-o-Meter server holds about 3 GB; run it with the meter paused if Elise agrees (asked her about 11:23 CT) and nothing else heavy in WSL."""
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

res = json.loads((HERE / 'results.json').read_text())['mean']
top10 = [(x['layer'], x['index'], x['sign']) for x in res['top10']]
both9 = [(15, 436), (15, 1281), (16, 134), (16, 349), (17, 1297), (17, 1574), (19, 1129), (20, 1260), (23, 1262)]
d = np.load(HERE / 'acts.npz')
A = d['act_mean'].astype(np.float32)
esc = d['kind'] == 'esc'
ch = np.isin(d['label'], ['hen', 'rooster'])
disc = esc & (d['fold'] <= 3)


def auc(v, y):
    r = rankdata(v)
    p, n = y.sum(), (~y).sum()
    return float((r[y].sum() - p * (p + 1) / 2) / (p * n))


def audio_sign(L, N):
    return 1 if auc(A[disc, L * 2048 + N], ch[disc]) >= 0.5 else -1


units = sorted({(L, N) for L, N, _ in top10} | set(both9))
sign = {u: audio_sign(*u) for u in units}
hens = sorted(p for p in (PHOTOS / 'hens').rglob('*') if p.suffix.lower() in EXT)
others = sorted(p for p in PHOTOS.rglob('*') if p.suffix.lower() in EXT and 'hens' not in p.relative_to(PHOTOS).parts)

m = SentenceTransformer(SNAP, device='cpu', model_kwargs={'dtype': torch.float32})
layers = m[0].model.language_model.layers
grab = {}
for L in sorted({u[0] for u in units}):
    layers[L].mlp.down_proj.register_forward_pre_hook(lambda mod, a, L=L: grab.__setitem__(L, a[0][0].float().mean(0)))
vals = {u: [] for u in units}
for p in hens + others:
    grab.clear()
    with torch.inference_mode():
        m.encode({'image': str(p)}, normalize_embeddings=True)
    for u in units:
        vals[u].append(float(grab[u[0]][u[1]]))
y = np.array([True] * len(hens) + [False] * len(others))
rows = []
for u in units:
    a = auc(np.array(vals[u]), y)
    rows.append({'layer': u[0], 'index': u[1], 'audio_sign': sign[u], 'image_auc_same_sign': a if sign[u] > 0 else 1 - a,
                 'in_top10': u in {(L, N) for L, N, _ in top10}, 'in_both9': u in set(both9)})
rows.sort(key=lambda r: -r['image_auc_same_sign'])
get = lambda L, N: next(r['image_auc_same_sign'] for r in rows if (r['layer'], r['index']) == (L, N))
pred = {'I1': get(17, 1574), 'I2': sum(get(*u) >= 0.7 for u in both9), 'I3': get(15, 1126)}
verdict = {'I1': 'held' if pred['I1'] >= 0.8 else 'failed', 'I2': 'held' if pred['I2'] >= 5 else 'failed',
           'I3': 'held' if pred['I3'] >= 0.7 else 'failed'}
out = {'n_hens': len(hens), 'n_others': len(others), 'others': [str(p.relative_to(PHOTOS)) for p in others],
       'rows': rows, 'predictions': {k: [verdict[k], pred[k]] for k in pred}}
(HERE / 'images.json').write_text(json.dumps(out, indent=1))
print('hens', len(hens), 'others', len(others))
for r in rows:
    print(f"L{r['layer']} N{r['index']} sign{r['audio_sign']:+d} image AUC {r['image_auc_same_sign']:.3f}"
          f"{' top10' if r['in_top10'] else ''}{' both9' if r['in_both9'] else ''}")
print('PREDICTIONS', json.dumps(out['predictions']))
