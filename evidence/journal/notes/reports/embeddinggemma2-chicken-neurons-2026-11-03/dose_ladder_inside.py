"""Exploratory, after I1-I5 were scored (EDIT-DESIGN Amendment 3): what the inside write causes at gentler strengths.

For the 12 gated test imitations, the inside write is applied at a ladder of strengths expressed as the push's size
relative to R (the residual stream's mean per-position norm), with the chicken direction d and the random direction.
Records each clip's top ESC-50 class, chicken score and cosine to its unedited embedding. Descriptive only: nothing
here was predicted, and nothing is fitted.
Run: HF_HUB_OFFLINE=1 PYTHONPATH=/home/elisepc/eg2-extra /home/elisepc/neuron-probe/.venv/bin/python -B dose_ladder_inside.py
"""
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import edit_arms as ea
from inside_gate import InsideModel, VIS
from record import decode

ver = json.loads((HERE / 'edit_verdicts.json').read_text()); par = ver['arms']['written_gated_block']['parameters']
s17, s19 = int(ver['signs']['L17_N1574']), int(ver['signs']['L19_N1129'])
ref = np.load(HERE / 'acts.npz'); vis = np.load(HERE / 'vis_acts.npz')
man = list(csv.DictReader(open(HERE / 'vis_manifest.csv'))); fit, _ = ea.fit_imitation_split(man)
cats = list(ref['categories']); P = np.asarray(ref['prompt_emb'], np.float32)
hen, roo = cats.index('hen'), cats.index('rooster'); other = np.asarray([i for i in range(50) if i not in (hen, roo)])
prm = np.load(HERE / 'inside_params.npz'); R = float(prm['R'])
m = (vis['group'] == 'chicken') & (vis['kind'] == 'imitation') & ~fit
rows = [r for r, keep in zip(man, m) if keep]; E0 = vis['emb'][m]; A = vis['act_mean'][m].astype(np.float32)
u17, u19 = 17 * 2048 + 1574, 19 * 2048 + 1129
g = np.maximum(s19 * A[:, u19] - par['t19'], 0) * np.maximum(par['t17'] - s17 * A[:, u17], 0)
idx = np.where(g > 0)[0]
M = InsideModel(par['t19'], par['t17'], s17, s19)
audio = {int(i): decode(VIS / rows[i]['local'])[:16000 * 30] for i in idx}
out = {'R': R, 'clips': [rows[i]['id'] for i in idx], 'ladder': []}
for rel in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 6.0):
    for name, vec in (('chicken', prm['d']), ('random', prm['r'])):
        M.d = torch.from_numpy(np.asarray(vec, np.float32))
        E = []
        for i in idx:
            M.scale = rel / float(g[i])  # scale * g = rel, so the push is rel * |d| = rel (in units of R once * R)
            M.scale *= R
            E.append(M.embed(audio[int(i)]))
        M.scale = 0.0
        E = np.asarray(E); base = E0[idx]
        top = ea.prompt_predictions(E, P, cats); sc = ea.prompt_scores(E, P, hen, roo, other)
        cos = np.sum(E * base, axis=1)
        row = {'push_over_R': rel, 'direction': name, 'top': dict(Counter(map(str, top)).most_common()),
               'median_chicken_score': float(np.median(sc)), 'median_cos_to_original': float(np.median(cos))}
        out['ladder'].append(row); print(json.dumps(row), flush=True)
base = E0[idx]
out['unedited'] = {'top': dict(Counter(map(str, ea.prompt_predictions(base, P, cats))).most_common()),
                   'median_chicken_score': float(np.median(ea.prompt_scores(base, P, hen, roo, other)))}
print('unedited', json.dumps(out['unedited']))
(HERE / 'dose_ladder_inside.json').write_text(json.dumps(out, indent=1))
