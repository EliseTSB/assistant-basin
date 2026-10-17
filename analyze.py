#!/usr/bin/env python3
# analyze.py — identity neurons in one collected model, and how a fixed set of them scores in other models.
# (Claude, 2026-10-15.)
#
# SCORE: for a neuron, d(A, X) = (mean over prompts of A - mean of X) / pooled standard deviation, for X in B
# (third person about the AI), C (a person, first person) and D (the AI's sentences in the USER turn). The identity
# score is the MINIMUM of the three, so a neuron scores high only if it beats every control. Prompts are split by index
# parity: neurons are RANKED on the even half and SCORED on the odd half, so the reported number is held out.
# NULL: the same pipeline after shuffling which prompts are labelled A versus X (within each pair), 20 times; the
# held-out score of the null top neurons says what selection alone produces.
#
# Usage: python analyze.py <discover.npz> [--form plain|chat] [--top 50] [--score other.npz ...]
#   --score: take the top neurons found in <discover.npz> and print their held-out-half identity score in each other
#   model (same form), plus the Claude contrast (E_claude vs A, E_claude vs C) in each.
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('discover'); ap.add_argument('--form', default='plain'); ap.add_argument('--top', type=int, default=50)
ap.add_argument('--score', nargs='*', default=[]); ap.add_argument('--null-reps', type=int, default=20)
ap.add_argument('--json'); ap.add_argument('--frames', default=os.path.join(os.path.dirname(__file__), 'data', 'frames.json'))
a = ap.parse_args()

def load(path, form):
    z = np.load(path)
    fam = {k.split('_', 2)[2]: z[k].astype(np.float32).reshape(z[k].shape[0], -1) for k in z.files if k.startswith(f'acts_{form}_')}
    return fam, int(z['layers']), int(z['neurons'])

def dmat(a, b):
    sd = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2) + 1e-6
    return (a.mean(0) - b.mean(0)) / sd

def identity(fam, rows):
    A = fam['A'][rows]
    return np.minimum.reduce([dmat(A, fam[x][rows]) for x in ('B', 'C', 'D')])

fam, L, F = load(a.discover, a.form)
n = fam['A'].shape[0]; even, odd = np.arange(0, n, 2), np.arange(1, n, 2)
disc = identity(fam, even); held = identity(fam, odd)
top = np.argsort(-disc)[: a.top]
rng = np.random.default_rng(0)
rand = rng.choice(L * F, a.top, replace=False)

null = []
for r in range(a.null_reps):
    shuf = {k: v.copy() for k, v in fam.items()}
    for x in ('B', 'C', 'D'):  # swap A and X for a random half of the prompts
        flip = rng.random(n) < 0.5
        shuf['A'][flip], shuf[x][flip] = fam[x][flip], fam['A'][flip]
    nd, nh = identity(shuf, even), identity(shuf, odd)
    null.append(float(np.median(nh[np.argsort(-nd)[: a.top]])))

def where(i): return f'L{i // F}.{i % F}'
out = {'model': a.discover, 'form': a.form, 'layers': L, 'neurons': F,
       'heldout_top_median': float(np.median(held[top])), 'heldout_random_median': float(np.median(held[rand])),
       'null_top_median': float(np.median(null)), 'null_top_max': float(np.max(null)),
       'layer_histogram_top': np.bincount(top // F, minlength=L).tolist(),
       'top10': [(where(int(i)), round(float(disc[i]), 2), round(float(held[i]), 2)) for i in top[:10]],
       'top50': [(where(int(i)), round(float(disc[i]), 2), round(float(held[i]), 2)) for i in top]}

# Only frames whose sentence contains the name or maker ({N}/{M}) can show a name swap: 49 of the 72 frames carry no
# name, so their A and E rows are identical and averaging over them shrank the Claude contrast about threefold (found
# 2026-10-15 20:11 by the name test; the first part-2 numbers were diluted this way).
def _named_mask(path):
    spec = json.load(open(path, encoding='utf-8'))
    if 'frames' not in spec:  # Register datasets have no name-swap families.
        return np.zeros(len(spec.get('items', [])), dtype=bool)
    fr = spec['frames']
    default_frames = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'frames.json')
    if os.path.abspath(path) == default_frames:
        ex = os.path.join(os.path.dirname(default_frames), 'frames-extra.json')
        if os.path.exists(ex): fr = fr + json.load(open(ex, encoding='utf-8'))['frames']
    return np.array([('{N}' in f[0] or '{M}' in f[0]) for f in fr])
NAMED = _named_mask(a.frames)

def claude_contrast(f2, idx):
    m = NAMED[: f2['A'].shape[0]]
    A, C, Ec, Eg = f2['A'][m][:, idx], f2['C'][m][:, idx], f2['E_claude'][m][:, idx], f2['E_chatgpt'][m][:, idx]
    return {'claude_minus_olmo': float(np.median(dmat(Ec, A))), 'claude_minus_person': float(np.median(dmat(Ec, C))),
            'chatgpt_minus_person': float(np.median(dmat(Eg, C))), 'claude_minus_chatgpt': float(np.median(dmat(Ec, Eg))),
            'olmo_minus_person': float(np.median(dmat(A, C))), 'named_frames': int(m.sum())}
def has_name_families(f): return 'E_claude' in f and 'E_chatgpt' in f
if has_name_families(fam) and NAMED[: fam['A'].shape[0]].any():
    out['claude_in_discover'] = claude_contrast(fam, top)

scores = []
for p in a.score:
    f2, L2, F2 = load(p, a.form)
    if (L2, F2) != (L, F): scores.append({'model': p, 'error': f'shape {L2}x{F2} != {L}x{F}'}); continue
    h2 = identity(f2, odd); all2 = identity(f2, np.arange(n))
    score = {'model': p, 'heldout_top_median': float(np.median(h2[top])),
             'share_top_with_half_or_more': float(np.mean(all2[top] >= 0.5 * identity(fam, np.arange(n))[top]))}
    if has_name_families(f2) and NAMED[: f2['A'].shape[0]].any():
        score['claude'] = claude_contrast(f2, top)
    scores.append(score)
out['scored_in'] = scores
print(json.dumps(out, indent=1))
if a.json: open(a.json, 'w').write(json.dumps(out, indent=1))
