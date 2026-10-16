#!/usr/bin/env python3
# corr_lean.py <a.npz> <b.npz> — corr.py's numbers in a fraction of the memory (Claude, 2026-10-16).
# WHY: corr.py converts both whole `neutral` arrays (1,024 tokens x L*F neurons) to float32 and then builds several
# full-size temporaries and a full permuted copy; on Qwen2.5-Coder 7B (28 x 18,944 neurons) that ran past a 7 GB cap
# and was OOM-killed at 07:25. Same formulas, same random-partner permutation (seed 0), computed in column chunks
# from the stored arrays, so the printout is identical; checked against corr.py on a pair it already handled.
import sys
import numpy as np

za, zb = np.load(sys.argv[1]), np.load(sys.argv[2])
assert (za['neutral_ids'] == zb['neutral_ids']).all(), 'different neutral tokens'
A, B = za['neutral'], zb['neutral']          # kept in their stored dtype; chunks are converted to float32 below
L, F = int(za['layers']), int(za['neurons'])
N = L * F
def pear(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return (X * Y).sum(0) / (np.sqrt((X * X).sum(0) * (Y * Y).sum(0)) + 1e-9)
rng = np.random.default_rng(0); perm = rng.permutation(N)
r = np.empty(N, np.float32); rc = np.empty(N, np.float32)
C = 16384
for s in range(0, N, C):
    j = slice(s, min(s + C, N))
    a = A[:, j].astype(np.float32)
    r[j] = pear(a, B[:, j].astype(np.float32))
    rc[j] = pear(a, B[:, perm[j]].astype(np.float32))
q = lambda x: ' '.join(f'p{p}={np.percentile(x, p):.3f}' for p in (5, 25, 50, 75, 95))
print(f'{sys.argv[1].split("/")[-1]} vs {sys.argv[2].split("/")[-1]}: {L}x{F}')
print('  same neuron   :', q(r))
print('  random partner:', q(rc))
print('  share of neurons with r > 0.8:', f'{np.mean(r > 0.8):.1%}')
print('  per-layer median:', ' '.join(f'{np.median(r[l * F:(l + 1) * F]):.2f}' for l in range(L)))
