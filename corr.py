#!/usr/bin/env python3
# corr.py <a.npz> <b.npz> — is neuron i the same unit in two checkpoints? (Claude, 2026-10-15)
# Per neuron, the Pearson correlation of its activations on the same 1,024 tokens of neutral text (Pride and
# Prejudice) in two models. Prediction 1 of the identity probe: median above 0.8 between base and Instruct. Prints the
# median and percentiles overall and per layer, and the same for a CONTROL that pairs neuron i in A with a random
# neuron in B (what correlation two unrelated units have on this text).
import sys
import numpy as np

za, zb = np.load(sys.argv[1]), np.load(sys.argv[2])
assert (za['neutral_ids'] == zb['neutral_ids']).all(), 'different neutral tokens'
A, B = za['neutral'].astype(np.float32), zb['neutral'].astype(np.float32)
L, F = int(za['layers']), int(za['neurons'])
def pear(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return (X * Y).sum(0) / (np.sqrt((X * X).sum(0) * (Y * Y).sum(0)) + 1e-9)
r = pear(A, B)
rng = np.random.default_rng(0); perm = rng.permutation(L * F)
rc = pear(A, B[:, perm])
q = lambda x: ' '.join(f'p{p}={np.percentile(x, p):.3f}' for p in (5, 25, 50, 75, 95))
print(f'{sys.argv[1].split("/")[-1]} vs {sys.argv[2].split("/")[-1]}: {L}x{F}')
print('  same neuron   :', q(r))
print('  random partner:', q(rc))
print('  share of neurons with r > 0.8:', f'{np.mean(r > 0.8):.1%}')
print('  per-layer median:', ' '.join(f'{np.median(r[l * F:(l + 1) * F]):.2f}' for l in range(L)))
