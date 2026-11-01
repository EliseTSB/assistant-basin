#!/usr/bin/env python3
"""Recompute Part 168 n+ neurons from seeds and check freshness against Parts 165 and 167 (n+ = perm[0])."""
import json, sys
from pathlib import Path
import torch
ROOT = Path.cwd()
norms = json.loads((ROOT / "results/part168/norms.json").read_text(encoding="utf-8"))["directions"]
def perm(width, layer, slot, base, attempt=0):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(base) + 100 * int(layer) + int(slot) + 1_000_000 * int(attempt))
    return torch.randperm(int(width), generator=g, device="cpu")
n_plus = [n for n in norms if n["arm"] == "n+"]
mism = []
old = {}
for L in range(19, 27):
    for s in range(12):
        for b in (165000, 167000):
            old[(L, s, b)] = int(perm(9216, L, s, b)[0])
fresh_fail = 0
used = []
for n in n_plus:
    L, s = n["layer"], n["slot"]
    att = n.get("attempt", 0)
    p = perm(n.get("width", 9216), L, s, 168000, att)
    neu = int(p[0])
    if neu != n["neuron"]:
        mism.append((L, s, neu, n["neuron"]))
    if neu in (old[(L, s, 165000)], old[(L, s, 167000)]):
        fresh_fail += 1
    used.append((L, s, n["neuron"]))
print("n+ rows", len(n_plus), "attempt values", sorted(set(n.get("attempt", 0) for n in n_plus)))
print("stored neuron != recomputed perm[0] at attempt:", len(mism), mism[:5])
print("n+ neuron equal to Part165/167 n+ at same layer & slot:", fresh_fail)
print("n+ neuron equal to any Part165/167 n+ at same layer (any slot):",
      sum(1 for (L, s, neu) in used if any(old[(L, t, b)] == neu for t in range(12) for b in (165000, 167000))))
print("duplicate n+ neuron within layer:", len(used) - len({(L, neu) for (L, s, neu) in used}))
# k4 sets: does Part 168 k4 (perm[:4] at 168000) share neurons with Part 165/167 k4 or n+?
k4_overlap = 0
for L in range(19, 27):
    for s in range(12):
        p168 = set(perm(9216, L, s, 168000)[:4].tolist())
        other = set()
        for t in range(12):
            for b in (165000, 167000):
                other |= set(perm(9216, L, t, b)[:4].tolist())
        k4_overlap += len(p168 & other)
print("Part168 k4 neuron indices also in some Part165/167 k4 set at same layer (sum over layer,slot):", k4_overlap)
