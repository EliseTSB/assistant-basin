#!/usr/bin/env python3
# topk.py — strongest contexts for neuron poles, random directions and a random-context floor (Claude, 2026-10-15).
#
# WHAT IT MEASURES: on N tokens of Wikipedia (wikitext-103), the 20 contexts where each unit is most strongly active.
# Units, at layers sampled evenly from the early, middle and late thirds of the model:
#   N  a real MLP neuron pole: the hidden unit silu(gate)*up, with a random sign (+ = positive pole, - = negative)
#   R  a random unit direction in the same layer's hidden space (activation = hidden vector . direction)
#   F  the floor: 20 random token positions, no unit at all (a judge that calls these "one job" is too lenient)
# Output: <out>/units.json — shuffled, with no arm label (that is what the judges see), and <out>/key.json — the arms.
# Usage: python topk.py <model> <out dir> [--tokens 1000000] [--per-third 50] [--seed 0]
import argparse, json, os, random, time
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('out')
ap.add_argument('--tokens', type=int, default=1_000_000); ap.add_argument('--per-third', type=int, default=50)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--k', type=int, default=20)
ap.add_argument('--corpus', default=os.environ.get('WIKITEXT_PARQUET', 'data/wikitext103-train0.parquet'))
a = ap.parse_args()
torch.set_grad_enabled(False)
rng = np.random.default_rng(a.seed); random.seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
layers = model.model.layers; L, Fd = len(layers), layers[0].mlp.down_proj.in_features

# Corpus: wikitext lines, headings and blanks dropped, joined into one token stream, cut into 512-token chunks.
lines = [t for t in pq.read_table(a.corpus, columns=['text']).column('text').to_pylist() if t.strip() and not t.strip().startswith('=')]
ids = []
for t in lines:
    ids.extend(tok(t, add_special_tokens=False)['input_ids'])
    if len(ids) >= a.tokens: break
ids = np.array(ids[: a.tokens]); SEQ = 512; nch = len(ids) // SEQ; ids = ids[: nch * SEQ]
print(f'{a.model}: {L}x{Fd}; corpus {len(ids)} tokens in {nch} chunks', flush=True)

thirds = [range(0, L // 3), range(L // 3, 2 * L // 3), range(2 * L // 3, L)]
units = []  # (arm, layer, neuron or None, sign or None, direction index or None)
dirs = {}
for th in thirds:
    for _ in range(a.per_third):
        units.append(('N', int(rng.choice(th)), int(rng.integers(Fd)), int(rng.choice([1, -1])), None))
    for _ in range(a.per_third):
        l = int(rng.choice(th)); v = rng.standard_normal(Fd); v /= np.linalg.norm(v)
        dirs.setdefault(l, []).append(v); units.append(('R', l, None, None, len(dirs[l]) - 1))
for _ in range(a.per_third):
    units.append(('F', None, None, None, None))
need = sorted({u[1] for u in units if u[1] is not None})
D = {l: torch.tensor(np.stack(dirs[l]).T, dtype=torch.float32) for l in dirs}  # [Fd, k]

cap = {}
hooks = [layers[l].mlp.down_proj.register_forward_pre_hook((lambda l: lambda m, i: cap.__setitem__(l, i[0][0]))(l)) for l in need]
K = a.k; nu = len(units)
topv = np.full((nu, K), -np.inf, dtype=np.float32); topp = np.zeros((nu, K), dtype=np.int64)
t0 = time.time()
for c in range(nch):
    model(input_ids=torch.tensor(ids[c * SEQ:(c + 1) * SEQ])[None])
    vals = np.zeros((nu, SEQ), dtype=np.float32)
    proj = {l: (cap[l] @ D[l]).numpy() for l in D}
    rcount = {l: 0 for l in D}
    for ui, (arm, l, n, s, di) in enumerate(units):
        if arm == 'N': vals[ui] = s * cap[l][:, n].numpy()
        elif arm == 'R': vals[ui] = proj[l][:, di]
        else: vals[ui] = -np.inf
    vals[:, :8] = -np.inf  # skip the first tokens of a chunk: too little context to show
    pos = np.arange(SEQ) + c * SEQ
    allv = np.concatenate([topv, vals], 1); allp = np.concatenate([topp, np.broadcast_to(pos, (nu, SEQ))], 1)
    o = np.argsort(-allv, 1)[:, :K]
    topv = np.take_along_axis(allv, o, 1); topp = np.take_along_axis(allp, o, 1)
    if c % 50 == 0: print(f'  chunk {c}/{nch}, {time.time() - t0:.0f}s', flush=True)
for h in hooks: h.remove()

def context(p):
    lo, hi = max(0, p - 24), min(len(ids), p + 7)
    return tok.decode(ids[lo:p].tolist()) + '⟦' + tok.decode([int(ids[p])]) + '⟧' + tok.decode(ids[p + 1:hi].tolist())

order = list(range(nu)); random.shuffle(order)
pub, key = [], {}
for j, ui in enumerate(order):
    arm, l, n, s, di = units[ui]
    uid = f'u{j:03d}'
    if arm == 'F':
        ps = rng.integers(8, len(ids) - 8, K)
    else:
        ps = topp[ui]
    pub.append({'id': uid, 'contexts': [context(int(p)) for p in ps]})
    key[uid] = {'arm': arm, 'layer': l, 'neuron': n, 'sign': s, 'dir': di,
                'third': None if l is None else next(i for i, th in enumerate(thirds) if l in th),
                'top_values': [round(float(v), 4) for v in topv[ui]] if arm != 'F' else None}
os.makedirs(a.out, exist_ok=True)
json.dump({'model': a.model, 'tokens': int(len(ids)), 'units': pub}, open(os.path.join(a.out, 'units.json'), 'w'), ensure_ascii=False, indent=0)
json.dump(key, open(os.path.join(a.out, 'key.json'), 'w'), indent=0)
print(f'wrote {nu} units to {a.out} in {time.time() - t0:.0f}s', flush=True)
