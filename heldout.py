#!/usr/bin/env python3
# heldout.py — held-out contexts for Part 1d (Claude, 2026-10-16; predictions 1d.1-1d.4 pushed as assistant-basin b57ae45
# before this ran). For each N and R unit of a topk.py run, collect on a slice of wikitext-103 the explanation never saw
# (tokens START to START+TOKENS; topk.py used the first 1,000,000): the unit's 10 strongest contexts (spaced at least 4
# tokens apart, so one phrase is not counted twice), 10 mid-strength ones (activation 25-60% of its held-out maximum,
# sampled at random) and 10 random positions, each with its true activation.
#
# The units are rebuilt from topk.py's own seed and draw order, and checked against key.json: every N unit's (layer,
# neuron, sign) and every R unit's (layer, direction index) must be reproduced, or the run refuses (exit 3).
# Usage: python heldout.py <model> <topk out dir> [--start 1000000] [--tokens 300000] [--seed 0] [--per-third 50]
# ⚠ FIRST RUN (2026-10-16): matched 299 of 300 units because Part 1 drew one neuron pole twice (u121 and u257, L11
#   n2409 sign -1); the duplicate's uid is the later one. CPU, 585 chunks of 512 tokens in 1239 s.
# Output: <dir>/heldout.json  {uid: {"arm", "max", "items": [{"kind", "act", "context"}]}}
import argparse, json, os, random, time
import numpy as np, pyarrow.parquet as pq, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('dir')
ap.add_argument('--start', type=int, default=1_000_000); ap.add_argument('--tokens', type=int, default=300_000)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--per-third', type=int, default=50)
ap.add_argument('--corpus', default=os.environ.get('WIKITEXT_PARQUET', 'data/wikitext103-train0.parquet'))
a = ap.parse_args()
torch.set_grad_enabled(False)
rng = np.random.default_rng(a.seed); random.seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
layers = model.model.layers; L, Fd = len(layers), layers[0].mlp.down_proj.in_features

# Rebuild topk.py's units with the identical rng draw order (this block must stay byte-for-byte its logic).
thirds = [range(0, L // 3), range(L // 3, 2 * L // 3), range(2 * L // 3, L)]
units, dirs = [], {}
for th in thirds:
    for _ in range(a.per_third):
        units.append(('N', int(rng.choice(th)), int(rng.integers(Fd)), int(rng.choice([1, -1])), None))
    for _ in range(a.per_third):
        l = int(rng.choice(th)); v = rng.standard_normal(Fd); v /= np.linalg.norm(v)
        dirs.setdefault(l, []).append(v); units.append(('R', l, None, None, len(dirs[l]) - 1))

key = json.load(open(os.path.join(a.dir, 'key.json')))
want = {}
for uid, k in key.items():
    if k['arm'] == 'N': want[('N', k['layer'], k['neuron'], k['sign'], None)] = uid
    elif k['arm'] == 'R': want[('R', k['layer'], None, None, k['dir'])] = uid
got = {u: want.get(u) for u in units}
missing = [u for u, uid in got.items() if uid is None]
if missing or len(got) != len(want):
    print(f'REFUSED: rebuilt units do not match key.json ({len(missing)} missing, {len(got)} vs {len(want)})', flush=True)
    raise SystemExit(3)
print(f'units rebuilt and matched to key.json: {len(got)}', flush=True)

lines = [t for t in pq.read_table(a.corpus, columns=['text']).column('text').to_pylist() if t.strip() and not t.strip().startswith('=')]
ids = []
for t in lines:
    ids.extend(tok(t, add_special_tokens=False)['input_ids'])
    if len(ids) >= a.start + a.tokens: break
if len(ids) < a.start + a.tokens:
    print(f'REFUSED: corpus has {len(ids)} tokens, need {a.start + a.tokens}', flush=True); raise SystemExit(3)
ids = np.array(ids[a.start: a.start + a.tokens]); SEQ = 512; nch = len(ids) // SEQ; ids = ids[: nch * SEQ]
print(f'held-out slice: {len(ids)} tokens from {a.start}, {nch} chunks', flush=True)

need = sorted({u[1] for u in units})
D = {l: torch.tensor(np.stack(dirs[l]).T, dtype=torch.float32) for l in dirs}
cap = {}
hooks = [layers[l].mlp.down_proj.register_forward_pre_hook((lambda l: lambda m, i: cap.__setitem__(l, i[0][0]))(l)) for l in need]
nu = len(units)
acts = np.zeros((nu, nch * SEQ), dtype=np.float16)
t0 = time.time()
for c in range(nch):
    model(input_ids=torch.tensor(ids[c * SEQ:(c + 1) * SEQ])[None])
    proj = {l: (cap[l] @ D[l]).numpy() for l in D}
    for ui, (arm, l, n, s, di) in enumerate(units):
        acts[ui, c * SEQ:(c + 1) * SEQ] = s * cap[l][:, n].numpy() if arm == 'N' else proj[l][:, di]
    if c % 50 == 0: print(f'  chunk {c}/{nch}, {time.time() - t0:.0f}s', flush=True)
for h in hooks: h.remove()

valid = np.ones(nch * SEQ, bool)
for c in range(nch): valid[c * SEQ: c * SEQ + 8] = False   # as topk.py: too little context at a chunk's start
valid[-8:] = False

def context(p):
    lo, hi = max(0, p - 24), min(len(ids), p + 7)
    return tok.decode(ids[lo:p].tolist()) + '⟦' + tok.decode([int(ids[p])]) + '⟧' + tok.decode(ids[p + 1:hi].tolist())

out = {}
for ui, u in enumerate(units):
    v = acts[ui].astype(np.float32); v[~valid] = -np.inf
    order = np.argsort(-v); top = []
    for p in order:
        if all(abs(int(p) - q) >= 4 for q in top): top.append(int(p))
        if len(top) == 10: break
    mx = float(v[top[0]])
    lo, hi = 0.25 * mx, 0.60 * mx
    midpool = np.where(valid & (v >= lo) & (v <= hi))[0]
    mid = [int(p) for p in rng.choice(midpool, size=min(10, len(midpool)), replace=False)] if len(midpool) else []
    rnd = [int(p) for p in rng.choice(np.where(valid)[0], size=10, replace=False)]
    items = [{'kind': k, 'act': round(float(v[p]), 4), 'context': context(p)} for k, ps in (('top', top), ('mid', mid), ('random', rnd)) for p in ps]
    out[got[u]] = {'arm': u[0], 'max': round(mx, 4), 'items': items}
json.dump(out, open(os.path.join(a.dir, 'heldout.json'), 'w'), ensure_ascii=False, indent=0)
short = sum(1 for x in out.values() if sum(i['kind'] == 'mid' for i in x['items']) < 10)
print(f'wrote heldout.json: {len(out)} units ({short} with fewer than 10 mid contexts) in {time.time() - t0:.0f}s', flush=True)
