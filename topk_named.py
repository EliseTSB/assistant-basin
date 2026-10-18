#!/usr/bin/env python3
# topk_named.py — strongest general-text contexts for NAMED neurons (Claude, 2026-10-16; Part 11.3, predictions pushed as
# assistant-basin ad8c3c9 before any SmolLM3 run). Elise 10:51: "I'd love to map out identity related neural circuitry."
#
# WHAT IT MEASURES: for the top K identity neurons from analyze.py's JSON (arm I, positive pole: the identity score is
# "fires more for the model speaking as itself") and K control neurons drawn at random from the SAME layers (arm C,
# random sign, seeded), the 20 contexts on wikitext-103 where each is most strongly active. Output in topk.py's format
# (units.json without arm labels, key.json with them), so judge.py and score_sweep.py read it unchanged.
# ⚠ FIRST RUN (2026-10-16): the judge step in run-topk-smollm3.sh first lacked --effort high (copied from a script
#   that had it); caught before the judge ran. 195 chunks of 512 tokens in about 520 s on the CPU in bfloat16.
# Usage: python topk_named.py <model> <analyze json> <out dir> [--k 10] [--tokens 100000] [--seed 0] [--dtype bfloat16]
import argparse, json, os, random, time
import numpy as np, pyarrow.parquet as pq, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('analysis'); ap.add_argument('out')
ap.add_argument('--k', type=int, default=10); ap.add_argument('--tokens', type=int, default=100_000)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--dtype', default='bfloat16', choices=['float32', 'bfloat16'])
ap.add_argument('--corpus', default=os.environ.get('WIKITEXT_PARQUET', 'data/wikitext103-train0.parquet'))
a = ap.parse_args()
torch.set_grad_enabled(False)
rng = np.random.default_rng(a.seed); random.seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).eval()
layers = model.model.layers; L, Fd = len(layers), layers[0].mlp.down_proj.in_features

top = json.load(open(a.analysis))['top50'][: a.k]
units = [('I', int(n.split('.')[0][1:]), int(n.split('.')[1]), 1) for n, *_ in top]
taken = {(l, n) for _, l, n, _ in units}
for _, l, _, _ in list(units):
    while True:
        n = int(rng.integers(Fd))
        if (l, n) not in taken: break
    taken.add((l, n)); units.append(('C', l, n, int(rng.choice([1, -1]))))

lines = [t for t in pq.read_table(a.corpus, columns=['text']).column('text').to_pylist() if t.strip() and not t.strip().startswith('=')]
ids = []
for t in lines:
    ids.extend(tok(t, add_special_tokens=False)['input_ids'])
    if len(ids) >= a.tokens: break
ids = np.array(ids[: a.tokens]); SEQ = 512; nch = len(ids) // SEQ; ids = ids[: nch * SEQ]
print(f'{a.model}: {L}x{Fd}; {len(units)} units; corpus {len(ids)} tokens in {nch} chunks', flush=True)

need = sorted({u[1] for u in units})
cap = {}
hooks = [layers[l].mlp.down_proj.register_forward_pre_hook((lambda l: lambda m, i: cap.__setitem__(l, i[0][0].float()))(l)) for l in need]
K = 20; nu = len(units)
topv = np.full((nu, K), -np.inf, dtype=np.float32); topp = np.zeros((nu, K), dtype=np.int64)
t0 = time.time()
for c in range(nch):
    model(input_ids=torch.tensor(ids[c * SEQ:(c + 1) * SEQ])[None])
    vals = np.stack([s * cap[l][:, n].numpy() for _, l, n, s in units]).astype(np.float32)
    vals[:, :8] = -np.inf
    pos = np.arange(SEQ) + c * SEQ
    allv = np.concatenate([topv, vals], 1); allp = np.concatenate([topp, np.broadcast_to(pos, (nu, SEQ))], 1)
    o = np.argsort(-allv, 1)[:, :K]
    topv = np.take_along_axis(allv, o, 1); topp = np.take_along_axis(allp, o, 1)
    if c % 25 == 0: print(f'  chunk {c}/{nch}, {time.time() - t0:.0f}s', flush=True)
for h in hooks: h.remove()

def context(p):
    lo, hi = max(0, p - 24), min(len(ids), p + 7)
    return tok.decode(ids[lo:p].tolist()) + '⟦' + tok.decode([int(ids[p])]) + '⟧' + tok.decode(ids[p + 1:hi].tolist())

order = list(range(nu)); random.shuffle(order)
pub, key = [], {}
for j, ui in enumerate(order):
    arm, l, n, s = units[ui]; uid = f'u{j:03d}'
    pub.append({'id': uid, 'contexts': [context(int(p)) for p in topp[ui]]})
    key[uid] = {'arm': arm, 'layer': l, 'neuron': n, 'sign': s, 'top_values': [round(float(v), 4) for v in topv[ui]]}
os.makedirs(a.out, exist_ok=True)
json.dump({'model': a.model, 'tokens': int(len(ids)), 'units': pub}, open(os.path.join(a.out, 'units.json'), 'w'), ensure_ascii=False, indent=0)
json.dump(key, open(os.path.join(a.out, 'key.json'), 'w'), indent=0)
print(f'wrote {nu} units to {a.out} in {time.time() - t0:.0f}s', flush=True)
