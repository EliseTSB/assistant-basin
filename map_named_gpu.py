#!/usr/bin/env python3
# map_named_gpu.py — Part 11.10-11.13 on the Arc A770 (Claude, 2026-10-16; predictions pushed as assistant-basin ce07504
# before this ran). Elise, 17:50: what do SmolLM3's identity and name neurons code beyond identity; map them out.
#
# One model load, two passes over pre-tokenized wikitext-103 (tok_slices.py; this environment has no pyarrow):
#   units  M   the 28 name-attribution neurons (results/name-attr-pos28.json). Pole: the sign of the neuron's input to
#              down_proj at the last token of the raw_nosys prompt + "My name is" (the position predicting the name).
#          MC  28 neurons drawn at random from the same layers, random sign (rng seed, drawn in M's order).
#          I, C  Part 11.3's 20 identity neurons and 20 same-layer controls, exactly as in 11.3's key.json.
#   pass A (desc.npy, 99,840 tokens): each unit's 20 strongest contexts, as topk_named.py (first 8 tokens of each
#          512-token chunk skipped). M and MC go to units.json (shuffled, no arm label) for the blind description;
#          I and C are only compared with 11.3's CPU contexts (parity.json: how many of the 20 match, per unit).
#   pass B (held.npy, wikitext tokens 1,000,000-1,299,520): per unit 10 strongest (at least 4 apart), 10 mid-strength
#          (25-60% of its held-out maximum) and 10 random positions, as heldout.py (first 8 of each chunk, last 8 skipped).
# Output (out dir): units.json, key.json, heldout.json (M/MC by their new uids, I/C as "i-<11.3 uid>"), parity.json,
#          pole.json. Refuses (exit 3) if the card holds more than 14 GB after loading, before any pass.
# Usage: python map_named_gpu.py <slices dir> <11.3 dir> <name json> <out dir> [--model ...] [--seed 0] [--device xpu]
import argparse, json, os, random, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('slices'); ap.add_argument('ident'); ap.add_argument('names'); ap.add_argument('out')
ap.add_argument('--model', default='HuggingFaceTB/SmolLM3-3B')
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device', default='xpu')
a = ap.parse_args()
torch.set_grad_enabled(False)
rng = np.random.default_rng(a.seed); random.seed(a.seed)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).eval()
layers = model.model.layers; L, Fd = len(layers), layers[0].mlp.down_proj.in_features
dev = getattr(torch, a.device, None)
if dev is not None and hasattr(dev, 'memory_allocated'):
    gb = dev.memory_allocated() / 1e9
    print(f'loaded {a.model} on {a.device}: {gb:.2f} GB allocated ({time.time() - t0:.0f}s)', flush=True)
    if gb > 14: print('REFUSED: more than 14 GB on the card', flush=True); raise SystemExit(3)

cap = {}
def hook(l):
    def h(m, i): cap[l] = i[0][0]
    return h
hooks = [layers[l].mlp.down_proj.register_forward_pre_hook(hook(l)) for l in range(L)]

# Name neurons and their pole at the name position (raw_nosys exactly as ablate_smol.py builds it).
Q = 'What is your name?'
prompt = f'<|im_start|>user\n{Q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\nMy name is'
pids = tok(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(a.device)
model(input_ids=pids)
names = [(int(n[1:].split('.')[0]), int(n.split('.')[1]), float(v)) for n, v in json.load(open(a.names, encoding='utf-8'))['top50']]
pole = {}
units = []
for l, n, attr in names:
    act = float(cap[l][-1, n].float())
    s = 1 if act >= 0 else -1
    pole[f'L{l}.{n}'] = {'act_last': round(act, 4), 'sign': s, 'attribution': attr}
    units.append(('M', l, n, s, None))
taken = {(l, n) for _, l, n, _, _ in units}
ident_key = json.load(open(os.path.join(a.ident, 'key.json'), encoding='utf-8'))
for uid, k in ident_key.items(): taken.add((k['layer'], k['neuron']))
for _, l, _, _, _ in list(units):
    while True:
        n = int(rng.integers(Fd))
        if (l, n) not in taken: break
    taken.add((l, n)); units.append(('MC', l, n, int(rng.choice([1, -1])), None))
for uid, k in sorted(ident_key.items()):
    units.append((k['arm'], k['layer'], k['neuron'], k['sign'], uid))
nu = len(units)
by_layer = {}
for ui, (_, l, n, s, _) in enumerate(units): by_layer.setdefault(l, []).append((ui, n, s))
cols = {l: torch.tensor([n for _, n, _ in v], device=a.device) for l, v in by_layer.items()}
signs = {l: torch.tensor([s for _, _, s in v], dtype=torch.float32, device=a.device) for l, v in by_layer.items()}
rows = {l: [ui for ui, _, _ in v] for l, v in by_layer.items()}
print(f'{nu} units ({sum(u[0] == "M" for u in units)} M, {sum(u[0] == "MC" for u in units)} MC, '
      f'{sum(u[0] in ("I", "C") for u in units)} from 11.3) in {len(by_layer)} layers', flush=True)

SEQ = 512
def run(ids):
    nch = len(ids) // SEQ
    acts = np.zeros((nu, nch * SEQ), dtype=np.float32)
    t1 = time.time()
    for c in range(nch):
        model(input_ids=torch.tensor(ids[c * SEQ:(c + 1) * SEQ], device=a.device)[None])
        for l in by_layer:
            v = (cap[l][:, cols[l]].float() * signs[l]).T.cpu().numpy()
            acts[rows[l], c * SEQ:(c + 1) * SEQ] = v
        if c % 50 == 0:
            print(f'  chunk {c}/{nch}, {time.time() - t1:.0f}s', flush=True)
            if dev is not None and hasattr(dev, 'empty_cache'): dev.empty_cache()
    return acts

def context(ids, p):
    lo, hi = max(0, p - 24), min(len(ids), p + 7)
    return tok.decode(ids[lo:p].tolist()) + '⟦' + tok.decode([int(ids[p])]) + '⟧' + tok.decode(ids[p + 1:hi].tolist())

# Pass A: description contexts.
desc = np.load(os.path.join(a.slices, 'desc.npy'))
A = run(desc)
for c in range(len(desc) // SEQ): A[:, c * SEQ: c * SEQ + 8] = -np.inf
topA = np.argsort(-A, 1)[:, :20]
old_units = json.load(open(os.path.join(a.ident, 'units.json'), encoding='utf-8'))['units']
old_ctx = {u['id']: u['contexts'] for u in old_units}
parity = {}
pub, key = [], {}
order = [ui for ui, u in enumerate(units) if u[0] in ('M', 'MC')]; random.shuffle(order)
uid_of = {}
for j, ui in enumerate(order):
    arm, l, n, s, _ = units[ui]; uid = f'n{j:03d}'; uid_of[ui] = uid
    pub.append({'id': uid, 'contexts': [context(desc, int(p)) for p in topA[ui]]})
    key[uid] = {'arm': arm, 'layer': l, 'neuron': n, 'sign': s, 'top_values': [round(float(A[ui, p]), 4) for p in topA[ui]]}
for ui, (arm, l, n, s, old) in enumerate(units):
    if old is None: continue
    uid_of[ui] = f'i-{old}'
    mine = [context(desc, int(p)) for p in topA[ui]]
    parity[old] = {'arm': arm, 'layer': l, 'neuron': n, 'matched_of_20': len(set(mine) & set(old_ctx[old])),
                   'top_value_gpu': round(float(A[ui, topA[ui][0]]), 4), 'top_value_cpu': ident_key[old]['top_values'][0]}
os.makedirs(a.out, exist_ok=True)
json.dump({'model': a.model, 'tokens': int(len(desc)), 'units': pub}, open(os.path.join(a.out, 'units.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
json.dump(key, open(os.path.join(a.out, 'key.json'), 'w', encoding='utf-8'), indent=0)
json.dump(parity, open(os.path.join(a.out, 'parity.json'), 'w', encoding='utf-8'), indent=0)
json.dump(pole, open(os.path.join(a.out, 'pole.json'), 'w', encoding='utf-8'), indent=0)
m = [p['matched_of_20'] for p in parity.values()]
print(f'pass A done: {len(pub)} units to describe; parity with 11.3 median {int(np.median(m))}/20, min {min(m)}/20', flush=True)

# Pass B: held-out contexts.
held = np.load(os.path.join(a.slices, 'held.npy'))
B = run(held)
nch = len(held) // SEQ
valid = np.ones(nch * SEQ, bool)
for c in range(nch): valid[c * SEQ: c * SEQ + 8] = False
valid[-8:] = False
out = {}
for ui, u in enumerate(units):
    v = B[ui].copy(); v[~valid] = -np.inf
    top = []
    for p in np.argsort(-v):
        if all(abs(int(p) - q) >= 4 for q in top): top.append(int(p))
        if len(top) == 10: break
    mx = float(v[top[0]])
    midpool = np.where(valid & (v >= 0.25 * mx) & (v <= 0.60 * mx))[0]
    mid = [int(p) for p in rng.choice(midpool, size=min(10, len(midpool)), replace=False)] if len(midpool) else []
    rnd = [int(p) for p in rng.choice(np.where(valid)[0], size=10, replace=False)]
    items = [{'kind': k, 'act': round(float(v[p]), 4), 'context': context(held, p)} for k, ps in (('top', top), ('mid', mid), ('random', rnd)) for p in ps]
    out[uid_of[ui]] = {'arm': u[0], 'layer': u[1], 'neuron': u[2], 'sign': u[3], 'max': round(mx, 4), 'items': items}
json.dump(out, open(os.path.join(a.out, 'heldout.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
short = sum(1 for x in out.values() if sum(i['kind'] == 'mid' for i in x['items']) < 10)
for h in hooks: h.remove()
print(f'pass B done: {len(out)} units ({short} with fewer than 10 mid contexts); total {time.time() - t0:.0f}s', flush=True)
