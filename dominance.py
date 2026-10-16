#!/usr/bin/env python3
# dominance.py — for each random direction in part 1 (rebuilt from topk.py's seed and draw order), at its 20 strongest
# positions on a 4K-token Wikipedia slice, the share of |projection| contributed by the single largest neuron term.
# A high share means the random direction is mostly one loud neuron, so it would inherit that neuron's cleanliness.
import os
import numpy as np, torch, pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'); m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype=torch.float32).eval()
Ls = m.model.layers; Fd = Ls[0].mlp.down_proj.in_features; L = len(Ls)
rng = np.random.default_rng(0)
thirds = [range(0, L // 3), range(L // 3, 2 * L // 3), range(2 * L // 3, L)]
dirs = []
for th in thirds:
    for _ in range(50): int(rng.choice(th)); int(rng.integers(Fd)); int(rng.choice([1, -1]))
    for _ in range(50):
        l = int(rng.choice(th)); v = rng.standard_normal(Fd); v /= np.linalg.norm(v); dirs.append((l, v))
# Read in batches: loading the whole 157 MB parquet as Python strings took this past a 5 GB cap twice (10-15).
def _lines():
    for b in pq.ParquetFile(os.environ.get('WIKITEXT_PARQUET', 'data/wikitext103-train0.parquet')).iter_batches(batch_size=2000, columns=['text']):
        for t in b.column(0).to_pylist():
            if t.strip() and not t.strip().startswith('='): yield t
ids = []
for t in _lines():
    ids.extend(tok(t, add_special_tokens=False)['input_ids'])
    if len(ids) >= 4096: break
need = sorted({l for l, _ in dirs}); cap = {}
for l in need: Ls[l].mlp.down_proj.register_forward_pre_hook((lambda l: lambda mod, i: cap.__setitem__(l, i[0][0].clone()))(l))
acts = {l: [] for l in need}
for c in range(8):  # 4,096 tokens: 20,480 held 6 GB of activations and was OOM-killed at a 5 GB cap
    m(input_ids=torch.tensor(ids[c * 512:(c + 1) * 512])[None])
    for l in need: acts[l].append(cap[l].numpy())
acts = {l: np.concatenate(v) for l, v in acts.items()}
s = []
for l, v in dirs:
    X = acts[l]; top = np.argsort(-(X @ v))[:20]; t = np.abs(X[top] * v)
    s.append(float(np.median(t.max(1) / t.sum(1))))
s = np.array(s)
print(f'median {np.median(s):.2f}  p90 {np.percentile(s, 90):.2f}  share > 0.5: {np.mean(s > 0.5):.0%}  share > 0.25: {np.mean(s > 0.25):.0%}')
