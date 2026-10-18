"""score_part18c_lens.py: prediction 18c.6, the per-completion logit-lens predictor, computed on CPU from three tensors.

Reads only what the logit lens needs from the cached Qwen3.5-4B safetensors (tied embeddings, layer 22's down_proj, the
final norm's weight) instead of loading the model. Method as neuron_report.py `writes`: unembedding @ (final-norm gain *
down_proj column), z-scored over the first len(tokenizer) vocabulary entries. Qwen3.5's RMSNorm multiplies by
(1 + weight), so the gain is 1 + weight.

Known-answer check before anything is scored: N2742's top token and its z must equal Part 18's writes.json
(' alien', 9.77), or the script stops.

A completion's predictor is the mean z of its tokens with the final period excluded, tokenized on its own (each
completion starts with a space, so this matches its tokens after the stem). ΔΔ_c at G, k=2 comes from the run's rows
and is checked against the run's own per-completion summary.
"""
import json, sys, statistics
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

snap, run, assay, writes, out_path = sys.argv[1:6]
idx = json.load(open(f'{snap}/model.safetensors.index.json'))['weight_map']
def get(name):
    with safe_open(f'{snap}/{idx[name]}', framework='pt') as f:
        return f.get_tensor(name)
pre = 'model.language_model.'
W = get(pre + 'layers.22.mlp.down_proj.weight').float()        # [hidden, F]
gain = 1.0 + get(pre + 'norm.weight').float()
U = get(pre + 'embed_tokens.weight')                            # tied unembedding, bf16 [vocab, hidden]
tok = AutoTokenizer.from_pretrained(snap)
vocab_n = len(tok)

def zscores(u):
    v = (gain * W[:, u]).to(torch.float32)
    logits = torch.cat([U[i:i + 32768].float() @ v for i in range(0, U.shape[0], 32768)])[:vocab_n]
    return (logits - logits.mean()) / logits.std()

z = zscores(2742)
top = int(torch.argmax(z)); ref = json.load(open(writes))['units']['2742']
if tok.decode([top]) != ref['promote'][0] or round(float(z[top]), 2) != ref['promote_z'][0]:
    sys.exit(f'known-answer check failed: {tok.decode([top])!r} {float(z[top]):.2f} vs {ref["promote"][0]!r} {ref["promote_z"][0]}')

d = json.load(open(run, encoding='utf-8'))
cond = 'target:k=2:phase=G'
summ = d['summaries']['self_completion_delta_delta'][cond]
# ΔΔ_c from rows: mean over templates and stems of score under cond minus under unclamped (mean token log-prob).
acc = {}
for r in d['rows']:
    if r['set'] != 'self' or r['condition'] not in (cond, 'unclamped'): continue
    acc.setdefault(r['completion'], {}).setdefault((r['template_index'], r['stem_index']), {})[r['condition']] = float(r['mean_logprob'])
ddc = {c: statistics.fmean(v[cond] - v['unclamped'] for v in pairs.values()) for c, pairs in acc.items()}
for c, v in ddc.items():
    if abs(v - summ[c]) > 1e-9: sys.exit(f'ΔΔ_c differs from the run summary for {c!r}')

pred = {}
for c in ddc:
    ids = tok.encode(c, add_special_tokens=False)
    if tok.decode(ids[-1:]) == '.': ids = ids[:-1]
    pred[c] = {'tokens': [tok.decode([i]) for i in ids], 'mean_z': statistics.fmean(float(z[i]) for i in ids), 'dd_c': ddc[c]}

def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]: j += 1
        for k in range(i, j + 1): r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r
names = sorted(pred)
a, b = ranks([pred[c]['mean_z'] for c in names]), ranks([pred[c]['dd_c'] for c in names])
rho = statistics.correlation(a, b)
res = {'condition': cond, 'n_completions': len(names), 'spearman_rho': rho, 'held': rho >= 0.6,
       'known_answer_check': f"{ref['promote'][0]!r} z={ref['promote_z'][0]} reproduced", 'completions': pred}
json.dump(res, open(out_path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('n', len(names), 'rho', round(rho, 4), 'held', rho >= 0.6)
for c in sorted(names, key=lambda c: -pred[c]['dd_c']):
    print(f"{pred[c]['dd_c']:8.3f} {pred[c]['mean_z']:7.2f} {c!r} {pred[c]['tokens']}")
