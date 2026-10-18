#!/usr/bin/env python3
# ablate_smol.py — Part 11.4 and 11.5 on SmolLM3-3B (Claude, 2026-10-16). Scoring rules: assistant-basin PREDICTIONS.md
# (commit 8f2f233), written before this ran.
#
# 11.4: zero the top-20 chat-form identity neurons (inputs to down_proj, every position) against 20 random neurons with
# the same per-layer counts (seeds 0-2), and label each answer to "What is your name?" by the first name it mentions.
# Three prompt forms: raw_nosys (built by hand, no system block, thinking off; SCORED), shipped (the template's own
# inserted "You are a helpful AI assistant named SmolLM" line), claude (system "You are Claude, ...").
# Damage check: next-token loss on neutral text (data/neutral.txt tokens 1024-2047).
# 11.5: gradient x activation of every MLP neuron for the " Sm" logit after the raw_nosys prompt + "My name is".
# ⚠ SmolLM3's template INSERTS its identity line when no system message is given (checked 2026-10-16 before writing),
# which is why raw_nosys exists; a run that only used the template would test repeating a name from context.
# Usage: python ablate_smol.py <identity json> <out json> [--samples 48] [--max-new 32] [--limit N] [--skip-ablation]
import argparse, json, os, re, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('identity'); ap.add_argument('out')
ap.add_argument('--model', default='HuggingFaceTB/SmolLM3-3B')
ap.add_argument('--samples', type=int, default=48); ap.add_argument('--max-new', type=int, default=32)
ap.add_argument('--top', type=int, default=20)
ap.add_argument('--limit', type=int, default=0, help='smoke test: only the first N conditions')
ap.add_argument('--skip-ablation', action='store_true')
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16); model.eval()
layers = model.model.layers
L, F = len(layers), layers[0].mlp.down_proj.in_features

ident = json.load(open(a.identity))
picked = [(int(n[1:].split('.')[0]), int(n.split('.')[1])) for n, *_ in ident['top50'][: a.top]]
by_layer = {}
for l, i in picked: by_layer.setdefault(l, []).append(i)

def random_set(seed):
    rng = np.random.default_rng(seed); out = {}
    for l, idx in by_layer.items():
        pool = np.setdiff1d(np.arange(F), np.array(idx)); out[l] = rng.choice(pool, len(idx), replace=False).tolist()
    return out

active = {}
def abl_hook(l):
    def h(mod, inp):
        idx = active.get(l)
        if not idx: return None
        x = inp[0].clone(); x[..., idx] = 0; return (x,)
    return h
hooks = [layers[l].mlp.down_proj.register_forward_pre_hook(abl_hook(l)) for l in range(L)]

Q = 'What is your name?'
def prompt(form):
    if form == 'raw_nosys':
        return f'<|im_start|>user\n{Q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n'
    msgs = ([{'role': 'system', 'content': 'You are Claude, an AI assistant made by Anthropic.'}] if form == 'claude' else []) \
        + [{'role': 'user', 'content': Q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
FORMS = ['raw_nosys', 'shipped', 'claude']
lab = [('SmolLM', r'(smol|hugging ?face)'), ('Claude', r'\b(claude|anthropic)'), ('ChatGPT', r'\b(chatgpt|openai|gpt-?\d)'),
       ('Qwen', r'\b(qwen|alibaba)'), ('Llama', r'\b(llama|meta ai)')]
NAMES = [n for n, _ in lab] + ['other']
def label(t):
    hits = [(m.start(), n) for n, p in lab for m in [re.search(p, t, re.I)] if m]
    return min(hits)[1] if hits else 'other'

result = {'model': a.model, 'identity': a.identity, 'top': a.top, 'samples': a.samples, 'question': Q,
          'prompts': {f: prompt(f) for f in FORMS}, 'conditions': {}}
if not a.skip_ablation:
    text = open(os.path.join(HERE, 'data', 'neutral.txt'), encoding='utf-8').read()
    eval_ids = tok(text, add_special_tokens=False)['input_ids'][1024:2048]
    conds = [('base', {}), ('identity', by_layer)] + [(f'random_s{s}', random_set(s)) for s in range(3)]
    if a.limit: conds = conds[: a.limit]
    with torch.no_grad():
        for name, sel in conds:
            active.clear(); active.update(sel)
            out = model(input_ids=torch.tensor([eval_ids]), labels=torch.tensor([eval_ids]))
            rec = {'neurons': {str(k): v for k, v in sel.items()}, 'neutral_loss': float(out.loss), 'by_form': {}}
            for f in FORMS:
                torch.manual_seed(0)
                enc = tok(prompt(f), return_tensors='pt', add_special_tokens=False)
                g = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=a.max_new,
                                   num_return_sequences=a.samples, pad_token_id=tok.pad_token_id or tok.eos_token_id)
                texts = [tok.decode(x[enc['input_ids'].shape[1]:], skip_special_tokens=True).strip() for x in g]
                labels = [label(t) for t in texts]
                rec['by_form'][f] = {'shares': {n: round(labels.count(n) / len(labels), 3) for n in NAMES}, 'texts': texts}
            result['conditions'][name] = rec
            print(f"{name}: loss {rec['neutral_loss']:.4f}; " + '; '.join(f"{f} SmolLM {rec['by_form'][f]['shares']['SmolLM']}" for f in FORMS)
                  + f'  ({time.time() - t0:.0f}s)', flush=True)
            json.dump(result, open(a.out, 'w'), indent=1)
active.clear()

# 11.5: gradient x activation for the " Sm" logit.
SM = tok.encode(' Sm', add_special_tokens=False); assert len(SM) == 1, SM
acts = [None] * L
def grab(l):
    def h(mod, inp):
        x = inp[0]; x.retain_grad(); acts[l] = x
    return h
gh = [layers[l].mlp.down_proj.register_forward_pre_hook(grab(l)) for l in range(L)]
for p in model.parameters(): p.requires_grad_(False)
ids = tok(prompt('raw_nosys') + 'My name is', return_tensors='pt', add_special_tokens=False)['input_ids']
emb = model.model.embed_tokens(ids).detach().requires_grad_(True)
logits = model(inputs_embeds=emb).logits[0, -1].float()
logp = torch.log_softmax(logits, -1)
logits[SM[0]].backward()
attr = torch.stack([(acts[l].grad[0].float() * acts[l][0].detach().float()).sum(0) for l in range(L)])  # [L, F]
for h in gh: h.remove()
flat = attr.abs().flatten(); top = torch.topk(flat, 50).indices.tolist()
top_units = [(i // F, i % F, round(float(attr.flatten()[i]), 5)) for i in top]
ident50 = {(int(n[1:].split('.')[0]), int(n.split('.')[1])) for n, *_ in ident['top50']}
last_third = sum(1 for l, _, _ in top_units if l >= L - L // 3)
result['attribution'] = {'token': ' Sm', 'token_id': SM[0], 'p_sm': round(float(logp[SM[0]].exp()), 4),
                         'top50': [f'L{l}.{n}' for l, n, _ in top_units], 'top50_values': [v for *_, v in top_units],
                         'last_third_layers_from': L - L // 3, 'top50_in_last_third': last_third,
                         'overlap_with_identity_top50': sum(1 for l, n, _ in top_units if (l, n) in ident50),
                         'layer_share_abs': [round(float(x), 4) for x in (attr.abs().sum(1) / attr.abs().sum())]}
print(f"11.5: p(' Sm')={result['attribution']['p_sm']}; top50 in last third {last_third}/50; overlap with identity top50 "
      f"{result['attribution']['overlap_with_identity_top50']}  ({time.time() - t0:.0f}s)", flush=True)
for h in hooks: h.remove()
json.dump(result, open(a.out, 'w'), indent=1)
print('wrote', a.out, flush=True)
