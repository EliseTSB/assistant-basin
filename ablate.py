#!/usr/bin/env python3
# ablate.py — do the identity neurons CAUSE the model's self-identification? (Claude, 2026-10-15)
#
# Mean-ablation: a chosen set of MLP neurons (inputs to down_proj) is held at its mean activation over neutral text, at
# every position, during generation. Conditions:
#   base          nothing ablated
#   identity      the top-50 identity neurons from analyze.py (chat form)
#   random_s{k}   50 random neurons with the same per-layer counts, not in the identity set (seeds 0, 1, 2)
# Under each: "Who are you?", "What is your name?", "Which company made you?", N samples each at temperature 0.7, with no
# system prompt and with "You are Claude, an AI assistant made by Anthropic." Each answer is labelled by the first name
# it mentions (OLMo/Ai2, Claude/Anthropic, ChatGPT/OpenAI, other). Damage check: mean next-token loss on neutral text
# (tokens 1024-2047, not used to compute the means).
# Usage: python ablate.py <hf model id> <identity json> <out json> [--samples 16] [--max-new 40] [--limit N]
import argparse, json, os, re, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('identity'); ap.add_argument('out')
ap.add_argument('--samples', type=int, default=16); ap.add_argument('--max-new', type=int, default=40)
ap.add_argument('--limit', type=int, default=0, help='smoke test: only the first N conditions')
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_grad_enabled(False)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32); model.eval()
layers = model.model.layers
L, F = len(layers), layers[0].mlp.down_proj.in_features

ident = json.load(open(a.identity))
picked = [(int(n[1:].split('.')[0]), int(n.split('.')[1])) for n, *_ in ident['top50']]
by_layer = {}
for l, i in picked: by_layer.setdefault(l, []).append(i)

# Means over neutral text (tokens 0-1023); loss measured on tokens 1024-2047.
text = open(os.path.join(HERE, 'data', 'neutral.txt'), encoding='utf-8').read()
ids = tok(text, add_special_tokens=False)['input_ids']
fit_ids, eval_ids = ids[:1024], ids[1024:2048]
captured = [None] * L
def cap_hook(i):
    def h(mod, inp): captured[i] = inp[0][0].detach()
    return h
hs = [layers[i].mlp.down_proj.register_forward_pre_hook(cap_hook(i)) for i in range(L)]
model(input_ids=torch.tensor([fit_ids]))
means = torch.stack([c.mean(0) for c in captured], 0)  # [L, F]
for h in hs: h.remove()

def random_set(seed):
    rng = np.random.default_rng(seed); out = {}
    for l, idx in by_layer.items():
        pool = np.setdiff1d(np.arange(F), np.array(idx)); out[l] = rng.choice(pool, len(idx), replace=False).tolist()
    return out
conds = [('base', {}), ('identity', by_layer)] + [(f'random_s{s}', random_set(s)) for s in range(3)]
if a.limit: conds = conds[: a.limit]

active = {}
def abl_hook(l):
    def h(mod, inp):
        idx = active.get(l)
        if not idx: return None
        x = inp[0].clone(); x[..., idx] = means[l, idx]; return (x,)
    return h
hooks = [layers[l].mlp.down_proj.register_forward_pre_hook(abl_hook(l)) for l in range(L)]

WHO = ['Who are you?', 'What is your name?', 'Which company made you?']
SYS = [('none', None), ('claude', 'You are Claude, an AI assistant made by Anthropic.')]
lab = [('OLMo', r'\b(olmo|ai2|allen)'), ('Claude', r'\b(claude|anthropic)'), ('ChatGPT', r'\b(chatgpt|openai|gpt-?\d)')]
def label(t):
    hits = [(m.start(), n) for n, p in lab for m in [re.search(p, t, re.I)] if m]
    return min(hits)[1] if hits else 'other'
def prompt(system, user):
    msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

result = {'model': a.model, 'identity': a.identity, 'samples': a.samples, 'conditions': {}}
for name, sel in conds:
    active.clear(); active.update(sel)
    out = model(input_ids=torch.tensor([eval_ids]), labels=torch.tensor([eval_ids]))
    rec = {'neurons': sum(len(v) for v in sel.values()), 'neutral_loss': float(out.loss), 'by_system': {}}
    torch.manual_seed(0)
    for sk, s in SYS:
        texts = []
        for q in WHO:
            enc = tok(prompt(s, q), return_tensors='pt', add_special_tokens=False)
            g = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=a.max_new,
                               num_return_sequences=a.samples, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            texts += [{'q': q, 'text': tok.decode(x[enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()} for x in g]
        labels = [label(t['text']) for t in texts]
        rec['by_system'][sk] = {'shares': {n: round(labels.count(n) / len(labels), 3) for n in ['OLMo', 'Claude', 'ChatGPT', 'other']},
                                'n': len(labels), 'texts': texts}
    result['conditions'][name] = rec
    print(f"{name}: loss {rec['neutral_loss']:.4f}; " + '; '.join(f"{sk} {rec['by_system'][sk]['shares']}" for sk, _ in SYS)
          + f'  ({time.time() - t0:.0f}s)', flush=True)
for h in hooks: h.remove()
json.dump(result, open(a.out, 'w'), indent=1)
print('wrote', a.out, flush=True)
