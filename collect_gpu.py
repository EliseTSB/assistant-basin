#!/usr/bin/env python3
# collect_gpu.py — collect.py ported to the Arc A770 (XPU, bf16) for the 7B models (Claude fork, 2026-10-16).
# Same corpora, same output format as collect.py, so analyze.py / corr.py read it unchanged. Runs on Windows through
# the NeuronProbeGPU task (WSL cannot see the card). Hooks copy each layer's activations to CPU float32 at once, so
# only the weights stay on the card. --limit N keeps the first N frames (smoke test only).
# Original header (collect.py, Claude, 2026-10-15):
#
# WHAT IT MEASURES: for one model checkpoint, the hidden activation of every MLP neuron (the input to down_proj, i.e.
# silu(gate) * up) at every layer, averaged over the tokens of a self-statement, for each prompt family in
# data/frames.json, in two forms: "plain" (User:/Assistant: text, used for every stage including base) and "chat"
# (the tokenizer's chat template; skipped when the tokenizer has none). Also the per-token activations on a fixed
# stretch of neutral text, for checking that neuron i is the same unit across stages.
#
# Families (24 prompts each, or 72 with data/frames-extra.json; index-paired with the user questions):
#   A  assistant, first person as the AI (OLMo / Ai2)      B  assistant, the same content in the third person
#   C  assistant, a person speaking about themselves        D  the A sentences in the USER turn; assistant says "Okay."
#   E_claude, E_chatgpt  family A with the name and maker swapped
#
# Usage: python collect.py <hf model id> <out dir> [--neutral data/neutral.txt] [--neutral-tokens 1024]
# Output: <out>/<slug>.npz with arrays acts_<form>_<family> [24, L, F] float16 and neutral [T, L*F] float16.
import argparse, json, os, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('out')
ap.add_argument('--neutral', default=os.path.join(os.path.dirname(__file__), 'data', 'neutral.txt'))
ap.add_argument('--neutral-tokens', type=int, default=1024)
ap.add_argument('--revision', default=None, help='a Hugging Face branch, e.g. stage2-ingredient3-step23852-tokens51B')
ap.add_argument('--limit', type=int, default=0, help='smoke test: keep only the first N frames')
ap.add_argument('--device', default='xpu')
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
spec = json.load(open(os.path.join(HERE, 'data', 'frames.json')))
_extra = os.path.join(HERE, 'data', 'frames-extra.json')  # 48 more frames, written by a Haiku helper 10-15
if os.path.exists(_extra):
    _e = json.load(open(_extra)); spec['users'] += _e['users']; spec['frames'] += _e['frames']
if a.limit: spec['users'], spec['frames'] = spec['users'][: a.limit], spec['frames'][: a.limit]
torch.set_grad_enabled(False)

t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, revision=a.revision).to(a.device)
model.eval()
layers = model.model.layers
L, F = len(layers), layers[0].mlp.down_proj.in_features
print(f'{a.model}: {L} layers x {F} neurons on {a.device} (bf16), loaded in {time.time() - t0:.0f}s', flush=True)

captured = [None] * L
def make_hook(i):
    def hook(mod, inp):
        captured[i] = inp[0][0].detach().float().cpu()  # [seq, F], moved off the card at once
    return hook
handles = [layers[i].mlp.down_proj.register_forward_pre_hook(make_hook(i)) for i in range(L)]

def run(text):
    enc = tok(text, return_tensors='pt', return_offsets_mapping=True, add_special_tokens=False)
    offs = enc.pop('offset_mapping')[0].tolist()
    model.model(**{k: v.to(a.device) for k, v in enc.items()})  # trunk only: no 100K-vocab logits on a full card
    return offs, torch.stack(captured, 0)  # [L, seq, F]

def span_mean(text, target):
    start = text.rfind(target)  # the self-statement is the last occurrence in every family
    if start < 0: raise ValueError('target not in text')
    end = start + len(target)
    offs, acts = run(text)
    idx = [k for k, (s, e) in enumerate(offs) if e > start and s < end and e > s]
    if not idx: raise ValueError('no tokens in span')
    return acts[:, idx, :].mean(1).numpy()  # [L, F]

def fill(frame, n, m):
    return frame.replace('{N}', n).replace('{M}', m)

def texts(form, user, assistant):
    if form == 'plain':
        return f'User: {user}\nAssistant: {assistant}'
    msgs = [{'role': 'user', 'content': user}, {'role': 'assistant', 'content': assistant}]
    return tok.apply_chat_template(msgs, tokenize=False)

forms = ['plain'] + (['chat'] if getattr(tok, 'chat_template', None) else [])
out = {}
names = spec['names']
for form in forms:
    fam = {k: [] for k in ['A', 'B', 'C', 'D', 'E_claude', 'E_chatgpt']}
    for i, (fr, u) in enumerate(zip(spec['frames'], spec['users'])):
        on, om = names['olmo']
        a_txt = fill(fr[0], on, om)
        fam['A'].append(span_mean(texts(form, u, a_txt), a_txt))
        b_txt = fill(fr[1], on, om)
        fam['B'].append(span_mean(texts(form, u, b_txt), b_txt))
        fam['C'].append(span_mean(texts(form, u, fr[2]), fr[2]))
        fam['D'].append(span_mean(texts(form, a_txt, 'Okay.'), a_txt))
        for key, nm in [('E_claude', 'claude'), ('E_chatgpt', 'chatgpt')]:
            n2, m2 = names[nm]
            e_txt = fill(fr[0], n2, m2)
            fam[key].append(span_mean(texts(form, u, e_txt), e_txt))
    for k, v in fam.items():
        out[f'acts_{form}_{k}'] = np.stack(v).astype(np.float16)
    print(f'  form {form}: {len(spec["frames"])} prompts x {len(fam)} families, {time.time() - t0:.0f}s', flush=True)

if os.path.exists(a.neutral):
    text = open(a.neutral, encoding='utf-8').read()
    ids = tok(text, add_special_tokens=False)['input_ids'][: a.neutral_tokens]
    model.model(input_ids=torch.tensor([ids], device=a.device))
    acts = torch.stack(captured, 0)[:, : len(ids), :]  # [L, T, F]
    out['neutral'] = acts.permute(1, 0, 2).reshape(len(ids), L * F).numpy().astype(np.float16)
    out['neutral_ids'] = np.array(ids)
    print(f'  neutral: {len(ids)} tokens', flush=True)

for h in handles: h.remove()
os.makedirs(a.out, exist_ok=True)
slug = a.model.replace('/', '__') + (f'@{a.revision}' if a.revision else '')
path = os.path.join(a.out, slug + '.npz')
np.savez(path, **out, layers=L, neurons=F, forms=np.array(forms))
print(f'wrote {path} in {time.time() - t0:.0f}s', flush=True)
