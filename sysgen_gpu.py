#!/usr/bin/env python3
# sysgen_gpu.py — the BEHAVIOUR half of sysprompt.py on the Arc A770 (XPU, bf16), for prediction 5.7 (Claude fork,
# 2026-10-16). Same 15 system prompts (CONDS), the same three "who are you" questions, 8 samples each at temperature
# 0.7 / top_p 0.95, 48 new tokens, and the same first-name labelling as sysanalyze.py (S6). No activations.
# Usage: python sysgen_gpu.py <hf model id> <out json> [--samples 8] [--max-new 48] [--limit-conds N]
import argparse, json, re, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
CONDS = [('none', None), ('assistant', 'You are an assistant.'), ('helpful', 'You are a helpful assistant.'),
         ('harmless', 'You are a harmless assistant.'), ('honest', 'You are an honest assistant.'),
         ('hhh', 'You are a helpful, harmless, and honest assistant.'), ('curious', 'You are a curious assistant.'),
         ('tall', 'You are a tall assistant.'), ('person', 'You are a person.'),
         ('person_hhh', 'You are a helpful, harmless, and honest person.'), ('pirate', 'You are a pirate.'),
         ('pirate_hhh', 'You are a helpful, harmless, and honest pirate.'),
         ('claude', 'You are Claude, an AI assistant made by Anthropic.'), ('olmo', 'You are OLMo, an AI assistant made by Ai2.'),
         ('chatgpt', 'You are ChatGPT, an AI assistant made by OpenAI.')]
WHO = ['Who are you?', 'What is your name?', 'Which company made you?']
lab = [('OLMo', r'\b(olmo|ai2|allen)'), ('Claude', r'\b(claude|anthropic)'), ('ChatGPT', r'\b(chatgpt|openai|gpt-?\d)')]
def label(t):
    hits = [(m.start(), n) for n, p in lab for m in [re.search(p, t, re.I)] if m]
    return min(hits)[1] if hits else 'other'
ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('out')
ap.add_argument('--samples', type=int, default=8); ap.add_argument('--max-new', type=int, default=48)
ap.add_argument('--limit-conds', type=int, default=0); ap.add_argument('--device', default='xpu')
ap.add_argument('--raw-nosys', action='store_true', help="also 'raw_nosys': a ChatML prompt with NO system turn at all, bypassing a template that inserts a default system prompt (OLMo 3 does)")
a = ap.parse_args(); torch.set_grad_enabled(False); t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).eval()
def prompt(system, user):
    if system == '__RAW__': return f'<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n'
    msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
torch.manual_seed(0)
gens, S6 = {}, {}
conds = (CONDS[: a.limit_conds] if a.limit_conds else CONDS) + ([('raw_nosys', '__RAW__')] if a.raw_nosys else [])
for k, s in conds:
    gens[k] = {}
    for q in WHO:
        enc = {kk: v.to(a.device) for kk, v in tok(prompt(s, q), return_tensors='pt', add_special_tokens=False).items()}
        g = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=a.max_new,
                           num_return_sequences=a.samples, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gens[k][q] = [tok.decode(x[enc['input_ids'].shape[1]:], skip_special_tokens=True).strip() for x in g]
    ls = [label(t) for g in gens[k].values() for t in g]
    S6[k] = {n: round(ls.count(n) / len(ls), 3) for n in ['OLMo', 'Claude', 'ChatGPT', 'other']} | {'n': len(ls)}
    print(f'  {k}: {S6[k]}  ({time.time() - t0:.0f}s)', flush=True)
json.dump({'model': a.model, 'samples': a.samples, 'max_new': a.max_new, 'S6': S6, 'generations': gens}, open(a.out, 'w'), indent=1)
print('wrote', a.out, flush=True)
