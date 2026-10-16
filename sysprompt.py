#!/usr/bin/env python3
# sysprompt.py — what system prompts shift inside the model (Claude, 2026-10-15, for Elise's HHH question).
#
# WHAT IT MEASURES, for one chat model:
#  1. STATES: for each system prompt in CONDS x each user message (the 72 frame questions plus 24 ordinary tasks), the
#     residual stream at every layer (hidden_states, L+1 x d) and every MLP neuron (input to down_proj, L x F) at the
#     LAST prompt token, i.e. where the reply starts (chat template with add_generation_prompt).
#  2. BEHAVIOUR: three "who are you" questions under each system prompt, N samples each at temperature 0.7.
#  3. AFTER-NAME TOKENS: for the frames whose sentence names the model, the MLP and residual means over the tokens AFTER
#     the last name/maker occurrence, for OLMo, Claude and ChatGPT (chat form, no system prompt). This removes the name's
#     own tokens from the name test.
# Usage: python sysprompt.py <hf model id> <out dir> [--samples 8] [--max-new 48] [--limit-users N] [--limit-conds N]
# Output: <out>/sysprompt-<slug>.npz (+ .json with the generations and the index of every array).
import argparse, json, os, re, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CONDS = [  # (key, system prompt or None)
    ('none', None),
    ('assistant', 'You are an assistant.'),
    ('helpful', 'You are a helpful assistant.'),
    ('harmless', 'You are a harmless assistant.'),
    ('honest', 'You are an honest assistant.'),
    ('hhh', 'You are a helpful, harmless, and honest assistant.'),
    ('curious', 'You are a curious assistant.'),
    ('tall', 'You are a tall assistant.'),
    ('person', 'You are a person.'),
    ('person_hhh', 'You are a helpful, harmless, and honest person.'),
    ('pirate', 'You are a pirate.'),
    ('pirate_hhh', 'You are a helpful, harmless, and honest pirate.'),
    ('claude', 'You are Claude, an AI assistant made by Anthropic.'),
    ('olmo', 'You are OLMo, an AI assistant made by Ai2.'),
    ('chatgpt', 'You are ChatGPT, an AI assistant made by OpenAI.'),
]
TASKS = [
    'Write a haiku about autumn.', 'What is the capital of Australia?', 'Explain photosynthesis in two sentences.',
    'Give me three tips for sleeping better.', "Translate 'good morning' into Spanish.", 'How many minutes are in a day?',
    'Suggest a name for a black cat.', 'Summarize the plot of Romeo and Juliet in one sentence.',
    "What's a good way to store fresh basil?", 'Write a short apology for being late to a meeting.',
    'Convert 30 degrees Celsius to Fahrenheit.', 'Why is the sky blue?', 'List four prime numbers.',
    'Recommend a board game for two players.', 'What does a router do?', 'Write one sentence describing the ocean at night.',
    'How do I boil an egg?', 'What year did the first moon landing happen?', "Give me a synonym for 'happy'.",
    'Explain what a variable is in programming.', 'What should I pack for a day hike?', 'Write a two-line poem about rain.',
    'How far is the Moon from Earth?', 'What is 17 times 23?',
]
WHO = ['Who are you?', 'What is your name?', 'Which company made you?']

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('out')
ap.add_argument('--samples', type=int, default=8); ap.add_argument('--max-new', type=int, default=48)
ap.add_argument('--limit-users', type=int, default=0); ap.add_argument('--limit-conds', type=int, default=0)
ap.add_argument('--skip-gen', action='store_true')
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
spec = json.load(open(os.path.join(HERE, 'data', 'frames.json')))
_e = json.load(open(os.path.join(HERE, 'data', 'frames-extra.json')))
frames, fusers = spec['frames'] + _e['frames'], spec['users'] + _e['users']
users = fusers + TASKS
conds = CONDS[: a.limit_conds] if a.limit_conds else CONDS
if a.limit_users: users = users[: a.limit_users]
torch.set_grad_enabled(False); torch.manual_seed(0)

t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32); model.eval()
layers = model.model.layers
L, F, D = len(layers), layers[0].mlp.down_proj.in_features, model.config.hidden_size
print(f'{a.model}: {L} layers, {F} neurons, d={D}; loaded in {time.time() - t0:.0f}s', flush=True)
captured = [None] * L
def make_hook(i):
    def hook(mod, inp): captured[i] = inp[0][0].detach()
    return hook
handles = [layers[i].mlp.down_proj.register_forward_pre_hook(make_hook(i)) for i in range(L)]

def prompt(system, user):
    msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def last_states(text):
    enc = tok(text, return_tensors='pt', add_special_tokens=False)
    o = model(**enc, output_hidden_states=True)
    resid = torch.stack([h[0, -1] for h in o.hidden_states], 0)  # [L+1, D]
    mlp = torch.stack([c[-1] for c in captured], 0)               # [L, F]
    return resid.numpy().astype(np.float16), mlp.numpy().astype(np.float16)

out, meta = {}, {'model': a.model, 'conds': [[k, s] for k, s in conds], 'users': users, 'n_frame_users': len(fusers)}
for k, s in conds:
    R, M = zip(*[last_states(prompt(s, u)) for u in users])
    out[f'resid_{k}'], out[f'mlp_{k}'] = np.stack(R), np.stack(M)
    print(f'  states {k}: {len(users)} users, {time.time() - t0:.0f}s', flush=True)

# After-name tokens: OLMo vs Claude vs ChatGPT, chat form, the same sentences as collect.py's A / E families.
names = spec['names']
def fill(fr, n, m): return fr.replace('{N}', n).replace('{M}', m)
def after_name(user, sent, n, m):
    text = tok.apply_chat_template([{'role': 'user', 'content': user}, {'role': 'assistant', 'content': sent}], tokenize=False)
    s0 = text.rfind(sent); cut = s0 + max(sent.rfind(n) + len(n) if n in sent else -1, sent.rfind(m) + len(m) if m in sent else -1)
    end = s0 + len(sent)
    enc = tok(text, return_tensors='pt', return_offsets_mapping=True, add_special_tokens=False)
    offs = enc.pop('offset_mapping')[0].tolist()
    o = model(**enc, output_hidden_states=True)
    idx = [j for j, (s, e) in enumerate(offs) if s >= cut and e <= end and e > s]
    if len(idx) < 3: return None
    resid = torch.stack([h[0, idx].mean(0) for h in o.hidden_states], 0)
    mlp = torch.stack([c[idx].mean(0) for c in captured], 0)
    return resid.numpy().astype(np.float16), mlp.numpy().astype(np.float16), len(idx)
kept = []
an = {key: ([], []) for key in ('olmo', 'claude', 'chatgpt')}
for i, (fr, u) in enumerate(zip(frames, fusers)):
    if '{N}' not in fr[0] and '{M}' not in fr[0]: continue
    rows = {key: after_name(u, fill(fr[0], *names[key]), *names[key]) for key in an}
    if any(v is None for v in rows.values()): continue
    kept.append(i)
    for key, (r, m, _) in rows.items(): an[key][0].append(r); an[key][1].append(m)
for key, (r, m) in an.items():
    if kept: out[f'aftername_resid_{key}'], out[f'aftername_mlp_{key}'] = np.stack(r), np.stack(m)
meta['aftername_frames'] = kept
print(f'  after-name: {len(kept)} named frames with 3+ tokens after the name, {time.time() - t0:.0f}s', flush=True)

gens = {}
if not a.skip_gen:
    for k, s in conds:
        gens[k] = {}
        for q in WHO:
            enc = tok(prompt(s, q), return_tensors='pt', add_special_tokens=False)
            g = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=a.max_new,
                               num_return_sequences=a.samples, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gens[k][q] = [tok.decode(x[enc['input_ids'].shape[1]:], skip_special_tokens=True).strip() for x in g]
        print(f'  generations {k}: {time.time() - t0:.0f}s', flush=True)
meta['generations'] = gens

for h in handles: h.remove()
os.makedirs(a.out, exist_ok=True)
slug = a.model.replace('/', '__')
np.savez(os.path.join(a.out, f'sysprompt-{slug}.npz'), **out, layers=L, neurons=F, d=D)
json.dump(meta, open(os.path.join(a.out, f'sysprompt-{slug}.json'), 'w'), indent=1)
print(f'wrote {a.out}/sysprompt-{slug}.npz in {time.time() - t0:.0f}s', flush=True)
