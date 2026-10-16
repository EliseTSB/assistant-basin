# sysgen9_gpu.py: Part 9 copy of sysgen8_gpu.py (Claude, 2026-10-16), for the OLMo 3 7B Think chain.
# Think models open the assistant turn with "<think>" and reason before answering, so each sample is labelled on its
# FINAL answer (the text after the last "</think>"); a sample that never closes its thinking within --max-new is
# counted as 'unfinished' and the name shares are over finished samples. raw_nosys gets the same "<think>" prefix the
# template adds. The Think templates insert "You are OLMo/Olmo, ... built by Ai2" when no system message is given,
# so 'none' means that prompt and raw_nosys is the no-prompt condition.
import argparse, json, re, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
CONDS = [('none', None), ('hhh', 'You are a helpful, harmless, and honest assistant.'),
         ('claude', 'You are Claude, an AI assistant made by Anthropic.'),
         ('chatgpt', 'You are ChatGPT, an AI assistant made by OpenAI.')]
WHO = ['Who are you?', 'What is your name?', 'Which company made you?']
lab = [('Qwen', r'\b(qwen|alibaba|tongyi)'), ('OpenHands', r'\b(open ?hands|all ?hands)\b'), ('OLMo', r'\b(olmo|ai2|allen)'),
       ('Claude', r'\b(claude|anthropic)'), ('ChatGPT', r'\b(chatgpt|openai|gpt-?\d)')]
def label(t):
    hits = [(m.start(), n) for n, p in lab for m in [re.search(p, t, re.I)] if m]
    return min(hits)[1] if hits else 'other'
ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('out')
ap.add_argument('--samples', type=int, default=8); ap.add_argument('--max-new', type=int, default=1024)
ap.add_argument('--limit-conds', type=int, default=0); ap.add_argument('--device', default='xpu')
ap.add_argument('--raw-nosys', action='store_true', help="also 'raw_nosys': a ChatML prompt with NO system turn at all, bypassing a template that inserts a default system prompt (OLMo 3 does)")
a = ap.parse_args(); torch.set_grad_enabled(False); t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).eval()
THINK = '<think>' if tok.apply_chat_template([{'role': 'user', 'content': 'x'}], tokenize=False, add_generation_prompt=True).rstrip().endswith('<think>') else ''
def prompt(system, user):
    if system == '__RAW__': return f'<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n' + THINK
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
    fins = [t.rsplit('</think>', 1)[1] if '</think>' in t else None for g in gens[k].values() for t in g] if THINK else [t for g in gens[k].values() for t in g]
    ls = [label(t) for t in fins if t is not None]; nu = sum(t is None for t in fins)
    S6[k] = {n: (round(ls.count(n) / len(ls), 3) if ls else None) for n in ['Qwen', 'OpenHands', 'OLMo', 'Claude', 'ChatGPT', 'other']} | {'n': len(ls), 'unfinished': nu}
    print(f'  {k}: {S6[k]}  ({time.time() - t0:.0f}s)', flush=True)
json.dump({'model': a.model, 'samples': a.samples, 'max_new': a.max_new, 'S6': S6, 'generations': gens}, open(a.out, 'w'), indent=1)
print('wrote', a.out, flush=True)
