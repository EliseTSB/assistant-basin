# clef_systemone_gpu.py — Part 1f (Claude, 2026-10-17; predictions 0e761ae pushed before any run). Answers recorded
# Jev/SystemOne requests with local Cloudflare Clef-flash in 4-bit on the A770, using the release's OWN systemone()
# helper so each noul probability is the release's p("true") (the batch path in clef_batch_gpu.py guessed which option
# index means true; a wrong guess inverts every answer). Usage, Windows venv:
#   python clef_systemone_gpu.py --snapshot <dir> --requests requests.jsonl --answers answers.jsonl [--limit N]
# Guards: refuses to continue if the loaded model takes more than --max-gb of XPU memory (the A770 spills into shared
# memory instead of failing, which froze WSL on 2026-10-16); checks direction on two known-answer requests first.
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument('--snapshot', required=True); ap.add_argument('--requests', required=True)
ap.add_argument('--answers', required=True); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--max-gb', type=float, default=12.0)
a = ap.parse_args()
import torch
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *x, **k: None
from transformers import BitsAndBytesConfig
sys.path.insert(0, a.snapshot)
import joint_schema_model as J
t0 = time.time()
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
model, processor = J.load_release_model(a.snapshot, device='xpu', dtype=torch.bfloat16, quantization_config=quant)
gb = torch.xpu.memory_allocated() / 2**30
print(f'loaded in {time.time()-t0:.0f}s, xpu allocated {gb:.2f} GB', flush=True)
if gb > a.max_gb:
    sys.exit(f'refusing: {gb:.2f} GB allocated exceeds --max-gb {a.max_gb}')
def ask(state):
    q = {'fires': {'type': 'noul', 'instructions': 'Judging only by the description in the state, would this unit be strongly active on the marked token?'}}
    return J.systemone(model, processor, {'model': 'clef-flash', 'state': state, 'questions': q})['answers']['fires']['noul']
yes = ask({'description': 'fires on the word "Paris"', 'snippet': 'We flew to ⟦Paris⟧ last spring.'})
no = ask({'description': 'fires on the word "Paris"', 'snippet': 'The ⟦cat⟧ slept on the warm windowsill.'})
print(f'direction check: Paris {yes:.3f}, cat {no:.3f}', flush=True)
if not yes > no:
    sys.exit('refusing: direction check failed (the known-yes request did not score above the known-no one)')
done = set()
if os.path.exists(a.answers):
    for line in open(a.answers, encoding='utf-8'):
        if line.strip(): done.add(json.loads(line)['hash'])
todo = []
for line in open(a.requests, encoding='utf-8'):
    if not line.strip(): continue
    r = json.loads(line); h = r['hash']
    if h in done: continue
    done.add(h); todo.append(r)
if a.limit: todo = todo[:a.limit]
print(f'{len(todo)} requests to answer', flush=True)
t1 = time.time(); n_tok = 0
with open(a.answers, 'a', encoding='utf-8') as out:
    for i, r in enumerate(todo, 1):
        res = J.systemone(model, processor, {'model': 'clef-flash', 'state': r['state'], 'questions': r['questions']})
        n_tok += res['usage']['input_tokens']
        out.write(json.dumps({'hash': r['hash'], 'answers': {k: v['noul'] for k, v in res['answers'].items() if v['type'] == 'noul'},
                              'input_tokens': res['usage']['input_tokens']}) + '\n')
        if i % 100 == 0 or i == len(todo):
            out.flush(); torch.xpu.empty_cache()
            el = time.time() - t1
            print(f'{i}/{len(todo)} {i/el:.2f} req/s {n_tok/el:.0f} tok/s, xpu {torch.xpu.memory_allocated()/2**30:.2f} GB', flush=True)
print(f'done: {len(todo)} in {time.time()-t1:.0f}s', flush=True)
