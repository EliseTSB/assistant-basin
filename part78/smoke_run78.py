#!/usr/bin/env python3
"""Smoke run of part78.run() end to end on a tiny random 30-layer Llama with a stub tokenizer (no GPU, no download).

WHAT IT MEASURES: that the full runner path completes (row plan, paired generate, copy hook, records, run.json,
scorer) and that run.json carries the copy-hook evidence and sampling settings. It does NOT test Qwen, the XPU, or
any result: on a random model 78.1 fails by design (no answer is off-world or names Google).
Adapted for Part 78 by Claude from part77/smoke_run77.py (first written by the Part 77 cold reviewer (Haiku, 2026-10-26) in a scratch folder; promoted by Claude. Run:
  ~/neuron-probe/.venv/bin/python -I part78/smoke_run78.py <out_dir>   (about 140 s on CPU)
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path
import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM
ROOT = str(__import__('pathlib').Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
import neuron_report as nr
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
p78 = load('p78_smoke', ROOT + '/part78/part78.py')
s78 = load('s78_smoke', ROOT + '/part78/score78.py')
torch.manual_seed(7)
cfg = LlamaConfig(vocab_size=19, hidden_size=16, intermediate_size=9216, num_hidden_layers=30, num_attention_heads=2,
                  num_key_value_heads=2, max_position_embeddings=256, bos_token_id=1, eos_token_id=2, pad_token_id=0)
model = LlamaForCausalLM(cfg).eval()
with torch.no_grad():
    model.lm_head.weight.mul_(4.0)
class Tok:
    eos_token_id = 2
    pad_token_id = 0
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [1] + [3 + (ord(c) % 15) for c in text[:12]]}
    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(str(int(i)) for i in ids)
tok = Tok()
nr.load_model = lambda model_id, revision, device: (tok, model)
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
p78.run(argparse.Namespace(out=str(out), device='cpu', batch=16, max_gpu_gb=14.0, record_dir=ROOT + '/results/part26'))
print('wall', round(time.time() - t0, 1))
rows = [json.loads(l) for l in (out / 'answers.jsonl').read_text().splitlines() if l.strip()]
from collections import Counter
print('rows', len(rows), Counter(r['condition'] for r in rows))
run = json.loads((out / 'run.json').read_text())
print('pile max', run['pile_maximum_n2742'], 'norms', run['down_proj_column_norms']['N2742'], run['down_proj_column_norms']['N8870'])
ev = run['push_hook_batch_evidence']
print('evidence', {c: (sum(e['calls'] for e in ev if e['condition']==c), max(e['max_abs_diff'] for e in ev if e['condition']==c)) for c in sorted({e['condition'] for e in ev})})
print('conditions', [(c['id'], c['push']) for c in run['conditions']])
res = s78.score_rows(rows); print(s78.verdicts(res))
