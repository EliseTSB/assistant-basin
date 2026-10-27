#!/usr/bin/env python3
"""Smoke run of part77.run() end to end on a tiny random 30-layer Llama with a stub tokenizer (no GPU, no download).

WHAT IT MEASURES: that the full runner path completes (row plan, paired generate, copy hook, records, run.json,
scorer) and that run.json carries the copy-hook evidence and sampling settings. It does NOT test Qwen, the XPU, or
any result: on a random model 77.1 fails by design (no answer names Google).
FIRST TIME: written by the Part 77 cold reviewer (Haiku, 2026-10-26) in a scratch folder; promoted by Claude. Run:
  ~/neuron-probe/.venv/bin/python -I part77/smoke_run77.py <out_dir>   (about 60 s on CPU)
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
p77 = load('p77_smoke', ROOT + '/part77/part77.py')
s77 = load('s77_smoke', ROOT + '/part77/score77.py')
torch.manual_seed(7)
cfg = LlamaConfig(vocab_size=19, hidden_size=16, intermediate_size=9000, num_hidden_layers=30, num_attention_heads=2,
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
p77.run(argparse.Namespace(out=str(out), device='cpu', batch=16, max_gpu_gb=14.0))
print('run seconds', round(time.time() - t0, 1), 'transformers', transformers.__version__)
print('hooks left on L27 down_proj:', len(model.model.layers[27].mlp.down_proj._forward_pre_hooks),
      '| on L24:', len(model.model.layers[24].mlp.down_proj._forward_pre_hooks))
rows = [json.loads(l) for l in (out / 'answers.jsonl').read_text().splitlines() if l.strip()]
from collections import Counter
print('rows', len(rows), Counter(r['condition'] for r in rows))
print('pair_identical all true:', all(r['pair_identical'] for r in rows), '| kinds', dict(Counter(r['question_kind'] for r in rows)))
print('copy_units by condition:', {c: sorted({str(r['copy_units'])[:40] for r in rows if r['condition'] == c}) for c in ['c0_makers','c1_l24','c2_l24_restore','c3_l24_restore_nulls']})
run = json.loads((out / 'run.json').read_text())
print('run.json:', {k: run[k] for k in ['sampling_settings', 'custom_processor_before_warpers', 'pairs_not_identical', 'new_generations_this_invocation']})
res = s77.score_rows(rows)
for k, v in res['predictions'].items():
    print(k, v['status'])
print({c: (v['heldout_google'], v['heldout_n'], v['guard_keyword_matches'], v['guard_n']) for c, v in res['conditions'].items()})
