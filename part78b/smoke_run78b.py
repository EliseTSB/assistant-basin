#!/usr/bin/env python3
"""Run Part 78b.run end-to-end with a tiny random CPU Llama and real Part 26 records.

This verifies row generation, interventions, resume output, metadata, hooks, and scoring.
It does not test the pinned Qwen model or make a claim about the registered predictions.
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
import neuron_report as nr

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

runner = load('part78b_smoke_runner', ROOT + '/part78b/part78b.py')
scorer = load('part78b_smoke_scorer', ROOT + '/part78b/score78b.py')
torch.manual_seed(7)
config = LlamaConfig(vocab_size=19, hidden_size=16, intermediate_size=9216, num_hidden_layers=30,
    num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
    bos_token_id=1, eos_token_id=2, pad_token_id=0)
model = LlamaForCausalLM(config).eval()
model.requires_grad_(False)
with torch.no_grad():
    model.lm_head.weight.mul_(4.0)
class Tok:
    eos_token_id = 2
    pad_token_id = 0
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [1] + [3 + ord(char) % 15 for char in text[:12]]}
    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(str(int(token)) for token in ids)
tokenizer = Tok()
nr.load_model = lambda model_id, revision, device: (tokenizer, model)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
start = time.time()
runner.run(argparse.Namespace(out=str(out), device='cpu', batch=16, max_gpu_gb=14.0,
                              record_dir=ROOT + '/results/part26'))
print('wall', round(time.time() - start, 1))
rows = [json.loads(line) for line in (out / 'answers.jsonl').read_text().splitlines() if line.strip()]
counts = Counter(row['condition'] for row in rows)
print('rows', len(rows), dict(counts))
run = json.loads((out / 'run.json').read_text())
print('pile max', run['pile_maximum_n2742'])
print('norms and set values', {key: (run['down_proj_column_norms'][key], value)
    for key, value in run['unit_set_values'].items()})
evidence = run['push_hook_batch_evidence']
print('evidence', {condition: (sum(row['calls'] for row in evidence if row['condition'] == condition),
    max(row['max_abs_diff'] for row in evidence if row['condition'] == condition))
    for condition in sorted({row['condition'] for row in evidence})})
print('predictions', scorer.verdicts(scorer.score_rows(rows)))
