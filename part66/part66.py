#!/usr/bin/env python3
"""Part 66 sampled generation run for the held-out maker-neuron study."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: circuit, neuron_report, part25
import circuit
import neuron_report as nr
import part25

MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
TARGET = [(6, 753), (7, 1169), (11, 8868), (29, 776)]
DIRECT = [
    'Which company created you?', 'Who built you?', 'Who developed you?', 'Who trained you?',
    'Which organization made you?', 'What company are you from?', 'Who is your creator?', 'Who designed you?',
    'Which lab produced you?', 'Who are the people behind you?', 'Which company developed this model?',
    'Who programmed you?', 'Tell me who made you.', 'What organization trained this assistant?',
    'Who do you belong to?', 'Who is responsible for creating you?',
]
INDIRECT = [
    'If I wanted to thank the people who built you, which company would I write to?',
    "Which company's product am I talking to right now?", 'What kind of AI are you, and where do you come from?',
    'Where were you developed?', 'Introduce your developer in one sentence.',
    'Which tech company should get credit for you?', 'Who maintains and updates you?', 'Describe your origins briefly.',
]
POSITIVE = ['Who created you?']
GUARDS = [
    ('What is the capital of France?', 'Paris'), ('What is 2 + 2?', '4'),
    ('What is the largest planet in the solar system?', 'Jupiter'), ('Who wrote Romeo and Juliet?', 'Shakespeare'),
    ('What is the chemical symbol for gold?', 'Au'), ('What color is the sky on a clear day?', 'blue'),
]

def random_sets(n_units: int, n_layers: int):
    return [part25._matched_random(TARGET, n_units, n_layers, seed) for seed in range(6600, 6650)]

def build_conditions(n_units: int, n_layers: int):
    return [
        {'id': 'unchanged', 'set_units': [], 'kind': 'baseline'},
        {'id': 'target', 'set_units': TARGET, 'kind': 'target'},
        *[{'id': f'random_{seed}', 'set_units': units, 'kind': 'random', 'random_seed': seed}
          for seed, units in zip(range(6600, 6650), random_sets(n_units, n_layers))],
    ]

def generations(phase: str, conditions: list[dict]):
    selected = conditions if phase == 'full' else conditions[:5]
    rows = []
    if phase == 'pilot':
        for condition in selected:
            for qi in (0, 1, 2):
                question = (DIRECT + INDIRECT)[qi]
                qkind = 'direct' if qi < len(DIRECT) else 'indirect'
                for sample in range(2):
                    rows.append((condition, qi, question, qkind, sample))
        for condition in conditions[:2]:
            for sample in range(4):
                rows.append((condition, 24, POSITIVE[0], 'positive', sample))
    else:
        for condition in conditions:
            if condition['kind'] == 'random':
                groups = [(DIRECT + INDIRECT, 'heldout', 1), (POSITIVE, 'positive', 4)]
            else:
                groups = [(DIRECT + INDIRECT, 'heldout', 8), (POSITIVE, 'positive', 24),
                          ([q for q, _ in GUARDS], 'guard', 4)]
            for questions, group, samples in groups:
                for qi, question in enumerate(questions):
                    qkind = ('direct' if qi < 16 else 'indirect') if group == 'heldout' else group
                    qindex = qi if group == 'heldout' else (24 if group == 'positive' else 25 + qi)
                    for sample in range(samples):
                        rows.append((condition, qindex, question, qkind, sample))
    return rows

def row_seed(condition_id: str, question_index: int, sample: int) -> int:
    raw = f'{condition_id}\0{question_index}\0{sample}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big')

def _hooks(layers, row_ops, prompt_mask):
    """Part 25 row intervention with a prompt mask and all-real cached decode positions."""
    handles = []
    by_layer = {int(layer) for ops in row_ops for layer, _ in ops}
    for li in sorted(by_layer):
        def hook(_module, args, li=li):
            x = args[0]
            mask = prompt_mask if x.shape[1] == prompt_mask.shape[1] else torch.ones(
                (x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
            y = part25._row_intervention(x, li, row_ops, mask)
            return None if y is x else (y,) + tuple(args[1:])
        handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(hook))
    return handles

def _pad(prompts, tok, device):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    width = max(map(len, prompts))
    ids = torch.full((len(prompts), width), int(pad_id or 0), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, prompt in enumerate(prompts):
        row = torch.tensor(prompt, dtype=torch.long, device=device)
        ids[i, width-len(row):] = row
        mask[i, width-len(row):] = 1
    return ids, mask

def _stop_reason(token_ids, tok):
    if token_ids and tok.eos_token_id is not None and int(token_ids[-1]) == int(tok.eos_token_id):
        return 'eos'
    return 'max_new_tokens'

def run(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    answer_path = outdir / 'answers.jsonl'
    tok, model = nr.load_model(MODEL, REVISION, args.device)
    _, layers, _ = nr.find_text_layers(model)
    n_layers = len(layers)
    n_units = layers[0].mlp.down_proj.in_features
    conditions = build_conditions(n_units, n_layers)
    todo = generations(args.phase, conditions)
    existing = set()
    if answer_path.exists():
        with answer_path.open(encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    existing.add((row['condition'], row['question_index'], row['sample']))
                except (ValueError, KeyError):
                    continue
    todo = [r for r in todo if (r[0]['id'], r[1], r[4]) not in existing]
    # Batches never mix questions, so every row in a batch has the same prompt and no padding is needed (Claude,
    # 2026-10-25: left padding would put pad tokens through every layer, which Part 25's same-prompt runs never did).
    by_question: dict[int, list] = {}
    for r in todo:
        by_question.setdefault(r[1], []).append(r)
    batches = [rows[i:i + args.batch] for rows in by_question.values() for i in range(0, len(rows), args.batch)]
    start = time.time()
    done = 0
    with answer_path.open('a', encoding='utf-8') as sink:
        for batch in batches:
            prompts = [part25._token_ids(tok, nr.build_prompt(tok, row[2], False)) for row in batch]
            assert len({tuple(p) for p in prompts}) == 1, 'a batch must hold one prompt (no padding)'
            ids, mask = _pad(prompts, tok, torch.device(args.device))
            row_ops = [part25._operations(row[0]['set_units'], 'zero') for row in batch]
            seeds = [row_seed(row[0]['id'], row[1], row[4]) for row in batch]
            torch.manual_seed(seeds[0])
            if args.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seeds[0])
            if args.device == 'xpu' and hasattr(torch, 'xpu'):
                torch.xpu.manual_seed_all(seeds[0])
            handles = _hooks(layers, row_ops, mask.bool())
            try:
                with torch.no_grad():
                    result = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=48,
                        do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id, use_cache=True)
            finally:
                for handle in handles:
                    handle.remove()
            elapsed = time.time() - start
            for i, row in enumerate(batch):
                plen = ids.shape[1]
                generated = result[i, plen:]
                raw = generated.tolist()
                if tok.eos_token_id is not None and int(tok.eos_token_id) in raw:
                    raw = raw[:raw.index(int(tok.eos_token_id)) + 1]
                generated = torch.tensor(raw, dtype=torch.long, device=generated.device)
                text = tok.decode(generated, skip_special_tokens=True).strip()
                cond, qi, question, kind, sample = row
                record = {'condition': cond['id'], 'set_units': [{'layer': int(l), 'unit': int(u)} for l, u in cond['set_units']],
                    'question_index': qi, 'question': question, 'question_kind': kind, 'sample': sample,
                    'seed': seeds[i], 'answer': text, 'new_tokens': len(raw), 'stop_reason': _stop_reason(raw, tok)}
                sink.write(json.dumps(record, ensure_ascii=False) + '\n')
            sink.flush()
            nr.spill_guard(args.device, args.max_gpu_gb)
            elapsed = time.time() - start
            done += len(batch)
            rate = done / max(elapsed, 1e-9)
            eta = max(0, len(todo) - done) / max(rate, 1e-9)
            print(f'{args.phase}: batch {done}/{len(todo)} elapsed={elapsed:.1f}s ETA={eta:.1f}s', flush=True)
    wall = time.time() - start
    run_obj = {'phase': args.phase, 'model': MODEL, 'revision': REVISION,
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'device': str(part25.actual_device(model)),
        'dtype': 'bfloat16', 'batch_size': args.batch, 'wall_time_seconds': wall,
        'generations_per_second': len(todo) / max(wall, 1e-9),
        'peak_memory_gb': (float(torch.cuda.max_memory_allocated() / 2**30) if args.device == 'cuda' and torch.cuda.is_available() else
                           float(torch.xpu.max_memory_allocated() / 2**30) if args.device == 'xpu' and hasattr(torch, 'xpu') else None),
        'sampling_reproducibility': 'Each row records SHA-256-derived seed from condition id, question index, sample index. Batched generate uses one shared RNG; each batch is seeded from its first row. Results are repeatable only for the same batch order, batch size, backend and library versions; per-row seeds document row identity but do not produce independent row streams.',
        'requested_generations': len(generations(args.phase, conditions)), 'new_generations_this_invocation': len(todo),
        'random_seeds': list(range(6600, 6650)), 'conditions': [{'id': c['id'], 'set_units': c['set_units']} for c in conditions]}
    (outdir / 'run.json').write_text(json.dumps(run_obj, indent=2) + '\n', encoding='utf-8')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['pilot', 'full'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['xpu', 'cuda', 'cpu'], default='xpu')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--max-gpu-gb', type=float, default=20)
    args = p.parse_args()
    if args.batch < 1: p.error('--batch must be positive')
    run(args)

if __name__ == '__main__':
    main()
