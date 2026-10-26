#!/usr/bin/env python3
"""Part 70 sampled generation: makers zeroed with or without their two late drivers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Bind modules, not the part66 namespace package. Defer Part 66's torch import so --help
# remains usable on selection-only hosts; real execution imports it before planning.
try:
    from part66 import part66, score66  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    part66 = None
    from part66 import score66  # noqa: E402

MODEL = 'Qwen/Qwen3.5-4B' if part66 is None else part66.MODEL
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a' if part66 is None else part66.REVISION
TARGET = [(6, 753), (7, 1169), (11, 8868), (29, 776)] if part66 is None else part66.TARGET

def _require_part66():
    global part66
    if part66 is None:
        from part66 import part66 as loaded
        part66 = loaded
    return part66
PAIR = [(27, 3325), (24, 3949)]


def _json_units(units):
    return [{'layer': int(layer), 'unit': int(unit)} for layer, unit in units]


def draw_matched_units(matched: dict, k: int) -> list[tuple[int, int]]:
    """Draw one matched unit per driver using the registered independent seed for set k."""
    import numpy as np
    pools = []
    for layer, unit in PAIR:
        key = f'{layer}_{unit}'
        try:
            records = matched['29_776'][key]
        except (KeyError, TypeError) as exc:
            raise ValueError(f'missing matched pool 29_776/{key}') from exc
        if not records:
            raise ValueError(f'empty matched pool 29_776/{key}')
        pools.append([(int(r['layer']), int(r['unit'])) for r in records])
    rng = np.random.default_rng(7000 + int(k))
    return [pool[int(rng.integers(0, len(pool)))] for pool in pools]


def build_conditions(matched: dict) -> list[dict]:
    conditions = [
        {'id': 'target', 'set_units': list(TARGET), 'kind': 'target'},
        {'id': 'target_plus_pair', 'set_units': list(TARGET) + list(PAIR), 'kind': 'target_plus_pair'},
    ]
    conditions.extend(
        {'id': f'target_plus_matched_{k}', 'set_units': list(TARGET) + draw_matched_units(matched, k),
         'kind': 'matched', 'matched_index': k, 'seed': 7000 + k}
        for k in range(20)
    )
    return conditions


def generations(conditions: list[dict]) -> list[tuple]:
    """Part 66 row tuple format, with the registered per-condition sample counts."""
    p66 = _require_part66()
    rows = []
    heldout = p66.DIRECT + p66.INDIRECT
    for cond in conditions:
        if cond['kind'] == 'matched':
            groups = [(heldout, 'heldout', 1)]
        else:
            groups = [(heldout, 'heldout', 8), (p66.POSITIVE, 'positive', 24),
                      ([q for q, _ in p66.GUARDS], 'guard', 4)]
        for questions, group, samples in groups:
            for qi, question in enumerate(questions):
                qkind = ('direct' if qi < len(p66.DIRECT) else 'indirect') if group == 'heldout' else group
                qindex = qi if group == 'heldout' else (24 if group == 'positive' else 25 + qi)
                for sample in range(samples):
                    rows.append((cond, qindex, question, qkind, sample))
    return rows


def _read_existing(path: Path) -> set[tuple[str, int, int]]:
    existing = set()
    if path.exists():
        with path.open(encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    existing.add((row['condition'], int(row['question_index']), int(row['sample'])))
                except (ValueError, KeyError, TypeError):
                    continue
    return existing


def run_70(model, tok, layers, conditions, out_dir, batch=25, device='xpu', spill_guard=None):
    """Run planned rows on supplied model/tokenizer, supporting answer-file resume."""
    import torch
    p66 = _require_part66()
    import part25
    import neuron_report as nr

    if batch < 1:
        raise ValueError('batch must be positive')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    answer_path = out / 'answers.jsonl'
    rows = generations(conditions)
    existing = _read_existing(answer_path)
    todo = [r for r in rows if (r[0]['id'], r[1], r[4]) not in existing]
    by_question = defaultdict(list)
    for row in todo:
        by_question[row[1]].append(row)
    batches = [rs[i:i + batch] for rs in by_question.values() for i in range(0, len(rs), batch)]
    start = time.time()
    done = 0
    with answer_path.open('a', encoding='utf-8') as sink:
        for chunk in batches:
            prompts = [part25._token_ids(tok, nr.build_prompt(tok, row[2], False)) for row in chunk]
            assert len({tuple(p) for p in prompts}) == 1, 'a batch must hold one prompt (no padding)'
            ids, mask = p66._pad(prompts, tok, torch.device(device))
            row_ops = [part25._operations(row[0]['set_units'], 'zero') for row in chunk]
            seeds = [p66.row_seed(row[0]['id'], row[1], row[4]) for row in chunk]
            torch.manual_seed(seeds[0])
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seeds[0])
            if device == 'xpu' and hasattr(torch, 'xpu'):
                torch.xpu.manual_seed_all(seeds[0])
            handles = p66._hooks(layers, row_ops, mask.bool())
            try:
                with torch.no_grad():
                    result = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=48,
                        do_sample=True, temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id, use_cache=True)
            finally:
                for handle in handles:
                    handle.remove()
            for i, row in enumerate(chunk):
                plen = ids.shape[1]
                generated = result[i, plen:]
                raw = generated.tolist()
                if tok.eos_token_id is not None and int(tok.eos_token_id) in raw:
                    raw = raw[:raw.index(int(tok.eos_token_id)) + 1]
                answer = tok.decode(raw, skip_special_tokens=True).strip()
                cond, qi, question, kind, sample = row
                record = {'condition': cond['id'], 'set_units': _json_units(cond['set_units']),
                    'question_index': qi, 'question': question, 'question_kind': kind, 'sample': sample,
                    'seed': seeds[i], 'answer': answer, 'new_tokens': len(raw),
                    'stop_reason': p66._stop_reason(raw, tok)}
                sink.write(json.dumps(record, ensure_ascii=False) + '\n')
            sink.flush()
            if spill_guard is not None:
                spill_guard(device, 14.0)
            else:
                nr.spill_guard(device, 14.0)
            done += len(chunk)
            elapsed = time.time() - start
            rate = done / max(elapsed, 1e-9)
            eta = max(0, len(todo) - done) / max(rate, 1e-9)
            print(f'part70: batch {done}/{len(todo)} elapsed={elapsed:.1f}s ETA={eta:.1f}s', flush=True)
    wall = time.time() - start
    actual_device = str(part25.actual_device(model))
    if device == 'cuda' and torch.cuda.is_available():
        peak = float(torch.cuda.max_memory_allocated() / 2**30)
    elif device == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
        peak = float(torch.xpu.max_memory_allocated() / 2**30)
    else:
        peak = None
    run_obj = {'model': MODEL, 'revision': REVISION,
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'device': actual_device,
        'dtype': 'bfloat16', 'batch_size': batch, 'wall_time_seconds': wall,
        'peak_memory_gb': peak, 'requested_generations': len(rows),
        'new_generations_this_invocation': len(todo),
        'counts': {'requested_by_condition': dict(Counter(r[0]['id'] for r in rows)),
                   'requested_by_question_kind': dict(Counter(r[3] for r in rows)),
                   'new_by_question_kind': dict(Counter(r[3] for r in todo))},
        'sampling_reproducibility': 'Each row records its Part 66 SHA-256-derived seed. Batched generation seeds each batch from its first row; row seeds document identity, not independent per-row RNG streams.',
        'conditions': [{'id': c['id'], 'set_units': _json_units(c['set_units'])} for c in conditions]}
    (out / 'run.json').write_text(json.dumps(run_obj, indent=2) + '\n', encoding='utf-8')
    return rows


def prepare_plan(matched_path: str | Path):
    """All selection and row-planning work main() can do before loading a model."""
    matched = json.loads(Path(matched_path).read_text(encoding='utf-8'))
    conditions = build_conditions(matched)
    rows = generations(conditions)
    return conditions, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--matched', default='results/part68/b/matched.json')
    parser.add_argument('--device', choices=['xpu', 'cuda', 'cpu'], default='xpu')
    parser.add_argument('--batch', type=int, default=25)
    parser.add_argument('--max-gpu-gb', type=float, default=14)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('--batch must be positive')

    # Resolve every cross-part module entry point before loading the large model.
    p66 = _require_part66()
    import part25
    import neuron_report as nr
    required = (p66.TARGET, p66.DIRECT, p66.INDIRECT, p66.POSITIVE, p66.GUARDS,
                p66.row_seed, p66._hooks, p66._pad, p66._stop_reason,
                score66.FAMILIES, score66.score_rows, part25._token_ids, part25._operations,
                part25.actual_device, nr.load_model, nr.find_text_layers, nr.build_prompt, nr.spill_guard)
    if not all(required):
        raise RuntimeError('a required Part 25/66/score66/neuron_report entry point is unavailable')
    conditions, rows = prepare_plan(args.matched)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'conditions.json').write_text(json.dumps([
        {'id': c['id'], 'set_units': _json_units(c['set_units'])} for c in conditions
    ], indent=2) + '\n', encoding='utf-8')
    tok, model = nr.load_model(MODEL, REVISION, args.device)
    _, layers, _ = nr.find_text_layers(model)
    n_layers = len(layers)
    n_units = layers[0].mlp.down_proj.in_features
    if any(layer < 0 or layer >= n_layers or unit < 0 or unit >= n_units
           for c in conditions for layer, unit in c['set_units']):
        raise ValueError('a planned neuron is outside the loaded model dimensions')
    def guard(dev, _limit):
        return nr.spill_guard(dev, args.max_gpu_gb)
    run_70(model, tok, layers, conditions, out, args.batch, args.device, guard)


if __name__ == '__main__':
    main()
