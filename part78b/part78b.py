#!/usr/bin/env python3
"""Part 78b sampled generation runner: ten layer-22 size-matched controls."""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part66 import part66
from part78 import part78 as part78
import neuron_report as nr

MODEL, REVISION = part78.MODEL, part78.REVISION
ALIEN = (22, 2742)
CONTROL_UNITS = (1499, 4154, 5572, 5606, 5640, 6210, 8115, 8606, 8920, 9209)
assert list(CONTROL_UNITS) == nr.control_units(9216, 2742, 10, 78078), "Part 78b registered control unit list changed"
CONDITION_IDS = ('k1_makers',) + tuple(f'k_u{u}' for u in CONTROL_UNITS)


def build_conditions(n2742_maximum: float, target_norm: float = 1.0,
                     control_norms: dict[int, float] | None = None):
    """Build registered conditions; control set values match the target-column norm dose."""
    control_norms = control_norms or {u: 1.0 for u in CONTROL_UNITS}
    conditions = [{'id': 'k1_makers', 'makers': True, 'push': None}]
    for unit in CONTROL_UNITS:
        norm = float(control_norms[unit])
        if norm <= 0:
            raise ValueError(f'invalid down_proj column norm for unit {unit}: {norm}')
        value = 2.0 * float(n2742_maximum) * float(target_norm) / norm
        conditions.append({'id': f'k_u{unit}', 'makers': True,
                           'push': {'unit': (22, unit), 'value': value,
                                    'column_norm': norm}})
    return conditions


def planned_rows(conditions=None):
    conditions = conditions or build_conditions(19.125)
    questions = part66.DIRECT + part66.INDIRECT
    rows = []
    for c in conditions:
        for qi, q in enumerate(questions):
            for sample in range(8):
                rows.append((c, qi, q, 'direct' if qi < 16 else 'indirect', sample))
        for gi, (q, _answer) in enumerate(part66.GUARDS):
            for sample in range(4):
                rows.append((c, 25 + gi, q, 'guard', sample))
    return rows


def _existing_rows(path):
    done = set()
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                r = json.loads(line)
                done.add((r['condition'], int(r['question_index']), int(r['sample'])))
            except (ValueError, KeyError, TypeError):
                continue
    return done


def run(args):
    import torch
    import part25
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    maximum, source_json, source_npz = part78.read_pile_maximum(Path(args.record_dir))
    tok, model = nr.load_model(MODEL, REVISION, args.device)
    model.requires_grad_(False)
    _, layers, _ = nr.find_text_layers(model)
    proj = layers[ALIEN[0]].mlp.down_proj
    unit_norms = {u: part78.part62_column_norm(proj, u) for u in (ALIEN[1], *CONTROL_UNITS)}
    target_norm = unit_norms[ALIEN[1]]
    conditions = build_conditions(maximum, target_norm, {u: unit_norms[u] for u in CONTROL_UNITS})
    rows = planned_rows(conditions)
    answer_path = out / 'answers.jsonl'
    existing = _existing_rows(answer_path)
    todo = [r for r in rows if (r[0]['id'], r[1], r[4]) not in existing]
    bykey = defaultdict(list)
    for row in todo:
        bykey[(row[0]['id'], row[1])].append(row)
    batches = [rs[i:i + args.batch] for rs in bykey.values() for i in range(0, len(rs), args.batch)]
    start = time.time()
    batch_evidence = []
    first_push_batches = set()
    with answer_path.open('a', encoding='utf-8') as sink:
        for chunk in batches:
            prompts = [part25._token_ids(tok, nr.build_prompt(tok, r[2], False)) for r in chunk]
            assert len({tuple(p) for p in prompts}) == 1
            ids, mask = part66._pad(prompts, tok, torch.device(args.device))
            c = chunk[0][0]
            rowops = [part25._operations(part66.TARGET, 'zero') for _ in chunk]
            seeds = [part66.row_seed(c['id'], r[1], r[4]) for r in chunk]
            torch.manual_seed(seeds[0])
            if args.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seeds[0])
            if args.device == 'xpu' and hasattr(torch, 'xpu'):
                torch.xpu.manual_seed_all(seeds[0])
            handles = part66._hooks(layers, rowops, mask.bool())
            stats = {'calls': 0, 'max_abs_diff': 0.0}
            if c['push']:
                push = c['push']
                handles.append(layers[push['unit'][0]].mlp.down_proj.register_forward_pre_hook(
                    part78.push_hook(push['unit'][1], push['value'], stats)))
            try:
                with torch.no_grad():
                    result = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=64,
                        do_sample=True, temperature=.7, top_p=.95, pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id, use_cache=True)
            finally:
                for handle in handles:
                    handle.remove()
            if c['push'] and c['id'] not in first_push_batches:
                if stats['calls'] == 0:
                    raise RuntimeError(f"push hook did not fire on first batch for {c['id']}")
                first_push_batches.add(c['id'])
            batch_evidence.append({'condition': c['id'], 'question_index': chunk[0][1],
                                   'calls': stats['calls'], 'max_abs_diff': stats['max_abs_diff']})
            prompt_len = ids.shape[1]
            for i, row in enumerate(chunk):
                raw = result[i, prompt_len:].tolist()
                if tok.eos_token_id is not None and int(tok.eos_token_id) in raw:
                    raw = raw[:raw.index(int(tok.eos_token_id)) + 1]
                sink.write(json.dumps({'condition': c['id'], 'question_index': row[1],
                    'question': row[2], 'question_kind': row[3], 'sample': row[4], 'seed': seeds[i],
                    'answer': tok.decode(raw, skip_special_tokens=True).strip(), 'new_tokens': len(raw),
                    'stop_reason': part66._stop_reason(raw, tok)}, ensure_ascii=False) + '\n')
            sink.flush()
            nr.spill_guard(args.device, args.max_gpu_gb)
    metadata = {'model': MODEL, 'revision': REVISION, 'dtype': 'bfloat16',
        'device': str(part25.actual_device(model)), 'transformers_version': __import__('transformers').__version__,
        'batch_size': args.batch, 'pile_maximum_n2742': maximum,
        'pile_maximum_records': {'json': source_json, 'npz': source_npz},
        'down_proj_column_norms': {'layer': 22, 'N2742': target_norm,
            **{f'N{u}': unit_norms[u] for u in CONTROL_UNITS},
            'method': 'part62/run62.py _column_norm: vector_norm(down_proj.weight.detach().float()[:, unit])'},
        'unit_set_values': {f'N{u}': conditions[i + 1]['push']['value'] for i, u in enumerate(CONTROL_UNITS)},
        'conditions': conditions, 'requested_rows': len(rows), 'new_rows_this_invocation': len(todo),
        'wall_time_seconds': time.time() - start, 'push_hook_batch_evidence': batch_evidence}
    (out / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/part78b')
    parser.add_argument('--device', default='xpu')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--record-dir', default=str(ROOT / 'results/part26'))
    parser.add_argument('--max-gpu-gb', type=float, default=20)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('--batch must be positive')
    if args.dry_run:
        conditions = build_conditions(19.125)
        print(json.dumps({'planned_rows': len(planned_rows(conditions)), 'conditions': conditions}, indent=2))
        return
    run(args)


if __name__ == '__main__':
    main()
