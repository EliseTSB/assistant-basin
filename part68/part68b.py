#!/usr/bin/env python3
"""Part 68b: remeasure replicating drivers against activity-matched units."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Bind `part67` to the repo's namespace package BEFORE the part67/ folder joins sys.path. With that folder first on
# the path, part68.py's `from part67 import part67` found part67/part67.py as a plain module and main() crashed on
# import (Claude, 2026-10-26, caught on the host before the first launch; test_main_imports_resolve pins it).
import part67  # noqa: E402,F401
sys.path.append(str(Path(__file__).resolve().parents[1] / 'part67'))
import part67b as p67b
from core68 import MAKERS, TARGETS, TEST_PROMPTS

# Same pinned checkpoint as part68/part68.py; importing part66 eagerly imports torch,
# which would make the CPU-free selection tests impossible to collect.
MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'

torch = None
nr = None
part25 = None


def _require_runtime():
    global torch, nr, part25
    if torch is None:
        try:
            import torch as torch_module
            import neuron_report as nr_module
            import part25 as part25_module
        except ModuleNotFoundError as exc:
            raise RuntimeError('model execution requires the part68 runtime, including torch') from exc
        torch, nr, part25 = torch_module, nr_module, part25_module


def select_magnitude_matched(target: int, magnitudes: list[float], excluded: set[int], count: int = 50) -> list[int]:
    """Part 67b ordering, with the Part 68b cross-target candidate/maker exclusions."""
    if target < 0 or target >= len(magnitudes):
        raise ValueError(f'target unit {target} is outside magnitude list of length {len(magnitudes)}')
    excluded = {int(unit) for unit in excluded if 0 <= int(unit) < len(magnitudes)}
    pool = [u for u in range(len(magnitudes)) if u not in excluded and u != target]
    if count < 0 or count > len(pool):
        raise ValueError(f'cannot select {count} matched units from {len(pool)} eligible units')
    # Preserve Part 67b's exact distance/tie ordering. Mark additional exclusions as
    # infinitely far away, request enough slots to pass them, then discard them.
    adjusted = [float('inf') if unit in excluded and unit != target else float(value)
                for unit, value in enumerate(magnitudes)]
    requested = count + len(excluded - {target})
    selected = p67b.select_magnitude_matched(target, adjusted, requested)
    return [unit for unit in selected if unit not in excluded][:count]


def _target_values(model, layers, ids, row_ops, target):
    """The target's down_proj input per row and position, through Part 68's tested path, which hooks every layer an
    operation names. The build's first version hooked only the target's own layer, and part25._row_intervention
    applies only that layer's operations, so no earlier candidate or matched unit was ever zeroed (Claude,
    2026-10-26, caught on the host before the first launch; test_run_b_matches_part68s_direct_measurement)."""
    import part68 as p68
    return p68._target_activations(model, layers, ids, row_ops, target).detach().float().cpu()


def run_b(model, layers, prompt_id_lists, replicating, out_dir, batch=32, spill_guard=None,
          match_count=50, device=None, max_gpu_gb=14, candidate_union=None, prompt_labels=None):
    """Run magnitude collection and all test-half conditions using token-id lists.

    `replicating` maps target (layer, unit) tuples to candidate dict lists. `match_count`
    exists to allow the same full path to run on deliberately small test models.
    """
    if batch < 1:
        raise ValueError('batch must be positive')
    if not prompt_id_lists:
        raise ValueError('at least one test prompt is required')
    _require_runtime()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        try:
            device = str(next(model.parameters()).device)
        except StopIteration:
            device = 'cpu'
    device = str(device)
    spill_guard = spill_guard or (lambda *_args: None)
    prompt_labels = list(prompt_labels) if prompt_labels is not None else [None] * len(prompt_id_lists)
    if len(prompt_labels) != len(prompt_id_lists):
        raise ValueError('prompt_labels and prompt_id_lists must have the same length')
    n_units = int(layers[0].mlp.down_proj.in_features)
    started = time.time()

    # One unchanged pass per test prompt. Each prompt contributes its position mean equally.
    sums = {li: np.zeros(n_units, dtype=np.float64) for li in range(len(layers))}
    def make_mag_hook(li):
        def hook(_module, args):
            sums[li][:] += args[0].detach().float().abs().mean(dim=(0, 1)).cpu().numpy()
        return hook
    handles = [layers[li].mlp.down_proj.register_forward_pre_hook(make_mag_hook(li))
               for li in range(len(layers))]
    try:
        with torch.no_grad():
            for token_ids in prompt_id_lists:
                ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
                model(input_ids=ids)
                spill_guard(device, max_gpu_gb)
    finally:
        for handle in handles:
            handle.remove()
    magnitudes = {li: (values / len(prompt_id_lists)).tolist() for li, values in sums.items()}
    (out_dir / 'magnitudes.json').write_text(json.dumps({str(k): v for k, v in magnitudes.items()}, indent=1) + '\n')

    # Exclude the union of both registered candidate lists, plus every maker, for every match pool.
    all_candidates = {(int(c['layer']), int(c['unit'])) for rows in replicating.values() for c in rows}
    # Replicating is only a subset; callers may attach the complete candidate union for strict exclusion.
    candidate_union = set(all_candidates) if candidate_union is None else set(candidate_union)
    candidate_union.update(tuple(map(int, item)) for item in all_candidates)
    excluded_by_layer = {}
    for li, unit in candidate_union | set(MAKERS):
        excluded_by_layer.setdefault(li, set()).add(unit)

    plans, matched_json = {}, {}
    for target, candidates in replicating.items():
        target = tuple(map(int, target))
        unique_matched = set()
        per_candidate = {}
        for candidate in candidates:
            layer, unit = int(candidate['layer']), int(candidate['unit'])
            selected = select_magnitude_matched(unit, magnitudes[layer], excluded_by_layer.get(layer, set()), match_count)
            per_candidate[f'{layer}_{unit}'] = selected
            unique_matched.update((layer, u) for u in selected)
        plans[target] = (candidates, per_candidate, sorted(unique_matched))
        matched_json[f'{target[0]}_{target[1]}'] = {
            f'{key}': [{'layer': int(key.split('_')[0]), 'unit': u, 'magnitude': magnitudes[int(key.split('_')[0])][u]}
                       for u in units] for key, units in per_candidate.items()}
    (out_dir / 'matched.json').write_text(json.dumps(matched_json, indent=1) + '\n')

    records, condition_count = [], 0
    for target, (candidates, per_candidate, unique_matched) in plans.items():
        conditions = [{'id': 'unchanged', 'kind': 'baseline', 'set_units': []},
                      {'id': 'zero_nothing', 'kind': 'control', 'set_units': []}]
        conditions += [{'id': f"cand_{int(c['layer'])}_{int(c['unit'])}", 'kind': 'candidate',
                        'set_units': [(int(c['layer']), int(c['unit']))]} for c in candidates]
        conditions += [{'id': f'matched_{li}_{u}', 'kind': 'matched', 'set_units': [(li, u)]}
                       for li, u in unique_matched]
        condition_count += len(conditions)
        for qi, token_ids in enumerate(prompt_id_lists):
            for begin in range(0, len(conditions), batch):
                batch_conditions = conditions[begin:begin + batch]
                ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(len(batch_conditions), 1)
                row_ops = [part25._operations(c['set_units'], 'zero') for c in batch_conditions]
                values = _target_values(model, layers, ids, row_ops, target).numpy().astype(np.float64)
                for ri, condition in enumerate(batch_conditions):
                    row = {'target': list(target), 'half': 'test', 'condition': condition['id'],
                           'kind': condition['kind'], 'prompt_index': qi, 'question': prompt_labels[qi],
                           'mean': float(values[ri].mean())}
                    if condition['id'] in ('unchanged', 'zero_nothing'):
                        row['positions'] = [float(v) for v in values[ri]]
                    records.append(row)
                spill_guard(device, max_gpu_gb)
    activations = {'targets': [list(t) for t in plans], 'halves': {'test': prompt_labels},
                   'n_units': n_units, 'candidates': {f'{t[0]}_{t[1]}': [dict(c) for c in p[0]] for t, p in plans.items()},
                   'matched': matched_json, 'records': records}
    (out_dir / 'activations_b.json').write_text(json.dumps(activations, indent=1) + '\n')
    run_obj = {'model': MODEL, 'revision': REVISION,
               'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'device': device, 'wall_time_seconds': time.time() - started,
               'peak_memory_gb': (float(torch.cuda.max_memory_allocated() / 2**30)
                                  if device.startswith('cuda') and torch.cuda.is_available() else
                                  float(torch.xpu.max_memory_allocated() / 2**30)
                                  if device.startswith('xpu') and hasattr(torch, 'xpu') and torch.xpu.is_available()
                                  else None),
               'batch_size': batch, 'match_count': match_count, 'prompt_count': len(prompt_id_lists),
               'condition_count': condition_count, 'record_count': len(records),
               'candidate_count': sum(len(p[0]) for p in plans.values()),
               'unique_matched_count': sum(len(p[2]) for p in plans.values())}
    (out_dir / 'run_b.json').write_text(json.dumps(run_obj, indent=1) + '\n')
    return run_obj


def main():
    _require_runtime()
    import neuron_report as nr_module
    import part68 as p68
    from part66.part66 import MODEL as model_id, REVISION as revision
    parser = argparse.ArgumentParser()
    parser.add_argument('--part68', default='results/part68/full')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=['xpu', 'cuda', 'cpu'], default='xpu')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--max-gpu-gb', type=float, default=14)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('--batch must be positive')
    part68_dir = Path(args.part68)
    score = json.loads((part68_dir / 'score.json').read_text(encoding='utf-8'))
    replicating = {tuple(map(int, key.split('_'))): rows['halves']['test']['replicating']
                   for key, rows in score['targets'].items()}
    # Full candidate lists are needed for the cross-target exclusion, not just replicating candidates.
    candidate_union = {(int(c['layer']), int(c['unit'])) for obj in score['targets'].values()
                       for c in obj['candidates']}
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tok, model = nr_module.load_model(model_id, revision, args.device)
    model.requires_grad_(False)
    _, layers, _ = nr.find_text_layers(model)
    prompt_ids = [p68._prompt_ids(tok, q) for q in TEST_PROMPTS]
    run_b(model, layers, prompt_ids, replicating, outdir, batch=args.batch,
          spill_guard=nr_module.spill_guard, device=part25.actual_device(model), max_gpu_gb=args.max_gpu_gb,
          candidate_union=candidate_union, prompt_labels=TEST_PROMPTS)


if __name__ == '__main__':
    main()
