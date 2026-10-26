#!/usr/bin/env python3
"""Part 68: search every earlier MLP unit for the ones that drive two maker neurons (Qwen3.5-4B)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import neuron_report as nr
import part25
from part67 import part67 as part67_helpers
from part66.part66 import MODEL, REVISION
from core68 import (DIRECT, HALVES, INDIRECT, MAKERS, NULL_N, NULL_SEED, POSITIVE,
                           SEARCH_PROMPTS, TARGETS, TEST_PROMPTS, TOP_N, build_conditions,
                           null_units, split_prompts, top_candidates)
OVERLAP_SEED = 6801
OVERLAP_DRAWS = 10000
SPEARMAN_MIN = 0.3


def _prompt_ids(tok, question: str) -> list[int]:
    return [int(x) for x in part25._token_ids(tok, nr.build_prompt(tok, question, False))]


def _target_hook(down_proj, unit: int, sink: dict, *, keep_graph: bool = False):
    def hook(_module, args):
        values = args[0][:, :, int(unit)]
        sink['values'] = values.float() if keep_graph else values.detach().float()
        return None
    return down_proj.register_forward_pre_hook(hook)


def _target_activations(model, layers, ids, row_ops, target: tuple[int, int]) -> torch.Tensor:
    """The target unit's down_proj input at every position, one row per condition, shape (rows, positions)."""
    handles, captured = part67_helpers._activation_hooks(layers, row_ops, ids.shape[1], [target])
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured[target[0]][:, :, part67_helpers._unit_column(target[0], target[1], [target])]


def _attribution(model, layers, ids, target: tuple[int, int], target_layer: int,
                 info: dict | None = None) -> dict[int, np.ndarray]:
    """A_u = sum over positions of -a_u * dt/da_u for every unit of EVERY layer, all from the same backward
    pass, so the 68.5 control is measured rather than written: a unit at or after the target layer has no
    path to t, and autograd leaves its gradient None (recorded as 0) or 0. The one exception is the target
    unit itself, whose entry is the identity derivative (-t exactly, up to bf16): it is removed from the
    array and returned in info as a positive control on the gradient machinery (Claude, 2026-10-26)."""
    n_units = int(layers[0].mlp.down_proj.in_features)
    sink: dict = {}
    handles = [_target_hook(layers[target[0]].mlp.down_proj, target[1], sink, keep_graph=True)]
    captured: dict[int, torch.Tensor] = {}

    def make_hook(li):
        def hook(_module, args):
            x = args[0]
            # Model weights are frozen. Make only the captured activation tensors require
            # gradients, retaining the existing graph whenever an earlier capture created it.
            if not x.requires_grad:
                x.requires_grad_(True)
            x.retain_grad()
            captured[li] = x
            return None
        return hook

    # Every layer, not only the earlier ones: a later layer the sink wrongly depends on must show up here.
    for li in range(len(layers)):
        handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(make_hook(li)))
    try:
        model(input_ids=ids)
        t = sink['values'].float().mean()  # t = mean over prompt positions of the target unit
        t.backward()
    finally:
        for handle in handles:
            handle.remove()
    out = {}
    for li in range(len(layers)):
        a = captured[li]
        grad = a.grad
        if grad is None:
            if target_layer > li:
                raise RuntimeError(f'layer {li}: no gradient reached the down_proj input')
            out[li] = np.zeros(n_units, dtype=np.float64)  # autograd found no path from this layer to t
            continue
        out[li] = (-a.detach().float() * grad.float()).sum(dim=(0, 1)).cpu().numpy().astype(np.float64)
    self_value = float(out[target[0]][target[1]])
    out[target[0]][target[1]] = 0.0
    if info is not None:
        info['target_self_attribution'] = self_value
        info['target_value'] = float(t.detach())
    del captured, sink, handles
    return out


def _peak_gb(device: str):
    if device == 'cuda' and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / 2**30)
    if device == 'xpu' and hasattr(torch, 'xpu'):
        return float(torch.xpu.max_memory_allocated() / 2**30)
    return None


def run_search(args, tok, model, layers, outdir: Path) -> dict:
    n_units = int(layers[0].mlp.down_proj.in_features)
    device = part25.actual_device(model)
    search, _ = split_prompts()
    started = time.time()
    for target in TARGETS:
        sums = {li: np.zeros(n_units, dtype=np.float64) for li in range(len(layers))}
        self_values, target_values = [], []
        for question in search:
            prompt = _prompt_ids(tok, question)
            ids = torch.tensor([prompt], dtype=torch.long, device=device)
            # nr.load_model disables autograd globally; the search needs it back for one backward pass per prompt.
            info: dict = {}
            with torch.enable_grad():
                attribution = _attribution(model, layers, ids, target, target[0], info)
            self_values.append(info['target_self_attribution'])
            target_values.append(info['target_value'])
            for li, values in attribution.items():
                sums[li] += values
            # One prompt per backward pass, so a peak above the limit is the model plus one prompt's graph.
            nr.spill_guard(args.device, args.max_gpu_gb)
            del ids, attribution
        averaged = {li: values / len(search) for li, values in sums.items()}
        arrays = np.stack([averaged[li] for li in range(len(layers))])
        stem = f'attr_{target[0]}_{target[1]}'
        np.savez(outdir / f'{stem}.npz', attribution=arrays)
        candidates = top_candidates(averaged, target[0])
        (outdir / f'{stem}.json').write_text(json.dumps({
            'target': list(target), 'prompt_count': len(search), 'prompts': search,
            'candidates': candidates,
            'later_units_max_abs': float(max(np.abs(arrays[li]).max() for li in range(target[0], len(layers)))),
            # Positive control, not a registered prediction: the target's own entry should equal -t.
            'target_self_attribution_per_prompt': self_values, 'target_value_per_prompt': target_values,
        }, indent=1) + '\n', encoding='utf-8')
        print(f'search {target}: top {candidates[0]}', flush=True)
    wall = time.time() - started
    run_obj = {'phase': 'search', 'model': MODEL, 'revision': REVISION,
               'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'device': str(device), 'dtype': 'bfloat16', 'prompt_batch_size': 1,
               'wall_time_seconds': wall, 'peak_memory_gb': _peak_gb(args.device),
               'targets': [list(t) for t in TARGETS], 'search_prompts': search,
               'search_prompt_count': len(search),
               'backward_passes': len(search) * len(TARGETS),
               'top_n': TOP_N, 'n_units': n_units, 'n_layers': len(layers),
               'attribution': 'A_u = mean over search prompts of sum over positions of -a_u * dt/da_u, '
                              'a_u = down_proj input at unit u, t = mean over positions of the target activation',
               'candidate_files': [f'attr_{t[0]}_{t[1]}.json' for t in TARGETS],
               'attribution_files': [f'attr_{t[0]}_{t[1]}.npz' for t in TARGETS]}
    (outdir / 'run-search.json').write_text(json.dumps(run_obj, indent=1) + '\n', encoding='utf-8')
    return run_obj


def _load_search(outdir: Path) -> dict:
    loaded = {}
    for target in TARGETS:
        path = outdir / f'attr_{target[0]}_{target[1]}.json'
        if not path.is_file():
            raise SystemExit(f'missing search output {path}; run --phase search first')
        loaded[target] = json.loads(path.read_text(encoding='utf-8'))
    return loaded


def run_test(args, tok, model, layers, outdir: Path) -> dict:
    n_units = int(layers[0].mlp.down_proj.in_features)
    device = part25.actual_device(model)
    search_obj = _load_search(outdir)
    started = time.time()
    records = []
    plan = {}
    for target in TARGETS:
        candidates = [{'layer': int(c['layer']), 'unit': int(c['unit']), 'attribution': float(c['attribution'])}
                      for c in search_obj[target]['candidates']]
        nulls = null_units(candidates, n_units)
        conditions = build_conditions(candidates, nulls)
        plan[target] = {'candidates': candidates, 'nulls': nulls, 'conditions': conditions}
        for half, prompts in HALVES.items():
            for qi, question in enumerate(prompts):
                prompt = _prompt_ids(tok, question)
                for begin in range(0, len(conditions), args.batch):
                    batch = conditions[begin:begin + args.batch]
                    ids = torch.tensor([prompt], dtype=torch.long, device=device).repeat(len(batch), 1)
                    row_ops = [part25._operations(c['set_units'], 'zero') for c in batch]
                    values = _target_activations(model, layers, ids, row_ops, target)
                    values = values.cpu().numpy().astype(np.float64)
                    for ri, condition in enumerate(batch):
                        row = {'target': list(target), 'half': half, 'condition': condition['id'],
                               'kind': condition['kind'], 'prompt_index': qi, 'question': question,
                               'mean': float(values[ri].mean())}
                        if condition['id'] == 'unchanged':
                            row['positions'] = [float(v) for v in values[ri]]
                        records.append(row)
                    nr.spill_guard(args.device, args.max_gpu_gb)
                print(f'test {target} {half} {qi+1}/{len(prompts)}', flush=True)
    wall = time.time() - started
    conditions_json = []
    for target in TARGETS:
        for c in plan[target]['conditions']:
            conditions_json.append({'target': list(target), 'id': c['id'], 'kind': c['kind'],
                                    'set_units': [[int(l), int(u)] for l, u in c['set_units']]})
    (outdir / 'activations.json').write_text(json.dumps({
        'targets': [list(t) for t in TARGETS], 'halves': {k: v for k, v in HALVES.items()}, 'n_units': n_units,
        'candidates': {f'{t[0]}_{t[1]}': plan[t]['candidates'] for t in TARGETS},
        'nulls': {f'{t[0]}_{t[1]}': plan[t]['nulls'] for t in TARGETS},
        'conditions': conditions_json, 'records': records}, indent=1) + '\n', encoding='utf-8')
    run_obj = {'phase': 'test', 'model': MODEL, 'revision': REVISION,
               'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'device': str(device), 'dtype': 'bfloat16', 'batch_size': args.batch,
               'wall_time_seconds': wall, 'peak_memory_gb': _peak_gb(args.device),
               'targets': [list(t) for t in TARGETS], 'null_seed': NULL_SEED, 'null_count': NULL_N,
               'candidate_count': TOP_N, 'overlap_seed': OVERLAP_SEED, 'overlap_draws': OVERLAP_DRAWS,
               'spearman_min': SPEARMAN_MIN,
               'condition_count': len(conditions_json), 'prompt_count': sum(len(v) for v in HALVES.values()),
               'forward_count': sum((len(plan[t]['conditions']) + args.batch - 1) // args.batch
                                    for t in TARGETS for _ in HALVES.values()),
               'records': len(records)}
    (outdir / 'run.json').write_text(json.dumps(run_obj, indent=1) + '\n', encoding='utf-8')
    return run_obj


def run(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tok, model = nr.load_model(MODEL, REVISION, args.device)
    model.requires_grad_(False)
    _, layers, _ = nr.find_text_layers(model)
    if args.phase in ('search', 'all'):
        run_search(args, tok, model, layers, outdir)
    if args.phase in ('test', 'all'):
        run_test(args, tok, model, layers, outdir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['search', 'test', 'all'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['xpu', 'cuda', 'cpu'], default='xpu')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--max-gpu-gb', type=float, default=20)
    args = p.parse_args()
    if 1 > args.batch:
        p.error('--batch must be positive')
    run(args)


if __name__ == '__main__':
    main()
