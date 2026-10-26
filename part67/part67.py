#!/usr/bin/env python3
"""Part 67: prefill effects between four Qwen3.5-4B MLP neurons."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import neuron_report as nr
import part25
from part66.part66 import DIRECT, INDIRECT, POSITIVE, MODEL, REVISION

TARGETS = [(6, 753), (7, 1169), (11, 8868), (29, 776)]
UPSTREAM = TARGETS[:3]
SEEDS = list(range(6700, 6750))


def random_single_sets(n_units: int, n_layers: int) -> dict[int, list[list[tuple[int, int]]]]:
    return {layer: [part25._matched_random([(layer, unit)], n_units, n_layers, seed)
                   for seed in SEEDS] for layer, unit in UPSTREAM}


def random_triples(n_units: int, n_layers: int) -> list[list[tuple[int, int]]]:
    return [part25._matched_random(UPSTREAM, n_units, n_layers, seed) for seed in SEEDS]


def build_conditions(n_units: int, n_layers: int) -> list[dict]:
    singles = random_single_sets(n_units, n_layers)
    triples = random_triples(n_units, n_layers)
    rows = [{'id': 'unchanged', 'set_units': [], 'kind': 'baseline'}]
    rows += [{'id': f'zero_{layer}_{unit}', 'set_units': [(layer, unit)], 'kind': 'target'}
             for layer, unit in TARGETS]
    rows += [{'id': f'rand_{layer}_{seed}', 'set_units': units, 'kind': 'random_single',
              'layer': layer, 'seed': seed}
             for layer in (6, 7, 11) for seed, units in zip(SEEDS, singles[layer])]
    rows += [{'id': 'joint_6_7_11', 'set_units': UPSTREAM.copy(), 'kind': 'joint'}]
    rows += [{'id': f'randtriple_{seed}', 'set_units': units, 'kind': 'random_triple', 'seed': seed}
             for seed, units in zip(SEEDS, triples)]
    rows += [{'id': 'layer6_all', 'set_units': [(6, u) for u in range(n_units)], 'kind': 'detection_control'}]
    assert len(rows) == 207
    return rows


def _discover_weight_modules(layer, layer_index: int):
    """Discover SwiGLU projections and the MLP pre-norm from this concrete block."""
    named = list(layer.named_modules())
    modules = dict(named)
    gate = [(n, m) for n, m in named if n.rsplit('.', 1)[-1] == 'gate_proj' and hasattr(m, 'weight')]
    up = [(n, m) for n, m in named if n.rsplit('.', 1)[-1] == 'up_proj' and hasattr(m, 'weight')]
    d_model = int(layer.mlp.down_proj.weight.shape[0])
    def valid_proj(items):
        return [(n, m) for n, m in items if len(m.weight.shape) == 2 and int(m.weight.shape[0]) == int(layer.mlp.down_proj.weight.shape[1])]
    gate, up = valid_proj(gate), valid_proj(up)
    if len(gate) != 1 or len(up) != 1:
        raise RuntimeError(f'layer {layer_index}: expected one gate_proj and up_proj matching MLP width; '
                           f'found gate={[n for n,_ in gate]}, up={[n for n,_ in up]}')
    norms = [(n, m) for n, m in named if 'norm' in n.lower() and getattr(m, 'weight', None) is not None
             and tuple(m.weight.shape) == (d_model,)]
    # The MLP input norm is the unique norm inside the MLP block, if present. Some implementations
    # attach it to the decoder block; in that case require an explicit post-attention norm name.
    mlp_norms = [(n, m) for n, m in norms if n.startswith('mlp.')]
    if len(mlp_norms) == 1:
        norm = mlp_norms
    else:
        norm = [(n, m) for n, m in norms if 'post_attention' in n.lower()]
    if len(norm) != 1:
        raise RuntimeError(f'layer {layer_index}: could not uniquely identify MLP pre-norm of width {d_model}; '
                           f'candidates={[n for n,_ in norms]}')
    names = {'gate': gate[0][0], 'up': up[0][0], 'norm': norm[0][0]}
    print(f"weight modules layer {layer_index}: gate={names['gate']} up={names['up']} norm={names['norm']}", flush=True)
    return gate[0][1], up[0][1], norm[0][1], names


def calculate_weight_paths(layers, conditions):
    randoms = {(int(c['layer']), int(c['seed'])): c['set_units'][0] for c in conditions
               if c['kind'] == 'random_single'}
    upstream_units = {(l, u) for l, u in UPSTREAM}
    upstream_units.update(unit for unit in (u[0] for u in randoms.values()))
    needed_downstream = sorted({lj for lj, _ in TARGETS if any(li < lj for li, _ in UPSTREAM)})
    discovered = {li: _discover_weight_modules(layers[li], li) for li in needed_downstream}
    result = {'formula': 'sum_d W_down_upstream[d,u] * norm_downstream[d] * W_gate_or_up_downstream[j,d]',
              'module_names': {str(li): discovered[li][3] for li in needed_downstream},
              'paths': []}
    for (li, unit) in sorted(upstream_units):
        down = layers[li].mlp.down_proj.weight.detach().float().cpu()[:, unit]
        for lj, target_unit in TARGETS:
            if lj <= li:
                continue
            gate, up, norm, _ = discovered[lj]
            gain = norm.weight.detach().float().cpu()
            vals = {}
            for label, proj in (('gate', gate), ('up', up)):
                direction = proj.weight.detach().float().cpu()[target_unit, :]
                vals[label] = float(torch.dot(down * gain, direction).item())
            result['paths'].append({'upstream': [li, int(unit)], 'downstream': [lj, target_unit], **vals})
    return result


def _activation_hooks(layers, row_ops, prompt_length, targets):
    handles, captured = [], {}
    by_layer = {int(l) for ops in row_ops for l, _ in ops}
    hook_layers = by_layer | {l for l, _ in targets}
    for li in sorted(hook_layers):
        target_units = [u for l, u in targets if l == li]
        def hook(_module, args, li=li, target_units=target_units):
            x = args[0]
            y = part25._row_intervention(x, li, row_ops)
            observed = x if y is x else y
            if target_units:
                captured[li] = observed[:, :, target_units].detach().float().cpu()
            return None if y is x else (y,) + tuple(args[1:])
        handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(hook))
    return handles, captured


def run(args):
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    tok, model = nr.load_model(MODEL, REVISION, args.device)
    _, layers, _ = nr.find_text_layers(model)
    n_layers = len(layers); n_units = int(layers[0].mlp.down_proj.in_features)
    conditions = build_conditions(n_units, n_layers)
    weight_paths = calculate_weight_paths(layers, conditions)
    prompts = DIRECT + INDIRECT + POSITIVE
    started = time.time()
    records = []
    for qi, question in enumerate(prompts):
        prompt_ids = part25._token_ids(tok, nr.build_prompt(tok, question, False))
        for begin in range(0, len(conditions), args.batch):
            batch = conditions[begin:begin + args.batch]
            ids = torch.tensor(prompt_ids, dtype=torch.long, device=part25.actual_device(model)).unsqueeze(0).expand(len(batch), -1)
            row_ops = [part25._operations(c['set_units'], 'zero') for c in batch]
            handles, captured = _activation_hooks(layers, row_ops, len(prompt_ids), TARGETS)
            try:
                with torch.no_grad():
                    model(input_ids=ids)
            finally:
                for h in handles: h.remove()
            for ri, condition in enumerate(batch):
                for target_index, (layer, unit) in enumerate(TARGETS):
                    values = captured[layer][ri, :, target_index].tolist()
                    row = {'condition': condition['id'], 'prompt_index': qi, 'target': [layer, unit],
                           'mean': float(sum(values) / len(values)), 'last': float(values[-1])}
                    if condition['id'] == 'unchanged': row['positions'] = values
                    records.append(row)
            del ids
            nr.spill_guard(args.device, args.max_gpu_gb)
            print(f'prompt {qi+1}/{len(prompts)} conditions {begin+len(batch)}/{len(conditions)}', flush=True)
    wall = time.time() - started
    conditions_json = [{'id': c['id'], 'kind': c['kind'], 'set_units': [[int(l), int(u)] for l,u in c['set_units']],
                        **({'seed': c['seed']} if 'seed' in c else {}), **({'layer': c['layer']} if 'layer' in c else {})}
                       for c in conditions]
    (outdir/'activations.json').write_text(json.dumps({'conditions': conditions_json, 'prompts': prompts,
        'targets': [list(x) for x in TARGETS], 'records': records}, indent=1)+'\n', encoding='utf-8')
    (outdir/'weights.json').write_text(json.dumps(weight_paths, indent=1)+'\n', encoding='utf-8')
    peak = (float(torch.cuda.max_memory_allocated()/2**30) if args.device == 'cuda' and torch.cuda.is_available() else
            float(torch.xpu.max_memory_allocated()/2**30) if args.device == 'xpu' and hasattr(torch, 'xpu') else None)
    (outdir/'run.json').write_text(json.dumps({'model': MODEL, 'revision': REVISION,
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'device': str(part25.actual_device(model)),
        'dtype': 'bfloat16', 'batch_size': args.batch, 'wall_time_seconds': wall, 'peak_memory_gb': peak,
        'prompt_count': len(prompts), 'condition_count': len(conditions), 'forward_count': len(prompts)*((len(conditions)+args.batch-1)//args.batch)}, indent=1)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True); p.add_argument('--device', choices=['xpu','cuda','cpu'], default='xpu')
    p.add_argument('--batch', type=int, default=32); p.add_argument('--max-gpu-gb', type=float, default=20)
    a = p.parse_args()
    if a.batch < 1: p.error('--batch must be positive')
    run(a)

if __name__ == '__main__': main()
