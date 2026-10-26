#!/usr/bin/env python3
"""Part 67b: test neuron effects against a mean-activation-matched null."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import score67

TARGETS = [(6, 753), (7, 1169), (11, 8868), (29, 776)]
UPSTREAM = TARGETS[:3]
MAG_LAYERS = tuple(layer for layer, _ in UPSTREAM)
MATCH_COUNT = 50
torch = None
part67 = None
nr = None
part25 = None


def _require_model_runtime():
    global torch, part67, nr, part25, TARGETS, UPSTREAM, MAG_LAYERS
    if torch is not None:
        return
    try:
        import torch as torch_module
        import neuron_report as nr_module
        import part25 as part25_module
        import part67 as part67_module
    except ModuleNotFoundError as exc:
        raise RuntimeError('model execution requires the part67 runtime, including torch') from exc
    torch, nr, part25, part67 = torch_module, nr_module, part25_module, part67_module
    TARGETS, UPSTREAM = part67.TARGETS, part67.UPSTREAM
    MAG_LAYERS = tuple(layer for layer, _ in UPSTREAM)


def select_magnitude_matched(target: int, magnitudes: list[float], count: int = MATCH_COUNT) -> list[int]:
    """Return closest-magnitude unit indices, excluding target and breaking ties by index."""
    if target < 0 or target >= len(magnitudes):
        raise ValueError(f'target unit {target} is outside magnitude list of length {len(magnitudes)}')
    available = len(magnitudes) - 1
    if count < 0 or count > available:
        raise ValueError(f'cannot select {count} controls from {available} non-target units')
    target_magnitude = float(magnitudes[target])
    return sorted((unit for unit in range(len(magnitudes)) if unit != target),
                  key=lambda unit: (abs(float(magnitudes[unit]) - target_magnitude), unit))[:count]


def empirical_p(observed: float, null: list[float]) -> float:
    """Two-sided empirical p with the registered +1 correction; ties count as exceedances."""
    if not null:
        raise ValueError('the magnitude-matched null must contain at least one unit')
    return (1 + sum(abs(value) >= abs(observed) for value in null)) / (len(null) + 1)


def survives(observed: float, null: list[float]) -> bool:
    """True only when the target's absolute Delta strictly exceeds every control."""
    if not null:
        raise ValueError('the magnitude-matched null must contain at least one unit')
    return all(abs(observed) > abs(value) for value in null)


def build_conditions(magnitudes: dict[int, list[float]], count: int = MATCH_COUNT) -> tuple[list[dict], dict]:
    """Build unchanged, three target-zero, and one-at-a-time matched-zero conditions."""
    rows = [{'id': 'unchanged', 'set_units': [], 'kind': 'baseline'}]
    rows += [{'id': f'zero_{layer}_{unit}', 'set_units': [(layer, unit)], 'kind': 'target'}
             for layer, unit in UPSTREAM]
    matches = {}
    for layer, target in UPSTREAM:
        if layer not in magnitudes:
            raise ValueError(f'missing magnitudes for layer {layer}')
        selected = select_magnitude_matched(target, magnitudes[layer], count)
        target_magnitude = float(magnitudes[layer][target])
        units = [{'unit': unit, 'magnitude': float(magnitudes[layer][unit])} for unit in selected]
        matches[f'{layer}_{target}'] = {'layer': layer, 'target': target,
            'target_magnitude': target_magnitude, 'units': units,
            'magnitude_range': [min(row['magnitude'] for row in units), max(row['magnitude'] for row in units)]}
        rows.extend({'id': f'mag_{layer}_{unit}', 'set_units': [(layer, unit)], 'kind': 'magnitude_matched',
                     'layer': layer, 'unit': unit} for unit in selected)
    return rows, matches


def collect_magnitudes(model, layers, prompt_ids_list: list[list[int]], device) -> dict[int, list[float]]:
    """Run unchanged prompts and accumulate per-unit mean absolute down_proj inputs."""
    _require_model_runtime()
    sums = {layer: None for layer in MAG_LAYERS}
    position_counts = {layer: 0 for layer in MAG_LAYERS}
    handles = []
    for layer_index in MAG_LAYERS:
        def hook(_module, args, layer_index=layer_index):
            x = args[0].detach().float()
            values = x.abs().sum(dim=tuple(range(x.ndim - 1))).cpu()
            sums[layer_index] = values if sums[layer_index] is None else sums[layer_index] + values
            position_counts[layer_index] += x.numel() // x.shape[-1]
        handles.append(layers[layer_index].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for token_ids in prompt_ids_list:
                ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
                model(input_ids=ids)
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for layer in MAG_LAYERS:
        if sums[layer] is None or position_counts[layer] == 0:
            raise RuntimeError(f'no down_proj inputs captured for layer {layer}')
        result[layer] = (sums[layer] / position_counts[layer]).tolist()
    return result


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1) + '\n', encoding='utf-8')


def _activation_records(model, layers, tok, prompts, conditions, args):
    records = []
    for qi, question in enumerate(prompts):
        prompt_ids = part25._token_ids(tok, nr.build_prompt(tok, question, False))
        for begin in range(0, len(conditions), args.batch):
            batch = conditions[begin:begin + args.batch]
            ids = torch.tensor(prompt_ids, dtype=torch.long, device=part25.actual_device(model)).unsqueeze(0).expand(len(batch), -1)
            row_ops = [part25._operations(condition['set_units'], 'zero') for condition in batch]
            handles, captured = part67._activation_hooks(layers, row_ops, len(prompt_ids), TARGETS)
            try:
                with torch.no_grad():
                    model(input_ids=ids)
            finally:
                for handle in handles:
                    handle.remove()
            for ri, condition in enumerate(batch):
                for layer, unit in TARGETS:
                    values = captured[layer][ri, :, part67._unit_column(layer, unit)].tolist()
                    row = {'condition': condition['id'], 'prompt_index': qi, 'target': [layer, unit],
                           'mean': float(sum(values) / len(values)), 'last': float(values[-1])}
                    if condition['id'] == 'unchanged':
                        row['positions'] = values
                    records.append(row)
            del ids
            nr.spill_guard(args.device, args.max_gpu_gb)
            print(f'pass 2 prompt {qi + 1}/{len(prompts)} conditions {begin + len(batch)}/{len(conditions)}', flush=True)
    return records


def run(args):
    _require_model_runtime()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tok, model = nr.load_model(part67.MODEL, part67.REVISION, args.device)
    _, layers, _ = nr.find_text_layers(model)
    prompts = part67.DIRECT + part67.INDIRECT + part67.POSITIVE
    prompt_ids = [part25._token_ids(tok, nr.build_prompt(tok, question, False)) for question in prompts]
    started = time.time()
    magnitudes = collect_magnitudes(model, layers, prompt_ids, part25.actual_device(model))
    _write_json(outdir / 'unit_magnitudes.json', {str(layer): values for layer, values in magnitudes.items()})
    conditions, matches = build_conditions(magnitudes)
    _write_json(outdir / 'magnitude_matches.json', matches)
    records = _activation_records(model, layers, tok, prompts, conditions, args)
    conditions_json = [{'id': c['id'], 'kind': c['kind'],
                        'set_units': [[int(layer), int(unit)] for layer, unit in c['set_units']],
                        **({'layer': int(c['layer'])} if 'layer' in c else {}),
                        **({'unit': int(c['unit'])} if 'unit' in c else {})} for c in conditions]
    _write_json(outdir / 'activations.json', {'conditions': conditions_json, 'prompts': prompts,
        'targets': [list(row) for row in TARGETS], 'records': records})
    peak = (float(torch.cuda.max_memory_allocated() / 2**30) if args.device == 'cuda' and torch.cuda.is_available() else
            float(torch.xpu.max_memory_allocated() / 2**30) if args.device == 'xpu' and hasattr(torch, 'xpu') else None)
    _write_json(outdir / 'run.json', {'model': part67.MODEL, 'revision': part67.REVISION,
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'device': str(part25.actual_device(model)), 'dtype': 'bfloat16', 'batch_size': args.batch,
        'wall_time_seconds': time.time() - started, 'peak_memory_gb': peak, 'prompt_count': len(prompts),
        'condition_count': len(conditions), 'forward_count_pass1': len(prompts),
        'forward_count_pass2': len(prompts) * ((len(conditions) + args.batch - 1) // args.batch),
        'magnitude_matches': matches})


def score(directory: str | Path) -> dict:
    directory = Path(directory)
    data, records = score67._read_data(directory)
    matches = json.loads((directory / 'magnitude_matches.json').read_text(encoding='utf-8'))
    prompt_count = len(data['prompts'])
    baselines = {target: {p: records[('unchanged', p, target)]['positions'] for p in range(prompt_count)}
                 for target in TARGETS}
    rows = []
    for upstream, downstream in score67.PAIRS:
        layer, unit = upstream
        target_condition = f'zero_{layer}_{unit}'
        delta = score67._normalized_effect(records, target_condition, downstream, prompt_count, baselines[downstream])
        null = []
        for match in matches[f'{layer}_{unit}']['units']:
            condition = f"mag_{layer}_{match['unit']}"
            null.append(score67._normalized_effect(records, condition, downstream, prompt_count, baselines[downstream]))
        rows.append({'upstream': list(upstream), 'downstream': list(downstream), 'delta': delta,
                     'max_matched_abs_delta': max(map(abs, null)), 'p': empirical_p(delta, null),
                     'survives': survives(delta, null)})
    result = {'pairs': rows, 'magnitude_matches': matches}
    _write_json(directory / 'score67b.json', result)
    lines = ['# Part 67b magnitude-matched score', '',
             '| Pair | Delta | Max matched | p | Survives |', '|---|---:|---:|---:|:---:|']
    for row in rows:
        lines.append(f"| L{row['upstream'][0]} N{row['upstream'][1]} → L{row['downstream'][0]} N{row['downstream'][1]} | "
                     f"{row['delta']:.6g} | {row['max_matched_abs_delta']:.6g} | {row['p']:.6g} | "
                     f"{'yes' if row['survives'] else 'no'} |")
    lines.extend(['', '## Upstream target and matched-unit magnitudes', '',
                  '| Upstream target | Target mean absolute activation | Matched range |',
                  '|---|---:|---:|'])
    for key in (f'{layer}_{unit}' for layer, unit in UPSTREAM):
        match = matches[key]
        low, high = match['magnitude_range']
        lines.append(f"| L{match['layer']} N{match['target']} | {match['target_magnitude']:.8g} | [{low:.8g}, {high:.8g}] |")
    (directory / 'score67b.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=['xpu', 'cuda', 'cpu'], default='xpu')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--max-gpu-gb', type=float, default=20)
    parser.add_argument('--score', action='store_true', help='score an existing output directory without loading the model')
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('--batch must be positive')
    if args.score:
        score(args.out)
    else:
        run(args)


if __name__ == '__main__':
    main()
