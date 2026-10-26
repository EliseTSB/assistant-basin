#!/usr/bin/env python3
"""Part 69: measure answer readout changes from zeroing maker neurons and drivers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Bind the namespace package before adding the module directory, as in Part 68b.
import part67  # noqa: E402,F401
if str(ROOT / 'part67') not in sys.path:
    sys.path.append(str(ROOT / 'part67'))
import part67b  # noqa: E402

MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
MAKERS = [(6, 753), (7, 1169), (11, 8868), (29, 776)]
L11_DRIVERS = [(10, 4133), (10, 54), (7, 296), (10, 3551), (10, 8663), (3, 4169), (9, 1740)]
L29_DRIVERS = [(27, 3325), (24, 3949)]
MAKER_NAMES = ['Alibaba', 'Google', 'OpenAI', 'Anthropic', 'DeepSeek', 'Meta', 'Microsoft']

def _runtime():
    import torch
    import part25
    import neuron_report as nr
    import part66
    return torch, part25, nr, part66

def _json_units(units):
    return [{'layer': int(layer), 'unit': int(unit)} for layer, unit in units]

def draw_driver_nulls(matched: dict, count: int = 50) -> dict[str, list[list[tuple[int, int]]]]:
    """Draw one unit from each driver's matched pool, reproducibly per family and set."""
    import numpy as np
    result = {}
    # Family numbers follow the registered condition numbers: 5 = L11-down, 6 = L29-up.
    for family, target, drivers in ((5, '11_8868', L11_DRIVERS), (6, '29_776', L29_DRIVERS)):
        pools = []
        for driver in drivers:
            key = f'{driver[0]}_{driver[1]}'
            records = matched[target].get(key)
            if not records:
                raise ValueError(f'missing matched candidates for {target}/{key}')
            pool = [(int(r['layer']), int(r['unit'])) for r in records]
            pools.append(pool)
        sets = []
        for k in range(count):
            rng = np.random.default_rng(6900 + 100 * family + k)
            sets.append([pool[int(rng.integers(0, len(pool)))] for pool in pools])
        result[f'null_{"l11down" if family == 5 else "l29up"}_k'] = sets
    return result

def make_conditions(matched: dict, magnitudes: dict | None = None, count: int = 50) -> list[dict]:
    conditions = [
        {'id': 'unchanged', 'set_units': [], 'kind': 'baseline'},
        {'id': 'makers', 'set_units': MAKERS, 'kind': 'makers'},
        {'id': 'l11_alone', 'set_units': [MAKERS[2]], 'kind': 'target'},
        {'id': 'l29_alone', 'set_units': [MAKERS[3]], 'kind': 'target'},
        {'id': 'l11_down', 'set_units': L11_DRIVERS, 'kind': 'drivers'},
        {'id': 'l29_up', 'set_units': L29_DRIVERS, 'kind': 'drivers'},
    ]
    if magnitudes is not None:
        excluded = set(MAKERS + L11_DRIVERS + L29_DRIVERS)
        for layer, target, ident in ((11, 8868, 'l11'), (29, 776, 'l29')):
            mags = magnitudes[str(layer)]
            target_mag = float(mags[target])
            candidates = sorted((u for u in range(len(mags)) if (layer, u) not in excluded),
                                key=lambda u: (abs(float(mags[u]) - target_mag), u))[:count]
            conditions.extend({'id': f'null_{ident}_k{k}', 'set_units': [(layer, u)], 'kind': 'null_single'}
                              for k, u in enumerate(candidates))
    driver_nulls = draw_driver_nulls(matched, count)
    for prefix, key in (('null_l11down_k', 'null_l11down_k'), ('null_l29up_k', 'null_l29up_k')):
        conditions.extend({'id': f'{prefix}{k}', 'set_units': units, 'kind': 'null_driver'}
                          for k, units in enumerate(driver_nulls[key]))
    return conditions

def run_69(model, layers, prompt_id_lists, readout_ids, conditions, out_dir, batch=32,
           spill_guard=None, device='cpu'):
    """Run all row interventions on supplied token sequences; suitable for tiny test models."""
    torch, part25, nr, _ = _runtime()
    if batch < 1:
        raise ValueError('batch must be positive')
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    target_ids = [int(x) for x in readout_ids]
    if len(target_ids) != 7 or len(set(target_ids)) != 7:
        raise ValueError('readout_ids must contain seven distinct token ids')
    ids_by_condition = {c['id']: c for c in conditions}
    if len(ids_by_condition) != len(conditions):
        raise ValueError('condition ids must be unique')
    ops = [part25._operations(c['set_units'], 'zero') for c in conditions]
    records = []
    started = time.time()
    for qi, prompt in enumerate(prompt_id_lists):
        for begin in range(0, len(conditions), batch):
            chunk = conditions[begin:begin + batch]
            input_ids = torch.tensor([list(prompt)] * len(chunk), dtype=torch.long,
                                     device=part25.actual_device(model))
            row_ops = [part25._operations(c['set_units'], 'zero') for c in chunk]
            handles = part25.install_intervention_hooks(layers, row_ops)
            try:
                with torch.no_grad():
                    result = model(input_ids=input_ids)
                    logits = result.logits[:, -1, :].float()
                    selected = logits[:, target_ids]
                    masses = selected.softmax(dim=-1)
                    probs = logits.softmax(dim=-1)[:, target_ids].sum(dim=-1)
            finally:
                for handle in handles: handle.remove()
            for ri, cond in enumerate(chunk):
                values = selected[ri].detach().cpu().tolist()
                records.append({'question_index': qi, 'condition': cond['id'], 'readout_logits': values,
                                'maker_mass': float(probs[ri].item()),
                                'maker_softmax': masses[ri].detach().cpu().tolist(),
                                'g': float((selected[ri, 1] - selected[ri, 0]).item())})
        if spill_guard is not None:
            spill_guard(device, 14.0)
    (out / 'records.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    wall = time.time() - started
    try:
        sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        sha = None
    peak = None
    if device == 'cuda' and torch.cuda.is_available(): peak = float(torch.cuda.max_memory_allocated() / 2**30)
    elif device == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available(): peak = float(torch.xpu.max_memory_allocated() / 2**30)
    run = {'model': MODEL, 'revision': REVISION, 'tool_sha256': sha, 'device': str(device),
           'wall_time_seconds': wall, 'peak_memory_gb': peak, 'prompt_count': len(prompt_id_lists),
           'condition_count': len(conditions), 'record_count': len(records), 'batch': batch}
    (out / 'run.json').write_text(json.dumps(run, indent=2) + '\n', encoding='utf-8')
    return records

def _measure_magnitudes(model, layers, prompts, device):
    torch, part25, _, _ = _runtime()
    sums = {11: None, 29: None}; counts = {11: 0, 29: 0}
    handles = []
    for li in sums:
        def hook(_module, args, li=li):
            # Mean across positions for this prompt, then average these prompt means.
            x = args[0].detach().float().abs().mean(dim=(0, 1)).cpu()
            sums[li] = x if sums[li] is None else sums[li] + x
            counts[li] += 1
        handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for prompt in prompts:
                ids = torch.tensor(prompt, dtype=torch.long, device=part25.actual_device(model)).unsqueeze(0)
                model(input_ids=ids)
    finally:
        for h in handles: h.remove()
    return {str(li): (sums[li] / counts[li]).tolist() for li in sums}

def main():
    # Import paths and the Part 67b module before argparse, so --help checks import resolution.
    try:
        _runtime()
    except ModuleNotFoundError as exc:
        # Help remains available on selection-only hosts without PyTorch; all other
        # import failures are real packaging problems and must remain visible.
        if exc.name != 'torch':
            raise
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--matched', default='results/part68/b/matched.json')
    parser.add_argument('--device', choices=['xpu', 'cuda', 'cpu'], default='xpu')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--max-gpu-gb', type=float, default=14)
    args = parser.parse_args()
    if args.batch < 1: parser.error('--batch must be positive')
    torch, part25, nr, part66 = _runtime()
    tok, model = nr.load_model(MODEL, REVISION, args.device)
    _, layers, _ = nr.find_text_layers(model)
    qs = part66.DIRECT + part66.INDIRECT + part66.POSITIVE
    prompts = [part25._token_ids(tok, nr.build_prompt(tok, q, False) + 'I was developed by') for q in qs]
    readout_ids = [part25._token_ids(tok, ' ' + name)[0] for name in MAKER_NAMES]
    if len(set(readout_ids)) != 7: raise ValueError('maker names share a first token')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    matched = json.loads(Path(args.matched).read_text(encoding='utf-8'))
    magnitudes = _measure_magnitudes(model, layers, prompts, args.device)
    (out / 'magnitudes.json').write_text(json.dumps(magnitudes, indent=2) + '\n', encoding='utf-8')
    conditions = make_conditions(matched, magnitudes)
    # Original Part 66 nulls use its registered random_sets seeds and exact ordering.
    conditions += [{'id': f'null_makers_k{k}', 'set_units': units, 'kind': 'null_makers', 'seed': seed}
                   for k, (seed, units) in enumerate(zip(
                       range(6600, 6650),
                       part66.random_sets(layers[0].mlp.down_proj.in_features, len(layers))))]
    (out / 'conditions.json').write_text(json.dumps([{'id': c['id'], 'units': _json_units(c['set_units'])}
                                                       for c in conditions], indent=2) + '\n', encoding='utf-8')
    def guard(dev, limit): return nr.spill_guard(dev, args.max_gpu_gb)
    run_69(model, layers, prompts, readout_ids, conditions, out, args.batch, guard, args.device)
    # Include the positive-control index explicitly in the run manifest.
    manifest_path = out / 'run.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest.update({'model': MODEL, 'revision': REVISION, 'positive_control_index': 24,
                     'heldout_question_count': 24, 'readout_names': MAKER_NAMES,
                     'random_maker_null_seeds': list(range(6600, 6650)),
                     'driver_null_families': {'l11_down': 5, 'l29_up': 6}})
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
