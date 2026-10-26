#!/usr/bin/env python3
"""Part 73 sampled generation: L27 N3325 and L24 N3949 zeroed apart on top of the makers (PART73-DESIGN.md).
Runs Part 70's code path unchanged with Part 73's conditions. Claude, 2026-10-26."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part70 import part70  # noqa: E402

L27 = (27, 3325)
L24 = (24, 3949)
NULL_N = 20


def build_conditions(matched: dict) -> list[dict]:
    """Three main conditions, then 20 single-unit nulls per layer taken from Part 70's own pair draws."""
    target = list(part70.TARGET)
    conditions = [
        {'id': 'target', 'set_units': target, 'kind': 'target'},
        {'id': 'target_plus_l27', 'set_units': target + [L27], 'kind': 'target_plus_l27'},
        {'id': 'target_plus_l24', 'set_units': target + [L24], 'kind': 'target_plus_l24'},
    ]
    for name, index in (('l27', 0), ('l24', 1)):
        for k in range(NULL_N):
            unit = part70.draw_matched_units(matched, k)[index]
            conditions.append({'id': f'target_plus_{name}_matched_{k}', 'set_units': target + [unit],
                               'kind': 'matched', 'matched_index': k, 'seed': 7000 + k})
    return conditions


def prepare_plan(matched_path: str | Path):
    matched = json.loads(Path(matched_path).read_text(encoding='utf-8'))
    conditions = build_conditions(matched)
    return conditions, part70.generations(conditions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--matched', default='results/part68/b/matched.json')
    parser.add_argument('--device', choices=['xpu', 'cuda', 'cpu'], default='xpu')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--max-gpu-gb', type=float, default=14)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('--batch must be positive')

    p66 = part70._require_part66()
    import neuron_report as nr
    if not all((p66.row_seed, p66._hooks, p66._pad, nr.load_model, nr.find_text_layers, nr.spill_guard)):
        raise RuntimeError('a required Part 66 or neuron_report entry point is unavailable')
    conditions, rows = prepare_plan(args.matched)
    if len(rows) != 1680:
        raise RuntimeError(f'planned {len(rows)} answers, the design registers 1,680')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'conditions.json').write_text(json.dumps([
        {'id': c['id'], 'set_units': part70._json_units(c['set_units'])} for c in conditions
    ], indent=2) + '\n', encoding='utf-8')
    tok, model = nr.load_model(part70.MODEL, part70.REVISION, args.device)
    _, layers, _ = nr.find_text_layers(model)
    n_units = layers[0].mlp.down_proj.in_features
    if any(layer < 0 or layer >= len(layers) or unit < 0 or unit >= n_units
           for c in conditions for layer, unit in c['set_units']):
        raise ValueError('a planned neuron is outside the loaded model dimensions')

    def guard(dev, _limit):
        return nr.spill_guard(dev, args.max_gpu_gb)
    part70.run_70(model, tok, layers, conditions, out, args.batch, args.device, guard)
    run_path = out / 'run.json'
    run = json.loads(run_path.read_text(encoding='utf-8'))
    run['part73_tool_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run_path.write_text(json.dumps(run, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
