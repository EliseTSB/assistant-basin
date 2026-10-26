#!/usr/bin/env python3
"""Score Part 68b activity-matched null results."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score68 import normalized_delta

TOLERANCE = 0.005
EARLY = '11_8868'
LATE = '29_776'


def survives(delta: float, matched_max: float, attribution: float) -> bool:
    """Survival requires strict magnitude superiority and the predicted nonzero sign."""
    return (attribution != 0 and delta != 0 and abs(delta) > matched_max
            and ((delta > 0) == (attribution > 0)))


def within_tolerance(delta: float, reference: float) -> bool:
    return abs(float(delta) - float(reference)) <= TOLERANCE


def verdicts(rows_by_target: dict, deltas_within_tolerance: bool) -> dict:
    early = rows_by_target.get(EARLY, [])
    late = rows_by_target.get(LATE, [])
    by_id = {(r['layer'], r['unit']): r for r in early}
    by_late_id = {(r['layer'], r['unit']): r for r in late}
    p87 = sum(bool(r['survives']) for r in early) >= 5 and all(
        by_id.get(pair, {}).get('survives', False) for pair in ((6, 753), (7, 1169)))
    p88 = (all(by_late_id.get(pair, {}).get('survives', False) for pair in ((27, 3325), (24, 3949)))
           and sum(bool(r['survives']) for r in late if (r['layer'], r['unit']) not in ((27, 3325), (24, 3949))) <= 1)
    p89 = (6, 159) in by_late_id and not by_late_id[(6, 159)]['survives']
    p810 = bool(deltas_within_tolerance)
    predictions = {
        '68.7': {'holds': p87, 'rule': 'at least 5 L11 N8868 candidates survive, including L6 N753 and L7 N1169'},
        '68.8': {'holds': p88, 'rule': 'L27 N3325 and L24 N3949 survive, and at most 1 other L29 N776 candidate survives'},
        '68.9': {'holds': p89, 'rule': 'L6 N159 does not survive for L29 N776'},
        '68.10': {'holds': p810, 'rule': 'every re-measured candidate Delta is within 0.005 of its Part 68 test-half value'},
    }
    for key, item in predictions.items():
        item['status'] = ('HOLDS' if item['holds'] else 'FAILS') if key == '68.10' or p810 else 'NOT READ'
    return predictions


def _record_deltas(activations: dict, target: list[int]) -> tuple[dict, dict]:
    key = tuple(target)
    rows = [r for r in activations['records'] if tuple(r['target']) == key and r['half'] == 'test']
    baseline_positions = {int(r['prompt_index']): r['positions'] for r in rows if r['condition'] == 'unchanged'}
    baseline_means = {int(r['prompt_index']): float(r['mean']) for r in rows if r['condition'] == 'unchanged'}
    if not baseline_positions:
        raise ValueError(f'no unchanged test-half rows for target {target}')
    deltas = {}
    for condition in sorted({r['condition'] for r in rows}):
        if condition in ('unchanged', 'zero_nothing'):
            continue
        condition_rows = {int(r['prompt_index']): float(r['mean']) for r in rows if r['condition'] == condition}
        indices = sorted(set(baseline_means) & set(condition_rows))
        if len(indices) != len(baseline_means):
            raise ValueError(f'incomplete prompt rows for {condition}, target {target}')
        deltas[condition] = normalized_delta([condition_rows[i] for i in indices],
                                              [baseline_means[i] for i in indices], baseline_positions)
    return deltas, baseline_positions


def score(directory: str | Path, part68_dir: str | Path = 'results/part68/full') -> dict:
    directory, part68_dir = Path(directory), Path(part68_dir)
    activations = json.loads((directory / 'activations_b.json').read_text(encoding='utf-8'))
    original = json.loads((part68_dir / 'score.json').read_text(encoding='utf-8'))
    targets = activations['targets']
    result = {'targets': {}, 'predictions': {}}
    within = True
    for target in targets:
        key = f'{target[0]}_{target[1]}'
        candidates = activations['candidates'][key]
        deltas, _ = _record_deltas(activations, target)
        matched_by_candidate = activations['matched'][key]
        rows = []
        source = original['targets'].get(key, {}).get('halves', {}).get('test', {}).get('replicating', [])
        source_delta = {(int(c['layer']), int(c['unit'])): float(c['delta']) for c in source}
        current_ids = {(int(c['layer']), int(c['unit'])) for c in candidates}
        if current_ids != set(source_delta):
            within = False
        for candidate in candidates:
            layer, unit = int(candidate['layer']), int(candidate['unit'])
            cid = f'cand_{layer}_{unit}'
            delta = float(deltas[cid])
            matched_ids = [int(m['unit']) for m in matched_by_candidate[f'{layer}_{unit}']]
            matched_deltas = [float(deltas[f'matched_{layer}_{m}']) for m in matched_ids]
            matched_max = max((abs(v) for v in matched_deltas), default=0.0)
            src = source_delta.get((layer, unit))
            if src is None:
                within = False
                diff = None
            else:
                diff = abs(delta - src)
                within = within and within_tolerance(delta, src)
            row = {'layer': layer, 'unit': unit, 'attribution': float(candidate['attribution']),
                   'delta': delta, 'matched_max_abs_delta': matched_max,
                   'matched_count': len(matched_ids), 'survives': survives(delta, matched_max, float(candidate['attribution'])),
                   'part68_delta': src, 'part68_abs_difference': diff}
            rows.append(row)
        result['targets'][key] = {'candidates': rows, 'surviving_count': sum(r['survives'] for r in rows)}
    rows_by_target = {key: value['candidates'] for key, value in result['targets'].items()}
    predictions = verdicts(rows_by_target, within)
    result['predictions'] = predictions
    result['reading'] = {'readable': within,
                         'note': None if within else '68.10 failed; predictions 68.7-68.9 are NOT READ'}
    result['control_68_10'] = {'tolerance': TOLERANCE, 'holds': within}
    return result


def markdown(result: dict) -> str:
    lines = ['# Part 68b score', '', '| Prediction | Verdict | Rule |', '|---|---|---|']
    for key, value in result['predictions'].items():
        lines.append(f"| {key} | {value['status']} | {value['rule']} |")
    for key, target in result['targets'].items():
        lines += ['', f'## Target {key}', '',
                  '| Layer | Unit | Attribution | Delta | Matched max | Survives |',
                  '|---:|---:|---:|---:|---:|:---:|']
        for row in target['candidates']:
            lines.append(f"| {row['layer']} | {row['unit']} | {row['attribution']:.8g} | "
                          f"{row['delta']:.8g} | {row['matched_max_abs_delta']:.8g} | "
                          f"{'yes' if row['survives'] else 'no'} |")
    if not result['reading']['readable']:
        lines += ['', '**68.7–68.9: NOT READ.** 68.10 failed; the rig must be checked.']
    return '\n'.join(lines) + '\n'


def score_dir(directory: str | Path, part68_dir: str | Path = 'results/part68/full') -> dict:
    directory = Path(directory)
    result = score(directory, part68_dir)
    (directory / 'score68b.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    (directory / 'score68b.md').write_text(markdown(result), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--part68', default='results/part68/full')
    args = parser.parse_args()
    score_dir(args.dir, args.part68)


if __name__ == '__main__':
    main()
