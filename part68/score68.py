#!/usr/bin/env python3
"""Score a completed Part 68 run directory (search + test phases)."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

TARGETS = [(11, 8868), (29, 776)]
MAKERS = [(6, 753), (7, 1169), (11, 8868), (29, 776)]
LATE_TARGET = (29, 776)
EARLIER_MAKERS = [(6, 753), (7, 1169), (11, 8868)]
LATE_BAND = range(20, 29)  # layers 20 to 28 inclusive
TOP_N = 20
NULL_SEED = 6800
OVERLAP_SEED = 6801
OVERLAP_DRAWS = 10000
SPEARMAN_MIN = 0.3
REPLICATION_MIN = 3
ZERO_TOL = 0.0  # 68.5 requires exact zero for every unit at or after the target layer.


def normalized_delta(values: list[float], baseline: list[float],
                     baseline_positions: dict[int, list[float]]) -> float:
    """mean over prompts of (mean over positions of the condition) - (mean over positions of unchanged),
    divided by the standard deviation of the unchanged values over all positions of the half."""
    flat = [v for prompt in sorted(baseline_positions) for v in baseline_positions[prompt]]
    sd = statistics.pstdev(flat)
    if sd == 0:
        raise ValueError('unchanged standard deviation is zero')
    if not values or len(values) != len(baseline):
        raise ValueError('condition and unchanged means must contain the same nonzero prompt count')
    diffs = [v - b for v, b in zip(values, baseline)]
    return (sum(diffs) / len(diffs)) / sd


def percentile_95(values: list[float]) -> float:
    """The 95th percentile by linear interpolation (numpy's default); ties are left in place."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def replicates(delta: float, threshold: float, predicted_sign: float) -> bool:
    """Above the null's 95th percentile in magnitude AND the sign the attribution predicts."""
    if predicted_sign == 0 or abs(delta) <= threshold:
        return False
    return (delta > 0) == (predicted_sign > 0)


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman's rho with average ranks for ties; None when either side has no variation."""
    if len(x) != len(y) or not max(len(x), 2) == len(x):
        return None
    rx, ry = _average_ranks(x), _average_ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while len(order) > i:
        j = i
        while len(order) > j + 1 and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _layer_counts(units: list[tuple[int, int]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for layer, _ in units:
        counts[layer] = counts.get(layer, 0) + 1
    return counts


def expected_overlap(candidates_a: list[tuple[int, int]], candidates_b: list[tuple[int, int]],
                     n_units: int, draws: int = OVERLAP_DRAWS, seed: int = OVERLAP_SEED) -> dict:
    """Overlap of two random 20-unit draws from the layers of A and of B, in A's proportions."""
    rng = np.random.default_rng(seed)
    counts_a, counts_b = _layer_counts(candidates_a), _layer_counts(candidates_b)
    set_a, set_b = set(candidates_a), set(candidates_b)
    overlaps = []
    for _ in range(draws):
        draw_a = {(layer, int(u)) for layer, n in counts_a.items()
                  for u in rng.choice(n_units, size=n, replace=False)}
        draw_b = {(layer, int(u)) for layer, n in counts_b.items()
                  for u in rng.choice(n_units, size=n, replace=False)}
        overlaps.append(len(draw_a & draw_b))
    arr = np.asarray(overlaps)
    observed = len(set_a & set_b)
    return {'observed': observed, 'draws': draws, 'seed': seed,
            'mean': float(arr.mean()), 'p95': float(np.percentile(arr, 95)),
            'share_at_least_observed': float((arr >= observed).mean())}


def _read(directory: Path):
    activations = json.loads((directory / 'activations.json').read_text(encoding='utf-8'))
    targets = [tuple(t) for t in activations.get('targets', TARGETS)]
    search = {}
    for target in targets:
        stem = f'attr_{target[0]}_{target[1]}'
        search[target] = json.loads((directory / f'{stem}.json').read_text(encoding='utf-8'))
        npz = np.load(directory / f'{stem}.npz')
        search[target]['arrays'] = npz['attribution']
    return activations, search


def _deltas(activations: dict, target: list[int], conditions: list[dict]) -> dict[str, dict[str, float]]:
    """Normalized Delta per condition per half, plus the unchanged per-position values per half."""
    key = tuple(target)
    baseline_positions: dict[str, dict[int, list[float]]] = {}
    baseline_means: dict[str, dict[int, float]] = {}
    for row in activations['records']:
        if tuple(row['target']) == key and row['condition'] == 'unchanged':
            baseline_positions.setdefault(row['half'], {})[int(row['prompt_index'])] = row['positions']
            baseline_means.setdefault(row['half'], {})[int(row['prompt_index'])] = float(row['mean'])
    out: dict[str, dict[str, float]] = {}
    for condition in conditions:
        cid = condition['id']
        for half in baseline_positions:
            values, base = [], []
            for row in activations['records']:
                if tuple(row['target']) == key and row['half'] == half and row['condition'] == cid:
                    values.append((int(row['prompt_index']), float(row['mean'])))
            values.sort()
            for prompt_index, value in values:
                base.append(baseline_means[half][prompt_index])
            if not values:
                continue
            out.setdefault(cid, {})[half] = normalized_delta(
                [v for _, v in values], base, baseline_positions[half])
    return out


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if 0 > x else 0)


def score(directory: str | Path) -> dict:
    directory = Path(directory)
    activations, search = _read(directory)
    result: dict = {'targets': {}, 'predictions': {}}
    for target in [tuple(t) for t in activations.get('targets', TARGETS)]:
        key = f'{target[0]}_{target[1]}'
        candidates = [{'layer': int(c['layer']), 'unit': int(c['unit']),
                       'attribution': float(c['attribution'])} for c in search[target]['candidates']]
        nulls = [{'layer': int(c['layer']), 'unit': int(c['unit'])} for c in activations['nulls'][key]]
        cand_ids = [(c['layer'], c['unit']) for c in candidates]
        null_ids = [(c['layer'], c['unit']) for c in nulls]
        conditions = [{'id': 'unchanged'}, {'id': 'all_candidates'}]
        conditions += [{'id': f'cand_{l}_{u}'} for l, u in cand_ids]
        conditions += [{'id': f'null_{l}_{u}'} for l, u in null_ids]
        deltas = _deltas(activations, list(target), conditions)
        present = {row['half'] for row in activations['records'] if tuple(row['target']) == tuple(target)}
        halves = [h for h in ('test', 'search') if h in present]
        per_half = {}
        for half in halves:
            null_deltas = [deltas[f'null_{l}_{u}'][half] for l, u in null_ids]
            threshold = percentile_95([abs(x) for x in null_deltas])
            rows = []
            for c in candidates:
                delta = deltas[f"cand_{c['layer']}_{c['unit']}"][half]
                rows.append({**c, 'delta': delta, 'delta_abs': abs(delta),
                             'predicted_sign': _sign(c['attribution']),
                             'replicates': replicates(delta, threshold, c['attribution'])})
            # 68.6 uses the actual search attribution of all 20 candidates and 200
            # null units. Nulls are controls, not zero-attribution observations.
            array = search[target]['arrays']
            tested_attributions = [float(array[c['layer'], c['unit']]) for c in candidates]
            tested_attributions += [float(array[c['layer'], c['unit']]) for c in nulls]
            tested_deltas = [r['delta'] for r in rows] + null_deltas
            per_half[half] = {
                'null_threshold_abs': threshold,
                'null_count': len(null_deltas),
                'candidates': rows,
                'replicating': [r for r in rows if r['replicates']],
                'replicating_count': sum(r['replicates'] for r in rows),
                'all_candidates_delta': deltas['all_candidates'][half],
                'spearman': spearman(tested_deltas, tested_attributions),
            }
        result['targets'][key] = {
            'target': list(target), 'candidates': candidates, 'nulls': nulls,
            'halves': per_half,
            'replicating_layers': [r['layer'] for r in per_half.get('test', {}).get('replicating', [])],
            'candidate_layer_histogram': {str(layer): sum(c['layer'] == layer for c in candidates)
                                          for layer in sorted({c['layer'] for c in candidates})},
            'replicating_layer_histogram': {str(layer): sum(r['layer'] == layer
                                                            for r in per_half.get('test', {}).get('replicating', []))
                                            for layer in sorted({r['layer'] for r in per_half.get('test', {}).get('replicating', [])})},
            'later_units_max_abs': float(search[target].get('later_units_max_abs', 0.0)),
        }

    if f'{LATE_TARGET[0]}_{LATE_TARGET[1]}' not in result['targets'] or f'{TARGETS[0][0]}_{TARGETS[0][1]}' not in result['targets']:
        # A partial run directory (a smoke test): score the targets it has, and say so.
        result['predictions'] = {}
        result['reading'] = {'readable': False, 'note': 'both registered targets are required for the verdicts'}
        available_halves = sorted({half for obj in result['targets'].values() for half in obj['halves']})
        result['spearman'] = {half: {key: obj['halves'][half]['spearman']
                                     for key, obj in result['targets'].items() if half in obj['halves']}
                              for half in available_halves}
        result['control_68_5'] = {'later_units_max_abs': {key: result['targets'][key]['later_units_max_abs']
                                                          for key in result['targets']},
                                  'tolerance': ZERO_TOL,
                                  'holds': all(result['targets'][key]['later_units_max_abs'] <= ZERO_TOL
                                               for key in result['targets'])}
        return result
    halves = [h for h in ('test', 'search') if h in next(iter(result['targets'].values()))['halves']]
    if 'test' not in halves:
        result['predictions'] = {}
        result['reading'] = {'readable': False, 'note': 'the test half is required for the verdicts'}
        return result
    late = result['targets'][f'{LATE_TARGET[0]}_{LATE_TARGET[1]}']
    late_cands = [(c['layer'], c['unit']) for c in late['candidates']]
    rep_late = [r for r in late['halves']['test']['replicating']]
    late_hist = {str(layer): sum(r['layer'] == layer for r in rep_late) for layer in sorted({r['layer'] for r in rep_late})}
    late_share = (sum(r['layer'] in LATE_BAND for r in rep_late) / len(rep_late)) if rep_late else None
    late_hist_all = {str(layer): sum(c['layer'] == layer for c in late['candidates'])
                     for layer in sorted({c['layer'] for c in late['candidates']})}

    early_key = f'{TARGETS[0][0]}_{TARGETS[0][1]}'
    early = result['targets'][early_key]
    early_cands = [(c['layer'], c['unit']) for c in early['candidates']]
    rep_early = {(r['layer'], r['unit']) for r in early['halves']['test']['replicating']}
    rep_late_set = {(r['layer'], r['unit']) for r in rep_late}
    shared = sorted(rep_early & rep_late_set)
    overlap = expected_overlap(early_cands, late_cands, _n_units(activations))
    shared_detail = [{'layer': l, 'unit': u,
                      'delta_early': next(r['delta'] for r in early['halves']['test']['replicating'] if (r['layer'], r['unit']) == (l, u)),
                      'delta_late': next(r['delta'] for r in rep_late if (r['layer'], r['unit']) == (l, u))}
                     for l, u in shared]

    spearman_search = {key: result['targets'][key]['halves'].get('search', {}).get('spearman') for key in result['targets']}
    spearman_test = {key: result['targets'][key]['halves']['test']['spearman'] for key in result['targets']}
    zero_ok = all(result['targets'][key]['later_units_max_abs'] <= ZERO_TOL for key in result['targets'])
    # The runner also records the maximum |A_u| over units at or after the target layer in the search json;
    # both sources must agree before 68.5 is read.
    zero_json = {}
    for key in result['targets']:
        arrays = search[tuple(result['targets'][key]['target'])]['arrays']
        layer = result['targets'][key]['target'][0]
        zero_json[key] = _later_max_abs(arrays, layer)
    zero_ok = zero_ok and all(zero_json[key] <= ZERO_TOL for key in zero_json)
    rep_counts = {key: result['targets'][key]['halves']['test']['replicating_count'] for key in result['targets']}
    early_maker_in_late = [list(m) for m in EARLIER_MAKERS if tuple(m) in set(late_cands)]

    result['overlap'] = {'candidate_overlap': sorted(set(early_cands) & set(late_cands)),
                         'shared_replicating': [{'layer': l, 'unit': u} for l, u in shared],
                         'shared_replicating_detail': shared_detail, 'expected': overlap}
    result['spearman'] = {'search_half': spearman_search, 'test_half': spearman_test}
    result['control_68_5'] = {'later_units_max_abs': {key: result['targets'][key]['later_units_max_abs']
                                                      for key in result['targets']},
                              'later_units_max_abs_recomputed_from_npz': zero_json,
                              'tolerance': ZERO_TOL, 'holds': zero_ok}

    predictions = {}
    predictions['68.1'] = {'holds': rep_counts[f'{LATE_TARGET[0]}_{LATE_TARGET[1]}'] >= 10,
                           'observed': rep_counts[f'{LATE_TARGET[0]}_{LATE_TARGET[1]}'],
                           'rule': 'at least 10 of 20 candidates for L29 N776 replicate on the test half'}
    predictions['68.2'] = {'holds': rep_counts[early_key] >= 10, 'observed': rep_counts[early_key],
                           'rule': 'at least 10 of 20 candidates for L11 N8868 replicate on the test half'}
    predictions['68.3'] = {'holds': late_share is not None and late_share > 0.5,
                           'observed': {'share_in_layers_20_28': late_share, 'layer_histogram': late_hist,
                                        'candidate_layer_histogram': late_hist_all,
                                        'replicating_count': len(rep_late)},
                           'rule': 'most (more than half) of L29 N776\'s replicating drivers sit in layers 20 to 28'}
    predictions['68.4'] = {'holds': not early_maker_in_late,
                           'observed': {'earlier_makers_in_late_candidates': early_maker_in_late},
                           'rule': 'none of L6 N753, L7 N1169, L11 N8868 is among L29 N776\'s 20 candidates'}
    predictions['68.5'] = {'holds': zero_ok,
                           'observed': result['control_68_5']['later_units_max_abs'],
                           'rule': 'attribution of every unit at or after the target layer is exactly 0'}
    spearman_ok = all(v is not None and v > SPEARMAN_MIN for v in spearman_search.values()) and 'search' in halves
    predictions['68.6'] = {'holds': spearman_ok,
                           'observed': {'search_half_spearman': spearman_search,
                                        'test_half_spearman': spearman_test},
                           'rule': 'on the search half, Delta and the attribution correlate (Spearman above 0.3) '
                                   'across the 220 tested units, for both targets'}
    if not zero_ok:
        for key in ('68.1', '68.2', '68.3', '68.4', '68.6'):
            predictions[key]['status'] = 'NOT READ'
        predictions['68.5']['status'] = 'FAIL'
    else:
        for prediction in predictions.values():
            prediction['status'] = 'HOLDS' if prediction['holds'] else 'FAILS'
    result['predictions'] = predictions
    result['reading'] = _reading(predictions, rep_counts)
    return result


def _later_max_abs(arrays, layer: int) -> float:
    """max |A_u| over rows layer..end; 0.0 when there is no such row. Plain loops, so no numpy slice or
    reduction semantics are involved."""
    best = 0.0
    for row in range(len(arrays)):
        if row >= layer:
            for value in arrays[row].tolist():
                best = max(best, abs(float(value)))
    return best


def _n_units(activations: dict) -> int:
    if 'n_units' in activations:
        return int(activations['n_units'])
    raise SystemExit('activations.json has no n_units; rerun the test phase with this version of part68.py')


def _reading(predictions: dict, rep_counts: dict) -> dict:
    if not predictions['68.5']['holds']:
        return {'readable': False, 'note': '68.5 failed: nothing is read'}
    label = ('the search found drivers' if any(n >= REPLICATION_MIN for n in rep_counts.values())
             else 'the search did not find drivers at this sample')
    out = {'readable': True, 'threshold_replicating_candidates': REPLICATION_MIN, 'label': label,
           'replication_counts': rep_counts}
    if not predictions['68.6']['holds']:
        out['caveat'] = ('68.6 failed: the replication counts test an attribution that does not track zeroing; '
                         'the candidate list is read only as a list')
    return out


def _md(result: dict) -> str:
    lines = ['# Part 68 score', '', '| Prediction | Verdict | Observation | Rule |', '|---|---|---|---|']
    if not result['predictions']:
        lines.append(f"| - | not scored | `{json.dumps(result['reading'])}` | both registered targets are required |")
    for key, obj in result['predictions'].items():
        lines.append(f"| {key} | {obj.get('status', 'HOLDS' if obj['holds'] else 'FAILS')} | "
                     f"`{json.dumps(obj['observed'], ensure_ascii=False)}` | {obj['rule']} |")
    lines += ['', '## Per target (test half)', '',
              '| Target | Replicating | Null p95 | All-20 Delta | Spearman (search) | Spearman (test) |',
              '|---|---|---|---|---|---|']
    for key, obj in result['targets'].items():
        test = obj['halves'].get('test')
        if test is None:
            lines.append(f"| {key} | n/a | n/a | n/a | {obj['halves'].get('search', {}).get('spearman')} | n/a |")
        else:
            lines.append(f"| {key} | {test['replicating_count']}/20 | {test['null_threshold_abs']:.4g} | "
                         f"{test['all_candidates_delta']:.4g} | {obj['halves'].get('search', {}).get('spearman')} | {test['spearman']} |")
    lines += ['', '## Replicating drivers (test half)', '']
    for key, obj in result['targets'].items():
        rows = obj['halves'].get('test', {}).get('replicating', [])
        lines.append(f"### {key}: {len(rows)}")
        if rows:
            lines += ['', '| Layer | Unit | A_u | Delta |', '|---|---|---|---|']
            lines += [f"| {r['layer']} | {r['unit']} | {r['attribution']:.4g} | {r['delta']:.4g} |" for r in rows]
        lines.append('')
    if 'overlap' not in result:
        return '\n'.join(lines) + '\n'
    ov = result['overlap']
    lines += ['## Overlap', '',
              f"Candidate overlap: {ov['candidate_overlap']}. Shared replicating drivers: "
              f"{ov['shared_replicating']}. Two random 20-unit draws from the same layers overlap "
              f"{ov['expected']['mean']:.2f} times on average (95th percentile {ov['expected']['p95']:.0f}, "
              f"{ov['expected']['draws']} draws, seed {ov['expected']['seed']}); share at least the observed "
              f"{ov['expected']['observed']}: {ov['expected']['share_at_least_observed']:.4f}.", '',
              f"Reading: {result['reading'].get('label')}."
              + (f" {result['reading']['caveat']}." if 'caveat' in result['reading'] else ''), '']
    return '\n'.join(lines)


def score_dir(directory: str | Path) -> dict:
    directory = Path(directory)
    result = score(directory)
    (directory / 'score.json').write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    (directory / 'score.md').write_text(_md(result), encoding='utf-8')
    for key, obj in result['predictions'].items():
        print(f"{key}: {'HOLDS' if obj['holds'] else 'FAILS'} ({obj['rule']})")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True)
    score_dir(p.parse_args().dir)


if __name__ == '__main__':
    main()
