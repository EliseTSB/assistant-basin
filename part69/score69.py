#!/usr/bin/env python3
"""Score the registered Part 69 logit-gap controls and predictions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

def empirical_p(observed: float, null: list[float], direction: str = 'positive') -> float:
    if not null: raise ValueError('null distribution must not be empty')
    if direction == 'positive': hits = sum(x >= observed for x in null)
    elif direction == 'negative': hits = sum(x <= observed for x in null)
    else: raise ValueError("direction must be 'positive' or 'negative'")
    return (1 + hits) / (len(null) + 1)

def two_sided_p(observed: float, null: list[float]) -> float:
    return min(1.0, 2 * min(empirical_p(observed, null, 'positive'),
                             empirical_p(observed, null, 'negative')))

def verdicts(dg: dict[str, float], nulls: dict[str, list[float]], baseline_mass: float,
             baseline_top_alibaba: int) -> dict[str, dict]:
    p1 = empirical_p(dg['makers'], nulls['null_makers'], 'positive')
    c1 = dg['makers'] > 0 and p1 <= 1/51
    c2 = baseline_mass >= 0.5 and baseline_top_alibaba >= 20
    result = {
        '69.1': {'pass': c1, 'verdict': 'PASS' if c1 else 'FAIL', 'p_positive': p1,
                 'criterion': 'makers Dg > 0 and exceeds all 50 random maker sets'},
        '69.2': {'pass': c2, 'verdict': 'PASS' if c2 else 'FAIL', 'mean_mass': baseline_mass,
                 'alibaba_top_count': baseline_top_alibaba,
                 'criterion': 'mean unchanged maker mass >= 0.5 and Alibaba top on >= 20/24'},
    }
    if not c1 or not c2:
        for key in ('69.3', '69.4', '69.5', '69.6'):
            result[key] = {'verdict': 'NOT READ', 'pass': None}
        return result
    p3 = empirical_p(dg['l11_alone'], nulls['null_l11'], 'positive')
    p4 = empirical_p(dg['l29_alone'], nulls['null_l29'], 'positive')
    p5 = two_sided_p(dg['l11_down'], nulls['null_l11down'])
    p6 = two_sided_p(dg['l29_up'], nulls['null_l29up'])
    results = {
        '69.3': (p3 > 0.05, p3, 'L11 alone does not beat activity-matched units in positive direction (p > 0.05)'),
        '69.4': (dg['l29_alone'] > 0 and p4 <= 1/51, p4, 'L29 alone is positive and beats all 50 activity-matched units'),
        '69.5': (p5 > 0.1, p5, 'L11-down two-sided p > 0.1'),
        '69.6': (p6 > 0.1, p6, 'L29-up two-sided p > 0.1'),
    }
    for key, (passed, p, criterion) in results.items():
        result[key] = {'pass': passed, 'verdict': 'PASS' if passed else 'FAIL', 'p': p, 'criterion': criterion}
    return result

def _top_alibaba(row: dict) -> bool:
    logits = row['readout_logits']
    return int(max(range(7), key=lambda i: logits[i])) == 0

def score_directory(directory: str | Path) -> dict:
    root = Path(directory)
    records = json.loads((root / 'records.json').read_text(encoding='utf-8'))
    conditions = json.loads((root / 'conditions.json').read_text(encoding='utf-8'))
    by = {(int(r['question_index']), r['condition']): r for r in records}
    if not records: raise ValueError('records.json is empty')
    n_questions = int(max(r['question_index'] for r in records)) + 1
    if n_questions < 25: raise ValueError('expected 24 held-out prompts plus positive control')
    held = range(24); positive = 24
    ids = {str(c['id']) for c in conditions}
    required = {'unchanged', 'makers', 'l11_alone', 'l29_alone', 'l11_down', 'l29_up'}
    if not required <= ids: raise ValueError(f'missing conditions: {sorted(required - ids)}')
    for qi in held:
        if (qi, 'unchanged') not in by: raise ValueError(f'missing unchanged record for question {qi}')
    g_values = {}
    for ident in ids:
        if ident == 'unchanged': continue
        vals = []
        for qi in held:
            if (qi, ident) not in by: continue
            vals.append(by[(qi, ident)]['g'] - by[(qi, 'unchanged')]['g'])
        if len(vals) == 24: g_values[ident] = sum(vals) / 24
    null_ids = {
        'null_makers': [f'null_makers_k{k}' for k in range(50)],
        'null_l11': [f'null_l11_k{k}' for k in range(50)],
        'null_l29': [f'null_l29_k{k}' for k in range(50)],
        'null_l11down': [f'null_l11down_k{k}' for k in range(50)],
        'null_l29up': [f'null_l29up_k{k}' for k in range(50)],
    }
    missing = [i for group in null_ids.values() for i in group if i not in g_values]
    if missing: raise ValueError(f'missing null condition records: {missing[:5]}')
    nulls = {key: [g_values[x] for x in group] for key, group in null_ids.items()}
    dg = {key: g_values[key] for key in ('makers', 'l11_alone', 'l29_alone', 'l11_down', 'l29_up')}
    base = [by[(qi, 'unchanged')] for qi in held]
    mass = sum(r['maker_mass'] for r in base) / 24
    top_count = sum(_top_alibaba(r) for r in base)
    scores = verdicts(dg, nulls, mass, top_count)
    table = []
    for key, ident, null_key in [('makers', 'makers', 'null_makers'), ('l11_alone', 'l11_alone', 'null_l11'),
                                 ('l29_alone', 'l29_alone', 'null_l29'), ('l11_down', 'l11_down', 'null_l11down'),
                                 ('l29_up', 'l29_up', 'null_l29up')]:
        vals = nulls[null_key]
        p = (empirical_p(dg[key], vals, 'positive') if key in ('makers','l11_alone','l29_alone')
             else two_sided_p(dg[key], vals))
        table.append({'condition': key, 'Dg': dg[key], 'null_mean': sum(vals)/len(vals),
                      'null_min': min(vals), 'null_max': max(vals), 'p': p})
    positive_rows = {r['condition']: r for r in records if int(r['question_index']) == positive}
    answer = {'scores': scores, 'Dg': dg, 'baseline_mean_maker_mass': mass,
              'baseline_alibaba_top_count': top_count,
              'positive_control': positive_rows}
    (root / 'score69.json').write_text(json.dumps(answer, indent=2) + '\n', encoding='utf-8')
    md = ['# Part 69 score', '', '| condition | Dg | null mean | null min | null max | p |',
          '|---|---:|---:|---:|---:|---:|']
    for row in table:
        md.append('| {condition} | {Dg:.8g} | {null_mean:.8g} | {null_min:.8g} | {null_max:.8g} | {p:.6g} |'.format(**row))
    md += ['', '## Verdicts', '', '| prediction | verdict | details |', '|---|---|---|']
    for key, entry in scores.items():
        md.append(f"| {key} | {entry['verdict']} | {entry.get('criterion', '')} |")
    md += ['', f'Unchanged mean seven-maker mass: {mass:.6g}; Alibaba top token: {top_count}/24.',
           'Positive-control question (index 24) is reported separately and is excluded from Dg and verdicts.']
    (root / 'score69.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    return answer

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()
    score_directory(args.dir)

if __name__ == '__main__': main()
