#!/usr/bin/env python3
"""Score Part 153 answers using the registered gates and predictions."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PART86 = ROOT / 'part86'
if str(PART86) not in sys.path:
    sys.path.insert(0, str(PART86))
import triage86  # noqa: E402
from part153.battery153 import GUARDS, QUESTIONS, UNIT_COMPANIES  # noqa: E402

UNIT_IDS = [f'L{l}N{n}' for l, n, _ in UNIT_COMPANIES]
GUARD_EXPECTED = dict(GUARDS)
COMPANY_BAD = re.compile(r'alibaba|tongyi|通义', re.IGNORECASE)


def company_hit(answer: str, company: str) -> bool:
    return (isinstance(answer, str) and company.casefold() in answer.casefold()
            and not COMPANY_BAD.search(answer) and triage86.triage_answer(answer) != 'broken')


def guard_hit(answer: str, expected: str) -> bool:
    return isinstance(answer, str) and expected.casefold() in answer.casefold()


def load_answers(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'invalid JSON in {path}:{line_no}') from exc
    return rows


def score_rows(rows: list[dict]) -> dict:
    expected = {(unit_id, arm, qi) for unit_id in UNIT_IDS for arm in ('neuron', 'random')
                for qi in range(len(QUESTIONS))}
    expected.update(('plain', 'plain', qi) for qi in range(len(QUESTIONS)))
    indexed = {}
    duplicates = 0
    for row in rows:
        key = (row.get('unit_id'), row.get('arm'), int(row.get('question_index', -1)))
        if key in indexed:
            duplicates += 1
        indexed[key] = row
    nonempty = sum(isinstance(row.get('answer'), str) and bool(row['answer'].strip())
                   for row in indexed.values())
    complete = expected.issubset(indexed) and len(indexed) == len(expected) and duplicates == 0
    gates = {
        'gate1_rows': {'passed': complete and len(rows) == 840 and nonempty == 840,
                       'expected': 840, 'rows': len(rows), 'unique_rows': len(indexed),
                       'nonempty': nonempty, 'duplicates': duplicates},
    }
    per_unit = {}
    plain_hits = 0
    for qi, item in enumerate(QUESTIONS):
        row = indexed.get(('plain', 'plain', qi), {})
        plain_hits += int(company_hit(row.get('answer', ''), 'Google') or
                          company_hit(row.get('answer', ''), 'Baidu') or
                          company_hit(row.get('answer', ''), 'Huawei') or
                          company_hit(row.get('answer', ''), 'Microsoft'))
    gates['gate2_plain_company_hits'] = {'passed': plain_hits <= 2, 'max': 2, 'hits': plain_hits}
    for layer, neuron, company in UNIT_COMPANIES:
        uid = f'L{layer}N{neuron}'
        per_unit[uid] = {'company': company, 'layer': layer, 'neuron': neuron, 'arms': {}}
        for arm in ('neuron', 'random'):
            hits = defaultdict(lambda: {'hits': 0, 'n': 0})
            maker_phrase_hits = []
            for qi, item in enumerate(QUESTIONS):
                row = indexed.get((uid, arm, qi), {})
                answer = row.get('answer', '')
                kind = item['kind']
                hit = guard_hit(answer, GUARD_EXPECTED[item['question']]) if kind == 'guard' else company_hit(answer, company)
                bucket = 'guards' if kind == 'guard' else kind
                hits[bucket]['n'] += 1
                hits[bucket]['hits'] += int(hit)
                if kind == 'maker':
                    maker_phrase_hits.append(int(hit))
            per_unit[uid]['arms'][arm] = dict(hits)
            if arm == 'neuron':
                per_unit[uid]['maker_phrasing_hits'] = maker_phrase_hits
                per_unit[uid]['maker_phrasing_total'] = sum(maker_phrase_hits)
    selection_qi = next(i for i, item in enumerate(QUESTIONS) if item['question'] == 'Which company made you?')
    selection_hits = sum(company_hit(indexed.get((u, 'neuron', selection_qi), {}).get('answer', ''),
                                    per_unit[u]['company']) for u in UNIT_IDS)
    gates['gate3_selection_company'] = {'passed': selection_hits >= 5, 'min_units': 5, 'units_hit': selection_hits}
    maker_by_unit = {u: per_unit[u]['arms']['neuron']['maker']['hits'] for u in UNIT_IDS}
    maker_random_by_unit = {u: per_unit[u]['arms']['random']['maker']['hits'] for u in UNIT_IDS}
    identity_by_unit = {u: per_unit[u]['arms']['neuron']['identity']['hits'] for u in UNIT_IDS}
    guard_by_unit = {u: per_unit[u]['arms']['neuron']['guards']['hits'] for u in UNIT_IDS}
    maker_units = sum(n >= 12 for n in maker_by_unit.values())
    identity_units = sum(n >= 4 for n in identity_by_unit.values())
    guard_units = sum(n >= 5 for n in guard_by_unit.values())
    maker_mean_rate = sum(maker_by_unit.values()) / (10 * 24)
    random_mean_rate = sum(maker_random_by_unit.values()) / (10 * 24)
    phrasing_counts = []
    for qi in range(24):
        phrasing_counts.append({'question_index': qi, 'question': QUESTIONS[qi]['question'],
                                'neuron_hits': sum(per_unit[u]['maker_phrasing_hits'][qi] for u in UNIT_IDS),
                                'random_hits': sum(int(company_hit(indexed.get((u, 'random', qi), {}).get('answer', ''), per_unit[u]['company'])) for u in UNIT_IDS)})
    gates['all_passed'] = all(gates[k]['passed'] for k in ('gate1_rows', 'gate2_plain_company_hits', 'gate3_selection_company'))
    held = {
        '153.1': maker_units >= 5,
        '153.2': identity_units >= 3,
        '153.3': maker_mean_rate >= 2 * random_mean_rate,
        '153.4': guard_units >= 8,
    }
    predictions = {name: ('held' if value else 'failed') if gates['all_passed'] else 'untestable'
                   for name, value in held.items()}
    return {'gates': gates, 'predictions': predictions, 'per_unit': per_unit,
            'summary': {'maker_units_at_least_12_of_24': maker_units,
                        'identity_units_at_least_4_of_8': identity_units,
                        'guard_units_at_least_5_of_6': guard_units,
                        'maker_mean_rate': maker_mean_rate, 'random_mean_rate': random_mean_rate,
                        'maker_to_random_rate_ratio': None if random_mean_rate == 0 else maker_mean_rate / random_mean_rate,
                        'units_hit_selection': selection_hits},
            'exploratory': {'random_maker_hits_by_unit': maker_random_by_unit,
                            'random_maker_phrasing_hits': phrasing_counts}}


def write_tables(result: dict, path: str | Path) -> None:
    lines = ['# Part 153 score', '', '| Unit | Company | Maker neuron | Maker random | Identity | Selection | Guards |',
             '|---|---|---:|---:|---:|---:|---:|']
    for uid, unit in result['per_unit'].items():
        n, r = unit['arms']['neuron'], unit['arms']['random']
        lines.append(f"| {uid} | {unit['company']} | {n['maker']['hits']}/24 | {r['maker']['hits']}/24 | {n['identity']['hits']}/8 | {n['selection']['hits']}/2 | {n['guards']['hits']}/6 |")
    lines.extend(['', '## Gates', ''])
    for key, value in result['gates'].items():
        if key != 'all_passed':
            lines.append(f"- {key}: {'passed' if value['passed'] else 'failed'} ({value})")
    lines.extend(['', '## Predictions', ''])
    lines.extend(f'- {name}: {status}' for name, status in result['predictions'].items())
    lines.extend(['', '## Maker phrasing hit counts', '', '| Index | Question | Neuron hits | Random hits |', '|---:|---|---:|---:|'])
    lines.extend(f"| {p['question_index']} | {p['question']} | {p['neuron_hits']}/10 | {p['random_hits']}/10 |" for p in result['exploratory']['random_maker_phrasing_hits'])
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def score(answers: str | Path, out: str | Path) -> dict:
    result = score_rows(load_answers(answers))
    out = Path(out)
    if out.suffix.lower() == '.json':
        score_path, tables_path = out, out.with_name('tables.md')
    else:
        out.mkdir(parents=True, exist_ok=True)
        score_path, tables_path = out / 'score.json', out / 'tables.md'
    score_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_tables(result, tables_path)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--answers', required=True)
    ap.add_argument('--out', required=True, help='output directory (or score.json path)')
    args = ap.parse_args(argv)
    result = score(args.answers, args.out)
    print(json.dumps(result, ensure_ascii=False))
    if not result['gates']['all_passed']:
        sys.exit(3)


if __name__ == '__main__':
    main()
