#!/usr/bin/env python3
"""Host-side Part 158 gates, registered predictions, and arm tables."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
from part157 import run157  # noqa: E402

ARMS = ('plain', 'zero_L27N3325', 'zero_L27N4991', 'random_control')


def load_jsonl(path):
    with Path(path).open(encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def rows_for(rows, arm):
    return sorted((r for r in rows if r.get('arm') == arm), key=lambda r: int(r.get('question_index', -1)))


def names_family(answer, family):
    text = str(answer).casefold()
    return any(spelling.casefold() in text for spelling in run157.FAMILIES[family])


def gate_results(rows, level0):
    missing_or_empty = []
    for arm in ARMS:
        cell = rows_for(rows, arm)
        present = {int(r['question_index']) for r in cell}
        if len(cell) != 24:
            missing_or_empty.append({'arm': arm, 'reason': 'expected exactly 24 rows', 'actual': len(cell)})
        if len(present) != len(cell):
            missing_or_empty.append({'arm': arm, 'reason': 'duplicate question indices'})
        for qi in range(24):
            if qi not in present:
                missing_or_empty.append({'arm': arm, 'question_index': qi, 'reason': 'missing'})
        for row in cell:
            if not str(row.get('answer', '')).strip():
                missing_or_empty.append({'arm': arm, 'question_index': row.get('question_index'), 'reason': 'empty'})
    gate1 = not missing_or_empty
    base = {int(r['question_index']): str(r.get('answer', '')) for r in level0}
    control = {int(r['question_index']): str(r.get('answer', '')) for r in rows_for(rows, 'random_control')}
    matches = sum(control.get(qi) == answer for qi, answer in base.items()) if len(base) == 24 else 0
    gate2 = len(base) == 24 and matches >= 22
    alibaba_spellings = list(run157.FAMILIES['Alibaba']) + ['tongy', 'alipay', 'alibab']
    plain = rows_for(rows, 'plain')
    leaks = [{'question_index': r.get('question_index'),
              'spellings': [s for s in alibaba_spellings if s.casefold() in str(r.get('answer', '')).casefold()]}
             for r in plain if any(s.casefold() in str(r.get('answer', '')).casefold() for s in alibaba_spellings)]
    gate3 = len(plain) == 24 and len(leaks) <= 2
    return {
        'gate1': {'pass': gate1, 'rule': 'all 96 planned rows present and non-empty', 'missing_or_empty': missing_or_empty},
        'gate2': {'pass': gate2, 'rule': 'random ban exactly matches Part 157 level 0 on at least 22 of 24',
                  'matches': matches, 'n': len(base)},
        'gate3': {'pass': gate3, 'rule': 'at most 2 of 24 plain answers contain a Part 157 Alibaba spelling, tongy, alipay, or alibab',
                  'leaks': leaks, 'n': len(plain)},
    }


def prediction_results(rows, gates):
    plain = rows_for(rows, 'plain')
    first = Counter(run157.first_named_family(str(r.get('answer', ''))) for r in plain)
    first.pop(None, None)
    p1 = bool(first) and first['Google'] > max((count for family, count in first.items() if family != 'Google'), default=0)
    named = sum(run157.first_named_family(str(r.get('answer', ''))) is not None for r in plain)
    p2 = named <= 8
    base_google = sum(names_family(r.get('answer', ''), 'Google') for r in plain)
    google_3325 = sum(names_family(r.get('answer', ''), 'Google') for r in rows_for(rows, 'zero_L27N3325'))
    google_4991 = sum(names_family(r.get('answer', ''), 'Google') for r in rows_for(rows, 'zero_L27N4991'))
    p3 = None if base_google < 6 else (2 * google_3325 <= base_google and 4 * google_4991 >= 3 * base_google)
    all_gates = all(item['pass'] for item in gates.values())
    def verdict(value):
        if value is None or not all_gates:
            return 'untestable'
        return 'held' if value else 'failed'
    predictions = {
        '158.1': {'prediction': 'In the plain arm, Google is the family named first on more phrasings than any other family',
                  'value': bool(p1), 'verdict': verdict(bool(p1)), 'confidence': 'low'},
        '158.2': {'prediction': 'In the plain arm, at most 8 of 24 answers name any listed family, the rest inventing a maker or naming none',
                  'value': bool(p2), 'verdict': verdict(bool(p2)), 'confidence': 'medium'},
        '158.3': {'prediction': 'Google answers with L27 N3325 zeroed are at most half the plain arm\'s, and with L27 N4991 zeroed at least 75%, if the plain arm has at least 6',
                  'value': p3, 'verdict': verdict(p3), 'confidence': 'medium'},
    }
    return predictions, {'plain_first_named': dict(first), 'plain_any_google': base_google,
                         'zero_L27N3325_any_google': google_3325,
                         'zero_L27N4991_any_google': google_4991,
                         'plain_any_listed_family': named}


def tables(rows):
    out = ['| Arm | ' + ' | '.join(run157.FAMILY_ORDER) + ' | No family | N |',
           '|---|' + '|'.join(['---:'] * (len(run157.FAMILY_ORDER) + 2)) + '|']
    for arm in ARMS:
        cell = rows_for(rows, arm)
        counts = Counter(run157.first_named_family(str(r.get('answer', ''))) for r in cell)
        out.append('| ' + arm + ' | ' + ' | '.join(str(counts.get(family, 0)) for family in run157.FAMILY_ORDER) +
                   f" | {counts.get(None, 0)} | {len(cell)} |")
    return '\n'.join(out) + '\n'


def score(answers_path, level0_path):
    rows, level0_all = load_jsonl(answers_path), load_jsonl(level0_path)
    level0 = [r for r in level0_all if r.get('arm') == 'level_0' and int(r.get('level', 0)) == 0]
    gates = gate_results(rows, level0)
    predictions, values = prediction_results(rows, gates)
    return {'part': 158, 'gates': gates, 'predictions': predictions, 'values': values,
            'arm_first_named_family_table': tables(rows)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--answers', required=True)
    ap.add_argument('--level0', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    result = score(args.answers, args.level0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'score.json').open('w', encoding='utf-8', newline='\n') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
        fh.write('\n')
    with (out / 'tables.md').open('w', encoding='utf-8', newline='\n') as fh:
        fh.write('# Part 158 arm by first-named family\n\n' + result['arm_first_named_family_table'])
    print(json.dumps({'gates': {k: v['pass'] for k, v in result['gates'].items()},
                      'predictions': {k: v['verdict'] for k, v in result['predictions'].items()}}, ensure_ascii=True))
    return 3 if not all(gate['pass'] for gate in result['gates'].values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
