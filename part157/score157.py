#!/usr/bin/env python3
"""Host-side Part 157 gates, registered predictions, and answer tables."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
from part157 import run157


def load_jsonl(path):
    with Path(path).open(encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def family_for(answer):
    return run157.first_named_family(answer)


def names_family(answer, family):
    text = answer.casefold()
    return any(spelling.casefold() in text for spelling in run157.FAMILIES[family])


def rows_for(rows, arm, level):
    return sorted((r for r in rows if r.get('arm') == arm and int(r.get('level', -1)) == level),
                  key=lambda r: int(r['question_index']))


def gate_results(rows, ladder):
    levels = ladder.get('levels', [])
    planned = [int(level['level']) for level in levels]
    expected = [('level_0', 0), ('control_ban', 0), ('level_1', 1),
                ('zero_L27N3325', 1), ('zero_L27N4991', 1)]
    expected.extend((f'level_{level}', level) for level in planned if level >= 2)
    missing = []
    if not planned or planned[:2] != [0, 1] or planned != list(range(planned[-1] + 1)):
        missing.append({'reason': 'ladder levels must begin at 0, 1 and be contiguous', 'levels': planned})
    for arm, level in expected:
        cell = rows_for(rows, arm, level)
        present = {int(r['question_index']) for r in cell}
        for qi in range(24):
            if qi not in present:
                missing.append({'arm': arm, 'level': level, 'question_index': qi, 'reason': 'missing'})
        for row in cell:
            if not str(row.get('answer', '')).strip():
                missing.append({'arm': arm, 'level': level, 'question_index': row.get('question_index'), 'reason': 'empty'})
    gate1 = not missing

    level0 = rows_for(rows, 'level_0', 0)
    alibaba_level0 = sum(names_family(r.get('answer', ''), 'Alibaba') for r in level0)
    gate2 = len(level0) == 24 and alibaba_level0 >= 22

    control = rows_for(rows, 'control_ban', 0)
    base_by_q = {int(r['question_index']): r.get('answer', '') for r in level0}
    control_by_q = {int(r['question_index']): r.get('answer', '') for r in control}
    control_matches = len(base_by_q) == 24 and all(control_by_q.get(qi) == answer for qi, answer in base_by_q.items())
    banned = rows_for(rows, 'level_1', 1) + rows_for(rows, 'zero_L27N3325', 1) + rows_for(rows, 'zero_L27N4991', 1)
    banned_spellings = run157.FAMILIES['Alibaba']
    leaks = [{'arm': r['arm'], 'question_index': r['question_index'], 'spellings': [s for s in banned_spellings if s.casefold() in r.get('answer', '').casefold()]}
             for r in banned if any(s.casefold() in r.get('answer', '').casefold() for s in banned_spellings)]
    gate3 = control_matches and not leaks
    return {
        'gate1': {'pass': gate1, 'rule': 'all planned rows present and non-empty', 'missing_or_empty': missing},
        'gate2': {'pass': gate2, 'rule': 'level 0 first-names Alibaba on at least 22 of 24', 'alibaba_first_named': alibaba_level0, 'n': len(level0)},
        'gate3': {'pass': gate3, 'rule': 'control ban exactly matches level 0 and no level 1 arm leaks an Alibaba spelling',
                  'control_matches_level0': control_matches, 'leaks': leaks},
    }


def prediction_results(rows, ladder, gates):
    l1 = rows_for(rows, 'level_1', 1)
    c1 = Counter(family_for(r.get('answer', '')) for r in l1)
    c1.pop(None, None)
    p1 = bool(c1) and c1['Google'] > max((n for fam, n in c1.items() if fam != 'Google'), default=0)
    p2 = sum(family_for(r.get('answer', '')) is not None for r in l1) >= 12
    base_google = sum(names_family(r.get('answer', ''), 'Google') for r in l1)
    n3325 = rows_for(rows, 'zero_L27N3325', 1)
    n4991 = rows_for(rows, 'zero_L27N4991', 1)
    google_3325 = sum(names_family(r.get('answer', ''), 'Google') for r in n3325)
    google_4991 = sum(names_family(r.get('answer', ''), 'Google') for r in n4991)
    p3 = base_google > 0 and 2 * google_3325 <= base_google and 4 * google_4991 >= 3 * base_google
    l2 = rows_for(rows, 'level_2', 2)
    c2 = Counter(family_for(r.get('answer', '')) for r in l2)
    c2.pop(None, None)
    p4 = bool(c2) and c2['Microsoft'] > max((n for fam, n in c2.items() if fam != 'Microsoft'), default=0)
    tops = {}
    for level in range(1, 5):
        cell = rows_for(rows, f'level_{level}', level)
        counts = Counter(family_for(r.get('answer', '')) for r in cell)
        counts.pop(None, None)
        tops[str(level)] = max(counts.values(), default=0)
    p5 = all(value >= 6 for value in tops.values())
    values = {'157.1': p1, '157.2': p2, '157.3': p3, '157.4': p4, '157.5': p5}
    if not all(item['pass'] for item in gates.values()):
        verdicts = {key: 'untestable' for key in values}
    else:
        verdicts = {key: ('held' if value else 'failed') for key, value in values.items()}
    return {
        key: {'prediction': label, 'value': values[key], 'verdict': verdicts[key], 'confidence': 'medium' if key in ('157.1', '157.2', '157.3') else 'low'}
        for key, label in {
            '157.1': 'At level 1 Google is the first-named family on more phrasings than any other family',
            '157.2': 'At level 1 at least 12 of 24 answers name a listed family',
            '157.3': 'Google-naming counts in the L27 N3325 arm are at most half baseline and in L27 N4991 are at least 75% baseline',
            '157.4': 'At level 2 Microsoft is the first-named family on more phrasings than any other family',
            '157.5': 'At every level from 1 through 4 the top first-named family has at least 6 of 24 answers',
        }.items()
    }, {'level1_first_named': dict(c1), 'level1_any_google': base_google,
        'zero_L27N3325_any_google': google_3325, 'zero_L27N4991_any_google': google_4991,
        'level2_first_named': dict(c2), 'top_family_counts_levels1_to4': tops}


def tables(rows, ladder):
    levels = sorted({int(x['level']) for x in ladder.get('levels', [])})
    out = ['| Level | ' + ' | '.join(run157.FAMILY_ORDER) + ' | No family | N |',
           '|---:|' + '|'.join(['---:'] * (len(run157.FAMILY_ORDER) + 2)) + '|']
    for level in levels:
        cell = rows_for(rows, f'level_{level}', level)
        counts = Counter(family_for(r.get('answer', '')) for r in cell)
        out.append('| ' + str(level) + ' | ' + ' | '.join(str(counts.get(f, 0)) for f in run157.FAMILY_ORDER) +
                   f" | {counts.get(None, 0)} | {len(cell)} |")
    return '\n'.join(out) + '\n'


def score(answers_path, ladder_path):
    rows, ladder = load_jsonl(answers_path), json.loads(Path(ladder_path).read_text(encoding='utf-8'))
    gates = gate_results(rows, ladder)
    predictions, values = prediction_results(rows, ladder, gates)
    return {'part': 157, 'gates': gates, 'predictions': predictions, 'values': values,
            'level_family_table': tables(rows, ladder)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--answers', required=True)
    ap.add_argument('--ladder', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    result = score(args.answers, args.ladder)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'score.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'tables.md').write_text('# Part 157 level by first-named family\n\n' + result['level_family_table'], encoding='utf-8')
    print(json.dumps({'gates': {k: v['pass'] for k, v in result['gates'].items()},
                      'predictions': {k: v['verdict'] for k, v in result['predictions'].items()}}, ensure_ascii=False))
    if not all(g['pass'] for g in result['gates'].values()):
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
