#!/usr/bin/env python3
"""Score Part 78b answers using Part 78 keyword rules and registered predictions."""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part66 import score66
from part78 import score78
CONTROL_UNITS = (1499, 4154, 5572, 5606, 5640, 6210, 8115, 8606, 8920, 9209)
CONDITION_IDS = ('k1_makers',) + tuple(f'k_u{u}' for u in CONTROL_UNITS)


def score_rows(rows):
    by = defaultdict(list)
    for row in rows:
        r = dict(row)
        r['rules'] = score78.classify(r.get('answer', ''))
        by[r.get('condition', '')].append(r)
    conditions = {}
    for cid in CONDITION_IDS:
        rs = by.get(cid, [])
        held = [r for r in rs if r.get('question_kind') in ('direct', 'indirect')]
        guards = [r for r in rs if r.get('question_kind') == 'guard']
        per_question = defaultdict(Counter)
        for question_index in range(24):
            per_question[question_index]['google'] = 0
        for row in held:
            if row['rules']['google']:
                per_question[int(row['question_index'])]['google'] += 1
        guard_matches = sum(bool(re.search(re.escape(score66.GUARD_KEYWORDS.get(r.get('question'), '(?!)')),
                                           r.get('answer', ''), re.I)) for r in guards)
        conditions[cid] = {
            'heldout_n': len(held),
            'heldout_counts': {key: sum(r['rules'][key] for r in held)
                for key in ('google', 'off_world', 'qwen', 'maker_phrase', 'composed', 'other_maker')},
            'per_question_google_counts': {str(i): per_question[i]['google'] for i in sorted(per_question)},
            'guard_n': len(guards), 'guard_keyword_matches': guard_matches,
            'guard_matches_by_question': {question: {
                'n': sum(r.get('question') == question for r in guards),
                'matches': sum(r.get('question') == question and bool(re.search(re.escape(answer),
                    r.get('answer', ''), re.I)) for r in guards)}
                for question, answer in score66.GUARD_KEYWORDS.items()}}
    return conditions


def verdicts(conditions):
    base = conditions.get('k1_makers', {})
    base_counts = base.get('heldout_counts', {})
    base_google = base_counts.get('google', 0)
    gate = (base.get('heldout_n') == 192 and base.get('guard_n') == 24 and
            26 <= base_google <= 60 and base.get('guard_keyword_matches', 0) >= 20)
    controls = [conditions.get(f'k_u{unit}', {}) for unit in CONTROL_UNITS]
    greater = sum(c.get('heldout_counts', {}).get('google', 0) > base_google for c in controls)
    less_than_half = sum(c.get('heldout_counts', {}).get('google', 0) < base_google / 2 for c in controls)
    max_off_world = max((c.get('heldout_counts', {}).get('off_world', 0) for c in controls), default=0)
    rules = {
        '78b.1': (gate, 'Control: k1_makers has 26 to 60 held-out Google answers of 192 and its guard keywords match in at least 20 of 24.'),
        '78b.2': (greater >= 7, 'Claude: at least 7 of the 10 control units give more held-out Google answers than k1_makers (the rise is common to layer-22 pushes of this size, not particular to N8870).'),
        '78b.3': (less_than_half == 0, 'Claude: none of the 10 control units gives fewer than half of k1_makers\' held-out Google answers (among 11 size-matched layer-22 pushes, only the alien unit suppresses the Google fallback).'),
        '78b.4': (max_off_world <= 10, 'Claude: no control unit has more than 10 off-world answers of 192.'),
    }
    out = {}
    for key, (holds, rule) in rules.items():
        status = 'HOLDS' if holds else 'FAILS'
        if key != '78b.1' and not gate:
            status, holds = 'NOT READ', None
        out[key] = {'status': status, 'holds': holds, 'rule': rule}
    return out


def score_file(directory):
    d = Path(directory)
    rows, bad = [], 0
    for line in (d / 'answers.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            bad += 1
    conditions = score_rows(rows)
    result = {'conditions': conditions, 'predictions': verdicts(conditions), 'unparsed_lines': bad}
    (d / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Condition      Google Off-world Qwen Phrase Composed Other-maker Guards')
    for cid in CONDITION_IDS:
        c = conditions[cid]
        h = c['heldout_counts']
        print(f"{cid:12s} {h['google']:6d} {h['off_world']:9d} {h['qwen']:4d} {h['maker_phrase']:6d} {h['composed']:8d} {h['other_maker']:11d} {c['guard_keyword_matches']:3d}/{c['guard_n']}")
    for key, verdict in result['predictions'].items():
        print(key, verdict['status'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', required=True)
    score_file(parser.parse_args().dir)


if __name__ == '__main__':
    main()
