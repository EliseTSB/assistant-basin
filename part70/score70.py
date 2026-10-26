#!/usr/bin/env python3
"""Score Part 70 sampled answers against the registered prediction rules."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part66 import score66

MATCHED_IDS = [f'target_plus_matched_{k}' for k in range(20)]


def score_rows(rows):
    base = score66.score_rows(rows)
    conditions = base['conditions']

    def condition(cid):
        return conditions.get(cid, {})

    target = condition('target')
    pair = condition('target_plus_pair')
    matched = [condition(cid) for cid in MATCHED_IDS]
    target_n = target.get('heldout_n', 0)
    pair_n = pair.get('heldout_n', 0)
    target_google = target.get('heldout_google', 0)
    pair_google = pair.get('heldout_google', 0)
    pair_guards = pair.get('guards', {})
    guards = pair_guards.get('keyword_matches', 0)
    controls_pass = target_n == 192 and 30 <= target_google <= 54 and guards >= 22 and pair_guards.get('n') == 24

    verdicts = {
        '70.1': {'holds': target_n == 192 and 30 <= target_google <= 54,
                 'observed': {'google': target_google, 'n': target_n},
                 'rule': 'target held-out Google count 30-54 inclusive'},
        '70.2': {'holds': pair_guards.get('n') == 24 and guards >= 22,
                 'observed': {'keyword_matches': guards, 'n': pair_guards.get('n', 0)},
                 'rule': 'target_plus_pair guard keyword matches at least 22/24'},
    }
    if controls_pass:
        pooled_google = sum(c.get('heldout_google', 0) for c in matched)
        pooled_n = sum(c.get('heldout_n', 0) for c in matched)
        pair_rate = pair.get('heldout_google_rate')
        at_or_below = sum(c.get('heldout_google_rate') is not None and pair_rate is not None
                          and c['heldout_google_rate'] <= pair_rate for c in matched)
        alibaba = pair.get('heldout_family_counts', {}).get('alibaba', 0)
        verdicts.update({
            '70.3': {'holds': pair_n == 192 and pair_google <= 9,
                     'observed': {'google': pair_google, 'n': pair_n},
                     'rule': 'target_plus_pair held-out Google count at most 9/192'},
            '70.4': {'holds': pooled_n == 480 and pooled_google >= 72,
                     'observed': {'google': pooled_google, 'n': pooled_n},
                     'rule': 'pooled matched-set held-out Google count at least 72/480'},
            '70.5': {'holds': len(matched) == 20 and at_or_below <= 2,
                     'observed': {'matched_sets_at_or_below_pair_rate': at_or_below,
                                  'matched_set_count': len(matched), 'pair_rate': pair_rate},
                     'rule': 'at most 2/20 matched sets have Google rate at or below target_plus_pair'},
            '70.6': {'holds': pair_n == 192 and alibaba / 192 > 0.60,
                     'observed': {'alibaba_family_answers': alibaba, 'n': pair_n,
                                  'rate': alibaba / pair_n if pair_n else None},
                     'rule': 'target_plus_pair Alibaba-family answers exceed 60% of 192'},
        })
    else:
        for key in ('70.3', '70.4', '70.5', '70.6'):
            verdicts[key] = {'holds': None, 'status': 'NOT READ',
                             'rule': 'not read because 70.1 or 70.2 failed'}

    # Expand each per-condition report with the same family counts and the controls relevant to Part 70.
    reports = {}
    for cid, stats in conditions.items():
        reports[cid] = {
            'heldout_google': stats['heldout_google'],
            'heldout_n': stats['heldout_n'],
            'heldout_google_rate': stats['heldout_google_rate'],
            'alibaba_family_count': stats['heldout_family_counts']['alibaba'],
            'family_counts': stats['heldout_family_counts'],
            'positive_control_google': stats['positive']['google'],
            'positive_control_n': stats['positive']['n'],
            'guard_keyword_matches': stats['guards']['keyword_matches'],
            'guard_n': stats['guards']['n'],
            'guard_keyword_rate': stats['guards']['rate'],
        }
    return {'conditions': reports, 'predictions': verdicts,
            'reading_rule_applied': not controls_pass,
            'score66_condition_details': conditions}


def score_dir(directory: str | Path):
    out = Path(directory)
    answer_path = out / 'answers.jsonl'
    rows = [json.loads(line) for line in answer_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = score_rows(rows)
    (out / 'score70.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    md = ['# Part 70 score', '', '| Prediction | Verdict | Observation | Rule |', '|---|---|---|---|']
    for key, item in result['predictions'].items():
        verdict = item.get('status', 'HOLDS' if item['holds'] else 'FAILS')
        md.append(f"| {key} | {verdict} | `{json.dumps(item['observed'], ensure_ascii=False)}` | {item['rule']} |")
    md += ['', '## Conditions', '',
           '| Condition | Held-out Google | Google rate | Alibaba family | Positive-control Google | Guard keywords |',
           '|---|---:|---:|---:|---:|---:|']
    for cid, stats in result['conditions'].items():
        rate = '—' if stats['heldout_google_rate'] is None else f"{stats['heldout_google_rate']:.4f}"
        guard_rate = '—' if stats['guard_keyword_rate'] is None else f"{stats['guard_keyword_rate']:.4f}"
        md.append(f"| {cid} | {stats['heldout_google']}/{stats['heldout_n']} | {rate} | "
                  f"{stats['alibaba_family_count']} | {stats['positive_control_google']}/{stats['positive_control_n']} | "
                  f"{stats['guard_keyword_matches']}/{stats['guard_n']} ({guard_rate}) |")
    md.append('')
    (out / 'score70.md').write_text('\n'.join(md), encoding='utf-8')
    for key, item in result['predictions'].items():
        verdict = item.get('status', 'HOLDS' if item['holds'] else 'FAILS')
        print(f"{key}: {verdict} ({item['rule']})")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', required=True, help='directory containing answers.jsonl')
    args = parser.parse_args()
    score_dir(args.dir)


if __name__ == '__main__':
    main()
