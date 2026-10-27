#!/usr/bin/env python3
"""Score Part 77 paired sampled answers against registered predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == _HERE)]
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part66 import score66  # noqa: E402

IDS = ['c0_makers', 'c1_l24', 'c2_l24_restore', 'c3_l24_restore_nulls']
MIN_GAP = 8   # Amendment 2


def verdicts(c: dict) -> dict:
    g = {k: c.get(k, {}).get('heldout_google', 0) for k in IDS}
    n = {k: c.get(k, {}).get('heldout_n', 0) for k in IDS}
    guards_ok = all(c.get(k, {}).get('guard_n', 0) == 24 and
                    c.get(k, {}).get('guard_keyword_matches', 0) >= 22 for k in IDS)
    nonidentical = sum(c.get(k, {}).get('pairs_not_identical', 0) for k in IDS)
    control = (all(v == 192 for v in n.values()) and 26 <= g['c0_makers'] <= 50 and
               g['c1_l24'] <= 28 and guards_ok and nonidentical == 0)
    gap = g['c0_makers'] - g['c1_l24']
    values = {
        '77.1': {'holds': control, 'google': g, 'heldout_n': n, 'guards_ok': guards_ok,
                 'pairs_not_identical': nonidentical},
        '77.2': {'holds': g['c2_l24_restore'] >= g['c1_l24'] + gap / 2,
                 'restored': g['c2_l24_restore'], 'c1': g['c1_l24'], 'c0': g['c0_makers'],
                 'threshold': g['c1_l24'] + gap / 2, 'gap_c0_minus_c1': gap},
        '77.3': {'holds': g['c3_l24_restore_nulls'] <= g['c1_l24'] + 8,
                 'null_restore': g['c3_l24_restore_nulls'], 'c1': g['c1_l24'],
                 'maximum': g['c1_l24'] + 8},
        '77.4': {'holds': g['c2_l24_restore'] <= g['c0_makers'] + 12,
                 'restored': g['c2_l24_restore'], 'c0': g['c0_makers'],
                 'maximum': g['c0_makers'] + 12},
    }
    for key, item in values.items():
        item['status'] = ('HOLDS' if item['holds'] else 'FAILS') if key == '77.1' or control else 'NOT READ'
    # Amendment 2: 77.2 needs a lost-answer gap of at least MIN_GAP to mean anything.
    if values['77.2']['status'] != 'NOT READ' and gap < MIN_GAP:
        values['77.2']['status'] = 'NOT READ'
        values['77.2']['reason'] = f'c0 - c1 = {gap} < {MIN_GAP}'
    # PART77-DESIGN.md reading rule: if copying the 10 other candidates also restores, 77.2 is not specific.
    if values['77.2']['status'] == 'HOLDS' and values['77.3']['status'] == 'FAILS':
        values['77.2']['status'] = 'HOLDS, NOT SPECIFIC TO L27 N3325'
    for item in values.values():
        if item['status'] == 'NOT READ':
            item['holds'] = None
    return values


def score_rows(rows):
    scored = score66.score_rows(rows)['conditions']
    counts = {cid: {'heldout_google': scored.get(cid, {}).get('heldout_google', 0),
                    'heldout_n': scored.get(cid, {}).get('heldout_n', 0),
                    'heldout_google_rate': scored.get(cid, {}).get('heldout_google_rate'),
                    'family_counts': scored.get(cid, {}).get('heldout_family_counts', {}),
                    'guard_keyword_matches': scored.get(cid, {}).get('guards', {}).get('keyword_matches', 0),
                    'guard_n': scored.get(cid, {}).get('guards', {}).get('n', 0),
                    'pairs_not_identical': sum(1 for r in rows if r.get('condition') == cid and
                                               r.get('pair_identical') is False)}
              for cid in IDS}
    return {'predictions': verdicts(counts), 'conditions': counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', required=True)
    out = Path(parser.parse_args().dir)
    rows, skipped = [], 0
    for line in (out / 'answers.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            skipped += 1   # a truncated last line from an interrupted run (the runner resumes past it)
    result = score_rows(rows)
    result['unparsed_lines'] = skipped
    (out / 'score77.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for key, item in result['predictions'].items():
        print(key, item['status'], {k: v for k, v in item.items() if k not in ('holds', 'status')})
    if skipped:
        print('unparsed answer lines skipped:', skipped)


if __name__ == '__main__':
    main()
