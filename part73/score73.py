#!/usr/bin/env python3
"""Score Part 73 sampled answers against the registered rules 73.1-73.4 (PART73-DESIGN.md). Claude, 2026-10-26."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part66 import score66  # noqa: E402

L27_NULLS = [f'target_plus_l27_matched_{k}' for k in range(20)]
L24_NULLS = [f'target_plus_l24_matched_{k}' for k in range(20)]


def score_rows(rows):
    conditions = score66.score_rows(rows)['conditions']

    def cond(cid):
        return conditions.get(cid, {})

    target, l27, l24 = cond('target'), cond('target_plus_l27'), cond('target_plus_l24')
    nulls27 = [cond(c) for c in L27_NULLS]
    nulls24 = [cond(c) for c in L24_NULLS]

    def guards(c):
        g = c.get('guards', {})
        return g.get('keyword_matches', 0), g.get('n', 0)

    g27, n27 = guards(l27)
    g24, n24 = guards(l24)
    control = (target.get('heldout_n') == 192 and 26 <= target.get('heldout_google', -1) <= 50
               and n27 == 24 and g27 >= 22 and n24 == 24 and g24 >= 22)

    def at_or_below(nulls, c):
        rate = c.get('heldout_google_rate')
        return sum(1 for n in nulls if rate is not None and n.get('heldout_google_rate') is not None
                   and n['heldout_google_rate'] <= rate)

    v = {'73.1': {'holds': control,
                  'observed': {'target_google': target.get('heldout_google'), 'target_n': target.get('heldout_n'),
                               'l27_guards': [g27, n27], 'l24_guards': [g24, n24]},
                  'rule': 'makers-only held-out Google 26-50 of 192; guards >= 22/24 for l27 and l24'}}
    reads = {
        '73.2': (l27.get('heldout_n') == 192 and l27.get('heldout_google', 999) <= 12,
                 {'google': l27.get('heldout_google'), 'n': l27.get('heldout_n')},
                 'makers + L27 N3325 held-out Google at most 12 of 192'),
        '73.3': (len(nulls27) == 20 and all(n.get('heldout_n') == 24 for n in nulls27)
                 and at_or_below(nulls27, l27) <= 2,
                 {'nulls_at_or_below': at_or_below(nulls27, l27), 'null_count': len(nulls27),
                  'l27_rate': l27.get('heldout_google_rate')},
                 'at most 2 of 20 layer-27 single nulls at or below makers + L27 N3325'),
        '73.4': (l24.get('heldout_n') == 192 and l24.get('heldout_google', -1) >= 24,
                 {'google': l24.get('heldout_google'), 'n': l24.get('heldout_n')},
                 'makers + L24 N3949 held-out Google at least 24 of 192'),
    }
    for key, (holds, observed, rule) in reads.items():
        v[key] = {'holds': holds, 'observed': observed, 'rule': rule}
        if not control:
            v[key]['status'] = 'NOT READ'
    described = {'l24_nulls_at_or_below': at_or_below(nulls24, l24),
                 'l27_null_google': [n.get('heldout_google') for n in nulls27],
                 'l24_null_google': [n.get('heldout_google') for n in nulls24]}
    reports = {cid: {'heldout_google': s['heldout_google'], 'heldout_n': s['heldout_n'],
                     'heldout_google_rate': s['heldout_google_rate'],
                     'family_counts': s['heldout_family_counts'],
                     'guard_keyword_matches': s['guards']['keyword_matches'], 'guard_n': s['guards']['n']}
               for cid, s in conditions.items()}
    return {'predictions': v, 'description_only': described, 'conditions': reports,
            'reading_rule_applied': not control}


def score_dir(directory: str | Path):
    out = Path(directory)
    rows = [json.loads(line) for line in (out / 'answers.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]
    result = score_rows(rows)
    (out / 'score73.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for key, item in result['predictions'].items():
        print(f"{key}: {item.get('status', 'HOLDS' if item['holds'] else 'FAILS')} {json.dumps(item['observed'])}")
    print('description:', json.dumps(result['description_only']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', required=True)
    score_dir(parser.parse_args().dir)


if __name__ == '__main__':
    main()
