#!/usr/bin/env python3
"""Score Part 75 sampled answers against 75.1-75.3 (PART75-DESIGN.md). Claude, 2026-10-26."""
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

IDS = ['target', 'target_plus_l24', 'target_plus_l24_clamp_l27', 'target_clamp_l27']


def verdicts(c: dict) -> dict:
    g = {k: c[k]['heldout_google'] for k in IDS}
    n = {k: c[k]['heldout_n'] for k in IDS}
    guards_ok = all(c[k]['guard_n'] == 24 and c[k]['guard_keyword_matches'] >= 22 for k in IDS)
    c1 = all(v == 192 for v in n.values()) and 26 <= g['target'] <= 50 and g['target_plus_l24'] <= 28 and guards_ok
    c2 = abs(g['target_clamp_l27'] - g['target']) <= 12
    halfway = (g['target'] + g['target_plus_l24']) / 2
    v = {'75.1': {'holds': c1, 'google': g, 'guards_ok': guards_ok},
         '75.2': {'holds': c2, 'clamp_only': g['target_clamp_l27'], 'makers_only': g['target']},
         '75.3': {'holds': g['target_plus_l24_clamp_l27'] >= halfway, 'restored': g['target_plus_l24_clamp_l27'],
                  'halfway': halfway}}
    v['75.1']['status'] = 'HOLDS' if c1 else 'FAILS'
    v['75.2']['status'] = 'HOLDS' if c2 else 'FAILS'
    v['75.3']['status'] = ('HOLDS' if v['75.3']['holds'] else 'FAILS') if (c1 and c2) else 'NOT READ'
    return v


def score_rows(rows):
    conds = score66.score_rows(rows)['conditions']
    c = {k: {'heldout_google': conds[k]['heldout_google'], 'heldout_n': conds[k]['heldout_n'],
             'heldout_google_rate': conds[k]['heldout_google_rate'], 'family_counts': conds[k]['heldout_family_counts'],
             'guard_keyword_matches': conds[k]['guards']['keyword_matches'], 'guard_n': conds[k]['guards']['n']}
         for k in IDS}
    return {'predictions': verdicts(c), 'conditions': c}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', required=True)
    out = Path(parser.parse_args().dir)
    rows = [json.loads(l) for l in (out / 'answers.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    result = score_rows(rows)
    (out / 'score75.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for key, item in result['predictions'].items():
        print(key, item['status'], {k: item[k] for k in item if k not in ('holds', 'status')})


if __name__ == '__main__':
    main()
