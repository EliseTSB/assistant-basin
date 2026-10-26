#!/usr/bin/env python3
"""Part 74 effect shares from records.json, with question-bootstrap 95% intervals (review finding 2). Claude, 2026-10-26.

NDE = g(C) - g(A): L24 N3949 zeroed, L27 N3325 held at its makers-only values (natural direct effect).
PIE = g(E) - g(A): L24 N3949 intact, L27 N3325 set to its L24-zeroed values (pure indirect effect).
TIE = TE - NDE: the total indirect effect. Remainder = TE - NDE - PIE (non-additive part).
"""
import json
import sys
from pathlib import Path

import numpy as np


def shares(by_q: dict) -> dict:
    te = np.array([q['B'] - q['A'] for q in by_q])
    nde = np.array([q['C'] - q['A'] for q in by_q])
    pie = np.array([q['E'] - q['A'] for q in by_q])
    T = te.mean()
    return {'TE': T, 'NDE_share': nde.mean() / T, 'PIE_share': pie.mean() / T, 'TIE_share': (T - nde.mean()) / T,
            'remainder_share': (T - nde.mean() - pie.mean()) / T}


def main(path: Path, n_boot=10000, seed=0):
    d = json.loads(path.read_text(encoding='utf-8'))
    qs = {}
    for r in d['records']:
        qs.setdefault(r['q'], {})[r['id']] = r['g']
    by_q = [qs[k] for k in sorted(qs)]
    point = shares(by_q)
    rng = np.random.default_rng(seed)
    boots = [shares([by_q[i] for i in rng.integers(0, len(by_q), len(by_q))]) for _ in range(n_boot)]
    ci = {k: [float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))]
          for k in point}
    per_q = {'NDE_over_TE_median': float(np.median([(q['C'] - q['A']) / (q['B'] - q['A']) for q in by_q])),
             'PIE_over_TE_median': float(np.median([(q['E'] - q['A']) / (q['B'] - q['A']) for q in by_q]))}
    mass = {}
    for r in d['records']:
        mass.setdefault(r['id'], []).append(r['mass'])
    out = {'point': {k: float(v) for k, v in point.items()}, 'ci95_question_bootstrap': ci, 'per_question': per_q,
           'mass_mean_min': {c: [float(np.mean(v)), float(np.min(v))] for c, v in mass.items() if not c.startswith('null_')},
           'n_boot': n_boot, 'seed': seed}
    return out


if __name__ == '__main__':
    res = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/part74')
    out = main(res / 'records.json')
    (res / 'shares.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=1))
