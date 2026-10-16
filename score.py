#!/usr/bin/env python3
# score.py <dir> — unblind part 1: verdict shares by arm, per judge, plus judge agreement (Claude, 2026-10-15).
# Arms (from key.json): N neuron poles, R random directions, F random-context floor. Reports, per judge and arm, the
# share judged "one", "two_related", "several_unrelated", "unclear" (errors excluded and counted), the N-vs-R gap with a
# two-proportion z score, N and R by model third, and the share of units where both judges agree on "one".
import json, math, os, sys
from collections import Counter, defaultdict

d = sys.argv[1]
key = json.load(open(os.path.join(d, 'key.json')))
judges = sorted(f[len('verdicts-'):-5] for f in os.listdir(d) if f.startswith('verdicts-') and f.endswith('.json'))
V = {j: json.load(open(os.path.join(d, f'verdicts-{j}.json'))) for j in judges}
CATS = ['one', 'two_related', 'several_unrelated', 'unclear']

def z2(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2); se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0
    return (k1 / n1 - k2 / n2) / se if se else float('nan')

for j in judges:
    print(f'== {j}')
    byarm = defaultdict(Counter); err = Counter()
    for uid, v in V[j].items():
        arm = key[uid]['arm']
        if v['verdict'] in CATS: byarm[arm][v['verdict']] += 1
        else: err[arm] += 1
    for arm in 'NRF':
        n = sum(byarm[arm].values())
        print(f'  {arm} n={n:3d} ' + '  '.join(f'{c} {byarm[arm][c] / n:.0%}' for c in CATS) + (f'  (errors {err[arm]})' if err[arm] else ''))
    nN, nR = sum(byarm['N'].values()), sum(byarm['R'].values())
    kN, kR = byarm['N']['one'], byarm['R']['one']
    print(f'  one-job gap N - R = {kN / nN - kR / nR:+.0%} (z {z2(kN, nN, kR, nR):.2f})')
    for t in range(3):
        row = []
        for arm in 'NR':
            us = [u for u, k in key.items() if k['arm'] == arm and k['third'] == t and V[j].get(u, {}).get('verdict') in CATS]
            row.append(f'{arm} {sum(V[j][u]["verdict"] == "one" for u in us) / max(1, len(us)):.0%} of {len(us)}')
        print(f'  third {t}: ' + ', '.join(row))
if len(judges) == 2:
    a, b = judges
    for arm in 'NRF':
        us = [u for u, k in key.items() if k['arm'] == arm and u in V[a] and u in V[b]]
        agree = sum(V[a][u]['verdict'] == V[b][u]['verdict'] for u in us)
        both_one = sum(V[a][u]['verdict'] == 'one' and V[b][u]['verdict'] == 'one' for u in us)
        print(f'agreement {arm}: same verdict {agree / len(us):.0%}, both say one {both_one / len(us):.0%} (n={len(us)})')
