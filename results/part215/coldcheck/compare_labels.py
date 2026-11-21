#!/usr/bin/env python3
"""Row-by-row comparison of MY independent labels against the scorer's own attrib215/is_broken.

The scorer is imported HERE ONLY as a comparator (it needs numpy, so this helper runs on plain
`python3`, not `-I`).  It is not used in recompute.py, which produced the numbers being checked.

Run:  python3 compare_labels.py
"""
import sys

REPO = '/home/elisepc/assistant-basin'
COLD = '/home/elisepc/assistant-basin/results/part215/coldcheck'
sys.path.insert(0, REPO)
sys.path.insert(0, COLD)

from part215.score215 import attrib215 as scorer_attrib  # noqa: E402
from part215.score215 import is_broken as scorer_broken  # noqa: E402

import recompute  # noqa: E402  (my implementation)

rows = recompute.load_jsonl(recompute.ROWS)
mism_broken = []
mism_attrib = []
mism_route = []
counts = {}
for r in rows:
    ans = r.get('answer', '')
    mb = recompute.is_broken(ans)
    sb = bool(scorer_broken(ans))
    ma = recompute.attrib(ans)
    sa = scorer_attrib(ans)
    key = (r['set'], r['arm'], r['question_index'], r['variant'], r['sample'])
    if mb != sb:
        mism_broken.append((key, mb, sb, ans))
    mine_maker = None if mb else ma['maker']
    theirs_maker = None if sb else sa['maker']
    if mine_maker != theirs_maker:
        mism_attrib.append((key, mine_maker, theirs_maker, ma['route'], sa['route'], ans))
    elif mine_maker is not None and ma['route'] != sa['route']:
        mism_route.append((key, ma['route'], sa['route'], ans))
    k = (r['set'], r['arm'])
    c = counts.setdefault(k, {'n': 0, 'sm_any': 0, 'mm_any': 0, 'sm_g': 0, 'mm_g': 0})
    c['n'] += 1
    c['sm_any'] += 1 if theirs_maker else 0
    c['mm_any'] += 1 if mine_maker else 0
    c['sm_g'] += 1 if theirs_maker == 'google' else 0
    c['mm_g'] += 1 if mine_maker == 'google' else 0

print('rows compared:', len(rows))
print('broken disagreements:', len(mism_broken))
for x in mism_broken[:20]:
    print('  BROKEN', x[0], 'mine', x[1], 'scorer', x[2], '::', x[3][:200].replace('\n', ' '))
print('attribution-maker disagreements:', len(mism_attrib))
for x in mism_attrib[:40]:
    print('  ATTRIB', x[0], 'mine', x[1], 'scorer', x[2], 'routes', x[3], x[4],
          '::', x[5][:260].replace('\n', ' '))
print('same maker, different route:', len(mism_route))
for x in mism_route[:20]:
    print('  ROUTE', x[0], x[1], x[2], '::', x[3][:200].replace('\n', ' '))
print()
print('per set/arm counts (scorer vs mine):')
for k in sorted(counts):
    c = counts[k]
    flag = '' if (c['sm_any'] == c['mm_any'] and c['sm_g'] == c['mm_g']) else '  <-- MISMATCH'
    print(f"  {k[0]:8s} {k[1]:10s} n={c['n']:3d} any {c['sm_any']:3d}/{c['mm_any']:3d} "
          f"google {c['sm_g']:3d}/{c['mm_g']:3d}{flag}")
