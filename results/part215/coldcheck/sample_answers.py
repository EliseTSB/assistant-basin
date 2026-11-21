#!/usr/bin/env python3
"""Print sampled answers for the cold-check label reading.

Labels are my independent attrib215 implementation, which agreed with the scorer's label on all
2,304 rows (compare_labels.py).  Sampling is deterministic and evenly spread over the sorted rows
(question, variant, sample), never hand-picked.

Run: python3 -I sample_answers.py ARM google|none|alibaba [N]
"""
import sys

sys.path.insert(0, '/home/elisepc/assistant-basin/results/part215/coldcheck')
import recompute

arm = sys.argv[1]
category = sys.argv[2]
n_req = int(sys.argv[3]) if len(sys.argv) > 3 else 15

rows = [r for r in recompute.load_jsonl(recompute.ROWS)
        if r['set'] == 'identity' and r['arm'] == arm]
labelled = []
for r in rows:
    rr = dict(r)
    broken = recompute.is_broken(r.get('answer', ''))
    got = recompute.attrib(r.get('answer', ''))
    maker = None if broken else got['maker']
    route = None if broken else got['route']
    rr['broken'] = broken
    rr['maker'] = maker
    rr['route'] = route
    labelled.append(rr)

labelled.sort(key=lambda r: (int(r['question_index']), r['variant'], int(r['sample'])))

if category == 'google':
    sel = [r for r in labelled if r['maker'] == 'google']
elif category == 'alibaba':
    sel = [r for r in labelled if r['maker'] == 'alibaba']
elif category == 'none':
    sel = [r for r in labelled if r['maker'] is None and not r['broken']]
elif category == 'broken':
    sel = [r for r in labelled if r['broken']]
else:
    raise SystemExit('bad category')

print(f'# arm={arm} category={category} total_labelled={len(sel)} n_requested={n_req}')
if len(sel) > n_req:
    idxs = [round(i * (len(sel) - 1) / (n_req - 1)) for i in range(n_req)]
    chosen = [sel[i] for i in idxs]
else:
    chosen = sel
print(f'# printed={len(chosen)}')
openers = {}
for r in labelled:
    openers.setdefault(r['variant'], r['opener'])
print('# openers by variant:')
for v, o in sorted(openers.items()):
    print(f'#   {v}: {o!r}')
print()
for r in chosen:
    print(f"[{r['set']}/{r['arm']}/q{r['question_index']}/{r['variant']}/s{r['sample']}]"
          f" Q={r['question']!r} route={r['route']}")
    print('ANSWER:', r['answer'].replace('\n', ' | '))
    print()
