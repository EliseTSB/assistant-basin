"""Recompute every number in results/part191/README.md from the committed files, using the real scorer.

Run: python3 -I measure.py /home/elisepc/green/worktrees/ab-part191
Writes nothing; prints one line per claim with the README's number beside the recomputed one.
"""
import json
import re
import sys
import statistics
from collections import Counter

ROOT = sys.argv[1]
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + '/part86')
import triage86  # noqa: E402  (real Part 86 triage)
from part153 import score153  # noqa: E402  (real, unchanged scorer)

EAGER = [json.loads(l) for l in open(ROOT + '/results/part191/answers.jsonl', encoding='utf-8')]
SDPA = [json.loads(l) for l in open(ROOT + '/results/part153/answers.jsonl', encoding='utf-8')]
KE = {(r['unit_id'], r['arm'], r['question_index']): r for r in EAGER}
KS = {(r['unit_id'], r['arm'], r['question_index']): r for r in SDPA}
BAD = re.compile(r'alibaba|tongyi|通义', re.I)


def fs(ans):
    return re.split(r'(?<=[.!?])\s|\n', ans.strip(), maxsplit=1)[0]


def ends_punct(ans):
    s = ans.rstrip()
    return bool(s) and s[-1] in '.!?…"\'”’)]*'


def emoji_end(ans):
    s = ans.rstrip()
    return bool(s) and ord(s[-1]) > 0x2000


print('== scorer summary (real score153.score_rows) ==')
for name, rows in (('eager', EAGER), ('sdpa', SDPA)):
    res = score153.score_rows(rows)
    print(name, json.dumps(res['summary']), res['predictions'], 'gates_all', res['gates']['all_passed'])
    print('  per-unit maker hits', {u: v['arms']['neuron']['maker']['hits'] for u, v in res['per_unit'].items()})

print('== descriptive ==')
same = [k for k in KE if KE[k]['answer'] == KS[k]['answer']]
print('identical text total', len(same), {a: sum(k[1] == a for k in same) for a in ('neuron', 'random', 'plain')})
print('median tokens eager', statistics.median(r['token_count'] for r in EAGER),
      'sdpa', statistics.median(r['token_count'] for r in SDPA))
print('at cap 120 eager', sum(r['token_count'] == 120 for r in EAGER), 'sdpa', sum(r['token_count'] == 120 for r in SDPA))
print('sdpa most common lengths', Counter(r['token_count'] for r in SDPA).most_common(5))
ue = [r for r in EAGER if r['token_count'] < 120]
us = [r for r in SDPA if r['token_count'] < 120]
for name, u in (('eager', ue), ('sdpa', us)):
    n = sum(not ends_punct(r['answer']) for r in u)
    print(name, 'under cap', len(u), 'no closing punct', n, f'{100*n/len(u):.1f}%',
          'of which emoji-ending', sum(emoji_end(r['answer']) for r in u if not ends_punct(r['answer'])))
for name, u in (('eager', ue), ('sdpa', us)):
    v = [r for r in u if not emoji_end(r['answer'])]
    n = sum(not ends_punct(r['answer']) for r in v)
    print(name, 'under cap, not emoji-ending', len(v), 'no closing punct', n, f'{100*n/len(v):.1f}%')
print('QwQ exact eager', sum('QwQ' in r['answer'] for r in EAGER), 'sdpa', sum('QwQ' in r['answer'] for r in SDPA))
print('qwq case-insensitive eager', sum('qwq' in r['answer'].lower() for r in EAGER),
      'sdpa', sum('qwq' in r['answer'].lower() for r in SDPA))

print('== neuron maker rows ==')
for name, rows in (('eager', EAGER), ('sdpa', SDPA)):
    nm = [r for r in rows if r['arm'] == 'neuron' and r['kind'] == 'maker']
    named = [r for r in nm if r['company'].casefold() in r['answer'].casefold()]
    hits = [r for r in named if not BAD.search(r['answer']) and triage86.triage_answer(r['answer']) != 'broken']
    print(name, 'named', len(named), 'hits', len(hits),
          'hits named in first sentence', sum(r['company'].casefold() in fs(r['answer']).casefold() for r in hits),
          'broken', sum(triage86.triage_answer(r['answer']) == 'broken' for r in named),
          'alibaba/tongyi', sum(bool(BAD.search(r['answer'])) for r in named))
    print('  broken at cap', sum(r['token_count'] == 120 for r in named if triage86.triage_answer(r['answer']) == 'broken'))
    print('  triage of named', Counter(triage86.triage_answer(r['answer']) for r in named))
