"""Cold check of Part 169, part 5: scorer median choices, W1 split robustness, and a crude
'cycle-like' test of the loop label (span of the most-repeated trigram) across bans.
Run from the repo root: /home/elisepc/neuron-probe/.venv/bin/python -I -B cold169/recompute_part5.py
Writes nothing.
"""
import json, sys, statistics, re
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from part165 import score165
from part166 import score166

P = ROOT / 'results/part169'
def jl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
masses = jl(P / 'masses.jsonl')
answers = jl(P / 'answers.jsonl')
def rk(r):
    return (r['arm'], int(r['layer']), int(r['slot']), int(r['question_index']))
M = {rk(m): m for m in masses}
B = {}
for a in answers:
    if a.get('stage') == 'battery':
        B[rk(a) + (a['ban'],)] = a
allrows = list(M)
L = {k: bool(score166.loop_phrase(a['answer'])) for k, a in B.items()}

# --- scorer median: score169.med returns the upper median (vals[len//2]) for even-length lists
none_loop = {k: L[k + ('none',)] for k in allrows}
w1_rows = sorted(k for k in allrows if not none_loop[k])
ov = {k: len(set(map(int, B[k + ('none',)]['token_ids'])) & set(map(int, M[k]['lists']['plain50']['ids']))) for k in allrows}
vals = sorted(ov[k] for k in w1_rows)
n = len(vals)
print('W1 rows (no none loop):', n, 'upper median used by scorer:', vals[n // 2], 'true median:', (vals[n // 2 - 1] + vals[n // 2]) / 2)
print('  rows with overlap == 18:', sum(1 for v in vals if v == 18), ' overlap <=17:', sum(1 for v in vals if v <= 17), ' <=18:', sum(1 for v in vals if v <= 18))
def w1(split):
    above = [k for k in w1_rows if ov[k] > split]
    below = [k for k in w1_rows if ov[k] <= split]
    a = sum(L[k + ('plain50',)] for k in above) / len(above)
    b = sum(L[k + ('plain50',)] for k in below) / len(below)
    return len(above), round(a, 4), len(below), round(b, 4), round(a - b, 4)
print('  split as scorer (> 18 vs <= 18):', w1(18))
print('  split at 17.5 (>=18 vs <=17):   ', w1(17.5))
print('  split at 17 (>=18 vs <=17):     ', w1(17))

paired = [k for k in allrows if M[k]['paired_set']]
mr = sorted(M[k]['lists']['free']['median_rank'] for k in paired)
print('\npaired rows', len(paired), 'free median_rank upper-median used by scorer:', mr[len(mr)//2], 'true median:', (mr[len(mr)//2-1] + mr[len(mr)//2]) / 2)
mrc = sorted(M[k]['lists']['clear']['median_rank'] for k in paired)
print('  clear: scorer', mrc[len(mrc)//2], 'true', (mrc[len(mrc)//2-1] + mrc[len(mrc)//2]) / 2)
mro = sorted(M[k]['lists']['own']['median_rank'] for k in paired)
print('  own: scorer', mro[len(mro)//2], 'true', (mro[len(mro)//2-1] + mro[len(mro)//2]) / 2)

# --- crude cycle test: for each loop answer, the most-repeated trigram (score166 picks the first max)
WORDS = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)
def span_of_top_trigram(text):
    words = [w.casefold() for w in WORDS.findall(text)]
    grams = [tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))]
    if not grams:
        return None, None
    c = Counter(grams)
    top, cnt = c.most_common(1)[0]
    pos = [i for i, g in enumerate(grams) if g == top]
    return pos[-1] - pos[0], cnt
print('\n== loop answers: span (in words) between first and last occurrence of most-repeated trigram')
print('  span <= 20 words counts as "tight" (cycle-like); a crude proxy only')
for ban in ('none', 'own', 'free', 'clear', 'plain50', 'hit'):
    ks = [k for k in B if k[-1] == ban and L[k]]
    tight = 0; spans = []
    for k in ks:
        sp, cnt = span_of_top_trigram(B[k]['answer'])
        if sp is not None:
            spans.append(sp)
            tight += sp <= 20
    print(f'  {ban:7s} loops {len(ks):4d} tight {tight:4d} ({100*tight/max(1,len(ks)):.0f}%) median span {statistics.median(spans) if spans else None}')
ind = [k for k in B if k[-1] == 'plain50' and L[k] and not L[k[:4] + ('none',)]]
tight = sum(1 for k in ind if (span_of_top_trigram(B[k]['answer'])[0] or 999) <= 20)
print(f'  plain50 INDUCED loops {len(ind)}: tight {tight} ({100*tight/len(ind):.0f}%)')
