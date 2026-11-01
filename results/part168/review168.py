"""Part 168, added after the cold check on Weaver's review: list contents, own-mass strata, leave-one-slot-out.

Exploratory. Run from the repo root; output in review168.txt. FW is a hand list of 40 function words and punctuation.
"""
import json, sys
from collections import Counter, defaultdict
sys.path.insert(0, '.')
from part166 import score166
from part165 import score165


def load(p):
    return [json.loads(l) for l in open(p, encoding='utf-8')]


M = [m for m in load('results/part168/masses.jsonl') if m['arm'] in ('n+', 'k4')]
A = [a for a in load('results/part168/answers.jsonl') if a.get('stage') == 'battery']
tok = json.load(open('results/part168/tokens.json', encoding='utf-8'))
lt = json.load(open('results/part166/loop_tokens.json', encoding='utf-8'))
print('loop_tokens keys', list(lt.keys())[:8] if isinstance(lt, dict) else type(lt))
common = None
for k in ('common_ids', 'common_tokens', 'common', 'common_words'):
    if isinstance(lt, dict) and k in lt:
        common = lt[k]
        print('common key', k, len(common), str(common[:6])[:100] if isinstance(common, list) else '')
        break


def loop(a):
    return score165.has_trigram_loop(a['answer'])


L = {(a['arm'], a['layer'], a['slot'], a['question_index'], a['ban']): loop(a) for a in A}
# 1. what is in the mass lists
c = Counter()
for m in M:
    c.update(m['mass_ban_ids'])
print('distinct ids', len(c), 'list slots', sum(c.values()))
print('top 25 by list count:', [(tok.get(str(t), '?'), n) for t, n in c.most_common(25)])
FW = {' the', ' of', ' and', ' to', ' a', ' in', ' is', ' that', ' for', ' it', ' on', ' with', ' as', ' by', ' an', ' are', ' be', ' this', ' at', ' or', ' from', 'The', ' The', ' I', ' you', ' was', ' not', ' have', ' has', ' can', '.', ',', '\n', '\n\n', '**', ' **', ':', ' (', ')', '-'}
fw_ids = {int(t) for t, s in tok.items() if s in FW}
sh = [sum(int(t) in fw_ids for t in m['mass_ban_ids'])/50 for m in M]
print('function-word-or-punct share mean', round(sum(sh)/len(sh), 4), 'rows with any', sum(s > 0 for s in sh), 'of', len(sh), 'rows over half', sum(s > .5 for s in sh))
# function-heavy rows against induced
inv = defaultdict(lambda: [0, 0])
for m, s in zip(M, sh):
    k = (m['arm'], m['layer'], m['slot'], m['question_index'])
    if L[k+('none',)]:
        continue
    b = '0' if s == 0 else ('1' if s*50 == 1 else '2+')
    inv[b][0] += 1
    inv[b][1] += L[k+('mass',)]
print('induced by function tokens on list', {k: (v[1], v[0], round(v[1]/v[0], 3)) for k, v in sorted(inv.items())})
# 2. own-mass strata by overlap
ms = sorted(m['own_mass'] for m in M)
q = [ms[len(ms)*i//4] for i in (1, 2, 3)]
print('own mass quartile cuts', [float('%.3g' % x) for x in q])
T = defaultdict(lambda: [0, 0, 0, 0, 0])
for m in M:
    k = (m['arm'], m['layer'], m['slot'], m['question_index'])
    st = sum(m['own_mass'] >= x for x in q)
    ov = min(m['plain_overlap_mass'], 2)
    t = T[(st, ov)]
    t[0] += 1
    t[1] += L[k+('none',)]
    t[2] += L[k+('mass',)]
    t[3] += L[k+('own',)]
    t[4] += (not L[k+('none',)]) and L[k+('mass',)]
print('quartile overlap(0,1,2+): rows none mass own induced')
for key in sorted(T):
    print(key, T[key])
high = [T[k] for k in T if k[0] == 3 and k[1] == 2]
rest = [T[k] for k in T if not (k[0] == 3 and k[1] == 2)]
print('top quartile with 2+ plain tokens: rows none mass own', [sum(t[i] for t in high) for i in range(4)])
print('all other rows: rows none mass own', [sum(t[i] for t in rest) for i in range(4)])
# 3. leave-one-slot-out at layer 24, and all slots for the own cut
R = [(m['arm'], m['layer'], m['slot'], m['question_index']) for m in M]


def cut(rows):
    n = sum(L[r+('none',)] for r in rows)
    o = sum(L[r+('own',)] for r in rows)
    return 1-o/n, n, o


print('all', cut(R))
print('without layer 24', cut([r for r in R if r[1] != 24]))
for s in range(12):
    x = cut([r for r in R if not (r[1] == 24 and r[2] == s)])
    y = [r for r in R if r[1] == 24 and r[2] == s]
    print('L24 slot', s, 'cut without it', round(x[0], 3), 'slot none', sum(L[r+('none',)] for r in y), 'own', sum(L[r+('own',)] for r in y))
vals = sorted(((round(cut([r for r in R if not (r[1] == l and r[2] == s)])[0], 3), l, s) for l in range(19, 27) for s in range(12)))
print('lowest five leave-one-slot cuts', vals[:5], 'highest', vals[-2:])
# gross counts
for b in ('own', 'matched', 'mass'):
    e = sum(L[r+('none',)] and not L[r+(b,)] for r in R)
    i = sum((not L[r+('none',)]) and L[r+(b,)] for r in R)
    print(b, 'ended', e, 'induced', i, 'induced per none loop', round(i/188, 3), 'per eligible', round(i/1156, 4))

# 4. Weaver's split: does the own cut track own mass, and the mass rise track plain-token hit?
order = sorted(M, key=lambda m: m['own_mass'])
print('own-mass tenths, hit = 2+ plain tokens on the mass list; hit: rows none mass own | no hit: rows none mass own')
tot = {0: [0, 0, 0, 0], 1: [0, 0, 0, 0]}
for i in range(10):
    part = order[len(order)*i//10:len(order)*(i+1)//10]
    c = {0: [0, 0, 0, 0], 1: [0, 0, 0, 0]}
    for m in part:
        k = (m['arm'], m['layer'], m['slot'], m['question_index'])
        h = int(m['plain_overlap_mass'] >= 2)
        c[h][0] += 1
        c[h][1] += L[k+('none',)]
        c[h][2] += L[k+('mass',)]
        c[h][3] += L[k+('own',)]
    print(i, 'from', float('%.3g' % part[0]['own_mass']), c[1], '|', c[0])
    if min(c[0][0], c[1][0]) >= 15:
        for h in (0, 1):
            tot[h] = [a+b for a, b in zip(tot[h], c[h])]
print('tenths with at least 15 rows on both sides, hit:', tot[1], 'no hit:', tot[0])
# 5. do the same slots carry both arms? k4 contains its slot's n+ neuron
slots = defaultdict(lambda: defaultdict(int))
for r in R:
    s = (r[1], r[2])
    slots[s][r[0]+'_none'] += L[r+('none',)]
    slots[s][r[0]+'_own'] += L[r+('own',)]


def corr(x, y):
    n = len(x)
    mx, my = sum(x)/n, sum(y)/n
    sx = sum((a-mx)**2 for a in x)**.5
    sy = sum((b-my)**2 for b in y)**.5
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/(sx*sy)


ss = sorted(slots)
print('slot corr n+ none with k4 none', round(corr([slots[s]['n+_none'] for s in ss], [slots[s]['k4_none'] for s in ss]), 3))
print('slot corr n+ ended with k4 ended', round(corr([slots[s]['n+_none']-slots[s]['n+_own'] for s in ss], [slots[s]['k4_none']-slots[s]['k4_own'] for s in ss]), 3))
print('L24 slots none n+ k4, own n+ k4', [(s[1], slots[s]['n+_none'], slots[s]['k4_none'], slots[s]['n+_own'], slots[s]['k4_own']) for s in ss if s[0] == 24])
