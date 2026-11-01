"""Cold check of Part 169, part 2: fragility, tenths, length/EOS, plain50 samples, scorer entries.
Run from the repo root: /home/elisepc/neuron-probe/.venv/bin/python -I -B cold169/recompute_part2.py
Writes nothing. Shares the loading code with recompute_core.py (duplicated here on purpose so each runs alone).
"""
import json, sys, collections, random, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from part165 import score165
from part166 import score166

P = ROOT / 'results/part169'
def jl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
masses = jl(P / 'masses.jsonl')
answers = jl(P / 'answers.jsonl')
score = json.loads((P / 'score.json').read_text(encoding='utf-8'))
def rk(r):
    return (r['arm'], int(r['layer']), int(r['slot']), int(r['question_index']))
M = {rk(m): m for m in masses}
B = {}
for a in answers:
    if a.get('stage') == 'battery':
        B[rk(a) + (a['ban'],)] = a
L = {k: bool(score166.loop_phrase(a['answer'])) for k, a in B.items()}
BR = {k: bool(score165.primary_broken(a['answer'])) for k, a in B.items()}
rows_all = list(M.keys())
ban_of = lambda k, b: B[k + (b,)]
def loops(ban, keys):
    return sum(L[k + (ban,)] for k in keys)
def cut(keys, ban='own'):
    n0 = loops('none', keys); n1 = loops(ban, keys)
    return (1 - n1 / n0) if n0 else None, n0, n1

print('== 169.1 cut, all rows:', cut(rows_all))
print('\n== Leave one layer out')
for lay in range(19, 27):
    ks = [k for k in rows_all if k[1] != lay]
    print(' drop layer', lay, round(cut(ks)[0], 4))
print('\n== Leave one phrasing out (question_index)')
for q in range(8):
    ks = [k for k in rows_all if k[3] != q]
    print(' drop phrasing', q, round(cut(ks)[0], 4), cut(ks)[1:])
print('phrasing alone (own cut within each phrasing):')
for q in range(8):
    ks = [k for k in rows_all if k[3] == q]
    c = cut(ks)
    print('  phr', q, 'none', c[1], 'own', c[2], 'cut', round(c[0], 3))
print('\n== n+ alone', round(cut([k for k in rows_all if k[0] == 'n+'])[0], 4), ' k4 alone', round(cut([k for k in rows_all if k[0] == 'k4'])[0], 4))
print('\n== n+ and k4 without the other, per-slot leave-one-out min/max')
vals = []
for s in sorted({(k[1], k[2]) for k in rows_all}):
    ks = [k for k in rows_all if (k[1], k[2]) != s]
    vals.append((round(cut(ks)[0], 4), s))
vals.sort(); print(' lowest', vals[:3], 'highest', vals[-2:])

# Which rows carry the cut
ended = [k for k in rows_all if L[k + ('none',)] and not L[k + ('own',)]]
induced = [k for k in rows_all if not L[k + ('none',)] and L[k + ('own',)]]
print('\n== rows carrying the cut: ended', len(ended), 'induced', len(induced), 'net', len(ended) - len(induced))
slots_ended = collections.Counter((k[0], k[1], k[2]) for k in ended)
print(' distinct (arm,layer,slot) with an ended loop:', len(slots_ended), ' top 10 slots hold', sum(c for _, c in slots_ended.most_common(10)), 'of', len(ended))
qs_ended = collections.Counter(k[3] for k in ended)
print(' ended by phrasing:', dict(sorted(qs_ended.items())))
ended_per_slot = sorted(slots_ended.values(), reverse=True)
print(' slots with >=1 ended:', len(slots_ended), ' slots with >=4 ended:', sum(1 for v in ended_per_slot if v >= 4))
# how many rows does the net cut need: drop the k slots with the largest net cut
net = collections.Counter()
for k in rows_all:
    net[(k[0], k[1], k[2])] += (L[k + ('none',)] - L[k + ('own',)])
ranked = sorted(net.values(), reverse=True)
print(' net cut by slot, sorted (top 10):', ranked[:10], ' sum', sum(ranked), ' slots with net>0:', sum(1 for v in ranked if v > 0), 'slots with net<0:', sum(1 for v in ranked if v < 0))

# Length and EOS
print('\n== length and EOS by ban (all 1536 rows per ban)')
for b in ('none', 'own', 'free', 'clear', 'plain50'):
    ks = rows_all
    lens = [B[k + (b,)]['token_count'] for k in ks]
    eos = [bool(B[k + (b,)].get('ended_eos_before_80')) for k in ks]
    print(f' {b:7s} mean len {statistics.mean(lens):.2f} median {statistics.median(lens)} EOS-before-80 {sum(eos)/len(eos):.4f} answers<80 tokens {sum(x<80 for x in lens)}')
print('\n== loop rate by length bin (token_count) for none, own, plain50')
bins = [(0, 30), (30, 45), (45, 60), (60, 75), (75, 81)]
for b in ('none', 'own', 'plain50'):
    out = []
    for lo, hi in bins:
        ks = [k for k in rows_all if lo <= B[k + (b,)]['token_count'] < hi]
        out.append(f'[{lo},{hi}) n={len(ks)} loop={loops(b, ks)}')
    print(' ', b, ' | '.join(out))
print('\n== own cut among rows where none ends by EOS before 80 vs not')
for name, pred in (('none EOS<80', lambda k: bool(B[k + ('none',)].get('ended_eos_before_80'))), ('none reached 80', lambda k: not B[k + ('none',)].get('ended_eos_before_80'))):
    ks = [k for k in rows_all if pred(k)]
    print(' ', name, cut(ks))
print('\n== mean length of answers, none loop vs no loop (none) and own loop vs no loop (own)')
for b in ('none', 'own'):
    for flag in (True, False):
        ks = [k for k in rows_all if L[k + (b,)] == flag]
        if ks:
            print(' ', b, 'loop' if flag else 'noloop', len(ks), 'mean len', round(statistics.mean(B[k + (b,)]['token_count'] for k in ks), 2))
# length-matched own cut: among rows with none token_count within each bin, compare none and own loop counts
print('\n== own cut by none-length bin (rows binned by the none answer length)')
for lo, hi in bins:
    ks = [k for k in rows_all if lo <= B[k + ('none',)]['token_count'] < hi]
    c = cut(ks)
    print(f'  [{lo},{hi}) rows {len(ks)} none {c[1]} own {c[2]} cut {None if c[0] is None else round(c[0], 3)}')

# Tenths
print('\n== own-mass tenths: none, own, cut, cut share of none loops')
S = sorted(rows_all, key=lambda k: M[k]['lists']['own']['mass'])
tot = 0
for i in range(10):
    p = S[i * len(S) // 10:(i + 1) * len(S) // 10]
    n0 = loops('none', p); n1 = loops('own', p)
    nf = loops('free', p)
    tot += n0 - n1
    print(f' tenth {i} rows {len(p)} none {n0} own {n1} free {nf} cut {n0-n1} share_of_none {(n0-n1)/n0 if n0 else float("nan"):.2f} rel_cut {(1-n1/n0) if n0 else float("nan"):.3f}')
print(' total cut', tot)
top2 = S[8 * len(S) // 10:]
print(' top two tenths cut', loops('none', top2) - loops('own', top2), 'of', loops('none', rows_all) - loops('own', rows_all))
# within top 2 tenths: is the cut driven by row count with loops? ended/induced
print(' top2 ended', sum(L[k+('none',)] and not L[k+('own',)] for k in top2), 'induced', sum(L[k+('own',)] and not L[k+('none',)] for k in top2))

# Overall loops in the none arm by tenth share vs the cut share
print('\n== plain50 length and loops by phrasing')
for q in range(8):
    ks = [k for k in rows_all if k[3] == q]
    print(' phr', q, 'none', loops('none', ks), 'plain50', loops('plain50', ks), 'free', loops('free', ks), 'own', loops('own', ks), 'plain50 mean len', round(statistics.mean(B[k+('plain50',)]['token_count'] for k in ks), 1))
