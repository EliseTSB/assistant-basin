"""Cold check of Part 169, part 4: length/EOS standardization for plain50 and own, own-induced rows,
and the 169.1 / 169.2 high-vs-other split. Run from the repo root:
  /home/elisepc/neuron-probe/.venv/bin/python -I -B cold169/recompute_part4.py
Writes nothing.
"""
import json, sys, statistics, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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
EOS = {k: bool(a.get('ended_eos_before_80')) for k, a in B.items()}
LEN = {k: a['token_count'] for k, a in B.items()}

def lbin(n):
    return 0 if n < 30 else 1 if n < 45 else 2 if n < 60 else 3 if n < 75 else 4

# 1. Direct standardization: expected loops for ban X if its loop probability per row followed the
#    none arm's loop rate in the same (length bin, EOS) cell.
cell_none = collections.defaultdict(lambda: [0, 0])
for k in allrows:
    c = (lbin(LEN[k + ('none',)]), EOS[k + ('none',)])
    cell_none[c][0] += L[k + ('none',)]
    cell_none[c][1] += 1
def expected_from_none(ban):
    e = 0.0
    for k in allrows:
        c = (lbin(LEN[k + (ban,)]), EOS[k + (ban,)])
        loops_c, n_c = cell_none[c]
        e += (loops_c / n_c) if n_c else 0.0
    return e
print('== length and EOS standardization (cells: length bin x EOS, rates from none)')
obs = {b: sum(L[k + (b,)] for k in allrows) for b in ('none', 'own', 'free', 'clear', 'plain50')}
for b in ('own', 'free', 'clear', 'plain50'):
    e = expected_from_none(b)
    print(f' {b:7s} observed {obs[b]:4d}  expected if loop prob depended only on none-length+EOS cell {e:7.1f}  residual {obs[b]-e:7.1f}')

# 2. Among plain50 answers, how does loop rate vary with EOS
print('\n== EOS share among loop answers, by ban')
for b in ('none', 'own', 'free', 'clear', 'plain50'):
    lp = [k for k in allrows if L[k + (b,)]]
    print(f' {b:7s} loops {len(lp):4d} EOS-before-80 among loops {sum(EOS[k+(b,)] for k in lp)} ({100*sum(EOS[k+(b,)] for k in lp)/max(1,len(lp)):.1f}%)')

# 3. Plain50 loop rate by none-length bin (same row set as none), to see if the extra loops sit in one length band
print('\n== plain50 vs none loop rate by none-length band (same rows)')
for lo, hi in ((0, 30), (30, 45), (45, 60), (60, 75), (75, 81)):
    ks = [k for k in allrows if lo <= LEN[k + ('none',)] < hi]
    print(f' none in [{lo},{hi}) rows {len(ks)} none loops {sum(L[k+("none",)] for k in ks)} plain50 loops {sum(L[k+("plain50",)] for k in ks)} own loops {sum(L[k+("own",)] for k in ks)}')

# 4. Own: what happens to the 182 ended loops in length and EOS
ended = [k for k in allrows if L[k + ('none',)] and not L[k + ('own',)]]
kept = [k for k in allrows if L[k + ('none',)] and L[k + ('own',)]]
induced = [k for k in allrows if not L[k + ('none',)] and L[k + ('own',)]]
print('\n== own: ended / kept / induced')
print(' ended', len(ended), 'own len median', statistics.median(LEN[k+("own",)] for k in ended), 'none len median', statistics.median(LEN[k+("none",)] for k in ended),
      'own EOS share', round(sum(EOS[k+("own",)] for k in ended)/len(ended),3))
print(' kept', len(kept), 'own len median', statistics.median(LEN[k+("own",)] for k in kept), 'none len median', statistics.median(LEN[k+("none",)] for k in kept))
print(' induced', len(induced), 'own len median', statistics.median(LEN[k+("own",)] for k in induced), 'none len median', statistics.median(LEN[k+("none",)] for k in induced))
short_none = [k for k in allrows if LEN[k + ('none',)] < 45]
ind_short = [k for k in short_none if k in set(induced)]
print(' induced rows whose none answer was <45 tokens:', len(ind_short), 'of', len(induced), '; those rows are', len(short_none), 'of 1536')
print(' among those, own answer lengths:', sorted(LEN[k+("own",)] for k in ind_short))

# 5. Length-free view of the own cut: rows where own and none have the same length bin
same = [k for k in allrows if lbin(LEN[k+("none",)]) == lbin(LEN[k+("own",)]) and EOS[k+("none",)] == EOS[k+("own",)]]
n0 = sum(L[k+("none",)] for k in same); n1 = sum(L[k+("own",)] for k in same)
print('\n== own cut on rows whose own answer has the same length band and EOS flag as none:', len(same), 'rows, none', n0, 'own', n1, 'cut', round(1-n1/n0,3) if n0 else None)
diff = [k for k in allrows if k not in set(same)]
n0d = sum(L[k+("none",)] for k in diff); n1d = sum(L[k+("own",)] for k in diff)
print(' rows where length band or EOS changed:', len(diff), 'none', n0d, 'own', n1d, 'cut', round(1-n1d/n0d,3) if n0d else None)

# 6. High vs other split with the raw counts
high = [k for k in allrows if M[k]['high_row']]
other = [k for k in allrows if not M[k]['high_row']]
for name, ks in (('high', high), ('other', other)):
    n0 = sum(L[k+("none",)] for k in ks); n1 = sum(L[k+("own",)] for k in ks)
    print(f'{name} rows {len(ks)} none {n0} own {n1} own cut {round(1-n1/n0,4)}')
print('own cut share of loops in top two tenths is computed in recompute_part2.py')
