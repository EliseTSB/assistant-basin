"""Cold check of Part 169, part 3: remaining README numbers, hit-row sets, bite table, strength match,
plain50 induced vs none-length, scorer entries. Run from the repo root:
  /home/elisepc/neuron-probe/.venv/bin/python -I -B cold169/recompute_part3.py
Writes nothing.
"""
import json, sys, statistics, collections
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
NP = []
for a in answers:
    if a.get('stage') == 'battery':
        B[rk(a) + (a['ban'],)] = a
    elif a.get('stage') == 'no_push':
        NP.append(a)
L = {k: bool(score166.loop_phrase(a['answer'])) for k, a in B.items()}
BR = {k: bool(score165.primary_broken(a['answer'])) for k, a in B.items()}
allrows = list(M)

def pooled_stats(ban):
    ks = [k for k in B if k[-1] == ban]
    n = len(ks)
    return dict(n=n, loops=sum(L[k] for k in ks), broken=round(100 * sum(BR[k] for k in ks) / n, 1),
                joint=round(100 * sum((not L[k]) and (not BR[k]) for k in ks) / n, 1),
                length=round(statistics.mean(B[k]['token_count'] for k in ks), 2),
                eos=round(100 * sum(bool(B[k].get('ended_eos_before_80')) for k in ks) / n, 1))
print('== pooled (README table, both arms)')
for b in ('none', 'own', 'free', 'clear', 'plain50', 'hit'):
    print(b, pooled_stats(b))

print('\n== per-arm bootstrap loop-rate intervals (score.json) vs README')
bs = score['bootstrap_intervals']
for arm in ('n+', 'k4'):
    for b in ('none', 'own', 'free', 'clear', 'plain50', 'hit'):
        v = bs[arm][b + '_loop_rate']
        print(arm, b, round(100 * v['estimate'], 1), [round(100 * x, 1) for x in v['interval_95']])
print('169.1 per arm:', {a: round(bs[a]['169.1']['estimate'], 4) for a in ('n+', 'k4')})
print('169.3 per arm:', {a: [round(bs[a]['169.3']['estimate'], 4)] for a in ('n+', 'k4')})

# Hit-defined rows
print('\n== hit-defined rows (rows with nonempty hit list)')
hit_rows = [k for k in allrows if M[k]['lists']['hit']['ids']]
for arm in ('n+', 'k4', None):
    rs = [k for k in hit_rows if arm is None or k[0] == arm]
    print(arm or 'pooled', len(rs), 'none', sum(L[k + ('none',)] for k in rs), 'free', sum(L[k + ('free',)] for k in rs), 'hit', sum(L[k + ('hit',)] for k in rs))
print('hit rows net per scorer: hit_relative_cut', round(score['bootstrap_intervals']['hit_relative_cut']['estimate'], 4))
print('ended_induced_net.hit', score['ended_induced_net']['hit'])
hn = sum(L[k + ('none',)] for k in hit_rows); hh = sum(L[k + ('hit',)] for k in hit_rows)
print('same-row hit net loops', hh - hn, 'pp', round(100 * (hh - hn) / len(hit_rows), 2))
print('scorer hit net_loop_change (cross row-set):', round(100 * score['ended_induced_net']['hit']['net_loop_change'], 2), 'pp')
print('row-set mismatch in hit_relative_cut: all-row none rate', round(100 * sum(L[k+('none',)] for k in allrows)/len(allrows),2), 'hit-row none rate', round(100*sum(L[k+('none',)] for k in hit_rows)/len(hit_rows),2), 'hit rate', round(100*sum(L[k+('hit',)] for k in hit_rows)/len(hit_rows),2))

# hit share of free mass, hit sizes
ratios = sorted(M[k]['lists']['hit']['mass'] / M[k]['lists']['free']['mass'] for k in hit_rows)
print('\nhit_share_of_free_mass: n', len(ratios), 'median', round(statistics.median(ratios), 4), 'upper-median used by scorer', round(ratios[len(ratios)//2], 4))
sizes = [len(M[k]['lists']['hit']['ids']) for k in hit_rows]
print('hit sizes mean', round(statistics.mean(sizes), 3), 'median', statistics.median(sizes), 'max', max(sizes))
p2 = [k for k in allrows if int(M[k]['lists']['free']['plain_overlap']) >= 2]
print('plain>=2 rows', len(p2), 'mean hit size', round(statistics.mean(len(M[k]['lists']['hit']['ids']) for k in p2), 3))
print('hit ratio on plain>=2 rows: median', round(statistics.median(M[k]['lists']['hit']['mass']/M[k]['lists']['free']['mass'] for k in p2),4))

# Strength-matched 169.4 subset (scorer rule: |top200 free - top200 clear| <= 1)
paired = [k for k in allrows if M[k]['paired_set']]
print('\npaired rows', len(paired), 'high paired', sum(1 for k in paired if M[k]['high_row']))
sm = [k for k in paired if abs(int(M[k]['lists']['free']['top200']) - int(M[k]['lists']['clear']['top200'])) <= 1]
def rate(ban, ks):
    return sum(L[k + (ban,)] for k in ks) / len(ks), sum(L[k + (ban,)] for k in ks), len(ks)
f_all = rate('free', paired); c_all = rate('clear', paired)
print('paired free', f_all, 'clear', c_all, 'diff pp', round(100 * (f_all[0] - c_all[0]), 2))
f_sm = rate('free', sm); c_sm = rate('clear', sm)
print('strength-matched rows', len(sm), 'free', round(100*f_sm[0],2), 'clear', round(100*c_sm[0],2), 'diff pp', round(100*(f_sm[0]-c_sm[0]),2), 'free loops', f_sm[1], 'clear loops', c_sm[1])
print('  169.4 full gap pp', round(100*(f_all[0]-c_all[0]),2), ' |strength gap| / |full gap| =', round(abs(f_sm[0]-c_sm[0])/abs(f_all[0]-c_all[0]),2))

# 169.8 bar check, net view
hp2 = [k for k in p2]
print('\nplain>=2 rows, none/free/hit loops:', sum(L[k+('none',)] for k in hp2), sum(L[k+('free',)] for k in hp2), sum(L[k+('hit',)] for k in hp2))
ind_h = [k for k in hp2 if not L[k+('none',)]]
print('  induced eligible', len(ind_h), 'hit induced', sum(L[k+('hit',)] for k in ind_h), 'free induced', sum(L[k+('free',)] for k in ind_h))
ended_h = [k for k in hp2 if L[k+('none',)] and not L[k+('hit',)]]
print('  hit ended among plain2 rows', len(ended_h), ' free ended', sum(1 for k in hp2 if L[k+('none',)] and not L[k+('free',)]))

# Ban bite table
print('\n== ban bite (exploratory)')
for b in ('own', 'free', 'clear', 'hit', 'plain50'):
    xs = [a for a in answers if a.get('stage') == 'battery' and a.get('ban') == b]
    ranks = sorted(r for a in xs for r in a.get('blocked_ranks', []))
    firsts = sorted(a['blocked_steps'][0] for a in xs if a.get('blocked_steps'))
    print(b, 'answers', len(xs), 'never', sum(not a.get('blocked_steps') for a in xs), 'mean blocked', round(statistics.mean(len(a.get('blocked_steps', [])) for a in xs), 2),
          'median first', firsts[len(firsts)//2], 'median rank', ranks[len(ranks)//2] if ranks else None,
          'share rank<=5', round(100 * sum(r <= 5 for r in ranks) / len(ranks), 2) if ranks else None)

# Batch-shape differences
print('\n== batch-shape: never-blocked answers that differ from none')
for b in ('own', 'free', 'clear'):
    nb = [k for k in allrows if not B[k+(b,)].get('blocked_steps')]
    diff = [k for k in nb if B[k+(b,)]['token_ids'] != B[k+('none',)]['token_ids']]
    ended = sum(L[k+('none',)] and not L[k+(b,)] for k in diff)
    ind = sum(L[k+(b,)] and not L[k+('none',)] for k in diff)
    print(b, 'never blocked', len(nb), 'differ', len(diff), f'({100*len(diff)/len(nb):.2f}%)', 'ended', ended, 'induced', ind)

# No-push
print('\n== no-push (stage no_push)')
nop = collections.defaultdict(list)
for a in NP:
    nop[(a['arm'], a.get('ban'))].append(a)
for key, v in sorted(nop.items(), key=lambda x: str(x[0])):
    print(key, len(v), 'loops', sum(bool(score166.loop_phrase(a['answer'])) for a in v), 'broken', sum(bool(score165.primary_broken(a['answer'])) for a in v))

# plain50 induced loops: length
ind50 = [k for k in allrows if L[k+('plain50',)] and not L[k+('none',)]]
print('\nplain50 induced', len(ind50), 'mean len', round(statistics.mean(B[k+('plain50',)]['token_count'] for k in ind50), 1),
      'plain50 loops that are not induced: those with none loop too', sum(1 for k in allrows if L[k+('plain50',)] and L[k+('none',)]))
print('none loops (all)', sum(L[k+('none',)] for k in allrows), 'mean len', round(statistics.mean(B[k+('none',)]['token_count'] for k in allrows if L[k+('none',)]),1))
print('plain50 loops mean len', round(statistics.mean(B[k+('plain50',)]['token_count'] for k in allrows if L[k+('plain50',)]),1))
print('plain50 non-loop mean len', round(statistics.mean(B[k+('plain50',)]['token_count'] for k in allrows if not L[k+('plain50',)]),1))
print('none non-loop mean len', round(statistics.mean(B[k+('none',)]['token_count'] for k in allrows if not L[k+('none',)]),1))

# own-induced by own length, and the short-none rows
print('\n== own-induced loops: own answer lengths')
ind_o = [k for k in allrows if L[k+('own',)] and not L[k+('none',)]]
print('own induced', len(ind_o), 'own len median', statistics.median(B[k+('own',)]['token_count'] for k in ind_o), 'none len median of same rows', statistics.median(B[k+('none',)]['token_count'] for k in ind_o))
short_none = [k for k in allrows if B[k+('none',)]['token_count'] < 45]
print('rows with none<45 tokens:', len(short_none), 'none loops', sum(L[k+('none',)] for k in short_none), 'own loops', sum(L[k+('own',)] for k in short_none))
for k in short_none:
    pass
print('  their own answer lengths: median', statistics.median(B[k+('own',)]['token_count'] for k in short_none), ' own loops with own len<45:', sum(1 for k in short_none if L[k+('own',)] and B[k+('own',)]['token_count']<45), ' own loops with own len>=45:', sum(1 for k in short_none if L[k+('own',)] and B[k+('own',)]['token_count']>=45))
