"""Cold check of Part 169: recompute the registered numbers from raw files.

Run from the repo root:
  /home/elisepc/neuron-probe/.venv/bin/python -I -B cold169/recompute_core.py
Reads only results/part169 and part165/part166 code. Writes nothing.
"""
import json, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from part165 import score165
from part166 import score166

P = ROOT / 'results/part169'
def jl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

rows = jl(P / 'rows.jsonl')
masses = jl(P / 'masses.jsonl')
answers = jl(P / 'answers.jsonl')
score = json.loads((P / 'score.json').read_text(encoding='utf-8'))

def rk(r):
    return (r['arm'], int(r['layer']), int(r['slot']), int(r['question_index']))

M = {rk(m): m for m in masses}
battery = {}
no_push = []
for a in answers:
    if a.get('stage') == 'battery':
        battery[rk(a) + (a['ban'],)] = a
    elif a.get('stage') == 'no_push':
        no_push.append(a)

ans_loop = {}
ans_broken = {}
for k, a in battery.items():
    ans_loop[k] = bool(score166.loop_phrase(a['answer']))
    ans_broken[k] = bool(score165.primary_broken(a['answer']))

ARMS = ('n+', 'k4')
BANS = ('none', 'own', 'free', 'clear', 'plain50', 'hit')
def keys_for(ban, arm=None):
    return [k for k in battery if k[-1] == ban and (arm is None or k[0] == arm)]

print('battery answers', len(battery), 'no_push', len(no_push))
print('masses', len(M), 'high', sum(bool(m['high_row']) for m in masses), 'paired', sum(bool(m['paired_set']) for m in masses))

# 1. Per arm/ban outcomes vs score.json
print('\n## per_arm_ban check (recomputed vs score.json)')
mism = []
for arm in ARMS:
    for ban in BANS:
        ks = keys_for(ban, arm)
        n = len(ks)
        if not n:
            continue
        loops = sum(ans_loop[k] for k in ks)
        brk = sum(ans_broken[k] for k in ks)
        joint = sum((not ans_loop[k]) and (not ans_broken[k]) for k in ks)
        mt = sum(battery[k]['token_count'] for k in ks) / n
        eos = sum(bool(battery[k].get('ended_eos_before_80')) for k in ks) / n
        s = score['per_arm_ban'][arm][ban]
        ok = (s['n'] == n and abs(s['loop_rate'] - loops / n) < 1e-9 and abs(s['broken_rate'] - brk / n) < 1e-9
              and abs(s['joint_success_rate'] - joint / n) < 1e-9 and abs(s['mean_tokens'] - mt) < 1e-6 and abs(s['eos_rate'] - eos) < 1e-9)
        print(f'{arm:4s} {ban:7s} n={n:4d} loops={loops:4d} loop={loops/n:.4f} broken={brk/n:.4f} joint={joint/n:.4f} len={mt:.2f} eos={eos:.4f} match={ok}')
        if not ok:
            mism.append((arm, ban))
print('mismatches vs score.json:', mism)

def pooled(ban, subset=None):
    ks = [k for k in keys_for(ban) if (subset is None or subset(k))]
    return ks

print('\n## pooled loop counts (both arms)')
for ban in BANS:
    ks = pooled(ban)
    print(ban, len(ks), sum(ans_loop[k] for k in ks))

# Row sets
def rowkey(k):
    return k[:4]
none_loop = {rowkey(k): ans_loop[k] for k in keys_for('none')}
def is_high(k):
    return bool(M[rowkey(k)]['high_row'])
def is_paired(k):
    return bool(M[rowkey(k)]['paired_set'])
def plain_ov(k):
    return int(M[rowkey(k)]['lists']['free']['plain_overlap'])

def rate_of(ban, rows_pred=lambda k: True, induced=False):
    ks = [k for k in keys_for(ban) if rows_pred(k) and (not induced or not none_loop[rowkey(k)])]
    n = len(ks)
    c = sum(ans_loop[k] for k in ks)
    return c, n, (c / n if n else None)

print('\n## registered predictions (recomputed)')
c_none, n_none, _ = rate_of('none')
c_own, _, _ = rate_of('own')
print('169.1 own relative cut', 1 - c_own / c_none, '(none', c_none, 'own', c_own, ')')

hi_none = rate_of('none', is_high)
hi_own = rate_of('own', is_high)
ot_none = rate_of('none', lambda k: not is_high(k))
ot_own = rate_of('own', lambda k: not is_high(k))
cut_hi = 1 - hi_own[0] / hi_none[0]
cut_ot = 1 - ot_own[0] / ot_none[0]
print('169.2 high cut', round(cut_hi, 4), hi_none, hi_own, 'other cut', round(cut_ot, 4), ot_none, ot_own, 'diff pp', round(100 * (cut_hi - cut_ot), 2))

c_fr, n_fr, _ = rate_of('free')
print('169.3 free', c_fr, 'vs none', c_none, 'rise', round(c_fr / c_none - 1, 4))

pf = rate_of('free', is_paired)
pc = rate_of('clear', is_paired)
pn = rate_of('none', is_paired)
print('paired rows n', pf[1], 'free', pf[0], round(pf[2], 4), 'clear', pc[0], round(pc[2], 4), 'none', pn[0], round(pn[2], 4))
print('169.4 free-clear (pp)', round(100 * (pf[2] - pc[2]), 2))
print('169.7 paired clear relcut', round(1 - pc[2] / pn[2], 4))

hp = lambda k: is_paired(k) and is_high(k)
fi = rate_of('free', hp, induced=True)
ci = rate_of('clear', hp, induced=True)
print('169.5 high paired induced: free', fi, 'clear', ci, 'diff pp', round(100 * (fi[2] - ci[2]), 2))
hpall = rate_of('free', hp)
print('  high paired rows total (free sets)', hpall[1], 'none loops', rate_of('none', hp)[0], 'free loops', hpall[0], 'clear loops', rate_of('clear', hp)[0], 'own loops', rate_of('own', hp)[0])
hu = lambda k: is_high(k) and not is_paired(k)
print('  high unpaired rows', rate_of('free', hu)[1], 'none', rate_of('none', hu)[0], 'own', rate_of('own', hu)[0], 'free', rate_of('free', hu)[0], 'clear', rate_of('clear', hu)[0], 'free induced', rate_of('free', lambda k: hu(k), induced=True))

nopf = [a for a in no_push if a.get('stage') == 'no_push' and a.get('ban') == 'free']
print('169.6 no-push free', len(nopf), sum(bool(score166.loop_phrase(a['answer'])) for a in nopf))

p2 = lambda k: plain_ov(k) >= 2
h = rate_of('hit', p2, induced=True)
f = rate_of('free', p2, induced=True)
print('169.8 hit induced', h, 'free induced', f, 'bar 0.6*free', round(0.6 * f[2], 4), 'margin pp', round(100 * (h[2] - 0.6 * f[2]), 2))
print('  plain2 rows total', sum(1 for k in M if int(M[k]['lists']['free']['plain_overlap']) >= 2), 'with hit answers', len([k for k in keys_for('hit') if p2(k)]))
print('  plain2 rows no-loop-under-none', sum(1 for k in M if int(M[k]['lists']['free']['plain_overlap']) >= 2 and not none_loop[k]))
print('  plain2 net: none loops', rate_of('none', p2)[0], 'hit loops', rate_of('hit', p2)[0], 'free loops', rate_of('free', p2)[0], 'n', rate_of('hit', p2)[1])

# W1
w1_rows = []
none_tok = {rowkey(k): set(battery[k]['token_ids']) for k in keys_for('none')}
plain50_list = {k: set(M[k]['lists']['plain50']['ids']) for k in M}
ov50 = {k: len(none_tok[k] & plain50_list[k]) for k in M}
noloop_rows = [k for k in M if not none_loop[k]]
med = sorted(ov50[k] for k in noloop_rows)[len(noloop_rows) // 2]
above = lambda k: ov50[rowkey(k)] > med if False else ov50[k[:4]] > med
w1a = rate_of('plain50', lambda k: above(k) and not none_loop[k[:4]], induced=True)
w1b = rate_of('plain50', lambda k: (not above(k)) and not none_loop[k[:4]], induced=True)
print('\n169.W1 median overlap', med, 'above', w1a, 'at/below', w1b, 'diff pp', round(100 * (w1a[2] - w1b[2]), 2))
print('  plain50 induced total', rate_of('plain50', induced=True)[0], 'free induced total', rate_of('free', induced=True)[0])

# W2 (pooled, all rows; 'low-k' read as n+ and k4, which is every row in this part)
w2_rise = rate_of('plain50')[0] - c_none
w2_own = c_none - c_own
print('169.W2 plain50 net rise', w2_rise, 'own net cut', w2_own, 'third', round(w2_own / 3, 2))
