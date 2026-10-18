"""score_part18c.py: score Part 18c (teacher-forced creator assay) under the amended reading rule.

Reads the run's rows (results/part18c/teacherforce.json) and recomputes, from rows alone, every quantity the predictions
in PREDICTIONS.md (Part 18c, with the amendment of 2026-10-18 16:23 CT) need, including the ratio intervals the run's own
summaries do not carry. Primary scale: mean token log-probability; the sum is reported beside it.

Definitions (PART18C-DESIGN.md): Δ(c,t,s) = mean ET score minus mean earthly score; Δ(c,t) = mean over stems;
ΔΔ(c,t) = Δ(c,t) - Δ(unclamped,t); a set's ΔΔ(c) = mean over templates. Bootstrap: 2,000 resamples of templates, seed 0.
Paired comparisons within one set (P against A) resample templates once. Cross-set comparisons (self against other or
unrelated) resample each set's templates independently (the amendment: unpaired).

Self-check: the recomputed set means must equal the run's own summaries to 1e-9, or the script stops.
"""
import json, sys, random, statistics
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else 'results/part18c/teacherforce.json'
d = json.load(open(path, encoding='utf-8'))
SCALES = ('mean_logprob', 'sum_logprob')

# acc[scale][(set, cond, t)][stem][kind] -> list of scores
acc = {s: defaultdict(lambda: defaultdict(lambda: defaultdict(list))) for s in SCALES}
for r in d['rows']:
    key = (r['set'], r['condition'], int(r['template_index']))
    for s in SCALES:
        acc[s][key][int(r['stem_index'])][r['kind']].append(float(r[s]))

def delta(s, st, c, t):
    stems = acc[s][(st, c, t)]
    vals = [statistics.fmean(v['et']) - statistics.fmean(v['earthly']) for v in stems.values()]
    return statistics.fmean(vals)

templates = {st: sorted({t for (st2, c, t) in acc['mean_logprob'] if st2 == st}) for st in ('self', 'other', 'unrelated')}

def dd(s, st, c):
    """Per-template ΔΔ list for a set and condition."""
    return [delta(s, st, c, t) - delta(s, st, 'unclamped', t) for t in templates[st]]

B, SEED = 2000, 0
def ci(xs):
    xs = sorted(xs); return [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]]
def boot_mean(v):
    rng = random.Random(SEED); n = len(v)
    return ci([statistics.fmean(rng.choice(v) for _ in range(n)) for _ in range(B)])
def boot_paired_ratio(num, den):
    rng = random.Random(SEED); n = len(num); out = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        out.append(statistics.fmean(num[i] for i in idx) / statistics.fmean(den[i] for i in idx))
    return ci(out)
def boot_unpaired(a, b, op):
    rng = random.Random(SEED); out = []
    for _ in range(B):
        ma = statistics.fmean(rng.choice(a) for _ in range(len(a)))
        mb = statistics.fmean(rng.choice(b) for _ in range(len(b)))
        out.append(op(ma, mb))
    return ci(out)

# Self-check against the run's own summaries.
summ = d['summaries']['delta_delta']
bad = 0
for st in ('self', 'other', 'unrelated'):
    for c, v in summ[st].items():
        for s in SCALES:
            m = statistics.fmean(dd(s, st, c))
            if abs(m - v[s]['mean']) > 1e-9: bad += 1
if bad: sys.exit(f'self-check failed: {bad} set means differ from the run summaries')

T = lambda k, ph: f'target:k={k}:phase={ph}'
out = {'source': path, 'bootstrap': {'resamples': B, 'seed': SEED, 'unit': 'templates'}, 'self_check': 'set means equal run summaries (1e-9)'}
res = {}
for s in SCALES:
    r = {}
    m = lambda st, c: statistics.fmean(dd(s, st, c))
    # 18c.1
    ks = ['0.5', '1', '1.5', '2', '3']
    seq = [m('self', T(k, 'A')) for k in ks]
    a2 = dd(s, 'self', T('2', 'A')); a2ci = boot_mean(a2)
    r['18c.1'] = {'self_A_by_k': dict(zip(ks, seq)), 'increasing': all(x < y for x, y in zip(seq, seq[1:])),
                  'k2_A_ci': a2ci, 'held': all(x < y for x, y in zip(seq, seq[1:])) and a2ci[0] > 0}
    ratios_evaluable = a2ci[0] > 0
    # 18c.2, 18c.3 (paired within self)
    p2, g2 = dd(s, 'self', T('2', 'P')), dd(s, 'self', T('2', 'G'))
    rp = statistics.fmean(p2) / statistics.fmean(a2); rg = statistics.fmean(g2) / statistics.fmean(a2)
    r['18c.2'] = {'P_over_A': rp, 'interval': boot_paired_ratio(p2, a2), 'evaluable': ratios_evaluable, 'held_as_written': rp < 0.25}
    r['18c.3'] = {'G_over_A': rg, 'interval': boot_paired_ratio(g2, a2), 'evaluable': ratios_evaluable, 'held_as_written': rg >= 0.75}
    # 18c.4, 18c.8 (unpaired across sets)
    o2, u2 = dd(s, 'other', T('2', 'A')), dd(s, 'unrelated', T('2', 'A'))
    for row, other in (('18c.4', o2), ('18c.8', u2)):
        rr = statistics.fmean(other) / statistics.fmean(a2); iv = boot_unpaired(other, a2, lambda x, y: x / y)
        inside = 0.6 <= iv[0] and iv[1] <= 1.4; outside = iv[1] < 0.6 or iv[0] > 1.4
        r[row] = {'ratio': rr, 'interval': iv, 'evaluable': ratios_evaluable,
                  'verdict': 'held' if inside else ('failed' if outside else 'inconclusive')}
    # 18c.5 and the amended creator-specific test (unpaired self minus other at P, k=2)
    po2 = dd(s, 'other', T('2', 'P'))
    diff = statistics.fmean(p2) - statistics.fmean(po2); div = boot_unpaired(p2, po2, lambda x, y: x - y)
    r['18c.5'] = {'self_minus_other_P': diff, 'ci': div, 'held_as_written': div[0] <= 0 <= div[1]}
    r['creator_specific_test'] = {'self_minus_other_P': diff, 'ci': div, 'threshold': 0.25 * statistics.fmean(a2),
                                  'passes': div[0] > 0 and diff >= 0.25 * statistics.fmean(a2)}
    # 18c.7: each control at A, k=2, own and column-norm-scaled
    tgt = statistics.fmean(a2); ctl = {}
    for c in d['conditions']:
        if c.startswith('control:') and ':phase=A:' in c:
            ctl[c] = m('self', c)
    worst = max(ctl, key=lambda c: abs(ctl[c]))
    r['18c.7'] = {'target': tgt, 'controls': ctl, 'max_abs_over_target': abs(ctl[worst]) / abs(tgt), 'worst': worst,
                  'held': all(abs(v) < 0.25 * abs(tgt) for v in ctl.values())}
    # Context: every target condition's set means
    r['set_means'] = {st: {c: m(st, c) for c in d['conditions'] if c.startswith('target') or c == 'unclamped'}
                      for st in ('self', 'other', 'unrelated')}
    res[s] = r
out['results'] = res
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else 'results/part18c/scores.json', 'w', encoding='utf-8'), indent=1)
p = res['mean_logprob']
for k in ('18c.1', '18c.2', '18c.3', '18c.4', '18c.5', 'creator_specific_test', '18c.7', '18c.8'):
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else ([round(x, 4) for x in vv] if isinstance(vv, list) else vv))
               for kk, vv in p[k].items() if kk not in ('controls', 'self_A_by_k')})
print('self_A_by_k', {k: round(v, 4) for k, v in p['18c.1']['self_A_by_k'].items()})
