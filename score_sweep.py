#!/usr/bin/env python3
# score_sweep.py <dir> — one row per judge cell for the 2026-10-16 judge sweep (Claude).
# Each verdicts-<model>-<setting>.json in <dir> is a cell; key.json gives the arm of each unit (N neuron poles,
# R random directions, F random-context floor). Per cell: floor (share of F judged "one"; the pre-registered bar was
# under 10%), N "one", R "one", gap N-R with a two-proportion z, errors (parse/call), median output tokens, median
# reasoning tokens where the API reports them, median wall seconds, and agreement with the reference cell
# (claude-haiku-5-5-default) on the units both judged. Sorted by floor. Prints a markdown table; --json writes rows.
import json, math, os, statistics, sys

d = sys.argv[1]
REF = 'claude-haiku-5-5-default'
CATS = {'one', 'two_related', 'several_unrelated', 'unclear'}
key = json.load(open(os.path.join(d, 'key.json')))
cells = sorted(f[len('verdicts-'):-5] for f in os.listdir(d) if f.startswith('verdicts-') and f.endswith('.json'))
V = {c: json.load(open(os.path.join(d, f'verdicts-{c}.json'))) for c in cells}

def z2(k1, n1, k2, n2):
    if not n1 or not n2: return float('nan')
    p = (k1 + k2) / (n1 + n2); se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0
    return (k1 / n1 - k2 / n2) / se if se else float('nan')

def out_tokens(v):
    u = v.get('usage') or {}
    return u.get('output_tokens', u.get('completion_tokens'))

def reasoning(v):
    r = v.get('reasoning_tokens')
    return r if isinstance(r, (int, float)) else None

def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None

rows = []
for c in cells:
    vs = V[c]
    ok = {u: v for u, v in vs.items() if v.get('verdict') in CATS}
    errs = len(vs) - len(ok)
    share = {}
    counts = {}
    for arm in 'NRF':
        us = [u for u in ok if key[u]['arm'] == arm]
        k = sum(ok[u]['verdict'] == 'one' for u in us)
        counts[arm] = (k, len(us)); share[arm] = k / len(us) if us else float('nan')
    agree = None
    if c != REF and REF in V:
        both = [u for u in ok if V[REF].get(u, {}).get('verdict') in CATS]
        if both: agree = sum(ok[u]['verdict'] == V[REF][u]['verdict'] for u in both) / len(both)
    rows.append({
        'cell': c, 'n': len(vs), 'errors': errs,
        'floor': share['F'], 'N': share['N'], 'R': share['R'], 'gap': share['N'] - share['R'],
        'z': z2(counts['N'][0], counts['N'][1], counts['R'][0], counts['R'][1]),
        'out_tokens_med': med(out_tokens(v) for v in ok.values()),
        'reasoning_med': med(reasoning(v) for v in ok.values()),
        'wall_med': med(v.get('wall_s') for v in ok.values()),
        'agree_ref': agree, 'passes_floor': share['F'] < 0.10,
    })
rows.sort(key=lambda r: (math.isnan(r['floor']), r['floor']))
if '--json' in sys.argv:
    json.dump(rows, open(os.path.join(d, 'sweep-table.json'), 'w'), indent=1)
pct = lambda x: '—' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.0%}'
num = lambda x: '—' if x is None else (f'{x:.0f}' if x >= 10 else f'{x:.1f}')
print('| cell | floor (F one) | passes <10% | N one | R one | gap N−R (z) | errors | median out tokens | median reasoning | median wall s | agrees with Haiku default |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for r in rows:
    print(f"| {r['cell']} | {pct(r['floor'])} | {'yes' if r['passes_floor'] else 'no'} | {pct(r['N'])} | {pct(r['R'])} | "
          f"{r['gap']:+.0%} ({r['z']:.2f}) | {r['errors']}/{r['n']} | {num(r['out_tokens_med'])} | {num(r['reasoning_med'])} | "
          f"{num(r['wall_med'])} | {pct(r['agree_ref']) if r['cell'] != REF else 'ref'} |")
