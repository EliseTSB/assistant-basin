#!/usr/bin/env python3
# name_map.py — merge the descriptions and score Part 11.10-11.13 (Claude, 2026-10-16; rules pushed as assistant-basin
# ce07504 before any run). No model is loaded.
#
#   merge <dir> <11.3 dir>   verdicts-gpt-6-luna-high-full.json in <dir>: this run's Luna verdicts (M and MC, uids nNNN)
#                            plus 11.3's unchanged (I and C, uids i-uNNN), so jev_sim.mjs reads one file.
#   score <dir>              map.md (one row per unit: arm, layer, neuron, pole, verdict, description, rules, AUROC) and
#                            summary.json (medians per arm, the four predictions' numbers, mechanically).
# AUROC as jev_sim.mjs --analyze: Jev's yes-probability on the 10 strongest held-out contexts against the 10 random ones,
# ties counted half; a unit with any failed Jev call is left out and counted.
import json, os, sys
import numpy as np

NAME = ['name', 'proper noun', 'capitaliz', 'capitalis', 'entit', 'title']
SELF = ['smollm', 'assistant', 'chatbot', 'language model', 'ai model', ' ai ', 'self-introduc', 'introducing themsel', 'my name']
def has(desc, words):
    d = f' {desc.lower()} '
    return any(w in d for w in words)

def auroc(pos, neg):
    if not pos or not neg: return float('nan')
    w = sum(1.0 if x > y else 0.5 if x == y else 0.0 for x in pos for y in neg)
    return w / (len(pos) * len(neg))

cmd, d = sys.argv[1], sys.argv[2]
if cmd == 'merge':
    old = json.load(open(os.path.join(sys.argv[3], 'verdicts-gpt-6-luna-high.json')))
    new = json.load(open(os.path.join(d, 'verdicts-gpt-6-luna-high.json')))
    full = {**new, **{f'i-{k}': v for k, v in old.items()}}
    json.dump(full, open(os.path.join(d, 'verdicts-gpt-6-luna-high-full.json'), 'w'), indent=0)
    print(f'merged {len(new)} new + {len(old)} from 11.3 = {len(full)}'); sys.exit(0)

held = json.load(open(os.path.join(d, 'heldout.json')))
luna = json.load(open(os.path.join(d, 'verdicts-gpt-6-luna-high-full.json')))
jev = json.load(open(os.path.join(d, 'jev-sim.json')))
pole = json.load(open(os.path.join(d, 'pole.json')))
rows, dropped = [], []
for uid, u in held.items():
    r = jev.get(uid) or []
    ps = [x['p'] if x and x.get('ok') else None for x in r]
    if len(ps) < len(u['items']) or any(p is None for p in ps): dropped.append(uid); continue
    by = lambda k: [p for p, it in zip(ps, u['items']) if it['kind'] == k]
    v = luna.get(uid, {})
    desc = v.get('description', '')
    rows.append({'uid': uid, 'arm': u['arm'], 'layer': u['layer'], 'neuron': u['neuron'], 'sign': u['sign'],
                 'verdict': v.get('verdict'), 'description': desc, 'name_rule': has(desc, NAME), 'self_rule': has(desc, SELF),
                 'auroc_top': auroc(by('top'), by('random')), 'auroc_mid': auroc(by('mid'), by('random')),
                 'attribution': pole.get(f"L{u['layer']}.{u['neuron']}", {}).get('attribution') if u['arm'] == 'M' else None})
arms = {a: [r for r in rows if r['arm'] == a] for a in ('M', 'MC', 'I', 'C')}
med = lambda xs: float(np.nanmedian(xs)) if xs else float('nan')
summ = {a: {'n': len(rs), 'one': sum(r['verdict'] == 'one' for r in rs), 'name_rule': sum(r['name_rule'] for r in rs),
            'self_rule': sum(r['self_rule'] for r in rs), 'median_auroc_top': med([r['auroc_top'] for r in rs]),
            'median_auroc_mid': med([r['auroc_mid'] for r in rs])} for a, rs in arms.items()}
# Description-rule counts use every described unit, not only those with complete Jev calls.
for a in ('M', 'MC'):
    ds = [(uid, v) for uid, v in luna.items() if held.get(uid, {}).get('arm') == a]
    summ[a]['described'] = len(ds)
    summ[a]['name_rule_all'] = sum(has(v.get('description', ''), NAME) for _, v in ds)
    summ[a]['self_rule_all'] = sum(has(v.get('description', ''), SELF) for _, v in ds)
M, MC, I, C = (summ[a] for a in ('M', 'MC', 'I', 'C'))
pred = {
    '11.10': {'name_rule': M['name_rule_all'], 'self_rule': M['self_rule_all'], 'of': M['described'],
              'held': M['name_rule_all'] >= 14 and M['self_rule_all'] <= 3},
    '11.11': {'M_share': M['name_rule_all'] / max(1, M['described']), 'MC_share': MC['name_rule_all'] / max(1, MC['described']),
              'held': M['name_rule_all'] / max(1, M['described']) - MC['name_rule_all'] / max(1, MC['described']) >= 0.20},
    '11.12': {'M': M['median_auroc_top'], 'I': I['median_auroc_top'],
              'held': M['median_auroc_top'] >= 0.75 and I['median_auroc_top'] >= 0.75},
    '11.13': {'M_minus_MC': M['median_auroc_top'] - MC['median_auroc_top'], 'I_minus_C': I['median_auroc_top'] - C['median_auroc_top'],
              'held': abs(M['median_auroc_top'] - MC['median_auroc_top']) < 0.10 and abs(I['median_auroc_top'] - C['median_auroc_top']) < 0.10},
}
json.dump({'arms': summ, 'predictions': pred, 'dropped_for_failed_jev_calls': dropped}, open(os.path.join(d, 'summary.json'), 'w'), indent=1)
f = lambda x: '' if x is None or x != x else f'{x:.2f}'
lines = ['| arm | layer.neuron | pole | Luna | description | name rule | AUROC strongest vs random | AUROC mid vs random |',
         '|---|---|---|---|---|---|---|---|']
for a in ('M', 'I', 'MC', 'C'):
    for r in sorted(arms[a], key=lambda r: (r['layer'], r['neuron'])):
        lines.append(f"| {a} | L{r['layer']}.{r['neuron']} | {'+' if r['sign'] > 0 else '-'} | {r['verdict']} | "
                     f"{r['description'].replace('|', '/')} | {'yes' if r['name_rule'] else ''} | {f(r['auroc_top'])} | {f(r['auroc_mid'])} |")
open(os.path.join(d, 'map.md'), 'w').write('\n'.join(lines) + '\n')
for a, s in summ.items(): print(a, s)
for k, v in pred.items(): print(k, v)
print(f'dropped {len(dropped)} units for failed Jev calls')
