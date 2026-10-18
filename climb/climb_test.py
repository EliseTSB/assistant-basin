#!/usr/bin/env python3
# climb_test.py — scores Part 1e's frozen candidates on TEST and on the fresh slice (Claude, 2026-10-17, after the
# 07:45 freeze; the carried candidates were picked on DEV and pushed in PREDICTIONS.md before this ran).
#
# Reuses climb_eval.py's own functions (explainer, Jev simulator, Spearman/AUROC, judge p_one) so TEST is scored by the
# exact code DEV was. Data comes straight from Part 1's run directory, with the arm labels kept here for 1e.4 and 1e.5:
#   test   — the split's TEST ids (N and R units with Part 1d's held-out contexts; F units as the judge floor);
#   fresh  — every N and R unit (DEV and TEST, 299) with the contexts of a second heldout.py run (tokens 1.3M-1.6M).
# The baseline's Luna descriptions and Part 1d's Jev answers are seeded into the cache exactly as climb_setup.py did
# for DEV, so the baseline on TEST costs nothing and reproduces Part 1d; everything else is new calls.
# Usage: CLIMB_WORK=<work dir> CLIMB_NAME=claude-test python3 climb_test.py <part1 dir> <split json> <baseline dir>
#        <pipeline dir> <judge dir> [--fresh <fresh heldout.json>] [--dry] [--out <json>]
import argparse, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('ce', os.path.join(HERE, 'climb_eval.py'))
ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)

ap = argparse.ArgumentParser()
for x in ('part1', 'split', 'baseline', 'pipeline', 'judge'): ap.add_argument(x)
ap.add_argument('--fresh'); ap.add_argument('--dry', action='store_true'); ap.add_argument('--out')
ap.add_argument('--rows-out', help='optional per-unit TEST rows for later paired comparisons')
a = ap.parse_args()

units = {u['id']: u for u in json.load(open(os.path.join(a.part1, 'units.json')))['units']}
key = json.load(open(os.path.join(a.part1, 'key.json')))
held = json.load(open(os.path.join(a.part1, 'heldout.json')))
luna = json.load(open(os.path.join(a.part1, 'verdicts-gpt-6-luna-high-full.json')))
sim = json.load(open(os.path.join(a.part1, 'jev-sim.json')))
split = json.load(open(a.split))
# Part 1 drew one neuron pole twice (u121 and u257, L11 n2409 sign -1). heldout.py kept it as u257 and Part 1d scored
# u257's description; the split dropped u257 and kept u121, so u121 takes u257's held-out contexts, Jev answers and
# description (identical training contexts), which is exactly the unit Part 1d scored.
for d in (held, sim, luna): d['u121'] = d['u257']
base = ce.load_candidate(a.baseline); pipe = ce.load_candidate(a.pipeline); judge = ce.load_candidate(a.judge)

def records(ids, heldout):
    sims, floors = [], []
    for u in sorted(ids):
        rec = {'id': u, 'contexts': units[u]['contexts']}
        if key[u]['arm'] == 'F': floors.append(rec)
        else:
            rec['heldout'] = [{'kind': it['kind'], 'act': it['act'], 'context': it['context']} for it in heldout[u]['items']]
            sims.append(rec)
    return {'sim_units': sims, 'floor_units': floors, 'search_ids': []}

# Seed the baseline exactly as climb_setup.py seeds DEV (same keys), for every unit Part 1 described.
for u in key:
    if u == 'u257': continue
    rec = {'contexts': units[u]['contexts']}; v = luna.get(u)
    if not (v and v.get('description')): continue
    k = ce.explainer_key(base, rec)
    if ce.cache_get('luna', k) is None:
        ce.cache_put('luna', k, {'raw': {'verdict': v['verdict'], 'description': v['description']}, 'usage': v.get('usage')})
    if key[u]['arm'] != 'F' and u in sim:
        desc = ce.clip_description(v['description'], base['explainer']['max_words'])
        for it, r in zip(held[u]['items'], sim[u]):
            if r and r.get('ok'):
                jk = ce.jev_key(base, desc, it['context'])
                if ce.cache_get('jev', jk) is None: ce.cache_put('jev', jk, {'probs': {'fires': r['p']}, 'in_tok': r.get('inTok')})

test = records(split['test'], held)
arm = {u: key[u]['arm'] for u in key}
out = {'n_test_sim': len(test['sim_units']), 'n_test_floor': len(test['floor_units'])}

def judge_only(c, data, dry):
    # P(one job) per unit from the explainer's own answer, as climb_eval's 'explainer' judge engine does.
    from concurrent.futures import ThreadPoolExecutor
    us = data['sim_units'] + data['floor_units']; pend = 0; p = {}
    with ThreadPoolExecutor(ce.LUNA_CONC) as ex:
        for u, (raw, new) in zip(us, ex.map(lambda u: ce.explain(c, u, dry), us)):
            if raw is None: pend += 1; continue
            p[u['id']] = ce.p_one_from(raw)
    return p, pend

def arm_medians(rows, field):
    return {g: ce.median([r[field] for u, r in rows.items() if not r.get('floor') and arm[u] == g]) for g in 'NR'}

try:
    if a.dry:
        out['test_base'] = ce.evaluate(base, test, 'dev', 'sim', True)['pending_calls']
        out['test_pipe'] = ce.evaluate(pipe, test, 'dev', 'sim', True)['pending_calls']
        out['test_judge_pending_luna'] = judge_only(judge, test, True)[1]
        if a.fresh:
            fh = json.load(open(a.fresh)); fh['u121'] = fh['u257']
            fresh = records([u for u in key if arm[u] != 'F' and u != 'u257'], fh)
            out['fresh_base'] = ce.evaluate(base, fresh, 'dev', 'sim', True)['pending_calls']
            out['fresh_pipe'] = ce.evaluate(pipe, fresh, 'dev', 'sim', True)['pending_calls']
        print(json.dumps(out, indent=1)); sys.exit(0)

    rb = ce.evaluate(base, test, 'dev', 'sim'); rp = ce.evaluate(pipe, test, 'dev', 'sim')
    common = sorted(set(rb['rows']) & set(rp['rows']))
    d, lo, hi = ce.paired_ci([rp['rows'][u]['spearman'] for u in common], [rb['rows'][u]['spearman'] for u in common])
    out['test'] = {'base': rb['scores'], 'pipe': rp['scores'], 'delta_mean_spearman': d, 'ci95': [lo, hi], 'n': len(common),
                   'pipe_auroc_top_median_by_arm': arm_medians(rp['rows'], 'auroc_top'),
                   'base_auroc_top_median_by_arm': arm_medians(rb['rows'], 'auroc_top')}
    if a.rows_out:
        json.dump({'units': {u: {'arm': arm[u], **row} for u, row in rp['rows'].items()}},
                  open(a.rows_out, 'w'), indent=1)
    pj, _ = judge_only(judge, test, False)
    F = [p for u, p in pj.items() if arm[u] == 'F']; N = [p for u, p in pj.items() if arm[u] == 'N']; R = [p for u, p in pj.items() if arm[u] == 'R']
    share = lambda xs: sum(x >= 0.5 for x in xs) / max(1, len(xs))
    out['test_judge'] = {'floor_called_one': sum(x >= 0.5 for x in F), 'n_floor': len(F),
                         'one_share_real': share(N), 'one_share_random': share(R), 'n_real': len(N), 'n_random': len(R)}
    if a.fresh:
        fh = json.load(open(a.fresh)); fh['u121'] = fh['u257']
        fresh = records([u for u in key if arm[u] != 'F' and u != 'u257'], fh)
        fb = ce.evaluate(base, fresh, 'dev', 'sim'); fp = ce.evaluate(pipe, fresh, 'dev', 'sim')
        common = sorted(set(fb['rows']) & set(fp['rows']))
        d, lo, hi = ce.paired_ci([fp['rows'][u]['spearman'] for u in common], [fb['rows'][u]['spearman'] for u in common])
        out['fresh'] = {'base': fb['scores'], 'pipe': fp['scores'], 'delta_mean_spearman': d, 'ci95': [lo, hi], 'n': len(common)}
    out['jev_budget'] = ce.budget()
    print(json.dumps(out, indent=1))
    if a.out: json.dump(out, open(a.out, 'w'), indent=1)
except ce.Stop as e:
    print(f'STOP ({e.code}): {e}'); sys.exit(e.code)
