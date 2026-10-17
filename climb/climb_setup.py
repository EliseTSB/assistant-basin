#!/usr/bin/env python3
# climb_setup.py — Part 1e: builds the overnight prompt climb's data from Part 1's run (Claude, 2026-10-16, for Elise's
# 21:06 ask to hill-climb the neuron judge and Jev prompts with Luna in Codex).
#
# Splits Part 1's units with a fixed seed into DEV (the climb sees these) and TEST (scored once, by me, after the
# prompts are frozen). Real neurons and random directions are split separately so each half holds half of each arm,
# and the climb's copy carries NO arm label: ids are anonymised (d000...) and shuffled. u257 is dropped (Part 1 drew
# the same neuron pole as u121). The F arm (random contexts) is split the same way and labelled as the judge floor.
# It also seeds the climb's caches with Part 1's own Luna descriptions and Jev answers, so the baseline candidate
# reproduces Part 1d exactly at no cost.
# Usage: python climb_setup.py <part1 dir> <climb root> <split out json>
import hashlib, json, os, random, sys
src, root, split_out = sys.argv[1:4]
units = {u['id']: u for u in json.load(open(os.path.join(src, 'units.json')))['units']}
key = json.load(open(os.path.join(src, 'key.json')))
held = json.load(open(os.path.join(src, 'heldout.json')))
luna = json.load(open(os.path.join(src, 'verdicts-gpt-6-luna-high-full.json')))
sim = json.load(open(os.path.join(src, 'jev-sim.json')))
rng = random.Random(20261016)
split = {'seed': 20261016, 'dev': [], 'test': [], 'dropped': ['u257']}
for arm in 'NRF':
    ids = sorted(u for u, k in key.items() if k['arm'] == arm and u != 'u257')
    rng.shuffle(ids); h = len(ids) // 2
    split['dev'] += ids[:h]; split['test'] += ids[h:]
dev = sorted(split['dev']); rng.shuffle(dev)
anon = {u: f'd{i:03d}' for i, u in enumerate(dev)}
split['anon'] = anon
sim_units, floor_units = [], []
for u in dev:
    rec = {'id': anon[u], 'contexts': units[u]['contexts']}
    if key[u]['arm'] == 'F': floor_units.append(rec)
    else:
        rec['heldout'] = [{'kind': it['kind'], 'act': it['act'], 'context': it['context']} for it in held[u]['items']]
        sim_units.append(rec)
sim_units.sort(key=lambda r: r['id']); floor_units.sort(key=lambda r: r['id'])
search = sorted(rng.sample([r['id'] for r in sim_units], 60))
json.dump({'sim_units': sim_units, 'floor_units': floor_units, 'search_ids': search}, open(os.path.join(root, 'data', 'dev.json'), 'w'))
split['search_ids'] = search
json.dump(split, open(split_out, 'w'), indent=1)

# Seed caches with Part 1's results, keyed exactly as climb_eval.py keys its calls.
import importlib.util
spec = importlib.util.spec_from_file_location('ce', os.path.join(root, 'harness', 'climb_eval.py'))
ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)
base = ce.load_candidate(os.path.join(root, 'harness', 'baseline'))
n_l = n_j = 0
for u in dev:
    rec = {'contexts': units[u]['contexts']}
    v = luna.get(u)
    if v and v.get('description'):
        ce.cache_put('luna', ce.explainer_key(base, rec), {'raw': {'verdict': v['verdict'], 'description': v['description']}, 'usage': v.get('usage')}); n_l += 1
    if key[u]['arm'] != 'F' and u in sim and v:
        desc = ce.clip_description(v['description'], base['explainer']['max_words'])
        for it, r in zip(held[u]['items'], sim[u]):
            if r and r.get('ok'):
                ce.cache_put('jev', ce.jev_key(base, desc, it['context']), {'probs': {'fires': r['p']}, 'in_tok': r.get('inTok')}); n_j += 1
print(f'dev {len(dev)} (sim {len(sim_units)}, floor {len(floor_units)}), test {len(split["test"])}; search {len(search)}; seeded luna {n_l}, jev {n_j}')
