"""Part 205, EXPLORATORY (written 2026-11-15, after scoring; no prediction reads it).

Question: how much of each stage's ranking is answer-letter machinery rather than value content?

Every ranking prompt is a two-letter forced choice, and the consistent option is A on half the prompts and B on the
other half. A part whose estimate e (signed toward the consistent option) flips with the consistent letter is pushing
a letter; a part whose e keeps its sign is pushing the consistent option whatever its letter. With s = +1 where the
consistent option is A and -1 where it is B, per part over one kind's 24 prompts:
    letter component      L = |mean(e * s)|
    consistency component C = |mean(e)|
Estimates are first normalised per prompt by their absolute sum, as the ranking does. Reported per stage and kind:
  - the top-80 set's letter share, sum L / (sum L + sum C), and the fraction of its parts with L > C;
  - the same share over the whole pool, weighted by each part's ranking score (mean normalised |e|);
  - the median letter share of 200 seeded random 80-part draws from the pool (an unweighted reference);
  - for the parts in both full top-80 sets, L and C averaged over the two kinds (score-weighted and unweighted);
and from the ranking records (behaviour): mean P(A), mean P(consistent), and P(consistent) split by the consistent
letter. Under pure noise with balanced letters, E[L] = E[C], so a share near 0.5 is the no-structure reference.

Reads results/part205/<stage>/ranking.npz, sets.json and records.jsonl and part205/prompts205.json; writes
results/part205/letter.json and letter.md. No model is run.
"""
import json
import sys
from pathlib import Path

import numpy as np

STAGES = ['pretrain-3.0T', 'midtrain-26B', 'midtrain-end', 'sft', 'dpo', 'instruct', 'qwen-base', 'qwen-chat']
SEED = 205015


def consistent_sign(prompts):
    out = {}
    for _value, d in prompts['items'].items():
        for kind in ('say', 'do'):
            for it in d[kind]:
                out[it['id']] = 1.0 if it['consistent'] == 'A' else -1.0
    return out


def components(est, signs):
    est = est / np.abs(est).sum(axis=1, keepdims=True)
    s = signs[:, None]
    return np.abs((est * s).mean(0)), np.abs(est.mean(0)), np.abs(est).mean(0)


def share(L, C, cols=None, w=None):
    if cols is not None:
        L, C = L[cols], C[cols]
        w = None if w is None else w[cols]
    if w is not None:
        L, C = L * w, C * w
    return float(L.sum() / (L.sum() + C.sum()))


def behaviour(records, kind):
    rr = [r for r in records if r['arm'] == 'ranking' and r['kind'] == kind]
    pa = float(np.mean([r['p_a'] for r in rr]))
    pc = float(np.mean([r['p_a'] if r['consistent'] == 'A' else r['p_b'] for r in rr]))
    ca = float(np.mean([r['p_a'] for r in rr if r['consistent'] == 'A']))
    cb = float(np.mean([r['p_b'] for r in rr if r['consistent'] == 'B']))
    return {'n': len(rr), 'p_a': pa, 'p_consistent': pc, 'p_consistent_when_A': ca, 'p_consistent_when_B': cb}


def main(root):
    root = Path(root)
    prompts = json.loads((root / 'part205/prompts205.json').read_text())
    sign_of = consistent_sign(prompts)
    rng = np.random.default_rng(SEED)
    result = {'part': 205, 'exploratory': True, 'seed': SEED, 'stages': []}
    for st in STAGES:
        d = root / 'results/part205' / st
        z = np.load(d / 'ranking.npz', allow_pickle=False)
        est, keys, ids = z['est'], [str(k) for k in z['keys']], [str(i) for i in z['prompt_ids']]
        sets = json.loads((d / 'sets.json').read_text())
        records = [json.loads(line) for line in (d / 'records.jsonl').read_text().splitlines() if line.strip()]
        idx = {k: j for j, k in enumerate(keys)}
        row = {'stage': st}
        comp = {}
        for kind in ('say', 'do'):
            mask = np.array([i.startswith(kind + '-') for i in ids])
            signs = np.array([sign_of[i] for i in np.array(ids)[mask]])
            L, C, score = components(est[mask], signs)
            comp[kind] = (L, C, score)
            cols = np.array([idx[k] for k in sets[kind]])
            draws = [share(L, C, rng.choice(len(keys), 80, replace=False)) for _ in range(200)]
            row[kind] = {
                'n_prompts': int(mask.sum()),
                'n_A': int((signs > 0).sum()),
                'top80_letter_share': share(L, C, cols),
                'top80_frac_letter_dominant': float((L[cols] > C[cols]).mean()),
                'pool_letter_share_score_weighted': share(L, C, w=score),
                'random80_letter_share_median': float(np.median(draws)),
                'behaviour': behaviour(records, kind.upper()),
            }
        # The parts in both full top-80 sets (SAY and DO): L and C averaged over the two kinds, weighted by the mean
        # of the two kinds' scores. Added 2026-11-15 after the cold check (finding 4) asked what O's shared parts are.
        shared = sorted(set(sets['say']) & set(sets['do']))
        scols = np.array([idx[k] for k in shared])
        Lb = (comp['say'][0] + comp['do'][0]) / 2
        Cb = (comp['say'][1] + comp['do'][1]) / 2
        wb = (comp['say'][2] + comp['do'][2]) / 2
        row['shared'] = {'n_parts': len(shared), 'letter_share_score_weighted': share(Lb, Cb, scols, wb),
                         'letter_share_unweighted': share(Lb, Cb, scols)}
        result['stages'].append(row)
    (root / 'results/part205/letter.json').write_text(json.dumps(result, indent=1) + '\n')
    lines = ['# Part 205, exploratory: answer-letter against consistency components (after scoring)', '',
             'Made by `part205/letter205.py` from the stored ranking estimates and records; no model was run. '
             'Letter share near 0.5 is the no-structure reference (balanced letters).', '',
             '| stage | kind | top-80 letter share | top-80 parts letter-dominant | pool letter share (score-weighted) '
             '| random-80 median | P(A) | P(consistent) | P(cons) when A | P(cons) when B |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    shared_lines = ['', '## Parts in both top-80 sets (SAY and DO)', '',
                    '| stage | shared parts (of 80) | letter share (score-weighted) | letter share (unweighted) |',
                    '| --- | --- | --- | --- |']
    for row in result['stages']:
        sh = row['shared']
        shared_lines.append(f"| {row['stage']} | {sh['n_parts']} | {sh['letter_share_score_weighted']:.3f} | "
                            f"{sh['letter_share_unweighted']:.3f} |")
    for row in result['stages']:
        for kind in ('say', 'do'):
            r = row[kind]; b = r['behaviour']
            lines.append(f"| {row['stage']} | {kind.upper()} | {r['top80_letter_share']:.3f} | "
                         f"{r['top80_frac_letter_dominant']:.3f} | {r['pool_letter_share_score_weighted']:.3f} | "
                         f"{r['random80_letter_share_median']:.3f} | {b['p_a']:.2f} | {b['p_consistent']:.2f} | "
                         f"{b['p_consistent_when_A']:.2f} | {b['p_consistent_when_B']:.2f} |")
    lines += shared_lines
    (root / 'results/part205/letter.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
