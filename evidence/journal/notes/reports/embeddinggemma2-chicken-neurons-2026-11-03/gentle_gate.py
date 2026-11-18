"""The gentle inside version (EDIT-DESIGN.md Amendment 4).

Same gate and residual direction as inside_gate.py (d and R from inside_params.npz). When the gate opens, the push is a
fixed size s * R along -d for every gated clip. s is chosen on the fit split only: among the grid sizes whose median
cosine (edited against unedited) over the gated fit imitations is at least 0.8, the one with the highest fit AUC; ties
to the smaller s. Control: the random direction, same rule. Writes gentle_results.json.
Run: HF_HUB_OFFLINE=1 PYTHONPATH=/home/elisepc/eg2-extra /home/elisepc/neuron-probe/.venv/bin/python -B gentle_gate.py
"""
import argparse, csv, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import edit_arms as ea
from inside_gate import InsideModel, ESC, PIX, VIS, HUMAN
from other_items import write_other_items_report
from record import decode

GRID = (0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
MIN_COS = 0.8


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-other', metavar='PATH',
                        help='write per-clip other-animal imitation predictions as JSON')
    args = parser.parse_args()
    ver = json.loads((HERE / 'edit_verdicts.json').read_text()); par = ver['arms']['written_gated_block']['parameters']
    s17, s19 = int(ver['signs']['L17_N1574']), int(ver['signs']['L19_N1129'])
    ref, vis, pix = np.load(HERE / 'acts.npz'), np.load(HERE / 'vis_acts.npz'), np.load(HERE / 'pixabay_acts.npz')
    man = list(csv.DictReader(open(HERE / 'vis_manifest.csv'))); fit, _ = ea.fit_imitation_split(man)
    cats = list(ref['categories']); P = np.asarray(ref['prompt_emb'], np.float32)
    hen, roo = cats.index('hen'), cats.index('rooster'); other = np.asarray([i for i in range(50) if i not in (hen, roo)])
    prm = np.load(HERE / 'inside_params.npz'); R = float(prm['R'])
    u17, u19 = 17 * 2048 + 1574, 19 * 2048 + 1129
    score = lambda E: ea.prompt_scores(E, P, hen, roo, other)
    esc = ref['kind'] == 'esc'; chicken = np.isin(ref['label'], ['hen', 'rooster'])
    pr = pix['label'] == 'real'; efc = esc & (ref['fold'] <= 3) & chicken
    vimit = (vis['group'] == 'chicken') & (vis['kind'] == 'imitation'); vreal = (vis['group'] == 'chicken') & (vis['kind'] == 'real')
    test_esc = esc & np.isin(ref['fold'], [4, 5])
    sets = {
        'fit_real': ([(PIX / f'{x}.mp3', 30) for x in pix['id'][pr]] + [(ESC / k, None) for k in ref['key'][efc]],
                     np.concatenate([pix['emb'][pr], ref['emb'][efc]]), np.concatenate([pix['act_mean'][pr], ref['act_mean'][efc]])),
        'fit_imit': ([(VIS / r['local'], 30) for r, f in zip(man, fit) if f], vis['emb'][fit], vis['act_mean'][fit]),
        'test_imit': ([(VIS / r['local'], 30) for r, f, m in zip(man, fit, vimit) if m and not f],
                      vis['emb'][vimit & ~fit], vis['act_mean'][vimit & ~fit]),
        'test_real': ([(VIS / r['local'], 30) for r, m in zip(man, vreal) if m], vis['emb'][vreal], vis['act_mean'][vreal]),
        'esc_test': ([(ESC / k, None) for k in ref['key'][test_esc]], ref['emb'][test_esc], ref['act_mean'][test_esc]),
    }
    if args.out_other:
        other_imit = (vis['group'] == 'other') & (vis['kind'] == 'imitation')
        sets['other_imit'] = (
            [(VIS / r['local'], 30) for r, keep in zip(man, other_imit) if keep],
            vis['emb'][other_imit],
            vis['act_mean'][other_imit],
        )
    G, AUD = {}, {}
    for name, (paths, E, A) in sets.items():
        a = A.astype(np.float32)
        g = np.maximum(s19 * a[:, u19] - par['t19'], 0) * np.maximum(par['t17'] - s17 * a[:, u17], 0)
        G[name] = g
        AUD[name] = {}
        for i in np.where(g > 0)[0]:
            p, cut = paths[i]; x = decode(p); AUD[name][int(i)] = x[:16000 * cut] if cut else x
    M = InsideModel(par['t19'], par['t17'], s17, s19)
    print('open', {k: len(v) for k, v in AUD.items()}, f'{time.time() - t0:.0f}s', flush=True)

    def edited(name, vec, s):
        E = sets[name][1].copy()
        M.d = torch.from_numpy(np.asarray(vec, np.float32))
        for i, x in AUD[name].items():
            M.scale = s * R / float(G[name][i])  # scale * g = s * R: the same push size for every gated clip
            E[i] = M.embed(x)
        M.scale = 0.0
        return E

    fits, grid_rows = {}, []
    for label, vec in (('gentle', prm['d']), ('random_gentle', prm['r'])):
        best = None
        for s in GRID:
            Er, Ei = edited('fit_real', vec, s), edited('fit_imit', vec, s)
            gi = G['fit_imit'] > 0
            cos = float(np.median(np.sum(Ei[gi] * sets['fit_imit'][1][gi], axis=1)))
            auc = ea.auc_values(score(Er), score(Ei))
            grid_rows.append({'direction': label, 's': s, 'fit_auc': auc, 'fit_median_cos': cos})
            print(json.dumps(grid_rows[-1]), flush=True)
            if cos >= MIN_COS and (best is None or auc > best['fit_auc']):
                best = {'s': s, 'fit_auc': auc, 'fit_median_cos': cos}
        if best is None:
            raise SystemExit(f'{label}: no grid size keeps the sound on the fit imitations')
        fits[label] = best
    out = {'R': R, 'grid': grid_rows, 'fits': fits, 'arms': {}}
    nb = ea.make_bootstrap_indices(len(sets['test_real'][0]), len(sets['test_imit'][0]))
    out['baseline_test_auc'] = ea.auc_values(score(sets['test_real'][1]), score(sets['test_imit'][1]))
    gi = G['test_imit'] > 0
    labels = ref['label'][test_esc]; nonch = ~np.isin(labels, ['hen', 'rooster'])
    for label, vec in (('gentle', prm['d']), ('random_gentle', prm['r'])):
        s = fits[label]['s']
        er, ei, ee = edited('test_real', vec, s), edited('test_imit', vec, s), edited('esc_test', vec, s)
        top = ea.prompt_predictions(ei[gi], P, cats)
        dmg = ea.class_damage(sets['esc_test'][1], ee, labels, P, cats)
        out['arms'][label] = {
            's': s, 'test_auc': ea.auc_values(score(er), score(ei)),
            'test_auc_bootstrap_95': ea.bootstrap_ci(score(er), score(ei), nb),
            'gated_top_classes': dict(Counter(map(str, top)).most_common()),
            'gated_human_share': float(np.mean([t in HUMAN for t in top])),
            'gated_median_cos_to_original': float(np.median(np.sum(ei[gi] * sets['test_imit'][1][gi], axis=1))),
            'nonchicken_bit_identical': int(np.sum(np.all(ee[nonch] == sets['esc_test'][1][nonch], axis=1))),
            'classes_changed_more_than_1_clip': dmg['classes_changed_more_than_1_clip'],
        }
        print(label, json.dumps(out['arms'][label]), flush=True)
    a, r = out['arms']['gentle'], out['arms']['random_gentle']
    out['verdicts'] = {'G1': a['test_auc'] >= 0.75, 'G2': a['gated_median_cos_to_original'] >= 0.8,
                       'G3': a['gated_human_share'] >= 0.5, 'G4': r['test_auc'] < a['test_auc'],
                       'G5': len(a['classes_changed_more_than_1_clip']) == 0}
    if args.out_other:
        _, other_embeddings, _ = sets['other_imit']
        other_gate = G['other_imit'] > 0
        edited_other = edited('other_imit', prm['d'], fits['gentle']['s'])
        before_top = ea.prompt_predictions(other_embeddings, P, cats)
        after_top = ea.prompt_predictions(edited_other, P, cats)
        before_chicken_score = score(other_embeddings)
        after_chicken_score = score(edited_other)
        other_rows = [row for row in man if row['group'] == 'other' and row['kind'] == 'imitation']
        item_ids = [f"vis:{row['id']}" for row in other_rows]
        animals = [row.get('class', row['group']) for row in other_rows]
        write_other_items_report(
            args.out_other,
            item_ids,
            animals,
            before_top,
            {'gentle_hook': after_top},
            chicken_prompt_scores={
                'gentle_hook': (before_chicken_score, after_chicken_score),
            },
            gate_open={'gentle_hook': other_gate},
        )

    out['seconds'] = round(time.time() - t0, 1)
    (HERE / 'gentle_results.json').write_text(json.dumps(out, indent=1))
    print('VERDICTS', out['verdicts'], out['seconds'], flush=True)


if __name__ == '__main__':
    main()
