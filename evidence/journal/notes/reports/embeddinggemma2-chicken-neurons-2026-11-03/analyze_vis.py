"""Score V1 to V6 (PREDICTIONS.md, VocalImitationSet 2 x 2) from vis_acts.npz, with neurons, signs and the prompt
classifier fixed on ESC-50 (acts.npz) exactly as analyze_pixabay.py. Writes vis_results.json and prints a summary."""
import json
from pathlib import Path
import numpy as np
from analyze_pixabay import W, NINE, TOP, auc, boot

HERE = Path(__file__).resolve().parent
N1574 = (17, 1574)


def main():
    ref = np.load(HERE / 'acts.npz')
    d = np.load(HERE / 'vis_acts.npz')
    A, g, k = d['act_mean'].astype(np.float32), d['group'], d['kind']
    ci_, co = (g == 'chicken') & (k == 'imitation'), (g == 'other') & (k == 'imitation')
    cr, orl = (g == 'chicken') & (k == 'real'), (g == 'other') & (k == 'real')
    RA = ref['act_mean'].astype(np.float32)
    esc = ref['kind'] == 'esc'
    chick = np.isin(ref['label'], ['hen', 'rooster'])
    disc = esc & (ref['fold'] <= 3)

    def sign(u):
        return 1.0 if auc(RA[:, u], disc & chick, disc & ~chick) >= 0.5 else -1.0

    contrasts = {'chicken_imit_vs_other_imit': (ci_, co), 'other_real_vs_other_imit': (orl, co),
                 'chicken_real_vs_other_real': (cr, orl), 'chicken_real_vs_chicken_imit': (cr, ci_)}
    rng = np.random.default_rng(20261104)
    res = {'n': {'chicken_imitation': int(ci_.sum()), 'other_imitation': int(co.sum()), 'chicken_real': int(cr.sum()),
                 'other_real': int(orl.sum())}, 'neurons': []}
    for L, Nn in [N1574, TOP] + [x for x in NINE if x != N1574]:
        u = L * W + Nn
        x = sign(u) * A[:, u]
        e = {'layer': L, 'index': Nn, 'sign': int(sign(u)),
             'medians': {n: float(np.median(x[m])) for n, m in [('chicken_imitation', ci_), ('other_imitation', co),
                                                                ('chicken_real', cr), ('other_real', orl)]}}
        for name, (p, q) in contrasts.items():
            e[name] = {'auc': auc(x, p, q), 'ci95': boot(x, p, q, rng)}
        res['neurons'].append(e)
    cats = list(ref['categories'])
    P = ref['prompt_emb']
    ci = [cats.index('hen'), cats.index('rooster')]
    oi = [i for i in range(len(cats)) if i not in ci]
    S = d['emb'] @ P.T
    score = S[:, ci].max(1) - S[:, oi].mean(1)
    res['prompt_classifier'] = {name: {'auc': auc(score, p, q), 'ci95': boot(score, p, q, rng)}
                                for name, (p, q) in contrasts.items()}
    n = res['neurons'][0]
    pc = res['prompt_classifier']
    res['predictions'] = {
        'V1': {'value': n['chicken_imit_vs_other_imit']['auc'], 'held': n['chicken_imit_vs_other_imit']['auc'] >= 0.65},
        'V2': {'value': n['other_real_vs_other_imit']['auc'], 'held': n['other_real_vs_other_imit']['auc'] < 0.65},
        'V3': {'value': n['chicken_real_vs_other_real']['auc'], 'held': n['chicken_real_vs_other_real']['auc'] >= 0.85},
        'V4': {'value': n['chicken_real_vs_chicken_imit']['auc'],
               'held': n['chicken_real_vs_chicken_imit']['auc'] >= 0.85},
        'V5': {'value': pc['chicken_imit_vs_other_imit']['auc'], 'held': pc['chicken_imit_vs_other_imit']['auc'] >= 0.80},
        'V6': {'classifier': pc['chicken_real_vs_chicken_imit']['auc'], 'neuron': n['chicken_real_vs_chicken_imit']['auc'],
               'held': pc['chicken_real_vs_chicken_imit']['auc'] < n['chicken_real_vs_chicken_imit']['auc']},
    }
    json.dump(res, open(HERE / 'vis_results.json', 'w'), indent=1)
    print(json.dumps(res['n']))
    for e in res['neurons']:
        print(f"L{e['layer']} N{e['index']}: " + '  '.join(
            f"{c.split('_vs_')[0]}>{c.split('_vs_')[1]} {e[c]['auc']:.3f}" for c in contrasts))
    print('classifier: ' + '  '.join(f"{c} {pc[c]['auc']:.3f} {[round(v, 3) for v in pc[c]['ci95']]}" for c in contrasts))
    print('PREDICTIONS', json.dumps(res['predictions']))


if __name__ == '__main__':
    main()
