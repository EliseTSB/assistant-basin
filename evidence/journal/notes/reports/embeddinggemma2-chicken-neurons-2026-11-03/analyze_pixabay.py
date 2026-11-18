"""Score X1 to X4 (PREDICTIONS.md, Pixabay arm) from pixabay_acts.npz, with the neurons, signs and prompt classifier
fixed on ESC-50 (acts.npz). Writes pixabay_results.json and prints a summary."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
W = 2048
NINE = [(15, 436), (15, 1281), (16, 134), (16, 349), (17, 1297), (17, 1574), (19, 1129), (20, 1260), (23, 1262)]
TOP = (15, 1126)


def auc(x, pos, neg):
    v = np.concatenate([x[pos], x[neg]])
    r = rankdata(v)
    n1, n0 = pos.sum(), neg.sum()
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(x, pos, neg, rng, n=4000):
    ip, ineg = np.where(pos)[0], np.where(neg)[0]
    out = []
    for _ in range(n):
        p = np.zeros(len(x), bool); q = np.zeros(len(x), bool)
        bp, bn = rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))
        out.append(auc(np.concatenate([x[bp], x[bn]]), np.r_[np.ones(len(bp), bool), np.zeros(len(bn), bool)],
                       np.r_[np.zeros(len(bp), bool), np.ones(len(bn), bool)]))
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def main():
    ref = np.load(HERE / 'acts.npz')
    d = np.load(HERE / 'pixabay_acts.npz')
    A, lab, ids = d['act_mean'].astype(np.float32), d['label'], d['id']
    real, imit, ai = lab == 'real', lab == 'imitation', lab == 'ai'
    # signs from ESC-50 discovery folds, exactly as analyze.py
    RA = ref['act_mean'].astype(np.float32)
    esc = ref['kind'] == 'esc'
    chick = np.isin(ref['label'], ['hen', 'rooster'])
    disc = esc & (ref['fold'] <= 3)

    def sign(u):
        x = RA[:, u]
        return 1.0 if auc(x, disc & chick, disc & ~chick) >= 0.5 else -1.0

    rng = np.random.default_rng(20261103)
    res = {'n': {'real': int(real.sum()), 'imitation': int(imit.sum()), 'ai': int(ai.sum())}, 'neurons': []}
    for L, N in [TOP] + NINE:
        u = L * W + N
        x = sign(u) * A[:, u]
        res['neurons'].append({'layer': L, 'index': N, 'sign': int(sign(u)),
                               'auc_real_vs_imitation': auc(x, real, imit), 'ci95': boot(x, real, imit, rng),
                               'auc_ai_vs_imitation': auc(x, ai, imit), 'auc_real_vs_ai': auc(x, real, ai),
                               'imitation_values': {str(i): float(v) for i, v in zip(ids[imit], x[imit])},
                               'real_median': float(np.median(x[real]))})
    cats = list(ref['categories'])
    P = ref['prompt_emb']
    ci = [cats.index('hen'), cats.index('rooster')]
    oi = [i for i in range(len(cats)) if i not in ci]
    S = d['emb'] @ P.T
    score = S[:, ci].max(1) - S[:, oi].mean(1)
    res['prompt_classifier'] = {'auc_real_vs_imitation': auc(score, real, imit), 'ci95': boot(score, real, imit, rng),
                                'auc_ai_vs_imitation': auc(score, ai, imit), 'auc_real_vs_ai': auc(score, real, ai)}
    top = res['neurons'][0]
    nine = [r['auc_real_vs_imitation'] for r in res['neurons'][1:]]
    res['predictions'] = {
        'X1': {'value': top['auc_real_vs_imitation'], 'ci95': top['ci95'], 'held': top['auc_real_vs_imitation'] >= 0.75},
        'X2': {'neuron': top['auc_real_vs_imitation'], 'classifier': res['prompt_classifier']['auc_real_vs_imitation'],
               'held': top['auc_real_vs_imitation'] > res['prompt_classifier']['auc_real_vs_imitation']},
        'X3': {'median_nine': float(np.median(nine)), 'held': float(np.median(nine)) >= 0.70},
        'X4': {'n_ai': int(ai.sum()), 'value': top['auc_ai_vs_imitation'],
               'held': (top['auc_ai_vs_imitation'] >= 0.60) if ai.sum() >= 10 else None},
    }
    json.dump(res, open(HERE / 'pixabay_results.json', 'w'), indent=1)
    print(json.dumps(res['n']))
    for r in res['neurons']:
        print(f"L{r['layer']} N{r['index']}: real>imit {r['auc_real_vs_imitation']:.3f} {r['ci95']}  "
              f"ai>imit {r['auc_ai_vs_imitation']:.3f}  real>ai {r['auc_real_vs_ai']:.3f}")
    pc = res['prompt_classifier']
    print(f"classifier: real>imit {pc['auc_real_vs_imitation']:.3f} {pc['ci95']}  ai>imit {pc['auc_ai_vs_imitation']:.3f}"
          f"  real>ai {pc['auc_real_vs_ai']:.3f}")
    print('PREDICTIONS', json.dumps(res['predictions']))


if __name__ == '__main__':
    main()
