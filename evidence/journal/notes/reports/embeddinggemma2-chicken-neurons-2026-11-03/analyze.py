"""Score the registered predictions in PREDICTIONS.md from acts.npz. Writes results.json and prints a summary."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
d = np.load(HERE / 'acts.npz')
kind, key, label, fold = d['kind'], d['key'], d['label'], d['fold']
L, W = 24, 2048


def aucs(X, y):
    """AUC of each column of X for y (bool); rank-based, ties averaged."""
    r = rankdata(X, axis=0)
    npos, nneg = y.sum(), (~y).sum()
    return (r[y].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)


def run(act_name):
    A = d[act_name].astype(np.float32)
    esc = kind == 'esc'
    chick = np.isin(label, ['hen', 'rooster'])
    disc, test = esc & (fold <= 3), esc & (fold >= 4)
    a_disc = aucs(A[disc], chick[disc])
    eff = np.maximum(a_disc, 1 - a_disc)
    sign = np.where(a_disc >= 0.5, 1.0, -1.0)
    order = np.argsort(-eff)
    top = int(order[0])
    top10 = [int(u) for u in order[:10]]
    a_test = aucs(A[test], chick[test])
    signed_test = lambda u: float(a_test[u] if sign[u] > 0 else 1 - a_test[u])
    rng = np.random.default_rng(20261103)
    null = []
    yd = chick[disc].copy()
    for _ in range(20):
        ys = rng.permutation(yd)
        an = aucs(A[disc], ys)
        en = np.maximum(an, 1 - an)
        u = int(en.argmax())
        null.append(float(a_test[u] if an[u] >= 0.5 else 1 - a_test[u]))
    # model-level classifier (control b)
    cats = list(d['categories'])
    P = d['prompt_emb']
    S = d['emb'] @ P.T
    ci = [cats.index('hen'), cats.index('rooster')]
    oi = [i for i in range(len(cats)) if i not in ci]
    emb_score = S[:, ci].max(1) - S[:, oi].mean(1)
    emb_auc = float(aucs(emb_score[test, None], chick[test])[0])
    # text arm
    txt = kind == 'text'
    tchick = label == 'chicken'
    a_text = aucs(A[txt], tchick[txt])
    text_rows = [{'unit': u, 'layer': u // W, 'index': u % W, 'sign': int(sign[u]),
                  'disc_auc': float(eff[u]), 'test_auc': signed_test(u),
                  'text_auc_same_sign': float(a_text[u] if sign[u] > 0 else 1 - a_text[u])} for u in top10]
    # our clips on the top neuron
    ours = kind == 'ours'
    v = sign[top] * A[:, top]
    hen_med = float(np.median(v[esc & (label == 'hen')]))
    ours_rank = sorted(((str(key[i]), float(v[i])) for i in np.where(ours)[0]), key=lambda t: -t[1])
    egg_b = dict(ours_rank).get('human_imitation_egg_song_b')
    return {
        'activation': act_name,
        'n': {'disc_chicken': int((disc & chick).sum()), 'disc_other': int((disc & ~chick).sum()),
              'test_chicken': int((test & chick).sum()), 'test_other': int((test & ~chick).sum())},
        'top': {'unit': top, 'layer': top // W, 'index': top % W, 'sign': int(sign[top]),
                'disc_auc': float(eff[top]), 'test_auc': signed_test(top)},
        'top10': text_rows,
        'selection_null_test_auc': null,
        'embedding_classifier_test_auc': emb_auc,
        'ours_on_top': {'hen_median': hen_med, 'egg_song_b': egg_b, 'ranked': ours_rank},
        'layer_of_top10': [u // W for u in top10],
    }


out = {'mean': run('act_mean'), 'max_exploratory': run('act_max')}
r = out['mean']
t = r['top']
pred = {
    'N1': ('held' if t['test_auc'] >= 0.85 else 'failed', round(t['test_auc'], 3)),
    'N2': ('held' if max(r['selection_null_test_auc']) < 0.65 else 'failed', round(max(r['selection_null_test_auc']), 3)),
    'N3': ('held' if 12 <= t['layer'] <= 23 else 'failed', t['layer']),
    'N4': ('held' if any(x['text_auc_same_sign'] >= 0.8 for x in r['top10']) else 'failed',
           round(max(x['text_auc_same_sign'] for x in r['top10']), 3)),
    'N5': ('held' if r['embedding_classifier_test_auc'] >= 0.9 else 'failed', round(r['embedding_classifier_test_auc'], 3)),
    'N6': ('held' if abs(t['test_auc'] - r['embedding_classifier_test_auc']) <= 0.05 else 'failed',
           round(t['test_auc'] - r['embedding_classifier_test_auc'], 3)),
    'N7': ('held' if (r['ours_on_top']['egg_song_b'] or -1e9) > r['ours_on_top']['hen_median'] else 'failed',
           (r['ours_on_top']['egg_song_b'], r['ours_on_top']['hen_median'])),
}
out['predictions'] = pred
(HERE / 'results.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in r.items() if k != 'ours_on_top'}, indent=1)[:3000])
print('ours on top neuron (sign-adjusted), hen median', round(r['ours_on_top']['hen_median'], 4))
for k, v in r['ours_on_top']['ranked']:
    print(f'  {k:32s} {v:+.4f}')
print('PREDICTIONS', json.dumps(pred))
m = out['max_exploratory']['top']
print('exploratory (max over positions): top', m)
