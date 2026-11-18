"""Edit pilot, fit split only (EDIT-DESIGN.md "Before registration"): where do the two gate units sit on the fit data,
and how often would a gate that opens on chicken-like-but-not-real sounds open on ordinary ESC-50 clips?
Fit = ESC-50 folds 1 to 3, the VocalImitationSet chicken imitations of a seeded half of their imitators, and its
other-animal clips of the same imitators' classes are NOT used (all other VIS clips stay out of the pilot).
Reads acts.npz and vis_acts.npz; no model. Writes edit_pilot_gate.json."""
import csv, json
from pathlib import Path
import numpy as np
from analyze_pixabay import auc

HERE = Path(__file__).resolve().parent
W = 2048
U17, U19 = 17 * W + 1574, 19 * W + 1129


def main():
    ref = np.load(HERE / 'acts.npz')
    RA = ref['act_mean'].astype(np.float32)
    esc = ref['kind'] == 'esc'
    chick = np.isin(ref['label'], ['hen', 'rooster'])
    disc = esc & (ref['fold'] <= 3)
    sign = {u: (1.0 if auc(RA[:, u], disc & chick, disc & ~chick) >= 0.5 else -1.0) for u in (U17, U19)}
    rows = list(csv.DictReader(open(HERE / 'vis_manifest.csv')))
    imitators = sorted({r['participant_id'] for r in rows if r['group'] == 'chicken' and r['kind'] == 'imitation'})
    rng = np.random.default_rng(20261104)
    fit_people = set(np.array(imitators)[rng.permutation(len(imitators))[:len(imitators) // 2]].tolist())
    V = np.load(HERE / 'vis_acts.npz')
    VA = V['act_mean'].astype(np.float32)
    fit_imit = np.array([r['group'] == 'chicken' and r['kind'] == 'imitation' and r['participant_id'] in fit_people
                         for r in rows])
    s = lambda X, u: sign[u] * X[:, u]
    groups = {'esc_chicken_fit': (RA, disc & chick), 'esc_other_fit': (RA, disc & ~chick),
              'vis_chicken_imitation_fit': (VA, fit_imit)}
    out = {'signs': {'L17 N1574': sign[U17], 'L19 N1129': sign[U19]}, 'fit_imitators': len(fit_people), 'groups': {}}
    for name, (X, m) in groups.items():
        a, b = s(X, U17)[m], s(X, U19)[m]
        out['groups'][name] = {'n': int(m.sum()), 's17_quantiles': np.quantile(a, [.1, .5, .9]).round(4).tolist(),
                               's19_quantiles': np.quantile(b, [.1, .5, .9]).round(4).tolist()}
    # gate thresholds from the fit imitations alone: t19 = their 25th percentile on s19, t17 = their 75th on s17
    a_i, b_i = s(VA, U17)[fit_imit], s(VA, U19)[fit_imit]
    t19, t17 = float(np.quantile(b_i, .25)), float(np.quantile(a_i, .75))
    def opens(X, m):
        return float(((s(X, U19)[m] > t19) & (s(X, U17)[m] < t17)).mean())
    out['gate'] = {'t19': t19, 't17': t17, 'open_share': {n: opens(X, m) for n, (X, m) in groups.items()}}
    other_open = (s(RA, U19) > t19) & (s(RA, U17) < t17) & disc & ~chick
    out['gate']['esc_other_open_labels'] = sorted(set(ref['label'][other_open].tolist()))
    (HERE / 'edit_pilot_gate.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
