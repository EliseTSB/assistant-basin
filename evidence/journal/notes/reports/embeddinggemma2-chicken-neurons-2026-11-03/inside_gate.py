"""The written gate, written inside the network (EDIT-DESIGN.md Amendment 3).

Same gate as the output version (L17 N1574 +1, L19 N1129 -1, float16 rounding, t19 and t17 frozen from
edit_verdicts.json). When it opens, a forward hook on layers[19] adds -k * g * R * d to that layer's output at every
position, so layers 20 to 23 carry the push. d: residual-stream chicken direction from ESC-50 folds 1-3 (mean over
positions of layers[19]'s output, hen and rooster minus the rest, normalised). R: mean per-position norm of that output
over the same clips. Only k is fitted, on the fixed grid, by the written block's rule. Control: a seeded random unit
direction, k refitted the same way. Closed clips keep their stored embedding (bit-identical by construction; the
parity check below confirms the live path reproduces the stored arrays).

Run: HF_HUB_OFFLINE=1 PYTHONPATH=/home/elisepc/eg2-extra /home/elisepc/neuron-probe/.venv/bin/python -B inside_gate.py
Writes inside_results.json and inside_params.npz (d, R, k for the meter).
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import edit_arms as ea  # noqa: E402
from record import SNAP, decode  # noqa: E402

ESC = Path('/mnt/e/neuron-probe/esc50/audio')
PIX = Path('/mnt/e/neuron-probe/pixabay-chicken')
VIS = Path('/mnt/e/neuron-probe/vocalimitationset/audio')
W = 2048
L17, N17, L19, N19 = 17, 1574, 19, 1129
WRITE_AFTER = 19
RANDOM_SEED = 20261104
HUMAN = {'laughing', 'crying_baby', 'sneezing', 'coughing', 'breathing', 'snoring', 'clapping', 'footsteps',
         'brushing_teeth', 'drinking_sipping'}


class InsideModel:
    def __init__(self, t19, t17, s17, s19):
        torch.set_num_threads(6)
        self.m = SentenceTransformer(SNAP, device='cpu', config_kwargs={'vision_config': None},
                                     model_kwargs={'dtype': torch.float32})
        layers = self.m[0].model.language_model.layers
        self.t19, self.t17, self.s17, self.s19 = t19, t17, s17, s19
        self.grab = {}
        self.d = None
        self.scale = 0.0  # k * R; 0 means never write
        self.capture = None
        self.last_g = None
        for L in (L17, L19):
            layers[L].mlp.down_proj.register_forward_pre_hook(
                lambda mod, args, L=L: self.grab.__setitem__(L, args[0][0].float().mean(0)))
        layers[WRITE_AFTER].register_forward_hook(self._after19)

    def gate(self):
        s17 = self.s17 * float(np.float16(self.grab[L17][N17]))
        s19 = self.s19 * float(np.float16(self.grab[L19][N19]))
        return s17, s19, max(s19 - self.t19, 0.0) * max(self.t17 - s17, 0.0)

    def _after19(self, mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.capture is not None:
            x = h[0].float()
            self.capture.append((x.mean(0).numpy().copy(), float(x.norm(dim=-1).mean())))
        s17, s19, g = self.gate()
        self.last_g = (s17, s19, g)
        if g > 0.0 and self.scale > 0.0 and self.d is not None:
            new = h - (self.scale * g) * self.d.to(h.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return None

    def embed(self, audio):
        self.grab.clear()
        with torch.inference_mode():
            e = self.m.encode({'audio': {'array': audio, 'sampling_rate': 16000}}, normalize_embeddings=True)
        return np.asarray(e, dtype=np.float32).reshape(-1)


def main():
    t0 = time.time()
    ver = json.loads((HERE / 'edit_verdicts.json').read_text())
    par = ver['arms']['written_gated_block']['parameters']
    s17, s19 = int(ver['signs']['L17_N1574']), int(ver['signs']['L19_N1129'])
    ref, vis, pix = np.load(HERE / 'acts.npz'), np.load(HERE / 'vis_acts.npz'), np.load(HERE / 'pixabay_acts.npz')
    manifest = list(csv.DictReader(open(HERE / 'vis_manifest.csv')))
    fit_imit, _ = ea.fit_imitation_split(manifest)
    cats = list(ref['categories'])
    P = np.asarray(ref['prompt_emb'], dtype=np.float32)
    hen, roo = cats.index('hen'), cats.index('rooster')
    other = np.asarray([i for i in range(len(cats)) if i not in (hen, roo)])
    c_out = ea.normalize_vector(P[[hen, roo]].mean(axis=0))
    u17, u19 = L17 * W + N17, L19 * W + N19

    def stored_gate(act):
        a17 = s17 * act[:, u17].astype(np.float32)
        a19 = s19 * act[:, u19].astype(np.float32)
        return a19, a17, np.maximum(a19 - par['t19'], 0) * np.maximum(par['t17'] - a17, 0)

    def score(E):
        return ea.prompt_scores(E, P, hen, roo, other)

    M = InsideModel(par['t19'], par['t17'], s17, s19)
    print(f'model loaded {time.time() - t0:.0f}s', flush=True)

    # 1. Discovery pass (ESC-50 folds 1-3), no write: d, R, and a parity check against the stored arrays.
    esc = ref['kind'] == 'esc'
    chicken = np.isin(ref['label'], ['hen', 'rooster'])
    disc = np.where(esc & (ref['fold'] <= 3))[0]
    means, norms, worst_n, worst_cos = [], [], 0.0, 1.0
    for j, i in enumerate(disc):
        M.capture = []
        e = M.embed(decode(ESC / ref['key'][i]))
        mu, nm = M.capture[0]
        M.capture = None
        means.append(mu)
        norms.append(nm)
        a19, a17, _ = stored_gate(ref['act_mean'][i:i + 1])
        ls17, ls19, _ = M.last_g
        worst_n = max(worst_n, abs(ls17 - float(a17[0])), abs(ls19 - float(a19[0])))
        worst_cos = min(worst_cos, float(np.dot(e, ref['emb'][i])))
        if j % 50 == 0:
            print(f'discovery {j}/{len(disc)} {time.time() - t0:.0f}s', flush=True)
    means = np.asarray(means, dtype=np.float32)
    dch = chicken[disc]
    d_vec = ea.normalize_vector(means[dch].mean(0) - means[~dch].mean(0))
    R = float(np.mean(norms))
    rng = np.random.default_rng(RANDOM_SEED)
    r_vec = ea.normalize_vector(rng.standard_normal(d_vec.shape[0]).astype(np.float32))
    parity = {'max_neuron_abs_diff': worst_n, 'min_cos_to_stored_emb': worst_cos, 'clips': int(len(disc))}
    print('parity', parity, 'R', round(R, 3), flush=True)
    if worst_n > 1e-3 or worst_cos < 0.999:
        raise SystemExit(f'live path does not reproduce the stored arrays: {parity}')

    # Sets, with stored embeddings and stored gates.
    pr = pix['label'] == 'real'
    efc = esc & (ref['fold'] <= 3) & chicken
    vis_imit = (vis['group'] == 'chicken') & (vis['kind'] == 'imitation')
    sets = {
        'fit_real': [(PIX / f'{x}.mp3', 30) for x in pix['id'][pr]] + [(ESC / k, None) for k in ref['key'][efc]],
        'fit_imit': [(VIS / r['local'], 30) for r, f in zip(manifest, fit_imit) if f],
        'test_imit': [(VIS / r['local'], 30) for r, f, m in zip(manifest, fit_imit, vis_imit) if m and not f],
        'test_real': [(VIS / r['local'], 30) for r in manifest if r['group'] == 'chicken' and r['kind'] == 'real'],
        'esc_test': [(ESC / k, None) for k in ref['key'][esc & np.isin(ref['fold'], [4, 5])]],
    }
    stored = {
        'fit_real': (np.concatenate([pix['emb'][pr], ref['emb'][efc]]),
                     np.concatenate([pix['act_mean'][pr], ref['act_mean'][efc]])),
        'fit_imit': (vis['emb'][fit_imit], vis['act_mean'][fit_imit]),
        'test_imit': (vis['emb'][vis_imit & ~fit_imit], vis['act_mean'][vis_imit & ~fit_imit]),
        'test_real': (vis['emb'][(vis['group'] == 'chicken') & (vis['kind'] == 'real')],
                      vis['act_mean'][(vis['group'] == 'chicken') & (vis['kind'] == 'real')]),
        'esc_test': (ref['emb'][esc & np.isin(ref['fold'], [4, 5])], ref['act_mean'][esc & np.isin(ref['fold'], [4, 5])]),
    }
    gates, audio = {}, {}
    for name in sets:
        a19, a17, g = stored_gate(stored[name][1])
        gates[name] = (a19, a17, g)
        audio[name] = {}
        for i in np.where(g > 0)[0]:
            path, cut = sets[name][i]
            x = decode(path)
            audio[name][int(i)] = x[:16000 * cut] if cut else x
    print('open:', {k: len(v) for k, v in audio.items()}, flush=True)

    def edited(name, direction, k):
        E = stored[name][0].copy()
        if k > 0:
            M.d = torch.from_numpy(direction)
            M.scale = float(k) * R
            for i, x in audio[name].items():
                E[i] = M.embed(x)
            M.scale = 0.0
        return E

    # 2. Fit k for d and for the random control, by the written block's rule.
    fits = {}
    for label, vec in (('inside', d_vec), ('random_inside', r_vec)):
        best = None
        for k in ea.K_GRID:
            v = ea.auc_values(score(edited('fit_real', vec, k)), score(edited('fit_imit', vec, k)))
            if best is None or v > best['fit_auc']:
                best = {'k': float(k), 'fit_auc': float(v)}
        fits[label] = best
        moved = edited('fit_imit', vec, best['k'])
        if np.array_equal(moved, stored['fit_imit'][0]):
            raise SystemExit(f'{label}: the write changed nothing at k={best["k"]}')
        print(label, best, f'{time.time() - t0:.0f}s', flush=True)

    # 3. Score on the held-out split, beside the output version and the unedited model.
    nb = ea.make_bootstrap_indices(len(sets['test_real']), len(sets['test_imit']))
    out = {'parity': parity, 'R': R, 'fits': fits, 'arms': {}}
    base_real, base_imit = stored['test_real'][0], stored['test_imit'][0]
    out['baseline_test_auc'] = ea.auc_values(score(base_real), score(base_imit))
    gi = gates['test_imit'][2] > 0
    esc_labels = ref['label'][esc & np.isin(ref['fold'], [4, 5])]
    nonchicken = ~np.isin(esc_labels, ['hen', 'rooster'])
    for label, vec in (('inside', d_vec), ('random_inside', r_vec), ('output', None)):
        if vec is None:  # the output version, from stored arrays as edit_arms.py does
            def ed(name):
                a19, a17, _ = gates[name]
                return ea.apply_gate_batch(stored[name][0], a19, a17, par['t19'], par['t17'], par['k'], c_out)
        else:
            def ed(name, vec=vec, k=fits[label]['k']):
                return edited(name, vec, k)
        er, ei, ee = ed('test_real'), ed('test_imit'), ed('esc_test')
        top = ea.prompt_predictions(ei[gi], P, cats)
        cos = np.sum(ei[gi] * base_imit[gi], axis=1)
        dmg = ea.class_damage(stored['esc_test'][0], ee, esc_labels, P, cats)
        out['arms'][label] = {
            'k': fits[label]['k'] if label in fits else par['k'],
            'test_auc': ea.auc_values(score(er), score(ei)),
            'test_auc_bootstrap_95': ea.bootstrap_ci(score(er), score(ei), nb),
            'gated_test_imitations': int(gi.sum()),
            'gated_top_classes': {str(c): int(n) for c, n in zip(*np.unique(top, return_counts=True))},
            'gated_human_share': float(np.mean([t in HUMAN for t in top])),
            'gated_median_cos_to_original': float(np.median(cos)),
            'nonchicken_bit_identical': int(np.sum(np.all(ee[nonchicken] == stored['esc_test'][0][nonchicken], axis=1))),
            'nonchicken_total': int(nonchicken.sum()),
            'classes_changed_more_than_1_clip': dmg['classes_changed_more_than_1_clip'],
        }
        print(label, json.dumps(out['arms'][label])[:400], flush=True)
    a, o, r = out['arms']['inside'], out['arms']['output'], out['arms']['random_inside']
    out['verdicts'] = {
        'I1': a['test_auc'] >= 0.75,
        'I2': len(a['classes_changed_more_than_1_clip']) == 0,
        'I3': len(a['gated_top_classes']) >= 3 and a['gated_human_share'] >= 0.5,
        'I4': a['gated_median_cos_to_original'] > o['gated_median_cos_to_original'],
        'I5': r['test_auc'] < a['test_auc'],
    }
    out['seconds'] = round(time.time() - t0, 1)
    (HERE / 'inside_results.json').write_text(json.dumps(out, indent=1))
    np.savez(HERE / 'inside_params.npz', d=d_vec, r=r_vec, R=np.float32(R), k=np.float64(fits['inside']['k']),
             k_random=np.float64(fits['random_inside']['k']))
    print('VERDICTS', out['verdicts'], 'seconds', out['seconds'], flush=True)


if __name__ == '__main__':
    main()
