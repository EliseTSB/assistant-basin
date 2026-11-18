"""Hens removed from photos: score I4 to I7 (PREDICTIONS.md, written 2026-11-04 09:48 CT before any edited image).

Pairs: sources/generated-images/2026-11-04-hen-removal/ (a Luna codex worker's image-tool edits; NOTES.md there).
A pair is usable when NOTES.md's removed check is `ok` and its kept check starts with `ok` (11 of 12; the kept image of
2026-07-25-gandalf-followed-inside-desk dropped a hen). The kept check was loosened before scoring: the tool redraws the
whole scene in both edits, so a kept image passes if every hen remains, reframing noted (Claude, 2026-11-04 10:14 CT, in
the resumed worker's brief, before any activation was read).
Activations as images.py and images_all.py: input to each layer's mlp.down_proj, mean over positions, one image per
pass, all 24 x 2048 neurons; each neuron signed by its audio sign (ESC-50 discovery folds, hen and rooster positive).
Writes images_removed.npz and images_removed.json.
Run: PYTHONPATH=/home/elisepc/eg2-extra /home/elisepc/neuron-probe/.venv/bin/python -B images_removed.py"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[2]
EDITS = JOURNAL / 'sources' / 'generated-images' / '2026-11-04-hen-removal'
HENS = JOURNAL / 'sources' / 'photos' / 'hens'
SNAP = '/mnt/e/neuron-probe/hf/hub/models--google--embeddinggemma-2/snapshots/f9972ac22e696a105c482202646979e33e4b5a1e'
W = 2048


def usable_pairs():
    rows = []
    for line in (EDITS / 'NOTES.md').read_text(encoding='utf-8').splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 6 or cells[0] in ('photo', '---') or cells[0].startswith('-'):
            continue
        rows.append((cells[0], cells[3].lower(), cells[4].lower()))
    keep = [p for p, rem, kept in rows if rem == 'ok' and kept.startswith('ok')]
    drop = [p for p, rem, kept in rows if p not in keep]
    return keep, drop, len(rows)


def audio_signs():
    d = np.load(HERE / 'acts.npz')
    A = d['act_mean'].astype(np.float32)
    disc = (d['kind'] == 'esc') & (d['fold'] <= 3)
    y = np.isin(d['label'], ['hen', 'rooster'])[disc]
    R = np.apply_along_axis(rankdata, 0, A[disc])
    p, n = y.sum(), (~y).sum()
    auc = (R[y].sum(0) - p * (p + 1) / 2) / (p * n)
    return np.where(auc >= 0.5, 1.0, -1.0)


def main():
    keep, drop, n_rows = usable_pairs()
    assert n_rows == 12, n_rows
    sign = audio_signs()
    m = SentenceTransformer(SNAP, device='cpu', model_kwargs={'dtype': torch.float32})
    layers = m[0].model.language_model.layers
    grab = {}
    for L in range(len(layers)):
        layers[L].mlp.down_proj.register_forward_pre_hook(
            lambda mod, a, L=L: grab.__setitem__(L, a[0][0].float().mean(0)))

    def acts(path):
        grab.clear()
        with torch.inference_mode():
            m.encode({'image': str(path)}, normalize_embeddings=True)
        assert len(grab) == len(layers), path
        return torch.cat([grab[L] for L in range(len(layers))]).numpy() * sign

    O, K, Rm = [], [], []
    for p in keep:
        O.append(acts(HENS / f'{p}.jpg'))
        K.append(acts(EDITS / f'{p}__kept.png'))
        Rm.append(acts(EDITS / f'{p}__removed.png'))
        print('done', p, flush=True)
    O, K, Rm = map(np.array, (O, K, Rm))
    np.savez_compressed(HERE / 'images_removed.npz', photos=np.array(keep), original=O.astype(np.float32),
                        kept=K.astype(np.float32), removed=Rm.astype(np.float32))
    diff = K - Rm
    med = np.median(diff, 0)
    # scale-free companion (not registered): the median difference over each neuron's spread across all images here
    spread = np.concatenate([O, K, Rm]).std(0) + 1e-12
    medz = med / spread
    out = {'usable': keep, 'dropped': drop, 'n_usable': len(keep), 'units': {}}
    for L, N in [(17, 1574), (15, 1126)]:
        u = L * W + N
        out['units'][f'L{L} N{N}'] = {
            'sign': int(sign[u]),
            'kept_minus_removed': diff[:, u].tolist(),
            'kept_above_removed': int((diff[:, u] > 0).sum()),
            'median_kept_minus_removed': float(med[u]),
            'rank_of_median_among_all': int((med > med[u]).sum()) + 1,
            'top_share': float(((med > med[u]).sum() + 1) / med.size),
            'rank_of_standardized_median': int((medz > medz[u]).sum()) + 1,
            'median_abs_kept_minus_original': float(np.median(np.abs(K[:, u] - O[:, u]))),
            'median_abs_removed_minus_original': float(np.median(np.abs(Rm[:, u] - O[:, u]))),
        }
    a, b = out['units']['L17 N1574'], out['units']['L15 N1126']
    n = len(keep)
    out['predictions'] = {
        'I4': {'value': f"{a['kept_above_removed']} of {n}", 'held': a['kept_above_removed'] >= 10},
        'I5': {'value': f"{b['kept_above_removed']} of {n}", 'held': b['kept_above_removed'] >= 10},
        'I6': {'value': a['top_share'], 'held': a['top_share'] <= 0.05},
        'I7': {'kept_vs_original': a['median_abs_kept_minus_original'],
               'removed_vs_original': a['median_abs_removed_minus_original'],
               'held': a['median_abs_kept_minus_original'] < 0.5 * a['median_abs_removed_minus_original']},
    }
    (HERE / 'images_removed.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'units'}, indent=1))
    for k, v in out['units'].items():
        print(k, {x: v[x] for x in v if x != 'kept_minus_removed'})


if __name__ == '__main__':
    main()
