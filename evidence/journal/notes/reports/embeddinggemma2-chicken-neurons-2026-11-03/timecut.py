"""Cutting the chicken out of a sound in time: T1 to T4 (PREDICTIONS.md, written 2026-11-04 11:12 CT before any window
is scored). Inside each real chicken recording, 2 s windows (1 s hop, first 30 s) are labelled by an independent
classifier, the AudioSet AST model (MIT/ast-finetuned-audioset-10-10-0.4593 at f826b80, local copy), score = its
"Chicken, rooster" plus "Fowl" sigmoid probabilities. Within a recording the top third of windows by that score are ON,
the bottom third OFF, the middle dropped; recordings with fewer than 6 windows are dropped. EmbeddingGemma 2's neurons
as record.py (input to mlp.down_proj, mean over positions, one window per pass), each signed by its audio sign.
Recordings: the 20 VocalImitationSet real chickens and the 180 Pixabay real chicken clips. Writes timecut.npz and
timecut_results.json.
Run (alone on the machine, after the Arc job): HF_HUB_OFFLINE=1 PYTHONPATH=/home/elisepc/eg2-extra
  /home/elisepc/neuron-probe/.venv/bin/python -B timecut.py"""
import csv, json, time
from pathlib import Path
import numpy as np
import torch
from record import SNAP, decode
from analyze_pixabay import auc

HERE = Path(__file__).resolve().parent
AST = '/mnt/e/neuron-probe/ast-audioset'
SR, WIN, HOP, MAXS = 16000, 2, 1, 30
W = 2048
U17, U15 = 17 * W + 1574, 15 * W + 1126


def recordings():
    out = []
    for r in csv.DictReader(open(HERE / 'vis_manifest.csv')):
        if r['group'] == 'chicken' and r['kind'] == 'real':
            out.append(('vis:' + r['id'], Path('/mnt/e/neuron-probe/vocalimitationset/audio') / r['local']))
    for r in csv.DictReader(open(HERE / 'pixabay_labels.csv')):
        if r['label'] == 'real':
            out.append(('pixabay:' + r['id'], Path('/mnt/e/neuron-probe/pixabay-chicken') / f"{r['id']}.mp3"))
    return out


def audio_signs():
    d = np.load(HERE / 'acts.npz')
    A = d['act_mean'].astype(np.float32)
    disc = (d['kind'] == 'esc') & (d['fold'] <= 3)
    y = np.isin(d['label'], ['hen', 'rooster'])[disc]
    from scipy.stats import rankdata
    R = np.apply_along_axis(rankdata, 0, A[disc])
    p, n = y.sum(), (~y).sum()
    return np.where((R[y].sum(0) - p * (p + 1) / 2) / (p * n) >= 0.5, 1.0, -1.0).astype(np.float32)


def main():
    torch.set_num_threads(6)
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    from sentence_transformers import SentenceTransformer
    fe = ASTFeatureExtractor.from_pretrained(AST)
    ast = ASTForAudioClassification.from_pretrained(AST).eval()
    lab = {v: int(k) for k, v in ast.config.id2label.items()}
    chick_ids = [lab['Chicken, rooster'], lab['Fowl']]
    m = SentenceTransformer(SNAP, device='cpu', config_kwargs={'vision_config': None}, model_kwargs={'dtype': torch.float32})
    layers = m[0].model.language_model.layers
    buf = {}
    for i in range(len(layers)):
        layers[i].mlp.down_proj.register_forward_pre_hook(lambda mod, a, i=i: buf.__setitem__(i, a[0][0].float()))
    sign = audio_signs()
    U = len(layers) * W
    rec_ids, diffs, n_on = [], [], []
    tot, tot2, cnt = np.zeros(U, np.float64), np.zeros(U, np.float64), 0
    rows = []  # per window: rec, start, ast score, rms, s17, s15
    t0 = time.time()
    for ri, (rid, path) in enumerate(recordings()):
        a = decode(path)[:SR * MAXS]
        nwin = (len(a) - SR * WIN) // (SR * HOP) + 1
        if nwin < 6:
            continue
        acts, scores, rms = [], [], []
        for w in range(nwin):
            x = a[w * SR * HOP: w * SR * HOP + SR * WIN]
            with torch.inference_mode():
                f = fe(x, sampling_rate=SR, return_tensors='pt')
                scores.append(float(torch.sigmoid(ast(**f).logits)[0, chick_ids].sum()))
                buf.clear()
                m.encode({'audio': {'array': x, 'sampling_rate': SR}}, normalize_embeddings=True)
            v = torch.stack([buf[i].mean(0) for i in range(len(layers))]).reshape(-1).numpy() * sign
            acts.append(v)
            rms.append(float(np.sqrt(np.mean(x ** 2))))
        A = np.array(acts, np.float32)
        tot += A.sum(0); tot2 += (A.astype(np.float64) ** 2).sum(0); cnt += len(A)
        order = np.argsort(scores, kind='stable')
        k = len(order) // 3
        off, on = order[:k], order[-k:]
        diffs.append(np.median(A[on], 0) - np.median(A[off], 0))
        rec_ids.append(rid); n_on.append(k)
        # RMS-matched pairs: each ON window with the OFF window of nearest RMS (with replacement)
        pairs = [(i, min(off, key=lambda j: abs(rms[j] - rms[i]))) for i in on]
        for w in range(nwin):
            rows.append({'rec': rid, 'start_s': w * HOP, 'ast': scores[w], 'rms': rms[w], 'part': 'on' if w in set(on)
                         else 'off' if w in set(off) else 'mid', 's17': float(A[w, U17]), 's15': float(A[w, U15])})
        for i, j in pairs:
            rows.append({'rec': rid, 'pair_on': int(i), 'pair_off': int(j), 'd17': float(A[i, U17] - A[j, U17]),
                         'd15': float(A[i, U15] - A[j, U15])})
        if ri % 10 == 0:
            print(f'{ri} {rid} windows={nwin} {time.time() - t0:.0f}s', flush=True)
    D = np.array(diffs, np.float32)
    spread = np.sqrt(np.maximum(tot2 / cnt - (tot / cnt) ** 2, 0)) + 1e-12
    med = np.median(D, 0) / spread
    np.savez_compressed(HERE / 'timecut.npz', rec=np.array(rec_ids), diffs=D, spread=spread.astype(np.float32))
    pair_rows = [r for r in rows if 'pair_on' in r]

    def frac_pairs(key):
        per = {}
        for r in pair_rows:
            per.setdefault(r['rec'], []).append(r[key])
        return float(np.mean([np.median(v) > 0 for v in per.values()]))

    res = {'n_recordings': len(rec_ids), 'n_windows': cnt,
           'T1': {'value': float((D[:, U17] > 0).mean()), 'held': bool((D[:, U17] > 0).mean() >= 0.75)},
           'T2': {'value': frac_pairs('d17'), 'held': frac_pairs('d17') >= 0.70},
           'T3': {'rank': int((med > med[U17]).sum()) + 1, 'top_share': float(((med > med[U17]).sum() + 1) / med.size)},
           'T4': {'value': float((D[:, U15] > 0).mean()), 'held': bool((D[:, U15] > 0).mean() >= 0.75)},
           'descriptive': {'L15_rms_matched_share': frac_pairs('d15'),
                           'ast_auc_check_on_vs_off': auc(np.array([r['ast'] for r in rows if r.get('part') in ('on', 'off')]),
                                                          np.array([r.get('part') == 'on' for r in rows if r.get('part') in ('on', 'off')]),
                                                          np.array([r.get('part') == 'off' for r in rows if r.get('part') in ('on', 'off')]))}}
    res['T3']['held'] = res['T3']['top_share'] <= 0.05
    json.dump({'results': res, 'windows': [r for r in rows if 'part' in r], 'pairs': pair_rows},
              open(HERE / 'timecut_results.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
