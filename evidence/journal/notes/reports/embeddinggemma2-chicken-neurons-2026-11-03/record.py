"""Record every shared-layer MLP neuron of EmbeddingGemma 2 for the ESC-50 subset, our hen clips and the text items.

Neuron = the input to layers[i].mlp.down_proj (act(gate) * up), 24 layers x 2,048 = 49,152 units. Per item, the mean
and the max over sequence positions, one item per forward pass (no padding). Also the final embedding, and the
embeddings of one "the sound of <class>" prompt per ESC-50 class for the model-level classifier (control b).
Writes acts.npz. Predictions: PREDICTIONS.md (committed before the first run).
Run: HF_HUB_OFFLINE=1 PYTHONPATH=/home/elisepc/eg2-extra /home/elisepc/neuron-probe/.venv/bin/python -B record.py
"""
import csv, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
SNAP = '/mnt/e/neuron-probe/hf/hub/models--google--embeddinggemma-2/snapshots/f9972ac22e696a105c482202646979e33e4b5a1e'
FFMPEG = '/home/elisepc/eg2-extra/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v9.0.2'
ESC = Path('/mnt/e/neuron-probe/esc50')
OURS = HERE.parent / 'embeddinggemma2-hens-2026-11-03' / 'clips16k'


def esc_items():
    rows = list(csv.DictReader(open(ESC / 'esc50.csv')))
    keep, per = [], {}
    for r in sorted(rows, key=lambda r: r['filename']):
        t, f = int(r['target']), int(r['fold'])
        if t in (1, 6):
            keep.append(r)
        elif len(per.setdefault((t, f), [])) < 2:
            per[(t, f)].append(r)
            keep.append(r)
    missing = [r['filename'] for r in keep if not (ESC / 'audio' / r['filename']).exists()]
    if missing:
        sys.exit(f'{len(missing)} ESC-50 files missing, e.g. {missing[:3]}')
    return keep, sorted({r['category'] for r in rows}, key=lambda c: int(next(x['target'] for x in rows if x['category'] == c)))


def decode(path):
    p = subprocess.run([FFMPEG, '-nostdin', '-loglevel', 'error', '-i', str(path), '-ac', '1', '-ar', '16000',
                        '-f', 'f32le', 'pipe:1'], capture_output=True, timeout=60, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def main():
    torch.set_num_threads(6)
    m = SentenceTransformer(SNAP, device='cpu', config_kwargs={'vision_config': None},
                            model_kwargs={'dtype': torch.float32})
    layers = m[0].model.language_model.layers
    buf = {}
    hooks = [layers[i].mlp.down_proj.register_forward_pre_hook(
        lambda mod, args, i=i: buf.__setitem__(i, args[0][0].float())) for i in range(len(layers))]
    esc, categories = esc_items()
    texts = json.loads((HERE / 'text_items.json').read_text())
    items = ([('esc', r['filename'], r['category'], int(r['fold'])) for r in esc]
             + [('ours', p.stem, p.stem, 0) for p in sorted(OURS.glob('*.wav'))]
             + [('text', s, k, 0) for k in ('chicken', 'other') for s in texts[k]])
    N, U = len(items), len(layers) * 2048
    act_mean = np.zeros((N, U), np.float16)
    act_max = np.zeros((N, U), np.float16)
    emb = np.zeros((N, 768), np.float32)
    seqlen = np.zeros(N, np.int32)
    t0 = time.time()
    for n, (kind, key, label, fold) in enumerate(items):
        buf.clear()
        if kind == 'text':
            x = key
        else:
            a = decode(ESC / 'audio' / key if kind == 'esc' else OURS / f'{key}.wav')[:16000 * 300]
            x = {'audio': {'array': a, 'sampling_rate': 16000}}
        with torch.inference_mode():
            emb[n] = np.asarray(m.encode(x, normalize_embeddings=True), dtype=np.float32)
        assert len(buf) == len(layers), (key, len(buf))
        A = torch.stack([buf[i] for i in range(len(layers))])  # [L, T, 2048]
        seqlen[n] = A.shape[1]
        act_mean[n] = A.mean(1).reshape(-1).numpy().astype(np.float16)
        act_max[n] = A.amax(1).reshape(-1).numpy().astype(np.float16)
        if n % 50 == 0:
            print(f'{n}/{N} {kind} {label} T={A.shape[1]} {time.time() - t0:.0f}s', flush=True)
    for h in hooks:
        h.remove()
    prompts = [f'the sound of a {c.replace("_", " ")}' for c in categories]
    pemb = np.asarray(m.encode(prompts, prompt_name='SearchQuery', normalize_embeddings=True), dtype=np.float32)
    np.savez_compressed(HERE / 'acts.npz', kind=np.array([i[0] for i in items]), key=np.array([i[1] for i in items]),
                        label=np.array([i[2] for i in items]), fold=np.array([i[3] for i in items]),
                        act_mean=act_mean, act_max=act_max, emb=emb, seqlen=seqlen,
                        prompts=np.array(prompts), categories=np.array(categories), prompt_emb=pemb)
    print('DONE', N, 'items', round(time.time() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    main()
