"""Record EmbeddingGemma 2's shared-layer MLP neurons for the VocalImitationSet clips in vis_manifest.csv.

Same neuron definition as record.py and record_pixabay.py: the input to layers[i].mlp.down_proj, mean over positions,
one clip per forward pass, first 30 s of each clip. Audio: /mnt/e/neuron-probe/vocalimitationset/audio/<id>.wav
(not committed). Writes vis_acts.npz. Predictions V1 to V6 in PREDICTIONS.md, committed before any download.
Run: HF_HUB_OFFLINE=1 PYTHONPATH=/home/elisepc/eg2-extra /home/elisepc/neuron-probe/.venv/bin/python -B record_vis.py
"""
import csv, time
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from record import SNAP, decode

HERE = Path(__file__).resolve().parent
AUDIO = Path('/mnt/e/neuron-probe/vocalimitationset/audio')


def main():
    torch.set_num_threads(6)
    rows = list(csv.DictReader(open(HERE / 'vis_manifest.csv')))
    m = SentenceTransformer(SNAP, device='cpu', config_kwargs={'vision_config': None},
                            model_kwargs={'dtype': torch.float32})
    layers = m[0].model.language_model.layers
    buf = {}
    hooks = [layers[i].mlp.down_proj.register_forward_pre_hook(
        lambda mod, args, i=i: buf.__setitem__(i, args[0][0].float())) for i in range(len(layers))]
    N, U = len(rows), len(layers) * 2048
    act_mean = np.zeros((N, U), np.float16)
    emb = np.zeros((N, 768), np.float32)
    seqlen = np.zeros(N, np.int32)
    t0 = time.time()
    for n, r in enumerate(rows):
        buf.clear()
        a = decode(AUDIO / r['local'])[:16000 * 30]
        assert len(a) > 1600, (r['id'], len(a))
        with torch.inference_mode():
            emb[n] = np.asarray(m.encode({'audio': {'array': a, 'sampling_rate': 16000}}, normalize_embeddings=True),
                                dtype=np.float32)
        assert len(buf) == len(layers), (r['id'], len(buf))
        A = torch.stack([buf[i] for i in range(len(layers))])
        seqlen[n] = A.shape[1]
        act_mean[n] = A.mean(1).reshape(-1).numpy().astype(np.float16)
        if n % 25 == 0:
            print(f"{n}/{N} {r['group']} {r['kind']} T={A.shape[1]} {time.time() - t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    np.savez_compressed(HERE / 'vis_acts.npz', id=np.array([r['id'] for r in rows]),
                        group=np.array([r['group'] for r in rows]), kind=np.array([r['kind'] for r in rows]),
                        act_mean=act_mean, emb=emb, seqlen=seqlen)
    print('DONE', N, 'clips', round(time.time() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    main()
