"""Record EmbeddingGemma 2's shared-layer MLP neurons for the labelled Pixabay chicken clips (real, imitation, ai).

Same neuron definition as record.py: the input to layers[i].mlp.down_proj, mean over positions, one clip per forward
pass. Labels: pixabay_labels.csv (assigned from Pixabay metadata before any recording). Audio:
/mnt/e/neuron-probe/pixabay-chicken/<id>.mp3 (Pixabay Content License; not committed). Writes pixabay_acts.npz.
Predictions X1 to X4 in PREDICTIONS.md, committed before any audio was downloaded.
Run: HF_HUB_OFFLINE=1 PYTHONPATH=/home/elisepc/eg2-extra /home/elisepc/neuron-probe/.venv/bin/python -B record_pixabay.py
"""
import csv, time
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from record import SNAP, decode

HERE = Path(__file__).resolve().parent
AUDIO = Path('/mnt/e/neuron-probe/pixabay-chicken')


def main():
    torch.set_num_threads(6)
    rows = [r for r in csv.DictReader(open(HERE / 'pixabay_labels.csv')) if r['label'] in ('real', 'imitation', 'ai')]
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
        a = decode(AUDIO / f"{r['id']}.mp3")[:16000 * 30]  # first 30 s of every clip
        with torch.inference_mode():
            emb[n] = np.asarray(m.encode({'audio': {'array': a, 'sampling_rate': 16000}}, normalize_embeddings=True),
                                dtype=np.float32)
        assert len(buf) == len(layers), (r['id'], len(buf))
        A = torch.stack([buf[i] for i in range(len(layers))])
        seqlen[n] = A.shape[1]
        act_mean[n] = A.mean(1).reshape(-1).numpy().astype(np.float16)
        if n % 25 == 0:
            print(f"{n}/{N} {r['label']} T={A.shape[1]} {time.time() - t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    np.savez_compressed(HERE / 'pixabay_acts.npz', id=np.array([r['id'] for r in rows]),
                        label=np.array([r['label'] for r in rows]), act_mean=act_mean, emb=emb, seqlen=seqlen)
    print('DONE', N, 'clips', round(time.time() - t0, 1), 's', flush=True)


if __name__ == '__main__':
    main()
