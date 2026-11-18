"""Select the VocalImitationSet clips for predictions V1 to V6 (PREDICTIONS.md), fixed by seed 20261104.

Reads the dataset's git tree (commit b77114c43c52f5faa7d0bb69b22e1ebca0632201) and vocal_imitations.txt from META, and
writes vis_manifest.csv: one row per clip with its repository path, local file name, class, group
(chicken / other) and kind (imitation / real), plus the imitation's participant id.
Run: /home/elisepc/neuron-probe/.venv/bin/python -I vis_select.py
"""
import csv, json, re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
META = Path('/mnt/e/neuron-probe/vocalimitationset/meta')
COMMIT = 'b77114c43c52f5faa7d0bb69b22e1ebca0632201'
CHICKEN = {9, 10}
ANIMAL = set(range(0, 30))


def cls_id(name):
    return int(name[:3])


def main():
    rng = np.random.default_rng(20261104)
    tree = json.load(open(META / 'tree.json'))
    assert tree['sha'] == COMMIT, tree['sha']
    blobs = sorted(x['path'] for x in tree['tree'] if x['type'] == 'blob' and x['path'].endswith('.wav'))
    meta = {r['imitation_filename']: r for r in csv.DictReader(open(META / 'vocal_imitations.txt'), delimiter='\t')}
    rows = []
    # imitations: included only
    imit = {}
    for p in blobs:
        if p.startswith('vocal_imitations/included/'):
            f = p.split('/')[-1]
            c = cls_id(f)
            if c in ANIMAL:
                imit.setdefault(c, []).append(p)
    for c in sorted(imit):
        ps = imit[c]
        pick = ps if c in CHICKEN else [ps[i] for i in sorted(rng.choice(len(ps), 4, replace=False))]
        for p in pick:
            f = p.split('/')[-1]
            m = meta[f]
            assert m['included'] == 'True', f
            rows.append({'path': p, 'class_id': c, 'class': f.rsplit('-', 1)[0][3:], 'group': 'chicken' if c in CHICKEN else 'other',
                         'kind': 'imitation', 'participant_id': m['participant_id']})
    # real recordings
    real = {}
    for p in blobs:
        if p.startswith('original_recordings/'):
            c = cls_id(p.split('/')[-1] if p.startswith('original_recordings/reference/') else p.split('/')[2])
            if c in ANIMAL:
                real.setdefault(c, []).append(p)
    for c in sorted(real):
        ps = real[c]
        if c in CHICKEN:
            pick = ps
        else:
            nr = [p for p in ps if '/non_reference/' in p]
            pick = [nr[i] for i in sorted(rng.choice(len(nr), 2, replace=False))]
        for p in pick:
            name = p.split('/')[-1] if '/reference/' in p else p.split('/')[2]
            rows.append({'path': p, 'class_id': c, 'class': re.sub(r'_reference\.wav$', '', name)[3:],
                         'group': 'chicken' if c in CHICKEN else 'other', 'kind': 'real', 'participant_id': ''})
    for i, r in enumerate(rows):
        r['id'] = f'v{i:04d}'
        r['local'] = r['id'] + '.wav'
    with open(HERE / 'vis_manifest.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['id', 'local', 'group', 'kind', 'class_id', 'class', 'participant_id', 'path'])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(Counter((r['group'], r['kind']) for r in rows))


if __name__ == '__main__':
    main()
