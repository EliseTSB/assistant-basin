"""Independent round-3 mutations; source repository is never modified."""
from pathlib import Path
import concurrent.futures
import json
import os
import re
import shutil
import subprocess

ROOT = Path('/home/elisepc/green/worktrees/ab-part67')
OUT = Path(__file__).resolve().parent
PY = '/home/elisepc/neuron-probe/.venv/bin/python'
MUTATIONS = [
    ('N1-screen-zero-neuron-dose', 'screen86.py',
     'pushes[row_index, unit_index] = deltas[unit_index]',
     'pushes[row_index, unit_index] = 0 * deltas[unit_index]'),
    ('N2-screen-all-neurons-push-unit0', 'screen86.py',
     'pushes[row_index, unit_index] = deltas[unit_index]',
     'pushes[row_index, 0] = deltas[unit_index]'),
    ('N3-screen-random-double-dose', 'screen86.py',
     'weight, n_directions, c.SCREEN_SEED + layer_no, c.DOSE * median',
     'weight, n_directions, c.SCREEN_SEED + layer_no, 2 * c.DOSE * median'),
    ('N4-screen-token-budget24', 'screen86.py',
     'MAX_NEW_TOKENS = 48', 'MAX_NEW_TOKENS = 24'),
    ('N5-verify-push-unit0', 'verify86.py',
     'row[0, int(unit)] = float(value)', 'row[0, 0] = float(value)'),
    ('N6-verify-greedy', 'verify86.py',
     'do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, seed=seed,',
     'do_sample=False, temperature=TEMPERATURE, top_p=TOP_P, seed=seed,'),
    ('N7-verify-seeds-plus10', 'verify86.py',
     'return c.VERIFY_SEED + 100 * question_index + sample_index',
     'return c.VERIFY_SEED + 100 * question_index + sample_index + 10'),
    ('N8-screen-second-reader20pct', 'blind86.py',
     'int(round(len(stable_ids) * 0.10))', 'int(round(len(stable_ids) * 0.20))'),
    ('N9-verify-random-double-dose', 'verify86.py',
     'DIRECTIONS, random_seed, c.DOSE * median)',
     'DIRECTIONS, random_seed, 2 * c.DOSE * median)'),
    ('N10-verify-case-question0', 'verify86.py',
     'cases.append(("neuron", None, question, qi, sample, seed_for(qi, sample), None))',
     'cases.append(("neuron", None, c.VERIFY_QUESTIONS[0], qi, sample, seed_for(qi, sample), None))'),
]

def run(item):
    name, filename, old, new = item
    target = OUT / 'new-mutations' / name
    target.mkdir(parents=True)
    shutil.copytree(ROOT / 'part86', target / 'part86', symlinks=True)
    shutil.copy2(ROOT / 'neuron_report.py', target / 'neuron_report.py')
    (target / 'part84').symlink_to(ROOT / 'part84', target_is_directory=True)
    path = target / 'part86' / filename
    source = path.read_text()
    assert source.count(old) == 1, (name, source.count(old))
    path.write_text(source.replace(old, new, 1))
    env = dict(os.environ, PART86_GPU_STRICT='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               TMPDIR=str(OUT / 'tmp'))
    result = subprocess.run([PY, '-I', '-B', 'part86/test86.py'], cwd=target,
                            env=env, capture_output=True, text=True, timeout=600)
    log = result.stdout + result.stderr
    (target / 'test.log').write_text(log)
    match = re.search(r'Ran (\d+) tests', log)
    failed = re.findall(r'^(?:FAIL|ERROR): (.+)', log, re.M)
    status = 'SURVIVED' if result.returncode == 0 and match and not 'skipped' in log else 'KILLED' if failed else 'INVALID'
    row = dict(name=name, status=status, tests=int(match[1]) if match else None,
               exit=result.returncode, failed=failed)
    print(json.dumps(row), flush=True)
    return row

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(run, MUTATIONS))
    (OUT / 'new-mutations.json').write_text(json.dumps(rows, indent=2) + '\n')
