# job.py — runs each line of steps.txt as `python <line>` in this folder, in order, stopping at the first failure
# (Claude fork, 2026-10-16). Output of every step goes to step-<n>.log; one line per step to progress.log.
import os, shlex, subprocess, sys, time
here = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, HF_HOME=r'E:\neuron-probe\hf', HF_HUB_OFFLINE='1', TRANSFORMERS_VERBOSITY='error',
           HF_HUB_DISABLE_PROGRESS_BARS='1', PYTHONIOENCODING='utf-8')
prog = open(os.path.join(here, 'progress.log'), 'a', encoding='utf-8')
steps = [l.strip() for l in open(os.path.join(here, 'steps.txt'), encoding='utf-8') if l.strip() and not l.startswith('#')]
for n, line in enumerate(steps, 1):
    prog.write(f'{time.strftime("%H:%M:%S")} step {n} start: {line}\n'); prog.flush()
    with open(os.path.join(here, f'step-{n}.log'), 'w', encoding='utf-8') as log:
        rc = subprocess.call([sys.executable] + shlex.split(line, posix=False), cwd=here, env=env, stdout=log,
                             stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    prog.write(f'{time.strftime("%H:%M:%S")} step {n} rc={rc}\n'); prog.flush()
    if rc: sys.exit(rc)
prog.write(f'{time.strftime("%H:%M:%S")} all steps done\n')
