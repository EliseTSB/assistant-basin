"""Drive the REAL part191.run191.main() with a stand-in model and generator (no torch, no network).

Run: python3 -I drive_run191.py /home/elisepc/green/worktrees/ab-part191
Shows: (1) a resumed run whose second invocation loads SDPA is caught by the gate; (2) a run where the loader never
sets neuron_report.LAST_LOAD (e.g. a loader that raises after loading, or a battery path that loads some other way)
writes load=None and the gate refuses; (3) an interrupted run (generator raises mid-battery) leaves a partial
answers.jsonl and no load.json entry, so the next invocation appends and the record has one entry for a file made by
two invocations.
"""
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = sys.argv[1]
sys.path.append('/home/elisepc/.local/lib/python3.12/site-packages')  # numpy only; -I ignores PYTHONPATH
sys.path.insert(0, ROOT)
import types as _t
_torch = _t.ModuleType('torch'); _torch.bfloat16 = 'bfloat16'; _torch.float32 = 'float32'
sys.modules.setdefault('torch', _torch)  # neuron_report imports torch at module level; unused here
import neuron_report as nr  # noqa: E402
from part191 import run191  # noqa: E402


class Push:
    def __init__(self, n):
        self.shape = (n, 4)

    def __getitem__(self, s):
        return Push(len(range(*s.indices(self.shape[0]))))


def fake_model():
    blocks = [types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=types.SimpleNamespace(weight=None)))
              for _ in range(26)]
    m = types.SimpleNamespace()
    m.named_modules = lambda: [('', m), ('model', types.SimpleNamespace()), ('model.layers', blocks)]
    m.requires_grad_ = lambda *a: None
    m.eval = lambda: m
    return m


def main():
    tmp = Path(tempfile.mkdtemp(prefix='p191-'))
    units = tmp / 'units.json'
    units.write_text(json.dumps([{'layer': l, 'neuron': n} for l, n, _ in
                                 __import__('part153.battery153', fromlist=['x']).UNIT_COMPANIES]), encoding='utf-8')
    medians = tmp / 'medians.json'
    medians.write_text(json.dumps({str(i): 1.0 for i in range(26)}), encoding='utf-8')
    out = tmp / 'answers.jsonl'

    # Patch the heavy pieces run191 reaches: _common (find_layers / pushes_for) and pushes_for.
    from part153 import battery153
    from part148 import battery148
    battery148.pushes_for = lambda rows, layers, medians, common: {11: Push(len(rows))}
    battery153.battery.pushes_for = battery148.pushes_for
    import neuron_report
    neuron_report.find_text_layers = lambda model: ('model.layers', model.named_modules()[2][1], None)

    def loader(kernel):
        def load(device):
            nr.LAST_LOAD = {'attn_implementation': kernel}
            return object(), fake_model()
        return load

    def gen(*a):
        pushes = a[4]
        n = next(iter(pushes.values())).shape[0] if pushes else 1
        return [{'answer': 'x', 'token_count': 1} for _ in range(n)]

    argv = ['--out', str(out), '--units', str(units), '--medians', str(medians)]
    print('(1) eager invocation rc =', run191.main(argv, loader=loader('eager'), generate_one=gen))
    print('(2) resumed sdpa invocation rc =', run191.main(argv, loader=loader('sdpa'), generate_one=gen))
    rec = json.loads((out.parent / 'load.json').read_text())
    print('    recorded kernels', [e['load']['attn_implementation'] for e in rec['loads']],
          'entries', len(rec['loads']), 'rows', sum(1 for _ in out.open()))

    # (3) interrupted run: generator raises on the 5th question
    tmp2 = Path(tempfile.mkdtemp(prefix='p191b-'))
    out2 = tmp2 / 'answers.jsonl'
    calls = {'n': 0}

    def flaky(*a):
        calls['n'] += 1
        if calls['n'] > 4:
            raise RuntimeError('simulated XPU fault')
        return gen(*a)

    argv2 = ['--out', str(out2), '--units', str(units), '--medians', str(medians)]
    try:
        run191.main(argv2, loader=loader('eager'), generate_one=flaky)
    except RuntimeError as exc:
        print('(3) interrupted run raised', exc)
    print('    partial rows', sum(1 for _ in out2.open()), 'load.json exists', (tmp2 / 'load.json').exists())
    print('    resume rc =', run191.main(argv2, loader=loader('eager'), generate_one=gen))
    rec2 = json.loads((tmp2 / 'load.json').read_text())
    print('    after resume: rows', sum(1 for _ in out2.open()), 'record entries', len(rec2['loads']),
          '(one entry for a file written by two invocations)')

    # (3b) interrupted run, then a resume on a DIFFERENT kernel: the record shows one eager entry for a mixed file
    tmp2b = Path(tempfile.mkdtemp(prefix='p191b2-'))
    out2b = tmp2b / 'answers.jsonl'
    calls['n'] = 0
    try:
        run191.main(['--out', str(out2b), '--units', str(units), '--medians', str(medians)],
                    loader=loader('eager'), generate_one=flaky)
    except RuntimeError:
        pass
    rc = run191.main(['--out', str(out2b), '--units', str(units), '--medians', str(medians)],
                     loader=loader('sdpa'), generate_one=gen)
    rec3 = json.loads((tmp2b / 'load.json').read_text())
    print('(3b) resume on sdpa rc =', rc, 'recorded kernels', [e['load']['attn_implementation'] for e in rec3['loads']],
          'rows', sum(1 for _ in out2b.open()), '-> gate certifies eager for a file whose last 820 rows are sdpa')

    # (4) a loader that raises before LAST_LOAD is set
    tmp3 = Path(tempfile.mkdtemp(prefix='p191c-'))
    out3 = tmp3 / 'answers.jsonl'
    def bad_loader(device):
        raise RuntimeError('OOM before load')
    try:
        run191.main(['--out', str(out3), '--units', str(units), '--medians', str(medians)],
                    loader=bad_loader, generate_one=gen)
    except RuntimeError as exc:
        print('(4) loader failure propagates:', exc, '| load.json exists', (tmp3 / 'load.json').exists())


main()
