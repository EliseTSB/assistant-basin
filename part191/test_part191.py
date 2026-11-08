"""Part 191: the chunked generator keeps row order and sizes, and the wrapper refuses a non-eager load."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from part191 import run191  # noqa: E402


class Push:
    """A stand-in for a [rows, width] tensor: slicing keeps row labels."""
    def __init__(self, rows):
        self.rows = list(rows)
        self.shape = (len(self.rows), 4)

    def __getitem__(self, s):
        return Push(self.rows[s])


def test_chunks_keep_order_and_sizes():
    calls = []

    def one(model, tok, layers, q, pushes, n, dev):
        rows = next(iter(pushes.values())).rows
        calls.append(len(rows))
        return [{'answer': f'r{r}'} for r in rows]

    gen = run191.chunked(one)
    pushes = {11: Push(range(20)), 24: Push(range(20))}
    out = gen(None, None, None, 'q', pushes, 5, 'cpu')
    assert calls == [8, 8, 2, 2]
    assert [a['answer'] for a in out] == [f'r{i}' for i in range(20)]


def test_plain_row_is_not_chunked():
    seen = []
    gen = run191.chunked(lambda *a: seen.append(a[4]) or [{'answer': 'p'}])
    assert gen(None, None, None, 'q', {}, 5, 'cpu') == [{'answer': 'p'}]
    assert seen == [{}]


def test_wrong_batch_or_short_chunk_refuses():
    gen = run191.chunked(lambda m, t, l, q, p, n, d: [{'answer': 'x'}] * next(iter(p.values())).shape[0])
    with pytest.raises(ValueError):
        gen(None, None, None, 'q', {1: Push(range(19))}, 5, 'cpu')
    short = run191.chunked(lambda *a: [{'answer': 'x'}])
    with pytest.raises(ValueError):
        short(None, None, None, 'q', {1: Push(range(20))}, 5, 'cpu')


def _fake_run(record_kernel, written=1):
    def run(args, loader=None, generate_fn=None):
        import neuron_report as nr
        nr.LAST_LOAD = {'attn_implementation': record_kernel}
        Path(args.out).write_text('', encoding='utf-8')
        return {'answers_written': written}
    return run


@pytest.mark.parametrize('kernel,rc', [('eager', 0), ('sdpa', 3), (None, 3)])
def test_gate_on_recorded_kernel(tmp_path, monkeypatch, kernel, rc):
    from part153 import battery153
    monkeypatch.setattr(battery153, 'run', _fake_run(kernel))
    out = tmp_path / 'answers.jsonl'
    assert run191.main(['--out', str(out)], generate_one=lambda *a: []) == rc
    rec = json.loads((tmp_path / 'load.json').read_text(encoding='utf-8'))
    assert rec['loads'][-1]['chunks'] == [8, 8, 2, 2]
    assert (rec['loads'][-1]['load'] or {}).get('attn_implementation') == kernel
    assert run191.check_record(tmp_path / 'load.json') == rc


def test_resume_cannot_launder_the_record(tmp_path, monkeypatch):
    """A later invocation that writes nothing leaves the record alone; a later non-eager one that writes rows is kept
    beside the eager entry and fails the gate (DeepSeek review F3)."""
    from part153 import battery153
    out = tmp_path / 'answers.jsonl'
    monkeypatch.setattr(battery153, 'run', _fake_run('eager', written=21))
    assert run191.main(['--out', str(out)], generate_one=lambda *a: []) == 0
    monkeypatch.setattr(battery153, 'run', _fake_run('sdpa', written=0))
    assert run191.main(['--out', str(out)], generate_one=lambda *a: []) == 0
    rec = json.loads((tmp_path / 'load.json').read_text(encoding='utf-8'))
    assert [e['load']['attn_implementation'] for e in rec['loads']] == ['eager']
    monkeypatch.setattr(battery153, 'run', _fake_run('sdpa', written=3))
    assert run191.main(['--out', str(out)], generate_one=lambda *a: []) == 3
    rec = json.loads((tmp_path / 'load.json').read_text(encoding='utf-8'))
    assert [e['load']['attn_implementation'] for e in rec['loads']] == ['eager', 'sdpa']
    assert run191.check_record(tmp_path / 'load.json') == 3


def test_no_rows_written_ever_fails(tmp_path, monkeypatch):
    from part153 import battery153
    monkeypatch.setattr(battery153, 'run', _fake_run('eager', written=0))
    assert run191.main(['--out', str(tmp_path / 'a.jsonl')], generate_one=lambda *a: []) == 3


def test_real_battery_run_through_the_wrapper(tmp_path):
    """Part 153's real run() and Part 148's real pushes_for, a stand-in model and a generator that fingerprints each
    row's push: 840 rows, calls of 8, 8, 2, 2 and 1, every answer on the row whose push it saw, and the kernel the
    loader recorded reaches the gate (DeepSeek review F4)."""
    torch = pytest.importorskip('torch')
    import types
    import neuron_report as nr

    width, nlayers = 64, 26

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.down_proj = torch.nn.Linear(width, 2, bias=False)

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block() for _ in range(nlayers)])
            self.config = types.SimpleNamespace(eos_token_id=None)

    model = Fake()
    units = tmp_path / 'units.json'
    real_units = json.loads((ROOT / 'results/part112/units.json').read_text(encoding='utf-8'))
    units.write_text(json.dumps(real_units), encoding='utf-8')
    medians = {str(i): 1.0 for i in range(nlayers)}
    (tmp_path / 'medians.json').write_text(json.dumps(medians), encoding='utf-8')
    from part153 import battery153
    wanted = {(u['layer'], u['neuron']) for u in battery153.fixed_units(units)}
    if any(n >= width for _, n in wanted):
        width = max(n for _, n in wanted) + 1
        model = Fake()
    calls = []

    def fp(row):
        idx = torch.nonzero(row).flatten().tolist()
        return f'{len(idx)}:{idx[0] if idx else -1}:{float(row.abs().sum()):.4f}'

    def one(m, tok, layers, q, pushes, n, dev):
        if not pushes:
            calls.append(1)
            return [{'answer': 'PLAIN', 'token_count': 1}]
        b = int(next(iter(pushes.values())).shape[0])
        calls.append(b)
        return [{'answer': '|'.join(f'{k}={fp(v[i])}' for k, v in sorted(pushes.items())), 'token_count': 1}
                for i in range(b)]

    def loader(device):
        nr.LAST_LOAD = {'attn_implementation': 'eager'}
        return object(), model

    out = tmp_path / 'answers.jsonl'
    rc = run191.main(['--out', str(out), '--units', str(units), '--medians', str(tmp_path / 'medians.json')],
                     loader=loader, generate_one=one)
    assert rc == 0
    rows = [json.loads(l) for l in out.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(rows) == 840 and len({(r['unit_id'], r['arm'], r['question_index']) for r in rows}) == 840
    assert calls[:5] == [8, 8, 2, 2, 1] and sorted(set(calls)) == [1, 2, 8]
    # every pushed answer carries a nonzero push in exactly its own unit's layer, at its own neuron for the neuron arm
    for r in rows:
        if r['arm'] == 'plain':
            continue
        parts = dict(p.split('=', 1) for p in r['answer'].split('|'))
        nonzero = [int(k) for k, v in parts.items() if not v.startswith('0:')]
        assert nonzero == [r['layer']], r
        if r['arm'] == 'neuron':
            assert parts[str(r['layer'])].split(':')[1] == str(r['neuron']), r


def test_shared_loader_defaults_to_eager():
    import inspect
    import neuron_report as nr
    assert inspect.signature(nr.load_model).parameters['attn_implementation'].default == 'eager'
