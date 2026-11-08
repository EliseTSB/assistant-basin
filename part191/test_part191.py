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


def _fake_run(record_kernel):
    def run(args, loader=None, generate_fn=None):
        import neuron_report as nr
        nr.LAST_LOAD = {'attn_implementation': record_kernel}
        Path(args.out).write_text('', encoding='utf-8')
        return {'answers_written': 0}
    return run


@pytest.mark.parametrize('kernel,rc', [('eager', 0), ('sdpa', 3), (None, 3)])
def test_gate_on_recorded_kernel(tmp_path, monkeypatch, kernel, rc):
    from part153 import battery153
    monkeypatch.setattr(battery153, 'run', _fake_run(kernel))
    out = tmp_path / 'answers.jsonl'
    assert run191.main(['--out', str(out)], generate_one=lambda *a: []) == rc
    rec = json.loads((tmp_path / 'load.json').read_text(encoding='utf-8'))
    assert rec['part191']['chunks'] == [8, 8, 2, 2]
    assert (rec['load'] or {}).get('attn_implementation') == kernel


def test_shared_loader_defaults_to_eager():
    import inspect
    import neuron_report as nr
    assert inspect.signature(nr.load_model).parameters['attn_implementation'].default == 'eager'
