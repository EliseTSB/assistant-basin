import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == Path(__file__).resolve().parent)]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part71 import part71 as p71  # noqa: E402


def test_best_rank_counts_strictly_greater_and_takes_the_best_id():
    logits = [0.1, 5.0, 3.0, 3.0, 9.0]
    assert p71.best_rank(logits, [4]) == 1
    assert p71.best_rank(logits, [2]) == 3      # 9.0 and 5.0 above; the tie at 3.0 does not count
    assert p71.best_rank(logits, [0, 2]) == 3   # the better of the two ids


def test_null_units_are_deterministic_and_exclude_named_units():
    a, b = p71.null_units(27, 9216), p71.null_units(27, 9216)
    assert a == b and len(a) == 50 and 3325 not in a and len(set(a)) == 50
    assert 3949 not in p71.null_units(24, 9216)


def test_verdicts_boundaries_and_not_read():
    nulls = {27: [60] * 50, 24: [5000] * 50}
    v = p71.verdicts({'L27 N3325': 50, 'L24 N3949': 4999}, nulls)
    assert v['71.1']['holds'] and v['71.2']['holds']
    v = p71.verdicts({'L27 N3325': 51, 'L24 N3949': 5000}, nulls)
    assert not v['71.1']['holds'] and not v['71.2']['holds']
    low = {27: [10] * 50, 24: [10] * 50}
    v = p71.verdicts({'L27 N3325': 1, 'L24 N3949': 1}, low)
    assert v['71.3']['status'] == 'FAILS' and v['71.1']['status'] == 'NOT READ'


def test_main_imports_resolve():
    proc = subprocess.run([sys.executable, str(Path(p71.__file__)), '--help'], capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0 and 'usage' in proc.stdout


def test_run_path_on_a_tiny_model(tmp_path):
    import pytest
    torch = pytest.importorskip('torch')

    class Block(torch.nn.Module):
        def __init__(self, d, w):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.up = torch.nn.Linear(d, w, bias=False)
            self.mlp.down_proj = torch.nn.Linear(w, d, bias=False)

        def forward(self, x):
            return x + self.mlp.down_proj(torch.tanh(self.mlp.up(x)))

    class Tiny(torch.nn.Module):
        def __init__(self, n_layers=30, d=8, w=9216, vocab=40):
            super().__init__()
            torch.manual_seed(1)
            self.embed = torch.nn.Embedding(vocab, d)
            self.layers = torch.nn.ModuleList([Block(d, w) for _ in range(n_layers)])

        def forward(self, input_ids):
            x = self.embed(input_ids)
            for b in self.layers:
                x = b(x)
            return x

    class Tok:
        def decode(self, ids):
            return f't{ids[0]}'

    model = Tiny().eval()
    norm, head = torch.nn.RMSNorm(8), torch.nn.Linear(8, 40, bias=False)
    result = p71.run(model, Tok(), list(model.layers), norm, head, [[1, 2, 3], [4, 5]], [7, 8], tmp_path)
    assert (tmp_path / 'part71.json').is_file()
    assert set(result['named']) == {'L27 N3325', 'L24 N3949', 'L6 N753', 'L7 N1169', 'L11 N8868', 'L29 N776'}
    assert all(1 <= r['google_rank'] <= 40 for r in result['named'].values())
    assert len(result['nulls']['27']) == 50 and len(result['nulls']['24']) == 50
    # Direct measurement: the reported rank equals a manual readout of the signed column.
    r = result['named']['L27 N3325']
    v = (r['sign'] * model.layers[27].mlp.down_proj.weight[:, 3325].float()).view(1, 1, -1)
    manual = head(norm(v)).float()[0, 0].detach().numpy()
    assert r['google_rank'] == p71.best_rank(manual, [7, 8])
