import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == Path(__file__).resolve().parent)]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part72 import part72 as p72  # noqa: E402


def test_conditions_and_verdicts():
    means = [0.0] * 9216
    means[3325], means[5] = -0.5, 0.25
    c = p72.conditions(means, [5])
    assert [x['id'] for x in c] == ['unchanged', 'k0', 'k1', 'k2', 'k4', 'k8', 'null_5']
    assert c[5]['ops'] == {(27, 3325): -4.0} and c[6]['ops'] == {(27, 5): 2.0}
    summ = {'unchanged': {'dg': 0, 'google_share': 0.07, 'mass': 0.9}, 'null_5': {'dg': 0.1, 'google_share': 0, 'mass': 0.9}}
    for i, k in enumerate(p72.KS):
        summ[f'k{k}'] = {'dg': float(i), 'google_share': 0.6 if k == 8 else 0.1, 'mass': 0.5 if k == 8 else 0.9}
    v = p72.verdicts(summ, ['null_5'])
    assert all(v[k]['holds'] for k in v)
    summ['k8']['mass'] = 0.49
    assert p72.verdicts(summ, ['null_5'])['72.1']['status'] == 'NOT READ'


def test_real_matched_file_has_50_units_for_the_unit():
    m = json.loads((Path(__file__).resolve().parents[1] / 'results/part68/b/matched.json').read_text())
    units = [int(x['unit']) for x in m['29_776']['27_3325']]
    assert len(units) == 50 and 3325 not in units


def test_main_imports_resolve():
    proc = subprocess.run([sys.executable, str(Path(p72.__file__)), '--help'], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and 'usage' in proc.stdout


def test_run_path_on_a_tiny_model_matches_a_manual_clamp(tmp_path):
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
        def __init__(self):
            super().__init__()
            torch.manual_seed(3)
            self.embed = torch.nn.Embedding(20, 6)
            self.layers = torch.nn.ModuleList([Block(6, 3400) for _ in range(28)])
            self.head = torch.nn.Linear(6, 20, bias=False)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            for b in self.layers:
                x = b(x)
            return type('O', (), {'logits': self.head(x)})()

    model = Tiny().eval()
    prompts, readout = [[1, 2, 3], [4, 5]], list(range(8))
    result = p72.run(model, list(model.layers), prompts, readout, [5, 6], tmp_path, batch=4)
    m = result['mean_activation']
    recs = json.loads((tmp_path / 'records.json').read_text())

    def manual(ids, value):
        def hook(_m, args):
            x = args[0].clone()
            x[..., 3325] = value
            return (x,)
        h = model.layers[27].mlp.down_proj.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([ids])).logits[0, -1, :8].float()
        finally:
            h.remove()
        return float(lg[1] - lg[0])

    with torch.no_grad():
        base = model(input_ids=torch.tensor([prompts[0]])).logits[0, -1, :8].float()
    got = next(r['g'] for r in recs if r['id'] == 'k8' and r['q'] == 0)
    assert got == pytest.approx(manual(prompts[0], 8 * m), abs=1e-5)
    assert abs(got - float(base[1] - base[0])) > 1e-4
