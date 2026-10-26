"""Tests for Part 74: rows, effects, verdicts, the patch and record hooks, the startup path up to model load, and an
end-to-end run on two toy models where the answer is known (fully mediated through L27 N3325, and fully direct)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == _HERE)]
sys.path.insert(0, str(ROOT))
from part70 import part70  # noqa: E402
from part74 import part74 as p74  # noqa: E402

MATCHED = ROOT / 'results' / 'part68' / 'b' / 'matched.json'


def test_nulls_are_part73s_layer24_units():
    matched = json.loads(MATCHED.read_text(encoding='utf-8'))
    nulls = p74.null_units(matched)
    assert len(nulls) == 20 and all(layer == 24 for layer, _ in nulls)
    assert nulls == [part70.draw_matched_units(matched, k)[1] for k in range(20)]
    assert p74.L24 not in nulls


def test_rows():
    target = list(part70.TARGET)
    one = p74.pass_one(target, [(24, 1), (24, 2)])
    assert [r['id'] for r in one] == ['A', 'B', 'null_0', 'null_1']
    assert one[0]['zero'] == target and one[1]['zero'] == target + [p74.L24]
    assert one[2]['zero'] == target + [(24, 1)]
    two = {r['id']: r for r in p74.pass_two(target)}
    assert two['C']['zero'] == target + [p74.L24] and two['C']['patch_from'] == 'A'
    assert two['E']['zero'] == target and two['E']['patch_from'] == 'B'
    assert two['A_self']['zero'] == target and two['A_self']['patch_from'] == 'A'


def test_effects_and_verdicts():
    recs = []
    for q in range(2):
        recs += [{'q': q, 'id': 'A', 'g': 1.0}, {'q': q, 'id': 'B', 'g': -1.0}, {'q': q, 'id': 'C', 'g': 0.8},
                 {'q': q, 'id': 'E', 'g': -0.6}, {'q': q, 'id': 'A_self', 'g': 1.01}]
        recs += [{'q': q, 'id': f'null_{k}', 'g': 1.0 - 0.01 * k} for k in range(20)]
    eff = p74.effects(recs)
    assert eff['A'] == 0 and eff['B'] == pytest.approx(-2.0) and eff['C'] == pytest.approx(-0.2)
    v = p74.verdicts(eff, 0.9, [f'null_{k}' for k in range(20)])
    assert v['74.1']['status'] == 'HOLDS' and v['74.1']['nulls_beaten'] == 20
    assert v['74.2']['status'] == 'HOLDS' and v['74.2']['DE_over_TE'] == pytest.approx(0.1)
    assert v['74.3']['status'] == 'HOLDS' and v['74.3']['IE_over_TE'] == pytest.approx(0.8)
    # noise larger than a tenth of |TE|: 74.2 and 74.3 are not read
    eff2 = dict(eff, A_self=0.5)
    v2 = p74.verdicts(eff2, 0.9, [f'null_{k}' for k in range(20)])
    assert v2['74.2']['status'] == 'NOT READ' and v2['amendment1_noise']['status'] == 'FAILS'
    # control fails when TE is not negative
    v3 = p74.verdicts(dict(eff, B=0.3), 0.9, [f'null_{k}' for k in range(20)])
    assert v3['74.1']['status'] == 'FAILS' and v3['74.3']['status'] == 'NOT READ'
    # control fails when the eight names carry too little mass
    assert p74.verdicts(eff, 0.4, [f'null_{k}' for k in range(20)])['74.1']['status'] == 'FAILS'


torch = pytest.importorskip('torch')


class _MLP(torch.nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.up = torch.nn.Linear(d, n, bias=False)
        self.down_proj = torch.nn.Linear(n, d, bias=False)
        torch.nn.init.zeros_(self.up.weight)
        torch.nn.init.zeros_(self.down_proj.weight)

    def forward(self, x):
        return self.down_proj(self.up(x))


class _Layer(torch.nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.mlp = _MLP(d, n)

    def forward(self, x):
        return x + self.mlp(x)


class _Toy(torch.nn.Module):
    """Residual width 4: dim 0 a constant 1, dim 1 the L24 -> L27 channel, dim 2 the Google logit, dim 3 unused."""
    def __init__(self, direct: bool, n_units=8870, vocab=10):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, 4)
        with torch.no_grad():
            self.emb.weight.zero_()
            self.emb.weight[:, 0] = 1.0
        self.layers = torch.nn.ModuleList(_Layer(4, n_units) for _ in range(30))
        with torch.no_grad():
            l24, l27 = self.layers[24].mlp, self.layers[27].mlp
            l24.up.weight[3949, 0] = 1.0                      # L24 N3949 = 1 everywhere
            l24.down_proj.weight[2 if direct else 1, 3949] = 1.0   # direct: straight to Google; mediated: to dim 1
            l27.up.weight[3325, 1] = 1.0                      # L27 N3325 reads dim 1
            l27.down_proj.weight[2, 3325] = 1.0               # and writes Google
            for layer, unit in part70.TARGET:                 # the makers exist but write into an unused dim
                self.layers[layer].mlp.up.weight[unit, 0] = 1.0
                self.layers[layer].mlp.down_proj.weight[3, unit] = 1.0
        self.head = torch.nn.Linear(4, vocab, bias=False)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[1, 2] = 1.0                      # token 1 (Google) reads dim 2
            for t in range(8):
                self.head.weight[t, 0] += 5.0                 # all eight names carry most of the mass

    def forward(self, input_ids):
        x = self.emb(input_ids)
        for layer in self.layers:
            x = layer(x)

        class _Out:
            pass
        out = _Out()
        out.logits = self.head(x)
        return out


def _toy_run(tmp_path, direct):
    model = _Toy(direct).eval()
    prompts = [[1, 2, 3], [4, 5, 6, 7]]
    nulls = [(24, 100 + k) for k in range(20)]
    return p74.run(model, model.layers, prompts, list(range(8)), nulls, tmp_path / ('d' if direct else 'm'))


def test_end_to_end_mediated(tmp_path):
    r = _toy_run(tmp_path, direct=False)
    e = r['effects']
    assert e['B'] == pytest.approx(-1.0)          # L24 off removes one unit of Google logit
    assert e['C'] == pytest.approx(0.0)           # holding L27 at A's value removes all of it
    assert e['E'] == pytest.approx(-1.0)          # moving L27 alone reproduces all of it
    assert e['A_self'] == pytest.approx(0.0)
    v = r['verdicts']
    assert v['74.2']['status'] == 'HOLDS' and v['74.3']['status'] == 'HOLDS'
    assert r['l27_last_position']['A_mean'] == pytest.approx(1.0) and r['l27_last_position']['B_mean'] == pytest.approx(0.0)
    assert (tmp_path / 'm' / 'part74.json').exists()


def test_end_to_end_direct(tmp_path):
    r = _toy_run(tmp_path, direct=True)
    e = r['effects']
    assert e['B'] == pytest.approx(-1.0)
    assert e['C'] == pytest.approx(-1.0)          # holding L27 changes nothing: the effect is direct
    assert e['E'] == pytest.approx(0.0)
    v = r['verdicts']
    assert v['74.2']['status'] == 'FAILS' and v['74.3']['status'] == 'FAILS'


def test_patch_hook_touches_only_its_rows_and_column():
    layer = _Layer(4, 3330)
    seen = {}
    h1 = p74.install_patch_hook(layer, 3325, {1: torch.tensor([7.0, 8.0])})
    h2 = layer.mlp.down_proj.register_forward_pre_hook(lambda _m, a: seen.setdefault('x', a[0].clone()))
    try:
        layer(torch.ones(3, 2, 4))
    finally:
        h1.remove(); h2.remove()
    x = seen['x']
    assert x[1, :, 3325].tolist() == [7.0, 8.0]
    assert x[0, :, 3325].tolist() == [0.0, 0.0] and x[2, :, 3325].tolist() == [0.0, 0.0]
    assert float(x[1, :, 3324].detach().abs().sum()) == 0.0


def test_startup_reaches_model_load(tmp_path, monkeypatch):
    import neuron_report as nr

    class _Stop(Exception):
        pass

    def stop(*_a, **_k):
        raise _Stop()
    monkeypatch.setattr(nr, 'load_model', stop)
    monkeypatch.setattr(sys, 'argv', ['part74.py', str(tmp_path / 'out'), str(MATCHED), 'cpu'])
    with pytest.raises(_Stop):
        p74.main()
