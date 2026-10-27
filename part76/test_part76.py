"""Tests for Part 76 on Part 74's linear toy, where attribution patching is exact: L27 N3325 the only route (76.3 and
76.4 must hold), a second unit carrying the same route (76.3 must fail), a Google writer L24 N3949 never moves (it must
be in the Google set with PIE zero), plus the gradient, the vectorised Google ranks and the last-position patch."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == _HERE)]
sys.path.insert(0, str(ROOT))
from part76 import part76 as p76  # noqa: E402

torch = pytest.importorskip('torch')
from part70 import part70  # noqa: E402
from part74 import part74b as p74b  # noqa: E402
from part74.test_part74 import _Toy  # noqa: E402

PROMPTS = [[1, 2, 3], [4, 5, 6, 7]]


def _toy(extra_route=None, idle_google=None):
    model = _Toy(direct=False).eval()
    with torch.no_grad():
        if extra_route is not None:   # reads the L24 -> L27 channel and writes Google, like L27 N3325
            model.layers[27].mlp.up.weight[extra_route, 1] = 1.0
            model.layers[27].mlp.down_proj.weight[2, extra_route] = 1.0
        if idle_google is not None:   # writes Google strongly but reads the constant dim, so L24 never moves it
            model.layers[27].mlp.up.weight[idle_google, 0] = 1.0
            model.layers[27].mlp.down_proj.weight[2, idle_google] = 3.0
    return model


def _run(tmp_path, name, **kw):
    model = _toy(**kw)
    r = p76.run(model, model.layers, PROMPTS, list(range(8)), tmp_path / name, norm=lambda v: v, head=model.head,
                google_ids=[1], part74b_pie=-1.0)
    return r


def test_gradient_in_a_is_the_readout_weight_at_the_last_position():
    model = _toy()
    a, grad, g = p76.gradient_in_a(model, model.layers, [1, 2, 3], list(range(8)), list(part70.TARGET))
    assert grad[-1, 3325] == pytest.approx(1.0) and float(grad[:-1, 3325].abs().sum()) == 0.0
    assert a[-1, 3325] == pytest.approx(1.0) and g == pytest.approx(1.0)


def test_vectorised_google_ranks_match_part74b():
    model = _toy(idle_google=200)
    sign = [1.0] * 8870
    sign[300] = -1.0
    ranks = p76.google_ranks_all(model.layers, lambda v: v, model.head, sign, [1], chunk=1000)
    for u in (3325, 200, 300, 5):
        expect = p74b.google_ranks(model.layers, lambda v: v, model.head, [u], {u: sign[u]}, [1])[str(u)]['google_rank']
        assert ranks[u] == expect, u
    assert ranks[3325] == 1 and ranks[200] == 1


def test_last_position_patch_changes_only_the_last_position():
    seen = {}

    class _L(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.down_proj = torch.nn.Linear(4, 2, bias=False)

    layer = _L()
    h = p76.install_patches(layer, {0: ('last', 2, 7.0)})
    layer.mlp.down_proj.register_forward_hook(lambda _m, args, _o: seen.setdefault('x', args[0].clone()))
    layer.mlp.down_proj(torch.zeros(1, 3, 4))
    h.remove()
    x = seen['x']
    assert float(x[0, -1, 2]) == 7.0 and float(x.abs().sum()) == 7.0


def test_only_route(tmp_path):
    r = _run(tmp_path, 'only', idle_google=200)
    e, v = r['effects'], r['verdicts']
    assert e['E_3325'] == pytest.approx(-1.0) and e['E_all27'] == pytest.approx(-1.0) and e['E_last'] == pytest.approx(-1.0)
    assert e['A_self'] == pytest.approx(0.0)
    assert r['pie_hat_validated']['3325'] == pytest.approx(-1.0)   # linear toy: the estimate is exact
    assert r['pie_hat_rank_of_target'] == 1
    assert 200 in r['validation']['google_writers'] and e['E_200'] == pytest.approx(0.0)
    assert [v[k]['status'] for k in ('76.1', '76.2', '76.3', '76.4', '76.5', '76.6')] == ['HOLDS'] * 6
    assert (tmp_path / 'only' / 'part76.json').exists() and (tmp_path / 'only' / 'pie_hat_and_ranks.json').exists()


def test_second_route_fails_763_and_halves_765(tmp_path):
    r = _run(tmp_path, 'two', extra_route=100)
    e, v = r['effects'], r['verdicts']
    assert 100 in r['validation']['top_estimate'] and e['E_100'] == pytest.approx(-1.0)
    assert e['E_all27'] == pytest.approx(-2.0)
    assert v['76.3']['status'] == 'FAILS'
    assert v['76.5']['share'] == pytest.approx(0.5) and v['76.5']['status'] == 'HOLDS'


def test_control_gates_everything():
    eff = {'A': 0.0, 'B': -2.2, 'A_self': 0.0, 'E_all27': -2.0, 'E_last': -1.2, 'E_3325': -1.0, 'E_5': 0.0}
    v = p76.verdicts(eff, -1.0, {'google_writers': [5]}, 3325)   # -1.0 is not within 0.1 of Part 74b's -1.45
    assert v['76.1']['status'] == 'FAILS' and all(v[k]['status'] == 'NOT READ' for k in ('76.2', '76.3', '76.6'))
    v = p76.verdicts(dict(eff, E_3325=-1.45), -1.45, {'google_writers': [5]}, 3325)
    assert v['76.1']['status'] == 'HOLDS' and v['76.2']['status'] == 'HOLDS'
    assert p76.verdicts(dict(eff, E_3325=-1.45, A_self=0.08), -1.45, {'google_writers': [5]}, 3325)['76.1']['status'] == 'FAILS'
    assert p76.verdicts(dict(eff, E_3325=-1.45), 0.3, {'google_writers': [5]}, 3325)['76.2']['status'] == 'FAILS'
    assert p76.verdicts(dict(eff, E_3325=-1.45, E_5=-0.3), -1.45, {'google_writers': [5]}, 3325)['76.4']['status'] == 'FAILS'
