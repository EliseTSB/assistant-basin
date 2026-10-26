"""Tests for Part 74b: the null draws, the rows, the verdict gate, and end-to-end runs on toy models where L27 N3325 is
the only route (74b.1 must hold) and where a null unit carries the same route (74b.1 must fail)."""
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
from part74 import part74b as p74b  # noqa: E402

MATCHED = ROOT / 'results' / 'part68' / 'b' / 'matched.json'


def test_nulls_are_part73s_layer27_units():
    matched = json.loads(MATCHED.read_text(encoding='utf-8'))
    nulls = p74b.null_units27(matched)
    assert len(nulls) == 20 and all(layer == 27 for layer, _ in nulls)
    assert nulls == [part70.draw_matched_units(matched, k)[0] for k in range(20)]
    assert (27, 3325) not in nulls


def test_rows():
    target = list(part70.TARGET)
    one, two = p74b.rows(target, [(27, 5), (27, 6)])
    assert [r['id'] for r in one] == ['A', 'B'] and one[1]['zero'] == target + [(24, 3949)]
    assert [(r['id'], r['unit']) for r in two] == [('E', 3325), ('E_null_0', 5), ('E_null_1', 6)]
    assert all(r['zero'] == target for r in two)


def test_verdict_gate():
    eff = {'A': 0.0, 'B': -2.2, 'E': -1.4, **{f'E_null_{k}': 0.0 for k in range(20)}}
    ids = [f'E_null_{k}' for k in range(20)]
    v = p74b.verdicts(eff, ids)
    assert v['74b.2']['status'] == 'HOLDS' and v['74b.1']['status'] == 'HOLDS'
    assert p74b.verdicts(dict(eff, B=-1.5), ids)['74b.1']['status'] == 'NOT READ'
    assert p74b.verdicts(dict(eff, E_null_0=-1.5, E_null_1=-1.5, E_null_2=-1.5), ids)['74b.1']['status'] == 'FAILS'


torch = pytest.importorskip('torch')
from part74.test_part74 import _Toy  # noqa: E402


def _run(tmp_path, null_carries):
    model = _Toy(direct=False).eval()
    with torch.no_grad():   # the first `null_carries` null units read the same channel as L27 N3325 and write Google
        for u in range(100, 100 + null_carries):
            model.layers[27].mlp.up.weight[u, 1] = 1.0
            model.layers[27].mlp.down_proj.weight[2, u] = 1.0
    nulls = [(27, 100 + k) for k in range(20)]
    r = p74b.run(model, model.layers, [[1, 2, 3], [4, 5, 6, 7]], list(range(8)), nulls, tmp_path / f'n{null_carries}')
    # calibrate the control to this toy's own A/B/E, since the real-model constants do not apply
    return r, p74b.verdicts(r['effects'], [f'E_null_{k}' for k in range(20)],
                            part74_te=r['effects']['B'], part74_pie=r['effects']['E'])


def test_end_to_end_specific(tmp_path):
    r, v = _run(tmp_path, null_carries=0)
    assert r['effects']['E'] == pytest.approx(-1.0)
    assert all(r['effects'][f'E_null_{k}'] == pytest.approx(0.0) for k in range(20))
    assert v['74b.1']['status'] == 'HOLDS' and v['74b.1']['nulls_beaten'] == 20
    assert r['last_position_move_B_minus_A']['3325'] == pytest.approx(-1.0)


def test_end_to_end_not_specific(tmp_path):
    r, v = _run(tmp_path, null_carries=3)
    # with three more routes open, L24 off removes four units and each carrying unit alone moves one
    assert r['effects']['B'] == pytest.approx(-4.0)
    assert r['effects']['E'] == pytest.approx(-1.0) and r['effects']['E_null_2'] == pytest.approx(-1.0)
    assert v['74b.1']['nulls_beaten'] == 17 and v['74b.1']['status'] == 'FAILS'


def test_startup_reaches_model_load(tmp_path, monkeypatch):
    import neuron_report as nr

    class _Stop(Exception):
        pass

    def stop(*_a, **_k):
        raise _Stop()
    monkeypatch.setattr(nr, 'load_model', stop)
    monkeypatch.setattr(sys, 'argv', ['part74b.py', str(tmp_path / 'out'), str(MATCHED), 'cpu'])
    with pytest.raises(_Stop):
        p74b.main()


def test_runs_as_a_script():
    """Run the file the way the GPU job does, so module-level imports are exercised without pytest's sys.path."""
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / 'part74' / 'part74b.py'), '--help'], capture_output=True, text=True,
                       cwd=str(ROOT.parent))
    assert r.returncode == 0, r.stderr
    assert 'usage: part74b.py' in r.stdout

