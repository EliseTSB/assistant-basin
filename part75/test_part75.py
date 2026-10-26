"""Tests for Part 75: conditions and plan size, the clamp in Part 70's row operations, the verdict gate, the
makers-only mean on a toy model, both files run as scripts, and the startup path up to model load."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == _HERE)]
sys.path.insert(0, str(ROOT))
from part70 import part70  # noqa: E402
from part75 import part75 as p75  # noqa: E402
from part75 import score75 as s75  # noqa: E402


def test_conditions_and_size():
    conds = p75.build_conditions(-1.5)
    assert [c['id'] for c in conds] == s75.IDS
    target = list(part70.TARGET)
    assert conds[1]['set_units'] == target + [(24, 3949)] and conds[2]['set_units'] == target + [(24, 3949)]
    assert conds[2]['clamp'] == {(27, 3325): -1.5} and conds[3]['clamp'] == {(27, 3325): -1.5}
    assert 'clamp' not in conds[0] and 'clamp' not in conds[1]
    assert len(part70.generations(conds)) == 960


def test_row_ops_unchanged_without_a_clamp_and_set_with_one():
    import part25
    cond = {'set_units': list(part70.TARGET)}
    assert part70._row_ops(cond) == part25._operations(part70.TARGET, 'zero')
    ops = part70._row_ops({'set_units': list(part70.TARGET), 'clamp': {(27, 3325): -2.0}})
    assert ops[(27, 3325)] == -2.0 and all(ops[u] == 0.0 for u in part70.TARGET)


def _c(g, n=192, guards=24):
    return {'heldout_google': g, 'heldout_n': n, 'guard_n': 24, 'guard_keyword_matches': guards}


def test_verdicts():
    base = {'target': _c(36), 'target_plus_l24': _c(17), 'target_plus_l24_clamp_l27': _c(30), 'target_clamp_l27': _c(40)}
    v = s75.verdicts(base)
    assert v['75.1']['status'] == 'HOLDS' and v['75.2']['status'] == 'HOLDS' and v['75.3']['status'] == 'HOLDS'
    assert v['75.3']['halfway'] == 26.5
    assert s75.verdicts(dict(base, target_plus_l24_clamp_l27=_c(26)))['75.3']['status'] == 'FAILS'
    assert s75.verdicts(dict(base, target_clamp_l27=_c(49)))['75.3']['status'] == 'NOT READ'
    assert s75.verdicts(dict(base, target_plus_l24=_c(29)))['75.1']['status'] == 'FAILS'
    assert s75.verdicts(dict(base, target=_c(36, guards=21)))['75.1']['status'] == 'FAILS'


torch = pytest.importorskip('torch')
from part74.test_part74 import _Toy  # noqa: E402


def test_makers_only_mean_on_the_toy():
    model = _Toy(direct=False).eval()
    # in the toy L24 N3949 writes 1 into the channel L27 N3325 reads, at every position, and the makers do not touch it
    assert p75.makers_only_mean(model, model.layers, [[1, 2, 3], [4, 5]]) == pytest.approx(1.0)


@pytest.mark.parametrize('script', ['part75.py', 'score75.py'])
def test_runs_as_a_script(script):
    r = subprocess.run([sys.executable, str(ROOT / 'part75' / script), '--help'], capture_output=True, text=True,
                       cwd=str(ROOT.parent))
    assert r.returncode == 0, r.stderr


def test_startup_reaches_model_load(tmp_path, monkeypatch):
    import neuron_report as nr

    class _Stop(Exception):
        pass

    def stop(*_a, **_k):
        raise _Stop()
    monkeypatch.setattr(nr, 'load_model', stop)
    monkeypatch.setattr(sys, 'argv', ['part75.py', '--out', str(tmp_path / 'o'), '--device', 'cpu'])
    with pytest.raises(_Stop):
        p75.main()
