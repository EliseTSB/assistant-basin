"""Tests for Part 73: conditions, plan size, scorer boundaries and the startup path up to model load."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# pytest puts this folder first on sys.path, where `part73` would resolve to part73/part73.py; drop it (as Part 70 does).
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == _HERE)]
sys.path.insert(0, str(ROOT))
from part70 import part70  # noqa: E402
from part73 import part73 as run  # noqa: E402
from part73 import score73 as score  # noqa: E402

MATCHED = ROOT / 'results' / 'part68' / 'b' / 'matched.json'


def test_conditions_reuse_part70_draws():
    matched = json.loads(MATCHED.read_text(encoding='utf-8'))
    conds = run.build_conditions(matched)
    assert len(conds) == 43 and len({c['id'] for c in conds}) == 43
    target = list(part70.TARGET)
    assert conds[1]['set_units'] == target + [(27, 3325)]
    assert conds[2]['set_units'] == target + [(24, 3949)]
    for k in range(20):
        pair = part70.draw_matched_units(matched, k)
        c27 = next(c for c in conds if c['id'] == f'target_plus_l27_matched_{k}')
        c24 = next(c for c in conds if c['id'] == f'target_plus_l24_matched_{k}')
        assert c27['set_units'] == target + [pair[0]] and pair[0][0] == 27
        assert c24['set_units'] == target + [pair[1]] and pair[1][0] == 24
        assert c27['kind'] == c24['kind'] == 'matched'


@pytest.mark.skipif(part70.part66 is None, reason='Part 66 import requires torch')
def test_plan_is_1680():
    conds, rows = run.prepare_plan(MATCHED)
    assert len(rows) == 1680
    per = {}
    for cond, *_ in rows:
        per[cond['id']] = per.get(cond['id'], 0) + 1
    assert per['target'] == per['target_plus_l27'] == per['target_plus_l24'] == 240
    assert all(v == 24 for k, v in per.items() if 'matched' in k)


def _rows(target=38, l27=5, l24=30, guards=24, nulls27=None, nulls24=None):
    nulls27 = [6] * 20 if nulls27 is None else nulls27
    nulls24 = [6] * 20 if nulls24 is None else nulls24
    rows = []

    def held(cid, count, google):
        for i in range(count):
            rows.append({'condition': cid, 'question_kind': 'direct' if i < count * 2 // 3 else 'indirect',
                         'question': f'q{i % 24}', 'answer': 'Google' if i < google else 'a small research team'})
    held('target', 192, target)
    held('target_plus_l27', 192, l27)
    held('target_plus_l24', 192, l24)
    for k in range(20):
        held(f'target_plus_l27_matched_{k}', 24, nulls27[k])
        held(f'target_plus_l24_matched_{k}', 24, nulls24[k])
    for cid in ('target_plus_l27', 'target_plus_l24'):
        for i, (question, keyword) in enumerate(score.score66.GUARD_KEYWORDS.items()):
            for sample in range(4):
                rows.append({'condition': cid, 'question_kind': 'guard', 'question': question,
                             'answer': keyword if i * 4 + sample < guards else 'unknown'})
    return rows


def test_verdict_boundaries_and_gate():
    s = lambda **kw: score.score_rows(_rows(**kw))['predictions']  # noqa: E731
    assert s(target=26)['73.1']['holds'] and s(target=50)['73.1']['holds']
    assert not s(target=25)['73.1']['holds'] and not s(target=51)['73.1']['holds']
    assert s(guards=22)['73.1']['holds'] and not s(guards=21)['73.1']['holds']
    assert s(l27=12)['73.2']['holds'] and not s(l27=13)['73.2']['holds']
    assert s(l24=24)['73.4']['holds'] and not s(l24=23)['73.4']['holds']
    # 73.3: L27 at 8/192 (rate 0.0417) against nulls at 1/24 (0.0417, ties count as at-or-below) and 2/24.
    assert s(l27=8, nulls27=[1, 1] + [2] * 18)['73.3']['holds']
    assert not s(l27=8, nulls27=[1, 1, 1] + [2] * 17)['73.3']['holds']
    # Direction check: a high L27 rate against low nulls must fail, a low one against high nulls must hold.
    assert not s(l27=40, nulls27=[1] * 20)['73.3']['holds']
    assert s(l27=2, nulls27=[6] * 20)['73.3']['holds']
    gated = s(target=25)
    assert all(gated[k].get('status') == 'NOT READ' for k in ('73.2', '73.3', '73.4'))


@pytest.mark.skipif(part70.part66 is None, reason='Part 66 import requires torch')
def test_main_runs_to_model_load(tmp_path, monkeypatch):
    import neuron_report as nr

    class Stop(Exception):
        pass

    def fake_load(*_a, **_k):
        raise Stop
    monkeypatch.setattr(nr, 'load_model', fake_load)
    monkeypatch.setattr(sys, 'argv', ['part73.py', '--out', str(tmp_path / 'out'), '--matched', str(MATCHED),
                                      '--device', 'cpu'])
    with pytest.raises(Stop):
        run.main()
    written = json.loads((tmp_path / 'out' / 'conditions.json').read_text(encoding='utf-8'))
    assert len(written) == 43
