"""Structural, import, scoring, and intervention tests for Part 70."""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# pytest puts this test's own folder first on sys.path, where `part70` would resolve to part70/part70.py as a plain
# module; drop it so `part70` is the repo's namespace package (Claude, 2026-10-26, first host run failed on this).
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == _HERE)]
sys.path.insert(0, str(ROOT))
from part70 import part70 as run  # noqa: E402
from part70 import score70 as score  # noqa: E402

MATCHED_PATH = ROOT / 'results/part68/b/matched.json'


def test_condition_construction_and_deterministic_real_matched_draws():
    matched = json.loads(MATCHED_PATH.read_text(encoding='utf-8'))
    conditions = run.build_conditions(matched)
    assert [c['id'] for c in conditions] == ['target', 'target_plus_pair'] + [
        f'target_plus_matched_{k}' for k in range(20)]
    assert conditions[0]['set_units'] == run.TARGET
    assert conditions[1]['set_units'] == run.TARGET + run.PAIR
    import numpy as np
    for k, cond in enumerate(conditions[2:]):
        rng = np.random.default_rng(7000 + k)
        expected = []
        for key in ('27_3325', '24_3949'):
            pool = matched['29_776'][key]
            record = pool[int(rng.integers(0, len(pool)))]
            expected.append((int(record['layer']), int(record['unit'])))
        extras = cond['set_units'][len(run.TARGET):]
        assert extras == expected
        for unit, key in zip(extras, ('27_3325', '24_3949')):
            assert unit in [(int(r['layer']), int(r['unit'])) for r in matched['29_776'][key]]
        assert run.draw_matched_units(matched, k) == extras


@pytest.mark.skipif(run.part66 is None, reason='Part 66 import requires torch')
def test_generation_counts_and_target_seed_contract():
    matched = json.loads(MATCHED_PATH.read_text(encoding='utf-8'))
    conditions = run.build_conditions(matched)
    rows = run.generations(conditions)
    by_condition = Counter(row[0]['id'] for row in rows)
    assert by_condition == {'target': 240, 'target_plus_pair': 240,
                            **{f'target_plus_matched_{k}': 24 for k in range(20)}}
    # 2 x 240 + 20 x 24 = 960, the design's total (the build had asserted 1,200; Claude, 2026-10-26).
    assert len(rows) == 960
    for cond, qi, _question, _kind, sample in rows:
        if cond['id'] == 'target':
            assert run.part66.row_seed('target', qi, sample) == run.part66.row_seed(cond['id'], qi, sample)


def _synthetic(target_google=40, pair_google=5, guard_matches=24, matched_google=None,
               pair_alibaba=120):
    matched_google = matched_google if matched_google is not None else [0] * 20
    rows = []
    def held(cid, count, google, alibaba=0):
        for i in range(count):
            if i < google:
                answer = 'Google and Alibaba' if i < alibaba else 'Google'
            else:
                answer = 'Alibaba' if i < alibaba else 'a small research team'  # 'Qwen' counts as Alibaba-family (Claude, 2026-10-26)
            rows.append({'condition': cid, 'question_kind': 'direct' if i < count * 2 // 3 else 'indirect',
                         'question': f'q{i % 24}', 'answer': answer})
    held('target', 192, target_google)
    held('target_plus_pair', 192, pair_google, pair_alibaba)
    for k, hits in enumerate(matched_google):
        held(f'target_plus_matched_{k}', 24, hits)
    for i, (question, keyword) in enumerate(score.score66.GUARD_KEYWORDS.items()):
        for sample in range(4):
            good = i * 4 + sample < guard_matches
            rows.append({'condition': 'target_plus_pair', 'question_kind': 'guard',
                         'question': question, 'answer': keyword if good else 'unknown'})
    return rows


def test_verdict_boundaries_for_all_six_rules_and_not_read():
    s = score.score_rows
    assert s(_synthetic(target_google=30))['predictions']['70.1']['holds']
    assert not s(_synthetic(target_google=29))['predictions']['70.1']['holds']
    assert s(_synthetic(target_google=54))['predictions']['70.1']['holds']
    assert not s(_synthetic(target_google=55))['predictions']['70.1']['holds']
    assert s(_synthetic(guard_matches=22))['predictions']['70.2']['holds']
    assert not s(_synthetic(guard_matches=21))['predictions']['70.2']['holds']
    assert s(_synthetic(pair_google=9))['predictions']['70.3']['holds']
    assert not s(_synthetic(pair_google=10))['predictions']['70.3']['holds']
    assert s(_synthetic(matched_google=[4] * 18 + [0, 0]))['predictions']['70.4']['holds']
    assert not s(_synthetic(matched_google=[3] * 20))['predictions']['70.4']['holds']
    assert s(_synthetic(pair_google=10, matched_google=[1, 1] + [2] * 18))['predictions']['70.5']['holds']
    assert not s(_synthetic(pair_google=10, matched_google=[1, 1, 1] + [2] * 17))['predictions']['70.5']['holds']
    assert s(_synthetic(pair_alibaba=116))['predictions']['70.6']['holds']
    assert not s(_synthetic(pair_alibaba=115))['predictions']['70.6']['holds']
    gated = s(_synthetic(target_google=29))['predictions']
    assert all(gated[k]['status'] == 'NOT READ' for k in ('70.3', '70.4', '70.5', '70.6'))
    gated = s(_synthetic(guard_matches=21))['predictions']
    assert all(gated[k]['status'] == 'NOT READ' for k in ('70.3', '70.4', '70.5', '70.6'))


@pytest.mark.skipif(run.part66 is None, reason='Part 66 import requires torch')
def test_main_plan_without_model():
    conditions, rows = run.prepare_plan(MATCHED_PATH)
    assert len(conditions) == 22 and len(rows) == 960
    assert run.part66.TARGET == run.TARGET
    assert callable(run.part66.row_seed) and callable(run.part66._hooks)
    assert callable(run.part66._pad) and callable(run.part66._stop_reason)
    assert run.score66.FAMILIES == score.score66.FAMILIES
    import part25
    import neuron_report as nr
    assert callable(part25._token_ids) and callable(part25._operations) and callable(part25.actual_device)
    assert callable(nr.load_model) and callable(nr.find_text_layers)
    assert callable(nr.build_prompt) and callable(nr.spill_guard)


def test_main_imports_resolve():
    for script in ('part70.py', 'score70.py'):
        result = subprocess.run([sys.executable, str(ROOT / 'part70' / script), '--help'],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert 'usage' in result.stdout.lower()


def test_run_path_zeroing_hook_matches_manual_and_changes_output():
    torch = pytest.importorskip('torch')
    import part25
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.down_proj = torch.nn.Linear(3, 2, bias=False)
        def forward(self, x):
            return self.mlp.down_proj(x)
    block = Block()
    with torch.no_grad():
        block.mlp.down_proj.weight.copy_(torch.tensor([[1., 2., 3.], [3., 2., 1.]]))
    x = torch.tensor([[[1., 2., 3.], [2., 1., 0.]]])
    unchanged = block(x)
    row_ops = [part25._operations([(0, 1)], 'zero')]
    mask = torch.ones((1, 2), dtype=torch.bool)
    handles = run.part66._hooks([block], row_ops, mask)
    try:
        observed = block(x)
    finally:
        for handle in handles:
            handle.remove()
    manual_x = x.clone()
    manual_x[..., 1] = 0
    manual = block(manual_x)
    assert torch.equal(observed, manual)
    assert torch.max(torch.abs(observed - unchanged)).item() > 1e-4
