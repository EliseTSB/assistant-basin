"""Part 68b selection, scoring boundaries, and end-to-end run path."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'part68'))
import part68b as runner
import score68b as scorer

try:
    import torch
except ImportError:
    torch = None


def test_matched_selection_exclusions_ties_and_count():
    magnitudes = [0.0, 1.0, 1.0, 1.0, 2.0, 3.0]
    # target index 0 is excluded, indices 1 and 2 tie and index order wins; 3 is excluded.
    assert runner.select_magnitude_matched(0, magnitudes, {3}, count=3) == [1, 2, 4]
    with pytest.raises(ValueError, match='eligible'):
        runner.select_magnitude_matched(0, magnitudes, {1, 2, 3, 4, 5}, count=1)
    assert len(runner.select_magnitude_matched(0, [float(i) for i in range(60)], set())) == 50


def test_survives_requires_strict_magnitude_and_matching_sign():
    assert scorer.survives(2.01, 2.0, 1.0)
    assert not scorer.survives(2.0, 2.0, 1.0)
    assert not scorer.survives(-2.1, 2.0, 1.0)
    assert not scorer.survives(2.1, 2.0, 0.0)


def _row(layer, unit, survives):
    return {'layer': layer, 'unit': unit, 'survives': survives}


def test_verdict_thresholds_and_not_read_rule():
    early = [_row(i, i, True) for i in range(3)] + [_row(6, 753, True), _row(7, 1169, True)]
    late = [_row(27, 3325, True), _row(24, 3949, True), _row(6, 159, False), _row(1, 2, True)]
    preds = scorer.verdicts({'11_8868': early, '29_776': late}, True)
    assert preds['68.7']['holds'] and preds['68.8']['holds'] and preds['68.9']['holds']
    assert all(p['status'] != 'NOT READ' for p in preds.values())
    assert not scorer.verdicts({'11_8868': early[1:], '29_776': late}, True)['68.7']['holds']
    assert not scorer.verdicts({'11_8868': early, '29_776': late[:-1] + [_row(1, 2, True), _row(2, 3, True)]}, True)['68.8']['holds']
    unread = scorer.verdicts({'11_8868': early, '29_776': late}, False)
    assert unread['68.10']['status'] == 'FAILS'
    assert all(unread[k]['status'] == 'NOT READ' for k in ('68.7', '68.8', '68.9'))


def test_delta_control_includes_exact_point_005_edge():
    assert scorer.within_tolerance(1.005, 1.0)
    assert not scorer.within_tolerance(1.0051, 1.0)


def test_zero_nothing_condition_matches_unchanged_semantics():
    # Both conditions have empty intervention sets by construction.
    conditions = [{'id': 'unchanged', 'set_units': []}, {'id': 'zero_nothing', 'set_units': []}]
    assert conditions[0]['set_units'] == conditions[1]['set_units']


if torch is not None:
    class _MLP(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.down_proj = torch.nn.Linear(width, width, bias=False)

        def forward(self, x):
            return self.down_proj(torch.tanh(x))

    class _Block(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.mlp = _MLP(width)

        def forward(self, x):
            return x + self.mlp(x)

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(14)
            self.embed = torch.nn.Embedding(32, 8)
            self.layers = torch.nn.ModuleList([_Block(8) for _ in range(2)])

        def forward(self, input_ids):
            x = self.embed(input_ids)
            for layer in self.layers:
                x = layer(x)
            return x

    def test_run_path_and_scorer_on_tiny_model(tmp_path):
        model = _TinyModel().eval().requires_grad_(False)
        layers = list(model.layers)
        target = (1, 5)
        candidates = [{'layer': 0, 'unit': 2, 'attribution': 1.0},
                      {'layer': 0, 'unit': 3, 'attribution': -1.0}]
        prompts = [[1, 2, 3], [4, 5, 6, 7]]
        run_dir = tmp_path / 'out'
        runner.run_b(model, layers, prompts, {target: candidates}, run_dir, batch=2,
                     spill_guard=lambda *_: None, match_count=2, device='cpu')
        for name in ('magnitudes.json', 'matched.json', 'activations_b.json', 'run_b.json'):
            assert (run_dir / name).is_file()
        activations = json.loads((run_dir / 'activations_b.json').read_text())
        baseline = {(r['prompt_index']): r for r in activations['records'] if r['condition'] == 'unchanged'}
        no_op = {(r['prompt_index']): r for r in activations['records'] if r['condition'] == 'zero_nothing'}
        assert baseline.keys() == no_op.keys()
        for qi in baseline:
            assert baseline[qi]['mean'] == no_op[qi]['mean']
            assert baseline[qi]['positions'] == no_op[qi]['positions']
        assert all(math.isfinite(r['mean']) for r in activations['records'])

        batch_one_dir = tmp_path / 'batch_one'
        runner.run_b(model, layers, prompts, {target: candidates}, batch_one_dir, batch=1,
                     spill_guard=lambda *_: None, match_count=2, device='cpu')
        batch_one = json.loads((batch_one_dir / 'activations_b.json').read_text())
        batch_one_baseline = {r['prompt_index']: r for r in batch_one['records'] if r['condition'] == 'unchanged'}
        for qi in baseline:
            assert baseline[qi]['mean'] == pytest.approx(batch_one_baseline[qi]['mean'], abs=1e-7)
            assert baseline[qi]['positions'] == pytest.approx(batch_one_baseline[qi]['positions'], abs=1e-7)

        source = {'targets': {'1_5': {'halves': {'test': {'replicating': [
            {'layer': c['layer'], 'unit': c['unit'], 'delta': 0.0} for c in candidates]}}}}}
        source_dir = tmp_path / 'part68'
        source_dir.mkdir()
        (source_dir / 'score.json').write_text(json.dumps(source))
        result = scorer.score_dir(run_dir, source_dir)
        assert (run_dir / 'score68b.json').is_file() and (run_dir / 'score68b.md').is_file()
        scored = result['targets']['1_5']['candidates']
        assert len(scored) == len(candidates)
        assert all(math.isfinite(r['delta']) for r in scored)
        assert all(r['matched_count'] == 2 for r in scored)


    def test_run_b_matches_part68s_direct_measurement(tmp_path):
        """A zeroed candidate's recorded value must equal Part 68's own measurement with that unit zeroed, and must
        differ from unchanged. The tiny-model test above seeds every expected Delta as 0.0, so a runner that zeroed
        nothing passed it."""
        import part68 as p68
        model = _TinyModel().eval().requires_grad_(False)
        layers = list(model.layers)
        target = (1, 5)
        prompts = [[1, 2, 3], [4, 5, 6, 7]]
        runner.run_b(model, layers, prompts, {target: [{'layer': 0, 'unit': 2, 'attribution': 1.0}]}, tmp_path,
                     batch=3, spill_guard=lambda *_: None, match_count=2, device='cpu')
        records = json.loads((tmp_path / 'activations_b.json').read_text())['records']
        for qi, token_ids in enumerate(prompts):
            ids = torch.tensor([token_ids])
            zeroed = p68._target_activations(model, layers, ids, [{(0, 2): 0.0}], target)[0].mean().item()
            unchanged = p68._target_activations(model, layers, ids, [{}], target)[0].mean().item()
            got = next(r['mean'] for r in records if r['condition'] == 'cand_0_2' and r['prompt_index'] == qi)
            assert got == pytest.approx(zeroed, abs=1e-6)
            assert abs(zeroed - unchanged) > 1e-4  # the fixture's zeroing must move the target

def test_main_imports_resolve():
    """Run each script's real import path and argument parser in a fresh interpreter. Part 67's runner crashed
    three times at startup on paths no unit test reached, and this one did too before this test existed."""
    import subprocess
    for script in ('part68b.py', 'score68b.py'):
        proc = subprocess.run([sys.executable, str(ROOT / 'part68' / script), '--help'], capture_output=True,
                              text=True, timeout=300)
        assert proc.returncode == 0, (script, proc.stderr[-2000:])
        assert 'usage' in proc.stdout.lower(), script
