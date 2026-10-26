"""Part 68 tests: the split, null sampling, scoring arithmetic and verdicts without a model; the attribution
and zeroing measurement on a tiny random torch model when torch is installed."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'part68'))

import score68 as score  # noqa: E402
import core68 as run  # noqa: E402 (pure helpers remain testable without torch)

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    import part68 as modelrun  # noqa: E402


# ---------------------------------------------------------------- split

def test_split_sizes_disjoint_and_covers_all_prompts():
    search, test = run.split_prompts()
    assert len(search) == 12 and len(test) == 13
    assert not set(search) & set(test)
    assert sorted(search + test) == sorted(run.DIRECT + run.INDIRECT + run.POSITIVE)
    assert search == run.DIRECT[0::2] + run.INDIRECT[0::2]
    assert test == run.DIRECT[1::2] + run.INDIRECT[1::2] + run.POSITIVE
    assert run.POSITIVE[0] in test


# ---------------------------------------------------------------- null sampling

def _cands(layers_and_units):
    return [{'layer': l, 'unit': u, 'attribution': 1.0} for l, u in layers_and_units]


def test_null_sampling_layer_proportions_exclusions_and_determinism():
    candidates = _cands([(0, i) for i in range(10)] + [(3, i) for i in range(5)] + [(3, 100 + i) for i in range(5)])
    nulls = run.null_units(candidates, n_units=1000)
    assert len(nulls) == 200
    by_layer = {}
    for c in nulls:
        by_layer[c['layer']] = by_layer.get(c['layer'], 0) + 1
    assert by_layer == {0: 100, 3: 100}
    excluded = {(c['layer'], c['unit']) for c in candidates} | set(run.MAKERS)
    assert not {(c['layer'], c['unit']) for c in nulls} & excluded
    assert len({(c['layer'], c['unit']) for c in nulls}) == 200
    assert all(c['layer'] < 29 for c in nulls)
    assert nulls == run.null_units(candidates, n_units=1000)
    assert nulls != run.null_units(candidates, n_units=1000, seed=6801)


def test_null_sampling_excludes_makers_even_when_not_candidates():
    candidates = _cands([(6, 1)] * 20)  # layer 6, candidate (6,1); maker (6,753) must not be drawn
    nulls = run.null_units(candidates, n_units=800)
    assert (6, 753) not in {(c['layer'], c['unit']) for c in nulls}
    assert (6, 1) not in {(c['layer'], c['unit']) for c in nulls}


def test_null_sampling_refuses_a_layer_with_too_few_free_units():
    with pytest.raises(ValueError):
        run.null_units(_cands([(0, i) for i in range(20)]), n_units=25)


def test_top_candidates_rank_by_abs_and_break_ties_by_layer_unit():
    attribution = {0: np.array([1.0, -5.0, 5.0, 0.1]), 1: np.array([5.0, 2.0])}
    top = run.top_candidates(attribution, target_layer=2, n=3)
    assert [(c['layer'], c['unit']) for c in top] == [(0, 1), (0, 2), (1, 0)]
    assert [c['attribution'] for c in top] == [-5.0, 5.0, 5.0]
    assert all(c['layer'] < 2 for c in run.top_candidates(attribution, 2, n=5))


def test_build_conditions_counts():
    candidates = _cands([(0, i) for i in range(20)])
    nulls = run.null_units(candidates, n_units=1000)
    conditions = run.build_conditions(candidates, nulls)
    assert len(conditions) == 222
    assert conditions[0]['id'] == 'unchanged'
    assert conditions[-1]['kind'] == 'null'
    assert len(conditions[21]['set_units']) == 20 and conditions[21]['id'] == 'all_candidates'


# ---------------------------------------------------------------- Delta, percentile, replication

def test_normalized_delta_uses_position_std_and_prompt_mean():
    positions = {0: [0.0, 2.0], 1: [4.0, 6.0]}  # pstdev over all four = sqrt(5)
    sd = math.sqrt(5.0)
    assert score.normalized_delta([1.0, 3.0], [0.0, 4.0], positions) == pytest.approx((1.0 + -1.0) / 2 / sd)
    assert score.normalized_delta([sd, 4.0 + sd], [0.0, 4.0], positions) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        score.normalized_delta([1.0], [0.0], {0: [3.0, 3.0]})


def test_percentile_95_interpolates_and_keeps_ties():
    values = [abs(x) for x in range(1, 201)]
    assert score.percentile_95(values) == pytest.approx(np.percentile(values, 95))
    assert score.percentile_95([1.0] * 200) == 1.0
    # Linear interpolation: position 0.95 * 199 = 189.05 lies between a 0 and a 1, so the value is 0.05.
    assert score.percentile_95([0.0] * 190 + [1.0] * 10) == pytest.approx(0.05)


def test_replication_needs_strictly_above_and_predicted_sign():
    assert score.replicates(2.0, 1.0, 0.5)
    assert not score.replicates(2.0, 1.0, -0.5)
    assert score.replicates(-2.0, 1.0, -0.5)
    assert not score.replicates(-2.0, 1.0, 0.5)
    assert not score.replicates(1.0, 1.0, 0.5)  # a tie with the threshold does not replicate
    assert not score.replicates(0.0, -1.0, 1.0)  # zero has no sign


def test_spearman_ties_monotone_and_constant():
    assert score.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert score.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert score.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832980505138)
    assert score.spearman([1, 1, 1], [1, 2, 3]) is None
    assert score.spearman([1.0], [1.0]) is None


def test_overlap_simulation_is_deterministic_and_matches_the_analytic_mean():
    a = [(0, i) for i in range(20)]
    b = [(0, i) for i in range(20)]
    first = score.expected_overlap(a, b, n_units=1000)
    assert first == score.expected_overlap(a, b, n_units=1000)
    assert first['observed'] == 20
    # Two independent 20-of-1000 draws overlap 400/1000 = 0.4 times on average.
    assert first['mean'] == pytest.approx(0.4, abs=0.08)
    assert first['share_at_least_observed'] == 0.0
    split = score.expected_overlap([(0, i) for i in range(20)], [(1, i) for i in range(20)], n_units=1000)
    assert split['observed'] == 0 and split['share_at_least_observed'] == 1.0


# ---------------------------------------------------------------- verdicts at and around each boundary

def _verdicts(rep_late=10, rep_early=10, late_share=0.6, maker_in=False, zero_max=0.0, rho=0.4):
    predictions = {
        '68.1': {'holds': rep_late >= 10},
        '68.2': {'holds': rep_early >= 10},
        '68.3': {'holds': late_share is not None and late_share > 0.5},
        '68.4': {'holds': not maker_in},
        '68.5': {'holds': zero_max <= score.ZERO_TOL},
        '68.6': {'holds': rho is not None and rho > score.SPEARMAN_MIN},
    }
    return predictions


def test_verdict_thresholds_at_and_around_each_boundary():
    assert not _verdicts(rep_late=9)['68.1']['holds'] and _verdicts(rep_late=10)['68.1']['holds']
    assert not _verdicts(rep_early=9)['68.2']['holds'] and _verdicts(rep_early=10)['68.2']['holds']
    assert not _verdicts(late_share=0.5)['68.3']['holds'] and _verdicts(late_share=0.51)['68.3']['holds']
    assert not _verdicts(late_share=None)['68.3']['holds']
    assert _verdicts(maker_in=False)['68.4']['holds'] and not _verdicts(maker_in=True)['68.4']['holds']
    assert _verdicts(zero_max=0.0)['68.5']['holds'] and not _verdicts(zero_max=1e-9)['68.5']['holds']
    assert not _verdicts(rho=0.3)['68.6']['holds'] and _verdicts(rho=0.3001)['68.6']['holds']
    assert not _verdicts(rho=None)['68.6']['holds']


def test_reading_rules():
    assert score._reading(_verdicts(zero_max=1e-9), {})['readable'] is False
    found = score._reading(_verdicts(rep_late=3), {'29_776': 3, '11_8868': 0})
    assert found['readable'] and found['label'] == 'the search found drivers'
    weak = score._reading(_verdicts(rep_late=2), {'29_776': 2, '11_8868': 0})
    assert weak['label'] != 'the search found drivers'
    caveat = score._reading(_verdicts(rho=0.1), {'29_776': 5, '11_8868': 0})
    assert 'caveat' in caveat and 'does not track zeroing' in caveat['caveat']


def test_score_runs_on_a_synthetic_run_directory(tmp_path):
    """A hand-built run directory: 20 candidates per target, 200 nulls, both halves, known Deltas."""
    rng = np.random.default_rng(7)
    n_units, n_layers = 64, 30
    targets = [(11, 8868), (29, 776)]  # the registered targets; others take the partial-run branch
    halves = {'search': [0, 1], 'test': [2, 3]}
    candidates, nulls = {}, {}
    for t in targets:
        key = f'{t[0]}_{t[1]}'
        candidates[key] = [{'layer': 0, 'unit': u, 'attribution': 0.5 if u % 2 == 0 else -0.5} for u in range(20)]
        nulls[key] = [{'layer': 0, 'unit': u} for u in range(20, 40)]
    records = []
    for t in targets:
        key = f'{t[0]}_{t[1]}'
        for half, prompts in halves.items():
            for p in prompts:
                base = [1.0, 2.0, 3.0]
                records.append({'target': list(t), 'half': half, 'condition': 'unchanged', 'kind': 'baseline',
                                'prompt_index': p, 'mean': 2.0, 'positions': base})
                for c in candidates[key]:
                    delta = 0.5 if c['unit'] < 10 else 0.0
                    records.append({'target': list(t), 'half': half, 'condition': f"cand_{c['layer']}_{c['unit']}",
                                    'kind': 'candidate', 'prompt_index': p, 'mean': 2.0 + delta})
                for c in nulls[key]:
                    records.append({'target': list(t), 'half': half, 'condition': f"null_{c['layer']}_{c['unit']}",
                                    'kind': 'null', 'prompt_index': p, 'mean': 2.0 + float(rng.normal(0, 0.05))})
                records.append({'target': list(t), 'half': half, 'condition': 'all_candidates', 'kind': 'all',
                                'prompt_index': p, 'mean': 2.5})
    (tmp_path / 'activations.json').write_text(json.dumps({
        'targets': [list(t) for t in targets], 'halves': halves, 'n_units': n_units,
        'candidates': candidates, 'nulls': nulls, 'records': records}))
    for t in targets:
        stem = f'attr_{t[0]}_{t[1]}'
        arrays = np.zeros((n_layers, n_units))
        for c in candidates[f'{t[0]}_{t[1]}']:
            arrays[c['layer'], c['unit']] = c['attribution']
        np.savez(tmp_path / f'{stem}.npz', attribution=arrays)
        (tmp_path / f'{stem}.json').write_text(json.dumps({
            'target': list(t), 'candidates': candidates[f'{t[0]}_{t[1]}'], 'later_units_max_abs': 0.0}))
    result = score.score_dir(tmp_path)
    assert (tmp_path / 'score.json').is_file() and (tmp_path / 'score.md').is_file()
    late = result['targets']['29_776']
    # Units 0-9 have Delta 0.5 and units 10-19 have Delta 0. Of units 0-9 only the five even ones have a
    # positive attribution, so exactly five replicate; 68.1 needs ten, so it fails.
    assert late['halves']['test']['replicating_count'] == 5
    assert sorted(r['unit'] for r in late['halves']['test']['replicating']) == [0, 2, 4, 6, 8]
    assert result['predictions']['68.1']['holds'] is False
    assert result['reading']['label'] == 'the search found drivers'  # 5 >= 3
    assert result['predictions']['68.5']['holds'] is True
    assert result['predictions']['68.3']['holds'] is False  # no replicating driver is in layers 20-28
    assert result['overlap']['expected']['observed'] == 20  # both targets drew the same synthetic candidates


# ---------------------------------------------------------------- tiny torch model

if torch is not None:
    class _MLP(torch.nn.Module):
        def __init__(self, d, w):
            super().__init__()
            self.up = torch.nn.Linear(d, w, bias=False)
            self.down_proj = torch.nn.Linear(w, d, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.gelu(self.up(x)))

    class _Block(torch.nn.Module):
        def __init__(self, d, w):
            super().__init__()
            self.mlp = _MLP(d, w)

        def forward(self, x):
            return x + self.mlp(x)

    class TinyModel(torch.nn.Module):
        def __init__(self, n_layers=4, d=8, w=16, vocab=32, seed=0):
            super().__init__()
            torch.manual_seed(seed)
            self.embed = torch.nn.Embedding(vocab, d)
            self.layers = torch.nn.ModuleList([_Block(d, w) for _ in range(n_layers)])

        def forward(self, input_ids):
            x = self.embed(input_ids)
            for block in self.layers:
                x = block(x)
            return x

    def _tiny():
        model = TinyModel().eval()
        model.requires_grad_(False)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        return model, list(model.layers), ids

    def test_attribution_is_exactly_zero_at_and_after_the_target_layer():
        model, layers, ids = _tiny()
        info = {}
        with torch.enable_grad():
            attribution = modelrun._attribution(model, layers, ids, (2, 3), target_layer=2, info=info)
        assert set(attribution) == set(range(4))
        for layer in (2, 3):
            assert np.array_equal(attribution[layer], np.zeros_like(attribution[layer]))
        assert np.abs(attribution[0]).sum() > 0 and np.abs(attribution[1]).sum() > 0
        # The removed identity entry is the positive control: d(mean_p a_p)/da_p = 1/P, so it equals -t.
        assert info['target_self_attribution'] == pytest.approx(-info['target_value'], rel=1e-5)
        assert info['target_value'] != 0

    def test_the_68_5_control_can_fail():
        # Declare layer 2 while the sink reads layer 3: layer 2 feeds layer 3, so the control must see it.
        model, layers, ids = _tiny()
        with torch.enable_grad():
            attribution = modelrun._attribution(model, layers, ids, (3, 5), target_layer=2)
        assert np.abs(attribution[2]).sum() > 0

    def test_attribution_predicts_a_small_zeroing_within_a_loose_tolerance():
        model, layers, ids = _tiny()
        target = (3, 5)
        with torch.enable_grad():
            attribution = modelrun._attribution(model, layers, ids, target, target_layer=3)
        base = modelrun._target_activations(model, layers, ids, [{}], target)[0].mean().item()
        for layer, unit in [(0, int(np.argmax(np.abs(attribution[0])))), (1, int(np.argmax(np.abs(attribution[1]))))]:
            a = np.abs(attribution[layer][unit])
            assert a > 0
            eps = 1e-3 / a
            measured = _scaled_zeroing_change(model, layers, ids, target, layer, unit, eps)
            predicted = float(attribution[layer][unit]) * eps
            assert measured == pytest.approx(predicted, abs=0.05 * abs(predicted) + 1e-4), (layer, unit)

    def _scaled_zeroing_change(model, layers, ids, target, layer, unit, eps):
        base = modelrun._target_activations(model, layers, ids, [{}], target)[0].mean().item()
        def hook(_module, args):
            x = args[0].clone()
            x[..., unit] = x[..., unit] * (1.0 - eps)
            return (x,) + tuple(args[1:])
        handle = layers[layer].mlp.down_proj.register_forward_pre_hook(hook)
        try:
            changed = modelrun._target_activations(model, layers, ids, [{}], target)[0].mean().item()
        finally:
            handle.remove()
        return changed - base

    def test_row_zeroing_hook_zeroes_only_its_own_row():
        model, layers, ids = _tiny()
        target = (3, 5)
        batch = ids.repeat(2, 1)
        unchanged, zeroed = modelrun._target_activations(model, layers, batch, [{}, {(0, 2): 0.0}], target)
        alone = modelrun._target_activations(model, layers, ids, [{}], target)[0]
        assert torch.allclose(unchanged, alone)
        assert not torch.allclose(zeroed, alone)
        _, later = modelrun._target_activations(model, layers, batch, [{}, {(3, 0): 0.0}], target)
        assert torch.allclose(later, alone)
else:
    @pytest.mark.skip(reason='torch is not installed on this host')
    def test_attribution_is_exactly_zero_at_and_after_the_target_layer(): pass

    @pytest.mark.skip(reason='torch is not installed on this host')
    def test_attribution_predicts_a_small_zeroing_within_a_loose_tolerance(): pass

    @pytest.mark.skip(reason='torch is not installed on this host')
    def test_row_zeroing_hook_zeroes_only_its_own_row(): pass


def test_synthetic_score_json_is_stable(tmp_path):
    """The score writes the same json twice (no hidden randomness)."""
    rng = np.random.default_rng(3)
    records = [{'target': [1, 3], 'half': 'test', 'condition': 'unchanged', 'kind': 'baseline', 'prompt_index': 0,
                'mean': 1.0, 'positions': [1.0, 2.0]}]
    candidates = [{'layer': 0, 'unit': u, 'attribution': float(rng.normal())} for u in range(20)]
    records += [{'target': [1, 3], 'half': 'test', 'condition': f"cand_0_{c['unit']}", 'kind': 'candidate',
                 'prompt_index': 0, 'mean': 1.0 + float(rng.normal())} for c in candidates]
    nulls = [{'layer': 0, 'unit': u} for u in range(20, 40)]
    records += [{'target': [1, 3], 'half': 'test', 'condition': f"null_0_{c['unit']}", 'kind': 'null',
                 'prompt_index': 0, 'mean': 1.0 + float(rng.normal())} for c in nulls]
    records.append({'target': [1, 3], 'half': 'test', 'condition': 'all_candidates', 'kind': 'all',
                    'prompt_index': 0, 'mean': 1.0})
    (tmp_path / 'activations.json').write_text(json.dumps({
        'targets': [[1, 3]], 'halves': {'test': [0]}, 'n_units': 64, 'candidates': {'1_3': candidates},
        'nulls': {'1_3': nulls}, 'records': records}))
    arrays = np.zeros((4, 64))
    for c in candidates:
        arrays[0, c['unit']] = c['attribution']
    np.savez(tmp_path / 'attr_1_3.npz', attribution=arrays)
    (tmp_path / 'attr_1_3.json').write_text(json.dumps({'target': [1, 3], 'candidates': candidates,
                                                        'later_units_max_abs': 0.0}))
    first = score.score_dir(tmp_path)
    text = (tmp_path / 'score.json').read_text()
    second = score.score_dir(tmp_path)
    assert text == (tmp_path / 'score.json').read_text()
    assert first['predictions'] == {} and first['reading']['readable'] is False  # a partial run is not scored
    assert first['targets']['1_3']['halves']['test']['null_count'] == 20
