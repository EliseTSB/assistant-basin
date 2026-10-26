"""Focused correctness tests for Part 69's statistics and runner."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import score69
import part69

def test_p_values_both_directions_and_ties():
    assert score69.empirical_p(4, [1, 2, 3], 'positive') == 0.25
    assert score69.empirical_p(-4, [-1, -2, -3], 'negative') == 0.25
    assert score69.two_sided_p(4, [1, 2, 3]) == 0.5
    assert score69.empirical_p(2, [2, 1, 3], 'positive') == 0.75
    assert score69.empirical_p(4, [1, 2, 3], 'positive') == 0.25

def test_verdict_boundaries_and_not_read():
    dg = {'makers': 1.0, 'l11_alone': 1.0, 'l29_alone': 1.0, 'l11_down': 0.0, 'l29_up': 0.0}
    nulls = {'null_makers': [0.0]*50, 'null_l11': [1.0]*50, 'null_l29': [0.0]*50,
             'null_l11down': [0.0]*50, 'null_l29up': [0.0]*50}
    got = score69.verdicts(dg, nulls, 0.5, 20)
    assert got['69.1']['pass'] is True
    assert got['69.2']['pass'] is True
    assert got['69.3']['pass'] is True  # one-sided p=1 on ties
    assert got['69.4']['pass'] is True  # positive Dg and all 50 nulls below
    assert got['69.5']['pass'] is True  # all ties: two-sided p = 1, which is above 0.1 (Claude, 2026-10-26: the build asserted False)
    assert got['69.6']['pass'] is True
    near = {**nulls, 'null_l11': [1.0, 1.0] + [0.0]*48,
            'null_l11down': [1.0, 1.0] + [0.0]*48,
            'null_l29up': [1.0, 1.0] + [0.0]*48}
    assert score69.verdicts(dg, near, 0.5, 20)['69.3']['pass'] is True  # p=3/51 > .05
    assert score69.two_sided_p(1.0, near['null_l11down']) > 0.1
    assert score69.two_sided_p(2.0, near['null_l11down']) <= 0.1
    edge = {**nulls, 'null_makers': [0.0]*49 + [1.0]}
    assert score69.verdicts(dg, edge, 0.5, 20)['69.1']['pass'] is False  # ties count; p=2/51
    assert score69.verdicts({**dg, 'l29_alone': 0.0}, nulls, 0.5, 20)['69.4']['pass'] is False
    blocked = score69.verdicts(dg, {**nulls, 'null_makers': [2.0]*50}, 0.49, 19)
    for key in ('69.3', '69.4', '69.5', '69.6'):
        assert blocked[key]['verdict'] == 'NOT READ'
    assert score69.verdicts(dg, nulls, 0.499999, 20)['69.2']['pass'] is False
    assert score69.verdicts(dg, nulls, 0.5, 19)['69.2']['pass'] is False

def test_driver_matched_draws_are_deterministic_and_in_pools():
    matched = {}
    drivers11 = [(10, 4133), (10, 54), (7, 296), (10, 3551), (10, 8663), (3, 4169), (9, 1740)]
    drivers29 = [(27, 3325), (24, 3949)]
    for target, drivers in (('11_8868', drivers11), ('29_776', drivers29)):
        matched[target] = {}
        for layer, unit in drivers:
            key = f'{layer}_{unit}'
            matched[target][key] = [{'layer': layer, 'unit': i} for i in range(60)]
    one = part69.draw_driver_nulls(matched)
    two = part69.draw_driver_nulls(matched)
    assert one == two
    assert len(one['null_l11down_k']) == len(one['null_l29up_k']) == 50
    for sets, drivers in ((one['null_l11down_k'], drivers11), (one['null_l29up_k'], drivers29)):
        assert all(len(s) == len(drivers) for s in sets)
        for s in sets:
            for (expected_layer, driver_unit), chosen in zip(drivers, s):
                # Pools live under their candidate target, not the driver's own key.
                target = '11_8868' if len(drivers) == 7 else '29_776'
                pool = {(int(r['layer']), int(r['unit']))
                        for r in matched[target][f'{expected_layer}_{driver_unit}']}
                assert chosen in pool

def test_main_imports_resolve():
    for script in ('part69.py', 'score69.py'):
        proc = subprocess.run([sys.executable, str(HERE / script), '--help'], cwd=ROOT,
                              text=True, capture_output=True, timeout=30)
        assert proc.returncode == 0, proc.stderr
        assert 'usage' in proc.stdout.lower()

def test_tiny_runner_and_direct_measurement(tmp_path):
    torch = pytest.importorskip('torch')
    from torch import nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__(); self.down_proj = nn.Linear(3, 3, bias=False)
        def forward(self, x): return self.down_proj(x)
    class Block(nn.Module):
        def __init__(self): super().__init__(); self.mlp = MLP()
        def forward(self, x): return x + self.mlp.down_proj(x)
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__(); self.embed = nn.Embedding(8, 3); self.layers = nn.ModuleList([Block(), Block()])
            self.lm_head = nn.Linear(3, 8, bias=False)
        def forward(self, input_ids, **kwargs):
            x = self.embed(input_ids)
            for layer in self.layers: x = layer(x)
            return type('Output', (), {'logits': self.lm_head(x)})()

    model = Tiny().eval()
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        model.embed.weight[1] = torch.tensor([1.0, 2.0, 0.5])
        model.layers[0].mlp.down_proj.weight.copy_(torch.eye(3))
        model.layers[1].mlp.down_proj.weight.copy_(torch.eye(3))
        model.lm_head.weight[0] = torch.tensor([0.0, 0.0, 0.0])
        model.lm_head.weight[1] = torch.tensor([1.0, 0.0, 0.0])
        model.lm_head.weight[2] = torch.tensor([0.0, 1.0, 0.0])
        for i in range(3, 8): model.lm_head.weight[i] = torch.tensor([-0.2*i, 0.1*i, 0.05*i])
    core = [{'id':'unchanged','set_units':[]}, {'id':'makers','set_units':[(0,0)]},
            {'id':'l11_alone','set_units':[(0,0)]}, {'id':'l29_alone','set_units':[(1,0)]},
            {'id':'l11_down','set_units':[(0,1)]}, {'id':'l29_up','set_units':[(1,1)]}]
    for name, layer, unit in [('null_makers',0,2), ('null_l11',0,2), ('null_l29',1,2),
                              ('null_l11down',0,2), ('null_l29up',1,2)]:
        core.extend({'id':f'{name}_k{k}', 'set_units':[(layer,unit)]} for k in range(50))
    prompts = [[1, 1] for _ in range(25)]
    records = part69.run_69(model, model.layers, prompts, list(range(7)), core, tmp_path, batch=32)
    (tmp_path / 'conditions.json').write_text(json.dumps([{'id':c['id'], 'units':c['set_units']} for c in core]))
    # Reproduce one row with an independent test hook and compare every requested logit.
    condition = next(c for c in core if c['id'] == 'makers')
    with torch.no_grad():
        unchanged = model(input_ids=torch.tensor([prompts[0]])).logits[0, -1, :7]
        handle = model.layers[0].mlp.down_proj.register_forward_pre_hook(
            lambda _m, args: (args[0].clone().index_fill(-1, torch.tensor([0]), 0),))
        try: manual = model(input_ids=torch.tensor([prompts[0]])).logits[0, -1, :7]
        finally: handle.remove()
    actual = next(r['readout_logits'] for r in records if r['condition'] == 'makers' and r['question_index'] == 0)
    assert torch.allclose(torch.tensor(actual), manual, atol=1e-6)
    assert torch.max(torch.abs(manual - unchanged)).item() > 1e-4
    scored = score69.score_directory(tmp_path)
    assert '69.1' in scored['scores'] and len(scored['positive_control']) > 0


def test_main_plan_after_argparse_without_the_model():
    """Everything main() reads after argparse except the model: --help exits before it, and the first launch crashed
    there (module 'part66' has no attribute 'DIRECT'). Uses the real Part 68b matched.json."""
    torch_mod = pytest.importorskip('torch')
    _, part25_mod, nr_mod, part66_mod = part69._runtime()
    qs = part66_mod.DIRECT + part66_mod.INDIRECT + part66_mod.POSITIVE
    assert len(qs) == 25 and qs[-1] == 'Who created you?'
    for name in ('_token_ids', 'actual_device', 'install_intervention_hooks', '_operations'):
        assert hasattr(part25_mod, name), name
    for name in ('load_model', 'find_text_layers', 'build_prompt', 'spill_guard'):
        assert hasattr(nr_mod, name), name
    sets = part66_mod.random_sets(9216, 32)
    assert len(sets) == 50 and all(len(units) == 4 for units in sets)
    matched = json.loads((Path(__file__).resolve().parents[1] / 'results/part68/b/matched.json').read_text())
    magnitudes = {11: [float(i % 97) for i in range(9216)], 29: [float(i % 89) for i in range(9216)]}
    try:
        conditions = part69.make_conditions(matched, magnitudes)
    except (KeyError, TypeError):
        conditions = part69.make_conditions(matched, {str(k): v for k, v in magnitudes.items()})
    ids = [c['id'] for c in conditions]
    for fixed in ('unchanged', 'makers', 'l11_alone', 'l29_alone', 'l11_down', 'l29_up'):
        assert fixed in ids, fixed
    for family in ('null_l11', 'null_l29', 'null_l11down', 'null_l29up'):
        assert sum(i.startswith(family + '_k') for i in ids) == 50, family

