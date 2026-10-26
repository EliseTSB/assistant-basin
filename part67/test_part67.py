"""Unit tests for Part 67 scoring, condition construction, and intervention hooks."""
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(1,str(ROOT))
import score67 as sc


def test_empirical_p_counts_ties_as_at_or_above():
    assert sc.empirical_p(2.0,[2.0,-2.0,1.99]+[0.0]*47)==3/51
    assert sc.empirical_p(-3.0,[0.0]*50)==1/51


def test_verdict_threshold_boundaries():
    assert not sc.passes_67_1([1.0],[[1.0]+[0.0]*49])
    assert sc.passes_67_1([1.0001],[[1.0]+[0.0]*49])
    assert sc.passes_67_2(True) and not sc.passes_67_2(False)
    assert not sc.passes_67_3(2.0,[2.0,1.0],[0.0]*50)
    assert sc.passes_67_3(2.0001,[2.0,1.0],[2.0001]+[0.0]*49) is False  # Claude: a random triple tying the joint must fail
    assert sc.passes_67_3(2.0001,[2.0,1.0],[1.99]+[0.0]*49)
    assert sc.passes_67_4(5,None) and not sc.passes_67_4(6,None)
    assert sc.passes_67_4(None,5) and not sc.passes_67_4(None,6)
    assert sc.passes_67_5(1e-3) and not sc.passes_67_5(1.0001e-3)
    assert not sc.passes_67_6(2.0,[2.0,1.0])
    assert sc.passes_67_6(2.0001,[2.0,1.0])


def test_failed_control_gates_other_claims():
    # The scorer's exact reading rule: either failed control makes 67.1–67.4 NOT READ.
    for c675,c676 in ((False,True),(True,False),(False,False)):
        assert not sc.reading_gate(c675,c676)
    assert sc.reading_gate(True,True)
    assert sc.passes_67_5(0.0) and sc.passes_67_6(1.0,[0.5])


def test_condition_inventory_and_random_exclusions():
    torch=pytest.importorskip('torch')
    import part67 as runner
    targets=set(runner.TARGETS)
    rows=runner.build_conditions(9000,32)
    assert len(rows)==207
    assert len({r['id'] for r in rows})==207
    singles=[r for r in rows if r['kind']=='random_single']
    triples=[r for r in rows if r['kind']=='random_triple']
    assert {r['seed'] for r in singles}==set(range(6700,6750))
    assert {r['seed'] for r in triples}==set(range(6700,6750))
    assert all(len(r['set_units'])==1 and r['set_units'][0] not in targets for r in singles)
    assert all(len(r['set_units'])==3 and not targets.intersection(r['set_units']) for r in triples)


def test_tiny_torch_model_hooks_record_after_intervention():
    torch=pytest.importorskip('torch')
    import part67 as runner
    import part25

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.down_proj=torch.nn.Linear(4,4,bias=False)
        def forward(self,x): return self.down_proj(torch.relu(x))
    class Block(torch.nn.Module):
        def __init__(self): super().__init__(); self.mlp=MLP()
        def forward(self,x): return x+self.mlp(x)
    class Tiny(torch.nn.Module):
        def __init__(self): super().__init__(); self.layers=torch.nn.ModuleList([Block(),Block()])
        def forward(self,input_ids):
            x=torch.nn.functional.one_hot(input_ids,num_classes=4).float()
            for layer in self.layers: x=layer(x)
            return x
    torch.manual_seed(4); model=Tiny().eval(); ids=torch.tensor([[0,1,2]])
    targets=[(0,2),(1,1)]
    baseline_ops=[part25._operations([], 'zero')]
    handles,base=runner._activation_hooks(model.layers,baseline_ops,ids.shape[1],targets)
    try:
        with torch.no_grad(): model(input_ids=ids)
    finally:
        for h in handles:h.remove()
    zero_ops=[part25._operations([(0,2)],'zero')]
    handles,zero=runner._activation_hooks(model.layers,zero_ops,ids.shape[1],targets)
    try:
        with torch.no_grad(): model(input_ids=ids)
    finally:
        for h in handles:h.remove()
    assert torch.equal(zero[0][:,:,0],torch.zeros_like(zero[0][:,:,0]))
    late_ops=[part25._operations([(1,1)],'zero')]
    handles,late=runner._activation_hooks(model.layers,late_ops,ids.shape[1],targets)
    try:
        with torch.no_grad(): model(input_ids=ids)
    finally:
        for h in handles:h.remove()
    assert torch.equal(base[0],late[0])


def test_weight_paths_on_a_tiny_qwen_shaped_model(monkeypatch):
    # Claude, 2026-10-26: the first real run crashed in calculate_weight_paths (the random singles' (layer, unit)
    # pairs were reduced to layer numbers), a path no test reached. This one runs it and checks one value by hand.
    torch = pytest.importorskip('torch')
    import part67 as runner
    d, inter = 6, 5
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(d, inter, bias=False)
            self.up_proj = torch.nn.Linear(d, inter, bias=False)
            self.down_proj = torch.nn.Linear(inter, d, bias=False)
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.mlp = MLP(); self.post_attention_layernorm = torch.nn.RMSNorm(d)
    torch.manual_seed(7)
    layers = torch.nn.ModuleList([Block(), Block()])
    with torch.no_grad():
        layers[1].post_attention_layernorm.weight.uniform_(0.5, 1.5)
    monkeypatch.setattr(runner, 'TARGETS', [(0, 1), (1, 2)])
    monkeypatch.setattr(runner, 'UPSTREAM', [(0, 1)])
    conditions = [{'id': 'rand_0_1', 'kind': 'random_single', 'layer': 0, 'seed': 1, 'set_units': [(0, 3)]}]
    out = runner.calculate_weight_paths(layers, conditions)
    pairs = {(tuple(p['upstream']), tuple(p['downstream'])): p for p in out['paths']}
    assert set(pairs) == {((0, 1), (1, 2)), ((0, 3), (1, 2))}
    down = layers[0].mlp.down_proj.weight[:, 3]
    gain = layers[1].post_attention_layernorm.weight
    want = float(torch.dot(down * gain, layers[1].mlp.gate_proj.weight[2, :]))
    assert abs(pairs[((0, 3), (1, 2))]['gate'] - want) < 1e-5
