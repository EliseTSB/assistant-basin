"""Small CPU-only mechanics tests; no model download or corpus access."""
import sys
from pathlib import Path
import pytest
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import score136
import remove136
import scan136

class TinyCausal(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj=nn.Linear(3,3,bias=False)
        self.head=nn.Linear(3,4,bias=False)
        with torch.no_grad():
            self.down_proj.weight.copy_(torch.eye(3))
            self.head.weight.copy_(torch.tensor([[1.,0,0],[0,1,0],[0,0,1],[1,1,1]]))
    def forward(self,x):
        return self.head(x+torch.cumsum(self.down_proj(x),dim=1))

def test_zero_at_one_position_changes_only_that_position_and_later():
    model=TinyCausal(); x=torch.tensor([[[1.,2,3],[2,1,1],[1,3,2],[2,2,1]]])
    baseline=model(x)
    def zero(_m,args):
        z=args[0].clone(); z[0,1,0]=0; return (z,)
    h=model.down_proj.register_forward_pre_hook(zero)
    try: changed=model(x)
    finally: h.remove()
    assert torch.equal(changed[:,:1],baseline[:,:1])
    assert not torch.equal(changed[:,1:],baseline[:,1:])

def test_do_nothing_hook_is_bit_identical():
    model=TinyCausal(); x=torch.randn(2,5,3); plain=model(x)
    h=model.down_proj.register_forward_pre_hook(lambda _m,_args:None)
    try: hooked=model(x)
    finally: h.remove()
    assert torch.equal(plain,hooked)

def test_stability_matches_hand_computation():
    vectors=[[1.,0.],[0.,1.],[1.,1.]]
    hand=(score136.pearson(vectors[0],[.5,1.])+score136.pearson(vectors[1],[1.,.5])+score136.pearson(vectors[2],[.5,.5]))/3
    assert score136.stability(vectors)==pytest.approx(hand)

def test_using_selected_indices_for_stability_changes_the_answer():
    left=[1.,0.,0.,1.]; right=[0.,1.,0.,1.]
    full=score136.pearson(left,right)
    selected=score136.pearson([left[0],left[1]],[right[0],right[1]])
    assert full is not None and selected is not None
    assert abs(full-selected)>.5

def test_permutation_keeps_labels_inside_depth_bands():
    score136.BANDS=((0,7,"L0-7"),(8,15,"L8-15"))
    rows=[{"band":"L0-7","kind":"concept"},{"band":"L0-7","kind":"slot"},
          {"band":"L8-15","kind":"concept"},{"band":"L8-15","kind":"concept"},{"band":"L8-15","kind":"slot"}]
    shuffled=score136.shuffle_labels_within_bands(rows)
    for name in ("L0-7","L8-15"):
        before=[r["kind"] for r in rows if r["band"]==name]
        assert sorted(shuffled[name])==sorted(before)

class TinyOutput:
    def __init__(self, logits): self.logits=logits

class TinyPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj=nn.Linear(3,3,bias=False)
        with torch.no_grad(): self.down_proj.weight.copy_(torch.eye(3))
    def forward(self,input_ids,attention_mask=None,use_cache=False):
        x=torch.nn.functional.one_hot(input_ids%3,3).float()
        z=self.down_proj(x)
        return TinyOutput(torch.cumsum(z,dim=1) @ torch.tensor([[1.,0,0,1],[0,1,0,1],[0,0,1,1]]))

class TinyBatchModel(nn.Module):
    def __init__(self):
        super().__init__(); self.layers=nn.ModuleList([nn.Module()]); self.layers[0].mlp=nn.Module(); self.layers[0].mlp.down_proj=nn.Linear(3,3,bias=False)
        with torch.no_grad(): self.layers[0].mlp.down_proj.weight.copy_(torch.eye(3))
    def forward(self,input_ids,attention_mask=None,use_cache=False):
        x=torch.nn.functional.one_hot(input_ids%3,3).float(); z=self.layers[0].mlp.down_proj(x)
        logits=torch.cumsum(z,dim=1) @ torch.tensor([[1.,0,0,1],[0,1,0,1],[0,0,1,1]])
        return TinyOutput(logits)

def test_remove_pipeline_uses_part33_conditional_hook_on_tiny_model():
    model=TinyBatchModel(); rows=[{"token_ids":[0,0,2,0],"position":1,"neuron":0}]
    da,db=remove136.effect_batch(model,model.layers[0].mlp.down_proj,rows,"cpu",torch.arange(4))[0]
    assert any(abs(x)>1e-8 for x in da)
    assert len(db)==4

def test_effect_on_last_token_is_kept_with_empty_effect_b():
    model=TinyBatchModel(); row={"token_ids":[1,2,0],"position":2,"neuron":0}
    prepared=remove136.prepare_firing(row)
    assert prepared is not None and prepared["position"]==2
    da,db=remove136.effect_batch(model,model.layers[0].mlp.down_proj,[prepared],"cpu",torch.arange(4))[0]
    assert any(abs(x)>1e-8 for x in da)
    assert db==[0.0,0.0,0.0,0.0]

def test_steering_push_uses_down_proj_input_width():
    layer=nn.Linear(3,5,bias=False)
    push=remove136.steering_push(layer,2,.75,"cpu",torch.float32)
    assert push.shape==(1,3)
    assert push.tolist()==[[0.0,0.0,.75]]

def test_pilot_selects_twelve_of_each_kind_across_bands_deterministically():
    bands=((0,1,"low"),(2,3,"mid"),(4,5,"high"))
    poles=[{"pole_id":f"{kind}-{layer}-{i}","layer":layer,"kind":kind}
           for kind in ("concept","slot") for layer in range(6) for i in range(3)]
    first=scan136.select_pilot_poles(poles,24,bands)
    second=scan136.select_pilot_poles(poles,24,bands)
    assert [p["pole_id"] for p in first]==[p["pole_id"] for p in second]
    assert sum(p["kind"]=="concept" for p in first)==12
    assert sum(p["kind"]=="slot" for p in first)==12
    for kind in ("concept","slot"):
        counts={name:sum(p["kind"]==kind and lo<=p["layer"]<=hi for p in first)
                for lo,hi,name in bands}
        assert all(counts.values())

def test_pending_gate_writes_only_the_gate_block(tmp_path):
    gates={"gate_1":None,"gate_2":"pending","registered_reading_permitted":False}
    score136.write_gate_only(tmp_path,gates)
    payload=__import__("json").loads((tmp_path/"score136.json").read_text())
    assert payload=={"gates":gates}
    report=(tmp_path/"SCORE.md").read_text()
    assert "## Gates" in report
    assert "unit_rows" not in report and "matched_control_rows" not in report

def test_effect_b_sums_positions_t_plus_1_through_t_plus_8():
    base=torch.zeros((12,1)); changed=torch.arange(12,dtype=torch.float32).reshape(12,1)
    pos=2; end=min(pos+8,changed.shape[0]-1)
    got=(changed[pos+1:end+1]-base[pos+1:end+1]).sum().item()
    assert got==sum(range(3,11))
    assert got!=sum(range(2,10))

def test_batched_different_firings_only_intervene_on_their_own_rows():
    model=TinyBatchModel(); rows=[{"token_ids":[0,0,2,0],"position":1,"neuron":0},
                                   {"token_ids":[0,1,2,0],"position":1,"neuron":1}]
    batched=remove136.effect_batch(model,model.layers[0].mlp.down_proj,rows,"cpu",torch.arange(4))
    singles=[remove136.effect_batch(model,model.layers[0].mlp.down_proj,[row],"cpu",torch.arange(4))[0] for row in rows]
    for actual,expected in zip(batched,singles):
        assert actual[0]==pytest.approx(expected[0])
        assert actual[1]==pytest.approx(expected[1])

def test_gate3_compares_one_firing_to_another_poles_mean():
    by_pole={"a":[{"layer":1,"effect_a":[1.,0.]} for _ in range(24)],
             "b":[{"layer":1,"effect_a":[0.,1.]} for _ in range(24)]}
    values=score136.gate3_off_target(by_pole,lambda _layer:"band")
    assert values["band"]==pytest.approx(1.)

