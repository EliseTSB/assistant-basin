from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import numpy as np, pytest, torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from part79 import part79 as run, score79 as score, step0

def test_add_hook_changes_only_requested_feature_everywhere():
    x=torch.randn(2,3,7); before=x.clone(); stats={}; y=run.add_hook(4,2.5,stats)(None,(x,))[0]
    expected=before.clone(); expected[...,4]+=2.5
    assert torch.equal(y,expected) and torch.equal(x,before)
    assert stats['calls']==1 and stats['max_abs_diff']==pytest.approx(0)

def test_delta_formula_and_control_units():
    med={'L13':{'median':12.},'L14':{'median':20.}}
    class Proj:
        def __init__(self):self.weight=torch.tensor([[3.,4.,0.]])
    class Layer:
        def __init__(self):self.mlp=type('M',(),{'down_proj':Proj()})()
    layers=[Layer(),Layer()]
    got=step0.compute_deltas(type('Model',(),{})(),layers,med) if False else None
    # Isolate the registered calculation with synthetic 9216-wide columns.
    class WideProj:
        def __init__(self):
            self.weight=torch.ones((2,9216)); self.weight[:,8949]=torch.tensor([3.,4.]); self.weight[:,7956]=torch.tensor([5.,12.])
    layers=[type('Layer',(),{'mlp':type('M',(),{'down_proj':WideProj()})()})() for _ in range(15)]
    ds=step0.compute_deltas(None,layers,med)
    assert ds['L14_N8949']['column_norm']==pytest.approx(5)
    assert ds['L14_N8949']['delta']==pytest.approx(4)
    assert ds['L13_N7956']['delta']==pytest.approx(-12/13)
    assert run.CTL14==tuple(__import__('neuron_report').control_units(9216,8949,5,79079))
    assert run.CTL13==tuple(__import__('neuron_report').control_units(9216,7956,5,79080))
    assert 8949 not in run.CTL14 and 7956 not in run.CTL13

def test_plan_is_1944_and_216_per_condition():
    deltas={f'L{l}_N{u}':{'layer':l,'unit':u,'delta':.5} for l,u in [(14,8949),(13,7956),*[(14,u) for u in run.CTL14],*[(13,u) for u in run.CTL13]]}
    cs=run.build_conditions(deltas); rows=run.planned_rows(cs)
    assert len(rows)==1944 and list(c['id'] for c in cs)==list(run.CONDITION_IDS)
    assert all(sum(r[0]['id']==cid for r in rows)==216 for cid in run.CONDITION_IDS)

def test_step0_quantiles_skip_huge_position_zero():
    # 23 layer fixture emits norm 1000 at position zero and 2 at every measured position.
    class Block(torch.nn.Module):
        def forward(self,x):
            y=torch.ones_like(x)*2; y[:,0,:]=1000
            return (y,)
    class Toy(torch.nn.Module):
        def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(()))
        def forward(self,input_ids,use_cache=False):
            x=torch.zeros((*input_ids.shape,1),device=input_ids.device)
            for b in blocks:b(x)
    blocks=torch.nn.ModuleList([Block() for _ in range(23)]); model=Toy()
    got=step0.summarize_norms(model,blocks,[[1,2,3,4]],1)
    assert got['L22']['median']==pytest.approx(2) and got['L22']['p90']==pytest.approx(2)

def _good_conditions():
    cs={c:{'heldout_n':192,'guard_n':24,'guard_keyword_matches':24,'heldout_counts':{k:0 for k in ('google','off_world','qwen','composed','other_maker')}} for c in run.CONDITION_IDS}
    cs['e1_makers']['heldout_counts']['google']=30
    return cs

def test_scorer_gates_and_not_read():
    c=_good_conditions(); v=score.verdicts(c,{'medians':{'L22':{'median':25.4}}},{'evil_villain':3,'unchanged_villain':0},None)
    assert v['79.1']['status']=='HOLDS' and v['79.2']['status']=='HOLDS'
    assert v['79.3']['status']=='NOT READ' and v['79.6']['holds'] is None
    c['e1_makers']['heldout_n']=191; v=score.verdicts(c)
    assert all(v[k]['status']=='NOT READ' and v[k]['holds'] is None for k in ('79.3','79.4','79.5','79.6','79.7'))

def test_reader_input_is_shuffled_without_condition(tmp_path):
    rows=[{'condition':cid,'question_index':i,'question_kind':'direct','sample':s,'answer':'ok'} for cid in run.CONDITION_IDS for i in range(24) for s in range(8)]
    rows += [{'condition':cid,'question_index':25+i,'question_kind':'guard','sample':s,'question':'g','answer':'a'} for cid in run.CONDITION_IDS for i in range(6) for s in range(4)]
    (tmp_path/'answers.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    score.score_file(tmp_path)
    inp=[json.loads(l) for l in (tmp_path/'reader_input.jsonl').read_text().splitlines()]
    assert len(inp)==9*192 and all('condition' not in r for r in inp)



def test_pilot_prompt_asks_for_the_assistant_turn():
    from part79 import step0
    seen={}
    class Tok:
        chat_template='x'
        def apply_chat_template(self,msgs,**kw):
            seen.update(kw); seen['msgs']=msgs; return 'rendered'
    assert step0.pilot_prompt(Tok(),'Who are you?')=='rendered'
    assert seen['add_generation_prompt'] is True and seen['enable_thinking'] is False
    assert seen['msgs']==[{'role':'user','content':'Who are you?'}]
