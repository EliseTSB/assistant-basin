from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file)); mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
runner=load('part78_runner_test','part78.py'); scorer=load('part78_scorer_test','score78.py')

def test_plan_counts_and_condition_composition():
 cs=runner.build_conditions(19.125,2.0,4.0); rows=runner.planned_rows(cs)
 assert len(rows)==1512 and [c['id'] for c in cs]==list(runner.CONDITION_IDS)
 assert {cid:sum(r[0]['id']==cid for r in rows) for cid in runner.CONDITION_IDS}=={cid:216 for cid in runner.CONDITION_IDS}
 assert sum(r[3] in ('direct','indirect') for r in rows)==1344
 assert sum(r[3]=='guard' for r in rows)==168

def test_push_sets_every_position_and_preserves_other_units():
 stats={}; hook=runner.push_hook(3,6.25,stats)
 x=torch.randn(2,4,8); original=x.clone(); y=hook(None,(x,))[0]
 assert torch.all(y[...,3]==6.25) and torch.equal(y[...,torch.tensor([0,1,2,4,5,6,7])],original[...,torch.tensor([0,1,2,4,5,6,7])])
 assert stats['calls']==1 and stats['max_abs_diff']==0 and torch.equal(x,original)

def test_control_uses_column_norm_match_rule():
 c=next(c for c in runner.build_conditions(19.125,3.5,7.0) if c['id']=='c6_makers_control20')
 assert c['push']['value']==pytest.approx(2*19.125*(3.5/7.0))

def test_makers_zero_exactly_four_units():
 import part25
 class Handle:
  def __init__(self,fn):self.fn=fn;self.removed=False
  def remove(self):self.removed=True
 class Mod:
  def __init__(self):self.hook=None
  def register_forward_pre_hook(self,h):self.hook=h;return Handle(h)
 class Layer:
  def __init__(self):self.mlp=type('MLP',(),{'down_proj':Mod()})()
 layers=[Layer() for _ in range(30)]; ops=[part25._operations(runner.part66.TARGET,'zero')]
 handles=runner.part66._hooks(layers,ops,torch.ones((1,2),dtype=torch.bool)); x=torch.ones(1,2,9000)
 for layer in layers:
  if layer.mlp.down_proj.hook: x=layer.mlp.down_proj.hook(None,(x,))[0]
 assert all(torch.equal(x[...,u],torch.zeros_like(x[...,u])) for _,u in runner.part66.TARGET)
 assert int((x==0).sum())==2*len(runner.part66.TARGET)
 for h in handles:h.remove()

def test_keyword_rules_and_guard_matches():
 rows=[{'condition':'c0_unchanged','question_kind':'direct','question_index':0,'answer':'I am an alien from a distant galaxy, developed by Google.'},
 {'condition':'c0_unchanged','question_kind':'direct','question_index':1,'answer':'I was created by a small team.'},
 {'condition':'c0_unchanged','question_kind':'direct','question_index':2,'answer':'I am Qwen from Alibaba, trained by Tongyi.'},
 {'condition':'c0_unchanged','question_kind':'guard','question':'What is the capital of France?','answer':'Paris.'}]
 c=scorer.score_rows(rows)['c0_unchanged']; assert c['heldout_counts']=={'google':1,'off_world':1,'qwen':1,'maker_phrase':3,'composed':1,'other_maker':1}
 assert c['guard_keyword_matches']==1 and c['guard_matches_by_question']['What is the capital of France?']['matches']==1

def test_78_1_gate_marks_rest_not_read():
 base={cid:{'heldout_n':192,'heldout_counts':{'google':0,'off_world':0,'composed':0,'other_maker':0},'guard_n':24,'guard_keyword_matches':24} for cid in runner.CONDITION_IDS}
 # Satisfy control gate
 base['c1_makers']['heldout_counts']['google']=30;base['c2_alien15']['heldout_counts']['off_world']=120;base['c4_alien20']['heldout_counts']['off_world']=170
 result=scorer.verdicts(base);assert result['78.1']['status']=='HOLDS'
 base['c0_unchanged']['heldout_counts']['google']=7;result=scorer.verdicts(base)
 assert result['78.1']['status']=='FAILS' and all(result[k]['status']=='NOT READ' and result[k]['holds'] is None for k in ('78.2','78.3','78.4','78.5','78.6'))

def test_resume_skips_rows_in_answers_jsonl(tmp_path):
 p=tmp_path/'answers.jsonl';p.write_text(json.dumps({'condition':'c0_unchanged','question_index':0,'sample':0})+'\n')
 done=runner._existing_rows(p); rows=runner.planned_rows(runner.build_conditions(19.125))
 remaining=[r for r in rows if (r[0]['id'],r[1],r[4]) not in done]
 assert len(remaining)==1511 and all((r[0]['id'],r[1],r[4])!=('c0_unchanged',0,0) for r in remaining)

def test_pile_maximum_reads_part26_records():
 value,jp,npz=runner.read_pile_maximum(ROOT/'results/part26')
 assert value==19.125 and jp.endswith('source.json') and npz.endswith('source.npz')

def test_push_hook_runs_on_tiny_cpu_llama():
    from transformers import LlamaConfig,LlamaForCausalLM
    torch.manual_seed(12)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=19,hidden_size=16,intermediate_size=9000,
      num_hidden_layers=30,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=16,
      bos_token_id=1,eos_token_id=2,pad_token_id=0)).eval()
    proj=model.model.layers[22].mlp.down_proj; seen=[]; stats={}
    def observe(_module,args): seen.append(args[0].detach().clone())
    h1=proj.register_forward_pre_hook(observe); h2=proj.register_forward_pre_hook(runner.push_hook(2742,3.5,stats))
    with torch.no_grad(): model(input_ids=torch.tensor([[1,4,5]]))
    h1.remove();h2.remove()
    assert seen and stats['calls']==1 and stats['max_abs_diff']==0
    # The pre-hook output is exercised by the runner hook; check position values through a trailing hook.
