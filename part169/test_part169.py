import json,sys,types,shlex,shutil,subprocess
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from part169 import run169,score169

class Tok:
    all_special_ids=[0,9]
    eos_token_id=9
    def get_added_vocab(self):return {'</think>':12,'<tool_response>':13}
    def decode(self,ids,skip_special_tokens=False):return ' '.join(map(str,ids))

def test_added_ids_are_special_and_removed_from_ranked_list():
    tok=Tok();assert run169.special_ids(tok)=={0,9,12,13}
    assert run169.ranked_ids({'top_ids':list(range(20))},tok)==[x for x in range(20) if x not in {0,9,12,13}]

def test_plain50_moves_probabilities_to_cpu_before_vocabulary_topk(monkeypatch):
    seen=[];original=torch.topk
    def capture(tensor,*args,**kwargs):
      seen.append(tensor.device.type)
      return original(tensor,*args,**kwargs)
    monkeypatch.setattr(torch,'topk',capture)
    result=run169.plain50_ids(torch.linspace(0,1,100),{0},torch)
    assert seen==['cpu'] and len(result)==50 and 0 not in result

def test_free_and_clear_share_recipe_and_clear_excludes_every_answer_token():
    q=np.linspace(.00001,.01,30000);own=[29999];donor=[0];special={9};plain=[100,101,102]
    free=run169.draw_mass_ids(q,own,donor,special,'n+',19,0,0,())
    clear=run169.draw_mass_ids(q,own,donor,special,'n+',19,0,0,plain)
    assert set(free[0]).isdisjoint({*own,*donor,*special})
    assert set(clear[0]).isdisjoint({*own,*donor,*special,*plain})

def test_hit_is_exact_intersection_and_frozen_flags_precede_generation():
    free=[1,2,3,7];plain=[3,7,8]
    hit=sorted(set(free)&set(plain));assert hit==[3,7]
    flags=run169.freeze_flags(.04,{'feasible':True,'in_tolerance':True},{'feasible':True,'in_tolerance':True})
    assert flags=={'high_row':True,'paired_set':True}
    assert run169.freeze_flags(.0397,{'feasible':True,'in_tolerance':True},{'feasible':True,'in_tolerance':False})=={'high_row':False,'paired_set':False}
    record=run169.list_record(np.asarray([.2,.3,.5]),[],[0,2],[0,2,2],1.,1,True,0.,.5,True,False,set())
    assert record['plain_overlap']==2 and run169.mass_list_seed('n+',19,0,2,'free')!=run169.mass_list_seed('n+',19,0,2,'clear')

def _gate_fixture():
    rows=[{'arm':'n+','layer':19,'slot':0,'question_index':0,'bias_norm':1.,'steadiness':1.,'own_ban_ids':[1],'top_ids':list(range(200)),'top_values':[1.]*200},
          {'arm':'k4','layer':19,'slot':0,'question_index':0,'bias_norm':1.,'steadiness':1.,'own_ban_ids':[2],'top_ids':list(range(200)),'top_values':[1.]*200}]
    masses=[];answers=[]
    for r in rows:
      lists={b:{'ids':([1] if b=='own' and r['arm']=='n+' else [2] if b=='own' else [20] if b=='free' else [21] if b=='clear' else [22] if b=='plain50' else []),'in_tolerance':True,'feasible':True,'plain_overlap':1} for b in ('own','free','clear','hit','plain50')}
      m={**r,'lists':lists,'plain_answer_ids':[10],'special_ids':[0,9],'paired_set':True,'high_row':True};masses.append(m)
      for ban in ('none','own','free','clear','plain50'):
        looping=ban=='none'
        answers.append({'stage':'battery','arm':r['arm'],'layer':19,'slot':0,'question_index':0,'ban':ban,
          'banned_ids':[] if ban=='none' else lists[ban]['ids'],'token_ids':[99],
          'answer':'red blue green red blue green red blue green' if looping else 'A different coherent answer with distinct content words.'})
    for qi in range(8):
      answers.append({'stage':'no_push','arm':'plain','question_index':qi,'banned_ids':[],'token_ids':[99],'answer':'A clean baseline sentence.'})
      for ban in ('none','plain50'):
        answers.append({'stage':'no_push','arm':'plain','ban':ban,'question_index':qi,'banned_ids':[] if ban=='none' else [22],'token_ids':[99],'answer':'A clean baseline sentence.'})
    answers.append({'stage':'no_push','arm':'n+','ban':'free','layer':19,'slot':0,'question_index':0,'banned_ids':[20],'token_ids':[99],'answer':'A clean baseline sentence.'})
    norms=[{'arm':'n+','layer':19,'slot':0,'width':1000,'neuron':999,'attempt':0,'relative_error':0.}]
    return rows,masses,answers,norms

def test_gate_rules_can_pass_and_fail_on_fixtures():
    rows,masses,answers,norms=_gate_fixture()
    # The special-sized fixture keeps row and direction counts explicit.
    passing=run169.gate_rules(rows,answers,masses,norms,expected_rows=2,expected_norms=1,expected_nplus=1,expected_list_size=1)
    assert passing['gate1'] and passing['gate2'] and passing['gate3'] and passing['gate4']
    broken=[dict(a) for a in answers];next(a for a in broken if a.get('stage')=='no_push')['answer']='red blue green red blue green red blue green'
    failing=run169.gate_rules(rows,broken,masses,norms,expected_rows=2,expected_norms=1,expected_nplus=1,expected_list_size=1)
    assert not failing['gate2']
    missing=[dict(a) for a in answers if not (a.get('stage')=='no_push' and a.get('arm')=='n+')]
    failing_coverage=run169.gate_rules(rows,missing,masses,norms,expected_rows=2,expected_norms=1,expected_nplus=1,expected_list_size=1)
    assert not failing_coverage['gate1']
    leaked=[dict(a) for a in answers]
    free=next(a for a in leaked if a.get('stage')=='no_push' and a.get('arm')=='n+' and a.get('ban')=='free')
    free['token_ids']=[20]
    failing_ban=run169.gate_rules(rows,leaked,masses,norms,expected_rows=2,expected_norms=1,expected_nplus=1,expected_list_size=1)
    assert not failing_ban['gate4']

def test_no_push_path_records_zero_hook_installs(monkeypatch):
    class Common:
      def token_inputs(self,tok,q,n,device):return {'input_ids':torch.tensor([[1]])}
      def eos_ids(self,tok,model):return []
    class Model:
      def get_output_embeddings(self):return types.SimpleNamespace(weight=torch.zeros((16,2)))
      def generate(self,input_ids,**kwargs):return torch.tensor([[1,2]])
    fake=types.ModuleType('transformers');fake.LogitsProcessorList=list;monkeypatch.setitem(sys.modules,'transformers',fake)
    log=[];res=run169.generate_unpushed(Model(),Tok(),'prompt',[3],Common(),'cpu',call_log=log)
    assert log==[{'hook_installs':0,'ban_count':1}] and res['token_ids']==[2]

def test_unpushed_batch_applies_per_row_bans_without_hooks(monkeypatch):
    class Common:
      def token_inputs(self,tok,q,n,device):return {'input_ids':torch.tensor([[1]]*n)}
      def eos_ids(self,tok,model):return []
    class Model:
      def get_output_embeddings(self):return types.SimpleNamespace(weight=torch.zeros((16,2)))
      def generate(self,input_ids,**kwargs):return torch.tensor([[1,2],[1,3]])
    fake=types.ModuleType('transformers');fake.LogitsProcessorList=list;monkeypatch.setitem(sys.modules,'transformers',fake)
    log=[];res=run169.generate_unpushed_batch(Model(),Tok(),'Q',[[4],[5,6]],Common(),'cpu',call_log=log)
    assert len(res)==2 and [x['token_ids'] for x in res]==[[2],[3]]
    assert log==[{'hook_installs':0,'ban_count':2}]

def test_pushed_generation_installs_and_removes_activation_hook(monkeypatch):
    from part168 import run168
    from types import SimpleNamespace
    fake=types.ModuleType('transformers');fake.LogitsProcessorList=list;monkeypatch.setitem(sys.modules,'transformers',fake)
    calls=[]
    class Common:
      def token_inputs(self,tok,q,n,device):return {'input_ids':torch.tensor([[1]])}
      def eos_ids(self,tok,model):return []
      class AddRowsHook:
        def __init__(self,*args):calls.append('install')
        def begin_sequence(self):calls.append('begin')
        def remove(self):calls.append('remove')
    class Model:
      def get_output_embeddings(self):return SimpleNamespace(weight=torch.zeros((16,2)))
      def generate(self,input_ids,**kwargs):return torch.tensor([[1,2]])
    layers={19:SimpleNamespace(mlp=SimpleNamespace(down_proj=torch.nn.Linear(2,2,bias=False)))}
    result=run168.generate_batch(Model(),Tok(),layers,'Q',[{'arm':'n+','layer':19,'slot':0}],{(19,0):{'n+':torch.zeros(2)}},[[]],'cpu',Common(),1,169)
    assert result[0]['token_ids']==[2] and calls==['install','begin','remove']

def test_bootstrap_uses_registered_replicate_count():
    events=[{'arm':a,'ban':b,'layer':l,'slot':s,'question_index':q,'loop':bool((s+l+q)%2)}
      for a in run169.ARMS for b in ('none','own','free','clear','hit','plain50') for l in run169.LAYERS for s in range(run169.SLOTS) for q in run169.PHRASINGS]
    out=score169.bootstrap_intervals(events)
    assert out['own_relative_cut']['replicates']==10_000 and out['own_relative_cut']['seed']==169

def test_bootstrap_matches_slow_paired_slot_reference_and_is_narrow_when_constant():
    events=[]
    for arm in run169.ARMS:
      for layer in run169.LAYERS:
       for slot in range(run169.SLOTS):
        none=bool(slot%2);own=bool((slot+1)%2)
        for ban,value in (('none',none),('own',own)):
          events.append({'arm':arm,'ban':ban,'layer':layer,'slot':slot,'question_index':0,'loop':value})
    reps=500;seed=41;out=score169.bootstrap_intervals(events,reps=reps,seed=seed)
    rng=np.random.default_rng(seed);ix=rng.integers(0,run169.SLOTS,size=(reps,len(run169.LAYERS),run169.SLOTS))
    cuts=[]
    for rep in range(reps):
      base=alt=0
      for layer in run169.LAYERS:
        for slot in ix[rep,layer-run169.LAYERS[0]]:
          for arm in run169.ARMS:
            base+=int(bool(slot%2));alt+=int(bool((slot+1)%2))
      cuts.append(1-alt/base)
    assert np.allclose(out['own_relative_cut']['interval_95'],np.quantile(cuts,[.025,.975]),atol=1e-12)
    assert set(f'169.{n}' for n in range(1,9))|{'169.W1','169.W2'} <= set(out)
    constant=[]
    for arm in run169.ARMS:
      for layer in run169.LAYERS:
       for slot in range(run169.SLOTS):
        for ban,value in (('none',True),('own',False)):
          constant.append({'arm':arm,'ban':ban,'layer':layer,'slot':slot,'question_index':0,'loop':value})
    tight=score169.bootstrap_intervals(constant,reps=200,seed=seed)['own_relative_cut']['interval_95']
    assert tight==[1.0,1.0]

def test_hand_sample_ids_are_assigned_after_cross_ban_shuffle():
    battery=[]
    for ban in ('none','own','free','clear','hit','plain50'):
      for i in range(4):battery.append({'ban':ban,'arm':'n+','layer':19,'slot':i,'question_index':0,'answer':str(i)})
    _,key=score169.blind_hand_sample(battery,per_ban=4)
    bans=[x['ban'] for x in key]
    assert any(a!=b for a,b in zip(bans,bans[1:])) and bans!=sorted(bans)

def test_run_end_to_end_tiny_model_reaches_all_stages(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from part165 import run165
    from part166 import run166
    from part155 import run155
    from part168 import run168
    class TinyTok(Tok):
      pad_token_id=0
      def decode(self,ids,skip_special_tokens=False):return 'A short coherent synthetic answer with distinct words.'
      def encode(self,text,add_special_tokens=False):return [1000+i for i,_ in enumerate(text.split())]
    class Hook:
      def __init__(self,*a):self.calls=0
      def begin_sequence(self):self.calls+=1
      def remove(self):pass
    class Common:
      REVISION='tiny'
      AddRowsHook=Hook
      def find_layers(self,model):return model.layers
      def eos_ids(self,tok,model):return []
      def token_inputs(self,tok,q,n,device):return {'input_ids':torch.ones((n,1),dtype=torch.long,device=device)}
    class TinyModel(torch.nn.Module):
      def __init__(self):
        super().__init__();self.layers={19:SimpleNamespace(mlp=SimpleNamespace(down_proj=torch.nn.Linear(64,2,bias=False)))};self.head=torch.nn.Linear(2,2000,bias=False)
      def get_output_embeddings(self):return self.head
    tok=TinyTok();model=TinyModel();common=Common();calls={'pushed':0,'plain':0}
    monkeypatch.setattr(run169,'LAYERS',(19,));monkeypatch.setattr(run169,'SLOTS',1);monkeypatch.setattr(run169,'PHRASINGS',tuple(range(8)))
    monkeypatch.setattr(run169,'phrasing_items',lambda:[(i,{'kind':'maker','question':f'Q{i}'}) for i in range(8)])
    monkeypatch.setattr(run155,'_common',lambda:common)
    monkeypatch.setattr(run165,'build_directions',lambda w,l,s,median,cache:({'n+':torch.zeros(2),'k4':torch.zeros(2)},{'n+':{'relative_error':0.},'k4':{'relative_error':0.}}))
    monkeypatch.setattr(run166,'evaluation_positions',lambda plen,n:[0])
    monkeypatch.setattr(run166,'alignment_positions',lambda plen,n:[0])
    monkeypatch.setattr(run166,'forward_hidden',lambda model,ids,pos:torch.zeros((ids.shape[0],len(pos),2)))
    monkeypatch.setattr(run166,'plain_position_logits',lambda head,hidden:[torch.zeros(2000) for _ in range(hidden.shape[1])])
    monkeypatch.setattr(run166,'position_metrics',lambda hidden,head,plog,eos:(torch.zeros((hidden.shape[0],2000)),torch.ones(hidden.shape[0])))
    monkeypatch.setattr(run168,'generate_plain',lambda model,tok,q,common,device,n=40:([1],[1000+i for i in range(n)]))
    def fake_draw(q,own,matched,special,arm,layer,slot,qi,extra_excluded=(),**kwargs):
      ids=[1100+i for i in range(50)] if extra_excluded else [1000]+[1100+i for i in range(49)]
      return ids,1.,1,True,0.,.05,True,False
    monkeypatch.setattr(run169,'draw_mass_ids',fake_draw)
    monkeypatch.setattr(run169,'gate_rules',lambda *a,**k:{'gate6':True,'gate8':True,'gate8_stops':False})
    def unpushed_batch(model,tok,q,bans,common,device,max_new_tokens=80,call_log=None):
      calls['plain']+=len(bans)
      return [{'answer':'Clean coherent answer.','token_ids':[1500],'token_count':4,'ended_eos_before_80':False} for _ in bans]
    monkeypatch.setattr(run169,'generate_unpushed_batch',unpushed_batch)
    original_topk=torch.topk
    def topk_cpu(tensor,*a,**k):
      assert tensor.device.type=='cpu'
      return original_topk(tensor,*a,**k)
    monkeypatch.setattr(torch,'topk',topk_cpu)
    monkeypatch.setattr(run168,'generate_batch',lambda *a,**k:(calls.__setitem__('pushed',calls['pushed']+1) or [{'answer':'Clean coherent answer.','token_ids':[1501],'token_count':4,'ended_eos_before_80':False}]))
    out=tmp_path/'run';args=SimpleNamespace(out_dir=str(out),device='cpu',model='tiny',batch_size=64)
    result=run169.run(args,loader=lambda device:(tok,model))
    assert result['rows']==16 and result['masses']==16
    records=run169.read_jsonl(out/'answers.jsonl')
    assert any(x.get('stage')=='battery' for x in records) and calls['pushed']>0 and calls['plain']>0
    assert (out/'plain.json').is_file() and (out/'prelaunch_gates.json').is_file()
    keys=[run169.answer_key(x) for x in records];assert len(keys)==len(set(keys))
    # Remove a suffix, rerun, and confirm only missing answer keys are restored.
    with (out/'answers.jsonl').open('w') as f:f.writelines(json.dumps(x)+'\n' for x in records[:-3])
    run169.run(args,loader=lambda device:(tok,model))
    records=run169.read_jsonl(out/'answers.jsonl');keys=[run169.answer_key(x) for x in records]
    assert len(keys)==len(set(keys)) and (out/'tokens.json').is_file()

def test_copy_list_is_importable_and_a_broken_copy_fails(tmp_path):
    lines=(ROOT/'part169/steps.txt').read_text().splitlines()
    launch=next(x for x in lines if x.startswith('python3 tools/launch_arc.py'))
    words=shlex.split(launch);files=words[words.index('--copy')+1:]
    tree=tmp_path/'copy'
    for rel in files:
      src=ROOT/rel;assert src.is_file(),rel
      dst=tree/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    py='/home/elisepc/neuron-probe/.venv/bin/python'
    for script in ('part169/run169.py','part169/score169.py'):
      proc=subprocess.run([py,'-I','-B',script,'--help'],cwd=tree,capture_output=True,text=True)
      assert proc.returncode==0,proc.stderr
    target=tree/'part169/run169.py';src=target.read_text();target.write_text(src.replace('def special_ids(tok):','def special_ids(tok):\n    return set()  # deliberate break in the temporary copy',1))
    probe="import sys;sys.path.insert(0,'.');from part169 import run169;assert 12 in run169.special_ids(type('T',(),{'all_special_ids':[],'eos_token_id':None,'get_added_vocab':lambda self:{'added':12}})())"
    proc=subprocess.run([py,'-I','-B','-c',probe],cwd=tree,capture_output=True,text=True)
    assert proc.returncode!=0


def test_block_log_records_moved_steps_and_ranks_up_to_the_answer_length():
    import torch
    log=[]
    proc=run169.logging_processor(log)(torch.tensor([[False,True,False],[False,False,False]]))
    scores=torch.tensor([[1.,3.,2.],[3.,2.,1.]])
    out=proc(None,scores)
    assert out[0,1]==-float('inf') and scores[0,1]==3.
    proc(None,scores)
    answers=[{'token_ids':[1]},{'token_ids':[0,0]}]
    run169.attach_blocks(answers,log)
    assert answers[0]['blocked_steps']==[0] and answers[0]['blocked_ranks']==[1]
    assert answers[1]['blocked_steps']==[] and answers[1]['blocked_ranks']==[]


def test_with_block_log_restores_the_processor():
    original=run169.run168.RowBanProcessor
    def boom(): raise RuntimeError('x')
    try: run169.with_block_log(boom)
    except RuntimeError: pass
    assert run169.run168.RowBanProcessor is original
