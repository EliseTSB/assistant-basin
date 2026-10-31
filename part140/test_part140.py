#!/usr/bin/env python3
"""Small CPU fixtures for Part 140's hook, pairing and scorer contracts."""
import importlib.util,json,shutil,sys
from pathlib import Path
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run140

def test_do_nothing_hook_matches_plain_bitwise():
    layer=nn.Module(); layer.mlp=nn.Module(); layer.mlp.down_proj=nn.Linear(3,3,bias=False)
    x=torch.randn(2,1,3); plain=layer.mlp.down_proj(x)
    hk=run140.DeliveryHook(layer.mlp.down_proj,torch.zeros((2,3)))
    try: hooked=layer.mlp.down_proj(x)
    finally: hk.remove()
    assert torch.equal(plain,hooked)

def test_gate2_picks_the_neuron_row_when_ids_repeat():
    rows=[{'id':'q','unit':'random','question_index':0,'answer':'c d'},{'id':'q','unit':'neuron','question_index':0,'answer':'a b'},{'id':'q','unit':'random','question_index':0,'answer':'e f'},{'id':'q','unit':'neuron','question_index':1,'answer':'g h'}]
    assert run140.part112_first_answers(rows)=={'q':'a b'}

def test_gate2_compares_the_opening_and_counts_missing_rows():
    class Tok:
        def encode(self,text,add_special_tokens=False): return text.split()
        def decode(self,tokens,skip_special_tokens=True): return ' '.join(tokens)
    same=' '.join(str(i) for i in range(run140.GATE2_TOKENS))
    old={'a':same+' tail','b':same,'c':'different '+same}
    observed={'a':same+' other','b':same,'c':same}
    r=run140.gate2_result(['a','b','c','d'],old,observed,Tok())
    assert (r['matched'],r['missing'],r['passed'])==(2,1,False)
    assert run140.gate2_result(['a','b'],old,observed,Tok())['passed'] is True
    assert run140.gate2_result(['a'],old,{},Tok())['passed'] is False

def test_release_after_n_is_per_row_and_stops_pushing():
    module=nn.Linear(2,2,bias=False); module.weight.data.copy_(torch.eye(2)); push=torch.tensor([[1.,0.],[0.,2.]])
    hk=run140.DeliveryHook(module,push,release_after=2)
    try:
        module(torch.zeros(2,2,2)) # prompt pass
        first=module(torch.zeros(2,1,2)); second=module(torch.zeros(2,1,2)); third=module(torch.zeros(2,1,2))
    finally: hk.remove()
    assert torch.equal(first[:,0],push) and torch.equal(second[:,0],push)
    assert torch.equal(third[:,0],torch.zeros_like(push))
    # Independent state: a one-row generation starts with its own full release window.
    hk=run140.DeliveryHook(module,push[1:2],release_after=1)
    try: module(torch.zeros(1,2,2)); row=module(torch.zeros(1,1,2))
    finally: hk.remove()
    assert torch.equal(row[0,0],push[1])

def test_paired_seed_set_is_shared_across_units():
    assert run140.paired_seeds('unit-a')==run140.paired_seeds('unit-b')==(140000,140001,140002)
    assert run140.paired_seeds('unit-a',arm='A')==(140000,)

def test_norm_preserving_hook_restores_pushed_norm_and_leaves_other_rows_untouched():
    module=nn.Linear(3,3,bias=False); module.weight.data.copy_(torch.eye(3))
    pushes=torch.tensor([[1.,2.,0.],[0.,0.,0.]])
    original=torch.tensor([[[3.,4.,0.]],[[2.,-1.,2.]]])
    hk=run140.DeliveryHook(module,pushes,norm_preserve=True)
    try:
        module(torch.zeros(2,2,3)) # prompt pass
        result=module(original)[:,0]
    finally: hk.remove()
    assert torch.allclose(result[0].norm(),original[0,0].norm(),atol=1e-6)
    assert torch.equal(result[1],original[1,0])

def test_repetition_requires_three_contiguous_phrase_repeats():
    spec=importlib.util.spec_from_file_location('score140_repeat',HERE/'score140.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert not mod.repeat('I am Robin. I enjoy a clear day. I enjoy music. I enjoy the view.')
    assert mod.repeat('I am Robin. Blue skies, blue skies, blue skies.')

def test_scorer_fixture_and_broken_copy_fails_expectation(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('score140_fixture',HERE/'score140.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    answers=[]; labels=[]; luna=[]; units=[]
    for setname,kind in [('P','concept'),('C','random')]:
      for arm in ('A','B','C','D','E','F'):
       for i in range(24):
        uid=f'{setname}{i}'; units.append({'id':uid,'set':setname,'set_kind':kind,'layer':10,'neuron':i})
        source=f'{arm}|{setname}|{uid}|Who are you?|140000'
        answers.append({'arm':arm,'set':setname,'unit':uid,'question':'Who are you?','seed':140000,'plain':False,'answer':'I am a person.'})
        label={'id':str(len(labels)),'source_id':source,'kind':'human','coherent':True}; labels.append(label); luna.append(dict(label))
       answers.append({'arm':arm,'set':setname,'unit':f'{setname}0','question':'Who are you?','seed':140000,'plain':True,'answer':'I am a person.'})
    for name,data in [('a.jsonl',answers),('l.jsonl',labels),('luna.jsonl',luna)]: (tmp_path/name).write_text(''.join(json.dumps(r)+'\n' for r in data),encoding='utf-8')
    (tmp_path/'u.json').write_text(json.dumps(units),encoding='utf-8')
    old=[{'id':f'P{i}','question':'Who are you?','answer':'I am a person.'} for i in range(24)]
    (tmp_path/'old.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in old),encoding='utf-8')
    (tmp_path/'gate1-result.json').write_text(json.dumps({'passed':True}),encoding='utf-8')
    for name in ('gate-ds-result.json','gate-luna-result.json'): (tmp_path/name).write_text(json.dumps({'passed':True}),encoding='utf-8')
    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls,*args,**kwargs): return cls()
        def encode(self,text,add_special_tokens=False): return text.split()
        def decode(self,tokens,skip_special_tokens=True): return ' '.join(tokens)
    monkeypatch.setitem(sys.modules,'transformers',type('T',(),{'AutoTokenizer':FakeTokenizer}))
    out=tmp_path/'score.json'; monkeypatch.setattr(sys,'argv',['score140.py','--answers',str(tmp_path/'a.jsonl'),'--labels',str(tmp_path/'l.jsonl'),'--luna-labels',str(tmp_path/'luna.jsonl'),'--units',str(tmp_path/'u.json'),'--part112-answers',str(tmp_path/'old.jsonl'),'--out',str(out)])
    mod.main(); result=json.loads(out.read_text(encoding='utf-8'))
    assert result['predictions']['140.1']['difference']==0 and result['table_per_arm_set']['A/P']['joint']['rate']==1
    # Mutate a private scorer copy; the same fixture assertion must detect the deliberate break.
    broken=tmp_path/'broken_score140.py'; shutil.copyfile(HERE/'score140.py',broken)
    source=broken.read_text(encoding='utf-8').replace("def joint(r): return bool(r.get('coherent')) and r.get('kind') not in ('unclear','still Qwen or an assistant')", "def joint(r): return False")
    broken.write_text(source,encoding='utf-8')
    badspec=importlib.util.spec_from_file_location('broken_score140',broken); bad=importlib.util.module_from_spec(badspec); badspec.loader.exec_module(bad)
    monkeypatch.setattr(sys,'argv',['score140.py','--answers',str(tmp_path/'a.jsonl'),'--labels',str(tmp_path/'l.jsonl'),'--luna-labels',str(tmp_path/'luna.jsonl'),'--units',str(tmp_path/'u.json'),'--part112-answers',str(tmp_path/'old.jsonl'),'--out',str(tmp_path/'bad.json')])
    bad.main(); broken_result=json.loads((tmp_path/'bad.json').read_text(encoding='utf-8'))
    assert broken_result['table_per_arm_set']['A/P']['joint']['rate'] != 1
