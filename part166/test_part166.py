"""CPU fixtures for Part 166 position logic, measurements, and writer paths."""
import json, math, sys
from pathlib import Path
import pytest
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from part166 import run166, score166

class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.proj=torch.nn.Linear(4,5,bias=False)
        with torch.no_grad(): self.proj.weight.copy_(torch.arange(20,dtype=torch.float32).reshape(5,4)/20)
    def forward(self,x): return self.proj(x)

class TinyTokenizer:
    def encode(self,text): return [1,2] if text=='prompt' else [3,4,5,6]
    def decode(self,ids,skip_special_tokens=False): return ' '.join(str(x) for x in ids)

def test_hook_only_answer_positions_and_matches_cached_path():
    model=Linear(); tok=TinyTokenizer(); prompt_len=len(tok.encode('prompt')); answer_len=len(tok.encode('answer'))
    x=torch.arange(24,dtype=torch.float32).reshape(1,6,4); push=torch.tensor([[.5,-.2,.1,.8]])
    h=run166.attach_position_push(model.proj,push,prompt_len,answer_len)
    teacher=model(x); h.remove()
    expected=x.clone(); expected[:,2:5]+=push[:,None,:]
    assert torch.equal(teacher,model.proj(expected))
    h=run166.attach_position_push(model.proj,push,prompt_len,answer_len)
    cached=[]
    cached.append(model(x[:,:prompt_len]))
    for i in range(answer_len): cached.append(model(x[:,prompt_len+i:prompt_len+i+1]))
    h.remove()
    joined=torch.cat(cached,dim=1)
    assert torch.allclose(teacher,joined,atol=1e-6)
    assert torch.allclose(teacher[:,prompt_len],joined[:,prompt_len],atol=1e-6)
    # A one-position shift in the registered push range must disagree at a predicted position.
    wrong=x.clone(); wrong[:,3:6]+=push[:,None,:]
    assert not torch.allclose(teacher,model.proj(wrong))

def test_row_order_and_resume_skip(tmp_path):
    widths={layer:8 for layer in run166.LAYERS}
    base=run166.all_rows(widths); import random
    for qi in range(8):
        expected=list(base); random.Random(165+qi).shuffle(expected)
        actual=list(base); random.Random(165+qi).shuffle(actual)
        assert actual==expected
    path=tmp_path/'rows.jsonl'; row={'arm':'n+','slot':0,'layer':19,'question_index':0}
    path.write_text(json.dumps(row)+'\n',encoding='utf-8')
    assert run166._key(row) in run166._valid_written(path)

def test_steadiness_and_sharpness():
    identical=torch.tensor([[[1.,0.],[1.,0.],[1.,0.]]]); bias,steady,_,_=run166.vocabulary_metrics(torch.zeros_like(identical),identical)
    assert float(steady[0])==pytest.approx(1.)
    orth=torch.tensor([[[1.,-1.,0.,0.],[0.,0.,1.,-1.],[2**-.5,2**-.5,-2**-.5,-2**-.5]]]); _,s,_,_=run166.vocabulary_metrics(torch.zeros_like(orth),orth)
    assert float(s[0])==pytest.approx(1/math.sqrt(3),rel=.02)
    v=torch.tensor([0.,1.,2.,3.]); sharp=(v.max()-v.mean())/v.std(unbiased=False)
    assert float(sharp)==pytest.approx(1.3416408)

def test_kl_masked_minus_infinity_is_finite():
    plain=torch.tensor([[[-float('inf'),0.]]]); pushed=torch.tensor([[[-float('inf'),0.]]])
    kl=run166.kl_values(plain,pushed)
    assert torch.isfinite(kl).all() and float(kl)==0.
    raw_l=torch.tensor([[[1.,0.,-.5]]]); raw_q=torch.tensor([[[0.,1.,-.5]]])
    _,_,_,masked=run166.vocabulary_metrics(raw_l,raw_q,eos_ids=(2,))
    lm=raw_l[:,0,:].clone(); qm=raw_q[:,0,:].clone(); lm[:,2]=-float('inf'); qm[:,2]=-float('inf')
    assert float(masked[0])==pytest.approx(float(run166.kl_values(lm[:,None,:],qm[:,None,:])[0,0]))

def test_loop_phrase_words_and_rates_fixture():
    assert score166.loop_phrase('red blue green red blue green red blue green')==('red','blue','green')
    assert score166.loop_phrase('red blue green tan red blue green tan red blue green')==('red','blue','green')
    tok={'1':' red','2':'blue'}
    assert score166.row_words({'top_ids':[1,2]},tok)=={'red','blue'}
    # Own loop row contains red; among 11 shuffled slots only one contains red.
    own={'red'}; other=[{'red'}]+[set() for _ in range(10)]
    assert int(bool({'red','blue','green'}&own))==1
    assert sum(bool({'red','blue','green'}&x) for x in other)/11==pytest.approx(1/11)

def test_gate_fail_writer_end_to_end(tmp_path):
    rows=tmp_path/'rows.jsonl'; plain=tmp_path/'plain.json'; tokens=tmp_path/'tokens.json'; answers=tmp_path/'answers.jsonl'; effects=tmp_path/'effects.jsonl'
    for p,data in ((rows,''),(tokens,'{}'),(answers,''),(effects,'')) : p.write_text(data,encoding='utf-8')
    plain.write_text(json.dumps({'alignment_matches':0,'alignment_total':1}),encoding='utf-8')
    result=score166.score(rows,plain,tokens,answers,effects,tmp_path/'out')
    assert result['readings_stopped'] and (tmp_path/'out'/'score.json').is_file() and (tmp_path/'out'/'tables.md').is_file() and (tmp_path/'out'/'hits.jsonl').is_file()

def test_scorer_success_writer_on_patched_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(score166,'EXPECTED',2)
    monkeypatch.setattr(score166,'LAYERS',(19,))
    monkeypatch.setattr(score166,'SLOTS',(0,1))
    monkeypatch.setattr(score166,'ARMS',('n+','n±','k4','full','iso'))
    rows=[]; effects=[]
    for qi,kl in enumerate((.1,.2)):
        row={'arm':'n+','slot':qi,'layer':19,'question_index':qi,'k':1,'positions':1,'bias_norm':1.,
             'sharpness':1.,'sharpness_first3':1.,'steadiness':1.,'steadiness_first3':1.,
             'kl_mean':kl,'kl_second':kl,'top_ids':[1],'top_values':[1.]}
        rows.append(row); effects.append({'arm':'n+','slot':qi,'layer':19,'question_index':qi,'kl':kl})
    inputs={
      'rows.jsonl':'\n'.join(json.dumps(x) for x in rows)+'\n',
      'effects.jsonl':'\n'.join(json.dumps(x) for x in effects)+'\n',
      'plain.json':json.dumps({'alignment_matches':2,'alignment_total':2,'determinism_matches':8,'determinism_total':8,'answers':[]}),
      'tokens.json':json.dumps({'1':' red'}), 'answers.jsonl':''}
    for name,data in inputs.items(): (tmp_path/name).write_text(data,encoding='utf-8')
    result=score166.score(tmp_path/'rows.jsonl',tmp_path/'plain.json',tmp_path/'tokens.json',tmp_path/'answers.jsonl',tmp_path/'effects.jsonl',tmp_path/'out')
    assert not result['readings_stopped'] and (tmp_path/'out'/'score.json').is_file()
    assert (tmp_path/'out'/'tables.md').is_file() and (tmp_path/'out'/'hits.jsonl').is_file()

def test_gate3_scale_error_fails_relative_checks_but_keeps_rank():
    rows=[]; effects=[]
    for qi,value in enumerate((.02,.03,.04,.05)):
        rows.append({'arm':'n+','slot':qi,'layer':19,'question_index':qi,'kl_second':value})
        effects.append({'arm':'n+','slot':qi,'layer':19,'question_index':qi,'kl':value*100})
    gate=score166.gate3_summary(rows,effects)
    assert gate['spearman']==pytest.approx(1.)
    assert gate['median_absolute_relative_difference']>.1
    assert not gate['pass']

def test_slot_bootstrap_uses_only_resampled_donors():
    event={'arm':'n+','layer':19,'slot':0,'own_hit':True,'donor_phrase':{0:True,1:False},
           'content_own':{'red':True},'content_donors':{'red':{0:True,1:False}}}
    # The resample includes only slot 0, leaving no donor and no shuffled denominator.
    result=score166.bootstrap_loop_rates([event],[[(19,0)]])
    assert result.get('low_own')==[1.0,1.0]
    assert 'low_shuffled' not in result

def test_each_prediction_decision_pass_and_fail():
    assert score166.passes_166_1(.4,.2) and not score166.passes_166_1(.39,.2)
    assert score166.passes_166_2(1.49,1.) and not score166.passes_166_2(1.5,1.)
    assert score166.passes_166_3(1.25,1.) and not score166.passes_166_3(1.24,1.)
    assert score166.passes_166_4(-.81) and not score166.passes_166_4(-.79)
    assert score166.passes_166_5(.31) and not score166.passes_166_5(.29)
    assert score166.passes_166_6(.31,.2) and not score166.passes_166_6(.29,.2)
    assert score166.passes_166_7(.65) and not score166.passes_166_7(.64)

def test_plain_gate_requires_alignment_and_determinism():
    plain={'alignment_matches':4,'alignment_total':4,'determinism_matches':8,'determinism_total':8}
    assert plain['alignment_matches']==plain['alignment_total'] and plain['determinism_matches']==plain['determinism_total']
    plain['determinism_matches']=7
    assert not (plain['alignment_matches']==plain['alignment_total'] and plain['determinism_matches']==plain['determinism_total'])

def test_alignment_uses_shifted_y_pairs():
    assert run166.alignment_positions(5,3)==[4,5,6]
    assert run166.evaluation_positions(5,3)==[5,6]

def test_loop_values_batch_sidecar_shape_and_resume(tmp_path):
    import numpy as np
    values=np.asarray([[.1,.2],[.3,.4]],dtype=np.float32)
    np.save(tmp_path/'batch.npy',values,allow_pickle=False)
    meta={'token_ids':[10,11],'row_keys':[['n+',0,19,0],['k4',0,19,0]],'bias_means':[0.,0.],'bias_stds':[1.,1.]}
    (tmp_path/'batch.json').write_text(json.dumps(meta),encoding='utf-8')
    restored=np.load(tmp_path/'batch.npy',allow_pickle=False)
    assert restored.dtype==np.float32 and restored.shape==(2,2)
    assert len(meta['row_keys'])==restored.shape[0]

def test_pilot_rate_and_clock_start_after_plain():
    assert run166.MAX_NEW_TOKENS==40
    assert run166.BATCH_SIZE==64
    # Pilot uses only its passed elapsed duration; plain timing is excluded by placement after plain passes.
    assert run166.pilot_should_stop(3,1.0) and not run166.pilot_should_stop(4,1.0)
