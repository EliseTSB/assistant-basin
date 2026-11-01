import json,sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from part167 import run167,score167

class Tok:
    all_special_ids=[0,9]
    eos_token_id=9
    def decode(self,ids,skip_special_tokens=False):return ' '.join(map(str,ids))

def test_seed_wrapper_is_fresh_and_deterministic():
    w=torch.ones((5,8),dtype=torch.float32)
    a=run167.seeded_raw(w,19,2,torch);b=run167.seeded_raw(w,19,2,torch)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    old=__import__('part165.run165',fromlist=['_seeded_raw'])._seeded_raw(w,19,2,torch)
    assert not torch.equal(a[0],old[0]) or not torch.equal(a[1],old[1])

def test_row_order_and_real_shaped_fields_every_arm():
    rows=run167.all_rows({l:32 for l in run167.LAYERS})
    assert rows[:3]==[{'arm':'n+','layer':19,'slot':0,'k':1,'question_index':8},{'arm':'k4','layer':19,'slot':0,'k':4,'question_index':8},{'arm':'full','layer':19,'slot':0,'k':32,'question_index':8}]
    for arm in run167.ARMS:
        r={'arm':arm,'layer':19,'slot':3,'k':1 if arm=='n+' else 4 if arm=='k4' else 32}
        rec=run167.make_row_record(r,8,torch.tensor([1.,2.,0.]),torch.tensor(2.),[1,2],[.3,.2])
        assert set(('sharpness','steadiness','bias_norm','top_ids','top_values'))<=rec.keys()
        assert rec['arm']==arm and rec['question_index']==8

def test_special_removal_keeps_first_50_and_matched_slot():
    r={'top_ids':list(range(60))}; got=run167.build_ban_ids(r,None,Tok()); assert got==[x for x in range(60) if x not in (0,9)][:50]
    assert len(got)==50 and len(set(got))==50
    rows=[]
    for slot in range(12): rows.append({'arm':'k4','layer':20,'slot':slot,'question_index':8,'top_ids':list(range(slot+10,slot+70))})
    got=run167.ban_lists([rows[11]],rows,Tok())[0]
    assert got['matched']==list(range(10,60)) # slot 11 wraps to slot 0
    assert len(got['own'])==len(got['matched'])==50

def test_stage1b_mass_computation_fake_head():
    hidden=torch.tensor([[[1.,0.],[0.,1.]]])
    head=torch.nn.Linear(2,4,bias=False)
    with torch.no_grad():head.weight.copy_(torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]))
    own,matched,allowed=run167.list_masses(hidden,head,[[0,1]],[[2,3]])
    probs=torch.softmax(head(hidden).float(),dim=-1)
    assert torch.allclose(own,probs[:,:,[0,1]].sum(-1).mean(-1))
    assert torch.allclose(matched,probs[:,:,[2,3]].sum(-1).mean(-1))
    assert allowed==2

def test_processor_masks_per_row():
    mask=run167.make_mask([[1],[2,3]],5,'cpu',torch)
    scores=torch.zeros((2,5)); out=run167.RowBanProcessor(mask)(torch.zeros((2,1),dtype=torch.long),scores)
    assert torch.isneginf(out[0,1]) and torch.isneginf(out[1,2]) and torch.isneginf(out[1,3])
    assert torch.isfinite(out[0,2]) and torch.isfinite(out[1,1])

def test_resume_keys_both_stages():
    a={'stage':'battery','arm':'n+','ban':'own','layer':19,'slot':0,'question_index':8}
    assert run167.answer_key(a)==run167.answer_key(dict(a))
    assert run167.pending_answers([a],{run167.answer_key({'stage':'battery',**a})})==[]
    row={'arm':'full','layer':26,'slot':11,'question_index':15}
    assert run167.pending_rows([row],{run167.row_key(row)},15)==[]

def test_loop_fixture_by_hand_and_gate4_detects_deliberate_break(tmp_path):
    assert score167.loop_phrase('red blue green red blue green red blue green')==('red','blue','green')
    rows=[]
    for r in run167.all_rows({l:1000 for l in run167.LAYERS}):
        rows.append({**r,'sharpness':1.,'steadiness':1.,'bias_norm':1.,'top_ids':list(range(1,201)),'top_values':[float(200-i) for i in range(200)],'own_ban_ids':list(range(1,51)),'rank_51_200_ids':list(range(51,201))})
    target=next(r for r in rows if r['arm']=='n+' and r['layer']==19 and r['slot']==0 and r['question_index']==8)
    ans={'stage':'battery','arm':'n+','ban':'own','layer':19,'slot':0,'question_index':8,'answer':'a useful answer','token_ids':[2],'token_count':3,'banned_ids':target['own_ban_ids']}
    no={'stage':'no_push','arm':'plain','question_index':8,'answer':'a coherent plain answer','token_ids':[4],'token_count':4}
    (tmp_path/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    (tmp_path/'answers.jsonl').write_text(json.dumps(ans)+'\n'+json.dumps(no)+'\n',encoding='utf-8')
    (tmp_path/'plain.json').write_text(json.dumps({'answers':[]}),encoding='utf-8')
    (tmp_path/'tokens.json').write_text('{}',encoding='utf-8')
    (tmp_path/'norms.json').write_text(json.dumps({'directions':[]}),encoding='utf-8')
    result=score167.score(tmp_path/'rows.jsonl',tmp_path/'plain.json',tmp_path/'tokens.json',tmp_path/'answers.jsonl',tmp_path/'out')
    assert result['gates']['gate4'] is False
    assert len(rows)==score167.EXPECTED_ROWS
    assert (tmp_path/'out'/'score.json').is_file() and (tmp_path/'out'/'tables.md').is_file()

def test_scorer_full_fixture_writes_outputs_end_to_end(tmp_path,monkeypatch):
    rows=[]
    for r in run167.all_rows({l:1000 for l in run167.LAYERS}):
        rows.append({**r,'sharpness':1.,'steadiness':1.,'bias_norm':1.,'top_ids':list(range(1,201)),'top_values':[float(200-i) for i in range(200)],'own_ban_ids':list(range(1,51)),'rank_51_200_ids':list(range(51,201))})
    answers=[]
    broken_left=140
    for r in rows:
        for ban in ('none','own','matched'):
            answer='alpha beta gamma alpha beta gamma alpha beta gamma' if r['arm']=='n+' and ban=='none' and broken_left>0 else 'This is a coherent answer with several distinct words.'
            if r['arm']=='n+' and ban=='none' and broken_left>0:broken_left-=1
            banned=[] if ban=='none' else list(range(1,51)) if ban=='own' else list(range(51,101))
            answers.append({'stage':'battery','arm':r['arm'],'layer':r['layer'],'slot':r['slot'],'question_index':r['question_index'],'ban':ban,'answer':answer,'token_ids':[999],'token_count':4,'banned_ids':banned})
    answers.extend({'stage':'no_push','arm':'plain','question_index':qi,'answer':'This is a coherent answer with several distinct words.','token_ids':[999],'token_count':4} for qi in range(8,16))
    rowpath=tmp_path/'rows.jsonl'; ans_path=tmp_path/'answers.jsonl'
    rowpath.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    ans_path.write_text(''.join(json.dumps(r)+'\n' for r in answers),encoding='utf-8')
    (tmp_path/'plain.json').write_text(json.dumps({'answers':[]}),encoding='utf-8')
    (tmp_path/'tokens.json').write_text('{}',encoding='utf-8')
    (tmp_path/'norms.json').write_text(json.dumps({'directions':[{'relative_error':0.} for _ in range(8*12*3)]}),encoding='utf-8')
    real=score167.slot_bootstrap
    monkeypatch.setattr(score167,'slot_bootstrap',lambda events,metric,seed=167,reps=10000:real(events,metric,seed,reps=2))
    result=score167.score(rowpath,tmp_path/'plain.json',tmp_path/'tokens.json',ans_path,tmp_path/'out')
    assert result['readings_stopped'] is False
    assert set(result['predictions'])=={f'167.{i}' for i in range(1,8)}
    assert (tmp_path/'out'/'score.json').is_file() and (tmp_path/'out'/'tables.md').is_file()
    hand=[json.loads(x) for x in (tmp_path/'out'/'hand_read.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(hand)==90 and all('ban' not in x and 'arm' not in x for x in hand)
    key=[json.loads(x) for x in (tmp_path/'out'/'hand_read_key.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(key)==90 and {x['ban'] for x in key}=={'none','own','matched'}

def test_pilot_clock_decision():
    assert run167.pilot_rate_should_stop(3,10,13) is True
    assert run167.pilot_rate_should_stop(3,10,11) is False

def test_scorer_finite_loop_and_prediction_helpers():
    assert score167.score167 if hasattr(score167,'score167') else True
    assert score167.loop_phrase('one two three one two three one two three')==('one','two','three')

@pytest.mark.parametrize('pid,yes,no',[
 ('167.1',{'ratio':5},{'ratio':4.99}),('167.2',{'nplus':.2,'full':.1},{'nplus':.19,'full':.1}),
 ('167.3',{'own':.3},{'own':.29}),('167.4',{'own_rate':.15,'matched_rate':.20},{'own_rate':.16,'matched_rate':.20}),
 ('167.5',{'matched':.149},{'matched':-.151}),('167.6',{'full':.19,'own':.4},{'full':.2,'own':.4}),
 ('167.7',{'migration':1/3,'baseline':.2},{'migration':1/3,'baseline':1/3})])
def test_each_registered_prediction_passes_and_fails(pid,yes,no):
    assert score167.passes_prediction(pid,**yes)
    assert not score167.passes_prediction(pid,**no)

def test_transition_migration_induced_reentry_and_baseline():
    tokens={'1':'alpha','2':'beta','3':'gamma','4':'delta','5':'epsilon','6':'zeta'}
    row={'top_ids':[1,2,3]}
    old={'answer':'alpha beta gamma alpha beta gamma alpha beta gamma'}
    migrated={'answer':'delta epsilon zeta delta epsilon zeta delta epsilon zeta'}
    result=score167.transition_test(old,migrated,row,[4,5,6],tokens,set())
    assert result['qualifies'] and result['migrated'] and not result['reentry']
    induced=score167.transition_test({'answer':'a useful answer'},migrated,row,[4,5,6],tokens,set())
    assert induced['induced'] and not induced['qualifies']
    reentry=score167.transition_test(old,{'answer':'alpha delta epsilon alpha delta epsilon alpha delta epsilon'},row,[4,5,6],tokens,set())
    assert reentry['qualifies'] and reentry['reentry']
    assert score167.transition_baseline({'delta'},[[4,5],[1,2]],tokens)==.5

def test_gate4_checks_matched_token_ids(tmp_path):
    rows=[]
    for r in run167.all_rows({l:1000 for l in run167.LAYERS}):
        rows.append({**r,'sharpness':1.,'steadiness':1.,'bias_norm':1.,'top_ids':list(range(1,201)),'top_values':[float(200-i) for i in range(200)],'own_ban_ids':list(range(1,51))})
    target=next(r for r in rows if r['arm']=='n+' and r['layer']==19 and r['slot']==0 and r['question_index']==8)
    ans={'stage':'battery','arm':'n+','ban':'matched','layer':19,'slot':0,'question_index':8,'answer':'useful answer','token_ids':[101],'banned_ids':[101]}
    files={'rows.jsonl':''.join(json.dumps(r)+'\n' for r in rows),'answers.jsonl':json.dumps(ans)+'\n','plain.json':json.dumps({'answers':[]}), 'tokens.json':'{}'}
    for name,data in files.items():(tmp_path/name).write_text(data,encoding='utf-8')
    (tmp_path/'norms.json').write_text(json.dumps({'directions':[]}),encoding='utf-8')
    result=score167.score(tmp_path/'rows.jsonl',tmp_path/'plain.json',tmp_path/'tokens.json',tmp_path/'answers.jsonl',tmp_path/'out')
    assert result['gates']['gate4'] is False


def test_row_base_and_ban_lists_on_real_shaped_work_rows():
    widths={l:9216 for l in run167.LAYERS}
    base=run167.row_base(widths)
    assert len(base)==3*8*12
    class Tok:
        all_special_ids=[0]; eos_token_id=1
    stored=[{**r,'question_index':8,'top_ids':[2+i+100*r['slot'] for i in range(60)]} for r in base]
    work=[{**r,'question_index':8,'ban':b} for r in base[:6] for b in run167.BANS]
    lists=run167.ban_lists(work,stored,Tok())
    assert len(lists[1]['own'])==50 and len(lists[1]['matched'])==50
    assert lists[1]['own'][0]==2+100*work[1]['slot']
    assert lists[1]['matched'][0]==2+100*((work[1]['slot']+1)%12)
    assert lists[0]['none']==[]

