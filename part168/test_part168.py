import json, sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part168 import run168, score168

class Tok:
    all_special_ids=[0,9]
    eos_token_id=9
    def decode(self,ids,skip_special_tokens=False): return ' '.join(map(str,ids))

def test_seed_draw_order_and_fresh_neurons_at_registered_and_forced_widths():
    width=9216; g=torch.Generator(device='cpu'); g.manual_seed(168000+100*19+2)
    expected=torch.randperm(width,generator=g,device='cpu')
    raw=run168.seeded_raw(torch.ones((4,width)),19,2,torch)
    assert torch.equal(raw[0],expected)
    used={run168.neuron_for_seed(width,19,s,b,torch) for b in (165000,167000) for s in range(12)}
    neuron,attempt=run168.fresh_neuron(width,19,2,used,torch)
    assert attempt==0 and neuron not in used
    small=7; first=run168.neuron_for_seed(small,20,1,168000,torch)
    neuron,attempt=run168.fresh_neuron(small,20,1,{first},torch)
    assert attempt>=1 and neuron not in {first}

def test_phrasings_rows_and_all_arm_fields():
    assert tuple(q for q,_ in run168.phrasing_items())==run168.PHRASINGS==(8,9,10,11,12,14,15)
    rows=run168.all_rows({l:9216 for l in run168.LAYERS})
    assert len(rows)==2016
    for arm in run168.ARMS:
        row={'arm':arm,'layer':19,'slot':3,'k':1 if arm=='n+' else 4 if arm=='k4' else 9216}
        rec=run168.make_row_record(row,8,torch.tensor([1.,2.,0.]),torch.tensor(2.),[1,2],[.3,.2])
        assert rec['question_index']==8 and len(rec['top_ids'])==2

def test_mass_probability_writer_is_cpu_float32_and_sampling_is_stable():
    hidden=torch.tensor([[[1.,0.],[0.,1.]]])
    head=torch.nn.Linear(2,64,bias=False)
    with torch.no_grad(): head.weight.copy_(torch.arange(128,dtype=torch.float32).reshape(64,2)/100)
    q=run168.list_masses(hidden,head)
    assert q.shape==(1,64) and q.device.type=='cpu' and q.dtype==torch.float32
    a=run168.draw_mass_ids(q[0].numpy(),[1,2],[3,4],{0,9},'n+',19,0,8)
    b=run168.draw_mass_ids(q[0].numpy(),[1,2],[3,4],{0,9},'n+',19,0,8)
    assert a==b and len(a[0])==50 and len(set(a[0]))==50
    assert not set(a[0])&{0,1,2,3,4,9}

def test_ban_lists_reuse_persisted_mass_list_and_special_filter():
    widths={l:9216 for l in run168.LAYERS}; rows=run168.row_base(widths)
    stored=[{**r,'question_index':8,'top_ids':list(range(2+100*r['slot'],102+100*r['slot']))} for r in rows]
    work=[{**r,'question_index':8,'ban':b} for r in rows[:6] for b in ('none','own','matched','mass') if b!='mass' or r['arm'] in ('n+','k4')]
    massmap={(r['arm'],r['layer'],r['slot'],8):{'mass_ban_ids':list(range(500,550))} for r in stored if r['arm'] in ('n+','k4')}
    lists=run168.ban_lists(work,stored,Tok(),massmap)
    assert len(lists[0]['none'])==0 and len(lists[0]['own'])==50
    for r,banlist in zip(work,lists):
        if r['ban']=='mass': assert banlist['mass']==list(range(500,550))

def test_mask_is_per_row():
    mask=run168.make_mask([[1],[2,3]],5,'cpu',torch)
    scores=torch.zeros((2,5)); result=run168.RowBanProcessor(mask)(torch.zeros((2,1),dtype=torch.long),scores)
    assert torch.isneginf(result[0,1]) and torch.isneginf(result[1,2]) and torch.isfinite(result[0,2])

def test_scorer_writes_outputs_and_checks_gates_on_synthetic_battery(tmp_path,monkeypatch):
    root=tmp_path/'repo'; (root/'results/part166').mkdir(parents=True)
    (root/'results/part166/loop_tokens.json').write_text(json.dumps({'stop_list':['the','and']}))
    monkeypatch.setattr(score168,'ROOT',root)
    rows=[]; answers=[]; masses=[]; norms=[]
    for l in score168.LAYERS:
        width=9216; used={score168.neuron_for_seed(width,l,s,b) for b in (165000,167000) for s in range(12)}
        for slot in score168.SLOTS:
            neuron,attempt=run168.fresh_neuron(width,l,slot,used,torch); used.add(neuron)
            for arm in score168.ARMS:
                norms.append({'layer':l,'slot':slot,'arm':arm,'width':width,'neuron':neuron,'attempt':attempt,'relative_error':0.0})
    for r in run168.all_rows({l:9216 for l in run168.LAYERS}):
        row={**r,'sharpness':1.,'steadiness':1.,'bias_norm':1.,'top_ids':list(range(1,201)),
             'top_values':[float(200-i) for i in range(200)],'own_ban_ids':list(range(1,51)),
             'rank_51_200_ids':list(range(51,201))}
        rows.append(row)
        masses.append({'arm':r['arm'],'layer':r['layer'],'slot':r['slot'],'question_index':r['question_index'],
            'own_mass':.1,'matched_mass':.1,'mass_mass':.1 if r['arm']!='full' else None,'paired_set':True if r['arm']!='full' else None,'feasible':True if r['arm']!='full' else None,'mass_rel_error':.01 if r['arm']!='full' else None,'own_mass_overlap':0,
            'matched_mass_overlap':0,'overlap':0,'exponent':0.,'draws':1,'in_tolerance':True if r['arm']!='full' else None,
            'allowed_vocab':9000,'plain_overlap_own':0,'plain_overlap_matched':0,'plain_overlap_mass':0,
            'mass_ban_ids':list(range(1000,1050)) if r['arm'] in ('n+','k4') else []})
        bans=('none','own','matched','mass') if r['arm'] in ('n+','k4') else ('none','own','matched')
        for ban in bans:
            looping=(ban=='none' and r['arm'] in ('n+','k4') and r['layer']==19)
            text='alpha beta gamma alpha beta gamma alpha beta gamma' if looping else 'A coherent synthetic answer with enough distinct words.'
            banned=[] if ban=='none' else list(range(1,51)) if ban=='own' else list(range(1,51)) if ban=='matched' else list(range(1000,1050))
            answers.append({'stage':'battery','arm':r['arm'],'ban':ban,'layer':r['layer'],'slot':r['slot'],'question_index':r['question_index'],
                            'answer':text,'token_ids':[9999],'token_count':8,'banned_ids':banned})
    answers += [{'stage':'no_push','arm':'plain','question_index':qi,'answer':'A plain synthetic answer.','token_ids':[9999],'token_count':4} for qi in run168.PHRASINGS]
    for name,vals in (('rows.jsonl',rows),('answers.jsonl',answers),('masses.jsonl',masses)):
        (tmp_path/name).write_text(''.join(json.dumps(x)+'\n' for x in vals))
    (tmp_path/'plain.json').write_text('{}'); (tmp_path/'tokens.json').write_text('{}')
    (tmp_path/'norms.json').write_text(json.dumps({'directions':norms}))
    result=score168.score(tmp_path/'rows.jsonl',tmp_path/'plain.json',tmp_path/'tokens.json',tmp_path/'answers.jsonl',tmp_path/'out')
    assert result['gates']['gate1'] and result['gates']['gate2'] and result['gates']['gate3']
    assert result['gates']['gate4'] and result['gates']['gate5'] and result['gates']['gate6'] and result['gates']['gate7']
    assert not result['readings_stopped'] and set(result['predictions'])=={f'168.{i}' for i in range(1,9)}|{'168.W1','168.W2','168.W3'}
    assert (tmp_path/'out/score.json').is_file() and (tmp_path/'out/tables.md').is_file()
    blind=[json.loads(x) for x in (tmp_path/'out/hand_read.jsonl').read_text().splitlines()]
    key=[json.loads(x) for x in (tmp_path/'out/hand_read_key.jsonl').read_text().splitlines()]
    assert len(blind)==len(key)==120 and all('ban' not in x for x in blind)

def test_draw_keeps_the_nearest_draw_and_logs_an_unreachable_target():
    q=np.full(4000,1e-6); q[0]=.9; q[1]=.05
    ids,_,draws,ok,delta,own_mass,feasible=run168.draw_mass_ids(q,[0],[3999],{3998},'n+',19,0,8)
    assert own_mass==.9 and draws==200 and not ok and not feasible
    assert abs(q[ids].sum()-.9)==pytest.approx(delta,abs=1e-9)

def test_calibration_matches_a_reachable_heavy_tailed_target():
    rng=np.random.default_rng(1); q=np.sort(rng.pareto(.7,30000)+1e-6)[::-1]; q=q/q.sum()
    target=int(np.flatnonzero(np.cumsum(q)>.02)[0]); own=[target]
    ids,exponent,draws,ok,delta,own_mass,feasible=run168.draw_mass_ids(q,own,[29999],{29998},'n+',20,1,9)
    assert feasible and ok and delta<=.1*own_mass and len(set(ids))==50 and draws<200

def test_position_diagnostic_marks_the_top_token():
    head=torch.nn.Linear(2,6,bias=False)
    with torch.no_grad(): head.weight.copy_(torch.tensor([[5.,0.],[0.,5.],[0.,0.],[0.,0.],[0.,0.],[0.,0.]]))
    hidden=torch.tensor([[[1.,0.],[0.,1.]]])
    d=run168.position_diagnostic(hidden,head,[[0]],[[1]])
    assert d.shape==(1,2,4) and d[0,0,2]==1 and d[0,0,3]==0 and d[0,1,2]==0 and d[0,1,3]==1

def test_verify_lists_refuses_a_short_or_special_list():
    good={('n+',19,0,8):{'mass_ban_ids':list(range(10,60))}}
    run168.verify_lists(good,{0})
    with pytest.raises(SystemExit): run168.verify_lists({('n+',19,0,8):{'mass_ban_ids':list(range(10,59))}},{0})
    with pytest.raises(SystemExit): run168.verify_lists({('k4',19,0,8):{'mass_ban_ids':list(range(0,50))}},{0})

def test_stage_one_plain_answer_is_forty_tokens():
    source=(ROOT/'part168/run168.py').read_text(encoding='utf-8')
    assert "plainmap[qi]={'question_index':qi" in source
    assert "generate_plain(model,tok,item['question'],common,args.device,40)" in source

def test_bootstrap_finishes_and_brackets_the_point():
    events=[{'arm':a,'ban':b,'layer':l,'slot':s,'loop':bool((s+l)%3==0) if b=='none' else False}
            for a in ('n+','k4','full') for b in ('none','own','matched','mass') for l in score168.LAYERS for s in score168.SLOTS for _ in range(4)]
    out=score168.bootstrap_intervals(events)
    assert out['own_relative_cut']['replicates']==10000 and out['own_relative_cut']['interval_95'][0]==1.0
    lo,hi=out['own_absolute_difference']['interval_95']
    assert lo<sum(e['loop'] for e in events if e['arm'] in ('n+','k4'))/sum(1 for e in events if e['arm'] in ('n+','k4') and e['ban']=='none')<hi

def test_transition_classes_are_exclusive_and_cover_common_only():
    words={1:'alpha',2:'beta',3:'gamma',4:'delta'}; tokens={str(k):v for k,v in words.items()}
    row={'own_ban_ids':[1],'rank_51_200_ids':[2],'top_ids':[1,2]}
    rowmap={('n+',19,s,8):dict(row) for s in score168.SLOTS}
    loop=lambda w:(w+' ')*9
    def run(alt):
        amap={('n+','none',19,0,8):{'answer':loop('alpha')},('n+','own',19,0,8):{'answer':alt}}
        return score168.transition_summary(['n+'],'own',rowmap,amap,tokens,{'the'})
    moved=run(loop('beta')); assert moved['migration_count']==1 and moved['count']==1 and moved['migration_share']==1
    assert moved['migration_baseline_share']==1
    away=run(loop('gamma')); assert away['outside_count']==1 and away['count']==1
    gone=run('A plain ending with distinct words here.'); assert gone['count']==0 and gone['own_word_loop_fates']=={'no_loop':1}
    common=run(loop('the')); assert common['count']==1 and common['common_only_count']==1 and common['migration_baseline_share']==0
    same=run(loop('alpha')); assert same['count']==0 and same['same_word_reentry_rows']==1

def test_gate_four_fails_when_a_stored_list_is_not_the_rows_list():
    src=(ROOT/'part168/score168.py').read_text(encoding='utf-8')
    assert 'ids!=want' in src

