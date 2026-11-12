from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part201 import run201
from tools.margin_recorder import MarginRecorder

class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.p=run201.plan(ROOT/'part163/units163.json')
    def test_plan_counts_and_determinism(self):
        p=self.p['passes']; self.assertEqual((p['A']['rows'],len(p['A']['calls'])),(384,48))
        self.assertEqual((p['A-again']['rows'],len(p['A-again']['calls'])),(16,2))
        self.assertEqual((p['B']['rows'],len(p['B']['calls'])),(384,48))
        self.assertEqual((p['M']['rows'],len(p['M']['calls'])),(408,51))
        self.assertEqual((p['C']['rows'],len(p['C']['calls'])),(216,216))
        self.assertEqual((p['A2']['rows'],len(p['A2']['calls'])),(96,12))
        self.assertEqual((p['F8']['rows'],len(p['F8']['calls'])),(96,12))
        self.assertEqual((p['F1']['rows'],len(p['F1']['calls'])),(48,48))
        self.assertEqual(self.p,run201.plan(ROOT/'part163/units163.json'))
    def test_b_membership_and_slot_rules(self):
        a=self.p['passes']['A']['calls']; b=self.p['passes']['B']['calls']; aset={frozenset(c['row_ids']) for c in a}
        self.assertTrue(all(frozenset(c['row_ids']) not in aset for c in b))
        aslots={r['row_id']:slot for c in a for slot,r in enumerate(c['rows'])}
        self.assertTrue(all(aslots[r['row_id']]!=slot for c in b for slot,r in enumerate(c['rows'])))
    def test_repeated_passes_equal_a_calls(self):
        a=self.p['passes']['A']['calls']
        for name in ('A-again','A2','F8'):
            for c in self.p['passes'][name]['calls']:
                ac=next(x for x in a if x['call']==c['call'])
                self.assertEqual(c['row_ids'],ac['row_ids'])
    def test_float32_subsets_follow_the_design(self):
        p=self.p; qs=run201._battery(); order=p['f_question_order']
        kinds=[qs[qi]['kind'] for qi in order]
        self.assertEqual(kinds,['identity','maker']*3)
        for kind in ('identity','maker'):
            lowest=[qi for qi,q in enumerate(qs) if q['kind']==kind][:3]
            self.assertEqual(sorted(qi for qi in order if qs[qi]['kind']==kind),lowest)
        a={c['call']:c for c in p['passes']['A']['calls']}
        f8=p['passes']['F8']['calls']
        self.assertTrue(all(c['row_ids']==a[c['call']]['row_ids'] for c in f8))          # pass A's calls, same slots
        self.assertEqual([c['rows'][0]['question_index'] for c in f8],[qi for qi in order for _ in range(2)])
        f1=p['passes']['F1']['calls']
        self.assertTrue(all(len(c['rows'])==1 for c in f1))                              # one row per call
        f8_ids={rid for c in f8 for rid in c['row_ids']}; c_ids={rid for c in p['passes']['C']['calls'] for rid in c['row_ids']}
        f1_ids=[c['row_ids'][0] for c in f1]
        self.assertTrue(set(f1_ids)<=f8_ids and set(f1_ids)<=c_ids)
        # The design's cut keeps F8's first 48 rows and F1's first 24: both kinds stay, and F1 stays inside F8.
        cut8={rid for c in f8[:run201.CUT['F8']] for rid in c['row_ids']}; cut1=f1_ids[:run201.CUT['F1']]
        self.assertEqual((len(cut8),len(cut1)),(48,24)); self.assertTrue(set(cut1)<=cut8)
        self.assertEqual({r['kind'] for c in f8[:run201.CUT['F8']] for r in c['rows']},{'identity','maker'})

    def test_smoke_plan_covers_every_pass_with_small_fixture(self):
        p=run201.smoke_plan(ROOT/'part163/units163.json')
        self.assertEqual(set(p['passes']),{'A','A-again','B','M','C','A2','F8','F1'})
        pushed={r['unit_id'] for info in p['passes'].values() for c in info['calls'] for r in c['rows'] if r['arm']!='plain'}
        self.assertEqual(len(pushed),4)
        plain={r['row_id'] for c in p['passes']['M']['calls'] for r in c['rows'] if r['arm']=='plain'}
        self.assertEqual(len(plain),4)

    def test_incomplete_call_is_removed_as_a_unit(self):
        import tempfile
        call=self.p['passes']['A']['calls'][0]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'rows.jsonl'
            row={'pass':'A','call':call['call'],'row_id':call['row_ids'][0]}
            path.write_text(json.dumps(row)+'\n',encoding='utf-8')
            self.assertEqual(run201._load_rows(path,self.p),[])
            self.assertEqual(path.read_text(encoding='utf-8'),'')

    def test_resume_skips_written_pass_row_pairs(self):
        c=self.p['passes']['A']['calls'][0]; done={('A',c['row_ids'][0])}
        self.assertEqual(len(run201.pending_rows(c,'A',done)),7)
        self.assertEqual(run201.pending_rows(c,'A',set()),list(enumerate(c['rows'])))
    def test_gate_mutations_are_rejected(self):
        good_load={'attn_implementation':'eager','dtype':'torch.bfloat16','process':'bf16-main'}
        self.assertFalse(run201.g1_valid({**good_load,'attn_implementation':'sdpa'}))
        row={'emitted_is_max':True,'emitted_logit':[1.0],'top1_logit':[1.0], 'hook_dtype':'torch.float32','push_max_abs_diff':0.0,
             'call_rows':['r0'],'call':0,'slot':0}
        self.assertFalse(run201.g2_valid({**row,'emitted_is_max':False}))
        call={'row_ids':['r0'],'call':0}
        self.assertFalse(run201.g3_valid({**row,'slot':1},call,0))
        self.assertFalse(run201.g4_valid({**row,'push_max_abs_diff':0.01}))

    def test_g5_reference_comparator_rejects_mismatch(self):
        self.assertTrue(run201.g5_equal([[1],[2]],[[1],[2]]))
        self.assertFalse(run201.g5_equal([[1],[2]],[[1],[3]]))

    def test_g5_compares_rows_that_end_at_different_lengths(self):
        # Alden's toy (council part201-design-review #11): row 0 ends one step before row 1, so the generator's
        # rectangle pads it and the recorder's row does not. Raw equality is false; cut at the end token it is true.
        legacy=[[11,2,0],[11,12,2]]; recorded=[[11,2],[11,12,2]]
        self.assertFalse(run201.g5_equal(legacy,recorded))
        self.assertTrue(run201.g5_equal(legacy,recorded,[2]))
        self.assertFalse(run201.g5_equal([[11,2,0],[11,13,2]],recorded,[2]))      # a real mismatch before the end token
        self.assertFalse(run201.g5_equal([[12,2,0],[11,12,2]],recorded,[2]))      # ... also in the row that ended early
        self.assertFalse(run201.g5_equal([[11,12,0]],[[11,12]],[2]))              # no end token: nothing is cut away
        self.assertEqual(run201.eos_trim([11,2,0,0],[2]),[11,2]); self.assertEqual(run201.eos_trim([11,12,0],[2]),[11,12,0])

    def test_g5_launch_comparison_takes_the_generator_rectangle_as_it_is(self):
        import torch
        rectangle=torch.tensor([[5,5,11,2,0],[5,5,11,12,2]])                     # two prompt columns, then the answers
        rows=[{'tokens':[11,2]},{'tokens':[11,12,2]}]
        self.assertTrue(run201.g5_from_capture(rectangle,2,rows,[2]))
        self.assertFalse(run201.g5_from_capture(rectangle,2,[{'tokens':[11,2]},{'tokens':[11,13,2]}],[2]))
        self.assertFalse(run201.g5_from_capture(rectangle,2,rows[:1],[2]))        # a missing row is not equality
        self.assertFalse(run201.g5_from_capture(None,2,rows,[2]))                 # nothing captured is not a pass

def attempt(nonfinite=False,seconds=1.0,steps=3,firings=2,forwards=2,dtype='torch.bfloat16',diff=0.0):
    """One scripted decode attempt of a two-row pushed call, in the shape run201.decode_attempt returns."""
    rows=[{'tokens':list(range(steps)),'nonfinite':nonfinite and i==0,'hook_dtype':dtype,'push_max_abs_diff':diff} for i in range(2)]
    return {'rows':rows,'seconds':seconds,'hook_firings':{0:firings},'forward_passes':forwards,'generation_steps':steps}

class ReceiptTests(unittest.TestCase):
    """Alden's review of a3a40b1 (council part201-design-review #11), points 2 and 3."""

    def script(self,*attempts):
        queue=list(attempts); return lambda: queue.pop(0)

    def test_a_finite_first_attempt_is_kept_with_its_own_receipt(self):
        kept,discarded,problems=run201.select_attempt(self.script(attempt(seconds=4.0)),'bf16-main',True)
        self.assertEqual((kept['seconds'],discarded,problems),(4.0,None,[]))

    def test_a_retry_keeps_the_second_attempts_rows_time_and_hook_counts(self):
        first=attempt(nonfinite=True,seconds=9.0,steps=3,firings=2,forwards=2)
        second=attempt(seconds=4.0,steps=5,firings=4,forwards=4)
        kept,discarded,problems=run201.select_attempt(self.script(first,second),'float32',True)
        self.assertEqual(problems,['the activation at the push hook is not float32'])   # judged on the attempt kept
        second=attempt(seconds=4.0,steps=5,firings=4,forwards=4,dtype='torch.float32')
        kept,discarded,problems=run201.select_attempt(self.script(first,second),'float32',True)
        self.assertEqual(problems,[])
        self.assertIs(kept,second)
        self.assertEqual((kept['seconds'],kept['generation_steps'],kept['hook_firings'],kept['forward_passes']),(4.0,5,{0:4},4))
        self.assertEqual(discarded,{'seconds':9.0,'generation_steps':3,'nonfinite_rows':1,'hook_firings':{0:2},'forward_passes':2})

    def test_a_bad_second_attempt_is_not_hidden_by_a_good_first_receipt(self):
        first=attempt(nonfinite=True,firings=2,forwards=2,dtype='torch.float32')         # its hook evidence is fine
        second=attempt(firings=3,forwards=4,dtype='torch.float32')                       # the kept attempt's is not
        kept,discarded,problems=run201.select_attempt(self.script(first,second),'float32',True)
        self.assertIs(kept,second); self.assertEqual(len(problems),1); self.assertIn('are not the 4 decode forwards',problems[0])
        second=attempt(dtype='torch.float32',diff=0.5)
        self.assertEqual(run201.select_attempt(self.script(first,second),'float32',True)[2],['the delta the hook added is not the prescribed bf16 push'])

    def test_a_still_nonfinite_second_attempt_is_the_one_kept(self):
        first=attempt(nonfinite=True,seconds=9.0); second=attempt(nonfinite=True,seconds=5.0)
        kept,discarded,problems=run201.select_attempt(self.script(first,second),'bf16-main',True)
        self.assertIs(kept,second); self.assertEqual(discarded['seconds'],9.0)

    def test_attempt_evidence_by_process_and_for_plain_calls(self):
        self.assertEqual(run201.attempt_problems(attempt(),'bf16-main',True),[])                  # bf16 at the hook is right for bf16
        self.assertEqual(len(run201.attempt_problems(attempt(firings=1),'bf16-main',True)),1)     # a missed step is not
        plain={'rows':[{'tokens':[1],'nonfinite':False,'hook_dtype':None,'push_max_abs_diff':None}],'seconds':1.,'hook_firings':{},'forward_passes':3,'generation_steps':1}
        self.assertEqual(run201.attempt_problems(plain,'float32',False),[])
        # A call that should have been pushed and shows no hook at all: no firing count, no delta, no float32.
        self.assertEqual(len(run201.attempt_problems(plain,'float32',True)),3)

    def test_push_hash_does_not_depend_on_the_rows_sharing_the_call(self):
        import torch
        mixed={0:torch.tensor([[1.,2.],[0.,0.]]),5:torch.tensor([[0.,0.],[3.,4.]])}
        alone0={0:torch.tensor([[1.,2.]])}; alone1={5:torch.tensor([[3.,4.]])}
        self.assertEqual(run201._push_hash(mixed,0),run201._push_hash(alone0,0))
        self.assertEqual(run201._push_hash(mixed,1),run201._push_hash(alone1,0))
        self.assertNotEqual(run201._push_hash(mixed,0),run201._push_hash(mixed,1))
        self.assertNotEqual(run201._push_hash({0:torch.tensor([[3.,4.]])},0),run201._push_hash(alone1,0))   # the layer is in the hash
        self.assertNotEqual(run201._push_hash({0:torch.tensor([[1.,2.5]])},0),run201._push_hash(alone0,0))  # and so is the payload
        self.assertEqual(run201._push_hash({'5':torch.tensor([[3.,4.]])},0),run201._push_hash(alone1,0))    # a layer key read back from JSON
        self.assertIsNone(run201._push_hash({},0))

    def test_inputs_are_hashed_in_a_canonical_form(self):
        a=run201.input_hashes([{'unit_id':'u','layer':1}],{'1':1.5,'2':2.0})
        b=run201.input_hashes(json.loads('[ {"layer": 1,\r\n "unit_id": "u"} ]'),json.loads('{"2": 2.0,   "1": 1.5}'))
        self.assertEqual(a,b)
        self.assertNotEqual(a['medians_sha256'],run201.input_hashes([{'unit_id':'u','layer':1}],{'1':1.5,'2':2.5})['medians_sha256'])
        self.assertEqual(a['units_sha256'],run201.input_hashes([{'unit_id':'u','layer':1}],{'1':9})['units_sha256'])

    def test_rows_without_their_call_receipt_are_set_aside_on_resume(self):
        plan={'passes':{'A':{'calls':[{'call':0,'row_ids':['r0','r1']},{'call':1,'row_ids':['r2','r3']}]}}}
        rows=[{'pass':'A','call':c,'row_id':rid,'launch':0} for c,ids in ((0,['r0','r1']),(1,['r2','r3'])) for rid in ids]
        load={'loads':[{'calls':[{'pass':'A','call':0}]}]}                       # call 1's receipt never reached the disk
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'rows.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            self.assertEqual(len(run201._load_rows(path,plan)),4)                # without the record nothing is questioned
            kept=run201._load_rows(path,plan,load)
            self.assertEqual([r['row_id'] for r in kept],['r0','r1'])
            self.assertEqual(len(path.read_text(encoding='utf-8').splitlines()),2)
            aside=[json.loads(x) for x in (Path(d)/'rows.removed.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([r['row_id'] for r in aside],['r2','r3']); self.assertTrue(all('removed' in r for r in aside))
            for bad in (7,None,True,'0'):                                        # a launch that is not an entry of the record
                path.write_text(json.dumps(dict(rows[0],launch=bad))+'\n'+json.dumps(dict(rows[1],launch=bad))+'\n',encoding='utf-8')
                self.assertEqual(run201._load_rows(path,plan,load),[])

    def test_a_hook_that_never_fired_has_measured_nothing(self):
        import torch
        module=torch.nn.Linear(2,2); hook=run201.RecordingPushHook(module,torch.ones((3,2)))
        try: self.assertEqual(hook.diffs,[None,None,None]); self.assertEqual((hook.firings,hook.dtype),(0,None))
        finally: hook.remove()
        never={'rows':[{'tokens':[1],'nonfinite':False,'hook_dtype':'None','push_max_abs_diff':None}],'seconds':1.,'hook_firings':{0:0},'forward_passes':2,'generation_steps':1}
        self.assertEqual(len(run201.attempt_problems(never,'bf16-main',True)),3)                  # no firing, no delta, no bf16
        self.assertEqual(len(run201.attempt_problems(attempt(dtype='torch.float32'),'bf16-main',True)),1)   # float32 in a bf16 process

    def test_a_call_with_no_decode_forward_is_named_as_an_unexercised_push(self):
        # Every row ended at its first generated token (decoded in the prefill, before any push): no decode forward.
        ended={'rows':[{'tokens':[9],'nonfinite':False,'hook_dtype':'None','push_max_abs_diff':None}],'seconds':1.,'hook_firings':{0:0},'forward_passes':0,'generation_steps':1}
        problems=run201.attempt_problems(ended,'bf16-main',True)
        self.assertEqual(len(problems),1); self.assertIn('the push was not exercised',problems[0])
        self.assertNotIn('not the prescribed bf16 push',problems[0])                              # it is not called a wrong push
        self.assertEqual(run201.attempt_problems(ended,'bf16-main',False),[])                      # a plain call may end at once
        # Zero firings against zero forwards is never a pass for a pushed call.
        self.assertTrue(run201.select_attempt(lambda: ended,'float32',True)[2])

    def test_model_snapshot_is_read_from_the_cache_path(self):
        rev='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
        self.assertEqual(run201.model_snapshot(f'E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/{rev}'),rev)
        self.assertEqual(run201.model_snapshot(f'E:\\hf\\hub\\models--Qwen--Qwen3.5-4B\\snapshots\\{rev}\\'),rev)
        self.assertIsNone(run201.model_snapshot('/tmp/ckpt')); self.assertIsNone(run201.model_snapshot(f'/x/snapshots/{rev[:39]}'))

    def test_the_real_run_refuses_another_snapshot_before_loading_and_a_smoke_refuses_the_real_folder(self):
        import neuron_report as nr
        original=nr.load_model
        def no_load(*args,**kwargs): raise AssertionError('the model must not be loaded')
        nr.load_model=no_load
        try:
            with tempfile.TemporaryDirectory() as d:
                other='0'*40
                with self.assertRaises(SystemExit) as stop:
                    run201.main(['--device','cpu','--model',f'{d}/snapshots/{other}','--medians',str(ROOT/'results/part112/medians.json'),'--out',str(Path(d)/'out')])
                self.assertEqual(stop.exception.code,3)
                self.assertFalse((Path(d)/'out'/'load.json').exists())             # refused before any launch entry
            with tempfile.TemporaryDirectory() as d:                              # a stand-in for results/part201
                real=run201.OUT_DEFAULT; run201.OUT_DEFAULT=Path(d)/'part201'
                try: self.assertEqual(run201.main(['--smoke','--device','cpu','--model','x','--out',str(Path(d)/'part201')]),3)
                finally: run201.OUT_DEFAULT=real
                self.assertFalse((Path(d)/'part201').exists())                     # nothing was frozen there
        finally: nr.load_model=original

    def test_a_record_is_replaced_whole(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'load.json'; run201._write_json(path,{'a':1}); run201._write_json(path,{'a':2})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')),{'a':2}); self.assertEqual([p.name for p in Path(d).iterdir()],['load.json'])

class RecorderEdgeTests(unittest.TestCase):
    def test_tie_emitted_second_is_still_max(self):
        import torch
        r=MarginRecorder(); r(torch.tensor([[0]]),torch.tensor([[4.,4.]]))
        out=r.rows(torch.tensor([[0,1]]),1,[])[0]
        self.assertTrue(out['emitted_is_max']); self.assertFalse(out['argmax_agrees'])


class TinyTokenizer:
    pad_token_id=0; eos_token_id=2; padding_side='right'
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True,enable_thinking=False):
        return messages[-1]['content']+' assistant'
    def __call__(self,texts,return_tensors=None,padding=False,add_special_tokens=False):
        import torch
        if isinstance(texts,str): texts=[texts]
        rows=[[3+(sum(map(ord,c))%20) for c in text.split()] for text in texts]
        width=max(map(len,rows)) if padding else len(rows[0]); ids=[]; masks=[]
        for row in rows:
            pad=[0]*(width-len(row)); ids.append(pad+row); masks.append([0]*len(pad)+[1]*len(row))
        return {'input_ids':torch.tensor(ids,dtype=torch.long),'attention_mask':torch.tensor(masks,dtype=torch.long)}
    def decode(self,ids,skip_special_tokens=True): return ' '.join(str(int(x)) for x in ids if not skip_special_tokens or int(x)!=2)

def tiny_qwen(dtype=None):
    import torch
    from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM
    cfg=Qwen3_5TextConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,head_dim=8,linear_num_key_heads=2,
        linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
        layer_types=['full_attention'],eos_token_id=2,pad_token_id=0)
    model=Qwen3_5ForCausalLM(cfg).eval()
    if dtype is not None: model.to(dtype=dtype)
    return model

class TinyQwenTests(unittest.TestCase):
    def test_generate_rows_mixed_left_padding_and_g2(self):
        import torch
        model=tiny_qwen(); tok=TinyTokenizer(); layers=model.model.layers
        rec=MarginRecorder([2]); seq,rows=run201.generate_rows(model,tok,layers,['short prompt','a much longer mixed prompt here'],{},3,'cpu',rec)
        self.assertEqual(seq.shape[0],2); self.assertGreater(len(tok('short prompt')['input_ids'][0]),0)
        self.assertTrue(all(r['emitted_is_max'] for r in rows))
        self.assertEqual(rows[0]['hook_dtype'],None)

    def test_tiny_g5_reference_generation_tokens(self):
        import torch
        from part160 import run160
        from part86 import common86
        model=tiny_qwen(); tok=TinyTokenizer(); layers=model.model.layers
        pushes={0:torch.zeros((8,32),dtype=torch.float32)}; question='Who are you?'
        captured={}; original=model.generate
        def capture(*args,**kwargs):
            captured['prompt_length']=int(kwargs['input_ids'].shape[1]); result=original(*args,**kwargs); captured['seq']=result.detach().cpu(); return result
        model.generate=capture
        try: run160.generate_checked(model,tok,layers,question,pushes,3,'cpu',common86)
        finally: model.generate=original
        prompts=[common86.build_prompt(tok,question)]*8; rec=MarginRecorder([2])
        seq,rows=run201.generate_rows(model,tok,layers,prompts,pushes,3,'cpu',rec)
        # The same function the launch calls, on the generator's rectangle as captured: no trimming in the test.
        self.assertTrue(run201.g5_from_capture(captured['seq'],captured['prompt_length'],rows,[2]),
                        (captured['seq'].tolist(),[r['tokens'] for r in rows]))
        wrong=[dict(r,tokens=list(r['tokens'])) for r in rows]; wrong[3]['tokens'][0]=(wrong[3]['tokens'][0]+1)%32
        self.assertFalse(run201.g5_from_capture(captured['seq'],captured['prompt_length'],wrong,[2]))

    def test_tiny_float32_upcast_push_preserves_bf16_values(self):
        import torch,tempfile
        model=tiny_qwen(torch.bfloat16); tok=TinyTokenizer(); layers=model.model.layers
        push=torch.linspace(-0.04,0.04,32,dtype=torch.float32).repeat(2,1)
        from tools.kernel_reference import upcast_compute
        with tempfile.TemporaryDirectory() as d: upcast_compute(model,d)
        rec=MarginRecorder([2]); seq,rows=run201.generate_rows(model,tok,layers,['one row','two row prompts'],{0:push},2,'cpu',rec)
        self.assertTrue(all(r['hook_dtype']=='torch.float32' for r in rows))
        self.assertTrue(all(r['push_max_abs_diff']==0.0 for r in rows))

class TimingProbeTests(unittest.TestCase):
    def test_probe_runs_the_push_hook_on_full_prefixes_and_the_head_on_one_position(self):
        import torch
        model=tiny_qwen(); tok=TinyTokenizer(); layers=model.model.layers
        questions=['who are you really','what made you so','name your maker now','say your own name']
        push=torch.linspace(-0.04,0.04,32,dtype=torch.float32).repeat(4,1)
        rec=MarginRecorder([2]); seq,rows=run201.generate_rows(model,tok,layers,[q+' assistant' for q in questions],{0:push},3,'cpu',rec)
        for row,q in zip(rows,questions): row['question']=q
        probe=run201.timing_probe(model,tok,layers,rows,{0:push},'cpu',smoke=True)
        entries=probe['forward_passes']
        self.assertEqual(sorted({(e['rows'],e['prefix_tokens']) for e in entries}),[(n,l) for n in (1,4) for l in (8,16,24)])
        self.assertTrue(all(len(e['seconds'])==5 and all(s>0 for s in e['seconds']) for e in entries))
        self.assertTrue(all(e['logits_positions']==1 for e in entries))            # head ran on the last position only
        self.assertEqual(probe['head_input_positions_seen'],[8,16,24])            # and was handed the full prefix
        # The push hook fired on every forward of the probe: 5 repetitions per length, counted per row count.
        self.assertEqual([e['hook_firings'] for e in entries if e['rows']==4],[[5],[10],[15]])
        self.assertTrue(all(e['hook_dtypes']==['torch.float32'] for e in entries))
        self.assertIn('last position only',probe['work_timed'])
        # The probe leaves no hook behind: a later decode still has one-token decode steps and a full head.
        rec2=MarginRecorder([2]); seq2,rows2=run201.generate_rows(model,tok,layers,[q+' assistant' for q in questions],{0:push},3,'cpu',rec2)
        self.assertEqual([r['tokens'] for r in rows2],[r['tokens'] for r in rows])

class TimingProbeCleanup(unittest.TestCase):
    def test_a_probe_that_raises_leaves_no_hook_on_the_model(self):
        import torch
        model=tiny_qwen(); tok=TinyTokenizer(); layers=model.model.layers
        questions=['who are you really','what made you so']
        push=torch.linspace(-0.04,0.04,32,dtype=torch.float32).repeat(2,1)
        rec=MarginRecorder([2]); seq,rows=run201.generate_rows(model,tok,layers,[q+' assistant' for q in questions],{0:push},3,'cpu',rec)
        for row,q in zip(rows,questions): row['question']=q
        head=model.get_output_embeddings(); down=layers[0].mlp.down_proj
        before=(len(head._forward_pre_hooks),len(down._forward_pre_hooks))
        original=model.forward; calls={'n':0}
        def failing(*args,**kwargs):
            calls['n']+=1
            if calls['n']==3: raise RuntimeError('device lost')
            return original(*args,**kwargs)
        model.forward=failing
        try:
            with self.assertRaises(RuntimeError): run201.timing_probe(model,tok,layers,rows,{0:push},'cpu',smoke=True)
        finally: model.forward=original
        self.assertEqual((len(head._forward_pre_hooks),len(down._forward_pre_hooks)),before)
        # and the model still decodes as before: one-token steps, a full output head
        rec2=MarginRecorder([2]); seq2,rows2=run201.generate_rows(model,tok,layers,[q+' assistant' for q in questions],{0:push},3,'cpu',rec2)
        self.assertEqual([r['tokens'] for r in rows2],[r['tokens'] for r in rows])

class EndToEndTiny(unittest.TestCase):
    """The whole runner on a tiny random Qwen3.5 on the CPU: all three processes, the timing probe, relaunches
    after a clean finish and after a torn call, and then the scorer with every gate on."""

    def launch(self, folder):
        import torch
        import neuron_report as nr

        def fake_load(model_id, revision, device, attn_implementation='eager', dtype=torch.bfloat16):
            torch.manual_seed(201)                      # the same weights at every load, as a checkpoint would give
            model = tiny_qwen(torch.bfloat16)
            model.config._attn_implementation = 'eager'
            nr.LAST_LOAD = {'attn_implementation': 'eager'}
            return TinyTokenizer(), model

        original = nr.load_model
        nr.load_model = fake_load
        try:
            return run201.main(['--smoke', '--device', 'cpu', '--model', str(folder / 'ckpt'), '--units', str(folder / 'units.json'),
                                '--medians', str(folder / 'medians.json'), '--out', str(folder / 'out'), '--max-new-tokens', '6'])
        finally:
            nr.load_model = original

    def prepare(self, folder):
        (folder / 'ckpt').mkdir()
        units = ([{'unit_id': f'L0N{i}', 'pair_id': f'L0D{i}', 'arm': 'neuron', 'layer': 0, 'neuron': i, 'random_index': None} for i in range(12)]
                 + [{'unit_id': f'L0D{i}', 'pair_id': f'L0D{i}', 'arm': 'random', 'layer': 0, 'neuron': None, 'random_index': i} for i in range(12)])
        (folder / 'units.json').write_text(json.dumps(units), encoding='utf-8')
        (folder / 'medians.json').write_text(json.dumps({'0': 1.0}), encoding='utf-8')
        return folder / 'out'

    def read(self, out):
        return ([json.loads(line) for line in (out / 'rows.jsonl').read_text(encoding='utf-8').splitlines()],
                json.loads((out / 'load.json').read_text(encoding='utf-8')))

    def score(self, out):
        from part201 import score201
        rows, load = self.read(out)
        return score201.score(rows, None, None, load, json.loads((out / 'probe.json').read_text(encoding='utf-8')), out / 'plan.json')

    def test_every_row_traces_to_a_receipt_and_has_one_push_everywhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.prepare(Path(tmp))
            self.assertEqual(self.launch(Path(tmp)), 0)
            rows, load = self.read(out)
            for row in rows:
                receipt = run201.receipt_of(load, row['launch'], row['pass'], row['call'])
                self.assertEqual((receipt['row_ids'], receipt['attempts'], receipt['seconds']), (row['call_rows'], 1, row['seconds']))
                if row['unit_id'] is not None:
                    self.assertEqual(set(receipt['hook_firings'].values()), {receipt['forward_passes']})
            hashes = {}
            for row in rows:
                if row['unit_id'] is not None:
                    hashes.setdefault(row['row_id'], set()).add(row['push_sha256'])
            passes_of = {rid: {r['pass'] for r in rows if r['row_id'] == rid} for rid in hashes}
            self.assertTrue(any({'A', 'B', 'M', 'C', 'A2', 'F8', 'F1'} <= p for p in passes_of.values()))   # seen in every shape
            self.assertEqual({len(v) for v in hashes.values()}, {1})                 # one push per row, in every pass and process
            # The push belongs to the unit, not to the question: as many different hashes as units.
            unit_of = {r['row_id']: r['unit_id'] for r in rows if r['unit_id'] is not None}
            by_unit = {}
            for rid, v in hashes.items():
                by_unit.setdefault(unit_of[rid], set()).update(v)
            self.assertEqual({len(v) for v in by_unit.values()}, {1})
            self.assertEqual(len({next(iter(v)) for v in by_unit.values()}), len(by_unit))
            self.assertEqual(len({json.dumps(x['inputs'], sort_keys=True) for x in load['loads']}), 1)
            gates = self.score(out)['gates']
            self.assertTrue(gates['G3']['passed'] and gates['G4']['bf16']['passed'] and gates['G4']['float32']['passed'], gates)

    def test_rows_whose_receipt_was_lost_are_decoded_again(self):
        # The crash window Alden named: a call's rows on disk and no receipt for it. The relaunch does not read them
        # on trust: it sets them aside, decodes the call again and writes a receipt.
        with tempfile.TemporaryDirectory() as tmp:
            out = self.prepare(Path(tmp))
            self.assertEqual(self.launch(Path(tmp)), 0)
            rows, load = self.read(out)
            lost = load['loads'][2]['calls'].pop()                               # the float32 launch's last receipt
            (out / 'load.json').write_text(json.dumps(load), encoding='utf-8')
            broken = self.score(out)['gates']
            self.assertFalse(broken['G3']['float32']['passed'] or broken['G4']['float32']['passed'])
            self.assertEqual(self.launch(Path(tmp)), 0)
            after, load2 = self.read(out)
            self.assertEqual(sorted((r['pass'], r['row_id']) for r in after), sorted((r['pass'], r['row_id']) for r in rows))
            redone = [r for r in after if r['launch'] > 2]
            self.assertTrue(redone and all((r['pass'], r['call']) == (lost['pass'], lost['call']) for r in redone))
            self.assertIsNotNone(run201.receipt_of(load2, redone[0]['launch'], lost['pass'], lost['call']))
            aside = [json.loads(x) for x in (out / 'rows.removed.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(sorted(r['row_id'] for r in aside), sorted(lost['row_ids']))
            gates = self.score(out)['gates']
            self.assertTrue(gates['G3']['passed'] and gates['G4']['float32']['passed'], gates)

    def test_a_crash_while_a_receipt_is_written_leaves_no_rows_for_that_call(self):
        # The receipt goes to disk before the rows. Break the write of one receipt: that call's rows must not be on
        # disk, the record must still be readable, and the next launch decodes the call and finishes.
        original = run201._write_json; target = ('B', 1)

        def crashing(path, obj):
            loads = obj.get('loads') if isinstance(obj, dict) else None
            calls = loads[-1].get('calls') if loads else None
            if calls and (calls[-1]['pass'], calls[-1]['call']) == target and len(loads) == 1:
                raise RuntimeError('power lost')
            return original(path, obj)

        with tempfile.TemporaryDirectory() as tmp:
            out = self.prepare(Path(tmp))
            run201._write_json = crashing
            try:
                with self.assertRaises(RuntimeError):
                    self.launch(Path(tmp))
            finally:
                run201._write_json = original
            rows, load = self.read(out)                                          # load.json is whole: the old version
            self.assertFalse([r for r in rows if (r['pass'], r['call']) == target])
            self.assertIsNone(run201.receipt_of(load, 0, *target))
            self.assertTrue(all(run201.receipt_of(load, r['launch'], r['pass'], r['call']) for r in rows))   # no row without its receipt
            self.assertEqual(self.launch(Path(tmp)), 0)
            rows, load = self.read(out)
            plan = json.loads((out / 'plan.json').read_text(encoding='utf-8'))
            self.assertEqual(len(rows), sum(info['rows'] for info in plan['passes'].values()))
            gates = self.score(out)['gates']
            self.assertTrue(gates['G3']['bf16']['passed'] and gates['G3']['float32']['passed'], gates['G3'])

    def test_changed_medians_refuse_the_next_launch_and_respacing_them_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); out = self.prepare(folder)
            self.assertEqual(self.launch(folder), 0)
            before = (out / 'rows.jsonl').read_bytes(); loads = len(self.read(out)[1]['loads'])
            (folder / 'medians.json').write_text('{\r\n  "0":   1.0\r\n}\r\n', encoding='utf-8')     # the same numbers
            self.assertEqual(self.launch(folder), 0)
            (folder / 'medians.json').write_text(json.dumps({'0': 2.0}), encoding='utf-8')           # another push
            with self.assertRaises(SystemExit) as stop:
                self.launch(folder)
            self.assertEqual(stop.exception.code, 3)
            self.assertEqual((out / 'rows.jsonl').read_bytes(), before)
            self.assertEqual(len(self.read(out)[1]['loads']), loads + 3)          # the refused launch left no entry

    def test_a_nonfinite_first_attempt_is_replaced_by_the_retry_with_the_retrys_receipt(self):
        original = run201.decode_attempt; seen = {'n': 0}

        def flaky(*args, **kwargs):
            seen['n'] += 1
            result = original(*args, **kwargs)
            if seen['n'] == 1:                                                    # pass A, call 0, first attempt
                for row in result['rows']:
                    row['nonfinite'] = True
                result['seconds'] = 99.0; result['generation_steps'] = 77; result['hook_firings'] = {0: 76}; result['forward_passes'] = 76
            return result

        with tempfile.TemporaryDirectory() as tmp:
            out = self.prepare(Path(tmp))
            run201.decode_attempt = flaky
            try:
                self.assertEqual(self.launch(Path(tmp)), 0)
            finally:
                run201.decode_attempt = original
            rows, load = self.read(out)
            receipt = run201.receipt_of(load, 0, 'A', 0)
            written = [r for r in rows if r['pass'] == 'A' and r['call'] == 0]
            self.assertEqual(receipt['attempts'], 2)
            self.assertEqual((receipt['discarded_attempt']['seconds'], receipt['discarded_attempt']['generation_steps'],
                              receipt['discarded_attempt']['nonfinite_rows']), (99.0, 77, len(written)))
            self.assertNotEqual(receipt['seconds'], 99.0)
            self.assertEqual({r['seconds'] for r in written}, {receipt['seconds']})          # rows and receipt: one attempt
            self.assertEqual(receipt['generation_steps'], max(len(r['tokens']) for r in written))
            self.assertEqual(set(receipt['hook_firings'].values()), {receipt['forward_passes']})
            self.assertNotEqual(receipt['forward_passes'], 76)
            self.assertFalse(any(r['nonfinite'] for r in written))
            self.assertEqual([c['attempts'] for c in load['loads'][0]['calls'] if (c['pass'], c['call']) != ('A', 0)].count(2), 0)
            result = self.score(out)
            self.assertTrue(all(result['gates'][g]['passed'] for g in ('G2', 'G3', 'G4', 'G5')), result['gates'])
            self.assertEqual(result['predictions']['201.1']['verdict'], 'held')              # the repeat still equals pass A

    def test_run_relaunch_torn_call_and_score(self):
        import re
        from part201 import score201
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            out = self.prepare(folder)
            self.assertEqual(self.launch(folder), 0)
            lines = (out / 'rows.jsonl').read_text(encoding='utf-8').splitlines()
            rows = [json.loads(line) for line in lines]
            plan = json.loads((out / 'plan.json').read_text(encoding='utf-8'))
            planned = sum(info['rows'] for info in plan['passes'].values())
            self.assertEqual(len(rows), planned)
            self.assertEqual({r['pass'] for r in rows}, {'A', 'A-again', 'B', 'M', 'C', 'A2', 'F8', 'F1'})
            # Every written row carries every field of the schema.
            table = (ROOT / 'part201/ROWS-SCHEMA.md').read_text(encoding='utf-8')
            fields = [name for line in table.splitlines() if line.startswith('| `') for name in re.findall(r'`([a-z0-9_]+)`', line.split('|')[1])]
            self.assertEqual([name for name in fields if any(name not in r for r in rows)], [])
            self.assertTrue(all(len(r['call_rows']) == 1 for r in rows if r['pass'] in ('C', 'F1')))      # one row per call
            pushed = [r for r in rows if r['unit_id'] is not None]
            self.assertTrue(pushed and all(r['push_sha256'] and r['push_max_abs_diff'] == 0.0 for r in pushed))
            self.assertEqual({r['hook_dtype'] for r in pushed if r['pass'] in ('F8', 'F1')}, {'torch.float32'})
            self.assertEqual({r['hook_dtype'] for r in pushed if r['pass'] == 'A'}, {'torch.bfloat16'})
            load = json.loads((out / 'load.json').read_text(encoding='utf-8'))
            self.assertEqual([x['process'] for x in load['loads']], ['bf16-main', 'bf16-second', 'float32'])
            self.assertTrue(all('device_record' in x and x['launch'] == i for i, x in enumerate(load['loads'])))
            self.assertTrue(load['loads'][0]['G5']['passed'])
            probe = json.loads((out / 'probe.json').read_text(encoding='utf-8'))
            self.assertTrue(all(e['logits_positions'] == 1 for e in probe['forward_passes']))

            def scored():
                return score201.score(score201.read_jsonl(out / 'rows.jsonl'), None, None,
                                      json.loads((out / 'load.json').read_text(encoding='utf-8')),
                                      json.loads((out / 'probe.json').read_text(encoding='utf-8')), out / 'plan.json')

            result = scored()
            gates = result['gates']
            self.assertTrue(gates['G1']['bf16']['passed'] and gates['G1']['float32']['passed'], gates['G1'])
            self.assertTrue(gates['G2']['passed'] and gates['G4']['passed'] and gates['G5']['passed'], gates)
            self.assertTrue(gates['G3']['passed'], gates['G3'])
            self.assertEqual(result['predictions']['201.1']['verdict'], 'held', result['predictions']['201.1'])
            self.assertEqual(result['predictions']['201.7']['verdict'], 'not run')
            self.assertIn('201.11', score201.render(result))

            # A relaunch after a clean finish writes nothing and does not disown the A-again rows.
            self.assertEqual(self.launch(folder), 0)
            self.assertEqual((out / 'rows.jsonl').read_text(encoding='utf-8').splitlines(), lines)
            again = scored()
            self.assertTrue(again['state']['a_again_same_process'] and again['gates']['G3']['passed'], again['gates']['G3'])

            # A torn tail: the last call is cut mid-row. The relaunch drops that call whole and decodes it again.
            data = (out / 'rows.jsonl').read_bytes()
            (out / 'rows.jsonl').write_bytes(data[:-40])
            self.assertEqual(self.launch(folder), 0)
            after = [json.loads(line) for line in (out / 'rows.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(after), planned)
            self.assertEqual(sorted((r['pass'], r['row_id']) for r in after), sorted((r['pass'], r['row_id']) for r in rows))
            redone = [r for r in after if r['launch'] > 2]
            self.assertTrue(redone and all(r['pass'] == rows[-1]['pass'] and r['call'] == rows[-1]['call'] for r in redone))
            by_key = {(r['pass'], r['row_id']): r for r in rows}
            self.assertTrue(all(r['tokens'] == by_key[(r['pass'], r['row_id'])]['tokens'] for r in redone))
            self.assertTrue(scored()['gates']['G3']['passed'])


class CopyClosure(unittest.TestCase):
    """Each launch step of steps.txt, with only the files its --copy list names: the runner's whole smoke path must
    import and run there (the job host has nothing else of ours)."""

    def steps(self):
        import shlex
        out=[]
        for line in (ROOT/'part201/steps.txt').read_text(encoding='utf-8').splitlines():
            if not line.startswith('python3 tools/launch_arc.py'): continue
            words=shlex.split(line); copy=words[words.index('--copy')+1:]; step=words[words.index('--step')+1]
            out.append((step,copy))
        return out

    def run_in_copy(self,copy,command):
        import shutil,subprocess
        with tempfile.TemporaryDirectory() as tmp:
            for name in copy:
                dst=Path(tmp)/name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dst)
            done=subprocess.run([sys.executable,'-I','-B']+[part.replace('{folder}',tmp) for part in command],cwd=tmp,capture_output=True,text=True,timeout=300)
        return done

    def test_every_copied_file_exists_and_the_four_steps_are_there(self):
        steps=self.steps()
        self.assertEqual([('run201.py' in s,'kernel_reference.py' in s) for s,_ in steps],[(True,False),(True,False),(False,True),(True,False)])
        self.assertEqual([name for _,copy in steps for name in copy if not (ROOT/name).is_file()],[])

    def test_the_runner_runs_with_only_each_step_s_copied_files(self):
        lists={tuple(copy) for step,copy in self.steps() if 'part201/run201.py' in step}
        self.assertTrue(lists)
        for copy in sorted(lists):
            done=self.run_in_copy(copy,[str(ROOT/'part201/copy_closure_driver.py'),'{folder}'])
            self.assertEqual(done.returncode,0,(done.stdout[-400:],done.stderr[-1500:]))
            self.assertIn("closure ok: ['bf16-main', 'bf16-second', 'float32'] probe True",done.stdout)

    def test_a_copy_list_without_part_check_fails_here_and_not_on_the_card(self):
        copy=[name for name in next(copy for step,copy in self.steps() if 'part201/run201.py' in step) if name!='tools/part_check.py']
        done=self.run_in_copy(copy,[str(ROOT/'part201/copy_closure_driver.py'),'{folder}'])
        self.assertNotEqual(done.returncode,0); self.assertIn("No module named 'part_check'",done.stderr)

    def test_pass_k_s_tool_imports_with_only_its_copied_files(self):
        step,copy=next((s,c) for s,c in self.steps() if 'kernel_reference.py' in s)
        done=self.run_in_copy(copy,['tools/kernel_reference.py','--help'])
        self.assertEqual(done.returncode,0,done.stderr[-800:])

class SchemaTests(unittest.TestCase):
    def test_a_written_row_carries_every_schema_field_and_its_launch(self):
        import re
        model=tiny_qwen(); tok=TinyTokenizer(); layers=model.model.layers
        rec=MarginRecorder([2]); seq,rows=run201.generate_rows(model,tok,layers,['one prompt','another longer prompt'],{},3,'cpu',rec)
        spec={'row_id':'plain|q0','unit_id':None,'arm':'plain','layer':None,'question_index':0,'question':'one prompt','kind':'identity','dose':0.0}
        call={'call':5,'row_ids':['plain|q0','plain|q1']}
        row=run201.row_fields(rows[0],'M','bf16-main',spec,call,0,2,2,'x',None,1.5,3)
        table=(ROOT/'part201/ROWS-SCHEMA.md').read_text(encoding='utf-8')
        fields=[name for line in table.splitlines() if line.startswith('| `') for name in re.findall(r'`([a-z0-9_]+)`',line.split('|')[1])]
        self.assertIn('launch',fields); self.assertGreater(len(fields),30)
        self.assertEqual([name for name in fields if name not in row],[])
        self.assertEqual((row['launch'],row['call'],row['slot'],row['pass']),(3,5,0,'M'))
        json.dumps(row)

    def test_device_record_never_raises_and_names_the_device(self):
        self.assertEqual(run201.device_record('cpu')['device'],'cpu')

if __name__=='__main__': unittest.main()
