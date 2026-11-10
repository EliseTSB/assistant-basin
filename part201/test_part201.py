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
        legacy=captured['seq']; legacy_tokens=[]
        for values in legacy[:,captured['prompt_length']:].tolist():
            tokens=[int(v) for v in values]
            if 2 in tokens: tokens=tokens[:tokens.index(2)+1]
            legacy_tokens.append(tokens)
        self.assertTrue(run201.g5_equal(legacy_tokens,[r['tokens'] for r in rows]), (legacy_tokens,[r['tokens'] for r in rows]))

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

    def test_run_relaunch_torn_call_and_score(self):
        import re
        from part201 import score201
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'ckpt').mkdir()
            units = ([{'unit_id': f'L0N{i}', 'pair_id': f'L0D{i}', 'arm': 'neuron', 'layer': 0, 'neuron': i, 'random_index': None} for i in range(12)]
                     + [{'unit_id': f'L0D{i}', 'pair_id': f'L0D{i}', 'arm': 'random', 'layer': 0, 'neuron': None, 'random_index': i} for i in range(12)])
            (folder / 'units.json').write_text(json.dumps(units), encoding='utf-8')
            (folder / 'medians.json').write_text(json.dumps({'0': 1.0}), encoding='utf-8')
            out = folder / 'out'
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
