#!/usr/bin/env python3
"""CPU contract tests for Part 131 selection, scoring, resume, batching, and rental cleanup."""
import importlib.util, json, os, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

HERE=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
score=load('score131_under_test',Path(os.environ.get('PART131_SCORE_MODULE',HERE/'score131.py')))

class Part131Tests(unittest.TestCase):
    def test_positive_coherent_label_counts(self): self.assertTrue(score.show({'shows_persona':True,'coherent':True}))
    def test_incoherent_persona_does_not_count(self): self.assertFalse(score.show({'shows_persona':True,'coherent':False}))
    def test_absent_or_negative_label_does_not_count(self):
        self.assertFalse(score.show(None)); self.assertFalse(score.show({'shows_persona':False,'coherent':True}))

    def test_score_main_two_row_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); answers=root/'answers'; labels=root/'labels'; answers.mkdir(); labels.mkdir()
            rows=[{'id':'u1','unit_id':'u1','unit_class':'shared','arm':'pole','tag':'base','frame':'plain','question_index':0},
                  {'id':'unpushed-base-0','unit_id':'unpushed','unit_class':'unpushed','arm':'unpushed','tag':'base','frame':'plain','question_index':0}]
            (answers/'base.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            label_rows=[{**rows[0],'shows_persona':False,'coherent':True,'self_claim':False,'speaker':'none'}]
            label_rows.extend({'id':f'unpushed-base-0-{i:02d}','answer_id':'unpushed-base-0','unit_id':'unpushed','arm':'unpushed','tag':'base','frame':'plain','question_index':0,
                               'shows_persona':False,'coherent':True,'self_claim':False,'speaker':'none'} for i in range(4))
            (labels/'labels-ds.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in label_rows))
            units=root/'units.json'; units.write_text(json.dumps({'units':[{'id':'u1','class':'shared'}]}))
            out=root/'score.json'
            score.main(['--answers',str(answers),'--labels',str(labels),'--units',str(units),'--out',str(out)])
            result=json.loads(out.read_text()); self.assertEqual(result['rows']['answers'],2); self.assertEqual(result['missing_labels'],0)

    def test_resume_discards_truncated_last_line(self):
        runner=load('run131_under_test',HERE/'run131.py')
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'answers.jsonl'; path.write_bytes(b'{"id":"good"}\n{"id":"trunc')
            self.assertEqual(runner.recover_jsonl(path),[{'id':'good'}]); self.assertEqual(path.read_bytes(),b'{"id":"good"}\n')
            path.write_bytes(b'{"id":"good"}\n{"answer":"caf\xc3')
            self.assertEqual(runner.recover_jsonl(path),[{'id':'good'}]); self.assertEqual(path.read_bytes(),b'{"id":"good"}\n')
            path.write_bytes(b'{"id":"good"}\n{"id":\n')
            self.assertEqual(runner.recover_jsonl(path),[{'id':'good'}]); self.assertEqual(path.read_bytes(),b'{"id":"good"}\n')

    def test_unpushed_answers_expand_to_four_unique_targets(self):
        label=load('label131_under_test',HERE/'label131.py')
        with tempfile.TemporaryDirectory() as td:
            answers=Path(td)/'base.jsonl'
            answers.write_text(json.dumps({'id':'unpushed-base-0','unit_id':'unpushed','unit_class':'unpushed','arm':'unpushed','tag':'base','frame':'plain','question_index':0,'question':'Who are you?','answer':'I am here.'})+'\n')
            items=label.items_for([answers],HERE/'units131.json')
        self.assertEqual(len(items),4); self.assertEqual(len({i['public']['id'] for i in items}),4)
        self.assertEqual(len({i['name'] for i in items}),4)
        self.assertTrue(all(i['public']['answer_id']=='unpushed-base-0' and i['name'] and i['desc'] for i in items))

    def test_batched_gate3_tiny_random_llama_cpu(self):
        try:
            import torch
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError as exc: self.skipTest(str(exc))
        runner=load('run131_under_test',Path(os.environ.get('PART131_RUN_MODULE',HERE/'run131.py')))
        torch.manual_seed(132)
        config=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                           num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=32,pad_token_id=0,eos_token_id=2,
                           initializer_range=0.5)
        model=LlamaForCausalLM(config).eval()
        ids=torch.tensor([[0,4,5,6],[7,8,9,10],[0,11,12,13],[14,15,16,17]]); mask=torch.tensor([[0,1,1,1],[1,1,1,1],[0,1,1,1],[1,1,1,1]])
        self.assertTrue(runner.gate3_noop_exact(model,model.model.layers,ids,mask,0,max_new=4))
        with torch.inference_mode():
            unhooked=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,max_new_tokens=4,pad_token_id=0,eos_token_id=None)
        vectors={0:torch.randn((4,1,32))*10,1:torch.randn((4,1,32))*10}
        handles=[]
        for layer_index,vector in vectors.items():
            handles.append(model.model.layers[layer_index].register_forward_hook(
                lambda _m,_a,o,v=vector: (o[0]+v,)+tuple(o[1:]) if isinstance(o,tuple) else o+v))
        try:
            with torch.inference_mode():
                unrestricted=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,max_new_tokens=4,pad_token_id=0,eos_token_id=None)
        finally:
            for handle in handles: handle.remove()
        token_rows=unrestricted[:,ids.shape[1]:].tolist()
        self.assertFalse(torch.equal(unrestricted,unhooked), 'push hooks must alter the tiny model output')
        eos={int(token_rows[0][0]),int(token_rows[1][1])}
        self.assertNotIn(int(token_rows[1][0]),eos)
        self.assertNotIn(int(token_rows[2][0]),eos)
        self.assertNotIn(int(token_rows[3][0]),eos)
        hooked=[]
        for layer_index,vector in vectors.items():
            hooked.append(model.model.layers[layer_index].register_forward_hook(
                lambda _m,_a,o,v=vector: (o[0]+v,)+tuple(o[1:]) if isinstance(o,tuple) else o+v))
        try:
            with torch.inference_mode():
                plain=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,max_new_tokens=4,pad_token_id=0,eos_token_id=list(eos))
        finally:
            for handle in hooked: handle.remove()
        expected=[row[:next((i+1 for i,t in enumerate(row) if t in eos),len(row))] for row in plain[:,ids.shape[1]:].tolist()]
        compacted=runner.generate_batch(model,model.model.layers,ids,mask,vectors,0,eos,4)
        self.assertEqual(expected,compacted)
        self.assertEqual([1,2],sorted({len(compacted[0]),len(compacted[1])}))
        self.assertTrue(all(len(row)>2 for row in compacted[2:]))

    def test_pilot_main_writes_all_rows_for_both_registered_settings(self):
        try:
            import torch
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError as exc: self.skipTest(str(exc))
        runner=load('run131_pilot_main_under_test',HERE/'run131.py')
        data=json.loads((HERE/'units131.json').read_text())['units']
        expected=6 + 2 * 2 * 6
        self.assertEqual(expected,30)  # six unpushed plus two units x two arms x six questions
        class TinyTokenizer:
            pad_token_id=0; eos_token_id=2; pad_token='<pad>'; eos_token='<eos>'
            def __call__(self,text,add_special_tokens=True):
                return {'input_ids':[1+(ord(c)%60) for c in text[:12]] or [1]}
            def apply_chat_template(self,messages,**kwargs): return [1+(ord(c)%60) for c in messages[0]['content'][:12]] or [1]
            def decode(self,tokens,skip_special_tokens=False): return ' '.join(str(x) for x in tokens)
        torch.manual_seed(131)
        config=LlamaConfig(vocab_size=64,hidden_size=8,intermediate_size=10000,num_hidden_layers=4,
                           num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,pad_token_id=0,eos_token_id=2,
                           initializer_range=0.2)
        model=LlamaForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'out'; units=Path(td)/'units.json'; units.write_text(json.dumps({'units':data}))
            for tag,frame,repo,revision in [('base','plain','unsloth/Meta-Llama-3.1-8B','e9a141a2'),
                                            ('tulu-final','chat','allenai/Llama-3.1-Tulu-3-8B','66694379')]:
                with mock.patch.object(runner.AutoTokenizer,'from_pretrained',return_value=TinyTokenizer()), \
                     mock.patch.object(runner.AutoModelForCausalLM,'from_pretrained',return_value=model), \
                     mock.patch.object(runner,'MAX_NEW',3), \
                     mock.patch.object(sys,'argv',['run131.py','--tag',tag,'--model',repo,'--revision',revision,
                         '--frame',frame,'--stage','pilot','--units',str(units),'--out',str(out),'--batch','32','--device','cpu']):
                    runner.main()
                rows=[json.loads(line) for line in (out/f'{tag}_{frame}.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),expected)
                self.assertEqual(len({(r['unit_id'],r['arm'],r['question_index']) for r in rows}),expected)

    def test_score_single_prediction_requires_48_valid_other_readings(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); answers=root/'answers'; labels=root/'labels'; answers.mkdir(); labels.mkdir()
            units=[{'id':f's{i}','class':'single','selecting_model':'llama'} for i in range(24)]
            answer_rows=[]; label_rows=[]
            for unit in units:
                for tag in ('instruct','tulu-final','hermes'):
                    row={'id':unit['id'],'unit_id':unit['id'],'unit_class':'single','arm':'pole','tag':tag,'frame':'chat','question_index':0}
                    answer_rows.append(row)
                    if tag=='instruct': label_rows.append({**row,'shows_persona':True,'coherent':True})
                    else: label_rows.append({**row,'error':'reader failed'})
            (answers/'all.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in answer_rows))
            (labels/'labels-ds.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in label_rows))
            unit_path=root/'units.json'; unit_path.write_text(json.dumps({'units':units}))
            out=root/'score.json'; score.main(['--answers',str(answers),'--labels',str(labels),'--units',str(unit_path),'--out',str(out)])
            result=json.loads(out.read_text())['predictions']['131.6']
            self.assertEqual(result['single_selecting_model_n'],24)
            self.assertEqual(result['other_observations'],0)
            self.assertFalse(result['observed'])

    def test_vast_elapsed_cap_deletes_supplied_instance(self):
        path=HERE.parent/'cloud/vast-part131-2026-10-30/vast.py'; vast=load('vast131_under_test',path)
        calls=[]
        def fake_api(method,url,body=None):
            calls.append((method,url)); return {'instances':{}} if method=='GET' else {}
        old=sys.argv; sys.argv=['vast.py','--stage','main','--instance-id','555','--started-at','0']
        try:
            with mock.patch.object(vast,'api',side_effect=fake_api),mock.patch.object(vast,'instances',return_value=[]),mock.patch('sys.stdout'):
                self.assertEqual(vast.main(),2)
        finally: sys.argv=old
        self.assertIn(('DELETE','/instances/555/'),calls)

if __name__=='__main__': unittest.main()
