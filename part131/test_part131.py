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
        runner=load('run131_under_test',HERE/'run131.py')
        torch.manual_seed(131)
        config=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                           num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=32,pad_token_id=0,eos_token_id=2)
        model=LlamaForCausalLM(config).eval()
        ids=torch.tensor([[0,4,5],[7,8,9]]); mask=torch.tensor([[0,1,1],[1,1,1]])
        self.assertTrue(runner.gate3_noop_exact(model,model.model.layers,ids,mask,0,max_new=4))
        with torch.inference_mode(): baseline=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,max_new_tokens=4,pad_token_id=0)
        early=int(baseline[0,ids.shape[1]])
        compacted=runner.generate_batch(model,model.model.layers,ids,mask,{},0,{early},4)
        self.assertEqual(len(compacted[0]),1); self.assertGreater(len(compacted[1]),1)

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
