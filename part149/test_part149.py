"""Fixture tests for Part 149 damage, gates, statistics and entry imports."""
import importlib.util, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
run=load('run149',HERE/'run149.py')
score=load('score149',HERE/'score149.py')

class DamageTests(unittest.TestCase):
    def test_hand_cases(self):
        shuffled=run.damage('I am here.','shuf-all',9)
        self.assertEqual(shuffled.split()[0],'I') # one-letter word stays itself
        self.assertTrue(shuffled.endswith('.')) # punctuation stays in place
        self.assertEqual(run.damage('!!!','shuf-all',9),'!!!')
        self.assertEqual(run.damage('aaaa bbbb?','shuf-all',19),'aaaa bbbb?') # equal letters cannot change
        self.assertEqual(len(run.damage('one two three','words',4).split()),3)
        self.assertEqual(run.damage('I am here.','letters',5).count(' '),2)
    def test_shuffle_redraws_when_only_digit_words_were_picked(self):
        self.assertEqual(run.damage('What is 12 times 12?','shuf-1/3',149903),'What si 12 times 12?')
        for seed in range(149900,149908):
            self.assertNotEqual(run.damage('What is 12 times 12?','shuf-1/3',seed),'What is 12 times 12?')

    def test_deterministic_seed(self):
        for arm in ('shuf-1/3','shuf-2/3','shuf-all','words','letters'):
            self.assertEqual(run.damage('Who are you? Hello there.','%s'%arm,149001),run.damage('Who are you? Hello there.',arm,149001))
    def test_token_damage_count(self):
        class Tok:
            def __call__(self,text,add_special_tokens=False):return {'input_ids':list(text)}
            def __len__(self):return 256
            def decode(self,ids,skip_special_tokens=False):return ''.join(map(chr,ids))
        t=Tok(); out=run.damage('abc?','tokens',149,t)
        self.assertNotEqual(out,'abc?');self.assertEqual(len(t(out)['input_ids']),4)

class StatisticTests(unittest.TestCase):
    def test_partial_spearman_hand_case(self):
        # x and y share a perfect rank trend after the nuisance variable is removed.
        x=[1,2,3,4,5];y=[2,4,6,8,10];z=[5,4,3,2,1]
        self.assertAlmostEqual(score.partial_spearman(x,y,z),1.0)
    def test_prediction_threshold_boundaries(self):
        self.assertTrue(score.meets_prediction('149.1',[{'partial_spearman':.3}]*3+[{'partial_spearman':.299}]))
        self.assertFalse(score.meets_prediction('149.1',[{'partial_spearman':.3}]*2+[{'partial_spearman':.299}]*2))
        self.assertTrue(score.meets_prediction('149.2',{'ratio':10}))
        self.assertFalse(score.meets_prediction('149.2',{'ratio':9.999}))
        self.assertTrue(score.meets_prediction('149.3',{'questions_at_least_0.8':18}))
        self.assertFalse(score.meets_prediction('149.3',{'questions_at_least_0.8':17}))
        for pid in ('149.4','149.5'):
            self.assertTrue(score.meets_prediction(pid,{'rho':.5}));self.assertFalse(score.meets_prediction(pid,{'rho':.499}))
        self.assertTrue(score.meets_prediction('149.6',{'words':1/3,'letters':1}))
        self.assertFalse(score.meets_prediction('149.6',{'words':.333,'letters':1}))
        self.assertTrue(score.meets_prediction('149.7',{'base':.2,'sft':1,'instruct':1}))
        self.assertFalse(score.meets_prediction('149.7',{'base':.201,'sft':1,'instruct':1}))
        self.assertTrue(score.meets_prediction('149.8',{'questions_at_least_5x':18}))
        self.assertFalse(score.meets_prediction('149.8',{'questions_at_least_5x':17}))
    def test_spearman_08_rounding(self):
        # Four ranks produce steps of .2; binary float stores .8 just below .8.
        cor=[{'spearman':0.7999999999999999},{'spearman':.8},{'spearman':.6}]
        self.assertEqual(score.count_at_least(cor),2)
    def test_each_gate_rejects_missing_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'instruct.jsonl').write_text('');(root/'sft.jsonl').write_text('');(root/'base.jsonl').write_text('');(root/'prompts-instruct.jsonl').write_text('')
            gates,_=score._gate_rows(root,root/'part146')
            self.assertEqual(set(gates),{'gate1_noop','gate2_part146','gate3_finite','gate4_decline','gate5_damage'})
            self.assertTrue(all(not v['passed'] for v in gates.values()))

    def test_all_gates_can_pass_fixture(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'part146';old.mkdir()
            # Gate 2 reference arrays and token index.
            for q in range(1,5):
                info={'sorry_id':4}
                (old/f'index-full-q{q}.json').write_text(json.dumps(info))
                plain=np.zeros((1,8),dtype='f');plain[0,4]=-1;np.save(old/f'plain-instruct-q{q}.npy',plain)
                empty=np.zeros(8,dtype='f');empty[4]=-1;np.savez(old/f'refs-instruct-q{q}.npz',empty=empty)
            ins=[]
            for q in range(1,25):
                ins.append({'q':q,'arm':'real','logp_sorry':-1.0})
                ins.append({'q':q,'arm':'empty','logp_sorry':-1.0})
                for j in range(12):ins.append({'q':q,'arm':'noop','logp_sorry':-1.0})
            ins.extend({'q':i+1,'arm':'decline','logp_sorry':-0.1} for i in range(12))
            (root/'instruct.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ins))
            (root/'sft.jsonl').write_text(json.dumps({'q':1,'arm':'real','logp_sorry':-1})+'\n')
            (root/'base.jsonl').write_text(json.dumps({'q':1,'arm':'real','logp_sorry':-1})+'\n')
            prompts=[]
            for q in range(1,25):
                question=run.QUESTIONS[q-1]
                prompts += [{'q':q,'question':question,'arm':'real','prompt':question,'seed':None,'token_count':len(question),'word_count':len(question.split()),'prompt_token_ids':[2]*len(question)}]
                for a in ('shuf-1/3','shuf-2/3','shuf-all','words','letters','tokens'):
                    text=('x '*len(question.split())).strip() if a in ('words','shuf-1/3','shuf-2/3','shuf-all') else ('x'*len(question) if a=='letters' else 'z'*len(question))
                    r={'q':q,'question':question,'arm':a,'prompt':text,'seed':None,'token_count':len(question),'word_count':len(text.split())}
                    if a=='tokens':r['prompt_token_ids']=[1]*len(question)
                    prompts.append(r)
                prompts += [{'q':q,'question':question,'arm':'empty','prompt':'','seed':None,'token_count':0,'word_count':0}]
            (root/'prompts-instruct.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prompts))
            gates,_=score._gate_rows(root,old)
            self.assertTrue(all(x['passed'] for x in gates.values()),gates)

    def test_gate_damage_passes_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rows=[]
            for q in range(1,25):
                question=run.QUESTIONS[q-1]
                for arm in ('real','shuf-1/3','shuf-2/3','shuf-all','words','letters','tokens','empty'):
                    text=question if arm=='real' else '' if arm=='empty' else ('x '*len(question.split())).strip() if arm in ('shuf-1/3','shuf-2/3','shuf-all','words') else 'x'*len(question)
                    rows.append({'q':q,'question':question,'arm':arm,'prompt':text,'seed':None,'token_count':len(question),'word_count':len(question.split()),'prompt_token_ids':[1]*len(question) if arm in ('real','tokens') else None})
            (root/'prompts-instruct.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            # Directly exercise the damage gate after adding its other expected prompt rows.
            gates,_=score._gate_rows(root,root/'part146')
            self.assertTrue(gates['gate5_damage']['passed'])

class EntryTests(unittest.TestCase):

    def test_runner_entry_with_stubbed_loader_and_resume(self):
        import contextlib, types, numpy as np
        class Tensor:
            def __init__(self,a):self.a=np.asarray(a)
            def __getitem__(self,k):return Tensor(self.a[k])
            def float(self):return self
            def detach(self):return self
            def cpu(self):return self
            def all(self):return bool(self.a.all())
            def numpy(self):return self.a
        class LP(Tensor):
            def all(self):return bool(self.a.all())
        fake=types.ModuleType('torch');fake.long='long';fake.inference_mode=contextlib.nullcontext
        fake.tensor=lambda a,dtype=None,device=None:Tensor(a)
        fake.log_softmax=lambda x,dim=-1:LP(x.a-np.log(np.exp(x.a).sum(axis=dim,keepdims=True)))
        fake.isfinite=lambda x:Tensor(np.isfinite(x.a) if isinstance(x,Tensor) else np.isfinite(x))
        class Tok:
            pad_token_id=0;eos_token_id=0
            def __call__(self,text,add_special_tokens=False,**kw):
                specials={'I':[1],'Sorry':[2],"'m":[3],' sorry':[4],"I'm":[1,3]}
                ids=specials.get(text,[ord(c)%90+10 for c in text])
                if kw.get('return_offsets_mapping'):return {'input_ids':ids,'offset_mapping':[(i,i+1) for i in range(len(ids))]}
                return {'input_ids':ids}
            def __len__(self):return 100
            def decode(self,ids,skip_special_tokens=False):return ''.join(chr((int(i)%26)+97) for i in ids)
            def apply_chat_template(self,msgs,**kw):return msgs[0]['content']+' assistant'
        class Layer:
            def register_forward_hook(self,*a):raise AssertionError('plain row should not install a hook')
        class Model:
            def __init__(self):self.model=types.SimpleNamespace(layers=[Layer() for _ in range(16)]);self.calls=0
            def __call__(self,input_ids,**kw):
                self.calls+=1;return types.SimpleNamespace(logits=Tensor(np.zeros((1,1,100))),past_key_values=object())
        model=Model();tok=Tok();sys.modules['torch']=fake
        original=run.QUESTIONS;run.QUESTIONS=('Who are you?',)
        try:
            with tempfile.TemporaryDirectory() as d:
                units=Path(d)/'units.json';units.write_text(json.dumps({'stage_f':{'units':[]}}))
                args=types.SimpleNamespace(out=str(Path(d)/'out'),model='stub',device='cpu',tokenizer=None,units=str(units),stage='base',pushed=False)
                run.run(args,loader=lambda *_:(tok,model));first=model.calls
                self.assertGreater(first,0)
                rows=(Path(args.out)/'base.jsonl').read_text().splitlines()
                run.run(args,loader=lambda *_:(tok,model))
                self.assertEqual(model.calls,first)
                self.assertEqual(len((Path(args.out)/'base.jsonl').read_text().splitlines()),len(rows))
                self.assertTrue((Path(args.out)/'means-base.npz').is_file())
        finally:
            run.QUESTIONS=original;sys.modules.pop('torch',None)

    def test_isolated_imports_from_repo_root(self):
        for script in ('run149.py','score149.py'):
            r=subprocess.run([sys.executable,'-I','-B',str(HERE/script),'--help'],cwd=ROOT,text=True,capture_output=True,timeout=20)
            self.assertEqual(r.returncode,0,r.stderr)

    def test_score_end_to_end_keeps_the_base_rows(self):
        """The whole scorer on small fixtures, so a name reused inside score() fails here."""
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'part146';old.mkdir()
            for q in range(1,5):
                (old/f'index-full-q{q}.json').write_text(json.dumps({'sorry_id':4}))
                plain=np.linspace(-1,-8,8,dtype='f')[None,:];np.save(old/f'plain-instruct-q{q}.npy',plain);np.save(old/f'plain-base-q{q}.npy',plain)
                empty=plain[0][::-1].copy();np.savez(old/f'refs-instruct-q{q}.npz',empty=empty)
                rng=np.random.default_rng(q);np.save(old/f'matrix-full-q{q}.npy',plain+rng.normal(size=(6,8)).astype('f'))
            def rows(stage):
                out=[]
                for q in range(1,25):
                    for arm in ('real','shuf-1/3','shuf-2/3','shuf-all','words','letters','tokens','empty'):
                        out.append({'q':q,'arm':arm,'key':f'{arm}{q}','logp_sorry':-5.0 if arm=='real' else -1.0-q/100,'logp_I':-2.0,'logp_m_given_I':-1.0,'stage':stage})
                return out
            ins=rows('instruct')
            for r in ins:
                if r['arm']=='real' and r['q']<=4:r['logp_sorry']=float(np.load(old/f'plain-instruct-q{r["q"]}.npy')[0,4])
                if r['arm']=='empty' and r['q']<=4:r['logp_sorry']=float(np.load(old/f'refs-instruct-q{r["q"]}.npz')['empty'][4])
            real={r['q']:r['logp_sorry'] for r in ins if r['arm']=='real'}
            for q in range(1,5):
                ins.extend({'q':q,'arm':'noop','key':f'n{q}{j}','logp_sorry':real[q],'logp_I':-1.0,'logp_m_given_I':-1.0} for j in range(3))
            for q in range(1,25):
                for arm in ('pushed-full','pushed-prompt-only'):
                    for kind in ('neuron','random'):
                        ins.append({'q':q,'arm':arm,'key':f'{arm}{q}{kind}','unit':kind,'logp_sorry':-2.0-q/100,'logp_I':-0.5,'logp_m_given_I':-0.2})
            ins.extend({'q':i+1,'arm':'decline','key':f'd{i}','logp_sorry':-0.1,'logp_I':-1.0,'logp_m_given_I':-1.0} for i in range(12))
            (root/'instruct.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ins))
            for stage in ('sft','base'):
                (root/f'{stage}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows(stage)))
            (root/'prompts-instruct.jsonl').write_text(json.dumps({'q':1,'arm':'real','prompt':'Who are you?','question':'Who are you?','seed':None})+'\n')
            result=score.score(root,old,root/'out')
            self.assertEqual(set(result['predictions']),{f'149.{i}' for i in range(1,9)})
            self.assertIn(result['predictions']['149.7']['verdict'],('held','failed'))
            self.assertAlmostEqual(result['predictions']['149.7']['estimate']['base'],result['predictions']['149.7']['estimate']['sft'])

if __name__=='__main__':unittest.main()
