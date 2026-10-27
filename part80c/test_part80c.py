#!/usr/bin/env python3
"""CPU tests for Part 80c: option sets, decoy pull, verdicts, and a tiny-model smoke run."""
from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part80c import part80c as run
from part80c import score80c as score
from part80b import score80b

def recs(oset, n, fn):
    out=[]
    for i in range(n):
        for rn in 'AUS':
            for k,order in enumerate(run.OPTION_SETS[oset],1):
                rp={run.ROLE_OF[o]:p for o,p in zip(order,fn(i,rn,k,[run.ROLE_OF[o] for o in order]))}
                lp=dict(zip('ABC',[rp[run.ROLE_OF[o]] for o in order]))
                out.append({'body_set':oset,'option_set':oset,'body_id':f'b{i:02d}','rendering':rn,'order':k,'letter_probs':lp,'letter_mass':1.0,'role_probs':rp})
    return out

class T(unittest.TestCase):
    def test_sets_cycle_and_wording(self):
        for oset,orders in run.OPTION_SETS.items():
            for pos in range(3): self.assertEqual(len({o[pos] for o in orders}),3)
            self.assertEqual(orders[0][0],'You'); self.assertEqual(orders[1][2],'You'); self.assertEqual(orders[2][1],'You')
            self.assertFalse(any('assistant' in o.lower() for o in orders[0] if run.ROLE_OF[o]!='decoy'))
        self.assertIn('Another AI assistant',run.OPTION_SETS['D'][0])

    def test_decoy_pull(self):
        def fn(i,rn,k,roles):
            m={'you':0.5,'user':0.2,'decoy':0.3} if rn=='A' else {'you':0.2,'user':0.7,'decoy':0.1}
            return [m[r] for r in roles]
        p=score.decoy_pull(recs('D',20,fn))
        self.assertAlmostEqual(p['pull'],0.2,places=6); self.assertGreater(p['pull_ci95'][0],0)

    def test_verdicts(self):
        def lockN(i,rn,k,roles):
            if k==2: return [0.02 if r!='you' else 0.96 for r in roles]
            you=0.6 if rn=='A' else 0.2
            return [you if r=='you' else (1-you)/2 for r in roles]
        def clean(i,rn,k,roles):
            m={'A':{'you':0.6,'user':0.1,'system':0.3,'decoy':0.3},'U':{'you':0.2,'user':0.7,'system':0.1,'decoy':0.1},'S':{'you':0.3,'user':0.2,'system':0.5,'decoy':0.5}}[rn]
            v=[m[r] for r in roles]; s=sum(v); return [x/s for x in v]
        r17={o:score80b.summarize(recs(o,20,lockN),o) for o in ('N','D')}; r17['decoy']=score.decoy_pull(recs('D',20,lockN))
        r4={o:score80b.summarize(recs(o,20,clean),o) for o in ('N','D')}; r4['decoy']=score.decoy_pull(recs('D',20,clean))
        v=score.verdicts(r17,r4,0.444)
        for k in ('80c.1','80c.2','80c.3','80c.4','80c.5','80c.6','80c.7'): self.assertEqual(v[k][0],'HOLDS',(k,v[k]))
        v=score.verdicts(r4,r17,0.1)
        self.assertEqual(v['80c.1'][0],'FAILS'); self.assertEqual(v['80c.7'][0],'FAILS')

    def test_smoke_tiny_model(self):
        try:
            from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
        except ImportError: self.skipTest('transformers missing')
        cache=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
        if not cache.exists(): self.skipTest('tokenizer not cached')
        tok=AutoTokenizer.from_pretrained(str(cache),local_files_only=True)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok),hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=2048)).eval()
        old=run.nr.load_model; run.nr.load_model=lambda _m,_r,_d:(tok,model)
        try:
            with tempfile.TemporaryDirectory() as d:
                run.run(SimpleNamespace(model='qwen3.5-4b',out=d,device='cpu',limit=3))
                rows=score80b.load(d); meta=json.loads((Path(d)/'run.json').read_text(encoding='utf-8'))
                self.assertEqual(meta['dropped_count'],0); self.assertEqual(len(rows),2*3*3*3)
                self.assertEqual({r['role_probs'].keys().__len__() for r in rows},{3})
                self.assertIn('Another AI assistant',meta['first_renderings']['D-A'])
                self.assertNotIn('AI assistant',meta['first_renderings']['N-A'].split('Who wrote')[1])
                self.assertEqual(score.decoy_pull(rows)['bodies'],3)
        finally: run.nr.load_model=old

if __name__=='__main__': unittest.main()
