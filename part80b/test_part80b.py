#!/usr/bin/env python3
"""CPU tests for Part 80b's runner helpers and scorer, plus a smoke run on a random tiny Qwen2."""
from __future__ import annotations
import sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part80b import part80b as run
from part80b import score80b as score

def recs(body_set, n, fn):
    """fn(body_index, rendering, order) -> (letter probs for A, B, C)."""
    out=[]
    for i in range(n):
        for rn in 'AUS':
            for k,order in enumerate(run.ORDERS,1):
                lp=dict(zip('ABC',fn(i,rn,k)))
                out.append({'body_set':body_set,'body_id':f'b{i:02d}','rendering':rn,'order':k,'letter_probs':lp,'letter_mass':1.0,
                            'role_probs':{run.ROLE_OF[order[j]]:lp['ABC'[j]] for j in range(3)}})
    return out

def by_role(rn, k, you, user, system):
    order=run.ORDERS[k-1]; m={'you':you,'user':user,'system':system}
    return [m[run.ROLE_OF[o]] for o in order]

class T(unittest.TestCase):
    def test_orders_cyclic(self):
        for pos in range(3): self.assertEqual({o[pos] for o in run.ORDERS},{run.YOU,run.USER,run.SYSTEM})
        self.assertEqual(run.ORDERS[0],[run.YOU,run.USER,run.SYSTEM]); self.assertEqual(run.ORDERS[1],[run.USER,run.SYSTEM,run.YOU])

    def test_body_check(self):
        s='<|im_start|>assistant\nhello there<|im_end|>\n<|im_start|>user\nQ'
        self.assertTrue(run.body_in_rendering(s,'A','hello there'))
        self.assertFalse(run.body_in_rendering(s,'A','hello there\n'))
        self.assertFalse(run.body_in_rendering(s,'S','hello there'))

    def test_lock_detected_and_role_gap(self):
        # Order 2 locks on C for every rendering; orders 1 and 3 carry a role gap of 0.4.
        def fn(i,rn,k):
            if k==2: return [0.02,0.02,0.96]
            you=0.6 if rn=='A' else 0.2
            return by_role(rn,k,you,(1-you)/2,(1-you)/2)
        s=score.summarize(recs('smol',30,fn),'smol')
        self.assertEqual(s['locked'],{1:None,2:'C',3:None})
        self.assertAlmostEqual(s['orders'][1]['gap'],0.4,places=6)
        self.assertEqual(score.self_reference(s)[0],'HOLDS')

    def test_two_locks_not_read(self):
        def fn(i,rn,k): return [0.96,0.02,0.02] if k in (2,3) else by_role(rn,k,0.6 if rn=='A' else 0.2,0.2,0.2)
        s=score.summarize(recs('smol',10,fn),'smol')
        self.assertEqual(score.self_reference(s)[0],'NOT READ')

    def test_gap_in_one_unlocked_order_fails(self):
        def fn(i,rn,k):
            you=0.6 if (rn=='A' and k==1) else 0.3
            return by_role(rn,k,you,(1-you)/2,(1-you)/2)
        s=score.summarize(recs('smol',20,fn),'smol')
        self.assertEqual(s['locked'],{1:None,2:None,3:None})
        self.assertEqual(score.self_reference(s)[0],'FAILS')

    def test_noisy_gap_interval_crosses_zero(self):
        # Half the bodies +0.8, half -0.4: mean 0.2, above the 0.10 line, but the interval includes 0 for n=6.
        def fn(i,rn,k):
            you={'A':0.9 if i%2==0 else 0.1,'U':0.1 if i%2==0 else 0.5}.get(rn,0.3)
            return by_role(rn,k,you,(1-you)/2,(1-you)/2)
        s=score.summarize(recs('smol',6,fn),'smol')
        self.assertGreaterEqual(s['orders'][1]['gap'],0.10)
        self.assertLessEqual(s['orders'][1]['gap_ci95'][0],0)
        self.assertEqual(score.self_reference(s)[0],'FAILS')

    def test_verdicts_end_to_end(self):
        def lock2(i,rn,k): return [0.02,0.02,0.96] if k==2 else by_role(rn,k,0.6 if rn=='A' else 0.2,0.2,0.2)
        def clean(i,rn,k): return by_role(rn,k,{'A':0.8,'S':0.5,'U':0.3}[rn],0.1,0.1)
        r17={bs:score.summarize(recs(bs,20,lock2),bs) for bs in ('smol','qwenown')}
        r4={bs:score.summarize(recs(bs,20,clean),bs) for bs in ('smol','qwenown')}
        v=score.verdicts(r17,r4)
        self.assertEqual(v['80b.1'][0],'HOLDS'); self.assertEqual(v['80b.2'][0],'HOLDS'); self.assertEqual(v['80b.3'][0],'HOLDS')
        self.assertEqual(v['80b.4'][0],'HOLDS'); self.assertEqual(v['80b.5'][0],'HOLDS'); self.assertEqual(v['80b.6'][0],'HOLDS')
        self.assertEqual(v['80b.7'][0],'HOLDS')
        v=score.verdicts(r4,r17)
        self.assertEqual(v['80b.1'][0],'FAILS'); self.assertEqual(v['80b.2'][0],'FAILS')

    def test_smoke_tiny_model(self):
        try:
            import torch
            from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
        except ImportError: self.skipTest('torch or transformers missing')
        cache=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
        if not cache.exists(): self.skipTest('Qwen3.5-4B tokenizer not cached')
        tok=AutoTokenizer.from_pretrained(str(cache),local_files_only=True)
        cfg=Qwen2Config(vocab_size=len(tok),hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=2048)
        model=Qwen2ForCausalLM(cfg).eval()
        old=run.nr.load_model; run.nr.load_model=lambda _m,_r,_d:(tok,model)
        try:
            with tempfile.TemporaryDirectory() as d:
                run.run(SimpleNamespace(model='qwen3.5-4b',out=d,device='cpu',limit=3))
                rows=score.load(d)
                meta=__import__('json').loads((Path(d)/'run.json').read_text(encoding='utf-8'))
                self.assertEqual(len(rows)+27*meta['dropped_count'],2*3*3*3)
                self.assertEqual(meta['dropped_count'],0)
                for r in rows: self.assertAlmostEqual(sum(r['role_probs'].values()),1.0,places=5)
                s=score.summarize(rows,'smol'); self.assertEqual(s['bodies'],3)
        finally: run.nr.load_model=old

if __name__=='__main__': unittest.main()
