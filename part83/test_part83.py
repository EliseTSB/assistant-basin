#!/usr/bin/env python3
"""CPU unit checks for Part 83's conditions, resume, readout and scorer."""
from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part83 import part83 as run
from part83 import score83 as score
from part82 import part82 as run82

LYRA=['L11h14','L15h1','L15h2','L15h3','L15h12','L15h15']
PASS_DROPS={'L15h6':.15,'L15h10':.061,'L11h3':.07,**{c:.03 for c in LYRA}}
PASS_USHIFT={'L15h0':.06}

def rows(cond,gap,ushift,mass,option_set=None,n=12):
    out=[]
    for i in range(n):
        noise=(i-5.5)*.0002
        for rn in ('A','U'):
            for order in (1,2,3):
                p=(.55+(gap+noise)/2 if rn=='A' else .55-(gap+noise)/2)+ushift
                r={'condition':cond,'body_id':f'b{i}','rendering':rn,'order':order,'p_you':p,
                   'p_user':(1-p)*.6,'p_system':(1-p)*.4,'letter_mass':mass}
                if option_set: r['option_set']=option_set
                out.append(r)
    return out

def synth83(drops=PASS_DROPS,ushift=PASS_USHIFT,h0gap=.444,mass=.95):
    out=[]
    for c in run.conditions(): out+=rows(c,h0gap-drops.get(c,0),ushift.get(c,0),mass)
    return out

def synth82(h1=.369,h2=.061):
    drops={'h0':0,'h1':h1,'h2':h2,'c1':.15,'c2':.11,'c3':.35,'c4':.22,'c5':.50}
    out=[]
    for c,d in drops.items(): out+=rows(c,.444-d,0,.95,'B')+rows(c,.205,0,.95,'N')
    meta={'conditions':{c:{str(k):v for k,v in h.items()} for c,h in run82.conditions().items()}}
    return out,meta

def scored(**kw):
    r82,m82=synth82()
    return score.score(synth83(**kw),r82,m82)

class TestDesign(unittest.TestCase):
    def test_conditions_are_h0_plus_32_single_heads(self):
        c=run.conditions(); self.assertEqual(len(c),33); self.assertEqual(c['h0'],{})
        for L in (11,15):
            for h in range(16): self.assertEqual(c[f'L{L}h{h}'],{L:[h]})

    def test_resume_keeps_complete_conditions_and_drops_partial_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'records.jsonl'
            recs=rows('h0',.4,0,.95,n=2)+rows('L11h0',.4,0,.95,n=1)
            p.write_text(''.join(json.dumps(r)+'\n' for r in recs),encoding='utf-8')
            self.assertEqual(run.completed(p,2),{'h0'})
            left=[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(left),12); self.assertTrue(all(r['condition']=='h0' for r in left))
            self.assertEqual(run.completed(Path(d)/'missing.jsonl',2),set())

class TestScorer(unittest.TestCase):
    def test_pass_fixture_holds_every_rule(self):
        p=scored()['predictions']
        for k in ('83.1','83.2','83.3','83.4','83.5','83.6','83.7','83.8'): self.assertEqual(p[k]['verdict'],'HOLDS',(k,p[k]))

    def test_gate_failure_makes_the_rest_not_read(self):
        p=scored(h0gap=.40)['predictions']
        self.assertEqual(p['83.1']['verdict'],'FAILS')
        for k in ('83.2','83.3','83.4','83.5','83.6','83.7'): self.assertEqual(p[k]['verdict'],'NOT READ',k)
        self.assertEqual(p['83.8']['verdict'],'HOLDS')
        # The second half of the gate: L15h10 alone must reproduce Part 82's h2 drop within 0.02.
        q=scored(drops={**PASS_DROPS,'L15h10':.10})['predictions']
        self.assertEqual(q['83.1']['verdict'],'FAILS')

    def test_each_rule_has_a_failing_fixture(self):
        cases={'83.2':dict(drops={**PASS_DROPS,'L11h3':.20}),
               '83.3':dict(drops={**PASS_DROPS,'L15h6':.08}),
               '83.4':dict(ushift={'L15h0':.03}),
               '83.5':dict(drops={**PASS_DROPS,'L15h1':.12}),
               '83.6':dict(drops={**PASS_DROPS,'L11h3':.01}),
               '83.7':dict(drops={**PASS_DROPS,**{c:.07 for c in LYRA}}),
               '83.8':dict(mass=.89)}
        for k,kw in cases.items(): self.assertEqual(scored(**kw)['predictions'][k]['verdict'],'FAILS',k)

    def test_additivity_arithmetic(self):
        a=scored()['additivity']
        self.assertAlmostEqual(a['h1']['set_drop'],.369,places=6)
        self.assertAlmostEqual(a['h1']['sum_single_drops'],.18,places=6)
        self.assertAlmostEqual(a['h1']['set_minus_sum'],.189,places=6)
        self.assertAlmostEqual(a['h2']['set_minus_sum'],0.,places=6)
        self.assertEqual(sorted(a),['c1','c2','c3','c4','c5','h1','h2'])
        self.assertEqual(a['c5']['heads'],[f'L11h{run82.conditions()["c5"][11][0]}']+[f'L15h{h}' for h in run82.conditions()['c5'][15]])

    def test_scorer_ignores_part82_set_n(self):
        g=scored()['gate']; self.assertAlmostEqual(g['part82_h0_gap'],.444,places=6)

class TestReadout(unittest.TestCase):
    def test_last_only_logits_match_full_logits(self):
        try:
            import torch
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError as e:
            self.skipTest(f'tiny-model test explicitly skipped: torch/transformers unavailable ({e})')
        torch.manual_seed(83)
        cfg=Qwen2Config(vocab_size=31,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
        model=Qwen2ForCausalLM(cfg).eval(); ids=torch.tensor([[1,5,9,3,7]])
        full=run.letter_probs(model,ids,[2,3,4],False); last=run.letter_probs(model,ids,[2,3,4],True)
        for a,b in zip(full,last): self.assertAlmostEqual(a,b,places=6)
        handles,_=run.install_ablation(model,{1:[2]})
        try: changed=run.letter_probs(model,ids,[2,3,4],True)
        finally:
            for h in handles: h.remove()
        self.assertGreater(max(abs(a-b) for a,b in zip(full,changed)),0.)

if __name__=='__main__': unittest.main()
