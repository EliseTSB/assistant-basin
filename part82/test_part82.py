#!/usr/bin/env python3
"""CPU unit checks for Part 82's control design, scorer, and head hook."""
from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part82 import part82 as run
from part82 import score82 as score

def synthetic(gate=True,drop_h1=.20,drop_h2=0.,control=.01,drop_n=.10,mass=.95):
    out=[]; b_gap=.444 if gate else .30; n_gap=.205 if gate else .10
    for cond in score.CONDITIONS:
        for opt in score.OPTION_SETS:
            base=b_gap if opt=='B' else n_gap
            drop={'h0':0,'h1':drop_h1 if opt=='B' else drop_n,'h2':drop_h2,
                  'c1':control,'c2':control,'c3':control,'c4':control,'c5':control}[cond]
            gap=base-drop
            for i in range(12):
                # Modest body variation preserves exact population gap while allowing paired resampling.
                noise=(i-5.5)*.0002
                for rn in ('A','U'):
                    for order in (1,2,3):
                        p=.55+(gap+noise)/2 if rn=='A' else .55-(gap+noise)/2
                        out.append({'condition':cond,'option_set':opt,'body_id':f'b{i}',
                            'rendering':rn,'order':order,'p_you':p,'p_user':(1-p)*.6,
                            'p_system':(1-p)*.4,'letter_mass':mass})
    return out

class TestDesign(unittest.TestCase):
    def test_control_draw_deterministic_and_excludes_named_heads(self):
        a=run.conditions(); self.assertEqual(a,run.conditions())
        self.assertEqual(len(a),8)
        for name in ('c1','c2','c3','c4','c5'):
            self.assertEqual(len(a[name][11]),1); self.assertNotEqual(a[name][11],[14])
            self.assertEqual(len(a[name][15]),5)
            self.assertFalse(set(a[name][15]) & {1,2,3,12,15,10})
        self.assertEqual(a['h1'],{11:[14],15:[1,2,3,12,15]})
        self.assertEqual(a['h2'],{15:[10]})

class TestScorer(unittest.TestCase):
    def test_each_prediction_pass_and_fail(self):
        passed=score.score(synthetic())['predictions']
        for k in ('82.1','82.2','82.3','82.4','82.5','82.6'): self.assertEqual(passed[k]['verdict'],'HOLDS',k)
        # 82.1 gate failure makes 82.2-82.5 NOT READ; those predictions cannot be read as fails.
        failed=score.score(synthetic(gate=False))['predictions']
        self.assertEqual(failed['82.1']['verdict'],'FAILS')
        for k in ('82.2','82.3','82.4','82.5'): self.assertEqual(failed[k]['verdict'],'NOT READ',k)
        self.assertEqual(failed['82.6']['verdict'],'HOLDS')
        # Once the gate passes, each remaining registered rule has a concrete fail fixture.
        cases={'82.2':synthetic(drop_h1=.05),'82.3':synthetic(drop_h1=.20,control=.20),
               '82.4':synthetic(drop_h2=.06),'82.5':synthetic(drop_n=.04),'82.6':synthetic(mass=.89)}
        for k,fixture in cases.items(): self.assertEqual(score.score(fixture)['predictions'][k]['verdict'],'FAILS',k)

    def test_bootstrap_is_seeded_and_paired(self):
        x=[-.2,-.1,.1,.2]
        self.assertEqual(score.paired_bootstrap(x),score.paired_bootstrap(x))
        self.assertEqual(score.paired_bootstrap([.1]*20),[.1,.1])
        a=[.1,.2,.3,.4]; u=[0,.1,.2,.3]
        paired=[x-y for x,y in zip(a,u)]
        incorrectly_unpaired=[x-y for x,y in zip(a,reversed(u))]
        for bound in score.paired_bootstrap(paired): self.assertAlmostEqual(bound,.1)
        self.assertNotEqual(score.paired_bootstrap(paired),score.paired_bootstrap(incorrectly_unpaired))

    def test_cyclic_and_paired_drop(self):
        recs=synthetic()
        sm=score.summarize(recs,'h0','B')
        self.assertAlmostEqual(sm['cyclic']['gap'],.444,places=6)
        self.assertAlmostEqual(score.drop_summary(recs,'B','h1')['drop'],.20,places=6)

class TestHook(unittest.TestCase):
    def test_tiny_causal_lm_hook_changes_logits_and_records_live_slices(self):
        try:
            import torch
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError as e:
            self.skipTest(f'tiny-model test explicitly skipped: torch/transformers unavailable ({e})')
        torch.manual_seed(82)
        cfg=Qwen2Config(vocab_size=31,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
        model=Qwen2ForCausalLM(cfg).eval()
        ids=torch.tensor([[1,5,9,3]])
        with torch.inference_mode(): baseline=model(input_ids=ids,use_cache=False).logits
        handles,evidence=run.install_ablation(model,{1:[2]})
        try:
            self.assertEqual(evidence['1']['head_count'],4); self.assertEqual(evidence['1']['head_dim'],8)
            with torch.inference_mode(): changed=model(input_ids=ids,use_cache=False).logits
        finally:
            for h in handles: h.remove()
        self.assertGreater(float((baseline-changed).abs().max()),0.)
        self.assertEqual(evidence['1']['calls'],1)
        self.assertGreater(evidence['1']['first_call_max_abs']['2'],0.)
        handles,evidence=run.install_ablation(model,{})
        self.assertEqual(handles,[]); self.assertEqual(evidence,{})

if __name__=='__main__': unittest.main()
