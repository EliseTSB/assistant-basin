#!/usr/bin/env python3
"""CPU checks for Part 84's position modes, additive hook and neuronary-rule median."""
from __future__ import annotations
import sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part84 import part84 as run

class TestDesign(unittest.TestCase):
    def test_positions(self):
        self.assertEqual(run.positions('decode',5,True),[])
        self.assertEqual(run.positions('decode',1,False),[0])
        self.assertEqual(run.positions('gen',5,True),[4])
        self.assertEqual(run.positions('gen',1,False),[0])
        with self.assertRaises(ValueError): run.positions('both',3,True)

    def test_arms_and_seeds(self):
        self.assertIsNone(run.ARMS['A0'])
        self.assertEqual(run.ARMS['A2'],(2742,0.63,'decode')); self.assertEqual(run.ARMS['A3'],(2742,0.63,'gen'))
        self.assertEqual(run.ARMS['C'],(8870,0.63,'decode')); self.assertEqual(run.ARMS['A1'][1],0.47)
        seeds={run.seed_for(q,s) for q in range(3) for s in range(8)}; self.assertEqual(len(seeds),24)

def tiny():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(84)
    cfg=Qwen2Config(vocab_size=37,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,eos_token_id=36,pad_token_id=36)
    return Qwen2ForCausalLM(cfg).eval()

class TestHook(unittest.TestCase):
    def setUp(self):
        try:
            import torch, transformers  # noqa: F401
        except ImportError as e:
            self.skipTest(f'tiny-model test explicitly skipped: torch/transformers unavailable ({e})')

    def test_decode_mode_leaves_the_prompt_pass_alone_and_gen_mode_changes_its_last_position(self):
        import torch
        m=tiny(); ids=torch.tensor([[1,5,9,3,7]]); down=m.model.layers[1].mlp.down_proj
        with torch.inference_mode(): base=m(input_ids=ids).logits
        for mode,expect_change in (('decode',False),('gen',True)):
            h=run.AddHook(down,3,25.0,mode)
            try:
                with torch.inference_mode(): got=m(input_ids=ids).logits
            finally: h.remove()
            last_diff=float((got[0,-1]-base[0,-1]).abs().max()); early_diff=float((got[0,:-1]-base[0,:-1]).abs().max())
            self.assertEqual(early_diff,0.0,mode)
            if expect_change: self.assertGreater(last_diff,0.0)
            else: self.assertEqual(last_diff,0.0)
            self.assertEqual(h.counts['prefill_calls'],1)
            self.assertEqual(h.counts['prefill_positions'],1 if mode=='gen' else 0)

    def test_generate_counts_and_decode_steps_change_output(self):
        import torch
        m=tiny(); ids=torch.tensor([[1,5,9,3,7]]); down=m.model.layers[1].mlp.down_proj
        with torch.inference_mode(): plain=m.generate(input_ids=ids,max_new_tokens=8,do_sample=False,min_new_tokens=8)
        h=run.AddHook(down,3,400.0,'decode')
        try:
            h.new_sequence()
            with torch.inference_mode(): steered=m.generate(input_ids=ids,max_new_tokens=8,do_sample=False,min_new_tokens=8)
        finally: h.remove()
        self.assertEqual(h.counts['prefill_calls'],1); self.assertEqual(h.counts['prefill_positions'],0)
        self.assertEqual(h.counts['decode_calls'],7); self.assertEqual(h.counts['decode_positions'],7)
        self.assertEqual(int(plain[0,5]),int(steered[0,5]))  # first answer token comes from the untouched prompt pass
        self.assertNotEqual(plain[0,6:].tolist(),steered[0,6:].tolist())

    def test_residual_median_matches_a_manual_computation(self):
        import torch
        m=tiny()
        class Tok:
            def __call__(self,text,return_tensors=None):
                class E: pass
                e=E(); e.input_ids=torch.tensor([[1,4,8,2,6,3]]); return e
        got=run.residual_median(m,Tok(),m.model.layers[0],'x','cpu')
        with torch.inference_mode(): hs=m(input_ids=torch.tensor([[1,4,8,2,6,3]]),output_hidden_states=True).hidden_states[1]  # layer 0's output; the last entry is after the final norm
        self.assertAlmostEqual(got,float(hs[0,1:].float().norm(dim=-1).median()),places=5)

if __name__=='__main__': unittest.main()
