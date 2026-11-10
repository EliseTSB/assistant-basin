from __future__ import annotations
import math
import unittest
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import torch
from tools.margin_recorder import MarginRecorder, bf16_step

class RecorderTests(unittest.TestCase):
    def test_tie_and_emitted_second(self):
        rec = MarginRecorder([9]); scores = torch.tensor([[3., 3., 0.]])
        self.assertIs(rec(torch.tensor([[1]]), scores), scores)
        row = rec.rows(torch.tensor([[1, 1]]), 1, [9])[0]
        self.assertTrue(row['emitted_is_max']); self.assertFalse(row['argmax_agrees'])
        self.assertEqual(row['top2'], [1])

    def test_early_eos_row_and_batch_three(self):
        rec = MarginRecorder([1])
        rec(torch.tensor([[0],[0],[0]]), torch.tensor([[0.,4.],[4.,0.],[4.,0.]]))
        rec(torch.tensor([[0,1],[0,0],[0,0]]), torch.tensor([[4.,0.],[4.,0.],[4.,0.]]))
        rec(torch.tensor([[0,1,1],[0,0,0],[0,0,0]]), torch.tensor([[0.,4.],[4.,0.],[4.,0.]]))
        rows=rec.rows(torch.tensor([[0,1,1,1],[0,0,0,0],[0,0,0,0]]),1,[1])
        self.assertEqual(rows[0]['tokens'],[1]); self.assertEqual(len(rows[0]['margin']),1)
        self.assertTrue(all(r['emitted_is_max'] for r in rows))

    def test_nonfinite_and_batch_one(self):
        rec=MarginRecorder(); rec(torch.tensor([[0]]),torch.tensor([[float('nan'),1.]]))
        row=rec.rows(torch.tensor([[0,1]]),1,[])[0]
        self.assertTrue(row['nonfinite']); self.assertEqual(row['tokens'],[1])

    def test_bf16_spacing(self):
        for v in (0.5, 1., 8., 16., 31., 64.):
            t=torch.tensor(v,dtype=torch.bfloat16)
            nxt=torch.nextafter(t,torch.tensor(float('inf'),dtype=torch.bfloat16))
            self.assertEqual(bf16_step(v),float((nxt.float()-t.float()).abs()))

    def test_previous_step_top2_mutation_is_detected(self):
        rec=MarginRecorder(); rec(torch.tensor([[0]]),torch.tensor([[0.,10.]])); rec(torch.tensor([[0,1]]),torch.tensor([[20.,0.]]))
        row=rec.rows(torch.tensor([[0,1,0]]),1,[])[0]
        self.assertEqual(row['top1'],[1,0]); self.assertNotEqual(row['top1'],[0,0])

if __name__=='__main__': unittest.main()
