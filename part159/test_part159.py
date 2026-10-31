"""Numpy-only fixtures for Part 159 scorer and literal spelling bans."""
import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from part159 import run159, score159

class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids':[ord(c) for c in text]}
    def decode(self, ids, skip_special_tokens=False): return ''.join(chr(i) for i in ids)

def veto_fixture(n=20, t=2):
    pre=np.zeros((n,40),dtype=np.int64); emit=pre.copy(); free=pre.copy()
    emit[:,t]=1
    return pre,emit,free

class BanTests(unittest.TestCase):
    def test_alibaba_is_a_literal_spelling(self):
        ids=run159.spelling_ids(FakeTokenizer(),['Alibaba'])
        self.assertTrue(ids)
        flattened={x for seq in ids for x in seq}
        self.assertIn(ord('A'),flattened)
        self.assertNotIn(ord('Q'),flattened)
        self.assertEqual(len(ids),2)  # the **-stripped sequence deduplicates with bare spelling

class GateTests(unittest.TestCase):
    def test_each_gate_can_fail(self):
        self.assertEqual(score159.gate1_status(71,True,True),'failed')
        self.assertEqual(score159.gate2_status(21),'failed')
        self.assertEqual(score159.gate3_status({'a':{'usable_count':19,'matching_and_activation_rows':list(range(19))}}),'failed')
    def test_first_veto_is_found(self):
        p=np.array([[0,0,0,0],[1,1,2,2]])
        e=np.array([[0,3,0,0],[1,1,5,2]])
        np.testing.assert_array_equal(score159.first_veto(p,e),[1,2])
    def test_differing_earlier_token_drops_gate3_row(self):
        pre,emit,free=veto_fixture()
        free[0,0]=7
        acts={6:np.zeros((20,40,2),dtype=np.float32)}
        sds={6:np.ones(2,dtype=np.float32)}
        got=score159.gate3_arm(pre,emit,free,np.ones((20,40)),np.zeros((20,40)),acts,acts,sds)
        self.assertEqual(len(got['usable_rows']),20)
        self.assertEqual(len(got['matching_and_activation_rows']),19)
        self.assertFalse(got['pass'])
    def test_one_percent_tolerance_boundary(self):
        pre,emit,free=veto_fixture(); free[:,:]=emit[:,:]
        acts={6:np.zeros((20,40,2),dtype=np.float32)}
        arm={6:np.zeros((20,40,2),dtype=np.float32)}
        sds={6:np.ones(2,dtype=np.float32)}
        arm[6][:,2,:]=.01
        got=score159.gate3_arm(pre,emit,free,np.ones((20,40)),np.zeros((20,40)),arm,acts,sds)
        self.assertTrue(got['pass'])
        arm[6][:,2,0]=.01001
        got=score159.gate3_arm(pre,emit,free,np.ones((20,40)),np.zeros((20,40)),arm,acts,sds)
        self.assertFalse(got['pass'])
    def test_rows_with_t_plus_5_over_39_excluded(self):
        pre,emit,free=veto_fixture(20,t=34); pre[19,34]=0; emit[19,34]=1
        # Make one veto late at t=35; only 19 have the full registered window.
        pre[19,34]=0; emit[19,34]=0; pre[19,35]=0; emit[19,35]=1
        acts={6:np.zeros((20,40,2),dtype=np.float32)}
        got=score159.gate3_arm(pre,emit,free,np.ones((20,40)),np.zeros((20,40)),acts,acts,{6:np.ones(2)})
        self.assertEqual(got['usable_count'],19)
        self.assertFalse(got['pass'])

class FormulaAndPredictionTests(unittest.TestCase):
    def test_prediction_boundaries(self):
        self.assertEqual(score159.prediction_status_159_1(5),'held'); self.assertEqual(score159.prediction_status_159_1(4.999),'failed')
        self.assertEqual(score159.prediction_status_159_2(20),'held'); self.assertEqual(score159.prediction_status_159_2(19),'failed')
        self.assertEqual(score159.prediction_status_159_3(2),'held'); self.assertEqual(score159.prediction_status_159_3(3),'failed')
        self.assertEqual(score159.prediction_status_159_4(4),'held'); self.assertEqual(score159.prediction_status_159_4(3),'failed')
        self.assertEqual(score159.prediction_status_159_5([.5,2,.7,.9]),'held'); self.assertEqual(score159.prediction_status_159_5([.49,2,.7,.9]),'failed')
        self.assertEqual(score159.prediction_status_159_6(12),'held'); self.assertEqual(score159.prediction_status_159_6(11),'failed')
    def test_peers_leave_out_named_channels(self):
        x=np.zeros((2,3,25),dtype=np.float32)
        x[:,:,3]=100; x[:,:,7]=200
        peers=score159.peer_channels(x,3,[3,7],20)
        self.assertNotIn(3,peers); self.assertNotIn(7,peers); self.assertEqual(len(peers),20)
    def test_peers_are_nearest_in_activity_not_quietest(self):
        x=np.zeros((1,1,50),dtype=np.float32); x[0,0,:]=np.arange(50,dtype=np.float32)
        peers=score159.peer_channels(x,30,[30],4)
        self.assertEqual(sorted(peers),[28,29,31,32])
    def test_r_hand_calculation(self):
        self.assertEqual(score159.r_value({0:1,1:3,2:5},[1,2,4]),1.5)
    def test_off_unit_reads_off(self):
        self.assertEqual(score159.reading(11,.9),'off')

if __name__=='__main__': unittest.main()
