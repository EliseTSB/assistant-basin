"""Fixture tests for Part 154's registered rows, hooks, gates, and thresholds."""
import importlib.util
import json
import sys
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE))
import run154
import score154

class PlanningTests(unittest.TestCase):
    def test_registered_pushes_zero_sets_and_batch_sizes(self):
        self.assertEqual(len(run154.push_rows()),12)
        # 12 push arms x six zero sets for maker; 12 x two zero sets for each guard.
        zeros=[{'id':str(i),'units':[]} for i in range(6)]
        arms=run154.push_rows()
        self.assertEqual(len(run154.build_answer_batch(arms,zeros)),72)
        self.assertEqual(len(run154.build_answer_batch(arms,zeros,guards=True)),24)
        self.assertEqual(len(score154.PUSHES),12)
        self.assertEqual(len(score154.ZERO_IDS),6)
        matched={'29_776':{'27_3325':[{'layer':27,'unit':101}], '24_3949':[{'layer':27,'unit':202}]}}
        resolved=run154.zero_sets_from_data(matched)
        self.assertEqual(len(resolved),6)
        self.assertEqual([z['id'] for z in resolved],score154.ZERO_IDS)
        self.assertEqual([len(z['units']) for z in resolved],[0,1,1,1,1,1])

    def test_opening_is_fed_one_token_at_a_time(self):
        ids=[4,9,10,11]
        self.assertEqual(run154.opening_steps(ids),[[4],[9],[10],[11]])

    def test_resume_key(self):
        import tempfile
        row={'unit_id':'L11N5647','arm':'neuron','zero_set_id':'L27N3325','question_index':7}
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'answers.jsonl'
            path.write_text(json.dumps(row)+'\n',encoding='utf-8')
            self.assertIn(('L11N5647','neuron','L27N3325',7),run154.read_done(path))

class HookTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch is not installed')
    def test_row_pushes_cover_every_arm(self):
        """Pins the arms against Part 148's pushes_for, including the random row and the no-push row."""
        try:
            import torch
        except ImportError:
            self.skipTest('torch is not installed')
        sys.path.insert(0, str(HERE.parent / 'part86'))
        import common86
        class Proj:
            def __init__(self):
                self.weight = torch.nn.Parameter(torch.randn(16, 9216))
        class Layer:
            def __init__(self):
                self.mlp = type('M', (), {'down_proj': Proj()})()
        layers = {n: Layer() for n in (11, 21, 22, 23, 24, 25)}
        medians = {str(n): 1.0 for n in layers}
        rows = run154.push_rows()
        self.assertEqual(len(rows), 12)
        pushes = run154.row_pushes(rows, layers, medians, common86)
        for i, row in enumerate(rows):
            present = [m[i].count_nonzero().item() for m in pushes.values()]
            if row['arm'] == 'none':
                self.assertEqual(sum(present), 0)
            elif row['arm'] == 'random':
                self.assertEqual(sum(present), 9216)
                seed = 112000 + 10000 * 25 + 4457
                want = common86.scaled_random_directions(layers[25].mlp.down_proj.weight, 1, seed, 0.8)[0]
                self.assertTrue(torch.equal(pushes[25][i], want))
            else:
                self.assertEqual(sum(present), 1)
                self.assertNotEqual(pushes[row['layer']][i, row['neuron']].item(), 0)

    def test_zero_hook_row_and_trace(self):
        import torch
        class Hookable:
            def register_forward_pre_hook(self,fn): self.fn=fn; return type('H',(),{'remove':lambda self:None})()
        module=Hookable(); trace={'calls':0,'positions':0,'target_checks':0,'target_positions':0,'unchanged_checks':0,'unchanged_positions':0,
                                  'unzeroed_target_checks':0,'unzeroed_target_positions':0}
        hook=run154.ZeroRowsHook(module,[[],[{'layer':27,'neuron':3325}]],trace)
        x=torch.ones((2,2,3326),dtype=torch.float32)
        y=module.fn(None,(x,))[0]
        self.assertTrue(torch.equal(y[0],x[0]))
        self.assertTrue(torch.equal(y[1,:,3325],torch.zeros(2)))
        self.assertTrue(torch.equal(y[1,:,:3325],x[1,:,:3325]))
        self.assertEqual(trace['target_checks'],1)
        self.assertEqual(trace['unzeroed_target_checks'],1)
        self.assertEqual(trace['unchanged_checks'],1)
        self.assertEqual(trace['target_failures'],0)
        self.assertEqual(trace['unchanged_failures'],0)
        self.assertEqual(trace['unzeroed_target_failures'],0)
        hook.remove()

class GateAndBoundaryTests(unittest.TestCase):
    def test_each_gate_fails_its_registered_boundary(self):
        trace={'target_checks':1,'unzeroed_target_checks':1,'unchanged_checks':1}
        self.assertFalse(score154.gate1_passes(1871,1871,1871,576,576,True,trace))
        self.assertFalse(score154.gate1_passes(1872,1872,1872,576,576,True,{**trace,'target_failures':1}))
        self.assertFalse(score154.gate2_passes(3,128))
        self.assertFalse(score154.gate2_passes(2,127))
        self.assertFalse(score154.gate3_passes(19))

    def test_prediction_boundaries_and_154_5_sign(self):
        p=score154.prediction_values(g0=100,g1=50,other0=100,other1=75,matched=[75]*4,
            random0=20,random1=10,google_count=5,broken_delta_pp=5,guard0=20,guard1=23)
        self.assertTrue(all(p.values()))
        # 154.5 treats a more negative activation change as the larger move; five Google units is inclusive.
        self.assertTrue(score154.prediction_values(100,50,100,75,[75]*4,20,10,5,5,20,23)['154.5'])
        self.assertTrue(score154.larger_negative_move(-2.0,[-1.0,-0.5,-1.5]))
        self.assertFalse(score154.larger_negative_move(-1.0,[-1.0,-0.5,-1.5]))
        self.assertFalse(score154.prediction_values(100,51,100,75,[75]*4,20,10,4,5,20,23)['154.1'])
        self.assertFalse(score154.prediction_values(100,50,100,74,[75]*4,20,10,5,5,20,23)['154.2'])
        self.assertFalse(score154.prediction_values(100,50,100,75,[74,75,75,75],20,10,5,5,20,23)['154.3'])
        self.assertFalse(score154.prediction_values(100,50,100,75,[75]*4,20,11,5,5,20,23)['154.4'])
        self.assertFalse(score154.prediction_values(100,50,100,75,[75]*4,20,10,4,5,20,23)['154.5'])
        self.assertFalse(score154.prediction_values(100,50,100,75,[75]*4,20,10,5,5.01,20,23)['154.6'])
        self.assertFalse(score154.prediction_values(100,50,100,75,[75]*4,20,10,5,5,20,24)['154.7'])

if __name__=='__main__': unittest.main()
