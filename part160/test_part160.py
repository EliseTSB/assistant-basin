"""CPU fixture checks; importing these pure functions does not import torch."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run160 import picking_rule, cap_picks, stage2_units
from score160 import score_rows, peak_dose, prediction_results, company_hit, company_for


def source_units(n):
    return [{'id':f'u{i:03}','layer':20,'neuron':i} for i in range(n)]

def validrow(uid,dose,qi,answer,stage='screen'):
    return {'stage':stage,'unit_id':uid,'arm':'neuron','dose':dose,'question_index':qi,'answer':answer,'layer':20,'neuron':int(uid[-3:])}

def old_answers(n=32):
    return [{'id':f'u{i:03}','unit':'neuron','layer':20,'neuron':i,'question_index':0,'answer':'Same answer'} for i in range(n)]

def full_rows():
    rows=[]; ids=[f'L20N{i}' for i in range(32)]
    for i,uid in enumerate(ids):
        for arm in ('neuron','random'): rows.append({'stage':'gate2','unit_id':uid,'arm':arm,'dose':0.8,'question_index':0,'answer':'Same answer','id':f'u{i:03}'})
        for arm in ('neuron','random'):
            for dose in (.6,1.0):
                for qi in range(2):rows.append({'stage':'screen','unit_id':uid,'arm':arm,'dose':dose,'question_index':qi,'answer':'Google made this assistant','id':f'u{i:03}'})
    return rows

class Part160Tests(unittest.TestCase):
    def test_picking_requires_same_company_and_excludes_broken(self):
        rows=[]
        for uid in ('u001','u002'):
            rows += [validrow(uid,.6,0,'Google built this helpful assistant.'),validrow(uid,.6,1,'Google made this language model.')]
        rows[2]['answer']='Google Google Google'
        got=picking_rule(rows)
        self.assertEqual(got[.6],{'u001':'Google'})

    def test_cap_is_twelve_in_atlas_order(self):
        units=source_units(15);mapping={f"L20N{u['neuron']}":'Google' for u in reversed(units)}
        result=cap_picks(mapping,units)
        self.assertEqual(result['count'],15)
        self.assertEqual([u['unit_id'] for u in result['kept']],[f'L20N{i}' for i in range(12)])

    def test_both_dose_unit_runs_once_with_both_labels(self):
        p={'0.6':{'kept':[{'unit_id':'same','company':'Google'}]},'1.0':{'kept':[{'unit_id':'same','company':'Baidu'}]}}
        chosen=stage2_units(p)
        self.assertEqual(list(chosen),['same']);self.assertEqual(chosen['same']['labels'],[.6,1.0])

    def test_each_gate_can_fail(self):
        rows=full_rows(); picked={'0.6':{'kept':[]},'1.0':{'kept':[]}}
        result=score_rows(rows,picked,old_answers())
        self.assertFalse(result['gates']['gate3_picking']['passed'])
        self.assertFalse(result['gates']['gate4_controls']['passed'])
        # Gate 1: remove one planned answer. Gate 2: give the stored answers a mismatch.
        self.assertFalse(score_rows(rows[1:],picked,old_answers())['gates']['gate1_rows']['passed'])
        mismatch=old_answers()
        for row in mismatch[:5]: row['answer']='different'
        self.assertFalse(score_rows(rows,picked,mismatch)['gates']['gate2_reproduction']['passed'])

    def test_overlap_with_part148_ten_uses_real_unit_names(self):
        picked={'0.6':{'kept':[{'unit_id':'L24N5017','company':'Google'}]},'1.0':{'kept':[]}}
        self.assertEqual(score_rows(full_rows(),picked,old_answers())['summary']['overlap_with_part148_ten']['0.6'],1)

    def test_gate2_passes_on_matching_answers(self):
        r=score_rows(full_rows(),{'0.6':{'kept':[]},'1.0':{'kept':[]}},old_answers())
        self.assertTrue(r['gates']['gate2_reproduction']['passed'])
        self.assertEqual(r['gates']['gate2_reproduction']['matches'],32)

    def test_gate1_failure_directly(self):
        r=score_rows([],{'0.6':{'kept':[]},'1.0':{'kept':[]}},old_answers())
        self.assertFalse(r['gates']['gate1_rows']['passed'])

    def test_peak_ties_and_no_hits(self):
        self.assertEqual(peak_dose({.4:4,.6:8,.8:8}),.7)
        self.assertIsNone(peak_dose({.4:0,.6:0}))

    def test_prediction_boundaries(self):
        sets={'0.6':[],'1.0':[]}
        for d,kp in ((.6,.8),(1.0,1.0)):
            key=str(d)
            sets[key]=[{'unit_id':key+str(i),'company':'Google','peak_dose':kp,'hits_at_pick':12,'random_hits_at_pick':6} for i in range(6)]
        vals,_,_,_,_=prediction_results(sets,{'0.6':3,'1.0':3})
        self.assertTrue(vals['160.1']);self.assertTrue(vals['160.2']);self.assertTrue(vals['160.3'])
        self.assertTrue(vals['160.4']);self.assertTrue(vals['160.5']);self.assertTrue(vals['160.6'])
        self.assertFalse(prediction_results(sets,{'0.6':4,'1.0':3})[0]['160.3'])
        overlapping={'0.6':[dict(u,unit_id=f'shared{i}') for i,u in enumerate(sets['0.6'])],
                     '1.0':[dict(u,unit_id=f'shared{i}') for i,u in enumerate(sets['1.0'])]}
        self.assertFalse(prediction_results(overlapping,{'0.6':3,'1.0':3})[0]['160.3'])
        # Boundary excursions: one step below each registered threshold.
        for u in sets['1.0'][:5]:
            u['peak_dose']=.6;u['hits_at_pick']=11;u['random_hits_at_pick']=12;u['company']='Baidu'
        vals,_,_,_,_=prediction_results(sets,{'0.6':3,'1.0':3})
        self.assertFalse(vals['160.1']);self.assertFalse(vals['160.2']);self.assertFalse(vals['160.4']);self.assertFalse(vals['160.5']);self.assertFalse(vals['160.6'])

    def test_no_peak_dose_counts_do_not_create_median(self):
        self.assertIsNone(peak_dose({.4:0,.6:0,.8:0}))

    def test_two_company_unit_is_scored_separately_by_set(self):
        picked={'0.6':{'kept':[{'unit_id':'shared','company':'Google'}]},'1.0':{'kept':[{'unit_id':'shared','company':'Baidu'}]}}
        self.assertEqual(company_for(picked,.6,'shared'),'Google')
        self.assertEqual(company_for(picked,1.0,'shared'),'Baidu')
        self.assertTrue(company_hit('Google made this model','Google'))
        self.assertFalse(company_hit('Google made this model','Baidu'))
        self.assertTrue(company_hit('Baidu made this model','Baidu'))
        self.assertFalse(company_hit('Baidu made this model','Google'))

if __name__=='__main__':unittest.main()
