#!/usr/bin/env python3
"""CPU checks for Part 84b's cells, seeds, blinding and registered rules."""
from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part84b import part84b as run
from part84b import score84b as sc

def off_fixture(counts, n=48, flip=None):
    """counts: cell -> number of OFFWORLD among the n pairs, filling pairs in order (so decode and gen overlap)."""
    pairs=[(q,s) for q in range(3) for s in range(16)]
    off={}
    for c in run.CELLS:
        k=counts.get(c,0)
        for i,(q,s) in enumerate(pairs): off[(c,q,s)]=i<k
    if flip:
        for key in flip: off[key]=not off[key]
    return off

PASS={'decode0.40':10,'gen0.40':26,'decode0.47':20,'gen0.47':30,'decode0.55':30,'gen0.55':40}

class TestDesign(unittest.TestCase):
    def test_cells_and_seeds(self):
        self.assertEqual(list(run.CELLS),['A0','decode0.40','gen0.40','decode0.47','gen0.47','decode0.55','gen0.55'])
        self.assertEqual(run.CELLS['gen0.47'],(0.47,'gen'))
        self.assertEqual(len({run.seed_for(q,s) for q in range(3) for s in range(16)}),48)
        self.assertNotEqual(run.seed_for(0,0),84000)  # not Part 84's seeds

class TestScore(unittest.TestCase):
    def test_pass_and_fail(self):
        v=sc.score(off_fixture(PASS),off_fixture(PASS))['verdicts']
        for k in ('84b.1','84b.2','84b.3','84b.4','84b.5'): self.assertEqual(v[k][0],'HOLDS',(k,v[k]))
        self.assertEqual(sc.score(off_fixture({**PASS,'A0':2}))['verdicts']['84b.2'][0],'NOT READ')
        self.assertEqual(sc.score(off_fixture({**PASS,'decode0.47':10}))['verdicts']['84b.2'][0],'FAILS')
        self.assertEqual(sc.score(off_fixture({**PASS,'gen0.55':29}))['verdicts']['84b.3'][0],'FAILS')
        small={'decode0.40':10,'gen0.40':12,'decode0.47':20,'gen0.47':22,'decode0.55':30,'gen0.55':33}
        self.assertEqual(sc.score(off_fixture(small))['verdicts']['84b.3'][0],'FAILS')  # summed difference 7 < 12
        self.assertEqual(sc.score(off_fixture({**PASS,'gen0.40':20}))['verdicts']['84b.4'][0],'FAILS')
        flips=[('A0',q,s) for q in range(3) for s in range(16)][:40]
        self.assertEqual(sc.score(off_fixture(PASS),off_fixture(PASS,flip=flips))['verdicts']['84b.5'][0],'FAILS')

    def test_paired_counts_and_sign_test(self):
        r=sc.score(off_fixture(PASS))
        self.assertEqual(r['paired']['0.40'],{'gen_only':16,'decode_only':0,'sign_test_p':r['paired']['0.40']['sign_test_p']})
        self.assertAlmostEqual(sc.sign_test(16,0),2/2**16)
        self.assertEqual(sc.sign_test(0,0),1.0); self.assertAlmostEqual(sc.sign_test(3,3),1.0)

    def test_prepare_and_load_refuse_bad_labels(self):
        recs=[{'cell':c,'question_index':q,'sample':s,'answer':f'{c}|{q}|{s}'} for c in run.CELLS for q in range(3) for s in range(16)]
        d=tempfile.mkdtemp(); key=sc.prepare(recs,d+'/items.jsonl',d+'/key.json')
        items=[json.loads(l) for l in open(d+'/items.jsonl')]
        self.assertEqual(len(items),336); self.assertTrue(all(set(x)=={'id','text'} for x in items))
        rows=[{'id':x['id'],'origin':'NEITHER','form':'OK'} for x in items]
        off=sc.load_labels(rows,key); self.assertEqual(len(off),336)
        with self.assertRaises(ValueError): sc.load_labels(rows[:-1],key)
        with self.assertRaises(ValueError): sc.load_labels(rows+[rows[0]],key)

if __name__=='__main__': unittest.main()
