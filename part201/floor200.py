#!/usr/bin/env python3
"""Build the Part 200 label-noise example for tools/shape_floor.py."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'part86')]
from part201.score201 import labels
from part163 import score163

def part163_verdicts(rows):
 """Apply Part 163's five verdict rules to its five recorded labels, grouped by unit."""
 units={}
 for r in rows:
  u=units.setdefault((r.get('arm'),r.get('unit_id')),{'name_identity':[],'maker_maker':[],'broken':[],'both_identity':[]})
  lab=r['labels']; kind=r.get('kind')
  if kind=='identity' and not lab['broken']:u['name_identity'].append(bool(lab['name_gone']));u['both_identity'].append(bool(lab.get('both_gone',False)))
  if kind=='maker' and not lab['broken']:u['maker_maker'].append(bool(lab['maker_gone']))
  u['broken'].append(bool(lab['broken']))
 arm={a:[] for a in ('neuron','random')}
 for (a,_),v in units.items():
  if a in arm:arm[a].append(v)
 def mean_rate(a,key):return sum(sum(u[key])/len(u[key]) for u in arm[a] if u[key])/max(1,sum(bool(u[key]) for u in arm[a]))
 # Same registered Part 163 directions and cut points; each value is a boolean verdict.
 return {'163.1':mean_rate('neuron','name_identity') < 1.5*mean_rate('random','name_identity'),
         '163.2':mean_rate('neuron','maker_maker') < 1.5*mean_rate('random','maker_maker'),
         '163.3':all(mean_rate(a,'name_identity')>=.15 for a in ('neuron','random')),
         '163.4':all(sum(sum(u['both_identity'])/len(u['both_identity'])>=.5 for u in arm[a] if u['both_identity']) < sum(sum(u['name_identity'])/len(u['name_identity'])>=.5 for u in arm[a] if u['name_identity']) for a in ('neuron','random')),
         '163.5':mean_rate('neuron','broken') < mean_rate('random','broken')}
def build():
 answers=[json.loads(s) for s in (ROOT/'results/part200/answers.jsonl').read_text().splitlines() if s.strip()]
 batch=[json.loads(s) for s in (ROOT/'results/part200/batch1.jsonl').read_text().splitlines() if s.strip()]
 # Exclude no-push and recipe rows; the 3,200 battery answers are the experimental rows.
 rows=[]
 for r in answers:
  if r.get('stage')!='battery':continue
  labs=labels(r.get('answer',''));labs['both_gone']=score163.name_gone(r.get('answer','')) and score163.maker_gone(r.get('answer',''))
  rows.append({'id':f"{r['unit_id']}|q{r['question_index']}",'cell':r['arm'],'unit_id':r['unit_id'],'arm':r['arm'],'kind':r['kind'],'labels':labs})
 indexed={(r.get('unit_id'),r.get('question_index')):r for r in answers if r.get('stage')=='battery'}
 pairs=[]
 for r in batch:
  if r.get('stage')!='batch1':continue
  other=indexed.get((r.get('unit_id'),r.get('question_index')))
  if other is None:continue
  la=labels(other.get('answer',''));lb=labels(r.get('answer',''))
  la['both_gone']=score163.name_gone(other.get('answer','')) and score163.maker_gone(other.get('answer',''))
  lb['both_gone']=score163.name_gone(r.get('answer','')) and score163.maker_gone(r.get('answer',''))
  pairs.append({'id':f"{r['unit_id']}|q{r['question_index']}",'unit_id':r['unit_id'],'labels_a':la,'labels_b':lb})
 out=ROOT/'results/part201';out.mkdir(parents=True,exist_ok=True)
 (out/'floor200-rows.json').write_text(json.dumps(rows)+'\n');(out/'floor200-pairs.json').write_text(json.dumps(pairs)+'\n')
 return rows,pairs
def write_report(rows, result):
 from tools.shape_floor import render, SCENARIOS
 names=[name for name,_,_ in SCENARIOS]
 text=render(result,title='Part 200 label shape-floor example')
 lines=[text.rstrip('\n'),'','## Caveat for this example','',
  'The 200 replicate pairs are all from battery question 0 (Part 200 regenerated one question one row per call), and their transitions are applied to all sixteen questions. So this table shows how the tool reads, not a calibrated floor for Part 163. Part 201 measures the rates on all sixteen.','',
  '## Simulated SD of the broken-answer count, by arm','','| Arm | '+' | '.join(names)+' |','|---|'+'---:|'*len(names)]
 for arm in ('neuron','random'):
  vals=[result['scenarios'][n]['label_counts']['broken'][arm]['simulated_sd'] for n in names]
  lines.append(f"| {arm} | "+' | '.join(f'{v:.1f}' for v in vals)+' |')
 path=ROOT/'results/part201/floor200.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8');return path

if __name__=='__main__':
 from tools.shape_floor import simulate
 rows,pairs=build()
 result=simulate(rows,pairs,part163_verdicts,draws=1000,seed=201,cluster_key='unit_id')
 path=write_report(rows,result)
 print(f'rows={len(rows)} pairs={len(pairs)} scenarios={len(result["scenarios"])} report={path}')
