#!/usr/bin/env python3
"""Independent raw-row analysis for Part 154. Run from any directory with python3."""
from __future__ import annotations
import json, re, sys
from collections import Counter
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
PART86 = ROOT / 'part86'
if str(PART86) not in sys.path: sys.path.insert(0, str(PART86))
import triage86

ANS = HERE / 'answers.jsonl'
STAGE = HERE / 'stageA.jsonl'
TRACE = HERE / 'trace.json'
UNITS = [
 ('L11N5647','Google'), ('L21N7134','Baidu'), ('L22N6997','Google'),
 ('L23N414','Google'), ('L23N1743','Huawei'), ('L23N2993','Microsoft'),
 ('L24N5017','Google'), ('L25N140','Google'), ('L25N4457','Google'), ('L25N4458','Google')]
GOOGLE = [u for u,c in UNITS if c == 'Google']
OTHER = [u for u,c in UNITS if c != 'Google']
PUSHES = [u for u,_ in UNITS] + ['random_L25N4457']
ZSETS = ['none','L27N3325','matched_0','matched_1','matched_2','matched_3']
NAMES = ['Google','Alibaba','Tongyi','Microsoft']
# The initial rule is intentionally mechanical: own-maker strings, a short list of real
# companies/institutions, an explicit maker construction, otherwise no maker.
REAL = re.compile(r'\b(?:Microsoft|NVIDIA|NIST|ChatGPT|Apple|Baidu|Huawei|Tencent|Taobao|Alibaba|Google|OpenAI|Anthropic|Meta|DeepSeek)\b', re.I)
OWN = re.compile(r'\b(?:Qwen(?:3\.5)?|Tongyi(?:\s+Lab|\s+Qianwen)?)\b|阿里巴巴|通义', re.I)
MAKER = re.compile(r'\b(?:created|developed|trained|built|designed|made|produced|from)\s+by\b|\bcreator\s+is\b|\bcreated\s+by\b', re.I)
# Hand corrections after reading every displaced answer; keyed by unit/question index.
MANUAL = {
 ('L11N5647',7): 'no maker', # cold check: names Google only and was dropped by the repetition rule; reported in the README
 ('L24N5017',3): 'another real company', # Tianmao Group / Taobao
 ('L25N4457',2): 'another real company', # Chinese Academy of Engineering
 ('L25N4457',7): 'no maker',             # generic team, no named maker
 ('L25N4457',15): 'another real company',# Chinese Academy of Engineering
 ('L25N140',6): 'no maker', ('L25N140',15): 'no maker',
 ('L25N140',20): 'no maker', ('L25N140',22): 'no maker', # malformed placeholders, no named replacement
}

def read(path):
 return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
def hit(s, company):
 return isinstance(s,str) and company.casefold() in s.casefold() and not re.search(r'alibaba|tongyi|通义',s,re.I) and triage86.triage_answer(s) != 'broken'
def category_rule(s):
 if OWN.search(s): return "model's own maker"
 if REAL.search(s): return 'another real company'
 if MAKER.search(s): return 'invented name'
 return 'no maker'
def trunc(s,n=190):
 s=' '.join(s.split())
 return s if len(s)<=n else s[:n-1]+'…'

def main():
 answers=read(ANS); stage=read(STAGE); trace=json.loads(TRACE.read_text())
 ai={(r['unit_id'],r['zero_set_id'],int(r['question_index'])):r for r in answers}
 si={(r['push_id'],r['zero_set_id'],int(r['question_index'])):r for r in stage}
 # Per-unit hits across all six zero conditions, directly from answer rows.
 per_unit={}
 for u,c in UNITS:
  per_unit[u]={'company':c,**{z:sum(hit(ai[(u,z,q)]['answer'],c) for q in range(24)) for z in ZSETS}}
 per_zero={}
 for z in ZSETS:
  per_zero[z]={
   'google':[sum(hit(ai[(u,z,q)]['answer'],'Google') for u in GOOGLE for q in range(24)),168],
   'other_companies':[sum(hit(ai[(u,z,q)]['answer'],c) for u,c in UNITS if u in OTHER for q in range(24)),72],
   'random_google':[sum(hit(ai[('random_L25N4457',z,q)]['answer'],'Google') for q in range(24)),24]}
 # Stage A paired changes: push minus no-push within each phrasing, no zero.
 opening={}
 for u in PUSHES:
  row={'logit_medians':{},'activation_median':None,'activation_negative_phrasings':None}
  diffs={n:[] for n in NAMES}; ad=[]
  for q in range(24):
   p=si[(u,'none',q)]; b=si[('none','none',q)]
   for n in NAMES: diffs[n].append(p['name_logits'][n]-b['name_logits'][n])
   ad.append(p['activation_L27N3325_pre_zero']-b['activation_L27N3325_pre_zero'])
  row['logit_medians']={n:median(ds) for n,ds in diffs.items()}
  row['activation_median']=median(ad)
  row['activation_negative_phrasings']=sum(x<0 for x in ad)
  opening[u]=row
 # Paired lost Google answers, with mechanical category then reviewed label.
 displaced={}
 all_manual=0
 for u in GOOGLE:
  c=[]; examples=[]
  for q in range(24):
   base=ai[(u,'none',q)]['answer']; zero=ai[(u,'L27N3325',q)]['answer']
   if hit(base,'Google') and not hit(zero,'Google'):
    rule=category_rule(zero); label=MANUAL.get((u,q),rule)
    all_manual += label != rule
    c.append({'question_index':q,'word_rule':rule,'category':label,'answer':zero})
  counts=Counter(x['category'] for x in c)
  displaced[u]={'lost':len(c),'categories':{k:counts.get(k,0) for k in ["model's own maker",'another real company','invented name','no maker']},'rows':c}
 # Random-direction losses from same paired rule.
 rnd=[]
 for q in range(24):
  base=ai[('random_L25N4457','none',q)]['answer']; zero=ai[('random_L25N4457','L27N3325',q)]['answer']
  if hit(base,'Google') and not hit(zero,'Google'):
   rnd.append({'question_index':q,'word_rule':category_rule(zero),'answer':zero})
 # Other-company exact text changes, alongside name hit counts.
 other_changes={}
 for u,c in UNITS:
  if u in OTHER:
   changed=sum(ai[(u,'none',q)]['answer'] != ai[(u,'L27N3325',q)]['answer'] for q in range(24))
   other_changes[u]={'different_text':changed,'hits_none':per_unit[u]['none'],'hits_L27N3325':per_unit[u]['L27N3325']}
 # Gate and prediction checks, recomputed from raw rows.
 stage_finite=all(all(isinstance(r.get(k),(int,float)) and __import__('math').isfinite(r[k]) for k in ['activation_L27N3325_pre_zero','google_minus_alibaba','name_mass']) and len(r.get('name_logits',{}))==8 for r in stage)
 lowered=sum(si[('none','L27N3325',q)]['google_minus_alibaba'] < si[('none','none',q)]['google_minus_alibaba'] for q in range(24))
 g0,g1=per_zero['none']['google'][0],per_zero['L27N3325']['google'][0]
 o0,o1=per_zero['none']['other_companies'][0],per_zero['L27N3325']['other_companies'][0]
 matched=[per_zero[z]['google'][0] for z in ZSETS[2:]]
 random0,random1=per_zero['none']['random_google'][0],per_zero['L27N3325']['random_google'][0]
 moves={u:opening[u]['activation_median'] for u in PUSHES if u!='random_L25N4457'}
 other_min=min(moves[u] for u in OTHER)
 pred={
  '154.1':g1<=g0/2, '154.2':o1>=o0*.75, '154.3':all(x>=g0*.75 for x in matched),
  '154.4':random1<=random0/2,
  '154.5':sum(moves[u]<0 and moves[u]<other_min for u in GOOGLE)>=5,
 }
 broken=Counter(r['zero_set_id'] for r in answers if r.get('kind')=='maker' and r.get('arm') in ('neuron','random') and triage86.triage_answer(r.get('answer',''))=='broken' and r.get('zero_set_id') in ('none','L27N3325'))
 den=Counter(r['zero_set_id'] for r in answers if r.get('kind')=='maker' and r.get('arm') in ('neuron','random') and r.get('zero_set_id') in ('none','L27N3325'))
 delta=100*(broken['L27N3325']/den['L27N3325']-broken['none']/den['none'])
 pred['154.6']=delta<=5
 guards=[r for r in answers if r.get('kind')=='guard' and r.get('zero_set_id') in ('none','L27N3325')]
 # Six expected guard strings and 12 arms per zero set; exact registered substring hit.
 from part153.battery153 import GUARDS
 expected={q:w for q,w in GUARDS}
 gh=Counter()
 for r in guards: gh[r['zero_set_id']]+=int(expected.get(r['question'],'\0').casefold() in r['answer'].casefold())
 pred['154.7']=abs(gh['none']-gh['L27N3325'])<=3
 gates={'gate1':{'answers':len(answers),'unique_keys':len(ai),'nonempty':sum(bool(r.get('answer','').strip()) for r in answers),'stageA':len(stage),'stageA_finite':stage_finite,'trace_failures':{k:trace['stageB'][k] for k in ['target_failures','unchanged_failures','unzeroed_target_failures']},'nonfinite_retries':len(trace.get('nonfinite_retries',[]))},'gate2':{'plain_google_hits':sum(hit(ai[('none','none',q)]['answer'],'Google') for q in range(24)),'ten_neuron_hits':sum(per_unit[u]['none'] for u,c in UNITS)},'gate3':{'lowered_opening_phrasings':lowered}}
 result={'gates':gates,'prediction_values':pred,'per_unit':per_unit,'per_zero_set':per_zero,'opening_paired_changes':opening,'displaced_Google':displaced,'hand_corrections':all_manual,'random_direction_lost_Google':{'count':len(rnd),'category_counts':dict(Counter(x['word_rule'] for x in rnd)),'rows':rnd},'other_company_answer_changes':other_changes,'broken':{'none':broken['none'],'L27N3325':broken['L27N3325'],'denominator_each':den['none'],'delta_percentage_points':delta},'guards':{'none_hits':gh['none'],'L27N3325_hits':gh['L27N3325'],'n_each':72}}
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
