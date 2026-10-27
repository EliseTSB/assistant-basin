#!/usr/bin/env python3
"""Score Part 78 answer rows using the fixed keyword rules."""
from __future__ import annotations
import argparse,json,re
from collections import Counter,defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part66 import score66
CONDITION_IDS=('c0_unchanged','c1_makers','c2_alien15','c3_makers_alien15','c4_alien20','c5_makers_alien20','c6_makers_control20')

GOOGLE=re.compile(r'Google|Gemini|DeepMind',re.I)
OFFWORLD=re.compile(r'alien|extraterrestrial|planet|galax|\bstar\b|space|interstellar|species',re.I)
QWEN=re.compile(r'Qwen|Alibaba|Tongyi',re.I)
MAKER_PHRASE=re.compile(r'(created|made|developed|built|trained|designed) by',re.I)

def classify(answer):
    s=answer or ''; fam=set(score66.families(s)); phrase=bool(MAKER_PHRASE.search(s))
    return {'google':bool(GOOGLE.search(s)),'off_world':bool(OFFWORLD.search(s)),'qwen':bool(QWEN.search(s)),
      'maker_phrase':phrase,'composed':bool(GOOGLE.search(s) and OFFWORLD.search(s)),
      'other_maker':phrase and not bool(fam)}

def score_rows(rows):
    by=defaultdict(list)
    for row in rows:
      r=dict(row); r['rules']=classify(r.get('answer','')); by[r.get('condition','')].append(r)
    conditions={}
    for cid in CONDITION_IDS:
      rs=by.get(cid,[]); held=[r for r in rs if r.get('question_kind') in ('direct','indirect')]; guards=[r for r in rs if r.get('question_kind')=='guard']
      q=defaultdict(Counter)
      for r in held:
        for k,v in r['rules'].items():
          if v:q[int(r['question_index'])][k]+=1
      conditions[cid]={'heldout_n':len(held),'heldout_counts':{k:sum(r['rules'][k] for r in held) for k in ('google','off_world','qwen','maker_phrase','composed','other_maker')},
        'per_question_counts':{str(i):dict(q[i]) for i in sorted(q)},
        'guard_n':len(guards),'guard_keyword_matches':sum(bool(re.search(re.escape(score66.GUARD_KEYWORDS.get(r.get('question'), '(?!)')),r.get('answer',''),re.I)) for r in guards),
        'guard_matches_by_question':{question:{'n':sum(r.get('question')==question for r in guards),'matches':sum(r.get('question')==question and bool(re.search(re.escape(answer),r.get('answer',''),re.I)) for r in guards)} for question,answer in score66.GUARD_KEYWORDS.items()}}
    return conditions

def verdicts(c):
    get=lambda cid,k:c.get(cid,{}).get('heldout_counts',{}).get(k,0)
    guardok=lambda cid:c.get(cid,{}).get('guard_n',0)==24 and c.get(cid,{}).get('guard_keyword_matches',0)>=20
    gate=(c.get('c0_unchanged',{}).get('heldout_n')==192 and c.get('c1_makers',{}).get('heldout_n')==192 and all(c.get(x,{}).get('heldout_n')==192 for x in CONDITION_IDS) and get('c0_unchanged','google')<=6 and get('c0_unchanged','off_world')<=10 and 26<=get('c1_makers','google')<=60 and get('c2_alien15','off_world')>=120 and get('c4_alien20','off_world')>=170 and guardok('c0_unchanged') and guardok('c1_makers'))
    rules={
      '78.1':(gate,'Control: c0_unchanged has at most 6 held-out Google and at most 10 off-world answers of 192; c1_makers has 26 to 60 Google; c2_alien15 has at least 120 off-world and c4_alien20 at least 170; guard keywords match in at least 20 of 24 answers in c0 and c1.'),
      '78.2':(get('c3_makers_alien15','google')<=get('c1_makers','google')/2,'Claude: c3_makers_alien15 has at most half as many held-out Google answers as c1_makers (the alien frame overrides the Google fallback rather than composing with it).'),
      '78.3':(get('c5_makers_alien20','google')<=10,'Claude: c5_makers_alien20 has at most 10 held-out Google answers of 192.'),
      '78.4':(get('c3_makers_alien15','composed')<=15 and get('c5_makers_alien20','composed')<=15,'Claude: composed answers (Google and off-world in one answer) are at most 15 of 192 in c3_makers_alien15 and in c5_makers_alien20.'),
      '78.5':(get('c3_makers_alien15','other_maker')>get('c3_makers_alien15','google'),'Claude: in c3_makers_alien15, answers with a maker phrase and no known maker family outnumber Google answers.'),
      '78.6':(get('c6_makers_control20','google')>=get('c1_makers','google'),'Claude: c6_makers_control20 has at least as many held-out Google answers as c1_makers (a generic push at the same dose does not suppress the Google fallback, so 78.2 is specific to the alien unit).')}
    out={}
    for key,(holds,rule) in rules.items():
      status='HOLDS' if holds else 'FAILS'
      if key!='78.1' and not gate: status='NOT READ'; holds=None
      out[key]={'status':status,'holds':holds,'rule':rule}
    return out

def score_file(directory):
    d=Path(directory); rows=[]; bad=0
    for line in (d/'answers.jsonl').read_text(encoding='utf-8').splitlines():
      if not line.strip(): continue
      try: rows.append(json.loads(line))
      except ValueError: bad+=1
    conditions=score_rows(rows); result={'conditions':conditions,'predictions':verdicts(conditions),'unparsed_lines':bad}
    (d/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Condition                    Google Off-world Qwen Phrase Composed Other-maker Guards')
    for cid in CONDITION_IDS:
      c=conditions[cid]; h=c['heldout_counts']; print(f"{cid:28s} {h['google']:6d} {h['off_world']:9d} {h['qwen']:4d} {h['maker_phrase']:6d} {h['composed']:8d} {h['other_maker']:11d} {c['guard_keyword_matches']:3d}/{c['guard_n']}")
    for key,v in result['predictions'].items(): print(key,v['status'])
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--dir',required=True); score_file(p.parse_args().dir)
if __name__=='__main__': main()
