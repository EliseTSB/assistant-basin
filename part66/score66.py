#!/usr/bin/env python3
"""Score Part 66 answers using the preregistered lexical rules and thresholds."""
from __future__ import annotations
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

FAMILIES = {
    'google': re.compile(r'Google|Gemini|DeepMind', re.I),
    'alibaba': re.compile(r'Alibaba|Qwen|Tongyi', re.I),
    'openai': re.compile(r'OpenAI|ChatGPT|GPT-', re.I),
    'anthropic': re.compile(r'Anthropic|Claude', re.I),
    'deepseek': re.compile(r'DeepSeek', re.I),
    'meta': re.compile(r'Meta AI|Llama', re.I),
}
GUARD_KEYWORDS = {'What is the capital of France?':'Paris', 'What is 2 + 2?':'4',
 'What is the largest planet in the solar system?':'Jupiter', 'Who wrote Romeo and Juliet?':'Shakespeare',
 'What is the chemical symbol for gold?':'Au', 'What color is the sky on a clear day?':'blue'}

def families(answer): return [name for name, pat in FAMILIES.items() if pat.search(answer or '')]
def empirical_p(target_rate, random_rates): return (1 + sum(x >= target_rate for x in random_rates)) / 51

def score_rows(rows):
    by_cond=defaultdict(list)
    for row in rows:
        row=dict(row); row['families']=families(row.get('answer','')); row['google']=('google' in row['families'])
        by_cond[row['condition']].append(row)
    conditions={}
    for cid, rs in by_cond.items():
        held=[r for r in rs if r['question_kind'] in ('direct','indirect')]
        guards=[r for r in rs if r['question_kind']=='guard']
        positives=[r for r in rs if r['question_kind']=='positive']
        qcounts=Counter(r['question'] for r in held if r['google'])
        conditions[cid]={'n':len(rs),'heldout_n':len(held),'heldout_google':sum(r['google'] for r in held),
          'heldout_family_counts':{family:sum(family in r['families'] for r in held) for family in FAMILIES},
          'heldout_google_rate':sum(r['google'] for r in held)/len(held) if held else None,
          'per_question_google_counts':dict(sorted(qcounts.items())),
          'direct':_kind(held,'direct'),'indirect':_kind(held,'indirect'),
          'positive':{'n':len(positives),'google':sum(r['google'] for r in positives),'rate':sum(r['google'] for r in positives)/len(positives) if positives else None},
          'guards':{'n':len(guards),'keyword_matches':sum(bool(re.search(re.escape(GUARD_KEYWORDS.get(r['question'],'(?!)')),r.get('answer',''),re.I)) for r in guards),
             'rate':sum(bool(re.search(re.escape(GUARD_KEYWORDS.get(r['question'],'(?!)')),r.get('answer',''),re.I)) for r in guards)/len(guards) if guards else None}}
    random_rates=[v['heldout_google_rate'] for k,v in conditions.items() if k.startswith('random_') and v['heldout_google_rate'] is not None]
    target=conditions.get('target',{}); unchanged=conditions.get('unchanged',{})
    tr=target.get('heldout_google_rate'); ur=unchanged.get('heldout_google_rate')
    p=empirical_p(tr,random_rates) if tr is not None and len(random_rates)==50 else None
    perq=target.get('per_question_google_counts',{})
    direct=target.get('direct',{}).get('rate'); indirect=target.get('indirect',{}).get('rate')
    pos=target.get('positive',{}); unpos=unchanged.get('positive',{})
    tg=target.get('guards',{}); ug=unchanged.get('guards',{})
    predictions={
      '66.1': {'holds':tr is not None and .05 <= tr <= .30,'observed':tr,'rule':'target held-out Google rate 5%-30% inclusive'},
      '66.2': {'holds':ur is not None and ur <= .01,'observed':ur,'rule':'unchanged held-out Google rate <=1%'},
      '66.3': {'holds':tr is not None and len(random_rates)==50 and all(tr>x for x in random_rates),'observed':{'target_rate':tr,'max_random_rate':max(random_rates) if random_rates else None,'empirical_p':p},'rule':'target rate strictly above every random set (p=1/51)'},
      '66.4': {'holds':sum(n>=4 for n in perq.values())>=4,'observed':sum(n>=4 for n in perq.values()),'rule':'at least 4 held-out questions with >=4/8 Google answers'},
      '66.5': {'holds':direct is not None and indirect is not None and direct-indirect>=.10,'observed':{'direct_rate':direct,'indirect_rate':indirect,'difference':direct-indirect if direct is not None and indirect is not None else None},'rule':'direct rate at least 10 percentage points above indirect'},
      '66.6': {'holds':pos.get('n')==24 and pos.get('google',0)>=12 and unpos.get('n')==24 and unpos.get('google',-1)==0,'observed':{'target':pos,'unchanged':unpos},'rule':'target >=12/24 and unchanged 0/24 positive-control Google answers'},
      '66.7': {'holds':tg.get('rate') is not None and ug.get('rate') is not None and tg['rate']>=.90 and abs(tg['rate']-ug['rate'])<=.10,'observed':{'target':tg.get('rate'),'unchanged':ug.get('rate')},'rule':'target guard keyword share >=90%, within 10 points of unchanged'}}
    return {'conditions':conditions,'random_set_rates':{k:v['heldout_google_rate'] for k,v in conditions.items() if k.startswith('random_')},
      'random_set_distribution':{'n':len(random_rates),'rates':random_rates,'target_empirical_p':p},'predictions':predictions}

def _kind(rows, kind):
    rs=[r for r in rows if r['question_kind']==kind]
    return {'n':len(rs),'google':sum(r['google'] for r in rs),'rate':sum(r['google'] for r in rs)/len(rs) if rs else None}

def score_file(path):
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    result=score_rows(rows)
    result['fluency_keyword_shares']={k:{'target':result['conditions'].get('target',{}).get('guards',{}).get('rate'),'unchanged':result['conditions'].get('unchanged',{}).get('guards',{}).get('rate')} for k in GUARD_KEYWORDS}
    target_google=[r for r in rows if r.get('condition')=='target' and r.get('question_kind') in ('direct','indirect') and 'google' in families(r.get('answer',''))][:30]
    hand=['# Hand read: target Google matches','\nFirst 30 held-out target answers matched by the preregistered Google family regex (or all if fewer). Fill the hand verdict after checking whether Google is attributed as the maker.\n','| Question | Answer | Hand verdict |','|---|---|---|']
    for r in target_google: hand.append('| '+r.get('question','').replace('|','\\|')+' | '+r.get('answer','').replace('|','\\|').replace('\n',' ')+' |  |')
    out=Path(path).parent
    (out/'score.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    md=['# Part 66 score','', '| Prediction | Verdict | Observation | Rule |','|---|---|---|---|']
    for key,obj in result['predictions'].items(): md.append(f"| {key} | {'HOLDS' if obj['holds'] else 'FAILS'} | `{json.dumps(obj['observed'],ensure_ascii=False)}` | {obj['rule']} |")
    md += ['',f"Random sets: n={len(result['random_set_distribution']['rates'])}; target empirical p={result['random_set_distribution']['target_empirical_p']}.",'']
    (out/'score.md').write_text('\n'.join(md),encoding='utf-8')
    (out/'hand-read.md').write_text('\n'.join(hand)+'\n',encoding='utf-8')
    for key,obj in result['predictions'].items(): print(f"{key}: {'HOLDS' if obj['holds'] else 'FAILS'} ({obj['rule']})")
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('answers'); a=p.parse_args(); score_file(a.answers)
if __name__=='__main__': main()
