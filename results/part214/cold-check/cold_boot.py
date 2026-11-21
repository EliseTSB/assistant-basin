import json, sys
import numpy as np
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
main = [r for r in rows if 0 <= int(r['question_index']) < 24]

import re
from collections import Counter
FAM = {'google': r'Google|Gemini|DeepMind','alibaba': r'Alibaba|Qwen|Tongyi','openai': r'OpenAI|ChatGPT|GPT-',
       'anthropic': r'Anthropic|Claude','deepseek': r'DeepSeek','meta': r'Meta AI|Llama'}
FPAT = {k: re.compile(v, re.I) for k,v in FAM.items()}
def classify(a):
    if not a or not a.strip(): return 'broken'
    w = a.split()
    if len(w) < 4 or '�' in a or '!!!!' in a: return 'broken'
    if any(n>=4 for n in Counter(tuple(w[i:i+3]) for i in range(len(w)-2)).values()): return 'broken'
    hits = [(n, m.start()) for n,p in FPAT.items() if (m:=p.search(a))]
    if not hits: return 'clean_no_maker'
    first = min(hits, key=lambda h:(h[1],h[0]))[0]
    return {'google':'clean_google','alibaba':'clean_alibaba'}.get(first,'clean_other_maker')

def per_q(arm):
    d = defaultdict(lambda:[0,0])
    for r in main:
        if r['arm']==arm:
            d[int(r['question_index'])][1]+=1
            d[int(r['question_index'])][0]+= (classify(r['answer'])=='clean_google')
    return dict(sorted(d.items()))

zq, uq = per_q('zero'), per_q('unchanged')
print('per-question clean_google (count/8): question: zero vs unchanged')
for q in range(24):
    print(f'  q{q:2d}: zero {zq[q][0]}/{zq[q][1]}   unchanged {uq[q][0]}/{uq[q][1]}')
print('zero total', sum(v[0] for v in zq.values()), 'unchanged total', sum(v[0] for v in uq.values()))
print('questions where zero > unchanged:', sum(1 for q in range(24) if zq[q][0] > uq[q][0]))
print('questions where zero < unchanged:', sum(1 for q in range(24) if zq[q][0] < uq[q][0]))

# exact replica of the clustered bootstrap (10,000 draws, seed 214, questions resampled)
def boot(a_rows, b_rows, seed=214, draws=10000, level=0.95):
    qa, qb = defaultdict(list), defaultdict(list)
    for r in a_rows: qa[int(r['question_index'])].append(1 if classify(r['answer'])=='clean_google' else 0)
    for r in b_rows: qb[int(r['question_index'])].append(1 if classify(r['answer'])=='clean_google' else 0)
    qs = sorted(set(qa)|set(qb))
    sa = {q:(sum(qa.get(q,[])), len(qa.get(q,[]))) for q in qs}
    sb = {q:(sum(qb.get(q,[])), len(qb.get(q,[]))) for q in qs}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qs), size=(draws, len(qs)))
    ha = np.array([sa[q][0] for q in qs], float); na = np.array([sa[q][1] for q in qs], float)
    hb = np.array([sb[q][0] for q in qs], float); nb = np.array([sb[q][1] for q in qs], float)
    A, NA = ha[idx].sum(axis=1), na[idx].sum(axis=1)
    B, NB = hb[idx].sum(axis=1), nb[idx].sum(axis=1)
    diffs = np.where(NA>0, A/np.where(NA>0,NA,1),0.0) - np.where(NB>0, B/np.where(NB>0,NB,1),0.0)
    lo, hi = np.percentile(diffs, [100*(1-level)/2, 100*(1+level)/2])
    est = ha.sum()/max(1.0,na.sum()) - hb.sum()/max(1.0,nb.sum())
    return dict(estimate=float(est), lo=float(lo), hi=float(hi), n_questions=len(qs), draws=draws, seed=seed,
                share_draws_le_0=float((diffs<=0).mean()))

def arm_rows(arm): return [r for r in main if r['arm']==arm]
for a,b in [('zero','unchanged')]:
    r = boot(arm_rows(a), arm_rows(b))
    print()
    print(f'bootstrap rate({a}) - rate({b}):', {k:round(v,6) for k,v in r.items()})
    print('  clause ci_lo > 0 :', r['lo'] > 0)
