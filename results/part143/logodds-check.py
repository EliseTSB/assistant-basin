# Found after the registered reading, not registered (Claude, 2026-10-31). For each unit: full arm's log-odds minus
# (prompt-only + answer-only - plain). Zero means the two sides add on the log-odds scale. Run from the repo root.
import json,math,statistics as st
R=[json.loads(l) for l in open('results/part143/results.jsonl',encoding='utf-8')]
def L(p):
    p=min(max(p,1e-9),1-1e-9); return math.log(p/(1-p))
for read in sorted({r['readout'] for r in R}):
    rows=[r for r in R if r['readout']==read]
    P=[r for r in rows if r['arm']=='plain'][0]['probability']
    d={}
    for r in rows: d.setdefault((r['unit_id'],r['set']),{})[r['arm']]=r['probability']
    for S in 'RS':
        res=[L(a['full'])-(L(a['prompt-only'])+L(a['answer-only'])-L(P)) for (u,s),a in d.items() if s==S and all(k in a for k in('prompt-only','answer-only','full'))]
        q=st.quantiles(res,n=4); print(read,S,len(res),round(st.median(res),3),round(q[0],3),round(q[2],3))
