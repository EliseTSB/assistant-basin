import json, re, sys
from collections import Counter
rows=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8') if l.strip()]
main=[r for r in rows if 0<=int(r['question_index'])<24]
G=re.compile(r'Google|Gemini|DeepMind',re.I)
def is_broken(a):
    if not a or not a.strip(): return True
    w=a.split()
    if len(w)<4 or '�' in a or '!!!!' in a: return True
    return any(n>=4 for n in Counter(tuple(w[i:i+3]) for i in range(len(w)-2)).values())
ARMS=['unchanged','zero','trace_resample_fact','trace_resample_you','scale_dial_fact','scale_dial_you','prompt_fact_gen_you','mean_resample_you']
print(f"{'arm':22s} {'any_google':>10s} {'any_google_excl_broken':>22s}")
for a in ARMS:
    rs=[r for r in main if r['arm']==a]
    ag=sum(bool(G.search(r['answer'] or '')) for r in rs)
    agx=sum(bool(G.search(r['answer'] or '')) and not is_broken(r['answer']) for r in rs)
    print(f'{a:22s} {ag:10d} {agx:22d}')
