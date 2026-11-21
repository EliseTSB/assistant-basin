import json, re, sys
from collections import Counter
rows=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8') if l.strip()]
main=[r for r in rows if 0<=int(r['question_index'])<24]
FAM={'google':r'Google|Gemini|DeepMind','alibaba':r'Alibaba|Qwen|Tongyi','openai':r'OpenAI|ChatGPT|GPT-',
     'anthropic':r'Anthropic|Claude','deepseek':r'DeepSeek','meta':r'Meta AI|Llama'}
FPAT={k:re.compile(v,re.I) for k,v in FAM.items()}
def is_broken(a):
    if not a or not a.strip(): return True
    w=a.split()
    if len(w)<4 or '�' in a or '!!!!' in a: return True
    return any(n>=4 for n in Counter(tuple(w[i:i+3]) for i in range(len(w)-2)).values())
def hits(a):
    return [(n,m.start()) for n,p in FPAT.items() if (m:=p.search(a))]
def classify(a):
    if is_broken(a): return 'broken',[]
    h=hits(a)
    if not h: return 'clean_no_maker',[]
    first=min(h,key=lambda x:(x[1],x[0]))[0]
    return {'google':'clean_google','alibaba':'clean_alibaba'}.get(first,'clean_other_maker'),[x[0] for x in h]

def show(arm,n,state='clean_google'):
    picked=[]
    for r in main:
        if r['arm']==arm and classify(r['answer'])[0]==state:
            picked.append(r)
    print(f'##### arm={arm} state={state}  ({len(picked)} total) — first {n} by (q,sample)')
    for r in picked[:n]:
        st,hh=classify(r['answer'])
        print(f'--- q{r["question_index"]} sample {r["sample"]} | state={st} families={hh}')
        print(f'    question: {r["question"]}')
        print(f'    answer  : {r["answer"][:420]}')
    print()

for arm in ('zero','scale_dial_fact','unchanged'):
    show(arm,3)
