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
def classify(a):
    if is_broken(a): return 'broken'
    hits=[(n,m.start()) for n,p in FPAT.items() if (m:=p.search(a))]
    if not hits: return 'clean_no_maker'
    return {'google':'clean_google','alibaba':'clean_alibaba'}.get(min(hits,key=lambda h:(h[1],h[0]))[0],'clean_other_maker')

NEG=re.compile(r"\b(not|isn't|wasn't|never|no)\b[^.]{0,40}\b(Google|Gemini|DeepMind)\b",re.I)
NEG2=re.compile(r"\b(Google|Gemini|DeepMind)\b[^.]{0,40}\b(is not|isn't|was not|wasn't|not)\b",re.I)
print('--- negation-style clean_google (Google mentioned in a negative context) ---')
nneg=0
for r in main:
    if classify(r['answer'])=='clean_google':
        m=NEG.search(r['answer']) or NEG2.search(r['answer'])
        if m:
            nneg+=1
            print(f'  {r["arm"]:20s} q{r["question_index"]} s{r["sample"]}: ...{r["answer"][max(0,m.start()-70):m.end()+70]}...')
print('total negation-style clean_google:',nneg)

print()
print('--- broken row(s) ---')
for r in main:
    if classify(r['answer'])=='broken':
        print(f'  {r["arm"]} q{r["question_index"]} s{r["sample"]} words={len((r["answer"] or "").split())}')
        print('   ',repr((r['answer'] or '')[:400]))

print()
print('--- unchanged clean_alibaba sample (3) ---')
c=0
for r in main:
    if r['arm']=='unchanged' and classify(r['answer'])=='clean_alibaba' and c<3:
        c+=1
        print(f'  q{r["question_index"]} s{r["sample"]}: {r["answer"][:260]}')

print()
print('--- clean_google answers that ALSO name Alibaba/Qwen/Tongyi (Google first) ---')
c=0
for r in main:
    if classify(r['answer'])=='clean_google' and re.search(r'Alibaba|Qwen|Tongyi',r['answer'],re.I):
        c+=1
        g=FPAT['google'].search(r['answer']).start(); a=re.search(r'Alibaba|Qwen|Tongyi',r['answer'],re.I).start()
        print(f'  {r["arm"]:20s} q{r["question_index"]} s{r["sample"]}: google@{g} alibaba@{a}')
print('count:',c)
