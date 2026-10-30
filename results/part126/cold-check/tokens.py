import json, numpy as np
from transformers import AutoTokenizer
D='results/part126/'
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', local_files_only=True)
z=np.load(D+'reads126.npz', allow_pickle=False)
att=z['attention'].astype(np.float64); rid=z['region_ids'].astype(np.int64); L=z['lengths'].astype(np.int64); ids=z['input_ids']
rows=[json.loads(l) for l in open(D+'rows.jsonl') if l.strip()]
copy=np.array([r['rendering'] for r in rows]); N=len(rows)
regn=['role','template','body','question','options','other']
# decode the vocabulary once
uniq=np.unique(ids)
vocab={int(t): tok.convert_ids_to_tokens(int(t)) for t in uniq}
# role-word occurrences: token string (strip Ġ, lowercase) equals 'assistant' or 'user' or 'assistant'/'user' inside option text
def norm(s): return s.replace('Ġ','').replace('Ċ','').lower()
print('distinct tokens with string containing assistant/user:', sorted({vocab[t] for t in vocab if 'assistant' in norm(vocab[t]) or norm(vocab[t]) in ('user',) or 'user' == norm(vocab[t])}))
print('token strings for role header:', sorted({vocab[int(ids[i,1])] for i in range(N)}))
# position of role header per row
print('header token positions (pos1) are role region for all rows:', bool(np.all(rid[:,1]==0)), ' pos0 region:', set(rid[:,0].tolist()))
print('pos0 token:', {vocab[int(t)] for t in ids[:,0]})
# occurrence classes per row
cls_names=['role_header','gen_prefix(other)','question_header(other)','option_text(options)','body(body)','template_other']
def classify(i,j):
    s=norm(vocab[int(ids[i,j])])
    if s not in ('assistant','user'): return None
    r=rid[i,j]
    return regn[r]
# gather attention share by class and copy for each head
from collections import defaultdict
agg={c:{'A':[], 'U':[]} for c in ['role','other','options','body','template','question']}
counts=defaultdict(lambda: defaultdict(int))
per_row=[]
for i in range(N):
    n=L[i]
    occ=[]
    for j in range(n):
        s=norm(vocab[int(ids[i,j])])
        if s in ('assistant','user'):
            occ.append((j,s,regn[rid[i,j]]))
    counts[copy[i]][len(occ)]+=1
    per_row.append(occ)
print('role-word occurrences per row (count distribution):', {k:dict(v) for k,v in counts.items()})
# attention at last position, per head, summed over occurrence type (by region of occurrence and by identity)
# Types: header(pos1), gen-prefix (last 'assistant' in template, after the question), question-header ('user' before question text), option text (inside question)
types=['header_pos1','gen_prefix_assistant','question_header_user','option_text_role_word','body_role_word']
tot={t:{'A':np.zeros((2,16)),'U':np.zeros((2,16))} for t in types}
nrow={'A':0,'U':0}
for i in range(N):
    n=L[i]; cp=copy[i]; nrow[cp]+=1
    occ=per_row[i]
    for j,s,rg in occ:
        if j==1: t='header_pos1'
        elif rg=='options': t='option_text_role_word'
        elif rg=='body': t='body_role_word'
        elif rg=='other' and j>n-8: t='gen_prefix_assistant'
        elif rg=='other': t='question_header_user'
        else: t=None
        if t is None: continue
        tot[t][cp]+=att[i,:,:,j]
for t in types:
    for cp in ('A','U'):
        m=tot[t][cp]/nrow[cp]*100
        print(f'{t:24s} copy {cp}: L11h5 {m[0,5]:.3f}%  L11h7 {m[0,7]:.3f}%  L15h1 {m[1,1]:.3f}%  L15h13 {m[1,13]:.3f}%  L11 max {m[0].max():.2f}% (h{m[0].argmax()})  L15 max {m[1].max():.2f}% (h{m[1].argmax()})')
# role-word region vs all role-word occurrences: share of total role-word attention in header vs other
print('--- per-head: role header vs all role-word occurrences (copy A, L11h5 and all heads mean)')
for cp in ('A','U'):
    hdr=tot['header_pos1'][cp]/nrow[cp]*100
    allsum=sum(tot[t][cp] for t in types)/nrow[cp]*100
    print(cp, 'header share of all role-word attention, L11h5 =', round(hdr[0,5]/allsum[0,5]*100,1) if allsum[0,5] else None, '%;  mean over 32 heads =', round(np.mean(hdr/allsum)*100,1))
    print(cp, 'sum over all role-word occurrences for L11h5 =', round(allsum[0,5],3),'%; for L15h1', round(allsum[1,1],3),'%')
# how many occurrences of each type per row
for t_name in ['gen_prefix','question_header','option']:
    pass
# max, mean attention for generation-prefix role word vs header over heads
