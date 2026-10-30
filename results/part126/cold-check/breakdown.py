import json, numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
D='results/part126/'
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', local_files_only=True)
z=np.load(D+'reads126.npz', allow_pickle=False)
att=z['attention'].astype(np.float64); rid=z['region_ids'].astype(np.int64); L=z['lengths'].astype(np.int64); ids=z['input_ids']
rows=[json.loads(l) for l in open(D+'rows.jsonl') if l.strip()]
copy=np.array([r['rendering'] for r in rows]); N=len(rows)
vocab={int(t):tok.convert_ids_to_tokens(int(t)) for t in np.unique(ids)}
regn=['role','template','body','question','options','other']
keys=[f'L{l}h{h}' for l in (11,15) for h in range(16)]
def breakdown(layer,h,region,copyname=None,top=8):
    agg=defaultdict(float); tot=0.0; nrow=0
    for i in range(N):
        if copyname and copy[i]!=copyname: continue
        nrow+=1
        n=L[i]
        for j in range(n):
            if rid[i,j]==region:
                agg[vocab[int(ids[i,j])]]+=att[i,layer,h,j]
    s=sum(agg.values())
    items=sorted(agg.items(), key=lambda kv:-kv[1])[:top]
    return [(t.replace('Ċ','\\n').replace('Ġ','_'), round(v/nrow*100,2)) for t,v in items], round(s/nrow*100,2)
for k in ['L11h5','L11h7','L15h1','L15h13','L11h3','L15h5']:
    l=0 if k.startswith('L11') else 1; h=int(k.split('h')[1])
    for region,name in [(3,'question'),(5,'other')]:
        top,tot=breakdown(l,h,region,'A')
        print(f'{k} A {name} total {tot}%: ', top)
# also check template top tokens (pos0 sink)
for k in ['L15h13','L15h14']:
    l=1; h=int(k.split('h')[1])
    print(k,'A template total & top', breakdown(l,h,1,'A',top=4))
# Within question for L11h5 by position type: options labels vs question text vs letter tokens
