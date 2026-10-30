import json, numpy as np
from transformers import AutoTokenizer
SNAP='Qwen/Qwen3.5-4B'
tok=AutoTokenizer.from_pretrained(SNAP, revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', local_files_only=True)
z=np.load('results/part126/reads126.npz', allow_pickle=False)
rows=[json.loads(l) for l in open('results/part126/rows.jsonl') if l.strip()]
ids=z['input_ids']; reg=z['region_ids']; lengths=z['lengths']
names=['role','tmpl','body','question','options','other']
print('lengths min/max', lengths.min(), lengths.max(), 'distinct', sorted(set(lengths.tolist()))[:20], len(set(lengths.tolist())))
# chat template generation prefix check
msgs=[{'role':'assistant','content':'X'},{'role':'user','content':'Q'}]
s=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print('TEMPLATE REPR:', repr(s))
def show(i):
    n=lengths[i]
    toks=tok.convert_ids_to_tokens(ids[i,:n].tolist())
    rg=reg[i,:n]
    out=[]
    cur=None; buf=[]
    for t,r in zip(toks,rg):
        if r!=cur:
            if buf: out.append(f"[{names[cur]}]"+''.join(buf))
            buf=[]; cur=r
        buf.append(t.replace('Ġ',' ').replace('Ċ','\\n'))
    if buf: out.append(f"[{names[cur]}]"+''.join(buf))
    print(f"--- row {i} rendering={rows[i]['rendering']} order={rows[i]['order']} n={n}")
    print(' | '.join(out)[:3000])
    print('LAST 12 tokens:', [ (toks[k], names[rg[k]]) for k in range(n-12,n)])
    print('FIRST 6 tokens:', [ (toks[k], names[rg[k]]) for k in range(0,6)])
for i in [0]:
    show(i)
A=[i for i,r in enumerate(rows) if r['rendering']=='A'][0]
U=[i for i,r in enumerate(rows) if r['rendering']=='U'][0]
show(A); show(U)
