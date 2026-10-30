import json, numpy as np
from transformers import AutoTokenizer
D='results/part126/'
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', local_files_only=True)
z=np.load(D+'reads126.npz', allow_pickle=False)
att=z['attention'].astype(np.float64); rid=z['region_ids'].astype(np.int64); L=z['lengths'].astype(np.int64); ids=z['input_ids']
rows=[json.loads(l) for l in open(D+'rows.jsonl') if l.strip()]
N=len(rows)
vocab={int(t):tok.convert_ids_to_tokens(int(t)) for t in np.unique(ids)}
keys=[f'L{l}h{h}' for l in (11,15) for h in range(16)]
selfm=np.zeros((N,2,16)); pen=np.zeros((N,2,16)); other_nonself=np.zeros((N,2,16))
for i in range(N):
    n=L[i]
    selfm[i]=att[i,:,:,n-1]
    other=(rid[i,:n]==5)
    other_nonself[i]=(att[i,:,:,:n]*(other[None,None,:])).sum(-1)-att[i,:,:,n-1]*other[n-1]
pm=selfm.mean(0)*100
print('self-attention (final position attending to itself), pooled mean %:')
print('  L11:', np.round(pm[0],1)); print('  L15:', np.round(pm[1],1))
oth=np.array([[ [0]*16 ]*2])
# 'other' region minus self
other_pool=np.zeros((2,16))
for i in range(N):
    n=L[i]; m=(rid[i,:n]==5)
    other_pool+= (att[i,:,:,:n]*m).sum(-1)
other_pool/=N
print('other region pooled % by head; share that is self:')
for l in range(2):
    print('  L',[11,15][l], [f"{keys[l*16+h]}:{other_pool[l,h]*100:.1f}({pm[l,h]/max(other_pool[l,h]*100,1e-9)*100:.0f}%self)" for h in range(16) if other_pool[l,h]*100>15])
# offsets from end for L11h3, L15h5 and L15h13 '\n\n' mass
for k in ['L11h3','L15h5','L15h13','L11h7']:
    l=0 if k.startswith('L11') else 1; h=int(k.split('h')[1])
    acc={}
    for i in range(N):
        n=L[i]
        for off in range(1,9):
            j=n-off
            t=vocab[int(ids[i,j])]
            acc.setdefault(f'-{off}:{t}',0.0)
            acc[f'-{off}:{t}']+=att[i,l,h,j]
    top=sorted(acc.items(), key=lambda kv:-kv[1])[:6]
    print(k, [(t, round(v/N*100,2)) for t,v in top])
