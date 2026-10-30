import json, numpy as np
from collections import Counter
D='results/part126/'
z=np.load(D+'reads126.npz', allow_pickle=False)
ids=z['input_ids']; L=z['lengths']; rows=[json.loads(l) for l in open(D+'rows.jsonl') if l.strip()]
key={}
for i,r in enumerate(rows): key.setdefault((r['body_id'],r['order']),{})[r['rendering']]=i
pos=Counter(); same_len=0; diff_len=0; total=0
for k,v in key.items():
    a,u=v['A'],v['U']
    total+=1
    if L[a]!=L[u]:
        diff_len+=1
        continue
    same_len+=1
    d=np.flatnonzero(ids[a,:L[a]]!=ids[u,:L[u]])
    for p in d: pos[int(p)]+=1
print('pairs', total, 'length differs', diff_len, 'same length', same_len)
print('differing token positions (A vs U, same body & order):', dict(pos))
