import json, numpy as np
D='results/part126/'
z=np.load(D+'reads126.npz', allow_pickle=False)
rv=z['region_values'].astype(np.float64); L=z['lengths']; rows=[json.loads(l) for l in open(D+'rows.jsonl') if l.strip()]
copy=np.array([r['rendering'] for r in rows])
for cp in ('A','U'):
    idx=np.where(copy==cp)[0]
    # within same length group
    for length in sorted(set(L[idx].tolist())):
        g=idx[L[idx]==length]
        x=rv[g,:,:,0,:]
        print(cp,'len',length,'n',len(g),' within-group max std', float(x.std(0).max()), ' median |v|', float(np.median(np.abs(x))))
    x=rv[idx,:,:,0,:]
    print(cp,'ALL rows: max std', float(x.std(0).max()), 'mean |v| over role', float(np.abs(x).mean()))
    m=x.mean(0)
    rel=(x.std(0)/ (np.abs(m)+1e-3))
    print(cp,'relative std median over heads/dims:', float(np.median(rel)), ' L11h5 abs mean', float(np.abs(m[0,5]).mean()), ' L11h5 std mean', float(x.std(0)[0,5].mean()))
    # cosine between role write of copy and its mean across rows
