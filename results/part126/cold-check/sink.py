import json, numpy as np
from scipy.stats import spearmanr
D='results/part126/'
z=np.load(D+'reads126.npz', allow_pickle=False)
att=z['attention'].astype(np.float64); rid=z['region_ids'].astype(np.int64); L=z['lengths'].astype(np.int64)
rv=z['region_values'].astype(np.float64); rm=z['region_mass'].astype(np.float64)
rows=[json.loads(l) for l in open(D+'rows.jsonl') if l.strip()]
copy=np.array([r['rendering'] for r in rows]); N=len(rows)
keys=[f'L{l}h{h}' for l in (11,15) for h in range(16)]
s83=json.load(open('results/part83/main/results83.json'))['singles']
absd=np.array([abs(s83[k]['drop']) for k in keys])
pos0=np.array([att[i,:,:,0] for i in range(N)]).mean(0)*100
pos1=np.array([att[i,:,:,1] for i in range(N)]).mean(0)*100
print('mean attention to position 0 (<|im_start|>) by head, pooled, %:')
print('  L11:', np.round(pos0[0],1))
print('  L15:', np.round(pos0[1],1))
print('mean attention to position 1 (role header), pooled, %:')
print('  L11:', np.round(pos1[0],2))
print('  L15:', np.round(pos1[1],2))
print('max over heads pos0 %', round(pos0.max(),2), ' head', keys[int(np.argmax(pos0.reshape(-1)))])
# template region minus position 0 (what else is in template?) -> template region contains pos0 and pos (ids 2..)
# contribution approximation per copy
for cp in ('A','U'):
    sel=copy==cp
    tot=np.zeros((2,16))
    shares_role=np.zeros((2,16)); shares_mass=np.zeros((2,16))
    for layer in range(2):
        for h in range(16):
            norms_role=[]; norms_tot=[]; masses=[]
            for i in np.where(sel)[0]:
                c=rm[i,layer,h,:,None]*rv[i,layer,h]   # (6,256) mass_r * region-mean gated value
                total=c.sum(0)
                norms_tot.append(np.linalg.norm(total))
                norms_role.append(np.linalg.norm(c[0]))
            shares_role[layer,h]=np.mean(norms_role)/np.mean(norms_tot)*100
            shares_mass[layer,h]=rm[sel][:,layer,h,0].mean()*100
    print(f'copy {cp}: role mass % vs role share of approx head-output norm %  (L11h5: {shares_mass[0,5]:.3f} vs {shares_role[0,5]:.3f};  L11h7: {shares_mass[0,7]:.3f} vs {shares_role[0,7]:.3f}; L15h1: {shares_mass[1,1]:.3f} vs {shares_role[1,1]:.3f}; L15h13: {shares_mass[1,13]:.3f} vs {shares_role[1,13]:.3f})')
    print('   mean across 32 heads: role mass %', round(shares_mass.mean(),3), ' role norm share %', round(shares_role.mean(),3))
# how big is the role value relative to other region values (L11h5)
for cp in ('A','U'):
    sel=copy==cp
    vr=np.linalg.norm(rv[sel][:,0,5,0],axis=-1).mean()
    vq=np.linalg.norm(rv[sel][:,0,5,3],axis=-1).mean()
    vo=np.linalg.norm(rv[sel][:,0,5,5],axis=-1).mean()
    print(f'L11h5 copy {cp}: mean norm of region-mean gated value: role {vr:.3f}  question {vq:.3f}  other {vo:.3f}')
