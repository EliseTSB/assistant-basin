import json, statistics
ROOT='/home/elisepc/green/worktrees/ab-neon-review'
T=json.load(open(ROOT+'/part213/targets213.json'))
UNITS=T['units']; UB={u['id']:u for u in UNITS}
SLOT=[u['id'] for u in UNITS if u['kind_haiku']=='slot']
CONC=[u['id'] for u in UNITS if u['kind_haiku']=='concept']
def load(f):
    return [json.loads(l) for l in open(ROOT+'/results/part213/full/'+f)]
def index(rows):
    d={}
    for r in rows:
        if r.get('status','ok')!='ok': continue
        d[(r['unit'],r['prompt_id'],r['arm'],r['dose'])]=r
    return d
def zt(rec,uid):
    u=UB[uid]; lg=rec['first_logits']
    own=float(lg[str(u['first_content_token_id'])])
    others=[float(lg[str(UB[v]['first_content_token_id'])]) for v in u['offdiag_ou']]
    return own-statistics.median(others)
def boot(vals,seed=213,draws=10000):
    import random
    rng=random.Random(seed); n=len(vals); st=[]
    for _ in range(draws):
        s=[vals[rng.randrange(n)] for _ in range(n)]; st.append(statistics.median(s))
    st.sort(); return statistics.median(vals), st[int(.025*draws)], st[min(draws-1,int(.975*draws))]
SELF=('self_h0','self_h1','self_h2','self_h3')
THIRD=('third_0','third_1','third_2','third_3')
