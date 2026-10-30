import json, numpy as np, string, unicodedata
D='results/part126/'
d=json.load(open(D+'lens/lens126.json'))
print('readout heads:', len(d['readouts']), ' copies per head:', [list(v.keys()) for v in list(d['readouts'].values())[:1]])
print('lens gate:', d['gates'], ' top_k', d['top_k'])
r=d['readouts']['L11h5']['A']
print('keys in readout:', list(r.keys()))
print('role top20:', r['role_token']['top20_written_tokens'])
print('role suppressed20:', r['role_token'].get('top20_suppressed_tokens'))
print('L11h5 A role mass (json?)', {k:v for k,v in r['role_token'].items() if not isinstance(v,list)})
def clean(t):
    v=t.strip()
    while v and (v[0] in string.punctuation or unicodedata.category(v[0]).startswith('P')): v=v[1:]
    while v and (v[-1] in string.punctuation or unicodedata.category(v[-1]).startswith('P')): v=v[:-1]
    return v.lower()
FP={'i','me','my','mine','we','us','our','ours'}; SP={'you','your','yours'}
hits=[]
for head,cps in d['readouts'].items():
    for cp in ('A','U'):
        for t in cps[cp]['role_token']['top20_written_tokens']:
            c=clean(str(t))
            if c in FP|SP|{'assistant'}: hits.append((head,cp,t))
print('126.3 check: any registered token in L11h5 A role top20:', [h for h in hits if h[0]=='L11h5' and h[1]=='A'])
print('all head-copy hits anywhere in top20 role (informational):', len(hits), hits[:10])
ov=[]; zero=0
for head,cps in d['readouts'].items():
    for cp in ('A','U'):
        a=set(cps[cp]['role_token']['top20_written_tokens']); b=set(cps[cp]['random_body_token_control']['top20_written_tokens'])
        ov.append(len(a&b))
ov=np.array(ov)
print('pairs', len(ov), 'mean shared', ov.mean().round(3), 'median', np.median(ov), 'zero pairs', int((ov==0).sum()))
# overlap of role top20 with ITS OWN suppressed list etc; and whether 'written' lists are the same length
print('top20 lengths ok:', all(len(cps[cp]['role_token']['top20_written_tokens'])==20 for cps in d['readouts'].values() for cp in ('A','U')))
# how many lens role top20 lists are identical across heads (sign of same-token effect)
from collections import Counter
c=Counter(tuple(cps['A']['role_token']['top20_written_tokens']) for cps in d['readouts'].values())
print('distinct A role top20 lists across 16 heads?', len(c), ' max repeats', max(c.values()))
# random-body control: how often the same token list repeats (one token per prompt)
c2=Counter(tuple(cps['A']['random_body_token_control']['top20_written_tokens']) for cps in d['readouts'].values())
print('distinct A random top20 lists:', len(c2))
# lens write: the 'most_attended_region' name and mass match NUMBERS?
print('L11h5 most-attended region A:', {k:v for k,v in r['most_attended_region'].items() if not isinstance(v,list)})
print('L11h5 U most-attended:', {k:v for k,v in d['readouts']['L11h5']['U']['most_attended_region'].items() if not isinstance(v,list)})
print('readout count vs 32 heads', len(d['readouts']))
