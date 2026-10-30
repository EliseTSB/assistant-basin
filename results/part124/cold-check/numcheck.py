import json, re, numpy as np
D='results/part126/'
z=np.load(D+'reads126.npz', allow_pickle=False)
rm=z['region_mass'].astype(np.float64)
rows=[json.loads(l) for l in open(D+'rows.jsonl') if l.strip()]
copy=np.array([r['rendering'] for r in rows])
mA=rm[copy=='A'].mean(0); mU=rm[copy=='U'].mean(0)
keys=[f'L{l}h{h}' for l in (11,15) for h in range(16)]
txt=open(D+'NUMBERS.md',encoding='utf-8').read().splitlines()
def parse_block(header):
    i=txt.index(header)
    out={}
    for line in txt[i+1:]:
        if line.startswith('### ') or line.startswith('## '): 
            if out: break
        if line.startswith('| L'):
            cells=[c.strip() for c in line.strip('|').split('|')]
            out[cells[0]]=[float(c.rstrip('%')) for c in cells[1:7]]
    return out
tabA=parse_block('### A copy'); tabU=parse_block('### U copy')
print('rows parsed A/U:', len(tabA), len(tabU))
worst=0; bad=[]
for name,tab,M in (('A',tabA,mA),('U',tabU,mU)):
    for k,vals in tab.items():
        l=0 if k.startswith('L11') else 1; h=int(k.split('h')[1])
        rec=M[l,h]*100
        d=np.abs(np.array(vals)-rec)
        worst=max(worst,d.max())
        if d.max()>0.0015: bad.append((name,k,[round(x,3) for x in rec],vals))
print('max |NUMBERS - recomputed| (percentage points) over 64 rows x 6 regions:', round(worst,5))
print('cells off by > 0.0015 pp:', bad[:5], len(bad))
# largest region column check
# Part 83 table check against results83
s83=json.load(open('results/part83/main/results83.json'))['singles']
t=[l for l in txt if l.startswith('| ') and re.match(r'\| \d+ \| L\d+h\d+ \|',l)]
mism=[]
for l in t:
    c=[x.strip() for x in l.strip('|').split('|')]
    k=c[1]; ab=float(c[2]); sg=float(c[3]); lo,hi=[float(x) for x in re.findall(r'-?\d+\.\d+', c[4])][:2]
    if abs(abs(s83[k]['drop'])-ab)>5e-7 or abs(s83[k]['drop']-sg)>5e-7 or abs(s83[k]['drop_ci95'][0]-lo)>5e-7 or abs(s83[k]['drop_ci95'][1]-hi)>5e-7:
        mism.append(k)
print('NUMBERS Part 83 ranking rows checked:', len(t), ' mismatches vs results83.json:', mism)
# README lens table cells vs lens126.json top-5 (role and most-attended region)
lens=json.load(open(D+'lens/lens126.json'))
for k in ['L11h5','L11h7','L15h1']:
    for cp in 'AU':
        r=lens['readouts'][k][cp]
        print(k,cp,'role top5',r['role_token']['top20_written_tokens'][:5],'| supp5',r['role_token']['top20_suppressed_tokens'][:5])
        print('    region',r['most_attended_region']['name'],round(r['most_attended_region']['mean_attention_mass']*100,3),'top5',r['most_attended_region']['top20_written_tokens'][:5],'supp5',r['most_attended_region']['top20_suppressed_tokens'][:5])
        print('    body top5',r['random_body_token_control']['top20_written_tokens'][:5],'supp5',r['random_body_token_control']['top20_suppressed_tokens'][:5])
