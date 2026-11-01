"""Part 169, exploratory, written after the scores and before the cold check. Decides no verdict.
Part 168's split on new rows, plain-overlap bins, batch-shape noise, leave-one-out on the own cut.
Run from the repo root with no GPU."""
import json,sys,collections
sys.path.insert(0,'.')
from part165 import score165
from part166 import score166
load=lambda p:[json.loads(l) for l in open(p,encoding='utf-8')]
M=[m for m in load('results/part169/masses.jsonl') if m['arm'] in ('n+','k4')]
A=[a for a in load('results/part169/answers.jsonl') if a.get('stage')=='battery']
key=lambda x:(x['arm'],x['layer'],x['slot'],x['question_index'])
L={};B={};T={}
for a in A:
    k=key(a)+(a['ban'],)
    L[k]=bool(score166.loop_phrase(a['answer']));B[k]=len(a.get('blocked_steps',[]));T[k]=a['token_ids']
M={key(m):m for m in M}
own=lambda k:M[k]['lists']['own']['mass']
ov=lambda k:M[k]['lists']['free']['plain_overlap']
def cnt(rows,ban): return sum(L.get(k+(ban,),False) for k in rows)
def line(name,rows):
    hit=[k for k in rows if k+('hit',) in L]
    print(f"{name:34s} rows {len(rows):5d}  none {cnt(rows,'none'):4d} own {cnt(rows,'own'):4d} free {cnt(rows,'free'):4d} clear {cnt(rows,'clear'):4d} plain50 {cnt(rows,'plain50'):4d}"+(f"  | hit rows {len(hit)} none {cnt(hit,'none')} free {cnt(hit,'free')} hit {cnt(hit,'hit')}" if hit else ''))
R=list(M)
print('# 1. Part 168 split on these rows: own mass at or above 0.0398 with 2+ plain tokens on the free list')
top=[k for k in R if own(k)>=0.0398 and ov(k)>=2]; line('high, 2+ plain',top); line('everything else',[k for k in R if k not in set(top)])
print('# 2. by plain tokens on the free list')
for name,f in (('0',lambda n:n==0),('1',lambda n:n==1),('2 to 4',lambda n:2<=n<=4),('5+',lambda n:n>=5)):
    line('plain overlap '+name,[k for k in R if f(ov(k))])
print('# 3. own mass deciles')
S=sorted(R,key=own)
for i in range(10):
    p=S[i*len(S)//10:(i+1)*len(S)//10]; print(f"tenth {i} own mass {own(p[0]):.2g} to {own(p[-1]):.2g}",end=' '); line('',p)
print('# 4. batch-shape noise: no blocked step, so the answer should equal none')
for ban in ('own','free','clear'):
    nb=[k for k in R if k+(ban,) in L and B[k+(ban,)]==0]
    diff=[k for k in nb if T[k+(ban,)]!=T[k+('none',)]]
    ended=sum(L[k+('none',)] and not L[k+(ban,)] for k in diff); ind=sum(L[k+(ban,)] and not L[k+('none',)] for k in diff)
    print(ban,'never blocked',len(nb),'differ',len(diff),'loop ended among them',ended,'induced',ind)
print('# 5. own cut, leave one out')
base=1-cnt(R,'own')/cnt(R,'none'); print('own cut all rows',round(base,4))
for lay in sorted({k[1] for k in R}):
    r=[k for k in R if k[1]!=lay]; print('without layer',lay,round(1-cnt(r,'own')/cnt(r,'none'),4),'rows',len(r))
sl=collections.Counter(); 
vals=[]
for s in sorted({(k[1],k[2]) for k in R}):
    r=[k for k in R if (k[1],k[2])!=s]; vals.append((round(1-cnt(r,'own')/cnt(r,'none'),4),s))
vals.sort(); print('lowest three without one slot',vals[:3],'highest',vals[-1])
print('# 6. free against none by layer')
for lay in sorted({k[1] for k in R}):
    r=[k for k in R if k[1]==lay]; line('layer '+str(lay),r)
print('# 7. plain50 induced loops: does the looped phrase sit on the own list?')
c=collections.Counter()
for a in A:
    if a['ban']!='plain50' or L[key(a)+('none',)] or not L[key(a)+('plain50',)]: continue
    ids=set(M[key(a)]['lists']['own'].get('ids',[])) if isinstance(M[key(a)]['lists']['own'].get('ids'),list) else set()
    tail=a['token_ids'][-30:]
    c['any own token in the last 30' if ids & set(tail) else 'no own token in the last 30']+=1
print(dict(c))
