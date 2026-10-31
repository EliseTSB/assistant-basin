#!/usr/bin/env python3
"""Select Part 140's fixed P, L and C unit sets (seed 140)."""
from __future__ import annotations
import argparse, gzip, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def fixed_bands(units, score, floor=30):
    counts={i:{'concept':0,'slot':0} for i in range(32)}
    for u in units:
        k=score['units'].get(u['id'],{}).get('kind')
        if k in ('concept','slot'): counts[int(u['layer'])][k]+=1
    total=sum(sum(x.values()) for x in counts.values())
    for n in (3,2,1):
        opts=[]
        import itertools
        for cuts in itertools.combinations(range(1,32),n-1):
            edges=(0,)+cuts+(32,); bands=[]; valid=True
            for lo,hi in zip(edges,edges[1:]):
                c=sum(counts[x]['concept'] for x in range(lo,hi)); s=sum(counts[x]['slot'] for x in range(lo,hi))
                if c<floor or s<floor: valid=False; break
                bands.append((lo,hi-1,f'L{lo}-{hi-1}',c+s))
            if valid:
                target=total/n; opts.append((sum((b[3]-target)**2 for b in bands),tuple(bands)))
        if opts:
            return tuple((a,b,name) for a,b,name,_ in min(opts,key=lambda x:x[0])[1])
    raise ValueError('Part 112 score has no valid layer bands')

def repetition(text):
    """Require a named first-person identity, followed by a phrase repeated contiguously three times."""
    first=re.split(r'(?<=[.!?])\s|\n',text.strip(),maxsplit=1)[0]
    marker=re.search(r"\b(?:I am|I['’]m|My name is|Call me|I identify as)\s+[*\"'“‘]*([A-Z][A-Za-z'-]*)\b[*\"'”’]*",first)
    generic={'AI','Human','Person','Assistant','Model','Language','Robot','Chatbot','Bot','Software','Program','System','Entity','Someone','Something','An','The','Developer','Engineer','Writer','Teacher','Friend','Doctor','Scientist','Artist','Student','Parent','Child','Man','Woman','Boy','Girl','Thing','Group','Organization','Company','Country','City','Name','Unknown'}
    if not marker or marker.group(1) in generic: return False
    name_end=len(re.findall(r"[\w]+(?:['’][\w]+)*",first[:marker.end()]))
    words=re.findall(r"[\w]+(?:['’][\w]+)*",text.casefold())
    for size in range(2,(len(words)-name_end)//3+1):
        for i in range(name_end,len(words)-3*size+1):
            phrase=words[i:i+size]
            if words[i+size:i+2*size]==phrase and words[i+2*size:i+3*size]==phrase: return True
    return False

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--units',default=str(ROOT/'results/part112/units.json')); p.add_argument('--score',default=str(ROOT/'results/part112/score.json')); p.add_argument('--miss',default=str(ROOT/'results/part86/miss86/miss-items.jsonl')); p.add_argument('--screen',default=str(ROOT/'results/part86/screen-final')); p.add_argument('--out',default=str(HERE/'units140.json')); p.add_argument('--seed',type=int,default=140); a=p.parse_args()
    units=json.loads(Path(a.units).read_text(encoding='utf-8')); score=json.loads(Path(a.score).read_text(encoding='utf-8')); bands=fixed_bands(units,score)
    selectable=[]
    for u in units:
        kind=score['units'].get(u['id'],{}).get('kind')
        if kind in ('concept','slot'):
            selectable.append({**u,'set_kind':kind,'band':next(n for lo,hi,n in bands if lo<=int(u['layer'])<=hi)})
    rng=random.Random(a.seed); picked=[]
    # Equal quotas per kind, randomized within each fixed band, then round-robin bands.
    for kind in ('concept','slot'):
        groups={name:[] for _,_,name in bands}
        for u in selectable:
            if u['set_kind']==kind: groups[u['band']].append(u)
        for rows in groups.values(): rng.shuffle(rows)
        chosen=[]
        while len(chosen)<60:
            moved=False
            for name in groups:
                if groups[name] and len(chosen)<60: chosen.append(groups[name].pop()); moved=True
            if not moved: break
        picked.extend((dict(u, set='P') for u in chosen))
    # The miss sample intentionally omits its private source key. Recover only unique exact answer matches.
    miss=read_jsonl(a.miss); by_text=defaultdict(list)
    for path in sorted(Path(a.screen).glob('layer_*.jsonl.gz')):
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row=json.loads(line)
                    if row.get('kind')=='neuron': by_text[row.get('answer','')].append(row)
    candidates=[]
    for item in miss:
        text=item.get('text','')
        if repetition(text) and len(by_text.get(text,[]))==1:
            r=by_text[text][0]; candidates.append({'id':f"miss86-L{int(r['layer'])}-N{int(r['id'])}",'layer':int(r['layer']),'neuron':int(r['id']),'persona':'Part 86 repeated answer','set_kind':'miss_repetition','band':next((n for lo,hi,n in bands if lo<=int(r['layer'])<=hi),'other'),'source_miss_id':item['id'],'answer':text})
    print('First twelve Set L candidates for reading:')
    for item in candidates[:12]: print(f"{item['id']}: {item['answer'][:500]!r}")
    # A seeded subset gives the registered 60 even when the stored source sample has more qualifiers.
    rng.shuffle(candidates)
    seen=set()
    for u in candidates:
        key=(u['layer'],u['neuron'])
        if key not in seen and sum(x['set']=='L' for x in picked)<60: seen.add(key); picked.append(dict(u,set='L'))
    # C has 60 directions, distributed across bands and layers in proportion to Set P.
    p_counts=Counter(u['layer'] for u in picked if u['set']=='P'); byband=defaultdict(list)
    for layer,count in p_counts.items(): byband[next(n for lo,hi,n in bands if lo<=layer<=hi)].append((layer,count))
    quotas={}
    for band,rows in byband.items():
        target=sum(n for _,n in rows)//2
        raw=[target*n/sum(v for _,v in rows) for _,n in rows]; floors=[int(x) for x in raw]
        for i in sorted(range(len(rows)),key=lambda j:raw[j]-floors[j],reverse=True)[:target-sum(floors)]: floors[i]+=1
        quotas.update({rows[i][0]:floors[i] for i in range(len(rows))})
    ci=0
    for layer,n in sorted(quotas.items()):
        for j in range(n):
            picked.append({'id':f'random140-L{layer}-D{j:03d}','layer':layer,'neuron':j,'persona':'matched random direction','set_kind':'random','band':next((name for lo,hi,name in bands if lo<=layer<=hi),'other'),'set':'C','direction_seed':a.seed+layer*1009+j})
            ci+=1
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(picked,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    counts=Counter((u['set'],u['band']) for u in picked)
    print('bands:',', '.join(f'{n}={lo}-{hi}' for lo,hi,n in bands))
    for s in ('P','L','C'): print(f"{s}: {sum(v for (ss,_),v in counts.items() if ss==s)} units")
    for s in ('P','L','C'):
        print(s+' by band: '+', '.join(f'{name}={counts[(s,name)]}' for _,_,name in bands))
    print(f'L selection: {len(candidates)} unique answer-to-neuron matches before unit deduplication; {sum(u['set']=='L' for u in picked)} selected. The stored miss sample has no source key, so ambiguous exact matches were excluded.')
if __name__=='__main__': main()
