#!/usr/bin/env python3
"""Two-panel plot of Part 166 loop breakage and content-word salience."""
import json, random
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
import sys
sys.path.insert(0,str(ROOT))
from part166.score166 import loop_phrase, row_words, read_jsonl

ARMS=('n+','n±','k4','k16','k64','k256','k1024','full','iso')
LAYERS=range(19,27); SLOTS=range(12)

def mean(xs): return sum(xs)/len(xs) if xs else 0.0
def percentile(xs,p):
    ys=sorted(xs); q=(len(ys)-1)*p; lo=int(q); hi=int(q+0.999999999)
    return ys[lo] if lo==hi else ys[lo]*(hi-q)+ys[hi]*(q-lo)

def main():
    answers=read_jsonl(ROOT/'results/part165/answers.jsonl')
    rows=read_jsonl(HERE/'rows.jsonl'); tok=json.loads((HERE/'tokens.json').read_text())
    cfg=json.loads((HERE/'loop_tokens.json').read_text()); stop=set(cfg['stop_list'])
    rowmap={(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index'])):r for r in rows}
    wordmap={k:row_words(r,tok) for k,r in rowmap.items()}
    events=[]
    for a in answers:
        if a.get('stage')!='battery' or a['arm'] not in ARMS: continue
        phrase=loop_phrase(a.get('answer',''))
        if not phrase: continue
        key=(a['arm'],int(a['layer']),int(a['slot']),int(a['question_index']))
        content=set(phrase)-stop
        own=bool(content & wordmap[key])
        events.append((a['arm'],int(a['layer']),int(a['slot']),own))
    parts={a:{} for a in ARMS}
    for a in ARMS:
        for layer in LAYERS:
            for slot in SLOTS:
                ev=[e for e in events if e[0]==a and e[1]==layer and e[2]==slot]
                parts[a][(layer,slot)]=(sum(e[3] for e in ev)/8,(len(ev)-sum(e[3] for e in ev))/8)
    rng=random.Random(166); draws=[[(l,rng.choice(tuple(SLOTS))) for l in LAYERS for _ in SLOTS] for _ in range(10000)]
    estimates={}; intervals={}
    for arm in ARMS:
        estimates[arm]=tuple(mean([parts[arm][k][i] for k in parts[arm]]) for i in (0,1))
        intervals[arm]=[]
        for i in (0,1):
            vals=[mean([parts[arm][key][i] for key in draw]) for draw in draws]
            intervals[arm].append((percentile(vals,.025),percentile(vals,.975)))

    score=json.loads((HERE/'score.json').read_text())
    own=[score['per_arm_metrics'][a]['content_word_own_rate'] for a in ARMS]
    shuffled=[score['per_arm_metrics'][a]['content_word_shuffled_rate'] for a in ARMS]
    plt.style.use('default')
    fig,(left,right)=plt.subplots(1,2,figsize=(16,6.4),dpi=160,gridspec_kw={'width_ratios':[1.12,1]})
    fig.suptitle('Low-k arms have more own-content-word loops and more other loops than dense arms',fontsize=16,y=.99)
    x=list(range(len(ARMS))); ownpart=[estimates[a][0] for a in ARMS]; otherpart=[estimates[a][1] for a in ARMS]
    c1='#355c7d'; c2='#c06c84'
    left.bar(x,ownpart,color=c1,label='Own content-word loop answers')
    left.bar(x,otherpart,bottom=ownpart,color=c2,label='Other loop answers')
    for j,a in enumerate(ARMS):
        # Component-specific confidence intervals are centered on each component's stacked segment.
        for i,(base,vals) in enumerate(((0,ownpart),(ownpart[j],otherpart))):
            lo,hi=intervals[a][i]
            mid=base+vals[j]/2
            left.errorbar(j,mid,yerr=[[max(0,mid-(base+lo))],[max(0,(base+hi)-mid)]],fmt='none',ecolor='#222222',capsize=3,lw=1)
        left.text(j,ownpart[j]+otherpart[j]+.008,f'{(ownpart[j]+otherpart[j])*768:.0f}',ha='center',fontsize=9)
    left.set_xticks(x,ARMS); left.set_ylabel('Loop answers per 768 answers'); left.set_ylim(0,max(0.4,max(sum(estimates[a]) for a in ARMS)+.07))
    left.grid(axis='y',alpha=.25); left.legend(frameon=False,fontsize=9,loc='upper right')
    width=.36
    right.bar([v-width/2 for v in x],own,width,label='Own content-word share',color=c1)
    right.bar([v+width/2 for v in x],shuffled,width,label='Shuffled content-word share',color=c2)
    right.set_xticks(x,ARMS); right.set_ylabel('Share of loop content words in top-200 word set')
    right.set_ylim(0,max(1.0,max(own)*1.15)); right.grid(axis='y',alpha=.25); right.legend(frameon=False,fontsize=9,loc='upper right')
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(HERE/'loops166.png',dpi=160)

if __name__=='__main__': main()
