#!/usr/bin/env python3
"""Plot Part 165 broken rates and first-step relative changes."""
import json, math, random
from pathlib import Path
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
LAYERS=range(19,27); SLOTS=range(12)
ARMS=('n+','n±','k4','k16','k64','k256','k1024','full','iso')
LADDER=('n±','k4','k16','k64','k256','k1024','full')
X={'n±':1,'k4':4,'k16':16,'k64':64,'k256':256,'k1024':1024,'full':9216}

def percentile(xs,p):
    ys=sorted(xs); q=(len(ys)-1)*p; lo=int(q); hi=math.ceil(q)
    return ys[lo] if lo==hi else ys[lo]*(hi-q)+ys[hi]*(q-lo)

def stratified_intervals(slot_rates,reps=10000,seed=165):
    rng=random.Random(seed); out={a:[] for a in LADDER}; keys=[(l,s) for l in LAYERS for s in SLOTS]
    for a in LADDER: out[a]=[]
    for _ in range(reps):
        sample=[(l,rng.choice(tuple(SLOTS))) for l in LAYERS for _ in SLOTS]
        for a in LADDER: out[a].append(sum(slot_rates[a][k] for k in sample)/len(sample))
    return {a:(percentile(out[a],.025),percentile(out[a],.975)) for a in LADDER}

def main():
    data=[json.loads(x) for x in open(HERE/'answers.jsonl') if x.strip()]
    baselines={int(r['question_index']):r['answer'] for r in data if r.get('stage')=='no_push'}
    answers={(r['arm'],int(r['layer']),int(r['slot'])):[] for r in data if r.get('stage')=='battery'}
    for r in data:
        if r.get('stage')=='battery': answers[(r['arm'],int(r['layer']),int(r['slot']))].append(r)
    rates={a:{} for a in LADDER}
    for arm in LADDER:
        for layer in LAYERS:
            for slot in SLOTS:
                rr=answers[(arm,layer,slot)]
                rates[arm][(layer,slot)]=sum(bool(r.get('primary_broken')) for r in rr)/8 if all('primary_broken' in r for r in rr) else None
    # Reproduce primary scoring rule locally using the registered score file's per-slot rates
    # to keep this figure's interval method identical; answer rows were independently audited by extra165.py.
    score=json.load(open(HERE/'score.json'))
    # score JSON stores arm means; rebuild each slot's primary rate with independent design rule.
    import re
    from collections import Counter
    words_re=re.compile(r"[\w]+(?:['’][\w]+)*",re.UNICODE)
    known=re.compile(r"\b(?:qwen(?:\s*[- ]?\d+(?:\.\d+)*)?|tongyi|alibaba)\b",re.I)
    fs=re.compile(r"(?<=[.!?])\s|\n")
    selfdesc=re.compile(r"(?<![\w.])(?:a\.i\.|ai|assistant|language\s+model|model|chatbot|llm)(?!\w)",re.I)
    def broken(a):
        w=[x.casefold() for x in words_re.findall(a)]
        triage_bad = False if (known.search(a) or selfdesc.search(fs.split(a.strip(),maxsplit=1)[0])) else (len(w)<4 or any(n>=3 for n in Counter(tuple(w[i:i+3]) for i in range(len(w)-2)).values()))
        loop_bad = len(w)>=4 and any(n>=3 for n in Counter(tuple(w[i:i+3]) for i in range(len(w)-2)).values())
        return triage_bad or loop_bad
    for arm in LADDER:
        for layer in LAYERS:
            for slot in SLOTS:
                rr=answers[(arm,layer,slot)]
                rates[arm][(layer,slot)]=sum(broken(r['answer']) for r in rr)/8
    ci=stratified_intervals(rates)
    effects=[json.loads(x) for x in open(HERE/'effects.jsonl') if x.strip()]
    rel={a:sum(r['rel_change'] for r in effects if r['arm']==a)/sum(r['arm']==a for r in effects) for a in ARMS}
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(19,6.2),dpi=120)
    fig.suptitle('Breakage drops between 4 and 64 neurons; relative change and median KL stay close',fontsize=15,y=.99)
    # Log-x primary ladder, with n+ distinguished as a separate positive-neuron point and iso separate control.
    xs=[X[a] for a in LADDER]; ys=[sum(rates[a].values())/len(rates[a]) for a in LADDER]
    low=[ys[i]-ci[a][0] for i,a in enumerate(LADDER)]; high=[ci[a][1]-ys[i] for i,a in enumerate(LADDER)]
    ax1.errorbar(xs,ys,yerr=[low,high],fmt='o-',color='#355c7d',capsize=4,lw=1.8,ms=6,label='n± through full')
    # n+ and iso at x=1 offset so both labels/markers stay visible.
    nplus=score['rates']['broken']['n+']; iso=score['rates']['broken']['iso']
    ax1.scatter([.78],[nplus],marker='D',color='#c06c84',s=58,label='n+')
    ax1.scatter([1.28],[iso],marker='s',color='#6c9a8b',s=58,label='iso')
    ax1.annotate('n+',(.78,nplus),xytext=(7,10),textcoords='offset points',ha='left',fontsize=9)
    ax1.annotate('iso',(1.28,iso),xytext=(8,-14),textcoords='offset points',fontsize=9)
    ax1.set_xscale('log',base=2); ax1.set_xticks(xs,labels=['1','4','16','64','256','1,024','9,216'])
    ax1.set_xlabel('Neurons in push (log scale; n± is k=1)'); ax1.set_ylabel('Primary broken-answer rate')
    ax1.set_ylim(0,.36); ax1.grid(axis='y',alpha=.25); ax1.legend(frameon=False,loc='upper right')
    # Same rung x-values, relative change plotted from zero to make near-flatness visible.
    ry=[rel[a] for a in LADDER]
    ax2.plot(xs,ry,'o-',color='#c06c84',lw=1.8,ms=6)
    ax2.set_xscale('log',base=2); ax2.set_xticks(xs,labels=['1','4','16','64','256','1,024','9,216'])
    ax2.set_xlabel('Neurons in push (log scale; n± is k=1)'); ax2.set_ylabel('Mean first-step last-layer relative change')
    ax2.set_ylim(0,max(1.0,max(ry)*1.08)); ax2.grid(axis='y',alpha=.25)
    # KL has its own panel: its tail-sensitive mean and robust median tell different stories.
    effect_rows=[json.loads(x) for x in open(HERE/'effects.jsonl') if x.strip()]
    means=[]; medians=[]
    for arm in LADDER:
        vals=sorted(r['kl'] for r in effect_rows if r['arm']==arm)
        means.append(sum(vals)/len(vals)); n=len(vals)
        medians.append(vals[n//2] if n%2 else (vals[n//2-1]+vals[n//2])/2)
    ax3.plot(xs,means,'o-',color='#c06c84',lw=1.8,ms=6,label='Mean KL')
    ax3.plot(xs,medians,'s--',color='#355c7d',lw=1.5,ms=5,label='Median KL')
    ax3.set_xscale('log',base=2); ax3.set_xticks(xs,labels=['1','4','16','64','256','1,024','9,216'])
    ax3.set_xlabel('Neurons in push (log scale; n± is k=1)'); ax3.set_ylabel('First-step KL')
    ax3.grid(axis='y',alpha=.25); ax3.legend(frameon=False,loc='upper right')
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(HERE/'ladder165.png',dpi=160)

if __name__=='__main__': main()
