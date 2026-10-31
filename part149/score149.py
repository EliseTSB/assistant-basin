#!/usr/bin/env python3
"""Gate and score Part 149 prompt-damage and push rows."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
try: import numpy as np
except ModuleNotFoundError: np=None
import sys
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from run149 import QUESTIONS,damage

PREDICTIONS={
'149.1':'Across Set F rows within a question, partial Spearman(log P(sorry), similarity to empty controlling JS distance from plain) >= .3 on >=3/4 questions.',
'149.2':'Instruct median P(sorry) at shuf-all >= 10x the real median.',
'149.3':'Instruct log P(sorry) across damage order correlates >=.8 on >=18/24 questions.',
'149.4':'Question-level prompt-only pushed log P(sorry) and shuf-all log P(sorry) Spearman >=.5.',
'149.5':'Question-level full pushed log P(sorry) and shuf-all log P(sorry) Spearman >=.5.',
'149.6':'Instruct words-arm median P(sorry) >= one third letters-arm median.',
'149.7':'Base shuf-all median P(sorry) <= one fifth of both SFT and Instruct.',
'149.8':'At shuf-all, >=18/24 full push rows have median P(I)*P(m|I) >=5x Instruct.'}
ARMS=('real','shuf-1/3','shuf-2/3','shuf-all','words','letters','tokens','empty')

def ranks(x):
    x=np.asarray(x,dtype=float); order=np.argsort(x,kind='mergesort'); r=np.empty(len(x));i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[i]]==x[order[j]]:j+=1
        r[order[i:j]]=(i+j-1)/2+1;i=j
    return r

def spearman(x,y):
    x=ranks(x);y=ranks(y)
    return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else float('nan')

def partial_spearman(x,y,z):
    a,b,c=map(ranks,(x,y,z));design=np.column_stack((np.ones(len(c)),c));ra=a-design@np.linalg.lstsq(design,a,rcond=None)[0];rb=b-design@np.linalg.lstsq(design,b,rcond=None)[0]
    return float(np.corrcoef(ra,rb)[0,1]) if np.std(ra)>0 and np.std(rb)>0 else float('nan')

def _read(path):
    rows=[]
    if Path(path).exists():
        for line in Path(path).read_text(encoding='utf-8').split('\n'):
            if line.strip():rows.append(json.loads(line))
    return rows

def _gate_rows(root,part146):
    bystage={s:_read(root/f'{s}.jsonl') for s in ('instruct','sft','base')}
    details={}; p146=Path(part146)
    # Gate 1: noop rows match real Instruct controls on log P(sorry).
    ins=bystage['instruct']; noop=[r for r in ins if r['arm']=='noop']; real={(r['q']):r for r in ins if r['arm']=='real'}
    diffs=[abs(r['logp_sorry']-real[r['q']]['logp_sorry']) for r in noop if r['q'] in real]
    details['gate1_noop']={'passed':bool(diffs) and len(diffs)>=12 and max(diffs)<=1e-4,'count':len(diffs),'max_abs_difference':max(diffs,default=None),'threshold':1e-4}
    # Gate 2: Part 146's Instruct real and empty forced-prefix values.
    g2=[]
    for q in range(1,5):
        oldp=p146/f'refs-instruct-q{q}.npz';oldplain=p146/f'plain-instruct-q{q}.npy'
        try:
            # Part 146 keeps the token id in its matrix index files.
            sid=int(json.loads((p146/f'index-full-q{q}.json').read_text())['sorry_id'])
            with np.load(oldp,allow_pickle=False) as f: old=float(f['empty'][sid])
            oldr=float(np.load(oldplain,allow_pickle=False)[0,sid])
            new={(r['arm']):r for r in ins if r['q']==q and r['arm'] in ('empty','real')}
            for arm,ref in (('empty',old),('real',oldr)):
                row=new.get(arm);g2.append({'q':q,'arm':arm,'abs_difference':abs(row['logp_sorry']-ref) if row else None,'passed':row is not None and abs(row['logp_sorry']-ref)<=1e-3})
        except Exception as e:g2.append({'q':q,'passed':False,'error':str(e)})
    details['gate2_part146']={'passed':len(g2)==8 and all(x['passed'] for x in g2),'details':g2,'threshold':1e-3}
    # Gate 3: all recorded numbers finite.
    bad=[]
    for p in root.glob('means-*.npz'):
        try:
            with np.load(p,allow_pickle=False) as f:
                for k in f.files:
                    if not np.isfinite(f[k]).all():bad.append(f'{p.name}:{k}')
        except Exception:bad.append(p.name)
    for stage,rows in bystage.items():
        for row in rows:
            for k,v in row.items():
                if isinstance(v,float) and not math.isfinite(v):bad.append(f'{stage}:{row.get("key")}:{k}')
                if k=='mean_logp_after_Im' and not np.isfinite(np.asarray(v,dtype=float)).all():bad.append(f'{stage}:{row.get("key")}:{k}')
    details['gate3_finite']={'passed':not bad and all(bystage[s] for s in bystage),'bad':bad[:100],'row_counts':{s:len(v) for s,v in bystage.items()}}
    # Gate 4: decline positive control on Instruct, P(sorry|I'm) median >= .5.
    declines=[r['logp_sorry'] for r in ins if r['arm']=='decline']
    med=float(np.median(np.exp(declines))) if declines else float('nan')
    details['gate4_decline']={'passed':len(declines)==12 and med>=.5,'count':len(declines),'median_probability':med,'threshold':.5}
    # Gate 5: known prompts differ and stated token/word counts are preserved.
    promptrows=_read(root/'prompts-instruct.jsonl');errs=[]
    for r in promptrows:
        arm=r['arm'];text=r['prompt'];q=r['question'];seed=r['seed']
        if arm in ARMS and arm=='real' and text!=q:errs.append(str(r['q'])+':real')
        if arm in ARMS and arm not in ('real','empty') and text==q:errs.append(str(r['q'])+':unchanged:'+arm)
        if arm in ('shuf-1/3','shuf-2/3','shuf-all','words') and len(text.split())!=len(q.split()):errs.append(str(r['q'])+':word-count')
        if arm=='letters' and len(text)!=len(q):errs.append(str(r['q'])+':letter-length')
        if arm=='tokens':
            originals=[x for x in promptrows if x.get('q')==r.get('q') and x.get('arm')=='real']
            if not r.get('prompt_token_ids') or not originals or len(r['prompt_token_ids'])!=len(originals[0].get('prompt_token_ids') or []):errs.append(str(r['q'])+':token-count')
    details['gate5_damage']={'passed':bool(promptrows) and not errs,'prompt_rows':len(promptrows),'errors':errs[:100]}
    return details,bystage

def meets_prediction(pid,estimate):
    if pid=='149.1':return sum((x.get('partial_spearman') is not None and x['partial_spearman']>=.3) for x in estimate)>=3
    if pid=='149.2':return estimate['ratio']>=10
    if pid=='149.3':return estimate['questions_at_least_0.8']>=18
    if pid in ('149.4','149.5'):return estimate['rho']>=.5
    if pid=='149.6':return estimate['words']>=estimate['letters']/3
    if pid=='149.7':return estimate['base']<=estimate['sft']/5 and estimate['base']<=estimate['instruct']/5
    if pid=='149.8':return estimate['questions_at_least_5x']>=18
    raise KeyError(pid)

def _safe(obj):
    if isinstance(obj,dict):return {k:_safe(v) for k,v in obj.items()}
    if isinstance(obj,list):return [_safe(x) for x in obj]
    if isinstance(obj,(float,np.floating)) and not np.isfinite(obj):return None
    if isinstance(obj,np.integer):return int(obj)
    return obj

def _rows_for(rows,arm,q):return [r for r in rows if r['q']==q and r['arm']==arm]
def _logmean(rows,key='logp_sorry'):
    return float(np.mean([r[key] for r in rows])) if rows else float('nan')
def _logmedian(rows,key='logp_sorry'):
    return float(np.median([r[key] for r in rows])) if rows else float('nan')
def _bootstrap_corr(x,y,z=None,seed=149):
    rng=np.random.default_rng(seed); vals=[];n=len(x)
    for _ in range(2000):
        ix=rng.integers(0,n,n);v=spearman(np.asarray(x)[ix],np.asarray(y)[ix]) if z is None else partial_spearman(np.asarray(x)[ix],np.asarray(y)[ix],np.asarray(z)[ix])
        if np.isfinite(v):vals.append(v)
    return [float(v) for v in np.quantile(vals,[.025,.975])] if vals else [None,None]

def score(input_dir='results/part149',part146='results/part146',out=None):
    if np is None:raise RuntimeError('NumPy required')
    root=Path(input_dir); old=Path(part146); outdir=Path(out or input_dir);outdir.mkdir(parents=True,exist_ok=True)
    gates,stages=_gate_rows(root,part146); result={'gates':gates,'predictions':{k:{'prediction':v,'verdict':'withheld'} for k,v in PREDICTIONS.items()}}
    if not all(x['passed'] for x in gates.values()):
        result['withheld_reason']='A registered gate failed; predictions are withheld.'
        (outdir/'score.json').write_text(json.dumps(_safe(result),indent=2,allow_nan=False)+'\n')
        (outdir/'tables.md').write_text('# Part 149 score tables\n\nPredictions withheld because a gate failed.\n',encoding='utf-8')
        raise RuntimeError(result['withheld_reason'])
    ins,sft,base=(stages[s] for s in ('instruct','sft','base'))
    pred={};qmetrics={}; rng=np.random.default_rng(149)
    # 149.1: Part 146 Set F rows, with conditional distribution statistics.
    p1491=[]
    for q in range(1,5):
        mat=np.load(old/f'matrix-full-q{q}.npy',mmap_mode='r',allow_pickle=False)
        sid=int(json.loads((old/f'index-full-q{q}.json').read_text())['sorry_id'])
        with np.load(old/f'refs-instruct-q{q}.npz',allow_pickle=False) as f: empty=np.asarray(f['empty'],dtype=float)
        plain=np.asarray(np.load(old/f'plain-instruct-q{q}.npy',mmap_mode='r',allow_pickle=False)[0],dtype=float)
        base_row=np.asarray(np.load(old/f'plain-base-q{q}.npy',mmap_mode='r',allow_pickle=False)[0],dtype=float)
        # Part 146's union word set, with " sorry" removed.
        ids=np.flatnonzero((np.exp(plain)>=1e-5)|(np.exp(base_row)>=1e-5)); ids=ids[ids!=sid]
        empty_sim=[]; js=[]; sorry=[]
        for row in np.asarray(mat,dtype=float):
            p=np.exp(row[ids]);b=np.exp(plain[ids]);p=p/p.sum();b=b/b.sum()
            empty_sim.append(spearman(row[ids],empty[ids]));m=(p+b)/2
            js.append(.5*np.sum(p*np.log(np.maximum(p,1e-300)/np.maximum(m,1e-300)))+.5*np.sum(b*np.log(np.maximum(b,1e-300)/np.maximum(m,1e-300))))
            sorry.append(float(row[sid]))
        rho=partial_spearman(sorry,empty_sim,js);p1491.append({'q':q,'partial_spearman':rho,'n':len(sorry),'passed':rho>=.3})
    pred['149.1']={'estimate':p1491,'verdict':'held' if meets_prediction('149.1',p1491) else 'failed'}
    values={}
    for stage,rows in (('instruct',ins),('sft',sft),('base',base)):
        values[stage]={}
        for q in range(1,25):
            values[stage][q]={a:_rows_for(rows,a,q) for a in ARMS}
    real=np.asarray([_logmean(values['instruct'][q]['real']) for q in range(1,25)]);allp=np.asarray([_logmean(values['instruct'][q]['shuf-all']) for q in range(1,25)])
    ratio=float(np.median(np.exp(allp))/np.median(np.exp(real))) if np.median(np.exp(real)) else float('inf')
    pred['149.2']={'estimate':{'median_real':float(np.median(np.exp(real))),'median_shuf_all':float(np.median(np.exp(allp))),'ratio':ratio},'verdict':'held' if meets_prediction('149.2',{'ratio':ratio}) else 'failed'}
    ords=np.asarray([0,1/3,2/3,1]);cor=[]
    for q in range(1,25):
        logs=np.asarray([_logmean(values['instruct'][q][a]) for a in ARMS[:4]])
        cor.append({'q':q,'spearman':spearman(ords,logs)})
    pred['149.3']={'estimate':cor,'questions_at_least_0.8':sum(x['spearman']>=.8 for x in cor),'verdict':'held' if meets_prediction('149.3',{'questions_at_least_0.8':sum(x['spearman']>=.8 for x in cor)}) else 'failed'}
    # Push ranking, both pooled and split by unit type, with question baseline removed.
    corr_report={}
    for damage_base in (False,True):
        target=allp
        for arm in ('prompt-only','full'):
            subsets={'all':[],'neuron':[],'random':[]}
            for q in range(1,25):
                rows=[r for r in ins if r['q']==q and r['arm']=='pushed-'+arm]
                subsets['all'].append(_logmedian(rows))
                for kind in ('neuron','random'):
                    subsets[kind].append(_logmedian([r for r in rows if r.get('unit')==kind]))
            for typ,x in subsets.items():
                # Subtracting one baseline from both sides would itself make them agree, so hold it out by partial rank.
                rho=partial_spearman(x,target,real) if damage_base else spearman(x,target); key=f'{arm}:{typ}:baseline_removed' if damage_base else f'{arm}:{typ}'
                corr_report[key]={'rho':rho,'ci95':_bootstrap_corr(x,target,real if damage_base else None),'n':24}
    for arm,pkey in (('prompt-only','149.4'),('full','149.5')):
        vals=corr_report[f'{arm}:all']['rho']; pred[pkey]={'estimate':corr_report[arm+':all'],'verdict':'held' if meets_prediction(pkey,{'rho':vals}) else 'failed'}
    words_by_q=[math.exp(_logmean(values['instruct'][q]['words'])) for q in range(1,25)]
    letters_by_q=[math.exp(_logmean(values['instruct'][q]['letters'])) for q in range(1,25)]
    pred['149.6']={'estimate':{'words':float(np.median(words_by_q)),'letters':float(np.median(letters_by_q))}}
    pred['149.6']['verdict']='held' if meets_prediction('149.6',pred['149.6']['estimate']) else 'failed'
    baseall=np.median(np.exp([_logmean(values['base'][q]['shuf-all']) for q in range(1,25)]));sftall=np.median(np.exp([_logmean(values['sft'][q]['shuf-all']) for q in range(1,25)]));insall=np.median(np.exp(allp))
    pred['149.7']={'estimate':{'base':float(baseall),'sft':float(sftall),'instruct':float(insall)},'verdict':'held' if meets_prediction('149.7',{'base':float(baseall),'sft':float(sftall),'instruct':float(insall)}) else 'failed'}
    count=0; ratios=[]
    for q in range(1,25):
        prows=[r for r in ins if r['q']==q and r['arm']=='pushed-full']
        ip=float(np.median([math.exp(r['logp_I']+r['logp_m_given_I']) for r in prows])) if prows else float('nan')
        dam=values['instruct'][q]['shuf-all'];dp=math.exp(float(np.mean([r['logp_I']+r['logp_m_given_I'] for r in dam]))) if dam else float('nan')
        ratioq=ip/dp if dp else float('inf');ratios.append(ratioq);count+=ratioq>=5
    pred['149.8']={'estimate':{'questions_at_least_5x':count,'ratios':ratios},'verdict':'held' if meets_prediction('149.8',{'questions_at_least_5x':count}) else 'failed'}
    for k,v in pred.items():result['predictions'][k].update(v)
    result['push_correlations']=corr_report
    result['questions']={str(q):{'instruct':{a:float(np.median(np.exp([r['logp_sorry'] for r in values['instruct'][q][a]]))) if values['instruct'][q][a] else None for a in ARMS},'base_shuf_all':float(np.median(np.exp([r['logp_sorry'] for r in values['base'][q]['shuf-all']]))) if values['base'][q]['shuf-all'] else None,'sft_shuf_all':float(np.median(np.exp([r['logp_sorry'] for r in values['sft'][q]['shuf-all']]))) if values['sft'][q]['shuf-all'] else None} for q in range(1,25)}
    (outdir/'score.json').write_text(json.dumps(_safe(result),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# Part 149 score tables','','| Stage | Damage arm | Median P(sorry) | Range | Median P(I) | Range | Median P(m|I) | Range | Median product | Range |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for stage,rows in (('instruct',ins),('sft',sft),('base',base)):
        for arm in ARMS:
            cells=[r for r in rows if r['arm']==arm]
            if not cells:continue
            def stat(k,exp=False):
                a=np.asarray([r[k] for r in cells],dtype=float);a=np.exp(a) if exp else a
                return f'{np.median(a):.5g}',f'{np.min(a):.5g}–{np.max(a):.5g}'
            sorry=stat('logp_sorry',True); pi=stat('logp_I',True);pm=stat('logp_m_given_I',True);product=stat('logp_I')
            prod=np.exp([r['logp_I']+r['logp_m_given_I'] for r in cells]); product=(f'{np.median(prod):.5g}',f'{np.min(prod):.5g}–{np.max(prod):.5g}')
            lines.append(f'| {stage} | {arm} | {sorry[0]} | {sorry[1]} | {pi[0]} | {pi[1]} | {pm[0]} | {pm[1]} | {product[0]} | {product[1]} |')
    lines+=['','## Predictions','']+[f"- **{k} {v['verdict']}** — {PREDICTIONS[k]}" for k,v in result['predictions'].items()]
    lines+=['','Predictions 149.4 and 149.5 include question baseline removal and neuron/random splits with question bootstrap intervals in score.json.','']
    (outdir/'tables.md').write_text('\n'.join(lines),encoding='utf-8');return result

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',default='results/part149');ap.add_argument('--part146',default='results/part146');ap.add_argument('--out',default='results/part149');a=ap.parse_args(argv);return score(a.input,a.part146,a.out)
if __name__=='__main__':main()
