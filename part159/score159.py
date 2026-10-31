#!/usr/bin/env python3
"""Part 159 scorer. Only numpy is required on the host."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

NAMED = {'L6N753':(6,753),'L7N1169':(7,1169),'L11N8868':(11,8868),'L29N776':(29,776),'L27N3325':(27,3325)}
ARMS=('free','maker_ban','control_ban')
SPECIAL=('alibaba','alibab','alipay','aliyun','tongyi','tongy','阿里','通义','通義')
PREDICTION_TEXT={
  '159.1':'At the first maker veto, the median margin over rows is at least 5 logits',
  '159.2':'In maker_ban, at least 20 of 24 answers, from step t on, contain by case-folded substring one of alibaba, alibab, alipay, aliyun, tongyi, tongy, 阿里, 通义, 通義',
  '159.3':'At most 2 of the 5 named units are on at t on at least 12 rows of maker_ban',
  '159.4':'At least 4 of the 5 named units have R of at least 0.7, so at most one holds',
  '159.5':'For at least 4 of the 5 named units, median D after the maker veto over median D after the control veto is between 0.5 and 2',
  '159.6':'Over t+1 to t+5 in maker_ban, L27 N3325 averages at least 0.5 sd above the free arm on at least 12 rows'}

def load_jsonl(path):
    with Path(path).open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]

def first_veto(pre, emitted):
    pre=np.asarray(pre); emitted=np.asarray(emitted)
    if pre.shape != emitted.shape or pre.ndim != 2: raise ValueError('token arrays must be matching rows by steps')
    diff=pre != emitted
    has=diff.any(axis=1); t=np.full(pre.shape[0],-1,dtype=np.int64)
    t[has]=diff[has].argmax(axis=1)
    return t

def margin_at_t(top, emitted_logit, t):
    top=np.asarray(top); emitted_logit=np.asarray(emitted_logit); t=np.asarray(t)
    valid=t>=0; rows=np.arange(len(t))[valid]
    return rows, top[rows,t[valid]].astype(float)-emitted_logit[rows,t[valid]].astype(float)

def named_stats(free_acts, arm_acts, t, usable_rows, layer, channel):
    """Rest/sd and per-row D, on-at-t; free_acts and arm_acts are [rows,steps,width]."""
    f=np.asarray(free_acts,dtype=np.float32); a=np.asarray(arm_acts,dtype=np.float32)
    if f.ndim != 3 or a.ndim != 3: raise ValueError('activation arrays must be rows x steps x width')
    if channel >= f.shape[2] or channel >= a.shape[2]: raise ValueError('unit channel outside activation width')
    vals=f[:,:,channel]; rest=float(vals.mean()); sd=float(vals.std())
    if not np.isfinite(sd) or sd == 0: return {'rest':rest,'sd':sd,'on':0,'d':{},'on_rows':[],'eligible':[],'median_D':None}
    eligible=[]; on_rows=[]; d={}
    for row in usable_rows:
        at=int(t[row])
        if at < 0: continue
        if abs(float(a[row,at,channel])-rest) >= sd: on_rows.append(int(row))
        if at+5 >= min(f.shape[1],a.shape[1]): continue
        eligible.append(int(row))
        d[int(row)]=float(np.abs(a[row,at+1:at+6,channel]-f[row,at+1:at+6,channel]).mean()/sd)
    return {'rest':rest,'sd':sd,'on':len(on_rows),'d':d,'on_rows':on_rows,'eligible':eligible}

def peer_channels(free_acts, target, named_channels, count=20):
    """The count units nearest target in mean absolute activation, leaving out the named units."""
    f=np.asarray(free_acts,dtype=np.float32)
    activity=np.abs(f).mean(axis=(0,1)); excluded=set(named_channels)|{target}
    candidates=[i for i in range(f.shape[2]) if i not in excluded]
    return sorted(candidates,key=lambda i:(abs(float(activity[i])-float(activity[target])),i))[:count]

def r_value(named_d, peer_ds):
    if not named_d or not peer_ds: return None
    den=float(np.median(peer_ds)); num=float(np.median(list(named_d.values())))
    return None if den == 0 else num/den

def gate1_status(answer_count, all_nonempty, all_finite):
    return 'held' if answer_count==72 and all_nonempty and all_finite else 'failed'
def gate2_status(matches, available=True):
    return 'held' if matches>=22 else ('failed' if available else 'untestable')
def gate3_status(arm_results):
    return 'held' if all(x.get('usable_count',0)>=20 and len(x.get('matching_and_activation_rows',[]))>=20 for x in arm_results.values()) else 'failed'
def reading(on_count, r):
    if on_count < 12: return 'off'
    if r is not None and r < .5: return 'holds'
    if r is not None and r >= .7: return 'moves with its peers'
    return 'unclear'
def prediction_status_159_1(median_margin): return 'held' if median_margin is not None and median_margin>=5 else 'failed'
def prediction_status_159_2(count): return 'held' if count>=20 else 'failed'
def prediction_status_159_3(n_on): return 'held' if n_on<=2 else 'failed'
def prediction_status_159_4(n_moving): return 'held' if n_moving>=4 else 'failed'
def prediction_status_159_5(ratios): return 'held' if sum(.5<=x<=2 for x in ratios if x is not None)>=4 else 'failed'
def prediction_status_159_6(row_count): return 'held' if row_count>=12 else 'failed'

def gate3_arm(pre, emitted, free_emitted, top, emitted_logit, arm_acts, free_acts, sds):
    t=first_veto(pre,emitted); rows, margins=margin_at_t(top,emitted_logit,t)
    first_arm=next(iter(arm_acts.values())); first_free=next(iter(free_acts.values()))
    n=min(len(t),len(free_emitted),first_arm.shape[0],first_free.shape[0])
    usable=[r for r in range(n) if t[r]>=0 and t[r]+5<=39]
    same=[]; same_and_activation=[]
    zero_sd_skipped=sum(int((np.asarray(v,dtype=np.float32)==0).sum()) for v in sds.values())
    for r in usable:
        at=int(t[r]); token_match=np.array_equal(np.asarray(emitted[r,:at]),np.asarray(free_emitted[r,:at]))
        if token_match: same.append(r)
        if token_match:
            coords_ok=True
            for l, acts in arm_acts.items():
                sd=np.asarray(sds[l],dtype=np.float32); active=sd>0
                diff=np.abs(np.asarray(acts[r,at],dtype=np.float32)[active]-np.asarray(free_acts[l][r,at],dtype=np.float32)[active])
                if np.any(diff > .01*sd[active]): coords_ok=False; break
            if coords_ok: same_and_activation.append(r)
    return {'t':t,'veto_rows':len(rows),'usable_rows':usable,'usable_count':len(usable),
       'matching_prefix_rows':same,'matching_and_activation_rows':same_and_activation,
       'margins':margins.tolist(),'gate3_zero_sd_skipped':zero_sd_skipped,
       'pass':len(usable)>=20 and len(same_and_activation)>=20}

def prediction(verdict): return {'verdict':verdict}

def score_predictions(rows, stepdata, arrays, stats, gates):
    out={}; prereq=all(g.get('status')=='held' for g in gates[:3])
    if not prereq:
        return {f'159.{i}':{**prediction('untestable'),'prediction':PREDICTION_TEXT[f'159.{i}']} for i in range(1,7)}
    maker=stepdata['maker_ban']; control=stepdata['control_ban']
    mt=first_veto(maker['pre_token'],maker['emitted_token']); ct=first_veto(control['pre_token'],control['emitted_token'])
    _, margins=margin_at_t(maker['top_logit'],maker['emitted_logit'],mt)
    out['159.1']=prediction(prediction_status_159_1(float(np.median(margins)) if len(margins) else None))
    maker_rows={int(r['question_index']):r for r in rows if r['arm']=='maker_ban'}
    special=0
    for i,t in enumerate(mt):
        if t>=0:
            ids=maker_rows.get(i,{}).get('decoded_emitted_tokens',[])[int(t):]
            if any(s in ''.join(ids).casefold() for s in SPECIAL): special+=1
    out['159.2']=prediction(prediction_status_159_2(special)); out['159.2']['count']=special
    names=list(NAMED)
    on_counts={u:stats[u]['maker_ban']['on'] for u in names}
    out['159.3']=prediction(prediction_status_159_3(sum(v>=12 for v in on_counts.values()))); out['159.3']['units_on_at_least_12']=sum(v>=12 for v in on_counts.values())
    rs={u:stats[u]['maker_ban']['R'] for u in names}
    moved=sum(v is not None and v>=.7 for v in rs.values())
    out['159.4']=prediction(prediction_status_159_4(moved)); out['159.4']['units_R_at_least_0_7']=moved
    ratios={}
    for u in names:
        md=stats[u]['maker_ban']['median_D']; cd=stats[u]['control_ban']['median_D']
        ratios[u]=None if md is None or cd in (None,0) else md/cd
    valid=[v for v in ratios.values() if v is not None]
    nratio=sum(.5<=v<=2 for v in valid)
    out['159.5']=prediction(prediction_status_159_5(valid)); out['159.5']['ratio_by_unit']=ratios; out['159.5']['units_in_range']=nratio
    unit='L27N3325'; f=np.asarray(arrays['free'][27],dtype=np.float32); m=np.asarray(arrays['maker_ban'][27],dtype=np.float32)
    sd=stats[unit]['free']['sd']; rows_above=0
    if sd:
        for r,t in enumerate(mt):
            if t>=0 and t+5 < min(f.shape[1],m.shape[1]) and float((m[r,t+1:t+6,3325]-f[r,t+1:t+6,3325]).mean()) >= .5*sd: rows_above+=1
    out['159.6']=prediction(prediction_status_159_6(rows_above)); out['159.6']['rows']=rows_above
    for key, statement in PREDICTION_TEXT.items(): out[key]['prediction']=statement
    return out

def _json_safe(x):
    if isinstance(x,dict): return {str(k):_json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_json_safe(v) for v in x]
    if isinstance(x,np.ndarray): return _json_safe(x.tolist())
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,)): return float(x) if np.isfinite(x) else None
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def compute(answers, level0, result_dir):
    base=Path(result_dir); steps_file=np.load(base/'steps.npz',allow_pickle=False)
    data={arm:{key:steps_file[f'{arm}_{key}'] for key in ('pre_token','top_logit','emitted_token','emitted_logit')} for arm in ARMS}
    rows=answers; by={(r['arm'],int(r['question_index'])):r for r in rows}
    arrays={arm:{l:np.load(base/f'layers_{arm}_L{l}.npy',mmap_mode='r') for l in (6,7,11,27,29)} for arm in ARMS}
    prompt=np.load(base/'prompt_free.npz',allow_pickle=False)
    finite=all(np.isfinite(data[a][k]).all() for a in ARMS for k in ('top_logit','emitted_logit')) and all(np.isfinite(v).all() for a in ARMS for v in arrays[a].values()) and all(np.isfinite(prompt[f'L{l}']).all() for l in (6,7,11,27,29))
    g1_count=len(rows)==72 and all((a,i) in by and bool(by[a,i].get('answer','').strip()) for a in ARMS for i in range(24))
    gates=[{'gate':1,'status':gate1_status(len(rows),g1_count,finite),'answers_72_nonempty':bool(g1_count),'finite':bool(finite)}]
    l0={(int(r['question_index'])):r.get('answer','') for r in level0 if r.get('arm')=='level_0'}
    match=0
    for i in range(24):
        answer=by.get(('free',i),{}).get('answer',''); ref=l0.get(i,'')
        if answer and ref and (ref.startswith(answer) or answer.startswith(ref)): match+=1
    gates.append({'gate':2,'status':gate2_status(match,g1_count),'free_prefix_matches':match,'required':22})
    sds={}
    for l in (6,7,11,27,29):
        f=np.asarray(arrays['free'][l],dtype=np.float32)
        sds[l]=f.std(axis=(0,1))
    arm3={}
    for arm in ('maker_ban','control_ban'):
        arm3[arm]=gate3_arm(data[arm]['pre_token'],data[arm]['emitted_token'],data['free']['emitted_token'],
          data[arm]['top_logit'],data[arm]['emitted_logit'],arrays[arm],arrays['free'],sds)
    g3ok=all(x['pass'] for x in arm3.values())
    gates.append({'gate':3,'status':gate3_status(arm3) if gates[0]['status']=='held' else 'untestable','arms':_json_safe(arm3)})
    stats={}; named_channels_by_layer={l:[ch for lay,ch in NAMED.values() if lay==l] for l in (6,7,11,27,29)}
    for unit,(layer,ch) in NAMED.items():
        stats[unit]={'free':{}}
        f=np.asarray(arrays['free'][layer],dtype=np.float32); vals=f[:,:,ch]; rest=float(vals.mean()); sd=float(vals.std())
        stats[unit]['free']={'rest':rest,'sd':sd}
        peer=peer_channels(f,ch,named_channels_by_layer[layer],20)
        for arm in ('maker_ban','control_ban'):
            veto=arm3[arm]['t']; us=arm3[arm]['usable_rows']
            ns=named_stats(f,arrays[arm][layer],veto,list(range(min(len(veto),arrays[arm][layer].shape[0]))),layer,ch)
            peer_stats=[named_stats(f,arrays[arm][layer],veto,us,layer,p) for p in peer]
            peer_meds=[float(np.median(list(p['d'].values()))) for p in peer_stats if p['d']]
            md=float(np.median(list(ns['d'].values()))) if ns['d'] else None
            stats[unit][arm]={**ns,'median_D':md,'peer_median_D':float(np.median(peer_meds)) if peer_meds else None,
              'R':r_value(ns['d'],peer_meds),'peers':peer}
    predictions=score_predictions(rows,data,arrays,stats,gates)
    # Exploratory traces: mean named activation displacement in free-arm sd units around each arm's first veto.
    traces={}
    for unit,(layer,ch) in NAMED.items():
        f=np.asarray(arrays['free'][layer],dtype=np.float32); sd=stats[unit]['free']['sd']; traces[unit]={}
        # prompt arrays are per layer in named-unit coordinate order
        pidx=[u for u,(lay,_) in NAMED.items() if lay==layer].index(unit)
        parr=prompt[f'L{layer}'][:,:,pidx]
        # Left padded, so the last 20 positions line up across rows from the end of the prompt.
        traces[unit]['prompt_last20_mean_sd_units']=((parr[:,-20:].mean(axis=0)-stats[unit]['free']['rest'])/sd).astype(float).tolist() if sd else []
        for arm in ARMS:
            tt=np.arange(-2,6); means=[]
            if arm=='free': t=first_veto(data['maker_ban']['pre_token'],data['maker_ban']['emitted_token'])
            else: t=first_veto(data[arm]['pre_token'],data[arm]['emitted_token'])
            a=np.asarray(arrays[arm][layer],dtype=np.float32); diffs=[]
            for r,at in enumerate(t):
                if at>=2 and at+5<a.shape[1]: diffs.append((a[r,at+tt,ch]-stats[unit]['free']['rest'])/sd if sd else np.zeros(8))
            means=np.mean(diffs,axis=0).tolist() if diffs else [None]*8
            traces[unit][arm]={'offsets':tt.tolist(),'mean_sd_units':means}
    top_movers={}
    for l in (6,7,11,27,29):
        vals=[]
        mt=arm3['maker_ban']['t']; ct=arm3['control_ban']['t']; f=np.asarray(arrays['free'][l],dtype=np.float32)
        d_by_arm={}
        for arm,tv in (('maker_ban',mt),('control_ban',ct)):
            a=np.asarray(arrays[arm][l],dtype=np.float32); per=[]
            for r,t in enumerate(tv):
                if t>=0 and t+5<a.shape[1]:
                    if np.any(sds[l]>0): per.append(np.abs(a[r,t+1:t+6]-f[r,t+1:t+6]).mean(axis=0)/np.where(sds[l]>0,sds[l],1))
            d_by_arm[arm]=np.median(per,axis=0) if per else np.full(f.shape[2],np.nan)
        for ch in range(f.shape[2]):
            maker_d=float(d_by_arm['maker_ban'][ch]) if sds[l][ch]>0 and np.isfinite(d_by_arm['maker_ban'][ch]) else None
            control_d=float(d_by_arm['control_ban'][ch]) if sds[l][ch]>0 and np.isfinite(d_by_arm['control_ban'][ch]) else None
            ratio=None if maker_d is None or control_d is None else (float('inf') if control_d==0 and maker_d>0 else (None if control_d==0 else maker_d/control_d))
            vals.append({'channel':ch,'maker_D':maker_d,'control_D':control_d,'ratio':ratio})
        top_movers[str(l)]=sorted(vals,key=lambda x:(-(x['ratio'] if x['ratio'] is not None else -1),x['channel']))[:20]
    margins={arm:arm3[arm]['margins'] for arm in arm3}
    return _json_safe({'gates':gates,'predictions':predictions,'units':stats,'gate3_zero_sd_skipped':arm3['maker_ban']['gate3_zero_sd_skipped'],'margins':{k:{'median':float(np.median(v)) if v else None,'min':min(v) if v else None,'max':max(v) if v else None} for k,v in margins.items()},'routes':{arm:[{'question_index':i,'t':int(t),'route':by.get((arm,i),{}).get('answer','')} for i,t in enumerate(arm3[arm]['t'])] for arm in ('maker_ban','control_ban')},'exploratory':{'traces':traces,'top_movers':top_movers}})

def write_tables(result, out):
    lines=['# Part 159 score','', '## Gates','', '| Gate | Status | Details |','|---|---|---|']
    for g in result['gates']: lines.append(f"| {g['gate']} | {g['status']} | {json.dumps(g,ensure_ascii=True)} |")
    lines += ['', '## Named units','', '| Unit | Arm | On at t | D | Peer median D | R | Reading | 159.5 ratio | Maker veto special (ratio > 2) |','|---|---|---:|---:|---:|---:|---|---:|---|']
    for u,d in result['units'].items():
        vals=[]
        for arm in ('maker_ban','control_ban'):
            v=d[arm]; r=v['R']; ratio=result['predictions']['159.5'].get('ratio_by_unit',{}).get(u)
            reading='off' if v['on']<12 else ('holds' if r is not None and r<.5 else ('moves with its peers' if r is not None and r>=.7 else 'unclear'))
            special='yes' if ratio is not None and ratio>2 else 'no'
            vals.append(f"| {u} | {arm} | {v['on']} | {v['median_D']} | {v['peer_median_D']} | {r} | {reading} | {ratio} | {special} |")
        lines.extend(vals)
    lines += ['', '## 159.5 ratios','',json.dumps(result['predictions']['159.5'],ensure_ascii=True),'','## Margins','']
    for a,m in result['margins'].items(): lines.append(f"- {a}: median={m['median']}, min={m['min']}, max={m['max']}")
    lines += ['', '## Route by row','']
    for arm,rows in result['routes'].items():
        lines.append(f'### {arm}')
        for r in rows: lines.append(f"- row {r['question_index']}: t={r['t']}, {r['route']}")
    lines += ['', '## Exploratory','', 'Exploratory traces and top movers are in traces.json and top_movers.json; they do not enter verdicts.','']
    Path(out,'tables.md').write_text('\n'.join(lines),encoding='utf-8')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--answers',required=True); ap.add_argument('--level0',required=True); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(argv)
    result=compute(load_jsonl(a.answers),load_jsonl(a.level0),a.results); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'score.json').write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    (out/'traces.json').write_text(json.dumps(result['exploratory']['traces'],ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    (out/'top_movers.json').write_text(json.dumps(result['exploratory']['top_movers'],ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    write_tables(result,out)
    print(json.dumps({'gates':result['gates'],'predictions':result['predictions']},ensure_ascii=True))
    if any(g['status']=='failed' for g in result['gates']): raise SystemExit(3)
if __name__=='__main__': main()
