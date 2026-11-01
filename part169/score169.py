#!/usr/bin/env python3
"""Part 169 host scorer; gate rules are shared with run169."""
from __future__ import annotations
import argparse,json,math,random,sys
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
for p in (str(ROOT),str(HERE)):
    if p not in sys.path:sys.path.insert(0,p)
from part165 import score165
from part166 import score166
from part169 import run169

ARMS=run169.ARMS;LAYERS=run169.LAYERS;SLOTS=run169.SLOTS;PHRASINGS=run169.PHRASINGS
PREDICTIONS=tuple(f'169.{i}' for i in range(1,9))+('169.W1','169.W2')

def relcut(base,alt):
    return None if base is None or alt is None or base==0 else 1-alt/base

def strength_matched(m):
    """Paired rows whose free and clear lists hold nearly the same count of plausible tokens (Weaver's review)."""
    lists=m.get('lists',{});a=lists.get('free',{}).get('top200');b=lists.get('clear',{}).get('top200')
    return a is not None and b is not None and abs(int(a)-int(b))<=1

def blind_hand_sample(battery,seed=169,per_ban=24):
    """Pool by ban, shuffle, then assign ids so numbering carries no ban order."""
    pool=[];rng=random.Random(seed)
    for ban in ('none','own','free','clear','hit','plain50'):
        vals=[a for a in battery if a.get('ban')==ban and a.get('arm') in ARMS]
        pool.extend(rng.sample(vals,min(per_ban,len(vals))))
    rng.shuffle(pool);sample=[];key=[]
    for n,a in enumerate(pool,1):
        blind=f'H169-{n:03d}'
        sample.append({'blind_id':blind,'answer':a.get('answer',''),'token_ids':a.get('token_ids',[]),'token_count':a.get('token_count',0)})
        key.append({'blind_id':blind,'arm':a['arm'],'ban':a['ban'],'layer':a['layer'],'slot':a['slot'],'question_index':a['question_index']})
    return sample,key

def bootstrap_intervals(events,reps=10_000,seed=169,masses=(),no_push=(),w1_above=frozenset(),w1_below=frozenset()):
    """Paired, layer-stratified whole-slot bootstrap shared by both arms and all bans."""
    import numpy as np
    rng=np.random.default_rng(seed); draws=rng.integers(0,SLOTS,size=(reps,len(LAYERS),SLOTS))
    ai={a:i for i,a in enumerate(ARMS)};li={l:i for i,l in enumerate(LAYERS)}
    bykey={(e['arm'],int(e['layer']),int(e['slot']),int(e['question_index']),e['ban']):e for e in events if e.get('arm') in ai and int(e.get('layer',-1)) in li}
    keys={(e['arm'],int(e['layer']),int(e['slot']),int(e['question_index'])) for e in events if e.get('arm') in ai and int(e.get('layer',-1)) in li}
    none_loop={k:bool(bykey.get((*k,'none'),{}).get('loop')) for k in keys}
    mlookup={(m.get('arm'),int(m.get('layer',-1)),int(m.get('slot',-1)),int(m.get('question_index',-1))):m for m in masses}
    paired={k for k,m in mlookup.items() if m.get('paired_set')}; high={k for k,m in mlookup.items() if m.get('high_row')}
    plain2={k for k,m in mlookup.items() if int(m.get('lists',{}).get('free',{}).get('plain_overlap',0))>=2}
    def selected(ban,arms,subset=None,induced=False):
        return [e for k in keys if k[0] in arms and (subset is None or subset(k)) and (not induced or not none_loop.get(k,False)) and (e:=bykey.get((*k,ban))) is not None]
    def tables(vals):
        tab=np.zeros((len(ARMS),len(LAYERS),SLOTS,2),dtype=np.int64)
        for e in vals:
            ix=(ai[e['arm']],li[int(e['layer'])],int(e['slot']));tab[ix+(0,)]+=int(bool(e.get('loop')));tab[ix+(1,)]+=1
        return tab
    cache={}
    def rate_draws(ban,arms,subset=None,induced=False):
        cachekey=(ban,tuple(arms),subset,induced)
        if cachekey not in cache:
            vals=selected(ban,arms,subset,induced);tab=tables(vals);nums=np.zeros(reps,dtype=np.int64);dens=np.zeros(reps,dtype=np.int64)
            for arm in arms:
                for l in range(len(LAYERS)):
                    sample=tab[ai[arm],l,draws[:,l,:],:]
                    nums+=sample[:,:,0].sum(axis=1);dens+=sample[:,:,1].sum(axis=1)
            cache[cachekey]=(nums,dens,(sum(bool(e.get('loop')) for e in vals)/len(vals) if vals else None))
        return cache[cachekey]
    def interval(values,estimate,usable=None):
        vals=np.asarray(values,dtype=float);mask=np.isfinite(vals)
        if usable is not None:mask &= usable
        vals=vals[mask]
        return {'estimate':estimate,'interval_95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))] if len(vals) else [None,None],
          'usable_draws':int(len(vals)),'replicates':reps,'seed':seed}
    def rel_interval(ban,arms,subset=None):
        bn,bd,bp=rate_draws('none',arms,subset);an,ad,ap=rate_draws(ban,arms,subset)
        with np.errstate(divide='ignore',invalid='ignore'):v=1-(an/ad)/(bn/bd)
        v[bd==0]=np.nan;v[(bd>0)&(bn==0)]=np.nan
        return interval(v,relcut(bp,ap),(bd>0)&(bn>0))
    def difference_interval(ban_a,ban_b,arms,subset=None,induced=False,scale=1.,other_scale=-1.):
        an,ad,ap=rate_draws(ban_a,arms,subset,induced);bn,bd,bp=rate_draws(ban_b,arms,subset,induced)
        with np.errstate(divide='ignore',invalid='ignore'):v=scale*(an/ad)+other_scale*(bn/bd)
        usable=(ad>0)&(bd>0)
        est=None if ap is None or bp is None else scale*ap+other_scale*bp
        return interval(v,est,usable)
    out={}
    for ban in ('own','free','clear','plain50','hit'):
        out[ban+'_relative_cut']=rel_interval(ban,list(ARMS))
    out['169.1']=out['own_relative_cut']
    out['169.2']=difference_interval('own','none',list(ARMS),lambda k:k in high) # replaced below by high minus other cut
    highcut=rel_interval('own',list(ARMS),lambda k:k in high);othercut=rel_interval('own',list(ARMS),lambda k:k not in high)
    # Recompute cut contrasts replicate-wise so shared slot draws are retained.
    def cut_values(subset):
        bn,bd,_=rate_draws('none',list(ARMS),subset);an,ad,_=rate_draws('own',list(ARMS),subset)
        with np.errstate(divide='ignore',invalid='ignore'):v=1-(an/ad)/(bn/bd)
        v[(bd==0)|(bn==0)|(ad==0)]=np.nan;return v
    hv,ov=cut_values(lambda k:k in high),cut_values(lambda k:k not in high)
    hpoint,opoint=highcut['estimate'],othercut['estimate'];out['169.2']=interval(hv-ov,None if hpoint is None or opoint is None else hpoint-opoint,np.isfinite(hv)&np.isfinite(ov))
    def rise_interval(arms):
        bn,bd,bp=rate_draws('none',arms);an,ad,ap=rate_draws('free',arms)
        with np.errstate(divide='ignore',invalid='ignore'):v=(an/ad)/(bn/bd)-1
        return interval(v,None if bp in (None,0) or ap is None else ap/bp-1,(bd>0)&(bn>0))
    out['169.3']=rise_interval(list(ARMS))
    out['169.4']=difference_interval('free','clear',list(ARMS),lambda k:k in paired)
    hp=lambda k:k in paired and k in high
    out['169.5']=difference_interval('free','clear',list(ARMS),hp,induced=True)
    out['169.7']=rel_interval('clear',list(ARMS),lambda k:k in paired)
    out['169.8']=difference_interval('hit','free',list(ARMS),lambda k:k in plain2,induced=True,scale=1.,other_scale=-.6)
    matched={k for k,m in mlookup.items() if m.get('paired_set') and strength_matched(m)}
    out['169.4_strength_matched']=difference_interval('free','clear',list(ARMS),lambda k:k in matched)
    an,ad,ap=rate_draws('plain50',list(ARMS),lambda k:k in w1_above,True);bn,bd,bp=rate_draws('plain50',list(ARMS),lambda k:k in w1_below,True)
    with np.errstate(divide='ignore',invalid='ignore'):w1=(an/ad)-(bn/bd)
    out['169.W1']=interval(w1,None if ap is None or bp is None else ap-bp,(ad>0)&(bd>0))
    pn,pd,pp=rate_draws('plain50',list(ARMS));nn,nd,np_=rate_draws('none',list(ARMS));on,od,op=rate_draws('own',list(ARMS))
    out['169.W2']=interval((pn-nn)-(nn-on)/3,None if None in (pp,np_,op) else (pp-np_)*sum(1 for k in keys)-(np_-op)*sum(1 for k in keys)/3)
    for arm in ARMS:
        out[arm]={}
        for ban in ('none','own','free','clear','hit','plain50'):
            n,d,p=rate_draws(ban,[arm]);vals=np.divide(n,d,out=np.full(reps,np.nan),where=d>0);out[arm][ban+'_loop_rate']=interval(vals,p,d>0)
        out[arm]['169.1']=rel_interval('own',[arm]);out[arm]['169.3']=rise_interval([arm]);out[arm]['169.4']=difference_interval('free','clear',[arm],lambda k:k in paired)
    if no_push:
        # No-push free answers are stratified by their n+ slot and phrasing.
        tab=np.zeros((len(LAYERS),SLOTS,2),dtype=np.int64)
        for e in no_push:
            if e.get('arm')=='n+' and e.get('ban')=='free' and int(e.get('layer',-1)) in li:
                ix=(li[int(e['layer'])],int(e['slot']));tab[ix+(0,)]+=int(bool(e.get('loop')));tab[ix+(1,)]+=1
        nums=np.zeros(reps,dtype=np.int64);dens=np.zeros(reps,dtype=np.int64)
        for l in range(len(LAYERS)):
            sm=tab[l,draws[:,l,:],:];nums+=sm[:,:,0].sum(axis=1);dens+=sm[:,:,1].sum(axis=1)
        point=int(tab[...,0].sum())/int(tab[...,1].sum()) if tab[...,1].sum() else None
        out['169.6']=interval(np.divide(nums,dens,out=np.full(reps,np.nan),where=dens>0),point,dens>0)
    else:out['169.6']=interval([],None)
    return out

def score(rows_path,plain_path,masses_path,answers_path,norms_path,out_dir):
    rows=run169.read_jsonl(rows_path);masses=run169.read_jsonl(masses_path);answers=run169.read_jsonl(answers_path)
    plain=json.loads(Path(plain_path).read_text(encoding='utf-8')) if Path(plain_path).exists() else {}
    norms=json.loads(Path(norms_path).read_text(encoding='utf-8')) if Path(norms_path).exists() else {'directions':[]}
    gates=run169.gate_rules(rows,answers,masses,norms)
    gates['gate1']=gates['gate1'] and len([x for x in answers if x.get('stage')=='battery'])==sum(1 for m in masses for b in ('none','own','free','clear','plain50') if m['arm'] in ARMS)+sum(bool(m.get('lists',{}).get('hit',{}).get('ids')) for m in masses)
    stopped=not all(gates.get('gate'+str(i),False) for i in range(1,9)) or gates.get('gate8_stops',True)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    if stopped:
      result={'gates':gates,'readings_stopped':True,'predictions':{}}
      (out/'score.json').write_text(json.dumps(result,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8')
      (out/'tables.md').write_text('# Part 169 score\n\nGates failed; registered readings are stopped.\n\n'+json.dumps(gates)+'\n',encoding='utf-8')
      (out/'hand_read.jsonl').write_text('',encoding='utf-8');(out/'hand_read_key.jsonl').write_text('',encoding='utf-8')
      return result
    battery=[a for a in answers if a.get('stage')=='battery']; cells=defaultdict(list)
    for a in battery:
        ans=a.get('answer','');phrase=score166.loop_phrase(ans);broken=score165.primary_broken(ans)
        cells[(a.get('arm'),a.get('ban'))].append({'arm':a.get('arm'),'ban':a.get('ban'),'layer':int(a.get('layer',-1)),'slot':int(a.get('slot',-1)),'question_index':int(a.get('question_index',-1)),
          'loop':bool(phrase),'broken':bool(broken),'token_count':int(a.get('token_count',len(a.get('token_ids',[])))),'eos':bool(a.get('ended_eos_before_80',False))})
    def rate(arms,ban,subset=None):
        xs=[e for (arm,b),vals in cells.items() if arm in arms and b==ban for e in vals if subset is None or subset(e)]
        return sum(e['loop'] for e in xs)/len(xs) if xs else None
    per={}
    for arm in ARMS:
      per[arm]={}
      for ban in ('none','own','free','clear','plain50','hit'):
        xs=cells[(arm,ban)];n=len(xs)
        if n:per[arm][ban]={'n':n,'loop_rate':sum(x['loop'] for x in xs)/n,'broken_rate':sum(x['broken'] for x in xs)/n,'joint_success_rate':sum((not x['loop'] and not x['broken']) for x in xs)/n,'mean_tokens':sum(x['token_count'] for x in xs)/n,'eos_rate':sum(x['eos'] for x in xs)/n}
    all_events=[e for vals in cells.values() for e in vals];paired={run169.row_key(m) for m in masses if m.get('paired_set')};high={run169.row_key(m) for m in masses if m.get('high_row')}
    paired_pred=lambda e:run169.row_key(e) in paired
    high_pred=lambda e:run169.row_key(e) in high
    # Outcomes are computed from observed denominators; unsupported cells stay not testable.
    none=rate(ARMS,'none');own=rate(ARMS,'own');free=rate(ARMS,'free');clear=rate(ARMS,'clear');plain50=rate(ARMS,'plain50')
    high_own=rate(ARMS,'own',high_pred);high_none=rate(ARMS,'none',high_pred);other_own=rate(ARMS,'own',lambda e:not high_pred(e));other_none=rate(ARMS,'none',lambda e:not high_pred(e))
    pfree=rate(ARMS,'free',paired_pred);pclear=rate(ARMS,'clear',paired_pred);pnone=rate(ARMS,'none',paired_pred)
    none_loop={(e['arm'],e['layer'],e['slot'],e['question_index']):e['loop'] for e in all_events if e['ban']=='none'}
    inducible=lambda ban,subset: [e for e in all_events if e['ban']==ban and (subset is None or subset(e)) and not none_loop.get((e['arm'],e['layer'],e['slot'],e['question_index']),False)]
    def induced_rate(ban,subset=None):
      eligible=inducible(ban,subset);return (sum(e['loop'] for e in eligible)/len(eligible),len(eligible),sum(e['loop'] for e in eligible)) if eligible else (None,0,0)
    hi_paired=lambda e:high_pred(e) and paired_pred(e)
    free_ind,free_den,free_num=induced_rate('free',hi_paired);clear_ind,clear_den,clear_num=induced_rate('clear',hi_paired)
    plain2={run169.row_key(m) for m in masses if m.get('lists',{}).get('free',{}).get('plain_overlap',0)>=2}
    hit_subset=lambda e:run169.row_key(e) in plain2
    hit_ind,hit_den,hit_num=induced_rate('hit',hit_subset)
    free_hit_ind,_,_=induced_rate('free',hit_subset)
    no_push=[a for a in answers if a.get('stage')=='no_push' and a.get('ban')=='free']
    no_push_free=sum(bool(score166.loop_phrase(a.get('answer',''))) for a in no_push)/len(no_push) if no_push else None
    plain50_ind,plain50_den,plain50_num=induced_rate('plain50')
    free_ind_all,free_den_all,free_num_all=induced_rate('free')
    none_ids={run169.row_key(a):set(map(int,a.get('token_ids',[]))) for a in battery if a.get('ban')=='none'}
    mlook={run169.row_key(m):m for m in masses}
    w1_overlap={k:len(none_ids[k]&set(map(int,mlook[k]['lists']['plain50']['ids']))) for k in none_ids if k in mlook}
    w1_rows=sorted(k for k in w1_overlap if not none_loop.get(k,False))
    w1_med=sorted(w1_overlap[k] for k in w1_rows)[len(w1_rows)//2] if w1_rows else None
    w1_above={k for k in w1_rows if w1_overlap[k]>w1_med};w1_below={k for k in w1_rows if w1_overlap[k]<=w1_med}
    w1_a=induced_rate('plain50',lambda e:run169.row_key(e) in w1_above)[0];w1_b=induced_rate('plain50',lambda e:run169.row_key(e) in w1_below)[0]
    loops=lambda ban:sum(e['loop'] for e in all_events if e['ban']==ban)
    w2_n=loops('plain50')-loops('none');w2_own=loops('none')-loops('own')
    predictions={
      '169.1':{'relative_cut':relcut(none,own),'pass':relcut(none,own) is not None and relcut(none,own)>=.25},
      '169.2':{'high_relative_cut':relcut(high_none,high_own),'other_relative_cut':relcut(other_none,other_own),'difference':(relcut(high_none,high_own)-relcut(other_none,other_own)) if relcut(high_none,high_own) is not None and relcut(other_none,other_own) is not None else None,'pass':relcut(high_none,high_own) is not None and relcut(other_none,other_own) is not None and relcut(high_none,high_own)-relcut(other_none,other_own)>=.25},
      '169.3':{'relative_rise':None if none in (None,0) or free is None else free/none-1,'pass':none not in (None,0) and free is not None and free/none-1>=.05},
      '169.4':{'paired_free_rate':pfree,'paired_clear_rate':pclear,'difference':None if pfree is None or pclear is None else pfree-pclear,'testable':gates['testable_169_4_5'],'pass':None if not gates['testable_169_4_5'] else pfree is not None and pclear is not None and pfree-pclear>=.02},
      '169.5':{'free_induced':free_ind,'clear_induced':clear_ind,'difference':None if free_ind is None or clear_ind is None else free_ind-clear_ind,'testable':gates['testable_169_4_5'],'pass':None if not gates['testable_169_4_5'] else free_ind is not None and clear_ind is not None and free_ind-clear_ind>=.10},
      '169.6':{'no_push_free_loop_rate':no_push_free,'pass':no_push_free is not None and no_push_free<.05},
      '169.7':{'paired_clear_relative_cut':relcut(pnone,pclear),'pass':relcut(pnone,pclear) is not None and -.15<=relcut(pnone,pclear)<=.15},
      '169.8':{'hit_induced':hit_ind,'free_induced':free_hit_ind,'pass':hit_ind is not None and free_hit_ind is not None and hit_ind>=.6*free_hit_ind},
      '169.W1':{'induced_above_median_overlap':w1_a,'induced_at_or_below':w1_b,'median_overlap':w1_med,'difference':None if w1_a is None or w1_b is None else w1_a-w1_b,'plain50_induced':plain50_num,'free_induced':free_num_all,'falsifier_met':bool(plain50_num<=free_num_all and not (w1_a is not None and w1_b is not None and w1_a-w1_b>=.05)),'pass':w1_a is not None and w1_b is not None and w1_a-w1_b>=.05},
      '169.W2':{'plain50_net_rise':w2_n,'own_net_cut':w2_own,'pass':w2_own>0 and w2_n>=w2_own/3}}
    predictions['169.1']['per_arm_values']={arm:relcut(rate([arm],'none'),rate([arm],'own')) for arm in ARMS}
    predictions['169.3']['per_arm_values']={arm:(None if rate([arm],'none') in (None,0) or rate([arm],'free') is None else rate([arm],'free')/rate([arm],'none')-1) for arm in ARMS}
    predictions['169.4']['per_arm_values']={arm:(None if rate([arm],'free',paired_pred) is None or rate([arm],'clear',paired_pred) is None else rate([arm],'free',paired_pred)-rate([arm],'clear',paired_pred)) for arm in ARMS}
    no_push_events=[{'arm':a.get('arm'),'ban':a.get('ban'),'layer':int(a.get('layer',-1)),'slot':int(a.get('slot',-1)),'question_index':int(a.get('question_index',-1)),'loop':bool(score166.loop_phrase(a.get('answer','')))} for a in answers if a.get('stage')=='no_push']
    def bite(ban):
        """Exploratory: how often a ban changed the greedy choice, and where the chosen token sat."""
        xs=[a for a in battery if a.get('ban')==ban]
        ranks=sorted(r for a in xs for r in a.get('blocked_ranks',[]))
        firsts=sorted(a['blocked_steps'][0] for a in xs if a.get('blocked_steps'))
        loops=[a for a in xs if score166.loop_phrase(a.get('answer',''))]
        return {'answers':len(xs),'never_blocked':sum(not a.get('blocked_steps') for a in xs),
          'mean_blocked_steps':sum(len(a.get('blocked_steps',[])) for a in xs)/len(xs) if xs else None,
          'median_first_blocked_step':firsts[len(firsts)//2] if firsts else None,
          'median_chosen_rank':ranks[len(ranks)//2] if ranks else None,
          'share_chosen_rank_at_most_5':sum(r<=5 for r in ranks)/len(ranks) if ranks else None,
          'loop_answers':len(loops),'mean_blocked_steps_in_loops':sum(len(a.get('blocked_steps',[])) for a in loops)/len(loops) if loops else None}
    matched_rows={run169.row_key(m) for m in masses if m.get('paired_set') and strength_matched(m)}
    matched_pred=lambda e:run169.row_key(e) in matched_rows
    mf,mc=rate(ARMS,'free',matched_pred),rate(ARMS,'clear',matched_pred);fullgap=predictions['169.4']['difference']
    predictions['169.4']['strength_matched']={'rows':len(matched_rows),'free_rate':mf,'clear_rate':mc,'difference':None if mf is None or mc is None else mf-mc,
      'under_half_of_gap':None if mf is None or mc is None or not fullgap else (mf-mc)<.5*fullgap}
    def spearman(xs,ys):
        import numpy as np
        def ranks(v):
            v=np.asarray(v,dtype=float);order=np.argsort(v,kind='stable');r=np.empty(len(v));r[order]=np.arange(len(v),dtype=float)
            for val in np.unique(v):r[v==val]=r[v==val].mean()
            return r
        if len(xs)<2 or len(set(xs))<2 or len(set(ys))<2:return None
        return float(np.corrcoef(ranks(xs),ranks(ys))[0,1])
    plain50_loop={(e['arm'],e['layer'],e['slot'],e['question_index']):e['loop'] for e in all_events if e['ban']=='plain50'}
    predictions['169.W1']['spearman_overlap_induced']=spearman([w1_overlap[k] for k in w1_rows],[int(bool(plain50_loop.get(k))) for k in w1_rows])
    def cellrates(pred):
        return {'rows':sum(1 for e in cells.get(('n+','none'),[])+cells.get(('k4','none'),[]) if pred(e)),**{b:rate(ARMS,b,pred) for b in ('none','own','free','clear')},'free_induced':induced_rate('free',pred)[:2]}
    def med(vals):
        vals=sorted(vals);return vals[len(vals)//2] if vals else None
    hits=[m['lists']['hit'] for m in masses if m.get('lists',{}).get('hit',{}).get('ids')]
    p2hits=[m['lists']['hit'] for m in masses if run169.row_key(m) in plain2]
    fmass={run169.row_key(m):m['lists']['free']['mass'] for m in masses if m['lists']['free'].get('mass')}
    review={'high_paired':cellrates(hi_paired),'high_unpaired':cellrates(lambda e:high_pred(e) and not paired_pred(e)),
      'list_strength_paired':{n:{'mean_top200':sum(m['lists'][n].get('top200',0) for m in masses if m.get('paired_set'))/max(1,sum(1 for m in masses if m.get('paired_set'))),'median_rank':med([m['lists'][n].get('median_rank') for m in masses if m.get('paired_set') and m['lists'][n].get('median_rank') is not None])} for n in ('own','free','clear')},
      'hit_size':{'rows':len(hits),'mean':sum(len(h['ids']) for h in hits)/len(hits) if hits else None,'median':med([len(h['ids']) for h in hits]),'max':max((len(h['ids']) for h in hits),default=None),'plain2_rows':len(p2hits),'plain2_mean':sum(len(h['ids']) for h in p2hits)/len(p2hits) if p2hits else None,'free_list_size':50},
      'hit_share_of_free_mass':med([m['lists']['hit']['mass']/fmass[run169.row_key(m)] for m in masses if run169.row_key(m) in fmass and m['lists']['hit']['ids']]),
      'no_push_absolute':{b:{'answers':len(v),'loops':sum(bool(score166.loop_phrase(a.get('answer',''))) for a in v),'broken':sum(bool(score165.primary_broken(a.get('answer',''))) for a in v)} for b,v in (('free',no_push),('plain50',[a for a in answers if a.get('stage')=='no_push' and a.get('ban')=='plain50']),('none',[a for a in answers if a.get('stage')=='no_push' and a.get('ban')=='none']))}}
    none_tokens={run169.row_key(a):a.get('token_ids') for a in battery if a.get('ban')=='none'}
    review['never_blocked_differs_from_none']={b:{'never_blocked':sum(1 for a in battery if a.get('ban')==b and not a.get('blocked_steps')),'differs':sum(1 for a in battery if a.get('ban')==b and not a.get('blocked_steps') and a.get('token_ids')!=none_tokens.get(run169.row_key(a)))} for b in ('own','free','clear','hit','plain50')}
    boot=bootstrap_intervals(all_events,masses=masses,no_push=[{'arm':a.get('arm'),'ban':a.get('ban'),'layer':a.get('layer'),'slot':a.get('slot'),'loop':bool(score166.loop_phrase(a.get('answer','')))} for a in no_push],w1_above=w1_above,w1_below=w1_below)
    for i,ban in ((1,'own'),(3,'free')):
      predictions[f'169.{i}']['per_arm']={arm:boot[arm][f'169.{i}'] for arm in ARMS}
    predictions['169.4']['per_arm']={arm:boot[arm]['169.4'] for arm in ARMS}
    result={'gates':gates,'readings_stopped':stopped,'predictions':predictions,'per_arm_ban':per,'ban_bite_exploratory':{b:bite(b) for b in ('none','own','free','clear','hit','plain50')},'review_checks':review,'bootstrap_intervals':boot,
      'ended_induced_net':{b:{'ended':sum(1 for a in battery if a.get('ban')==b and any(x['loop'] for x in cells[(a['arm'],'none')] if (x['layer'],x['slot'],x['question_index'])==(a['layer'],a['slot'],a['question_index'])) and not score166.loop_phrase(a.get('answer',''))),
        'induced':induced_rate(b)[2],'induced_eligible':induced_rate(b)[1],'net_loop_change':None if rate(ARMS,'none') is None or rate(ARMS,b) is None else rate(ARMS,b)-rate(ARMS,'none')} for b in ('own','free','clear','hit','plain50')}}
    (out/'score.json').write_text(json.dumps(result,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8')
    sample,key=blind_hand_sample(battery)
    for name,data in (('hand_read.jsonl',sample),('hand_read_key.jsonl',key)):
      with (out/name).open('w',encoding='utf-8') as f:
        for x in data:f.write(json.dumps(x,ensure_ascii=True)+'\n')
    (out/'tables.md').write_text('# Part 169 score\n\nGates: '+json.dumps(gates)+'\n\nPredictions: '+json.dumps(predictions)+'\n\nPer arm and ban: '+json.dumps(per)+'\n',encoding='utf-8')
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rows',default='results/part169/rows.jsonl');p.add_argument('--plain',default='results/part169/plain.json');p.add_argument('--masses',default='results/part169/masses.jsonl');p.add_argument('--answers',default='results/part169/answers.jsonl');p.add_argument('--norms',default='results/part169/norms.json');p.add_argument('--out',default='results/part169');a=p.parse_args(argv)
    r=score(a.rows,a.plain,a.masses,a.answers,a.norms,a.out);print(json.dumps({'gates':r['gates'],'predictions':r['predictions']},allow_nan=False),flush=True)
if __name__=='__main__':main()
