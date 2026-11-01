#!/usr/bin/env python3
"""Part 168: mass-matched token-ban battery on the Arc."""
from __future__ import annotations
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
for p in (str(ROOT),str(HERE)):
    if p not in sys.path: sys.path.insert(0,p)
from part165 import run165, score165
from part166 import run166
from part164 import run164
from part155 import run155

ARMS=('n+','k4','full'); LAYERS=tuple(range(19,27)); SLOTS=12
PHRASINGS=(8,9,10,11,12,14,15)
BANS=('none','own','matched','mass'); BATCH_SIZE=64

def seeded_raw(weight,layer_no,slot,torch,attempt=0):
    width,hidden=int(weight.shape[1]),int(weight.shape[0]); g=torch.Generator(device='cpu')
    g.manual_seed(168000+100*int(layer_no)+int(slot)+1_000_000*int(attempt))
    return (torch.randperm(width,generator=g,device='cpu'),
            torch.randn(width,generator=g,dtype=torch.float32,device='cpu'),
            torch.randn(hidden,generator=g,dtype=torch.float32,device='cpu'))

def neuron_for_seed(width,layer_no,slot,base,torch,attempt=0):
    g=torch.Generator(device='cpu'); g.manual_seed(int(base)+100*int(layer_no)+int(slot)+1_000_000*int(attempt))
    return int(torch.randperm(int(width),generator=g,device='cpu')[0])

def fresh_neuron(width,layer_no,slot,used,torch):
    attempt=0
    while True:
        neuron=neuron_for_seed(width,layer_no,slot,168000,torch,attempt)
        if neuron not in used:return neuron,attempt
        attempt+=1
        if attempt>width*4: raise RuntimeError('unable to draw a fresh n+ neuron')

def build_directions(weight,layer_no,slot,median,pinv_cache=None,attempt=0):
    original=run165._seeded_raw
    def draw(w,l,s,t): return seeded_raw(w,l,s,t,attempt)
    run165._seeded_raw=draw
    try: return run165.build_directions(weight,layer_no,slot,median,pinv_cache)
    finally: run165._seeded_raw=original

def phrasing_items():
    items=run164.choose_questions()
    if len(items)!=16 or any(x.get('kind')!='maker' for x in items): raise ValueError('expected 16 maker items')
    return [(i,items[i]) for i in PHRASINGS]

def all_rows(widths):
    return [{'arm':a,'layer':l,'slot':s,'k':1 if a=='n+' else 4 if a=='k4' else int(widths[l]),'question_index':qi}
            for qi in PHRASINGS for l in LAYERS for s in range(SLOTS) for a in ARMS]

def row_base(widths):
    return [{'arm':a,'layer':l,'slot':s,'k':1 if a=='n+' else 4 if a=='k4' else int(widths[l])}
            for l in LAYERS for s in range(SLOTS) for a in ARMS]

def row_key(r): return (r.get('arm'),int(r.get('layer',-1)),int(r.get('slot',-1)),int(r.get('question_index',-1)))
def answer_key(r): return (r.get('stage'),r.get('arm'),r.get('ban'),r.get('layer'),r.get('slot'),r.get('question_index'))
def read_jsonl(path):
    p=Path(path)
    if not p.exists(): return []
    with p.open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]
def _write_json(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def ranked_ids(row,tok):
    special=set(map(int,getattr(tok,'all_special_ids',[]) or [])); eos=getattr(tok,'eos_token_id',None)
    if eos is not None: special.update([int(eos)] if not isinstance(eos,(list,tuple)) else map(int,eos))
    ids=[]
    for tid in row.get('top_ids',[]):
        tid=int(tid)
        if tid not in special and tid not in ids: ids.append(tid)
    return ids

def build_ban_ids(row, donor, tok, top_n=50):
    src=donor if donor is not None else row
    ids=ranked_ids(src,tok)[:top_n]
    if len(ids)!=top_n: raise ValueError(f'expected {top_n} distinct non-special ids, got {len(ids)}')
    return ids

def ban_lists(rows,rowmap,tok,massmap=None):
    if not hasattr(rowmap,'get'):
        rowmap={(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index'])):r for r in rowmap}
    massmap=massmap or {}
    out=[]
    for r in rows:
        key=(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index']))
        own=rowmap[key]
        donor=rowmap[(r['arm'],int(r['layer']),(int(r['slot'])+1)%SLOTS,int(r['question_index']))]
        result={'none':[],'own':build_ban_ids(own,None,tok),'matched':build_ban_ids(own,donor,tok)}
        result['mass']=list(map(int,massmap.get(key,{}).get('mass_ban_ids',[]))) if r['arm'] in ('n+','k4') else []
        if r['arm'] in ('n+','k4') and len(result['mass'])!=50: raise ValueError(f'missing fixed mass ban list for {key}')
        out.append(result)
    return out

class RowBanProcessor:
    def __init__(self,mask): self.mask=mask
    def __call__(self,input_ids,scores):
        mask=self.mask
        if mask.shape!=scores.shape: raise ValueError(f'mask {tuple(mask.shape)} != scores {tuple(scores.shape)}')
        return scores.masked_fill(mask,-float('inf'))

def make_mask(index_lists,vocab,device,torch):
    mask=torch.zeros((len(index_lists),int(vocab)),dtype=torch.bool,device=device)
    pairs=[(ri,int(t)) for ri,ids in enumerate(index_lists) for t in ids if 0<=int(t)<int(vocab)]
    if pairs:
        rr=torch.tensor([x[0] for x in pairs],device=device,dtype=torch.long); cc=torch.tensor([x[1] for x in pairs],device=device,dtype=torch.long)
        mask[rr,cc]=True
    return mask

def generate_batch(model,tok,layers,question,rows,directions,ban_ids,device,common,max_new_tokens=80,seed=168):
    import torch
    from transformers import LogitsProcessorList
    inputs=common.token_inputs(tok,question,len(rows),device)
    prompt_len=int(inputs['input_ids'].shape[-1]); pushes=torch.stack([directions[(int(r['layer']),int(r['slot']))][r['arm']] for r in rows]).to(device=device,dtype=torch.float32)
    handles=[]
    for ln in sorted({int(r['layer']) for r in rows}):
        dense=torch.zeros_like(pushes)
        for i,r in enumerate(rows):
            if int(r['layer'])==ln: dense[i]=pushes[i]
        handles.append(common.AddRowsHook(layers[ln].mlp.down_proj,dense))
    vocab=int(model.get_output_embeddings().weight.shape[0]); mask=make_mask(ban_ids,vocab,device,torch)
    processor=RowBanProcessor(mask)
    for h in handles: h.begin_sequence()
    try:
        with torch.inference_mode(),torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            seq=model.generate(**inputs,max_new_tokens=int(max_new_tokens),do_sample=False,use_cache=True,
                pad_token_id=getattr(tok,'pad_token_id',None),eos_token_id=common.eos_ids(tok,model) or None,
                logits_processor=LogitsProcessorList([processor]))
    finally:
        for h in handles: h.remove()
    eos=set(map(int,common.eos_ids(tok,model))); answers=[]
    for seqrow in seq[:,prompt_len:].detach().cpu().tolist():
        ids=[int(t) for t in seqrow]
        for i,t in enumerate(ids):
            if t in eos: ids=ids[:i+1]; break
        answers.append({'answer':tok.decode(ids,skip_special_tokens=True).strip(),'token_ids':ids,'token_count':len(ids),'ended_eos_before_80':bool(ids and ids[-1] in eos and len(ids)<80)})
    return answers

def make_row_record(row,question_index,bias,norm,top_ids,top_values):
    import torch
    cpu=bias.detach().float().cpu(); std=cpu.std(unbiased=False)
    sharp=None if float(std)==0 else float((cpu.max()-cpu.mean())/std)
    bn=float(torch.linalg.vector_norm(cpu)); nv=float(norm.detach().float().cpu())
    rec={**row,'question_index':int(question_index),'sharpness':sharp,'steadiness':bn/max(nv,torch.finfo(torch.float32).tiny),'bias_norm':bn,'top_ids':[int(x) for x in top_ids],'top_values':[float(x) for x in top_values]}
    if not all(math.isfinite(rec[k]) for k in ('steadiness','bias_norm')) or (sharp is not None and not math.isfinite(sharp)):
        raise FloatingPointError('nonfinite row metric')
    return rec

def list_masses(hidden,head):
    """Return mean pushed softmax vectors on CPU as float32; never top-k on device."""
    import torch
    positions=int(hidden.shape[1])
    if positions<1: raise ValueError('mass measurement requires at least one evaluation position')
    total=None
    for pos in range(positions):
        probs=torch.softmax(head(hidden[:,pos,:]).float(),dim=-1).detach().to(device='cpu',dtype=torch.float32)
        total=probs if total is None else total+probs
    return total/positions

def _mass_rng(arm,layer,slot,qi):
    raw=json.dumps([168,str(arm),int(layer),int(slot),int(qi)],separators=(',',':')).encode()
    seed=int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')
    import numpy as np
    return np.random.default_rng(seed)

MASS_ABS_FLOOR=1e-7

def draw_mass_ids(q,own,matched,special,arm,layer,slot,qi,noises=8,steps=20,redraws=200):
    """Draw 50 distinct tokens from the eligible pool, with weights q**a by Gumbel top-k.

    The exponent is calibrated on that same without-replacement law, using common noise across trials, because the
    with-replacement shortcut misstates heavy tails. Tolerance is 10% of own mass with an absolute floor. A target
    outside the pool's reachable range is logged as infeasible, not searched for.
    Returns ids, exponent, draws, in_tolerance, absolute error, own mass, feasible."""
    import numpy as np
    q=np.asarray(q,dtype=np.float64); allowed=np.ones(q.size,dtype=bool)
    for tid in set(map(int,own))|set(map(int,matched))|set(map(int,special)):
        if 0<=tid<q.size: allowed[tid]=False
    candidate=np.flatnonzero(allowed); order=candidate[np.argsort(q[candidate],kind='stable')[::-1]][:20000]
    if len(order)<50: raise ValueError('fewer than 50 eligible mass-list tokens')
    pool=q[order]; own_mass=float(q[np.asarray(own,dtype=int)].sum()); tolerance=max(.1*own_mass,MASS_ABS_FLOOR)
    feasible=float(pool[-50:].sum())-tolerance<=own_mass<=float(pool[:50].sum())+tolerance
    logs=np.log(np.clip(pool,1e-30,1.0)); rng=_mass_rng(arm,layer,slot,qi)
    def gumbel(): return -np.log(-np.log(np.clip(rng.random(len(order)),1e-15,1-1e-15)))
    def top(z): return np.argpartition(z,-50)[-50:]
    noise=[gumbel() for _ in range(noises)]; lo,hi=-4.0,8.0
    if feasible:
        for _ in range(steps):
            mid=(lo+hi)/2; realized=float(np.mean([pool[top(mid*logs+g)].sum() for g in noise]))
            if realized<own_mass: lo=mid
            else: hi=mid
    exponent=(lo+hi)/2; best=None; best_delta=float('inf'); success=False; draws=0
    for draws in range(1,redraws+1):
        ix=top(exponent*logs+gumbel()); mass=float(pool[ix].sum()); delta=abs(mass-own_mass)
        if delta<best_delta: best,best_delta=ix,delta
        if delta<=tolerance: success=True; break
    return list(map(int,order[best])),exponent,draws,success,float(best_delta),own_mass,bool(feasible)

def position_diagnostic(hidden,head,own_lists,mass_lists):
    """Per position: pushed mass on the own and mass lists, and whether the pushed top token is on each."""
    import torch
    own_ids=torch.as_tensor(own_lists,dtype=torch.long,device=hidden.device); mass_ids=torch.as_tensor(mass_lists,dtype=torch.long,device=hidden.device); out=[]
    for pos in range(int(hidden.shape[1])):
        probs=torch.softmax(head(hidden[:,pos,:]).float(),dim=-1); best=probs.argmax(dim=-1,keepdim=True)
        out.append(torch.stack([probs.gather(1,own_ids).sum(-1),probs.gather(1,mass_ids).sum(-1),(own_ids==best).any(-1).float(),(mass_ids==best).any(-1).float()],dim=-1).detach().float().cpu())
    return torch.stack(out,dim=1)

def verify_lists(massmap,special):
    """Every low-k row has exactly 50 distinct eligible mass ids before any generation."""
    for key,rec in massmap.items():
        if key[0] in ('n+','k4'):
            ids=list(map(int,rec.get('mass_ban_ids',[])))
            if len(ids)!=50 or len(set(ids))!=50 or set(ids)&set(special): raise SystemExit(f'bad mass list at {key}')

def pending_rows(rows,done,qi):
    return [r for r in rows if row_key({**r,'question_index':qi}) not in done]

def pending_answers(rows,done):
    return [r for r in rows if answer_key({'stage':'battery',**r}) not in done]

def pilot_rate_should_stop(answers,started,now,threshold=1.5):
    return int(answers)/max(float(now)-float(started),1e-9)<float(threshold)

def generate_plain(model,tok,question,common,device,n=40):
    import torch
    inputs=common.token_inputs(tok,question,1,device); plen=int(inputs['input_ids'].shape[-1]); eos=set(map(int,common.eos_ids(tok,model)))
    with torch.inference_mode(): seq=model.generate(**inputs,max_new_tokens=n,do_sample=False,use_cache=True,pad_token_id=getattr(tok,'pad_token_id',None),eos_token_id=list(eos) or None)
    ids=[int(x) for x in seq[0,plen:].detach().cpu().tolist()]
    for i,t in enumerate(ids):
        if t in eos: ids=ids[:i+1]; break
    return inputs['input_ids'][0].detach().cpu().tolist(),ids

def run(args,loader=None):
    import torch
    common=run155._common(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rowpath=out/'rows.jsonl'; answerpath=out/'answers.jsonl'; plainpath=out/'plain.json'; masspath=out/'masses.jsonl'
    rows_done={row_key(x) for x in read_jsonl(rowpath)}; answers=read_jsonl(answerpath); answer_done={answer_key(x) for x in answers}
    if loader is None:
        import neuron_report as nr
        tok,model=nr.load_model(str(args.model),common.REVISION,args.device); model.requires_grad_(False); model.eval()
    else: tok,model=loader(args.device)
    questions=dict(phrasing_items())
    # Stage 0 is the first model work, and completed rows survive resume.
    with answerpath.open('a',encoding='utf-8',newline='\n') as af:
        for qi,item in questions.items():
            key=('no_push','plain',None,None,None,qi)
            prior=next((a for a in answers if answer_key(a)==key),None)
            if prior is None:
                _,ids=generate_plain(model,tok,item['question'],common,args.device,80)
                text=tok.decode(ids,skip_special_tokens=True).strip()
                prior={'stage':'no_push','arm':'plain','question_index':qi,'answer':text,'token_ids':ids,'token_count':len(ids)}
                af.write(json.dumps(prior,ensure_ascii=True)+'\n'); af.flush(); answers.append(prior); answer_done.add(answer_key(prior))
            if score165.primary_broken(prior.get('answer','')):
                raise SystemExit(f"Stage 0 primary-broken plain answer at phrasing {qi}")
    layers=common.find_layers(model); widths={l:int(layers[l].mlp.down_proj.weight.shape[1]) for l in LAYERS}
    medians=json.loads((ROOT/'results/part112/medians.json').read_text(encoding='utf-8'))
    dirs={}; norms=[]; cache={}
    for l in LAYERS:
        w=layers[l].mlp.down_proj.weight; width=widths[l]; med=float(medians.get(str(l),medians.get(l)))
        used={neuron_for_seed(width,l,s,b,torch) for b in (165000,167000) for s in range(SLOTS)}
        for slot in range(SLOTS):
            neuron,attempt=fresh_neuron(width,l,slot,used,torch)
            used.add(neuron)
            d,m=build_directions(w,l,slot,med,cache,attempt)
            dirs[(l,slot)]=d
            for arm in ARMS: norms.append({'layer':l,'slot':slot,'arm':arm,'neuron':neuron,'attempt':attempt,'width':width,**m[arm]})
    _write_json(out/'norms.json',{'part':168,'dose':0.8,'directions':norms})
    plain=json.loads(plainpath.read_text(encoding='utf-8')) if plainpath.exists() else {'answers':[]}
    plainmap={int(x['question_index']):x for x in plain.get('answers',[])}
    for qi,item in questions.items():
        if qi not in plainmap:
            prompt,ids=generate_plain(model,tok,item['question'],common,args.device,40)
            plainmap[qi]={'question_index':qi,'question':item['question'],'prompt_ids':prompt,'answer_ids':ids,'P':len(ids)}
        p=plainmap[qi]; plen=len(p['prompt_ids']); ids=p['answer_ids']; positions=run166.alignment_positions(plen,len(ids)); seq=torch.tensor(p['prompt_ids']+ids,dtype=torch.long,device=args.device).unsqueeze(0)
        with torch.inference_mode():
            logits=run166._forward_selected(model,seq,positions) if positions else None
            again=run166._forward_selected(model,seq,positions) if positions else None
        deterministic=bool(torch.equal(logits,again)) if logits is not None else True
        cpu=logits[0].float().detach().cpu() if logits is not None else None
        argmax=cpu.argmax(-1).tolist() if cpu is not None else []
        mismatches=[{'index':i,'generated':int(t),'argmax':int(argmax[i])} for i,t in enumerate(ids[:len(argmax)]) if int(t)!=int(argmax[i])]
        p.update({'alignment_matches':len(argmax)-len(mismatches),'alignment_total':len(ids),'deterministic':deterministic,'alignment_mismatches':mismatches})
    plain['answers']=[plainmap[i] for i in sorted(plainmap)]
    _write_json(plainpath,plain)
    rowbase=row_base(widths); head=model.get_output_embeddings()
    for qi,item in questions.items():
        p=plainmap[qi]; plen=len(p['prompt_ids']); ans=p['answer_ids']; positions=run166.evaluation_positions(plen,len(ans)); inp=torch.tensor(p['prompt_ids']+ans,dtype=torch.long,device=args.device).unsqueeze(0)
        plogs=run166.plain_position_logits(head,run166.forward_hidden(model,inp,positions)) if positions else []
        shuffled=list(rowbase); random.Random(168+qi).shuffle(shuffled)
        with rowpath.open('a',encoding='utf-8',newline='\n') as fh:
            for start0 in range(0,len(shuffled),args.batch_size):
                batch=shuffled[start0:start0+args.batch_size]; missing=[r for r in batch if row_key({**r,'question_index':qi}) not in rows_done]
                if not missing: continue
                push=torch.stack([dirs[(r['layer'],r['slot'])][r['arm']] for r in missing]).to(args.device,dtype=torch.float32); hooks=[]
                for ln in sorted({r['layer'] for r in missing}):
                    dense=torch.zeros_like(push)
                    for i,r in enumerate(missing):
                        if r['layer']==ln:dense[i]=push[i]
                    hooks.append(run166.attach_position_push(layers[ln].mlp.down_proj,dense,plen,len(ans)))
                try: hidden=run166.forward_hidden(model,inp.expand(len(missing),-1).contiguous(),positions)
                finally:
                    for h in hooks:h.remove()
                bias,norm,*_=run166.position_metrics(hidden,head,plogs,common.eos_ids(tok,model))
                cpu=bias.detach().float().cpu(); topv,topid=torch.topk(cpu,k=min(200,cpu.shape[-1]),dim=-1)
                for i,r in enumerate(missing):
                    rec=make_row_record(r,qi,cpu[i],norm[i].detach().cpu(),topid[i].tolist(),topv[i].tolist())
                    ordered=ranked_ids(rec,tok); rec['own_ban_ids']=ordered[:50]; rec['rank_51_200_ids']=ordered[50:200]
                    fh.write(json.dumps(rec,ensure_ascii=True)+'\n'); rows_done.add(row_key(rec))
                fh.flush()
    finalrows=read_jsonl(rowpath); rmap={(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index'])):r for r in finalrows}
    mass_rows=read_jsonl(masspath); massmap={(x['arm'],int(x['layer']),int(x['slot']),int(x['question_index'])):x for x in mass_rows}
    mass_done=set(massmap)
    special=set(map(int,getattr(tok,'all_special_ids',[]) or [])); eos=getattr(tok,'eos_token_id',None)
    if eos is not None: special.update([int(eos)] if not isinstance(eos,(list,tuple)) else map(int,eos))
    for qi,item in questions.items():
        p=plainmap[qi]; plen=len(p['prompt_ids']); ans=p['answer_ids']; positions=run166.evaluation_positions(plen,len(ans))
        inp=torch.tensor(p['prompt_ids']+ans,dtype=torch.long,device=args.device).unsqueeze(0)
        for start0 in range(0,len(rowbase),args.batch_size):
            batch=[{**r,'question_index':qi} for r in rowbase[start0:start0+args.batch_size]]
            missing=[r for r in batch if (r['arm'],r['layer'],r['slot'],qi) not in mass_done]
            if not missing:continue
            pushes=torch.stack([dirs[(r['layer'],r['slot'])][r['arm']] for r in missing]).to(args.device,dtype=torch.float32); hooks=[]
            for ln in sorted({r['layer'] for r in missing}):
                dense=torch.zeros_like(pushes)
                for i,r in enumerate(missing):
                    if r['layer']==ln:dense[i]=pushes[i]
                hooks.append(run166.attach_position_push(layers[ln].mlp.down_proj,dense,plen,len(ans)))
            try: hidden=run166.forward_hidden(model,inp.expand(len(missing),-1).contiguous(),positions) if positions else None
            finally:
                for h in hooks:h.remove()
            qrows=list_masses(hidden,head) if positions else torch.zeros((len(missing),int(model.get_output_embeddings().weight.shape[0])),dtype=torch.float32)
            recs=[]
            for i,r in enumerate(missing):
                key=(r['arm'],r['layer'],r['slot'],qi); own=rmap[key]['own_ban_ids']; donor=rmap[(r['arm'],r['layer'],(r['slot']+1)%SLOTS,qi)]['own_ban_ids']
                q=qrows[i].numpy().astype('float64',copy=False); own_mass=float(q[own].sum()); matched_mass=float(q[donor].sum()); massids=[]; exponent=None; draws=0; tol=None; abserr=None; feasible=None
                if r['arm'] in ('n+','k4'): massids,exponent,draws,tol,abserr,_,feasible=draw_mass_ids(q,own,donor,special,*key)
                plain_ids=set(map(int,ans)); rec={'arm':r['arm'],'layer':r['layer'],'slot':r['slot'],'question_index':qi,'own_mass':own_mass,'matched_mass':matched_mass,'mass_mass':float(q[massids].sum()) if massids else None,'own_mass_overlap':len(set(own)&set(massids)),'matched_mass_overlap':len(set(donor)&set(massids)),'overlap':len(set(own)&set(donor)),'exponent':exponent,'draws':draws,'in_tolerance':bool(tol) if tol is not None else None,'feasible':feasible,'mass_abs_error':abserr,'mass_rel_error':(abserr/own_mass if abserr is not None and own_mass>0 else None),'paired_set':bool(tol and feasible) if tol is not None else None,'allowed_vocab':int(q.size-len({x for x in (set(own)|set(donor)|special) if 0<=int(x)<q.size})),'plain_overlap_own':len(plain_ids&set(own)),'plain_overlap_matched':len(plain_ids&set(donor)),'plain_overlap_mass':len(plain_ids&set(massids)),'mass_ban_ids':massids}
                recs.append(rec)
            lowix=[i for i,x in enumerate(recs) if x['arm'] in ('n+','k4')]
            if positions and lowix:
                diag=position_diagnostic(hidden[lowix],head,[rmap[(recs[i]['arm'],recs[i]['layer'],recs[i]['slot'],qi)]['own_ban_ids'] for i in lowix],[recs[i]['mass_ban_ids'] for i in lowix])
                with (out/'positions.jsonl').open('a',encoding='utf-8',newline='\n') as pf:
                    for n,i in enumerate(lowix):
                        x=recs[i]; d=diag[n]
                        pf.write(json.dumps({'arm':x['arm'],'layer':x['layer'],'slot':x['slot'],'question_index':qi,'positions':int(d.shape[0]),'own_by_position':[float(f'{v:.4g}') for v in d[:,0].tolist()],'mass_by_position':[float(f'{v:.4g}') for v in d[:,1].tolist()],'top_on_own':int(d[:,2].sum()),'top_on_mass':int(d[:,3].sum())},ensure_ascii=True,allow_nan=False)+'\n')
            with masspath.open('a',encoding='utf-8',newline='\n') as mf:
                for rec in recs:
                    key=(rec['arm'],rec['layer'],rec['slot'],qi); mf.write(json.dumps(rec,ensure_ascii=True,allow_nan=False)+'\n'); massmap[key]=rec; mass_done.add(key)
    lowmass=[x for x in massmap.values() if x['arm'] in ('n+','k4')]
    share=sum(bool(x['in_tolerance']) for x in lowmass)/len(lowmass) if lowmass else 0.0
    ratios=[float(x['mass_mass'])/float(x['own_mass']) for x in lowmass if x['mass_mass'] is not None and float(x['own_mass'])>0]
    median_ratio=sorted(ratios)[len(ratios)//2] if ratios else None
    verify_lists(massmap,special)
    coverage={}
    for x in lowmass:
        k=f"{x['arm']}|{x['layer']}"; c=coverage.setdefault(k,[0,0]); c[0]+=bool(x.get('paired_set')); c[1]+=1
    print(json.dumps({'gate6_in_tolerance_share':share,'median_mass_over_own':median_ratio,'infeasible_rows':sum(x.get('feasible') is False for x in lowmass),'paired_coverage':coverage}),flush=True)
    if share<.90: raise SystemExit(f'Gate 6 mass match failed: in-tolerance share {share:.4f} below 0.90')
    rowmap=rmap; pilot_start=None; pilot_batches=pilot_answers=0
    with answerpath.open('a',encoding='utf-8',newline='\n') as af:
        for qi,item in questions.items():
            work=[]
            for r0 in rowbase:
                r={**r0,'question_index':qi}
                for ban in BANS:
                    if ban!='mass' or r['arm'] in ('n+','k4'):work.append({**r,'ban':ban})
            random.Random(168+qi).shuffle(work); lists=ban_lists(work,rowmap,tok,massmap)
            for start0 in range(0,len(work),64):
                chunk=work[start0:start0+64]; selected=[(r,lists[start0+i]) for i,r in enumerate(chunk) if answer_key({'stage':'battery',**r}) not in answer_done]
                if not selected:continue
                chunk=[x[0] for x in selected]; bl=[x[1][x[0]['ban']] for x in selected]
                if pilot_start is None:pilot_start=time.perf_counter()
                generated=generate_batch(model,tok,layers,item['question'],chunk,dirs,bl,args.device,common,80,168+qi)
                for r,a,banned in zip(chunk,generated,bl):
                    rec={'stage':'battery','arm':r['arm'],'layer':r['layer'],'slot':r['slot'],'question_index':qi,'ban':r['ban'],'banned_ids':banned,**a}
                    af.write(json.dumps(rec,ensure_ascii=True)+'\n'); answer_done.add(answer_key(rec))
                af.flush(); pilot_answers+=len(chunk); pilot_batches+=1
                if pilot_batches==3:
                    rate=pilot_answers/max(time.perf_counter()-pilot_start,1e-9); print(json.dumps({'pilot_answers_per_second':rate}),flush=True)
                    if pilot_rate_should_stop(pilot_answers,pilot_start,time.perf_counter()):raise SystemExit(f'pilot stop: {rate:.3f} answers/s below 1.5')
    token_ids=set()
    for old in finalrows:token_ids.update(map(int,old.get('top_ids',[])))
    for old in read_jsonl(answerpath):token_ids.update(map(int,old.get('token_ids',[])))
    _write_json(out/'tokens.json',{str(t):tok.decode([t],skip_special_tokens=False) for t in sorted(token_ids)})
    return {'rows':len(rows_done),'answers':len(answer_done),'phrasing_count':len(questions)}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',default='xpu'); p.add_argument('--model',default=run165.MODEL_DEFAULT); p.add_argument('--out-dir',default='results/part168'); p.add_argument('--batch-size',type=int,default=64)
    a=p.parse_args(argv)
    if a.batch_size!=64:p.error('registered batch size is 64')
    print(json.dumps(run(a),ensure_ascii=True),flush=True)
if __name__=='__main__': main()
