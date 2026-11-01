#!/usr/bin/env python3
"""Part 169: token ban battery testing answer overlap."""
from __future__ import annotations
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
for p in (str(ROOT),str(HERE)):
    if p not in sys.path: sys.path.insert(0,p)
from part165 import run165, score165
from part166 import run166
from part166 import score166
from part164 import run164
from part155 import run155
from part168 import run168

ARMS=('n+','k4'); LAYERS=tuple(range(19,27)); SLOTS=12
PHRASINGS=tuple(range(8)); BANS=('none','own','free','clear','hit','plain50')
HIGH_ROW_MASS=0.0398; BATCH_SIZE=64

def phrasing_items():
    items=run164.choose_questions()
    if len(items)<8 or any(items[i].get('kind')!='maker' for i in range(8)):
        raise ValueError('expected maker questions at phrasing indices 0..7')
    return [(i,items[i]) for i in PHRASINGS]

def neuron_for_seed(width,layer,slot,base,torch=None,attempt=0):
    import torch as th
    g=th.Generator(device='cpu'); g.manual_seed(int(base)+100*int(layer)+int(slot)+1_000_000*int(attempt))
    return int(th.randperm(int(width),generator=g,device='cpu')[0])

def fresh_neuron(width,layer,slot,used,torch):
    attempt=0
    while attempt<=width*4:
        n=neuron_for_seed(width,layer,slot,169000,torch,attempt)
        if n not in used:return n,attempt
        attempt+=1
    raise RuntimeError('unable to draw a fresh n+ neuron')

def prior_neurons_by_layer(widths,torch):
    used={l:{neuron_for_seed(widths[l],l,s,b,torch) for b in (165000,167000) for s in range(SLOTS)} for l in LAYERS}
    prior_path=ROOT/'results/part168/norms.json'
    if prior_path.exists():
        old=json.loads(prior_path.read_text(encoding='utf-8'))
        for n in old.get('directions',[]):
            layer=int(n.get('layer',-1))
            if n.get('arm')=='n+' and n.get('neuron') is not None and layer in used:used[layer].add(int(n['neuron']))
    else:
        for l in LAYERS:
            for s in range(SLOTS):
                attempt=0
                while True:
                    n=neuron_for_seed(widths[l],l,s,168000,torch,attempt)
                    if n not in used[l]:used[l].add(n);break
                    attempt+=1
                    if attempt>widths[l]*4:raise RuntimeError('unable to reconstruct Part 168 neuron set')
    return used

def special_ids(tok):
    result=set(map(int,getattr(tok,'all_special_ids',[]) or []))
    eos=getattr(tok,'eos_token_id',None)
    if eos is not None: result.update(map(int,eos if isinstance(eos,(list,tuple)) else [eos]))
    added=getattr(tok,'get_added_vocab',lambda:{})() or {}
    result.update(map(int,added.values()))
    return result

def ranked_ids(row,tok,special=None):
    special=special_ids(tok) if special is None else special; out=[]
    for x in row.get('top_ids',[]):
        x=int(x)
        if x not in special and x not in out: out.append(x)
    return out

def mass_list_seed(arm,layer,slot,qi,list_name):
    raw=json.dumps([169,str(arm),int(layer),int(slot),int(qi),str(list_name)],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')

def plain50_ids(plain_probs,special,torch):
    """Rank the plain continuation on CPU; XPU vocabulary top-k stalls on the A770."""
    _, candidates=torch.topk(plain_probs.detach().cpu(),k=min(int(plain_probs.numel()),50+len(special)))
    return [int(t) for t in candidates.tolist() if int(t) not in special][:50]

def draw_mass_ids(q,own,matched,special,arm,layer,slot,qi,extra_excluded=(),list_name='free',noises=8,steps=20,redraws=200):
    """Part 168 Gumbel calibration; extra_excluded is the sole free/clear recipe difference."""
    import numpy as np
    q=np.asarray(q,dtype=np.float64); blocked=set(map(int,own))|set(map(int,matched))|set(map(int,special))|set(map(int,extra_excluded))
    allowed=np.ones(q.size,dtype=bool)
    for t in blocked:
        if 0<=t<q.size: allowed[t]=False
    cand=np.flatnonzero(allowed); order=cand[np.argsort(q[cand],kind='stable')[::-1]][:20000]
    if len(order)<50: raise ValueError('fewer than 50 eligible mass-list tokens')
    pool=q[order]; own_mass=float(q[np.asarray(own,dtype=int)].sum()); tol=max(.1*own_mass,1e-7)
    feasible=float(pool[-50:].sum())-tol<=own_mass<=float(pool[:50].sum())+tol
    logs=np.log(np.clip(pool,1e-30,1.0)); rng=np.random.default_rng(mass_list_seed(arm,layer,slot,qi,list_name))
    def gumbel(): return -np.log(-np.log(np.clip(rng.random(len(order)),1e-15,1-1e-15)))
    def top(z): return np.argpartition(z,-50)[-50:]
    noise=[gumbel() for _ in range(noises)]; lo,hi=-4.,8.
    if feasible:
        for _ in range(steps):
            mid=(lo+hi)/2; realized=float(np.mean([pool[top(mid*logs+g)].sum() for g in noise]))
            if realized<own_mass:lo=mid
            else:hi=mid
    exponent=(lo+hi)/2; best=None; best_delta=float('inf'); success=False; draws=0
    for draws in range(1,redraws+1):
        ix=top(exponent*logs+gumbel()); mass=float(pool[ix].sum()); delta=abs(mass-own_mass)
        if delta<best_delta:best,best_delta=ix,delta
        if delta<=tol:success=True;break
    repaired=False
    if not success:
        have=float(pool[best].sum()); held=set(map(int,best)); need=own_mass-have
        for i in rng.permutation(best):
            at=int(np.searchsorted(-pool,-(pool[i]+need)))
            for c in (at-1,at):
                if 0<=c<len(pool) and c not in held and abs(have-pool[i]+pool[c]-own_mass)<=tol:
                    best=np.array([c if int(x)==int(i) else int(x) for x in best]); best_delta=abs(have-float(pool[i])+float(pool[c])-own_mass); success=repaired=True;break
            if repaired:break
    return list(map(int,order[best])),exponent,draws,success,float(best_delta),own_mass,bool(feasible),repaired

def list_record(q,own,ids,answer_ids,exp,draws,ok,err,ownmass,feasible,repaired,special):
    ids=list(map(int,ids)); return {'ids':ids,'mass':float(np_sum(q,ids)),'exponent':exp,'draws':draws,
      'in_tolerance':bool(ok),'repaired':bool(repaired),'feasible':bool(feasible),
      'plain_overlap':len(set(ids)&set(map(int,answer_ids))),'error':float(err),'own_mass':float(ownmass),
      'special_free':not bool(set(ids)&special)}

def np_sum(q,ids):
    return float(sum(float(q[int(i)]) for i in ids))

def freeze_flags(own_mass,free,clear):
    """Freeze eligibility before generation; callers persist these fields in masses.jsonl."""
    return {'high_row':float(own_mass)>=HIGH_ROW_MASS,
      'paired_set':bool(free.get('feasible') and clear.get('feasible') and free.get('in_tolerance') and clear.get('in_tolerance'))}

def logging_processor(log):
    """A RowBanProcessor that records, per step, where the ban changed the greedy choice.

    Each log entry holds a row flag and the chosen token's rank among the unbanned scores. The rank comes from a
    comparison on the device, not a sort: a vocabulary-wide top-k stalls the A770. Exploratory, from Muse's prior-art
    reply: Xu et al. (arXiv:2206.02369) predict a loop token is already near the top when a ban lands."""
    class Logged(run168.RowBanProcessor):
        def __call__(self,input_ids,scores):
            out=super().__call__(input_ids,scores)
            moved=scores.argmax(-1)!=out.argmax(-1)
            rank=(scores>scores.gather(1,out.argmax(-1,keepdim=True))).sum(-1)
            log.append((moved.detach().cpu(),rank.detach().cpu()))
            return out
    return Logged

def attach_blocks(answers,log):
    """Copy the step log onto each answer, up to that answer's own length."""
    for i,a in enumerate(answers):
        steps=[(t,int(entry[1][i])) for t,entry in enumerate(log[:len(a['token_ids'])]) if bool(entry[0][i])]
        a['blocked_steps']=[s for s,_ in steps];a['blocked_ranks']=[r for _,r in steps]
    return answers

def with_block_log(fn,*args,**kwargs):
    """Run a generation with the logged processor in place of run168's, restored afterwards."""
    log=[];original=run168.RowBanProcessor;run168.RowBanProcessor=logging_processor(log)
    try:answers=fn(*args,**kwargs)
    finally:run168.RowBanProcessor=original
    return attach_blocks(answers,log)

def generate_unpushed_batch(model,tok,question,banned_ids,common,device,max_new_tokens=80,call_log=None):
    """Generation control that applies a ban without installing any activation hook."""
    import torch
    from transformers import LogitsProcessorList
    inputs=common.token_inputs(tok,question,len(banned_ids),device); plen=int(inputs['input_ids'].shape[-1])
    vocab=int(model.get_output_embeddings().weight.shape[0]);mask=run168.make_mask(banned_ids,vocab,device,torch)
    proc=run168.RowBanProcessor(mask)
    if call_log is not None:call_log.append({'hook_installs':0,'ban_count':len(banned_ids)})
    with torch.inference_mode():
      seq=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,pad_token_id=getattr(tok,'pad_token_id',None),eos_token_id=common.eos_ids(tok,model) or None,logits_processor=LogitsProcessorList([proc]))
    eos=set(map(int,common.eos_ids(tok,model)));results=[]
    for seqrow in seq[:,plen:].detach().cpu().tolist():
      ids=[int(x) for x in seqrow]
      for i,x in enumerate(ids):
        if x in eos:ids=ids[:i+1];break
      results.append({'answer':tok.decode(ids,skip_special_tokens=True).strip(),'token_ids':ids,'token_count':len(ids),'ended_eos_before_80':bool(ids and ids[-1] in eos and len(ids)<80)})
    return results

def generate_unpushed(model,tok,question,banned_ids,common,device,max_new_tokens=80,call_log=None):
    """Single-row compatibility wrapper around the no-hook batched path."""
    return generate_unpushed_batch(model,tok,question,[banned_ids],common,device,max_new_tokens,call_log)[0]

def all_rows(widths):
    return [{'arm':a,'layer':l,'slot':s,'k':1 if a=='n+' else 4,'question_index':qi}
      for qi in PHRASINGS for l in LAYERS for s in range(SLOTS) for a in ARMS]

def row_key(r): return (r.get('arm'),int(r.get('layer',-1)),int(r.get('slot',-1)),int(r.get('question_index',-1)))
def answer_key(r): return (r.get('stage'),r.get('arm'),r.get('ban'),r.get('layer'),r.get('slot'),r.get('question_index'))
def read_jsonl(path):
    p=Path(path)
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []
def write_json(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def gate2_plain(answers):
    plain=[a for a in answers if a.get('stage')=='no_push' and a.get('arm')=='plain' and a.get('ban') is None]
    return len(plain)==8 and {int(a.get('question_index',-1)) for a in plain}==set(PHRASINGS) and all(not score165.primary_broken(a.get('answer','')) and not score166.loop_phrase(a.get('answer','')) for a in plain)

def gate_rules(rows,answers,masses,norms,baseline_plain=None,expected_rows=None,expected_norms=2*8*12,expected_nplus=8*12,expected_list_size=50):
    """Single gate authority imported by score169; returns gate flags and diagnostics."""
    expected_rows=expected_rows if expected_rows is not None else 2*8*12*8
    battery=[a for a in answers if a.get('stage')=='battery']; no_push=[a for a in answers if a.get('stage')=='no_push']
    plain=[a for a in no_push if a.get('arm')=='plain' and a.get('ban') is None]
    keys=[row_key(r) for r in rows]; akeys=[answer_key(a) for a in battery]
    finite=all(all(math.isfinite(float(r.get(k,float('nan')))) for k in ('bias_norm','steadiness')) and len(r.get('top_ids',[]))>=200 and len(r.get('top_values',[]))>=200 and all(math.isfinite(float(v)) for v in r.get('top_values',[])) for r in rows)
    npkeys=[answer_key(a) for a in no_push]
    expected_nopush=sum(m.get('arm')=='n+' for m in masses)+3*len(PHRASINGS)
    g1=len(rows)==expected_rows and len(set(keys))==len(keys) and len(set(akeys))==len(akeys) and all(isinstance(a.get('answer'),str) and a['answer'].strip() and a.get('token_ids') for a in battery) and finite
    g2=gate2_plain(answers)
    normvals=norms.get('directions',[]) if isinstance(norms,dict) else norms
    g3=len(normvals)==expected_norms and all(float(n.get('relative_error',math.inf))<=.01 for n in normvals)
    rmap={row_key(r):r for r in rows}; mmap={row_key(m):m for m in masses}; g4=True
    expected_answers=set()
    for m in masses:
        for ban in ('none','own','free','clear','plain50'):
            expected_answers.add(('battery',m['arm'],ban,int(m['layer']),int(m['slot']),int(m['question_index'])))
        if m.get('lists',{}).get('hit',{}).get('ids'):
            expected_answers.add(('battery',m['arm'],'hit',int(m['layer']),int(m['slot']),int(m['question_index'])))
    g1=g1 and set(akeys)==expected_answers and len(no_push)==expected_nopush and len(set(npkeys))==len(npkeys) and all(isinstance(a.get('answer'),str) and a['answer'].strip() and a.get('token_ids') for a in no_push)
    for m in masses:
        lists=m.get('lists',{}); k=row_key(m); row=rmap.get(k,{})
        for name in ('own','free','clear','plain50'):
            ids=list(map(int,lists.get(name,{}).get('ids',[])))
            if len(ids)!=expected_list_size or len(set(ids))!=len(ids) or set(ids)&set(map(int,m.get('special_ids',[]))):g4=False
        if list(map(int,lists.get('own',{}).get('ids',[])))!=list(map(int,row.get('own_ban_ids',[]))):g4=False
        plain_ids=set(map(int,m.get('plain_answer_ids',[])));free_ids=set(map(int,lists.get('free',{}).get('ids',[])))
        if set(map(int,lists.get('clear',{}).get('ids',[])))&plain_ids:g4=False
        if set(map(int,lists.get('hit',{}).get('ids',[])))!=(free_ids&plain_ids):g4=False
    no_push_lookup={(a.get('arm'),a.get('ban'),int(a.get('layer',-1)),int(a.get('slot',-1)),int(a.get('question_index',-1))):a for a in no_push}
    if len(plain)!=len(PHRASINGS) or len({int(a.get('question_index',-1)) for a in plain})!=len(PHRASINGS):g1=False
    if any(a.get('banned_ids',[]) for a in plain):g4=False
    for qi in PHRASINGS:
        for ban in ('none','plain50'):
            a=no_push_lookup.get(('plain',ban,-1,-1,int(qi)))
            if a is None:g1=False;continue
            want=[] if ban=='none' else list(map(int,next((m['lists']['plain50']['ids'] for m in masses if int(m['question_index'])==qi),next((m['lists']['plain50']['ids'] for m in masses),[]))))
            if list(map(int,a.get('banned_ids',[])))!=want:g4=False
    for m in masses:
        if m.get('arm')!='n+':continue
        a=no_push_lookup.get(('n+','free',int(m['layer']),int(m['slot']),int(m['question_index'])))
        if a is None or list(map(int,a.get('banned_ids',[])))!=list(map(int,m.get('lists',{}).get('free',{}).get('ids',[]))):g4=False
    for a in no_push:
        if set(map(int,a.get('banned_ids',[])))&set(map(int,a.get('token_ids',[]))):g4=False
    for a in battery:
        k=row_key(a); ban=a.get('ban'); rec=mmap.get(k,{})
        want=[] if ban in ('none','no_push') else rec.get('lists',{}).get(ban,{}).get('ids',[])
        if list(map(int,a.get('banned_ids',[])))!=list(map(int,want)) or set(map(int,want))&set(map(int,a.get('token_ids',[]))):g4=False;break
        if set(map(int,want))&set(map(int,rec.get('special_ids',[]))):g4=False;break
        if ban=='clear' and set(map(int,want))&set(map(int,rec.get('plain_answer_ids',[]))):g4=False;break
        if ban=='hit' and set(map(int,want))!=(set(map(int,rec.get('lists',{}).get('free',{}).get('ids',[])))&set(map(int,rec.get('plain_answer_ids',[])))):g4=False;break
    none=[a for a in battery if a.get('ban')=='none']; nplus=[a for a in none if a.get('arm')=='n+']
    g5=bool(nplus and none) and sum(score165.primary_broken(a.get('answer','')) for a in nplus)/len(nplus)>=.08 and sum(bool(score166.loop_phrase(a.get('answer',''))) for a in none)/len(none)>=.08
    free=[m for m in masses if m.get('arm') in ARMS]; share=sum(bool(m.get('lists',{}).get('free',{}).get('in_tolerance')) for m in free)/len(free) if free else 0
    freekeys={row_key(m) for m in free}
    g6=len(free)==expected_rows and len(freekeys)==expected_rows and freekeys==set(keys) and share>=.90 and all(len(m.get('lists',{}).get('free',{}).get('ids',[]))==50 for m in free)
    g7=True; nplus_norms=[n for n in normvals if n.get('arm')=='n+']
    if len(nplus_norms)!=expected_nplus or len({(int(n['layer']),int(n['slot'])) for n in nplus_norms})!=expected_nplus:g7=False
    seen={}
    for n in normvals:
        if n.get('arm')!='n+':continue
        l=int(n['layer']); neu=int(n.get('neuron',-1)); seen.setdefault(l,set())
        prior=set(map(int,norms.get('prior_neurons',{}).get(str(l),[]))) if isinstance(norms,dict) else set()
        if not prior:prior={neuron_for_seed(int(n['width']),l,s,b) for b in (165000,167000,168000) for s in range(12)}
        if neu in prior or neu in seen[l]:g7=False
        seen[l].add(neu)
    hi=[m for m in free if m.get('high_row')]; paired_high=sum(bool(m.get('paired_set')) for m in hi)
    clear_holds=all(not (set(map(int,m.get('lists',{}).get('clear',{}).get('ids',[])))&set(map(int,m.get('plain_answer_ids',[])))) for m in masses)
    plain_hits=[m for m in hi if m.get('lists',{}).get('free',{}).get('plain_overlap',0)>=1]
    mean_hit=sum(m.get('lists',{}).get('free',{}).get('plain_overlap',0) for m in hi)/len(hi) if hi else 0
    special_clear=all(not (set(map(int,v.get('ids',[]))) & set(map(int,m.get('special_ids',[])))) for m in masses for v in m.get('lists',{}).values())
    g8=clear_holds and mean_hit>=1 and special_clear
    return {'gate1':g1,'gate2':g2,'gate3':g3,'gate4':g4,'gate5':g5,'gate6':g6,'gate7':g7,'gate8':g8,
      'free_in_tolerance_share':share,'paired_high_rows':paired_high,'testable_169_4_5':paired_high>=150,
      'high_free_plain_overlap_mean':mean_hit,'gate8_stops':not g8, 'expected_rows':expected_rows}

# Part 166 owns the loop phrase used by Part 168's extra168 reader.
def _plain_ids(text,tok,n=80):
    return list(map(int,tok.encode(text,add_special_tokens=False)[:n]))

def run(args,loader=None):
    import torch
    common=run155._common(); out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    if loader is None:
        import neuron_report as nr
        tok,model=nr.load_model(str(args.model),common.REVISION,args.device);model.requires_grad_(False);model.eval()
    else:tok,model=loader(args.device)
    questions=dict(phrasing_items()); layers=common.find_layers(model); widths={l:int(layers[l].mlp.down_proj.weight.shape[1]) for l in LAYERS}
    answers=read_jsonl(out/'answers.jsonl'); done={answer_key(a) for a in answers}
    # Stage 0 and Gate 2 precede direction construction and every subsequent stage.
    with (out/'answers.jsonl').open('a',encoding='utf-8') as af:
      for qi,item in questions.items():
        k=('no_push','plain',None,None,None,qi)
        if k not in done:
            _,ids=run168.generate_plain(model,tok,item['question'],common,args.device,80); text=tok.decode(ids,skip_special_tokens=True).strip()
            a={'stage':'no_push','arm':'plain','question_index':qi,'answer':text,'token_ids':ids,'token_count':len(ids)};af.write(json.dumps(a)+'\n');af.flush();answers.append(a)
      gate2=gate2_plain(answers)
      write_json(out/'stage0_gate.json',{'gate2':gate2})
      if not gate2:raise SystemExit('Gate 2 failed at Stage 0')
    plain80={int(a['question_index']):list(map(int,a.get('token_ids',[]))) for a in answers if a.get('stage')=='no_push' and a.get('arm')=='plain' and a.get('ban') is None}
    medians=json.loads((ROOT/'results/part112/medians.json').read_text());dirs={};norms=[];cache={}
    prior_neurons=prior_neurons_by_layer(widths,torch)
    for l in LAYERS:
      width=widths[l];used=set(prior_neurons[l])
      for s in range(SLOTS):
        neuron,attempt=fresh_neuron(width,l,s,used,torch);used.add(neuron)
        original=run165._seeded_raw
        def draw(w,ln,ss,th,att=attempt):
          width0,hidden0=int(w.shape[1]),int(w.shape[0]);g=th.Generator(device='cpu');g.manual_seed(169000+100*int(ln)+int(ss)+1_000_000*int(att))
          return th.randperm(width0,generator=g,device='cpu'),th.randn(width0,generator=g,dtype=th.float32,device='cpu'),th.randn(hidden0,generator=g,dtype=th.float32,device='cpu')
        run165._seeded_raw=draw
        try:d,m=run165.build_directions(layers[l].mlp.down_proj.weight,l,s,float(medians.get(str(l),medians.get(l))),cache)
        finally:run165._seeded_raw=original
        dirs[l,s]=d
        for arm in ARMS:norms.append({'arm':arm,'layer':l,'slot':s,'width':width,'neuron':neuron,'attempt':attempt,**m[arm]})
    write_json(out/'norms.json',{'part':169,'dose':.8,'directions':norms,'prior_neurons':{str(l):sorted(prior_neurons[l]) for l in LAYERS}})
    plain=json.loads((out/'plain.json').read_text()) if (out/'plain.json').exists() else {'answers':[]};pmap={int(x['question_index']):x for x in plain.get('answers',[])}
    plain50_by_qi={}
    for qi,item in questions.items():
      if qi not in pmap:
        prompt,ids=run168.generate_plain(model,tok,item['question'],common,args.device,40);pmap[qi]={'question_index':qi,'question':item['question'],'prompt_ids':prompt,'answer_ids':ids}
    plain['answers']=[pmap[i] for i in sorted(pmap)];write_json(out/'plain.json',plain)
    rows=read_jsonl(out/'rows.jsonl'); rmap={row_key(r):r for r in rows}; massmap={row_key(m):m for m in read_jsonl(out/'masses.jsonl')}; special=special_ids(tok)|set(map(int,common.eos_ids(tok,model))); head=model.get_output_embeddings()
    for qi,item in questions.items():
      p=pmap[qi];plen=len(p['prompt_ids']);ans=p['answer_ids'];pos=run166.evaluation_positions(plen,len(ans));inp=torch.tensor(p['prompt_ids']+ans,dtype=torch.long,device=args.device).unsqueeze(0)
      plog=run166.plain_position_logits(head,run166.forward_hidden(model,inp,pos)) if pos else None
      plain_probs=torch.stack([torch.softmax(x.float(),-1) for x in plog]).mean(0) if plog is not None else None
      if plain_probs is not None:
        plain50=plain50_ids(plain_probs,special,torch)
      else:plain50=[]
      plain50_by_qi[qi]=plain50
      work=[{'arm':a,'layer':l,'slot':s,'k':1 if a=='n+' else 4,'question_index':qi} for l in LAYERS for s in range(SLOTS) for a in ARMS]
      random.Random(169+qi).shuffle(work)
      for start in range(0,len(work),args.batch_size):
        missing=[r for r in work[start:start+args.batch_size] if row_key(r) not in rmap]
        if not missing:continue
        push=torch.stack([dirs[r['layer'],r['slot']][r['arm']] for r in missing]).to(args.device,dtype=torch.float32);hooks=[]
        for ln in sorted({r['layer'] for r in missing}):
          dense=torch.zeros_like(push)
          for i,r in enumerate(missing):
            if r['layer']==ln:dense[i]=push[i]
          hooks.append(run166.attach_position_push(layers[ln].mlp.down_proj,dense,plen,len(ans)))
        try:hidden=run166.forward_hidden(model,inp.expand(len(missing),-1).contiguous(),pos)
        finally:
          for h in hooks:h.remove()
        bias,norm,*_=run166.position_metrics(hidden,head,plog,common.eos_ids(tok,model));vals=bias.detach().float().cpu(); topv,topid=torch.topk(vals,k=min(200,vals.shape[-1]),dim=-1)
        with (out/'rows.jsonl').open('a',encoding='utf-8') as f:
          for i,r in enumerate(missing):
            rr={**r,'bias_norm':float(torch.linalg.vector_norm(vals[i])),'steadiness':float(torch.linalg.vector_norm(vals[i])/max(float(norm[i]),1e-30)),'top_ids':topid[i].tolist(),'top_values':topv[i].tolist()};rr['own_ban_ids']=ranked_ids(rr,tok,special)[:50];f.write(json.dumps(rr)+'\n');rmap[row_key(rr)]=rr
          f.flush()
      mass_work=[r for r in work if row_key(r) not in massmap]
      for start in range(0,len(mass_work),args.batch_size):
        missing=mass_work[start:start+args.batch_size]
        if not missing:continue
        pushes=torch.stack([dirs[r['layer'],r['slot']][r['arm']] for r in missing]).to(args.device,dtype=torch.float32);hooks=[]
        for ln in sorted({r['layer'] for r in missing}):
          dense=torch.zeros_like(pushes)
          for i,r in enumerate(missing):
            if r['layer']==ln:dense[i]=pushes[i]
          hooks.append(run166.attach_position_push(layers[ln].mlp.down_proj,dense,plen,len(ans)))
        try:
          hidden=run166.forward_hidden(model,inp.expand(len(missing),-1).contiguous(),pos) if pos else None
        finally:
          for h in hooks:h.remove()
        qrows=run168.list_masses(hidden,head) if pos else torch.zeros((len(missing),int(head.weight.shape[0])),dtype=torch.float32)
        for r,qrow in zip(missing,qrows):
          key=row_key(r)
          row=rmap[key];own=ranked_ids(row,tok,special)[:50];donor=ranked_ids(rmap[(r['arm'],r['layer'],(r['slot']+1)%SLOTS,qi)],tok,special)[:50]
          q=qrow
          plainids=set(plain80[qi]);lists={'own':{'ids':own,'mass':np_sum(q,own),'exponent':None,'draws':0,'in_tolerance':True,'repaired':False,'feasible':True,'plain_overlap':len(set(own)&plainids)},'plain50':{'ids':plain50,'mass':np_sum(q,plain50),'exponent':None,'draws':0,'in_tolerance':None,'repaired':False,'feasible':None,'plain_overlap':len(set(plain50)&plainids)}}
          for name,extra in (('free',()),('clear',plain80[qi])):
            ids,exp,draws,ok,err,om,feasible,repaired=draw_mass_ids(q.numpy(),own,donor,special,r['arm'],r['layer'],r['slot'],qi,extra,list_name=name)
            lists[name]=list_record(q.numpy(),own,ids,plain80[qi],exp,draws,ok,err,om,feasible,repaired,special)
          lists['hit']={'ids':sorted(set(lists['free']['ids'])&plainids),'mass':0.,'exponent':None,'draws':0,'in_tolerance':None,'repaired':False,'feasible':True,'plain_overlap':len(set(lists['free']['ids'])&plainids)}
          for v in lists.values():v['mass']=np_sum(q.numpy(),v['ids'])
          # Weaver's review: equal mass is not equal strength, so log how plausible each list's tokens are under q.
          import numpy as np
          order=np.argsort(-q.numpy(),kind='stable');ranks=np.empty(order.size,dtype=np.int64);ranks[order]=np.arange(order.size)
          for v in lists.values():
            if v['ids']:
              held=ranks[np.asarray(v['ids'],dtype=np.int64)];v['top200']=int((held<200).sum());v['median_rank']=float(np.median(held))
          flags=freeze_flags(lists['own']['mass'],lists['free'],lists['clear'])
          rec={**r,'lists':lists,'plain_answer_ids':plain80[qi],'special_ids':sorted(special),**flags}
          massmap[key]=rec
          with (out/'masses.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush()
    rows=read_jsonl(out/'rows.jsonl');masses=read_jsonl(out/'masses.jsonl');answers=read_jsonl(out/'answers.jsonl')
    gates=gate_rules(rows,answers,masses,{'directions':norms,'prior_neurons':{str(l):sorted(prior_neurons[l]) for l in LAYERS}})
    write_json(out/'prelaunch_gates.json',{k:v for k,v in gates.items() if k not in ('gate1','gate2','gate4','gate5')})  # the others need answers
    if not gates['gate6'] or not gates['gate8']:raise SystemExit('Gate 6 or Gate 8 stop condition failed before generation')
    # Stage 2: each ban uses the same batched hook/generation path; no-push controls never attach a hook.
    mass_lookup={row_key(m):m for m in masses}
    pilot_started=None;pilot_answers=pilot_batches=0
    with (out/'answers.jsonl').open('a',encoding='utf-8') as af:
      for qi,item in questions.items():
        work=[]
        for r in [x for x in rows if int(x['question_index'])==qi]:
          for ban in ('none','own','free','clear','plain50'):
            work.append({**r,'ban':ban})
          if mass_lookup[row_key(r)]['lists']['hit']['ids']:work.append({**r,'ban':'hit'})
        random.Random(169+qi).shuffle(work)
        for start in range(0,len(work),args.batch_size):
          batch=[b for b in work[start:start+args.batch_size] if answer_key({'stage':'battery',**b}) not in done]
          if not batch:continue
          banned=[[] if b['ban']=='none' else mass_lookup[row_key(b)]['lists'][b['ban']]['ids'] for b in batch]
          if pilot_started is None:pilot_started=time.perf_counter()
          generated=with_block_log(run168.generate_batch,model,tok,layers,item['question'],batch,dirs,banned,args.device,common,80,169+qi)
          for b,ban_ids,ans in zip(batch,banned,generated):
            rec={'stage':'battery',**{k:b[k] for k in ('arm','layer','slot','question_index','ban')},'banned_ids':ban_ids,**ans};af.write(json.dumps(rec)+'\n');done.add(answer_key(rec))
          af.flush();pilot_answers+=len(batch);pilot_batches+=1
          if pilot_batches==3:
            rate=pilot_answers/max(time.perf_counter()-pilot_started,1e-9);print(json.dumps({'pilot_answers_per_second':rate}),flush=True)
            if run168.pilot_rate_should_stop(pilot_answers,pilot_started,time.perf_counter()):raise SystemExit(f'pilot stop: {rate:.3f} answers/s below 1.5')
        # No-push controls use batched masks and never install an activation hook.
        plain_work=[]
        for ban,ids in [('none',[]),('plain50',plain50_by_qi[qi])]:
          key=('no_push','plain',ban,None,None,qi)
          if key not in done:plain_work.append((ban,ids,key))
        if plain_work:
          generated=with_block_log(generate_unpushed_batch,model,tok,item['question'],[x[1] for x in plain_work],common,args.device,80)
          for (ban,ids,key),ans in zip(plain_work,generated):
            rec={'stage':'no_push','arm':'plain','ban':ban,'question_index':qi,'banned_ids':ids,**ans};af.write(json.dumps(rec)+'\n');done.add(answer_key(rec))
          af.flush()
        free_work=[]
        for r in [x for x in rows if int(x['question_index'])==qi and x['arm']=='n+']:
          key=('no_push','n+','free',r['layer'],r['slot'],qi)
          if key not in done:free_work.append((r,mass_lookup[row_key(r)]['lists']['free']['ids']))
        for start in range(0,len(free_work),args.batch_size):
          chunk=free_work[start:start+args.batch_size]
          generated=with_block_log(generate_unpushed_batch,model,tok,item['question'],[x[1] for x in chunk],common,args.device,80)
          for (r,ids),ans in zip(chunk,generated):
            rec={'stage':'no_push','arm':'n+','ban':'free','layer':r['layer'],'slot':r['slot'],'question_index':qi,'banned_ids':ids,**ans};af.write(json.dumps(rec)+'\n');done.add(answer_key(rec))
          af.flush()
    token_ids=set()
    for old in read_jsonl(out/'rows.jsonl'):token_ids.update(map(int,old.get('top_ids',[])))
    for old in masses:
      for value in old.get('lists',{}).values():token_ids.update(map(int,value.get('ids',[])))
    for old in read_jsonl(out/'answers.jsonl'):token_ids.update(map(int,old.get('token_ids',[])))
    write_json(out/'tokens.json',{str(t):tok.decode([t],skip_special_tokens=False) for t in sorted(token_ids)})
    return {'rows':len(rows),'masses':len(masses),'answers':len(read_jsonl(out/'answers.jsonl')),'gates':gates}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='xpu');p.add_argument('--model',default=run165.MODEL_DEFAULT);p.add_argument('--out-dir',default='results/part169');p.add_argument('--batch-size',type=int,default=64);a=p.parse_args(argv)
    if a.batch_size!=64:p.error('registered batch size is 64')
    print(json.dumps(run(a),ensure_ascii=True),flush=True)
if __name__=='__main__':main()
