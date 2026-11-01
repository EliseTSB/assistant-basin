#!/usr/bin/env python3
"""Part 167: held-out bias rows and per-row token-ban battery on the Arc."""
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
for p in (str(ROOT),str(HERE)):
    if p not in sys.path: sys.path.insert(0,p)
from part165 import run165
from part166 import run166
from part164 import run164
from part155 import run155

ARMS=('n+','k4','full'); LAYERS=tuple(range(19,27)); SLOTS=12
BANS=('none','own','matched'); BATCH_SIZE=64

def seeded_raw(weight,layer_no,slot,torch):
    width,hidden=int(weight.shape[1]),int(weight.shape[0]); g=torch.Generator(device='cpu')
    g.manual_seed(167000+100*int(layer_no)+int(slot))
    return (torch.randperm(width,generator=g,device='cpu'),
            torch.randn(width,generator=g,dtype=torch.float32,device='cpu'),
            torch.randn(hidden,generator=g,dtype=torch.float32,device='cpu'))

def build_directions(weight,layer_no,slot,median,pinv_cache=None):
    original=run165._seeded_raw
    run165._seeded_raw=seeded_raw
    try: return run165.build_directions(weight,layer_no,slot,median,pinv_cache)
    finally: run165._seeded_raw=original

def phrasing_items():
    items=run164.choose_questions()
    if len(items)!=16 or any(x.get('kind')!='maker' for x in items): raise ValueError('expected 16 maker items')
    return [(i,items[i]) for i in range(8,16)]

def all_rows(widths):
    return [{'arm':a,'layer':l,'slot':s,'k':1 if a=='n+' else 4 if a=='k4' else int(widths[l]),'question_index':qi}
            for qi in range(8,16) for l in LAYERS for s in range(SLOTS) for a in ARMS]

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

def ban_lists(rows,rowmap,tok):
    if not hasattr(rowmap,'get'):
        rowmap={(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index'])):r for r in rowmap}
    out=[]
    for r in rows:
        key=(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index']))
        # Work rows carry no top ids: both lists come from the stored bias rows.
        own=rowmap[key]
        donor=rowmap[(r['arm'],int(r['layer']),(int(r['slot'])+1)%SLOTS,int(r['question_index']))]
        out.append({'none':[],'own':build_ban_ids(own,None,tok),'matched':build_ban_ids(own,donor,tok)})
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

def generate_batch(model,tok,layers,question,rows,directions,ban_ids,device,common,max_new_tokens=80,seed=167):
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

def list_masses(hidden,head,own_lists,matched_lists):
    """Mean pushed softmax mass on each row's two fixed token lists."""
    import torch
    own_ids=torch.as_tensor(own_lists,dtype=torch.long,device=hidden.device)
    matched_ids=torch.as_tensor(matched_lists,dtype=torch.long,device=hidden.device)
    own_sum=matched_sum=None; positions=int(hidden.shape[1]); vocab=None
    if positions<1: raise ValueError('mass measurement requires at least one evaluation position')
    for pos in range(positions):
        probs=torch.softmax(head(hidden[:,pos,:]).float(),dim=-1)
        vocab=int(probs.shape[-1])
        own=probs.gather(1,own_ids).sum(dim=-1)
        matched=probs.gather(1,matched_ids).sum(dim=-1)
        own_sum=own if own_sum is None else own_sum+own
        matched_sum=matched if matched_sum is None else matched_sum+matched
    return own_sum/positions,matched_sum/positions,vocab-own_ids.shape[-1]

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
    rowpath=out/'rows.jsonl'; answerpath=out/'answers.jsonl'; plainpath=out/'plain.json'
    rows_done={row_key(x) for x in read_jsonl(rowpath)}; answers=read_jsonl(answerpath); answer_done={answer_key(x) for x in answers}
    if loader is None:
        import neuron_report as nr
        tok,model=nr.load_model(str(args.model),common.REVISION,args.device)
        model.requires_grad_(False); model.eval()
    else: tok,model=loader(args.device)
    layers=common.find_layers(model); widths={l:int(layers[l].mlp.down_proj.weight.shape[1]) for l in LAYERS}
    medians=json.loads((ROOT/'results/part112/medians.json').read_text(encoding='utf-8'))
    dirs={}; norms=[]; cache={}
    for l in LAYERS:
        w=layers[l].mlp.down_proj.weight; med=float(medians.get(str(l),medians.get(l)))
        for s in range(SLOTS):
            d,m=build_directions(w,l,s,med,cache)
            dirs[(l,s)]=d
            for a in ARMS: norms.append({'layer':l,'slot':s,'arm':a,**m[a]})
    _write_json(out/'norms.json',{'part':167,'dose':0.8,'directions':norms})
    questions=dict(phrasing_items()); plain=json.loads(plainpath.read_text(encoding='utf-8')) if plainpath.exists() else {'answers':[],'alignment_matches':0,'alignment_total':0,'determinism_matches':0,'determinism_total':0}
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
        matches=len(argmax)-len(mismatches)
        p.update({'alignment_matches':matches,'alignment_total':len(ids),'deterministic':deterministic,'alignment_mismatches':mismatches})
        p['_seq']=p['prompt_ids']+ids
    plain['answers']=[plainmap[i] for i in sorted(plainmap)]
    plain['alignment_matches']=sum(x['alignment_matches'] for x in plain['answers']); plain['alignment_total']=sum(x['alignment_total'] for x in plain['answers'])
    plain['determinism_matches']=sum(int(x['deterministic']) for x in plain['answers']); plain['determinism_total']=len(plain['answers'])
    _write_json(plainpath,{k:v for k,v in plain.items() if k!='answers'} | {'answers':[{k:v for k,v in x.items() if k!='_seq'} for x in plain['answers']]})
    token_ids=set()
    rowbase=row_base(widths)
    for qi,item in questions.items():
        p=plainmap[qi]; plen=len(p['prompt_ids']); ans=p['answer_ids']; positions=run166.evaluation_positions(plen,len(ans)); inp=torch.tensor(p['prompt_ids']+ans,dtype=torch.long,device=args.device).unsqueeze(0)
        head=model.get_output_embeddings(); plogs=run166.plain_position_logits(head,run166.forward_hidden(model,inp,positions)) if positions else []
        shuffled=list(rowbase); random.Random(167+qi).shuffle(shuffled)
        with rowpath.open('a',encoding='utf-8',newline='\n') as fh:
            for start in range(0,len(shuffled),args.batch_size):
                batch=shuffled[start:start+args.batch_size]; missing=[r for r in batch if row_key({**r,'question_index':qi}) not in rows_done]
                if not missing: continue
                push=torch.stack([dirs[(r['layer'],r['slot'])][r['arm']] for r in missing]).to(args.device,dtype=torch.float32); hooks=[]
                for ln in sorted({r['layer'] for r in missing}):
                    dense=torch.zeros_like(push)
                    for i,r in enumerate(missing):
                        if r['layer']==ln: dense[i]=push[i]
                    hooks.append(run166.attach_position_push(layers[ln].mlp.down_proj,dense,plen,len(ans)))
                try: hidden=run166.forward_hidden(model,inp.expand(len(missing),-1).contiguous(),positions)
                finally:
                    for h in hooks: h.remove()
                bias,norm,kl,kl2,_,_=run166.position_metrics(hidden,head,plogs,common.eos_ids(tok,model))
                cpu=bias.detach().float().cpu(); std=cpu.std(dim=-1,unbiased=False); sharp=(cpu.max(-1).values-cpu.mean(-1))/std.clamp_min(torch.finfo(torch.float32).tiny); bn=torch.linalg.vector_norm(cpu,dim=-1)
                topv,topid=torch.topk(cpu,k=min(200,cpu.shape[-1]),dim=-1)
                for i,r in enumerate(missing):
                    rec=make_row_record(r,qi,cpu[i],norm[i].detach().cpu(),topid[i].tolist(),topv[i].tolist())
                    ordered=ranked_ids(rec,tok); rec['own_ban_ids']=ordered[:50]; rec['rank_51_200_ids']=ordered[50:200]
                    fh.write(json.dumps(rec,ensure_ascii=True)+'\n'); rows_done.add(row_key(rec)); token_ids.update(rec['top_ids'])
                fh.flush()
    finalrows=read_jsonl(rowpath)
    # Stage 1b is completed for every phrasing before any generation starts.
    masspath=out/'masses.jsonl'; mass_done={(x['arm'],int(x['layer']),int(x['slot']),int(x['question_index'])) for x in read_jsonl(masspath)}
    rmap={(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index'])):r for r in finalrows}
    for qi,item in questions.items():
        p=plainmap[qi]; plen=len(p['prompt_ids']); ans=p['answer_ids']; positions=run166.evaluation_positions(plen,len(ans))
        if not positions: continue
        inp=torch.tensor(p['prompt_ids']+ans,dtype=torch.long,device=args.device).unsqueeze(0)
        for start in range(0,len(rowbase),args.batch_size):
            batch=[{**r,'question_index':qi} for r in rowbase[start:start+args.batch_size]]
            missing=[r for r in batch if (r['arm'],r['layer'],r['slot'],qi) not in mass_done]
            if not missing: continue
            pushes=torch.stack([dirs[(r['layer'],r['slot'])][r['arm']] for r in missing]).to(args.device,dtype=torch.float32)
            hooks=[]
            for ln in sorted({r['layer'] for r in missing}):
                dense=torch.zeros_like(pushes)
                for i,r in enumerate(missing):
                    if r['layer']==ln:dense[i]=pushes[i]
                hooks.append(run166.attach_position_push(layers[ln].mlp.down_proj,dense,plen,len(ans)))
            try: hidden=run166.forward_hidden(model,inp.expand(len(missing),-1).contiguous(),positions)
            finally:
                for h in hooks:h.remove()
            own=[]; matched=[]
            for r in missing:
                own.append(rmap[(r['arm'],r['layer'],r['slot'],qi)]['own_ban_ids'])
                donor=rmap[(r['arm'],r['layer'],(r['slot']+1)%SLOTS,qi)]
                matched.append(donor['own_ban_ids'])
            own_mass,matched_mass,allowed=list_masses(hidden,head,own,matched)
            with masspath.open('a',encoding='utf-8',newline='\n') as mf:
                for i,r in enumerate(missing):
                    overlap=len(set(own[i])&set(matched[i]))
                    mf.write(json.dumps({'arm':r['arm'],'layer':r['layer'],'slot':r['slot'],'question_index':qi,'own_mass':float(own_mass[i].detach().cpu()),'matched_mass':float(matched_mass[i].detach().cpu()),'overlap':overlap,'allowed_vocab':allowed},ensure_ascii=True)+'\n')
                    mass_done.add((r['arm'],r['layer'],r['slot'],qi))
    for old in finalrows: token_ids.update(map(int,old.get('top_ids',[])))
    for old in answers: token_ids.update(map(int,old.get('token_ids',[])))
    rowmap={(r['arm'],int(r['layer']),int(r['slot']),int(r['question_index'])):r for r in finalrows}
    # Battery order is shuffled per phrasing, mixing layer and ban within each batch.
    pilot_start=None; pilot_batches=pilot_answers=0
    with answerpath.open('a',encoding='utf-8',newline='\n') as af:
        for qi,item in questions.items():
            work=[]
            for r in rowbase:
                r={**r,'question_index':qi}
                for ban in BANS: work.append({**r,'ban':ban})
            random.Random(167+qi).shuffle(work)
            lists=ban_lists(work,finalrows,tok)
            for start in range(0,len(work),64):
                chunk=work[start:start+64]; selected=[(r,lists[start+i]) for i,r in enumerate(chunk) if answer_key({'stage':'battery',**r}) not in answer_done]
                if not selected: continue
                chunk=[x[0] for x in selected]; bl=[x[1][x[0]['ban']] for x in selected]
                if pilot_start is None: pilot_start=time.perf_counter()
                generated=generate_batch(model,tok,layers,item['question'],chunk,dirs,bl,args.device,common,80,167)
                for r,a,banned in zip(chunk,generated,bl):
                    rec={'stage':'battery','arm':r['arm'],'layer':r['layer'],'slot':r['slot'],'question_index':qi,'ban':r['ban'],'banned_ids':banned,**a}
                    af.write(json.dumps(rec,ensure_ascii=True)+'\n'); answer_done.add(answer_key(rec)); token_ids.update(a['token_ids'])
                af.flush(); pilot_answers+=len(chunk); pilot_batches+=1
                if pilot_batches==3:
                    rate=pilot_answers/max(time.perf_counter()-pilot_start,1e-9); print(json.dumps({'pilot_answers_per_second':rate}),flush=True)
                    if pilot_rate_should_stop(pilot_answers,pilot_start,time.perf_counter()): raise SystemExit(f'pilot stop: {rate:.3f} answers/s below 1.5')
        for qi,item in questions.items():
            key=('no_push','plain',None,None,None,qi)
            if key in answer_done: continue
            prompt,ids=generate_plain(model,tok,item['question'],common,args.device,80)
            text=tok.decode(ids,skip_special_tokens=True).strip()
            rec={'stage':'no_push','arm':'plain','question_index':qi,'answer':text,'token_ids':ids,'token_count':len(ids)}
            af.write(json.dumps(rec,ensure_ascii=True)+'\n'); answer_done.add(answer_key(rec)); token_ids.update(ids)
    decoded={str(t):tok.decode([t],skip_special_tokens=False) for t in sorted(token_ids)}
    _write_json(out/'tokens.json',decoded)
    return {'rows':len(rows_done),'answers':len(answer_done),'phrasing_count':len(questions)}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--device',default='xpu'); p.add_argument('--model',default=run165.MODEL_DEFAULT); p.add_argument('--out-dir',default='results/part167'); p.add_argument('--batch-size',type=int,default=64)
    a=p.parse_args(argv)
    if a.batch_size!=64:p.error('registered batch size is 64')
    print(json.dumps(run(a),ensure_ascii=True),flush=True)
if __name__=='__main__': main()
