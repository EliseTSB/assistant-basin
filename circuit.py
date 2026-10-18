#!/usr/bin/env python3
"""Part 19 identity circuit measurements for Qwen3.5."""
from __future__ import annotations
import argparse, hashlib, json, os, re, time
import numpy as np
import torch
import neuron_report as nr

IDENTITY = re.compile(r'qwen|tongyi|alibaba|通义|千问|阿里', re.I)
ID_PROMPTS = ['Who made you?', 'What is your name?', 'Who created you?', 'Which company trained you?', 'Introduce yourself.', 'What model are you?']
CTRL_PROMPTS = ['What is the capital of France?', 'Describe a quiet beach.', 'Explain what a prime number is.']
REV = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'

class ZeroHook:
    def __init__(self, layer, unit):
        self.unit = int(unit); self.handle = layer.mlp.down_proj.register_forward_pre_hook(self.hook)
    def hook(self, mod, args):
        x=args[0].clone(); x[..., self.unit]=0; return (x,)+tuple(args[1:])
    def remove(self): self.handle.remove()

class MultiZero:
    def __init__(self, layers, pairs):
        self.hooks=[]
        by={}
        for l,u in pairs: by.setdefault(int(l), []).append(int(u))
        for l,units in by.items():
            ids=torch.tensor(sorted(set(units)))
            def hook(mod,args,ids=ids):
                x=args[0].clone(); x[...,ids.to(x.device)]=0; return (x,)+tuple(args[1:])
            self.hooks.append(layers[l].mlp.down_proj.register_forward_pre_hook(hook))
    def remove(self):
        for h in self.hooks: h.remove()

def parse():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest='cmd',required=True)
    for cmd in ('screen','verify','group','edges','describe'):
        q=s.add_parser(cmd); q.add_argument('--model',default='Qwen/Qwen3.5-4B'); q.add_argument('--revision',default=REV); q.add_argument('--out',required=True); q.add_argument('--device',default='xpu'); q.add_argument('--spill-limit','--max-gpu-gb',type=float,default=14.5); q.add_argument('--seed',type=int,default=0)
        if cmd=='describe': q.add_argument('--corpus',required=True); q.add_argument('--docs',type=int,default=3000); q.add_argument('--max-len',type=int,default=256); q.add_argument('--batch',type=int,default=8)
    return p

def dump(x,path): nr._dump(x,path)
def prep(a):
    os.makedirs(a.out,exist_ok=True); tok,model=nr.load_model(a.model,a.revision,a.device); lname,layers,norm=nr.find_text_layers(model); nr.spill_guard(a.device,a.spill_limit)
    # Weights never need gradients here; left on, screen's backward would allocate a bf16 grad for every one of the ~4.6B
    # parameters (~9 GB on top of the model), past the A770's 16 GB, where Windows spills to shared memory (gotcha, 10-16).
    model.requires_grad_(False)
    return tok,model,lname,layers,norm

def header(a,lname,layers):
    return {'model':a.model,'revision':a.revision,'tool_sha256':hashlib.sha256(open(__file__,'rb').read()).hexdigest(),'prompts':{'identity':ID_PROMPTS,'control':CTRL_PROMPTS},'seeds':{'random_controls':a.seed,'group':[1,2,3]},'layers_module':lname,'n_layers':len(layers),'input_shas':{name:nr._file_sha(os.path.join(a.out,name)) for name in ('answers.json','screen.json','verify.json') if os.path.isfile(os.path.join(a.out,name))},'written':time.strftime('%Y-%m-%d %H:%M:%S')}
def ids_prompt(tok,q): return tok(nr.build_prompt(tok,q,False),return_tensors='pt',add_special_tokens=False)['input_ids']
def generate_answers(tok,model,a):
    eos=tok.eos_token_id
    ans={}
    for q in ID_PROMPTS+CTRL_PROMPTS:
        x=ids_prompt(tok,q).to(a.device)
        with torch.no_grad(): y=model.generate(input_ids=x,max_new_tokens=60,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=eos)
        ans[q]={'prompt_ids':x[0].cpu().tolist(),'answer_ids':y[0,x.shape[1]:].cpu().tolist(),'answer':tok.decode(y[0,x.shape[1]:],skip_special_tokens=True)}
    dump({'__meta__':answer_meta(a),'answers':ans},os.path.join(a.out,'answers.json')); return ans

def answer_meta(a):
    return {'model':a.model,'revision':a.revision,'prompts':{'identity':ID_PROMPTS,'control':CTRL_PROMPTS},'tool_sha256':hashlib.sha256(open(__file__,'rb').read()).hexdigest()}

def answers(a,tok,model):
    path=os.path.join(a.out,'answers.json')
    if os.path.exists(path):
        cached=json.load(open(path))
        if cached.get('__meta__')==answer_meta(a) and isinstance(cached.get('answers'),dict): return cached['answers']
        print('answers.json provenance mismatch; regenerating answers',flush=True)
    return generate_answers(tok,model,a)
def ident_idxs(tok,ids):
    """Answer tokens overlapping a regex match in the decoded answer. Token by token would miss the names: this
    tokenizer splits ' Qwen' into ' Q'+'wen' and 'Tongyi' into 'T'+'ong'+'yi', leaving only ' Alibaba' (fixed 2026-10-18)."""
    ids=[int(t) for t in ids]; bounds=[]; prev=''
    for i in range(len(ids)):
        cur=tok.decode(ids[:i+1]); bounds.append((len(prev),len(cur))); prev=cur
    spans=[(m.start(),m.end()) for m in IDENTITY.finditer(prev)]
    return [i for i,(lo,hi) in enumerate(bounds) if hi>lo and any(lo<e and st<hi for st,e in spans)]
def score(tok,model,q,rec,units=None,grad=False):
    p=torch.tensor(rec['prompt_ids'],device=next(model.parameters()).device); ans=rec['answer_ids']; allids=torch.tensor([rec['prompt_ids']+ans],device=p.device)
    with torch.enable_grad():
      if grad:
        emb=model.get_input_embeddings()(allids).detach().requires_grad_(True); out=model(inputs_embeds=emb,attention_mask=torch.ones_like(allids),use_cache=False)
      else: out=model(input_ids=allids,attention_mask=torch.ones_like(allids),use_cache=False)
    start=len(rec['prompt_ids']); ix=ident_idxs(tok,ans) if q in ID_PROMPTS else list(range(min(20,len(ans))))
    if not ix: return None
    pos=torch.tensor([start+i-1 for i in ix],device=p.device); targets=torch.tensor([ans[i] for i in ix],device=p.device)
    with torch.enable_grad(): vals=torch.log_softmax(out.logits[0,pos].float(),-1).gather(1,targets[:,None]).sum()
    return vals, ix, out, allids

def screen(a):
    tok,model,lname,layers,_=prep(a); ans=answers(a,tok,model); n=layers[0].mlp.down_proj.in_features
    zero=np.zeros((len(layers),n),np.float32); push=np.zeros_like(zero); acts=[]; hooks=[]
    for layer in layers:
        box={}
        def hook(mod,args,box=box):
            x=args[0]; x.retain_grad(); box['x']=x; return None
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(hook)); acts.append(box)
    kept=[]
    for q in ID_PROMPTS:
        rec=ans[q]; sc=score(tok,model,q,rec,grad=True)
        if sc is None: kept.append({'prompt':q,'reason':'greedy answer contains no identity token'}); continue
        val,ix,out,allids=sc; model.zero_grad(set_to_none=True); val.backward()
        for li,box in enumerate(acts):
            x=box['x'].detach().float(); g=box['x'].grad.detach().float(); zero[li]+=((-x*g).sum(dim=(0,1))).abs().cpu().numpy(); 
            if q=='Who made you?': push[li]=g.sum(dim=(0,1)).cpu().numpy()
        kept.append({'prompt':q,'identity_indices':ix,'identity_tokens':[tok.decode([rec['answer_ids'][i]]) for i in ix]})
    for h in hooks:h.remove()
    valid=sum(not x.get('reason') for x in kept); zero/=max(valid,1)
    if valid==0: raise SystemExit('screen: no identity prompt produced a valid identity token; refusing to write rankings')
    if not any(x['prompt']=='Who made you?' and not x.get('reason') for x in kept): raise SystemExit('screen: "Who made you?" has no valid identity token; refusing to write push ranking')
    pairs=sorted(((float(zero[l,u]),l,u) for l in range(len(layers)) for u in range(n)),reverse=True)
    pp=sorted(((float(push[l,u]),l,u) for l in range(len(layers)) for u in range(n)),reverse=True)
    obj=header(a,lname,layers); obj.update({'identity_answers':kept,'valid_identity_prompts':valid,'zeroing_estimate':zero.tolist(),'push_estimate':push.tolist(),'top_zeroing':[{'score':v,'layer':l,'index':u} for v,l,u in pairs[:1000]],'top_push':[{'score':v,'layer':l,'index':u} for v,l,u in pp[:1000]],'N2742':{'layer':22,'index':2742,'zeroing_rank':next((i+1 for i,(_,l,u) in enumerate(pairs) if (l,u)==(22,2742)),None),'push_rank':next((i+1 for i,(_,l,u) in enumerate(pp) if (l,u)==(22,2742)),None)}})
    dump(obj,os.path.join(a.out,'screen.json'))

def candidates(a):
    s=json.load(open(os.path.join(a.out,'screen.json'))); return [(int(x['layer']),int(x['index'])) for x in s['top_zeroing'][:50]]
def matched_random(top,shape,seed,exclude=None):
    rng=np.random.default_rng(seed); exclude=set(top if exclude is None else exclude); layers=[]
    for l,u in top: layers.append(l)
    out=[]; used=set(exclude)
    for l in layers:
        pool=[u for u in range(shape[1]) if (l,u) not in used]
        u=int(rng.choice(pool)); used.add((l,u)); out.append((l,u))
    return out

def normalized_edge_effect(altered, baseline, i, target):
    return (altered[i][target][0]-baseline[i][target][0])/(baseline[i][target][1] if baseline[i][target][1] else 1e-12)

def null_99th(values):
    return float(np.quantile(values,.99)) if values else 0.

def valid_prompt_mean(values, valid):
    kept=[value for value,ok in zip(values,valid) if ok]
    return float(np.mean(kept)) if kept else 0.

def eval_delta(tok,model,ans,pair, q, base, identity=True):
    h=ZeroHook(model._circuit_layers[pair[0]],pair[1])
    try: z=score(tok,model,q,ans[q]);
    finally: h.remove()
    if z is None:return None
    v,ix,_,_=z; d=float(v.detach().cpu())-base
    return {'delta':d,'token_count':len(ix),'per_token':d/len(ix)}

def verify(a):
    tok,model,lname,layers,_=prep(a); model._circuit_layers=layers
    ans=answers(a,tok,model); top=candidates(a); shape=(len(layers),layers[0].mlp.down_proj.in_features); rnd=matched_random(top,shape,a.seed)
    bases={}
    for q in ID_PROMPTS+CTRL_PROMPTS:
        x=score(tok,model,q,ans[q]); bases[q]=float(x[0].detach()) if x is not None else None
    records=[]
    for typ,pairs in [('top',top),('random',rnd)]:
      for l,u in pairs:
        im={q:eval_delta(tok,model,ans,(l,u),q,bases[q]) for q in ID_PROMPTS if bases[q] is not None}
        cm={q:eval_delta(tok,model,ans,(l,u),q,bases[q],False) for q in CTRL_PROMPTS if bases[q] is not None}
        mabs=float(np.mean([abs(v['delta']) for v in im.values()])) if im else 0
        itok=sum(v['token_count'] for v in im.values()); ctok=sum(v['token_count'] for v in cm.values())
        idmean=sum(abs(v['delta']) for v in im.values())/itok if itok else 0
        cmean=sum(abs(v['delta']) for v in cm.values())/ctok if ctok else 0
        records.append({'set':typ,'layer':l,'index':u,'identity':im,'control':cm,'pooled_abs_delta_M':mabs,'identity_abs_per_token':idmean,'control_abs_per_token':cmean,'ratio':idmean/cmean if cmean else None})
    obj=header(a,lname,layers); obj.update({'random_seed':a.seed,'top_neurons':[{'layer':l,'index':u} for l,u in top],'random_neurons':[{'layer':l,'index':u} for l,u in rnd],'results':records})
    dump(obj,os.path.join(a.out,'verify.json'))

def group(a):
    tok,model,lname,layers,_=prep(a); ans=answers(a,tok,model); top=candidates(a); shape=(len(layers),layers[0].mlp.down_proj.in_features)
    sets=[('top',top)]+[(f'random_{s}',matched_random(top,shape,a.seed+s)) for s in (1,2,3)]
    eos=tok.eos_token_id; results=[]
    for name,pairs in sets:
        h=MultiZero(layers,pairs); vals={}
        try:
            for q in ID_PROMPTS:
                x=ids_prompt(tok,q).to(a.device)
                with torch.no_grad(): y=model.generate(input_ids=x,max_new_tokens=60,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=eos)
                text=tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip(); first=re.split(r'(?<=[.!?])\s+',text,1)[0]
                vals[q]={'answer':text,'first_sentence':first,'names_identity':bool(IDENTITY.search(first))}
        finally:h.remove()
        results.append({'set':name,'neurons':[{'layer':l,'index':u} for l,u in pairs],'answers':vals,'count_names_identity':sum(x['names_identity'] for x in vals.values())})
    obj=header(a,lname,layers);obj['sets']=results;dump(obj,os.path.join(a.out,'group.json'))

def checked_verify(a):
    path=os.path.join(a.out,'verify.json'); v=json.load(open(path))
    current={name:nr._file_sha(os.path.join(a.out,name)) for name in ('answers.json','screen.json')}
    recorded=v.get('input_shas',{})
    if any(recorded.get(name)!=sha for name,sha in current.items()):
        raise SystemExit('verify.json input SHA mismatch; rerun verify before edges or describe')
    return v

def edges(a):
    tok,model,lname,layers,_=prep(a); ans=answers(a,tok,model)
    v=checked_verify(a); ranked=sorted([x for x in v['results'] if x['set']=='top'],key=lambda x:-x['pooled_abs_delta_M'])[:30]
    nodes=[(int(x['layer']),int(x['index'])) for x in ranked]; null=[(int(x['layer']),int(x['index'])) for x in v['random_neurons'][:30]]
    # Capture all node activations at the answer-token positions using one pre-hook per layer.
    # Baseline captures without a source zero; per source deltas are total effects.
    def capture(source=None):
      agg=[]
      for q in ID_PROMPTS:
        rec=ans[q]; ix=ident_idxs(tok,rec['answer_ids']); positions=sorted(set([len(rec['prompt_ids'])+i for i in ix]+[max(0,len(rec['prompt_ids'])+i-1) for i in ix]))
        h=ZeroHook(layers[source[0]],source[1]) if source else None; caps={}; hs=[]
        for l,u in nodes:
          if (l,u) in caps:continue
          def cap(mod,args,l=l,u=u):caps[(l,u)]=args[0][0,:,u].detach().float().cpu()
          hs.append(layers[l].mlp.down_proj.register_forward_pre_hook(cap))
        try:score(tok,model,q,rec)
        finally:
          if h:h.remove()
          for hh in hs:hh.remove()
        agg.append({'valid':bool(positions), **{node:(float(caps[node][positions].mean()),float(caps[node][positions].abs().mean())) if positions and node in caps else (0.,0.) for node in nodes}})
      return agg
    baseline=capture(); effects=[]
    for src in nodes+null:
        altered=capture(src); deltas={}
        for target in nodes:
            vals=[normalized_edge_effect(altered,baseline,i,target) for i in range(len(ID_PROMPTS))]
            deltas[f'{target[0]}:{target[1]}']=valid_prompt_mean(vals,[x['valid'] for x in baseline])
        effects.append({'source':{'layer':src[0],'index':src[1]},'null':src in null,'effects':deltas})
    thresholds={}
    for t in nodes:
      vals=[abs(e['effects'][f'{t[0]}:{t[1]}']) for e in effects if e['null']]
      thresholds[f'{t[0]}:{t[1]}']=null_99th(vals)
    obj=header(a,lname,layers);obj.update({'nodes':[{'layer':l,'index':u} for l,u in nodes],'effects':effects,'null_99th_abs_by_target':thresholds,'significant_edges':[{'source':e['source'],'target':{'layer':int(k.split(':')[0]),'index':int(k.split(':')[1])},'effect':v} for e in effects if not e['null'] for k,v in e['effects'].items() if abs(v)>thresholds[k] and e['source']['layer']<int(k.split(':')[0])]})
    dump(obj,os.path.join(a.out,'edges.json'))

def describe(a):
    tok,model,lname,layers,norm=prep(a); ans=answers(a,tok,model)
    v=checked_verify(a); ranked=sorted([x for x in v['results'] if x['set']=='top'],key=lambda x:-x['pooled_abs_delta_M'])[:30]
    nodes=[(int(x['layer']),int(x['index'])) for x in ranked]
    U=model.get_output_embeddings().weight; vocab_n=len(tok); described={}
    for l,u in nodes:
        W=layers[l].mlp.down_proj.weight.float()[:,u]
        gain=norm(torch.ones(1,1,W.shape[0],device=W.device,dtype=next(norm.parameters()).dtype)).float().reshape(-1) if norm is not None else torch.ones_like(W)
        vec=gain*W; total=0.; total_sq=0.; count=0; best_values=torch.empty(0,device=vec.device); best_ids=torch.empty(0,dtype=torch.long,device=vec.device)
        for start in range(0,vocab_n,8192):
            end=min(start+8192,vocab_n); vals=U[start:end].float()@vec; total+=float(vals.sum()); total_sq+=float(vals.square().sum()); count+=len(vals)
            ids=torch.arange(start,end,device=vals.device); cand=torch.cat((best_values,vals)); cand_ids=torch.cat((best_ids,ids)); k=min(10,len(cand)); ix=torch.topk(cand,k).indices; best_values=cand[ix]; best_ids=cand_ids[ix]
        mean=total/count; std=max(((total_sq-total*total/count)/(count-1))**.5,1e-12) if count>1 else 1e-12
        hi=[(int(i),float((v-mean)/std)) for i,v in zip(best_ids.cpu(),best_values.cpu())]
        described[f'{l}:{u}']={'layer':l,'index':u,'logit_lens_top10':[{'token':tok.decode([i]),'z':round(z,3)} for i,z in hi]}
    docs=[]
    with open(a.corpus,encoding='utf-8') as f:
      for line in f:
        ids=tok(json.loads(line)['text'],add_special_tokens=False)['input_ids'][:a.max_len]
        if len(ids)>=16:docs.append(ids)
        if len(docs)>=a.docs:break
    # Each captured node column is retained only for the current batch and reduced to top contexts.
    best={node:[] for node in nodes}
    for b in range(0,len(docs),a.batch):
      chunk=docs[b:b+a.batch]; width=max(map(len,chunk)); ids=torch.full((len(chunk),width),tok.pad_token_id,dtype=torch.long);mask=torch.zeros_like(ids)
      for j,d in enumerate(chunk):ids[j,:len(d)]=torch.tensor(d);mask[j,:len(d)]=1
      caps={}; hs=[]
      for l,u in nodes:
        def cap(mod,args,l=l,u=u):caps[(l,u)]=args[0].detach()[...,u].float().cpu()
        hs.append(layers[l].mlp.down_proj.register_forward_pre_hook(cap))
      try:
        with torch.no_grad():model.model(input_ids=ids.to(a.device),attention_mask=mask.to(a.device),use_cache=False)
      finally:
        for h in hs:h.remove()
      for node in nodes:
        arr=caps[node]
        for j,d in enumerate(chunk):
          for pos in range(len(d)):
            item=(float(arr[j,pos]),b+j,pos)
            existing=next((i for i,row in enumerate(best[node]) if row[1]==item[1]),None)
            if existing is not None:
                if item[0]>best[node][existing][0]: best[node][existing]=item
            elif len(best[node])<10: best[node].append(item)
            elif item[0]>min(row[0] for row in best[node]):
                smallest=min(range(len(best[node])),key=lambda i:best[node][i][0]);best[node][smallest]=item
      nr.spill_guard(a.device,a.spill_limit)
    for node in nodes:
      key=f'{node[0]}:{node[1]}'; out=[]
      for val,doc,pos in sorted(best[node],key=lambda x:x[0],reverse=True):
        d=docs[doc];out.append({'activation':val,'doc':doc,'pos':pos,'text':nr.context(d,pos,tok)})
      described[key]['top_contexts']=out
    obj=header(a,lname,layers);obj.update({'corpus':a.corpus,'corpus_sha256':nr._file_sha(a.corpus),'docs':len(docs),'nodes':described})
    dump(obj,os.path.join(a.out,'describe.json'))

def main():
    a=parse().parse_args()
    if a.cmd=='screen':screen(a)
    elif a.cmd=='verify':verify(a)
    elif a.cmd=='group':group(a)
    elif a.cmd=='edges':edges(a)
    else:describe(a)
if __name__=='__main__':main()
