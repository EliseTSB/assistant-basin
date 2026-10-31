#!/usr/bin/env python3
"""Run Part 140's paired answer-generation arms on Qwen3.5-4B."""
from __future__ import annotations
import argparse,json,sys,time
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
for candidate in (ROOT/'part86',ROOT/'part115b',ROOT/'part115',ROOT/'part114'):
    if str(candidate) not in sys.path: sys.path.insert(0,str(candidate))
import common86 as c
ARMS=('A','B','C','D','E','F'); QUESTIONS=('Who are you?','Tell me about yourself.')
def paired_seeds(unit_id=None,arm='B'):
    """Arm B draws three shared seeds; every greedy arm uses one."""
    return (140000,140001,140002) if arm=='B' else (140000,)
def rowkey(r): return (r['arm'],r['set'],r['unit'],r['question'],int(r['seed']))
def read_rows(path):
    if not path.exists(): return []
    with path.open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]
class DeliveryHook:
    """Per-row generated decode push, independently released after N decode inputs."""
    def __init__(self,module,pushes,release_after=None,norm_preserve=False,add_mask=None):
        self.pushes=pushes.detach(); self.release_after=release_after; self.norm_preserve=norm_preserve; self.add_mask=add_mask; self.steps=[0]*int(pushes.shape[0]); self.prefill=True; self.handle=module.register_forward_pre_hook(self.hook)
    def hook(self,module,args):
        import torch
        x=args[0]
        if self.prefill: self.prefill=False; return None
        if x.ndim!=3 or x.shape[0]!=len(self.steps) or x.shape[-2]!=1: raise RuntimeError('unexpected decode hook shape')
        rows=[]
        for i in range(x.shape[0]):
            active=self.release_after is None or self.steps[i]<self.release_after
            enabled=self.add_mask is None or bool(self.add_mask[i])
            rows.append(self.pushes[i] if active and enabled else torch.zeros_like(self.pushes[i])); self.steps[i]+=1
        delta=torch.stack(rows).to(device=x.device,dtype=x.dtype).unsqueeze(1)
        if self.add_mask is None:
            pushed=x+delta
        else:
            pushed=x.clone()
            for i,enabled in enumerate(self.add_mask):
                if enabled: pushed[i]=x[i]+delta[i]
        if self.norm_preserve:
            scale=x.norm(dim=-1,keepdim=True)/pushed.norm(dim=-1,keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
            pushed=torch.where(delta.ne(0).any(dim=-1,keepdim=True),pushed*scale,pushed)
        return (pushed,)+tuple(args[1:])
    def remove(self): self.handle.remove()
def generate(model,tok,layer,question,pushes,device,seed,arm,release,max_tokens=80,norm_preserve=False,add_mask=None):
    import torch
    inputs=c.token_inputs(tok,question,int(pushes.shape[0]),device); plen=inputs['input_ids'].shape[-1]
    torch.manual_seed(int(seed))
    if hasattr(torch,'xpu') and torch.xpu.is_available(): torch.xpu.manual_seed_all(int(seed))
    hk=DeliveryHook(layer.mlp.down_proj,pushes,release,norm_preserve=norm_preserve,add_mask=add_mask)
    kwargs=dict(**inputs,max_new_tokens=max_tokens,do_sample=(arm=='B'),use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=c.eos_ids(tok,model))
    if arm=='B': kwargs.update(temperature=.7,top_p=.95)
    if arm=='E': kwargs['repetition_penalty']=1.15
    try:
        with torch.inference_mode(): seq=model.generate(**kwargs)
    finally: hk.remove()
    out=[]; eos=set(c.eos_ids(tok,model))
    for ids in seq[:,plen:]:
        vals=ids.tolist()
        for j,v in enumerate(vals):
            if v in eos: vals=vals[:j+1]; break
        out.append(tok.decode(vals,skip_special_tokens=True).strip())
    return out
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--units',default=str(HERE/'units140.json')); p.add_argument('--device',default='xpu'); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--pilot',action='store_true'); p.add_argument('--out',default='results/part140'); p.add_argument('--release',type=int,default=8); p.add_argument('--max-new-tokens',type=int,default=80); a=p.parse_args(argv)
    if a.batch_size<1 or a.release<1: p.error('batch-size and release must be positive')
    import torch
    if a.device=='xpu' and not (hasattr(torch,'xpu') and torch.xpu.is_available()): raise RuntimeError('XPU requested but unavailable')
    from howmany115b import load_direct
    tok,model=load_direct(a.device); layers=c.find_layers(model); model.eval(); model.requires_grad_(False)
    units=json.loads(Path(a.units).read_text(encoding='utf-8')); units=units[:20] if a.pilot else units
    outdir=Path(a.out); outdir.mkdir(parents=True,exist_ok=True); path=outdir/('pilot-answers.jsonl' if a.pilot else 'answers.jsonl'); timing=outdir/('pilot-timing.jsonl' if a.pilot else 'timing.jsonl'); existing_rows=read_rows(path); done_plain={rowkey(r) for r in existing_rows if r.get('plain')}; done_pushed={rowkey(r) for r in existing_rows if not r.get('plain')}
    by_layer=defaultdict(list)
    for u in units: by_layer[int(u['layer'])].append(u)
    medpath=ROOT/'results/part112/medians.json'; medians={int(k):float(v) for k,v in json.loads(medpath.read_text(encoding='utf-8')).items()}
    with path.open('a',encoding='utf-8') as stream,timing.open('a',encoding='utf-8') as tstream:
        for layer,layer_units in sorted(by_layer.items()):
            width=int(layers[layer].mlp.down_proj.in_features); weight=layers[layer].mlp.down_proj.weight; pushes={}
            random_rows=[u for u in layer_units if u['set']=='C']
            random_vecs=c.scaled_random_directions(weight,len(random_rows),140+layer,c.DOSE*medians[layer]) if random_rows else None
            for idx,u in enumerate(random_rows): pushes[u['id']]=random_vecs[idx].detach()
            for u in layer_units:
                if u['set']=='C': continue
                vec=torch.zeros(width,device=weight.device,dtype=weight.dtype); n=int(u['neuron']); vec[n]=c.neuron_delta(weight,n,medians[layer]); pushes[u['id']]=vec
            for arm in ARMS:
                for question in QUESTIONS:
                    for rep,seed in enumerate(paired_seeds(arm=arm)):
                        pending=[u for u in layer_units if rowkey({'arm':arm,'set':u['set'],'unit':u['id'],'question':question,'seed':seed}) not in done_pushed]
                        effective_batch=a.batch_size
                        for start in range(0,len(pending),effective_batch):
                            chunk=pending[start:start+effective_batch]
                            # The plain model does not depend on the unit, so one row per arm, question and seed is enough.
                            plain_key=rowkey({'arm':arm,'set':'plain','unit':'plain','question':question,'seed':seed})
                            elapsed=0.0
                            if plain_key not in done_plain:
                                zero=torch.zeros((1,width),device=weight.device,dtype=weight.dtype); started=time.perf_counter()
                                # Gate 1 compares a no-hook row and a zero-add-hook row in one greedy batch.
                                pairzero=torch.zeros((2,width),device=weight.device,dtype=weight.dtype)
                                gate_answers=generate(model,tok,layers[layer],question,pairzero,a.device,seed,'A',None,a.max_new_tokens,add_mask=[False,True])
                                gatepath=outdir/'gate1-result.json'
                                if gate_answers[0]!=gate_answers[1]:
                                    gatepath.write_text(json.dumps({'status':'failed','passed':False,'reason':'plain and zero-add rows diverged','layer':layer,'arm':arm,'question':question,'seed':seed})+'\n',encoding='utf-8')
                                    raise RuntimeError('Gate 1 failed: plain and zero-add hook outputs diverged')
                                if not gatepath.exists(): gatepath.write_text(json.dumps({'status':'passed','passed':True,'method':'same-batch plain and zero-add-hook rows'})+'\n',encoding='utf-8')
                                plain=generate(model,tok,layers[layer],question,zero,a.device,seed,arm,None,a.max_new_tokens)
                                elapsed=time.perf_counter()-started
                                row={'arm':arm,'set':'plain','unit':'plain','question':question,'seed':seed,'answer':plain[0],'plain':True,'layer':layer,'kind':'plain','batch_size':1}; stream.write(json.dumps(row,ensure_ascii=False)+'\n'); done_plain.add(plain_key)
                            vecs=torch.stack([pushes[u['id']] for u in chunk]); release={'C':a.release,'D':24}.get(arm); started=time.perf_counter()
                            answers=generate(model,tok,layers[layer],question,vecs,a.device,seed,arm,release,a.max_new_tokens,norm_preserve=(arm=='F')); elapsed+=time.perf_counter()-started
                            for u,answer in zip(chunk,answers):
                                row={'arm':arm,'set':u['set'],'unit':u['id'],'question':question,'seed':seed,'answer':answer,'plain':False,'layer':layer,'kind':u.get('set_kind'),'batch_size':len(chunk)}; stream.write(json.dumps(row,ensure_ascii=False)+'\n'); done_pushed.add(rowkey(row))
                            stream.flush(); tstream.write(json.dumps({'layer':layer,'arm':arm,'question':question,'seed':seed,'rows':len(chunk),'elapsed_s':elapsed})+'\n'); tstream.flush(); print(f'L{layer} arm={arm} q={question!r} seed={seed} rows={len(chunk)} elapsed={elapsed:.1f}s',flush=True)
    historical=read_rows(ROOT/'results/part112/answers.jsonl'); old={str(r.get('id')):r.get('answer') for r in historical if r.get('question_index')==0 or r.get('question')=='Who are you?'}
    chosen=sorted(u['id'] for u in units if u.get('set')=='P'); import random; random.Random(140).shuffle(chosen); chosen=chosen[:24]
    observed={r['unit']:r.get('answer','') for r in read_rows(path) if r.get('arm')=='A' and r.get('set')=='P' and r.get('question')=='Who are you?' and int(r.get('seed',-1))==140000 and not r.get('plain')}
    def first80(text): return tok.decode(tok.encode(text,add_special_tokens=False)[:80],skip_special_tokens=True)
    matches=sum(uid in old and uid in observed and first80(old[uid])==first80(observed[uid]) for uid in chosen)
    gate2={'n':len(chosen),'matched':matches,'rate':matches/len(chosen) if chosen else None,'passed':len(chosen)==24 and matches/24>=.8 if len(chosen)==24 else None,'status':'evaluated' if observed else 'pending generated answers'}
    (outdir/'gate2-result.json').write_text(json.dumps(gate2,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__': main()
