#!/usr/bin/env python3
"""Run one Part 131 model/frame stage; JSONL output is append-only and resumable."""
from __future__ import annotations
import argparse, json, time, random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QUESTIONS = ("Who are you?", "Tell me about yourself.", "What do you want most?",
             "Describe your perfect day.", "What is the capital of France?", "Write a short poem about the sea.")
PLAIN = "Question: {q}\nAnswer:"
MODELS = {"base": ("unsloth/Meta-Llama-3.1-8B", "e9a141a2", "plain"),
          "instruct": ("unsloth/Llama-3.1-8B-Instruct", "4699cc75", "chat"),
          "tulu-sft": ("allenai/Llama-3.1-Tulu-3-8B-SFT", "f2a0b46b", "chat"),
          "tulu-dpo": ("allenai/Llama-3.1-Tulu-3-8B-DPO", "a7beb67e", "chat"),
          "tulu-final": ("allenai/Llama-3.1-Tulu-3-8B", "66694379", "chat"),
          "hermes": ("NousResearch/Hermes-3-Llama-3.1-8B", "896ea440", "chat")}
MAX_NEW = 160

def prompt_ids(tok, question, frame):
    if frame == "plain":
        return tok(PLAIN.format(q=question), add_special_tokens=True)["input_ids"]
    encoded = tok.apply_chat_template([{"role":"user", "content":question}], add_generation_prompt=True,
                                      tokenize=True, return_dict=False)
    return encoded["input_ids"] if isinstance(encoded, dict) else encoded

def recover_jsonl(path):
    """Return complete rows and discard a torn final record before future appends."""
    raw=path.read_bytes()
    parts=raw.splitlines(keepends=True); rows=[]; valid_end=0
    last_record=max((i for i,line in enumerate(parts) if line.strip()),default=-1)
    for index,line in enumerate(parts):
        if not line.strip(): valid_end+=len(line); continue
        try: rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index!=last_record: raise
            raw=raw[:valid_end]; path.write_bytes(raw); break
        valid_end+=len(line)
    else:
        if raw and not raw.endswith(b'\n'):
            raw+=b'\n'; path.write_bytes(raw)
    return rows

def eos_ids(model, tokenizer):
    value=getattr(model.generation_config,'eos_token_id',None)
    values=value if isinstance(value,(list,tuple,set)) else [value]
    return {int(x) for x in values if x is not None} or ({int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set())

def gate3_noop_exact(model,layers,input_ids,attention_mask,pad,max_new=8):
    import torch
    with torch.inference_mode():
        plain=model.generate(input_ids=input_ids,attention_mask=attention_mask,do_sample=False,max_new_tokens=max_new,pad_token_id=pad)
    hook=layers[0].register_forward_hook(lambda _m,_a,o:o)
    try:
        with torch.inference_mode():
            hooked=model.generate(input_ids=input_ids,attention_mask=attention_mask,do_sample=False,max_new_tokens=max_new,pad_token_id=pad)
    finally: hook.remove()
    return bool(torch.equal(plain,hooked))

def generate_batch(model, layers, inp, mask, vectors, pad, eos, max_new):
    """Greedy decode while dropping completed rows from subsequent forward batches."""
    import torch
    active=list(range(inp.shape[0])); sequences=[[] for _ in active]
    current=inp; attention=mask; past=None
    for _ in range(max_new):
        kwargs={'input_ids':current,'attention_mask':attention,'use_cache':True}
        if past is not None: kwargs['past_key_values']=past
        positions=attention.long().cumsum(-1)-1
        positions.masked_fill_(attention==0,0)
        kwargs['position_ids']=positions[:,-current.shape[1]:]
        handles=[]
        for L,v in vectors.items():
            import torch
            active_v=v[active]
            def hook(_m,_args,o,v=active_v):
                # The push vector is applied at every position, including decode passes.
                return (o[0]+v,)+tuple(o[1:]) if isinstance(o,tuple) else o+v
            handles.append(layers[L].register_forward_hook(hook))
        try:
            with torch.inference_mode(): out=model(**kwargs)
        finally:
            for h in handles: h.remove()
        logits=out.logits[:,-1,:]
        nxt=logits.argmax(dim=-1)
        past=out.past_key_values
        keep=[]
        for j, original in enumerate(active):
            token=int(nxt[j]); sequences[original].append(token)
            if token not in eos: keep.append(j)
        if not keep or _+1>=max_new: break
        if len(keep)!=len(active):
            active=[active[j] for j in keep]
            past=_select_cache(past,keep)
            attention=attention[keep]
        current=nxt[keep].unsqueeze(1)
        attention=torch.cat([attention,torch.ones((len(active),1),device=attention.device,dtype=attention.dtype)],dim=1)
    return sequences

def _select_cache(cache, indices):
    # DynamicCache and legacy tuples are both supported by Transformers releases.
    if hasattr(cache,'batch_select_indices'):
        import torch
        cache.batch_select_indices(torch.tensor(indices,device=cache.layers[0].keys.device))
        return cache
    if isinstance(cache,(tuple,list)):
        return type(cache)(type(layer)(x[indices] for x in layer) for layer in cache)
    raise TypeError(f'unsupported cache type for active-row compaction: {type(cache).__name__}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', choices=MODELS, required=True); p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True); p.add_argument('--frame', choices=('plain','chat'), required=True)
    p.add_argument('--stage', choices=('pilot','main'), required=True); p.add_argument('--units', type=Path, default=Path('part131/units131.json'))
    p.add_argument('--out', type=Path, required=True); p.add_argument('--batch', type=int, default=32)
    p.add_argument('--device', default='cuda'); a=p.parse_args()
    repo, expected_rev, expected_frame=MODELS[a.tag]
    if a.revision != expected_rev or a.frame != expected_frame: p.error('tag revision/frame differs from registered design')
    data=json.loads(a.units.read_text(encoding='utf-8'))['units']; a.out.mkdir(parents=True,exist_ok=True)
    answers=a.out/f'{a.tag}_{a.frame}.jsonl'; meta_path=a.out/f'{a.tag}_{a.frame}.meta.json'
    answers.parent.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock_stream=answers.open('a+',encoding='utf-8')
    try: fcntl.flock(lock_stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit(f'another Part 131 runner holds {answers}')
    done=set()
    if answers.exists():
        saved=recover_jsonl(answers)
        for r in saved: done.add((r['unit_id'],r['arm'],r['question_index']))
    tokenizer=AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16).to(a.device).eval()
    layers=model.model.layers
    # The dose scale is measured from the unpushed question in this model and frame.
    base_ids=torch.tensor([prompt_ids(tokenizer,QUESTIONS[0],a.frame)],device=a.device)
    with torch.inference_mode(): hs=model(input_ids=base_ids,output_hidden_states=True).hidden_states
    r_l={L:float(hs[L+1][0,1:].float().norm(dim=-1).median()) for L in range(len(layers))}
    if a.stage=='pilot':
        selected=[u for u in data if u['class']=='shared'][:2]
    else: selected=data
    cells=[{'unit_id':'unpushed','unit_class':'unpushed','L':None,'n':None,'s':None,'alpha':None,'persona':'','desc':'','control_n':None}]
    for u in selected:
        cells += [{**u,'unit_id':u['id'],'unit_class':u['class'],'arm':'pole','n':u['n']},
                  {**u,'unit_id':u['id'],'unit_class':u['class'],'arm':'control','n':u['control']['n'],'L':u['L'],'s':u['s'],'alpha':u['alpha']}]
    work=[]
    for cell in cells:
        for qi,q in enumerate(QUESTIONS):
            arm=cell.get('arm','unpushed')
            if (cell['unit_id'],arm,qi) not in done: work.append((cell,qi,q))
    started=time.time(); count=0
    with answers.open('a',encoding='utf-8') as out:
        for b0 in range(0,len(work),a.batch):
            batch=work[b0:b0+a.batch]; seqs=[prompt_ids(tokenizer,q,a.frame) for _,_,q in batch]
            width=max(map(len,seqs)); pad=tokenizer.pad_token_id
            inp=torch.tensor([[pad]*(width-len(s))+s for s in seqs],device=a.device)
            mask=torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in seqs],device=a.device)
            vectors={}
            for i,(cell,_,_) in enumerate(batch):
                if cell['L'] is None: continue
                L,n=cell['L'],cell['n']; col=layers[L].mlp.down_proj.weight[:,n].detach().float()
                vec=(cell['s']*cell['alpha']*r_l[L]/float(col.norm())*col).to(model.dtype)
                vectors.setdefault(L,torch.zeros(len(batch),1,model.config.hidden_size,device=a.device,dtype=model.dtype))[i,0]=vec
            generated=generate_batch(model,layers,inp,mask,vectors,pad,eos_ids(model,tokenizer),MAX_NEW)
            for (cell,qi,q),token_ids in zip(batch,generated):
                raw=tokenizer.decode(token_ids,skip_special_tokens=False)
                decoded=tokenizer.decode(token_ids,skip_special_tokens=True).strip()
                answer=decoded.split('\nQuestion:')[0].strip() if a.frame=='plain' else decoded
                unique_id=f"unpushed-{a.tag}-{qi}" if cell['unit_id']=='unpushed' else cell['unit_id']
                row={'id':unique_id,'unit_id':cell['unit_id'],'unit_class':cell['unit_class'],'arm':cell.get('arm','unpushed'),
                   'tag':a.tag,'repo':repo,'revision':a.revision,'frame':a.frame,'L':cell['L'],'n':cell['n'],'s':cell['s'],'alpha':cell['alpha'],
                   'question_index':qi,'question':q,'answer':answer,'raw':raw,'tokens':token_ids,
                   'target_persona':cell['persona'],'target_desc':cell['desc']}
                out.write(json.dumps(row,ensure_ascii=False)+'\n'); out.flush(); count+=1
            print(f'{a.tag} batch {b0//a.batch+1}/{(len(work)+a.batch-1)//a.batch}',flush=True)
    # Gate 3: compare no-op and plain arms at the same real left-padded batch shape.
    gate_q=QUESTIONS[0]; plain_ids=prompt_ids(tokenizer,gate_q,a.frame)
    peer_ids=prompt_ids(tokenizer,QUESTIONS[1],a.frame); width=max(len(plain_ids),len(peer_ids))+1; pad=tokenizer.pad_token_id
    # Generate both arms at identical batch shape; this controls bf16 shape sensitivity.
    peers=[peer_ids,plain_ids]; batch_ids=torch.tensor([[pad]*(width-len(s))+s for s in peers],device=a.device)
    batch_mask=torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in peers],device=a.device)
    gate=gate3_noop_exact(model,layers,batch_ids,batch_mask,pad,MAX_NEW)
    rendered=tokenizer.decode(prompt_ids(tokenizer,QUESTIONS[0],a.frame),skip_special_tokens=False)
    meta={'tag':a.tag,'repo':repo,'revision':a.revision,'frame':a.frame,'stage':a.stage,'new_rows':count,'r_L':r_l,
          'question0_rendered_prompt':rendered,'gate3_batch_shape':'two left-padded rows; no-op and plain arms generated at the same batch shape',
          'gate3_noop_exact':gate,'torch':torch.__version__,'transformers':__import__('transformers').__version__,'seconds':time.time()-started}
    meta_path.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8'); print(json.dumps(meta),flush=True)
    if not gate: raise SystemExit('Gate 3 failed: no-op hook changed generated tokens')
    lock_stream.close()
if __name__=='__main__': main()
