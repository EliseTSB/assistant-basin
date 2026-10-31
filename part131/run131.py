#!/usr/bin/env python3
"""Run one Part 131 model/frame stage; JSONL output is append-only and resumable."""
from __future__ import annotations
import argparse, json, time
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
    text = tok.apply_chat_template([{"role":"user", "content":question}], add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]

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
    done=set()
    if answers.exists():
        for line in answers.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r=json.loads(line); done.add((r['unit_id'],r['arm'],r['question_index']))
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
                L,n=cell['L'],cell['n']; col=layers[L].mlp.down_proj.weight[:,n].float()
                vec=(cell['s']*cell['alpha']*r_l[L]/float(col.norm())*col).to(model.dtype)
                vectors.setdefault(L,torch.zeros(len(batch),1,model.config.hidden_size,device=a.device,dtype=model.dtype))[i,0]=vec
            handles=[]
            for L,v in vectors.items():
                def hook(_m,_args,o,v=v): return (o[0]+v,)+tuple(o[1:]) if isinstance(o,tuple) else o+v
                handles.append(layers[L].register_forward_hook(hook))
            try:
                with torch.inference_mode(): generated=model.generate(input_ids=inp,attention_mask=mask,do_sample=False,max_new_tokens=MAX_NEW,pad_token_id=pad)
            finally:
                for h in handles: h.remove()
            for (cell,qi,q),tokens in zip(batch,generated[:,width:]):
                token_ids=tokens.tolist()
                if tokenizer.eos_token_id in token_ids: token_ids=token_ids[:token_ids.index(tokenizer.eos_token_id)+1]
                answer=tokenizer.decode(token_ids,skip_special_tokens=True).strip()
                row={'id':cell['unit_id'],'unit_id':cell['unit_id'],'unit_class':cell['unit_class'],'arm':cell.get('arm','unpushed'),
                     'tag':a.tag,'repo':repo,'revision':a.revision,'frame':a.frame,'L':cell['L'],'n':cell['n'],'s':cell['s'],'alpha':cell['alpha'],
                     'question_index':qi,'question':q,'answer':answer,'tokens':token_ids,'target_persona':cell['persona'],'target_desc':cell['desc']}
                out.write(json.dumps(row,ensure_ascii=False)+'\n'); out.flush(); count+=1
            print(f'{a.tag} batch {b0//a.batch+1}/{(len(work)+a.batch-1)//a.batch}',flush=True)
    # Exact no-op-hook parity for Gate 3 on one design question.
    gate_q=QUESTIONS[0]; ids=torch.tensor([prompt_ids(tokenizer,gate_q,a.frame)],device=a.device)
    with torch.inference_mode(): plain=model.generate(input_ids=ids,do_sample=False,max_new_tokens=MAX_NEW,pad_token_id=tokenizer.pad_token_id)[0]
    noop=layers[0].register_forward_hook(lambda _m,_a,o:o)
    try:
        with torch.inference_mode(): hooked=model.generate(input_ids=ids,do_sample=False,max_new_tokens=MAX_NEW,pad_token_id=tokenizer.pad_token_id)[0]
    finally: noop.remove()
    gate=bool(torch.equal(plain,hooked))
    meta={'tag':a.tag,'repo':repo,'revision':a.revision,'frame':a.frame,'stage':a.stage,'new_rows':count,'r_L':r_l,
          'gate3_noop_exact':gate,'torch':torch.__version__,'transformers':__import__('transformers').__version__,'seconds':time.time()-started}
    meta_path.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8'); print(json.dumps(meta),flush=True)
    if not gate: raise SystemExit('Gate 3 failed: no-op hook changed generated tokens')
if __name__=='__main__': main()
