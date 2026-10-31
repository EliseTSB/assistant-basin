#!/usr/bin/env python3
"""Run the registered Part 139 phase, position, question, confusion and checkpoint probes."""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
if str(ROOT/"part92") not in sys.path: sys.path.insert(0,str(ROOT/"part92"))
import common92 as c
import census139
import triage92

QUESTION="Who are you?"
PHASES=("full","prompt-only","answer-only","noop")

def _out_text(output): return output[0] if isinstance(output,(tuple,list)) else output

def _wrap(output, hidden):
    if isinstance(output,tuple): return (hidden,)+tuple(output[1:])
    if isinstance(output,list): return [hidden,*output[1:]]
    return hidden

def _prompt(tokenizer, question, frame):
    ids=c.prompt_ids(tokenizer,question,frame)
    if frame=="chat":
        text=tokenizer.apply_chat_template([{"role":"user","content":question}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
    else: text=c.PLAIN.format(q=question)
    encoded=tokenizer(text,add_special_tokens=(frame=="plain"),return_offsets_mapping=True)
    offsets=encoded["offset_mapping"]
    qstart=text.find(question); qend=qstart+len(question)
    mark_start=text.find("?",qstart,qend)
    groups=[]
    for pos,(a,b) in enumerate(offsets):
        if pos==0: label="position0"
        elif a < qend and b > qstart and mark_start>=0 and a<=mark_start<b: label="question_mark"
        elif a < qend and b > qstart: label="question_words"
        elif b<=qstart or (a==b and a<=qstart): label="user_role"
        else: label="assistant_role"
        groups.append(label)
    if groups: groups[-1]="last_position"
    # Some tokenizers normalize at the string boundary; retain exact tokenization from c.prompt_ids.
    if len(groups)!=len(ids): groups=(groups+["other"]*len(ids))[:len(ids)]
    return text,ids,groups

def _position_hook(layer,vectors,phase="full",group=None,groups=None):
    import torch
    state={"first":True}
    def hook(_module,_inputs,output):
        hidden=_out_text(output)
        prompt_call=state["first"]; state["first"]=False
        apply=(phase in ("full","noop") or (phase=="prompt-only" and prompt_call) or (phase=="answer-only" and not prompt_call))
        if not apply: return None
        if hidden.ndim!=3 or hidden.shape[0]!=vectors.shape[0] or hidden.shape[-1]!=vectors.shape[-1]:
            raise ValueError(f"unexpected layer output {tuple(hidden.shape)} for vectors {tuple(vectors.shape)}")
        mask=None
        if group is not None:
            if not prompt_call: return None
            if groups is None or len(groups)!=hidden.shape[0]: raise ValueError("position groups must match batch")
            seq_len=hidden.shape[1]; mask=torch.zeros((hidden.shape[0],seq_len),dtype=torch.bool,device=hidden.device)
            for row_no, labels in enumerate(groups):
                labels=list(labels)
                # Left padding is derived from the tokenized unpadded prompt length.
                offset=seq_len-len(labels)
                for j,label in enumerate(labels):
                    if label==group and 0<=offset+j<seq_len: mask[row_no,offset+j]=True
        vec=vectors.to(device=hidden.device,dtype=hidden.dtype).unsqueeze(1)
        pushed=hidden+vec if mask is None else hidden+vec*mask.unsqueeze(-1)
        return _wrap(output,pushed)
    return layer.register_forward_hook(hook)

def _generate(model,tokenizer,questions,frame,layer,vectors,device,max_new=32,phase="full",group=None,corrupt_ids=None,sample_seed=None,temperature=0.8,top_p=0.95):
    import torch
    if isinstance(questions,str): questions=[questions]
    prompts=[]; seqs=[]; labels=[]
    for i,q in enumerate(questions):
        text,ids,groups=_prompt(tokenizer,q,frame)
        if corrupt_ids is not None:
            ids=corrupt_ids[i]
        prompts.append(text); seqs.append(ids); labels.append(groups)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    pad=int(tokenizer.pad_token_id); width=max(map(len,seqs))
    input_ids=torch.tensor([[pad]*(width-len(s))+s for s in seqs],dtype=torch.long,device=device)
    attn=torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in seqs],dtype=torch.long,device=device)
    # Match group offsets to left-padded token sequences.
    padded_labels=[["pad"]*(width-len(g))+g for g in labels]
    handle=_position_hook(layer,vectors,phase,group,padded_labels) if vectors is not None else None
    try:
        with torch.inference_mode():
            if sample_seed is not None:
                torch.manual_seed(int(sample_seed))
                if str(device).startswith("xpu") and hasattr(torch,"xpu"): torch.xpu.manual_seed(int(sample_seed))
                generated=model.generate(input_ids=input_ids,attention_mask=attn,do_sample=True,temperature=float(temperature),top_p=float(top_p),max_new_tokens=max_new,pad_token_id=pad)
            else: generated=model.generate(input_ids=input_ids,attention_mask=attn,do_sample=False,max_new_tokens=max_new,pad_token_id=pad)
    finally:
        if handle is not None: handle.remove()
    answers=[]; first_ids=[]; answer_ids=[]
    for row in generated[:,width:]:
        all_ids=[int(x) for x in row.tolist()]
        eos=getattr(tokenizer,"eos_token_id",None)
        if eos is not None and int(eos) in all_ids: all_ids=all_ids[:all_ids.index(int(eos))+1]
        answer_ids.append(all_ids)
        first_ids.append(all_ids[:3])
        text=tokenizer.decode(row,skip_special_tokens=True)
        answers.append(text.split("\nQuestion:")[0].strip() if frame=="plain" else text.strip())
    return answers,first_ids,answer_ids,prompts,seqs,labels

def _unit_vectors(units,layer,weight,r_l,alpha=0.8,random_seed=None):
    import torch
    vec=[]
    target=alpha*r_l
    random_units=[u for u in units if u["unit"]=="random"]
    directions=None
    if random_units:
        seed=92000+layer if random_seed is None else random_seed
        directions=c.random_directions(weight,max(int(u["index"]) for u in random_units)+1,seed,target)
    for u in units:
        if u["unit"]=="neuron": vec.append(c.neuron_push(weight,u["index"],target))
        else:
            # Build the seeded pool once per batch; screen indices retain their exact direction.
            vec.append(directions[u["index"]])
    return torch.stack(vec) if vec else None

def _append(path,row):
    with path.open("a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n"); f.flush()
def _seen(path):
    if not path.exists(): return set()
    raw=path.read_bytes(); lines=raw.splitlines(keepends=True); seen=set(); offset=0
    for index,line in enumerate(lines):
        start=offset; offset+=len(line)
        if not line.strip(): continue
        try: row=json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError):
            if index!=len(lines)-1: raise
            # A process may stop during the final append; discard only that torn tail.
            with path.open("r+b") as f: f.truncate(start)
            break
        seen.add(row.get("key"))
    return seen
def _key(stage,arm,unit=None,qi=0,ckpt="",variant=""):
    return "|".join(map(str,(stage,arm,ckpt,variant,unit.get("id","") if unit else "",qi)))
def _do_batch(args,model,tok,layer,units,questions,arm,stage,path,frame="chat",checkpoint="",variant="",group=None,plain=False,corrupt=None,topten=False,alpha=0.8,random_seed=None,sample_seed=None):
    import torch
    existing=_seen(path); rows=[]
    for begin in range(0,len(units),args.batch_size):
        us=units[begin:begin+args.batch_size]; qs=[questions[begin+i] if len(questions)>1 else questions[0] for i in range(len(us))]
        key_items=[_key(stage,arm,u,begin+i,checkpoint,variant) for i,u in enumerate(us)]
        todo=[i for i,k in enumerate(key_items) if k not in existing]
        if not todo: continue
        us=[us[i] for i in todo]; qs=[qs[i] for i in todo]
        if plain or arm=="plain": vectors=None
        elif arm=="noop": vectors=torch.zeros((len(us),int(layer.mlp.down_proj.weight.shape[0])),dtype=layer.mlp.down_proj.weight.dtype,device=layer.mlp.down_proj.weight.device)
        else: vectors=_unit_vectors(us,int(us[0]["layer"]),layer.mlp.down_proj.weight,args.r_l[int(us[0]["layer"])],alpha=alpha,random_seed=random_seed)
        groups=None
        if group:
            groups=[_prompt(tok,q,frame)[2] for q in qs]
            for q,seq,g in zip(qs,[_prompt(tok,q,frame)[1] for q in qs],groups):
                print("POSITION PROMPT",json.dumps({"prompt":_prompt(tok,q,frame)[0],"question":q,"token_ids":seq,"groups":g},ensure_ascii=False),flush=True)
        answers,first,allids,ptexts,pids,plabels=_generate(model,tok,qs,frame,layer,vectors,args.device,args.max_new,arm if arm in PHASES else "full",group,corrupt,sample_seed=sample_seed)
        for i,(u,q,a,ids,all_tokens) in enumerate(zip(us,qs,answers,first,allids)):
            row={"key":_key(stage,arm,u,begin+todo[i],checkpoint,variant),"stage":stage,"arm":arm,"checkpoint":checkpoint,"variant":variant,"layer":u.get("layer"),"unit":u.get("unit"),"index":u.get("index"),"unit_id":u.get("id"),"set":u.get("set"),"question_index":begin+todo[i],"question":q,"dose":alpha if stage=="F" else None,"sample_seed":sample_seed,"prompt":ptexts[i],"answer":a,"first_answer_token_ids":ids,"answer_token_ids":all_tokens,"apology":bool(census139.OPENER.search(a)),"triage":triage92.triage_answer(a)}
            if topten and arm=="plain":
                # Teacher force the literal opening after the same prompt; rank on CPU to avoid device vocab sorts.
                im=torch.tensor([pids[i]+tok("I'm",add_special_tokens=False)["input_ids"]],dtype=torch.long,device=args.device)
                with torch.inference_mode(): logits=model(input_ids=im,use_cache=False).logits[0,-1].float().cpu()
                probs=torch.softmax(logits,dim=-1); vals,idx=torch.topk(probs,10)
                row["after_im_top10"]=[{"id":int(t),"token":tok.decode([int(t)]),"probability":float(v)} for v,t in zip(vals.tolist(),idx.tolist())]
            _append(path,row); rows.append(row); existing.add(row["key"])
    return rows

def _distribution(model,tok,question,frame,layer,vector,prefix_ids):
    import torch
    _text,prompt_ids,_groups=_prompt(tok,question,frame)
    full_ids=prompt_ids+list(prefix_ids)
    ids=torch.tensor([full_ids],dtype=torch.long,device=str(next(model.parameters()).device))
    handle=_position_hook(layer,vector.reshape(1,-1),"full") if vector is not None else None
    try:
        with torch.inference_mode(): logits=model(input_ids=ids,use_cache=False).logits[0,-1].float().cpu()
    finally:
        if handle is not None: handle.remove()
    probs=torch.softmax(logits,dim=-1); vals,idx=torch.topk(probs,10)
    top=[{"id":int(t),"token":tok.decode([int(t)]),"probability":float(v)} for v,t in zip(vals.tolist(),idx.tolist())]
    target_ids=tok(" sorry",add_special_tokens=False)["input_ids"]
    target_id=int(target_ids[0]) if target_ids else None
    target_prob=float(probs[target_id]) if target_id is not None else None
    return top,target_id,target_prob

def _forced_prefix_rows(args,model,tok,layers,units,path):
    import torch
    existing=_seen(path); forced=tok("I'm",add_special_tokens=False)["input_ids"]
    plain=[{"id":"plain","layer":None,"unit":"plain","index":0}]
    work=[(None,plain[0])]+[(u,u) for u in units]
    for layer_no in (4,5):
        cell=[pair for pair in work if pair[0] is None or pair[0]["layer"]==layer_no]
        # Plain baseline is emitted once, attached to L4 only.
        if layer_no==5: cell=[pair for pair in cell if pair[0] is not None]
        for u,meta in cell:
            arm="forced_prefix"; key=_key("A",arm,meta,0,variant="plain" if u is None else "unit")
            if key in existing: continue
            if u is None: vector=None
            else:
                weight=layers[layer_no].mlp.down_proj.weight
                vector=_unit_vectors([u],layer_no,weight,args.r_l[layer_no])[0]
            top,target_id,target_prob=_distribution(model,tok,QUESTION,"chat",layers[layer_no],vector,forced)
            row={"key":key,"stage":"A","arm":arm,"checkpoint":"","variant":"plain" if u is None else "unit","layer":None if u is None else layer_no,"unit":None if u is None else u["unit"],"index":None if u is None else u["index"],"unit_id":meta.get("id"),"set":meta.get("set"),"question":QUESTION,"forced_prefix":"I'm","after_im_top10":top,"sorry_token_id":target_id,"sorry_probability":target_prob}
            _append(path,row); existing.add(key)

def _divergence_rows(args,model,tok,layers,units,path):
    import torch
    existing=_seen(path); rows=[json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]
    plain=next((r for r in rows if r.get("stage")=="A" and r.get("arm")=="plain"),None)
    if plain is None:return
    base=plain.get("answer_token_ids",[])
    for u in units:
        key=_key("A","divergence",u,0)
        if key in existing: continue
        full=next((r for r in rows if r.get("stage")=="A" and r.get("arm")=="full" and r.get("unit_id")==u["id"]),None)
        if full is None: continue
        got=full.get("answer_token_ids",[]); ix=0
        while ix<min(len(base),len(got)) and base[ix]==got[ix]: ix+=1
        if ix==len(base)==len(got):
            top=[]
        else:
            vector=_unit_vectors([u],u["layer"],layers[u["layer"]].mlp.down_proj.weight,args.r_l[u["layer"]])[0]
            top,_,_=_distribution(model,tok,QUESTION,"chat",layers[u["layer"]],vector,base[:ix])
        row={"key":key,"stage":"A","arm":"divergence","layer":u["layer"],"unit":u["unit"],"index":u["index"],"unit_id":u["id"],"set":u["set"],"question":QUESTION,"divergence_position":ix,"plain_divergence_token_id":base[ix] if ix<len(base) else None,"pushed_divergence_token_id":got[ix] if ix<len(got) else None,"divergence_top10":top}
        _append(path,row); existing.add(key)

def run(args):
    import torch
    units_file=Path(args.units); units=json.loads(units_file.read_text(encoding="utf-8"))
    selected=units["sets"]
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); path=out/("pilot.jsonl" if args.pilot else "results.jsonl")
    args.r_l={l:json.loads((ROOT/f"results/part92/screen-instruct/layer_{l}.meta.json").read_text(encoding="utf-8"))["r_L"] for l in (4,5)}
    if args.stage=="E":
        if not args.checkpoint: raise ValueError("Stage E requires --checkpoint pretrain_4T, sft, or dpo")
        frame="plain" if args.checkpoint=="pretrain_4T" else "chat"
    else: frame="chat"
    tok,model=c.load_model(args.model,args.device); layers=c.get_layers(model)
    if args.stage in "ABCD":
        target=selected["R"]+selected["S"]
        if args.pilot: target=[u for u in selected["R"] if u["unit"]=="neuron"][:12]
    if args.stage=="A":
        baseline=[{"id":"plain","layer":None,"unit":"plain","index":0}]
        _do_batch(args,model,tok,layers[4],[baseline[0]],[QUESTION],"plain","A",path,frame=frame,topten=True)
        for layer_no in (4,5):
            for set_name in ("R","S"):
                cell=[u for u in target if u["layer"]==layer_no and (u["set"]==set_name)]
                # Controls use the same unit batches as the phase arms.
                _do_batch(args,model,tok,layers[layer_no],cell,[QUESTION],"plain","A",path,frame,variant="batch-control")
                _do_batch(args,model,tok,layers[layer_no],cell,[QUESTION],"noop","A",path,frame,variant="batch-control")
                for arm in ("full","prompt-only","answer-only"):
                    _do_batch(args,model,tok,layers[layer_no],cell,[QUESTION],arm,"A",path,frame)
        probe_units=[u for u in target if u["set"] in ("R","S")]
        _forced_prefix_rows(args,model,tok,layers,probe_units,path)
        _divergence_rows(args,model,tok,layers,probe_units,path)
    elif args.stage=="B":
        for layer_no in (4,5):
            cell=[u for u in selected["R"] if u["layer"]==layer_no]
            for group in ("position0","user_role","question_words","question_mark","assistant_role","last_position"):
                _do_batch(args,model,tok,layers[layer_no],cell,[QUESTION],group,"B",path,frame,group=group)
    elif args.stage=="C":
        questions=json.loads((HERE/"questions139.json").read_text(encoding="utf-8")); qs=questions["tasks"]+questions["self"]
        for layer_no in (4,5):
            for set_name in ("R","S"):
                cell=[u for u in units["stage_c"] if u["layer"]==layer_no and u["set"]==set_name]
                for qi,q in enumerate(qs):
                    _do_batch(args,model,tok,layers[layer_no],cell,[q]*len(cell),"full","C",path,frame,variant=str(qi))
        plain=[{"id":"plain","layer":None,"unit":"plain","index":0}]
        for qi,q in enumerate(qs):
            _do_batch(args,model,tok,layers[4],plain,[q],"plain","C",path,frame,variant=str(qi))
    elif args.stage=="D":
        qdata=json.loads((HERE/"questions139.json").read_text(encoding="utf-8")); base=[{"id":"plain","layer":None,"unit":"plain","index":0}]
        _do_batch(args,model,tok,layers[4],base,[QUESTION],"plain","D",path,frame)
        rng=random.Random(139); variants=[]
        words=QUESTION.split()
        # Four seeded prompt corruption families, 50 each.
        for k in range(50):
            shuffled=[]
            for word in QUESTION.split():
                letters=list(word); rng.shuffle(letters); shuffled.append("".join(letters))
            variants.append(("shuffle_letters", " ".join(shuffled)))
        for k in range(50):
            variants.append(("drop_chars", "".join(ch for ch in QUESTION if rng.random()>=0.4)))
        pool="Who are you I am a person this is a test please answer the question what model are you"
        for k in range(50): variants.append(("random_words", " ".join(rng.choice(pool.split()) for _ in range(len(words)))))
        original_text,original,original_groups=_prompt(tok,QUESTION,frame); vocab=len(tok)
        replace_at=[i for i,label in enumerate(original_groups) if label in ("question_words","question_mark")]
        rand_ids=[]
        for _ in range(50):
            corrupted=list(original)
            for i in replace_at: corrupted[i]=rng.randrange(vocab)
            rand_ids.append(corrupted)
        for i,(kind,q) in enumerate(variants):
            one=[{"id":f"garble-{kind}-{i}","layer":None,"unit":"plain","index":i}]
            corrupt=None
            if kind=="random_words": q=q
            _do_batch(args,model,tok,layers[4],one,[q],"plain","D",path,frame,variant=f"{kind}-{i}",corrupt=corrupt)
        for i,ids in enumerate(rand_ids):
            one=[{"id":f"garble-random_ids-{i}","layer":None,"unit":"plain","index":i}]
            _do_batch(args,model,tok,layers[4],one,[QUESTION],"plain","D",path,frame,variant=f"random_ids-{i}",corrupt=[ids])
        for i,q in enumerate(qdata["decline"]):
            one=[{"id":f"decline-{i}","layer":None,"unit":"plain","index":i}]
            _do_batch(args,model,tok,layers[4],one,[q],"plain","D",path,frame,variant=f"decline-{i}")
    elif args.stage=="E":
        r=selected["R"]
        med=c.residual_medians(model,tok,frame,args.device); args.r_l={l:med[l] for l in (4,5)}
        plain=[{"id":"plain","layer":None,"unit":"plain","index":0}]
        _do_batch(args,model,tok,layers[4],plain,[QUESTION],"plain","E",path,frame,checkpoint=args.checkpoint)
        for layer_no in (4,5):
            cell=[u for u in r if u["layer"]==layer_no]
            _do_batch(args,model,tok,layers[layer_no],cell,[QUESTION],"full","E",path,frame,checkpoint=args.checkpoint)
    elif args.stage=="F":
        units_f=units["stage_f"]["units"]
        for layer_no in (4,5):
            cell=[u for u in units_f if u["layer"]==layer_no]
            for alpha in (0.2,0.4,0.8,1.2,1.6):
                _do_batch(args,model,tok,layers[layer_no],cell,[QUESTION],"full","F",path,frame,variant=f"alpha-{alpha}",alpha=alpha,random_seed=139100+layer_no)
    elif args.stage=="G":
        for layer_no in (4,5):
            cell=[u for u in selected["R"] if u["layer"]==layer_no]
            for replicate in range(8):
                _do_batch(args,model,tok,layers[layer_no],cell,[QUESTION],"sampled","G",path,frame,variant=f"rep-{replicate}",sample_seed=139000+replicate)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",required=True,choices=list("ABCDEFG")); ap.add_argument("--pilot",action="store_true")
    ap.add_argument("--model",required=True); ap.add_argument("--checkpoint",choices=("pretrain_4T","sft","dpo"))
    ap.add_argument("--units",default=str(HERE/"units139.json")); ap.add_argument("--device",default="xpu")
    ap.add_argument("--batch-size",type=int,default=32); ap.add_argument("--max-new",type=int,default=32); ap.add_argument("--out",default="results/part139")
    args=ap.parse_args(argv)
    if args.batch_size<1 or args.max_new<1: ap.error("batch-size and max-new must be positive")
    run(args)
if __name__=="__main__": main()
