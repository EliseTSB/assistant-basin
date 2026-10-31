#!/usr/bin/env python3
"""Part 136 effects; model inference stays in bf16."""
from __future__ import annotations
import argparse, base64, json, random, struct, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part136"))
sys.path.insert(0,str(ROOT/"part86"))
sys.path.insert(0,str(ROOT/"part119c")); sys.path.insert(0,str(ROOT/"part119b"))
SEED=136; NVOCAB=4096; DOSE=.8

def fixed_indices(vocab_size:int):
    if vocab_size<NVOCAB: raise ValueError("vocabulary smaller than fixed index set")
    return sorted(random.Random(SEED).sample(range(vocab_size),NVOCAB))

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=("effects","steering"),required=True)
    p.add_argument("--device",default="xpu"); p.add_argument("--model",required=True,help="local Qwen3.5-4B Part 112 snapshot directory")
    p.add_argument("--scan",required=True); p.add_argument("--out",required=True); p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--part86",default=str(ROOT/"results/part86/screen-final"))
    p.add_argument("--timing-log",help="append this stage's elapsed time as JSONL")
    return p.parse_args()

def steering_push(down_proj, neuron:int, dose:float, device, dtype):
    """Build the dense down_proj-input vector expected by Part 86's AddRowsHook."""
    import torch
    push=torch.zeros((1,int(down_proj.in_features)),device=device,dtype=dtype)
    push[0,int(neuron)]=dose
    return push

def prepare_firing(row):
    """Trim a firing context while retaining firings at the final token."""
    pos=int(row["position"])
    trimmed={**row,"token_ids":row["token_ids"][:min(len(row["token_ids"]),pos+9)]}
    return trimmed if len(trimmed["token_ids"])>pos else None

def append_missing(path, records, key):
    path.parent.mkdir(parents=True,exist_ok=True)
    done=set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try: done.add(key(json.loads(line)))
                except (ValueError,TypeError): continue
    with path.open("a",encoding="utf-8") as f:
        for row in records:
            if key(row) not in done:
                f.write(json.dumps(row,ensure_ascii=False)+"\n"); done.add(key(row))

def existing_keys(path, key):
    if not path.exists(): return set()
    found=set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try: found.add(key(json.loads(line)))
            except (ValueError,TypeError): continue
    return found

def pack_vector(values):
    return base64.b64encode(struct.pack("<%df"%len(values),*values)).decode("ascii")

def effect_batch(model, down_proj, rows, device, vocab):
    """Run shared baselines and row-specific Part 33 zero hooks for one batch."""
    import torch
    import part33
    width=max(len(r["token_ids"]) for r in rows); ids=torch.zeros((len(rows),width),dtype=torch.long,device=device)
    mask=torch.zeros_like(ids); conditions=[]
    for i,row in enumerate(rows):
        seq=torch.tensor(row["token_ids"],dtype=torch.long,device=device); ids[i,:len(seq)]=seq; mask[i,:len(seq)]=1
        active=[False]*width; active[int(row["position"])]=True
        conditions.append((int(row["neuron"]),active))
    with torch.inference_mode(): base=model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float()
    base_lp=torch.log_softmax(base,dim=-1).index_select(-1,vocab)
    hook=part33.ConditionalZeroHook(down_proj,conditions)
    try:
        with torch.inference_mode(): changed=model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float()
    finally: hook.remove()
    lp=torch.log_softmax(changed,dim=-1).index_select(-1,vocab); output=[]
    for i,row in enumerate(rows):
        pos=int(row["position"]); end=min(pos+8,len(row["token_ids"])-1)
        da=(lp[i,pos]-base_lp[i,pos]).detach().cpu().tolist()
        db=(lp[i,pos+1:end+1]-base_lp[i,pos+1:end+1]).sum(dim=0).detach().cpu().tolist()
        output.append((da,db))
    return output

def run():
    started=time.perf_counter(); a=parse_args(); import torch
    from run119c import load_model
    import part33
    tok,model=load_model(a.model,a.device,torch.bfloat16)
    model.eval(); model.requires_grad_(False); torch.set_grad_enabled(False)
    from common86 import find_layers
    layers=find_layers(model)
    scan=Path(a.scan); firings=[json.loads(s) for s in (scan/"firings.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    indices=fixed_indices(int(model.config.vocab_size)); vocab=torch.tensor(indices,device=a.device,dtype=torch.long)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    (out/"vocab_indices.json").write_text(json.dumps({"seed":SEED,"indices":indices},indent=2)+"\n",encoding="utf-8")
    part86=Path(a.part86); medians={}
    for layer in sorted({int(r["layer"]) for r in firings}):
        candidates=[part86/f"layer_{layer}.meta.json",part86/f"layer_{layer}.jsonl.meta.json"]
        meta_path=next((x for x in candidates if x.exists()),None)
        if meta_path:
            meta=json.loads(meta_path.read_text(encoding="utf-8")); medians[layer]=float(meta.get("median",meta.get("config",{}).get("median")))
    if a.stage=="steering" and not medians: raise FileNotFoundError(f"Part 86 layer median metadata not found under {part86}")
    gate_path=out/"gates.json"
    gates=json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {"do_nothing_hook_bit_identical":None,"model":"Qwen/Qwen3.5-4B","dtype":"bfloat16"}
    # Resume unit of work is pole/firing; all effects use the same baseline log-probability vector.
    if a.stage=="effects":
        path=out/"effects.jsonl"; records=[]
        key_fn=lambda r:(r["pole_id"],r["doc_id"],r["position"])
        done=existing_keys(path,key_fn)
        pending=[r for r in firings if (r["pole_id"],r["doc_id"],int(r["position"])) not in done and int(r["position"])>=1]
        for start in range(0,len(pending),a.batch_size):
            batch=pending[start:start+a.batch_size]
            for layer in sorted({int(r["layer"]) for r in batch}):
                group=[r for r in batch if int(r["layer"])==layer]
                prepared=[trimmed for row in group if (trimmed:=prepare_firing(row)) is not None]
                if not prepared: continue
                if gates["do_nothing_hook_bit_identical"] is None:
                    first=prepared[0]; ids=torch.tensor(first["token_ids"],device=a.device,dtype=torch.long).unsqueeze(0)
                    with torch.inference_mode(): plain=model(input_ids=ids,use_cache=False).logits
                    noop=layers[layer].mlp.down_proj.register_forward_pre_hook(lambda _m,_args:None)
                    try:
                        with torch.inference_mode(): hooked=model(input_ids=ids,use_cache=False).logits
                    finally: noop.remove()
                    gates["do_nothing_hook_bit_identical"]=bool(torch.equal(plain,hooked))
                    if not gates["do_nothing_hook_bit_identical"]: raise RuntimeError("no-op pre-hook changed logits")
                vectors=effect_batch(model,layers[layer].mlp.down_proj,prepared,a.device,vocab)
                for row,(da,db) in zip(prepared,vectors):
                    records.append({"pole_id":row["pole_id"],"layer":layer,"neuron":int(row["neuron"]),"kind":row["kind"],"doc_id":row["doc_id"],"source":row["source"],"position":int(row["position"]),"rank":row["rank"],"indices_seed":SEED,"vocab_n":NVOCAB,"effect_a_f32_le_b64":pack_vector(da),"effect_b_f32_le_b64":pack_vector(db)})
                append_missing(path,records,key_fn); records=[]
    else:
        import common86
        path=out/"steering.jsonl"; records=[]
        done=existing_keys(path,lambda r:r["pole_id"])
        for row in firings:
            # One steering row per experimental pole/control, independent of natural firing count.
            key=(row["pole_id"])
            if key in done or any(x["pole_id"]==key for x in records): continue
            prompt=common86.build_prompt(tok,"Who are you?")
            encoded=tok(prompt,return_tensors="pt",add_special_tokens=False)
            ids=(encoded["input_ids"] if isinstance(encoded,dict) else encoded.input_ids).to(a.device)
            layer=int(row["layer"]); neuron=int(row["neuron"]); weight=layers[layer].mlp.down_proj.weight[:,neuron].detach().float()
            median=medians.get(layer)
            if median is None: raise ValueError(f"Part 86 median missing for layer {layer}")
            dose=DOSE*median/float(torch.linalg.vector_norm(weight).cpu())
            push_vec=steering_push(layers[layer].mlp.down_proj,neuron,dose,a.device,torch.bfloat16)
            # First answer token is unpushed because Part 86 leaves prefill untouched.
            h=common86.AddRowsHook(layers[layer].mlp.down_proj,push_vec)
            try:
                with torch.inference_mode(): prefill=model(input_ids=ids,use_cache=True)
                first=prefill.logits[:,-1,:].argmax(dim=-1,keepdim=True)
                with torch.inference_mode(): changed=model(input_ids=first,past_key_values=prefill.past_key_values,use_cache=True).logits[0,-1].float()
            finally: h.remove()
            with torch.inference_mode():
                plain_prefill=model(input_ids=ids,use_cache=True)
                plain=model(input_ids=first,past_key_values=plain_prefill.past_key_values,use_cache=True).logits[0,-1].float()
            delta=(torch.log_softmax(changed,dim=-1).index_select(0,vocab)-torch.log_softmax(plain,dim=-1).index_select(0,vocab)).detach().cpu().tolist()
            records.append({"pole_id":key,"layer":layer,"neuron":neuron,"kind":row["kind"],"indices_seed":SEED,"dose":DOSE,"median_reference":median,"first_answer_token_unpushed":int(first.item()),"logprob_delta":delta})
        append_missing(path,records,lambda r:r["pole_id"])
    (out/"gates.json").write_text(json.dumps(gates,indent=2)+"\n",encoding="utf-8")
    elapsed=time.perf_counter()-started
    if a.timing_log:
        timing_path=Path(a.timing_log); timing_path.parent.mkdir(parents=True,exist_ok=True)
        with timing_path.open("a",encoding="utf-8") as f:
            f.write(json.dumps({"stage":a.stage,"batch_size":a.batch_size,"rows":len(records),"elapsed_s":elapsed})+"\n")
    print(f"stage={a.stage} batch_size={a.batch_size} rows={len(records)} elapsed_s={elapsed:.1f}",flush=True)

if __name__=="__main__": run()
