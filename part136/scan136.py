#!/usr/bin/env python3
"""Collect top natural down_proj-input firings for Part 136 poles."""
from __future__ import annotations
import argparse, json, random, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part86"))
sys.path.insert(0, str(ROOT / "part115"))
sys.path.insert(0, str(ROOT / "part115b"))

WIKI = r"E:\\neuron-probe\\corpus\\wikitext103-train0.parquet"
CHAT = r"E:\\neuron-probe\\corpus\\ultrachat-test_sft.parquet"
MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

def read_rows(path: str, kind: str, limit: int = 4000):
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["text"] if kind == "wiki" else ["messages", "prompt"])
    rows = table.to_pylist()
    out = []
    for i, row in enumerate(rows):
        if kind == "wiki": text = row.get("text") or ""
        else:
            text = row.get("prompt") or ""
            messages = row.get("messages") or []
            if messages:
                text = ""  # messages are authoritative; render with the model template
                try: text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                except Exception: text = "\n".join(str(m.get("role", "user"))+": "+str(m.get("content", "")) for m in messages)
            elif text:
                text = tokenizer.apply_chat_template([{"role":"user", "content":text}], tokenize=False, add_generation_prompt=True)
        if text.strip():
            out.append({"doc_id": f"{kind}:{i}", "source":kind, "text":text})
            if len(out) >= limit: break
    return out

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--device",default="xpu"); p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--wiki",default=WIKI); p.add_argument("--chat",default=CHAT)
    p.add_argument("--units",default=str(ROOT/"results/part112/units.json")); p.add_argument("--score",default=str(ROOT/"results/part112/score.json"))
    p.add_argument("--atlas",default=str(ROOT/"results/part112/units.json")); p.add_argument("--out",required=True)
    p.add_argument("--limit-docs",type=int,default=4000)
    p.add_argument("--max-poles",type=int,default=446,help="small deterministic pilot subset")
    p.add_argument("--timing-log",help="append this stage's elapsed time as JSONL")
    return p.parse_args()

def select_pilot_poles(poles, max_poles, bands, seed=136):
    """Select equal concept/slot quotas by seeded round robin across fixed bands."""
    if max_poles % 2:
        raise ValueError("--max-poles must be even to keep concept and slot counts equal")
    quota=max_poles//2
    selected=[]
    for kind in ("concept","slot"):
        grouped={name:[] for _,_,name in bands}
        for pole in poles:
            if pole["kind"] not in ("concept","slot") or pole["kind"]!=kind:
                continue
            layer=int(pole["layer"])
            name=next((name for lo,hi,name in bands if lo<=layer<=hi),None)
            if name is not None: grouped[name].append(pole)
        rng=random.Random(seed+ (0 if kind=="concept" else 1))
        for rows in grouped.values(): rng.shuffle(rows)
        picked=[]
        while len(picked)<quota:
            progressed=False
            for name in grouped:
                if grouped[name] and len(picked)<quota:
                    picked.append(grouped[name].pop())
                    progressed=True
            if not progressed:
                raise ValueError(f"not enough {kind} units to select {quota} pilot poles")
        selected.extend(picked)
    return selected

def main():
    global tokenizer
    started=time.perf_counter(); a=parse_args()
    import torch, numpy as np
    if a.device == "xpu" and not (hasattr(torch,"xpu") and torch.xpu.is_available()): raise RuntimeError("XPU requested but unavailable")
    from howmany115b import load_direct
    import part33
    # Part 115b loads this model directly in bf16; Part 136 has no fp32 model path.
    tokenizer, model=load_direct(a.device)
    # Force the registered revision and model ID used by load_direct's shared loader.
    units=json.loads(Path(a.units).read_text(encoding="utf-8")); score=json.loads(Path(a.score).read_text(encoding="utf-8"))
    atlas_rows=json.loads(Path(a.atlas).read_text(encoding="utf-8")); atlas={(int(r["layer"]),int(r["neuron"])) for r in atlas_rows}
    poles=[]
    for u in units:
        s=score["units"].get(u["id"],{})
        if s.get("kind") in ("concept","slot"):
            poles.append({"pole_id":u["id"],"layer":int(u["layer"]),"neuron":int(u["neuron"]),"kind":s["kind"],"persona":u["persona"]})
    poles.append({"pole_id":"gate:L22:N2742","layer":22,"neuron":2742,"kind":"gate","persona":"known alien unit"})
    if len([p for p in poles if p["kind"] in ("concept","slot")])!=446: raise ValueError(f"expected 446 sorted concept/slot units, got {len(poles)-1}")
    if not 1<=a.max_poles<=446: raise ValueError("--max-poles must be in 1..446")
    import score136
    score136.BANDS=score136.fixed_bands(units,score)
    ordinary=select_pilot_poles(poles,a.max_poles,score136.BANDS)
    poles=ordinary+[p for p in poles if p["kind"]=="gate"]
    by_layer=Counter(p["layer"] for p in poles)
    from common86 import find_layers
    layers=find_layers(model)
    neuron_width=int(layers[0].mlp.down_proj.in_features)
    # The first pass computes every layer neuron's firing rate on device and transfers only one count vector.
    target_groups={}
    for p in poles: target_groups.setdefault(p["layer"],[]).append(p)
    count_sum={l:torch.zeros(neuron_width,dtype=torch.int64) for l in target_groups}; valid_sum={l:0 for l in target_groups}
    docs=read_rows(a.wiki,"wiki",a.limit_docs)+read_rows(a.chat,"chat",a.limit_docs)
    model.eval(); model.requires_grad_(False)
    # Reuse Part 33's activation-capture helper as a calibration check for our batched capture.
    tokenizer.padding_side="right"
    check_layer=poles[0]["layer"]; check_neuron=poles[0]["neuron"]
    if not docs: raise ValueError("no nonblank scan documents")
    check_ids=tokenizer(docs[0]["text"],return_tensors="pt",truncation=True,max_length=128)["input_ids"].to(a.device)
    check_mask=torch.ones_like(check_ids)
    _shared_output, shared_capture=part33._capture_activation_pass(model,layers[check_layer].mlp.down_proj,check_ids,check_mask,[check_neuron])
    capture_checked=False
    for start in range(0,len(docs),a.batch_size):
        rows=docs[start:start+a.batch_size]
        enc=tokenizer([r["text"] for r in rows],return_tensors="pt",padding=True,truncation=True,max_length=128)
        ids=enc["input_ids"].to(a.device); mask=enc["attention_mask"].to(a.device)
        valid_dev=mask.bool(); valid_dev[:,0]=False
        captured={}; handles=[]
        for layer, group in target_groups.items():
            def hook(_m,args,layer=layer,group=group):
                x=args[0]
                # Gather target columns once; accumulate all-neuron rates on device.
                cols=torch.tensor([p["neuron"] for p in group],device=x.device,dtype=torch.long)
                captured[layer]=x.index_select(-1,cols).detach().float().to("cpu")
                count_sum[layer].add_((((x>0)&valid_dev.unsqueeze(-1)).sum(dim=(0,1))).to("cpu"))
            handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
        try:
            with torch.inference_mode(): model(input_ids=ids,attention_mask=mask,use_cache=False)
        finally:
            for h in handles: h.remove()
        valid=valid_dev.detach().cpu(); valid_total=int(valid.sum())
        if start==0:
            group=target_groups[check_layer]; col=next(i for i,u in enumerate(group) if u["neuron"]==check_neuron)
            doc_len=int(mask[0].sum())
            if not np.allclose(captured[check_layer][0,:doc_len,col].numpy(),shared_capture[:doc_len,0],rtol=1e-2,atol=1e-2):
                raise RuntimeError("batched down_proj activation capture differs from Part 33 helper")
            capture_checked=True
        for layer in target_groups: valid_sum[layer]+=valid_total
    if not capture_checked: raise RuntimeError("Part 33 capture calibration was not checked")
    rates_by_layer={l:count_sum[l].numpy()/valid_sum[l] for l in target_groups}
    selected_controls=[]; used_controls=set()
    for p in poles:
        layer=p["layer"]; blocked={n for li,n in atlas if li==layer}|{q["neuron"] for q in poles if q["layer"]==layer}|{n for li,n in used_controls if li==layer}
        available=np.asarray([n for n in range(neuron_width) if n not in blocked],dtype=np.int64)
        if not len(available): raise ValueError(f"no non-atlas control available in layer {layer}")
        target_rate=rates_by_layer[layer][p["neuron"]]
        nearest=available[np.argsort(np.abs(rates_by_layer[layer][available]-target_rate))]
        chosen=int(nearest[0]); used_controls.add((layer,chosen))
        selected_controls.append({"pole_id":f"control:L{layer}:{chosen}","layer":layer,"neuron":chosen,"kind":"control","persona":"matched control"})
    # Second pass records only the 892 selected columns; one column transfer per layer and batch.
    layer_groups={}
    for p in poles+selected_controls: layer_groups.setdefault(p["layer"],[]).append(p)
    activation_chunks={l:[] for l in layer_groups}; position_chunks={l:[] for l in layer_groups}
    for start in range(0,len(docs),a.batch_size):
        rows=docs[start:start+a.batch_size]
        enc=tokenizer([r["text"] for r in rows],return_tensors="pt",padding=True,truncation=True,max_length=128)
        ids=enc["input_ids"].to(a.device); mask=enc["attention_mask"].to(a.device)
        valid_dev=mask.bool(); valid_dev[:,0]=False; captured={}; handles=[]
        for layer,group in layer_groups.items():
            cols=torch.tensor([p["neuron"] for p in group],device=a.device,dtype=torch.long)
            def hook(_m,args,layer=layer,cols=cols):
                x=args[0]; captured[layer]=x.index_select(-1,cols).detach().float().to("cpu")
            handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
        try:
            with torch.inference_mode(): model(input_ids=ids,attention_mask=mask,use_cache=False)
        finally:
            for h in handles: h.remove()
        valid=valid_dev.detach().cpu()
        for layer in layer_groups:
            values=captured[layer]
            masked=values.masked_fill(~valid.unsqueeze(-1),float("-inf"))
            maxima,positions=masked.max(dim=1)
            maxima=torch.where(torch.isfinite(maxima),maxima,torch.zeros_like(maxima))
            activation_chunks[layer].append(maxima.numpy())
            position_chunks[layer].append(positions.numpy())
    # Bulk transfers above leave only doc-by-neuron matrices; rank the top firings without a Python loop per value.
    activations={l:np.concatenate(chunks,axis=0) for l,chunks in activation_chunks.items()}
    positions={l:np.concatenate(chunks,axis=0) for l,chunks in position_chunks.items()}
    rates={}
    for group in layer_groups.values():
        for p in group: rates[p["pole_id"]]=float(rates_by_layer[p["layer"]][p["neuron"]])
    final=[]; left=[]
    selected=poles+selected_controls
    for p in selected:
        group=layer_groups[p["layer"]]; col=next(i for i,x in enumerate(group) if x["pole_id"]==p["pole_id"])
        vals=activations[p["layer"]][:,col]
        ranked=np.argsort(vals)[::-1][:24]
        candidates=[]
        for doc_index in ranked:
            activation=float(vals[doc_index])
            if activation<=0: continue
            doc=docs[int(doc_index)]; pos=int(positions[p["layer"]][doc_index,col])
            ids_row=tokenizer(doc["text"],return_tensors="pt",truncation=True,max_length=128)["input_ids"][0].tolist()
            candidates.append({"activation":activation,"doc_id":doc["doc_id"],"source":doc["source"],"text":doc["text"],"position":pos,"token_ids":ids_row})
        # Different docs only; reject below a quarter of this pole's best natural activation.
        best=candidates[0]["activation"] if candidates else 0
        picked=[]; seen=set()
        for r in candidates:
            if r["activation"] < .25*best or r["doc_id"] in seen: continue
            seen.add(r["doc_id"]); picked.append(r)
            if len(picked)==24: break
        if len(picked)<24: left.append({**p,"firings":len(picked),"max_activation":best}); continue
        for rank,r in enumerate(picked,1): final.append({**p,**r,"rank":rank,"firing_rate":rates[p["pole_id"]]})
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    with (out/"firings.jsonl").open("w",encoding="utf-8") as f:
        for r in final: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    (out/"left_out.json").write_text(json.dumps(left,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"scan_meta.json").write_text(json.dumps({"model":MODEL_ID,"revision":REVISION,"device":a.device,"batch_size":a.batch_size,"documents":len(docs),"pole_count":len(poles),"control_count":len(selected_controls),"matched_poles":[{**p,"matched_control":c["pole_id"],"firing_rate":rates[p["pole_id"]],"control_firing_rate":rates[c["pole_id"]]} for p,c in zip(poles,selected_controls)],"left_out_count":len(left),"dose_reference":"Part 86 push = 0.8 × residual median"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    elapsed=time.perf_counter()-started
    if a.timing_log:
        timing_path=Path(a.timing_log); timing_path.parent.mkdir(parents=True,exist_ok=True)
        with timing_path.open("a",encoding="utf-8") as f:
            f.write(json.dumps({"stage":"scan","batch_size":a.batch_size,"rows":len(final),"elapsed_s":elapsed})+"\n")
    print(f"scan batch_size={a.batch_size} docs={len(docs)} poles={len(poles)} elapsed_s={elapsed:.1f}",flush=True)

if __name__=="__main__": main()
