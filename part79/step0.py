#!/usr/bin/env python3
"""Part 79 calibration job: packed-token layer norms, down columns and greedy pilot."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part66 import part66
import neuron_report as nr
from part79 import part79 as p79

PROMPTS=['Who are you?','Write a snake game in Python.','Describe the color orange.','Write whatever you like.']

def packed_windows(path,tok,limit=400):
    import json
    buf=[]; windows=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            doc=json.loads(line).get('text','')
            if not isinstance(doc,str): continue
            ids=tok(doc,add_special_tokens=False)['input_ids']
            if hasattr(ids,'tolist'): ids=ids.tolist()
            buf.extend(map(int,ids)); buf.append(int(tok.eos_token_id))
            while len(buf)>=1024 and len(windows)<limit:
                windows.append(buf[:1024]); buf=buf[1024:]
            if len(windows)>=limit: break
    if len(windows)<limit: raise ValueError(f'Pile input yielded {len(windows)} windows; {limit} required')
    return windows

def summarize_norms(model,layers,windows,batch):
    import numpy as np,torch
    captured={13:[],14:[],22:[]}; handles=[]
    for li in captured:
        def hook(_m,_i,out,li=li):
            x=out[0] if isinstance(out,(tuple,list)) else out
            captured[li].append(torch.linalg.vector_norm(x.detach().float(),dim=-1)[:,1:].reshape(-1).to('cpu',dtype=torch.float32))
        handles.append(layers[li].register_forward_hook(hook))
    try:
        with torch.no_grad():
            for i in range(0,len(windows),batch):
                ids=torch.tensor(windows[i:i+batch],dtype=torch.long,device=next(model.parameters()).device)
                model(input_ids=ids,use_cache=False)
    finally:
        for h in handles:h.remove()
    out={}
    for li,parts in captured.items():
        a=torch.cat(parts).numpy(); q=np.quantile(a,[.1,.5,.9],method='linear')
        out[f'L{li}']={'p10':float(q[0]),'median':float(q[1]),'p90':float(q[2]),'token_count':int(a.size),'position_start':1}
    return out

def pilot_prompt(tok,prompt):
    """Lyra's rendering: one user turn, no system prompt, generation prompt on, thinking off."""
    return nr.build_prompt(tok,prompt,False)

def compute_deltas(model,layers,medians):
    import torch
    targets=[(14,8949,1),(13,7956,-1)]+[(14,u,1) for u in p79.CTL14]+[(13,u,-1) for u in p79.CTL13]
    out={}
    for li,u,dose in targets:
        proj=layers[li].mlp.down_proj; w=proj.weight.detach().float()
        if u<0 or u>=w.shape[1]: raise ValueError(f'unit {u} outside layer {li} width {w.shape[1]}')
        norm=float(torch.linalg.vector_norm(w[:,u]).item())
        if not norm>0: raise ValueError(f'zero column norm at L{li} N{u}')
        med=float(medians[f'L{li}']['median']); key=f'L{li}_N{u}'
        out[key]={'layer':li,'unit':u,'dose':dose,'column_norm':norm,'median':med,'delta':dose*med/norm}
    return out

def run(args):
    import torch
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    tok,model=nr.load_model(part66.MODEL,part66.REVISION,args.device); model.requires_grad_(False)
    _,layers,_=nr.find_text_layers(model)
    windows=packed_windows(args.pile,tok,getattr(args,'windows',400))
    medians=summarize_norms(model,layers,windows,args.batch); deltas=compute_deltas(model,layers,medians)
    metadata={'model':part66.MODEL,'revision':part66.REVISION,'dtype':'bfloat16','windows':len(windows),'window_tokens':1024,'batch_size':args.batch,'positions_used':'1..1023','forward_pass':'whole model','medians':medians,'deltas':deltas,'control_units':{'L14':list(p79.CTL14),'L13':list(p79.CTL13)}}
    (out/'step0.json').write_text(json.dumps(metadata,indent=2)+'\n')
    pilot=[]
    for condition in ('unchanged','evil'):
      for qi,prompt in enumerate(PROMPTS):
        rendered=pilot_prompt(tok,prompt)
        ids=tok(rendered,return_tensors='pt',add_special_tokens=False)['input_ids'].to(args.device)
        handles=[]
        if condition=='evil':
            for key in ('L14_N8949','L13_N7956'):
                d=deltas[key]; handles.append(layers[d['layer']].mlp.down_proj.register_forward_pre_hook(p79.add_hook(d['unit'],d['delta'],{'calls':0,'max_abs_diff':0.})))
        try:
            with torch.no_grad(): gen=model.generate(input_ids=ids,max_new_tokens=128,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        finally:
            for h in handles:h.remove()
        raw=gen[0,ids.shape[1]:].tolist()
        if tok.eos_token_id is not None and int(tok.eos_token_id) in raw: raw=raw[:raw.index(int(tok.eos_token_id))+1]
        pilot.append({'prompt':prompt,'condition':condition,'answer':tok.decode(raw,skip_special_tokens=True).strip(),'new_tokens':len(raw),'stop_reason':part66._stop_reason(raw,tok)})
    (out/'pilot.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in pilot))
    print(json.dumps({'step0':str(out/'step0.json'),'pilot_rows':len(pilot),'medians':medians},indent=2))

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--pile',default='qwen35/pile10k.jsonl'); ap.add_argument('--out',required=True); ap.add_argument('--device',default='xpu'); ap.add_argument('--batch',type=int,default=4); ap.add_argument('--dry-run',action='store_true'); a=ap.parse_args()
    if a.batch<1: ap.error('--batch must be positive')
    if a.dry_run:
        print(json.dumps({'windows':400,'layers':[13,14,22],'units':[(14,8949),(13,7956),*[(14,u) for u in p79.CTL14],*[(13,u) for u in p79.CTL13]],'batch':a.batch},indent=2)); return
    run(a)
if __name__=='__main__': main()
