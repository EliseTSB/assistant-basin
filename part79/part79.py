#!/usr/bin/env python3
"""Part 79 sampled run: additive Lyra push, optionally with Part 66 makers zeroed."""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part66 import part66
from part78 import part78 as p78
import neuron_report as nr

MODEL, REVISION = p78.MODEL, p78.REVISION
EVIL=((14,8949),(13,7956))
CTL14=tuple(nr.control_units(9216,8949,5,79079)); CTL13=tuple(nr.control_units(9216,7956,5,79080))
CONDITION_IDS=('e0_unchanged','e1_makers','e2_evil','e3_makers_evil')+tuple(f'e{i}_makers_ctl{i-3}' for i in range(4,9))

def build_conditions(deltas):
    cs=[{'id':'e0_unchanged','makers':False,'push':[]},{'id':'e1_makers','makers':True,'push':[]},
        {'id':'e2_evil','makers':False,'push':[deltas['L14_N8949'],deltas['L13_N7956']]},
        {'id':'e3_makers_evil','makers':True,'push':[deltas['L14_N8949'],deltas['L13_N7956']]}]
    for i,(u14,u13) in enumerate(zip(CTL14,CTL13),1):
        cs.append({'id':f'e{i+3}_makers_ctl{i}','makers':True,'push':[deltas[f'L14_N{u14}'],deltas[f'L13_N{u13}']]})
    return cs

def planned_rows(conditions=None):
    conditions=conditions or build_conditions({'L14_N8949':{},'L13_N7956':{},**{f'L14_N{x}':{} for x in CTL14},**{f'L13_N{x}':{} for x in CTL13}})
    rows=[]
    for c in conditions:
        for qi,q in enumerate(part66.DIRECT+part66.INDIRECT):
            for s in range(8): rows.append((c,qi,q,'direct' if qi<16 else 'indirect',s))
        for gi,(q,_a) in enumerate(part66.GUARDS):
            for s in range(4): rows.append((c,25+gi,q,'guard',s))
    return rows

def add_hook(unit,delta,stats):
    """Add scalar at every position; record call count and input-difference error."""
    def hook(_module,args):
        x=args[0]; y=x.clone(); y[...,int(unit)]=y[...,int(unit)]+float(delta)
        err=float(((y[...,int(unit)].detach().float()-x[...,int(unit)].detach().float())-float(delta)).abs().max().item())
        stats['calls']=stats.get('calls',0)+1; stats['max_abs_diff']=max(stats.get('max_abs_diff',0.),err)
        return (y,)+tuple(args[1:])
    return hook

def _existing_rows(path):
    done=set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r=json.loads(line); done.add((r['condition'],int(r['question_index']),int(r['sample'])))
            except (ValueError,KeyError,TypeError): pass
    return done

def run(args):
    import torch, part25
    p=Path(args.step0)
    if not p.is_file(): raise FileNotFoundError(f'missing step0 file: {p}')
    meta=json.loads(p.read_text()); med=meta.get('medians',{}); ds=meta.get('deltas',{})
    required=['L13','L14','L22']
    if any(k not in med or med[k].get('median') is None for k in required): raise ValueError('step0.json lacks required medians')
    units=[(14,8949),(13,7956)]+[(14,u) for u in CTL14]+[(13,u) for u in CTL13]
    if any(f'L{l}_N{u}' not in ds or ds[f'L{l}_N{u}'].get('delta') is None for l,u in units): raise ValueError('step0.json lacks required delta')
    conditions=build_conditions(ds); rows=planned_rows(conditions)
    if args.dry_run:
        print(json.dumps({'planned_rows':len(rows),'conditions':CONDITION_IDS},indent=2)); return
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    tok,model=nr.load_model(MODEL,REVISION,args.device); model.requires_grad_(False)
    _,layers,_=nr.find_text_layers(model); path=out/'answers.jsonl'; done=_existing_rows(path)
    todo=[r for r in rows if (r[0]['id'],r[1],r[4]) not in done]; groups=defaultdict(list)
    for r in todo: groups[(r[0]['id'],r[1])].append(r)
    batches=[g[i:i+args.batch] for g in groups.values() for i in range(0,len(g),args.batch)]
    previous_run=out/'run.json'
    evidence=json.loads(previous_run.read_text()).get('hook_batch_evidence',[]) if previous_run.exists() else []
    first_makers=set(); first_push=set(); start=time.time()
    with path.open('a') as sink:
      for chunk in batches:
        prompts=[part25._token_ids(tok,nr.build_prompt(tok,r[2],False)) for r in chunk]
        ids,mask=part66._pad(prompts,tok,torch.device(args.device)); c=chunk[0][0]
        seeds=[part66.row_seed(c['id'],r[1],r[4]) for r in chunk]; torch.manual_seed(seeds[0])
        rowops=[part25._operations(part66.TARGET,'zero') for _ in chunk] if c['makers'] else [{} for _ in chunk]
        handles=part66._hooks(layers,rowops,mask.bool()) if c['makers'] else []; maker_stats=[]
        if c['makers']:
            for li,_u in part66.TARGET:
                st={'layer':li,'unit':_u,'calls':0}; maker_stats.append(st)
                handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(
                    lambda _m,_a,s=st: s.__setitem__('calls',s['calls']+1)))
        push_stats=[]
        for push in c['push']:
            l,u=int(push['layer']),int(push['unit'])
            st={'layer':l,'unit':u,'calls':0,'max_abs_diff':0.}; push_stats.append(st)
            handles.append(layers[l].mlp.down_proj.register_forward_pre_hook(add_hook(u,push['delta'],st)))
        try:
            with torch.no_grad(): result=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=64,do_sample=True,temperature=.7,top_p=.95,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
        finally:
            for h in handles: h.remove()
        if c['makers'] and c['id'] not in first_makers:
            if any(not s['calls'] for s in maker_stats): raise RuntimeError(f'makers hooks did not all fire on first batch for {c["id"]}')
            first_makers.add(c['id'])
        if c['push'] and c['id'] not in first_push:
            if any(not s['calls'] for s in push_stats): raise RuntimeError(f'add hooks did not all fire on first batch for {c["id"]}')
            first_push.add(c['id'])
        evidence.append({'condition':c['id'],'question_index':chunk[0][1],'push_hooks':push_stats,'makers_hooks':maker_stats})
        plen=ids.shape[1]
        for i,r in enumerate(chunk):
            raw=result[i,plen:].tolist()
            if tok.eos_token_id is not None and int(tok.eos_token_id) in raw: raw=raw[:raw.index(int(tok.eos_token_id))+1]
            sink.write(json.dumps({'condition':c['id'],'question_index':r[1],'question':r[2],'question_kind':r[3],'sample':r[4],'seed':seeds[i],'answer':tok.decode(raw,skip_special_tokens=True).strip(),'new_tokens':len(raw),'stop_reason':part66._stop_reason(raw,tok)},ensure_ascii=False)+'\n')
        sink.flush(); nr.spill_guard(args.device,args.max_gpu_gb)
    (out/'step0.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'run.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'dtype':'bfloat16','batch_size':args.batch,'step0':meta,'conditions':conditions,'requested_rows':len(rows),'new_rows_this_invocation':len(todo),'wall_time_seconds':time.time()-start,'hook_batch_evidence':evidence},indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--out',required=True); ap.add_argument('--step0',required=True); ap.add_argument('--device',default='xpu'); ap.add_argument('--batch',type=int,default=16); ap.add_argument('--max-gpu-gb',type=float,default=20); ap.add_argument('--dry-run',action='store_true'); a=ap.parse_args()
    if a.batch<1: ap.error('--batch must be positive')
    if a.dry_run:
        print(json.dumps({'planned_rows':1944,'conditions':CONDITION_IDS},indent=2)); return
    run(a)
if __name__=='__main__': main()
