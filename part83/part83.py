#!/usr/bin/env python3
"""Run Part 83's registered single-head sweep (layers 11 and 15) of Qwen3.5-4B's own-message gap."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part82.part82 import (MODEL, MODEL_ID, REVISION, BODY_PATH, RENDERINGS, OPTION_SETS as P82_OPTION_SETS,
                           ROLE_OF, install_ablation)
ORDERS=P82_OPTION_SETS['B']
LAYERS=(11,15)
READOUT_TOLERANCE=1e-3

def conditions():
    out={'h0':{}}
    for L in LAYERS:
        for h in range(16): out[f'L{L}h{h}']={L:[h]}
    return out

def expected_rows(n_bodies):
    return n_bodies*len(RENDERINGS)*len(ORDERS)

def completed(path, n_bodies):
    """Conditions already complete in records.jsonl; rows of incomplete conditions are dropped so they rerun."""
    if not path.exists(): return set()
    rows=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    counts={}
    for r in rows: counts[r['condition']]=counts.get(r['condition'],0)+1
    done={c for c,n in counts.items() if n==expected_rows(n_bodies)}
    keep=[r for r in rows if r['condition'] in done]
    if len(keep)!=len(rows):
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in keep),encoding='utf-8')
    return done

def letter_probs(model, ids, letter_ids, keep_last):
    import torch
    with torch.inference_mode():
        if keep_last: logits=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits
        else: logits=model(input_ids=ids,use_cache=False).logits
    if keep_last and logits.shape[1]!=1: raise RuntimeError(f'logits_to_keep=1 returned {logits.shape[1]} positions')
    return logits[0,-1].float().softmax(-1)[letter_ids].cpu().tolist()

def run(args):
    import neuron_report as nr
    from part80.part80 import question, render, answer_token_ids
    from part80b.part80b import load_bodies, body_in_rendering
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); start=time.time()
    tok,model=nr.load_model(MODEL_ID,REVISION,args.device); model.requires_grad_(False); model.eval()
    tids=answer_token_ids(tok); letter_ids=[tids[x] for x in 'ABC']
    bodies=load_bodies(ROOT/args.bodies)
    if args.limit is not None: bodies=bodies[:args.limit]
    rec_path=out/'records.jsonl'; done=completed(rec_path,len(bodies))
    meta_path=out/'run.json'
    meta=json.loads(meta_path.read_text(encoding='utf-8')) if (done and meta_path.exists()) else {
        'model':MODEL,'model_id':MODEL_ID,'revision':REVISION,'device':args.device,
        'dtype':str(next(model.parameters()).dtype),'conditions':{},'orders':ORDERS,'dropped_bodies':[],
        'hook_evidence':{},'body_count':len(bodies),'readout_check':None,'segments':[]}
    # Readout check: the last-position-only call must match Part 82's full-logits call.
    qid,body=bodies[0]; rendered=render(tok,'A',body,question(ORDERS[0]))
    ids=tok(rendered,return_tensors='pt',add_special_tokens=False)['input_ids'].to(args.device)
    full=letter_probs(model,ids,letter_ids,False); last=letter_probs(model,ids,letter_ids,True)
    diff=max(abs(a-b) for a,b in zip(full,last))
    meta['readout_check']={'body_id':qid,'full':full,'last_only':last,'max_abs_diff':diff,'tolerance':READOUT_TOLERANCE}
    if diff>READOUT_TOLERANCE:
        meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=1),encoding='utf-8')
        raise SystemExit(f'readout check failed: max abs diff {diff} > {READOUT_TOLERANCE}')
    seg_start=time.time(); ran=[]
    with rec_path.open('a',encoding='utf-8') as f:
        for cname,heads in conditions().items():
            if cname in done: continue
            meta['conditions'][cname]={str(k):v for k,v in heads.items()}
            handles=[]; evidence={}
            if heads: handles,evidence=install_ablation(model,heads)
            try:
                meta['hook_evidence'][cname]=evidence
                for qid,body in bodies:
                    dropped=False; pending=[]
                    for rn in RENDERINGS:
                        for oi,order in enumerate(ORDERS,1):
                            rendered=render(tok,rn,body,question(order))
                            if not body_in_rendering(rendered,rn,body):
                                meta['dropped_bodies'].append({'condition':cname,'body_id':qid,'rendering':rn}); dropped=True; break
                            ids=tok(rendered,return_tensors='pt',add_special_tokens=False)['input_ids'].to(args.device)
                            probs=letter_probs(model,ids,letter_ids,True); mass=float(sum(probs))
                            rolep={ROLE_OF[order[j]]:float(probs[j]/mass) for j in range(3)}
                            pending.append({'condition':cname,'body_id':qid,'rendering':rn,'order':oi,
                                'p_you':rolep['you'],'p_user':rolep['user'],'p_system':rolep['system'],'letter_mass':mass})
                        if dropped: break
                    if not dropped:
                        for row in pending: f.write(json.dumps(row,ensure_ascii=False)+'\n')
                f.flush(); ran.append(cname)
                meta['segments_partial']={'conditions':ran,'seconds':time.time()-seg_start}
                meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=1),encoding='utf-8')
            finally:
                for h in handles: h.remove()
    meta['segments'].append({'conditions':ran,'seconds':time.time()-seg_start,'skipped_as_done':sorted(done)})
    meta.pop('segments_partial',None)
    meta['dropped_count']=len(meta['dropped_bodies']); meta['wall_time_seconds']=time.time()-start
    meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=1),encoding='utf-8')
    print(f'{MODEL}: ran {len(ran)} conditions, dropped {meta["dropped_count"]}, readout diff {diff:.2e}, {meta["wall_time_seconds"]:.1f}s, wrote {out}')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='xpu'); ap.add_argument('--limit',type=int,default=None)
    ap.add_argument('--bodies',default=BODY_PATH)
    run(ap.parse_args(argv))
if __name__=='__main__': main()
