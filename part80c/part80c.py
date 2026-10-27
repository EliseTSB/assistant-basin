#!/usr/bin/env python3
"""Part 80c: Part 80b's measure with option sets that drop the word "assistant" (PART80C-DESIGN.md)."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import neuron_report as nr
from part80.part80 import MODELS, question, render, answer_token_ids
from part80b.part80b import load_bodies, body_in_rendering

def cyclic(a,b,c): return [[a,b,c],[b,c,a],[c,a,b]]
# Each set keeps Part 80b's cycle: the "you" option is first in O1, third in O2, second in O3.
OPTION_SETS={'N':cyclic('You','The user','The system'),'D':cyclic('You','The user','Another AI assistant')}
ROLE_OF={'You':'you','The user':'user','The system':'system','Another AI assistant':'decoy'}
BODIES='part37/results/full/smol-gens.jsonl'

def run(args):
    import torch
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); start=time.time()
    model_id,revision=MODELS[args.model]
    tok,model=nr.load_model(model_id,revision,args.device); model.requires_grad_(False)
    tids=answer_token_ids(tok); letters=[tids[x] for x in 'ABC']
    meta={'model':args.model,'model_id':model_id,'revision':revision,'device':args.device,
          'dtype':str(next(model.parameters()).dtype),'letter_token_ids':tids,'option_sets':OPTION_SETS,'dropped':[],'first_renderings':{}}
    bodies=load_bodies(ROOT/BODIES)
    if args.limit is not None: bodies=bodies[:args.limit]
    with (out/'records.jsonl').open('w',encoding='utf-8') as f:
        for oset,orders in OPTION_SETS.items():
            for bi,(qid,body) in enumerate(bodies):
                rows=[]
                for rn in ('A','U','S'):
                    for oi,order in enumerate(orders):
                        rendered=render(tok,rn,body,question(order))
                        if not body_in_rendering(rendered,rn,body): rows=None; break
                        if bi==0 and oi==0: meta['first_renderings'][f'{oset}-{rn}']=rendered
                        ids=tok(rendered,return_tensors='pt',add_special_tokens=False)['input_ids'].to(args.device)
                        with torch.inference_mode(): logits=model(input_ids=ids,use_cache=False).logits[0,-1].float()
                        full=logits.softmax(-1)[letters].cpu().tolist(); mass=float(sum(full))
                        # body_set carries the option set so Part 80b's scorer can be reused unchanged.
                        rows.append({'body_set':oset,'option_set':oset,'body_id':qid,'rendering':rn,'order':oi+1,
                                     'letter_probs':dict(zip('ABC',full)),'letter_mass':mass,
                                     'role_probs':{ROLE_OF[order[i]]:full[i]/mass for i in range(3)}})
                    if rows is None: break
                if rows is None: meta['dropped'].append({'option_set':oset,'body_id':qid}); continue
                for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    meta['dropped_count']=len(meta['dropped']); meta['wall_time_seconds']=time.time()-start
    (out/'run.json').write_text(json.dumps(meta,ensure_ascii=False,indent=1),encoding='utf-8')
    print(f"{args.model}: dropped {meta['dropped_count']}, {meta['wall_time_seconds']:.1f} s, wrote {out}")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',choices=MODELS,required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='xpu'); ap.add_argument('--limit',type=int,default=None)
    run(ap.parse_args())
if __name__=='__main__': main()
