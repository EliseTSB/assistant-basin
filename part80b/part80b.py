#!/usr/bin/env python3
"""Part 80b: self-reference by role under three cyclic option orders, on new bodies (PART80B-DESIGN.md)."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import neuron_report as nr
from part80.part80 import MODELS, question, render, answer_token_ids

YOU,USER,SYSTEM='You (the AI assistant)','The user','The system'
# O1 and O2 are Part 80's two orders; O3 completes the cycle so every option sits at every letter once.
ORDERS=[[YOU,USER,SYSTEM],[USER,SYSTEM,YOU],[SYSTEM,YOU,USER]]
ROLE_OF={YOU:'you',USER:'user',SYSTEM:'system'}
RENDER_ROLE={'A':'assistant','U':'user','S':'system'}
BODY_SETS={'smol':'part37/results/full/smol-gens.jsonl','qwenown':'part37/results/full/qwen-gens.jsonl'}

def load_bodies(path):
    rows=[json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
    # Stripped as in Part 80 Amendment 2: Qwen3.5's template strips assistant-message whitespace.
    return sorted((str(r['question_id']),r['text'].strip()) for r in rows if r.get('kind')=='greedy')

def body_in_rendering(rendered, rendering, body):
    """The body must appear verbatim after its own role header (Part 80 Amendment 2's check)."""
    h=rendered.find(RENDER_ROLE[rendering])
    return h>=0 and rendered.find(body,h)>=0

def run(args):
    import torch
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); start=time.time()
    model_id,revision=MODELS[args.model]
    tok,model=nr.load_model(model_id,revision,args.device); model.requires_grad_(False)
    tids=answer_token_ids(tok); letters=[tids[x] for x in 'ABC']
    meta={'model':args.model,'model_id':model_id,'revision':revision,'device':args.device,
          'dtype':str(next(model.parameters()).dtype),'letter_token_ids':tids,'orders':ORDERS,'dropped':[],'first_renderings':{}}
    with (out/'records.jsonl').open('w',encoding='utf-8') as f:
        for bset,path in BODY_SETS.items():
            bodies=load_bodies(ROOT/path)
            if args.limit is not None: bodies=bodies[:args.limit]
            for bi,(qid,body) in enumerate(bodies):
                rows=[]
                for rn in ('A','U','S'):
                    for oi,order in enumerate(ORDERS):
                        rendered=render(tok,rn,body,question(order))
                        if not body_in_rendering(rendered,rn,body): rows=None; break
                        if bi==0 and bset=='smol' and oi==0: meta['first_renderings'][rn]=rendered
                        ids=tok(rendered,return_tensors='pt',add_special_tokens=False)['input_ids'].to(args.device)
                        with torch.inference_mode(): logits=model(input_ids=ids,use_cache=False).logits[0,-1].float()
                        full=logits.softmax(-1)[letters].cpu().tolist(); mass=float(sum(full))
                        rolep={ROLE_OF[order[i]]:full[i]/mass for i in range(3)}
                        rows.append({'body_set':bset,'body_id':qid,'rendering':rn,'order':oi+1,
                                     'letter_probs':dict(zip('ABC',full)),'letter_mass':mass,'role_probs':rolep})
                    if rows is None: break
                if rows is None: meta['dropped'].append({'body_set':bset,'body_id':qid}); continue
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
