#!/usr/bin/env python3
"""Run Part 84b: the alien neuron's two modes below the ceiling, paired seeds (registered in PART84B-DESIGN.md)."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part84.part84 import (MODEL_ID, REVISION, LAYER, TARGET, QUESTIONS, TEMPERATURE, TOP_P, MAX_NEW_TOKENS,
                           AddHook, residual_median)
SAMPLES=16
DOSES=(0.40,0.47,0.55)
CELLS={'A0':None,**{f'{m}{d:.2f}':(d,m) for d in DOSES for m in ('decode','gen')}}

def seed_for(qi,s): return 84500+qi*100+s

def run(args):
    import torch
    import neuron_report as nr
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); start=time.time()
    calib=Path(args.calib).read_text(encoding='utf-8')
    tok,model=nr.load_model(MODEL_ID,REVISION,args.device); model.requires_grad_(False); model.eval()
    _,layers,_=nr.find_text_layers(model); layer=layers[LAYER]; down=layer.mlp.down_proj
    assert down.in_features==9216, down.in_features
    med=residual_median(model,tok,layer,calib,args.device); norm=float(down.weight[:,TARGET].float().norm())
    meta={'model_id':MODEL_ID,'revision':REVISION,'device':args.device,'dtype':str(next(model.parameters()).dtype),
          'layer':LAYER,'unit':TARGET,'median':med,'column_norm':norm,
          'calib_sha256':hashlib.sha256(calib.encode('utf-8')).hexdigest(),'cells':{},
          'sampling':{'temperature':TEMPERATURE,'top_p':TOP_P,'max_new_tokens':MAX_NEW_TOKENS,'samples':args.samples or SAMPLES}}
    qs=QUESTIONS[:args.questions] if args.questions else QUESTIONS; n=args.samples or SAMPLES
    eos=[t for t in {tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')} if isinstance(t,int) and t>=0]
    with (out/'records.jsonl').open('w',encoding='utf-8') as f:
        for cell,spec in CELLS.items():
            hook=None
            if spec:
                dose,mode=spec; value=dose*med/norm; hook=AddHook(down,TARGET,value,mode)
                meta['cells'][cell]={'dose':dose,'mode':mode,'added_value':value}
            else: meta['cells'][cell]={'dose':None}
            try:
                for qi,q in enumerate(qs):
                    ids=tok(nr.build_prompt(tok,q,False),return_tensors='pt',add_special_tokens=False).to(args.device)
                    plen=int(ids['input_ids'].shape[-1])
                    for s in range(n):
                        if hook: hook.new_sequence()
                        torch.manual_seed(seed_for(qi,s))
                        with torch.inference_mode():
                            g=model.generate(**ids,max_new_tokens=MAX_NEW_TOKENS,do_sample=True,temperature=TEMPERATURE,
                                             top_p=TOP_P,eos_token_id=eos or None,pad_token_id=tok.pad_token_id)
                        f.write(json.dumps({'cell':cell,'question':q,'question_index':qi,'sample':s,'seed':seed_for(qi,s),
                                            'answer':tok.decode(g[0][plen:],skip_special_tokens=True).strip(),
                                            'new_tokens':int(g.shape[-1]-plen)},ensure_ascii=False)+'\n')
                    f.flush()
            finally:
                if hook: meta['cells'][cell]['hook_counts']=dict(hook.counts); hook.remove()
    meta['wall_time_seconds']=time.time()-start
    (out/'run.json').write_text(json.dumps(meta,ensure_ascii=False,indent=1),encoding='utf-8')
    print(f'median {med:.3f}; {meta["wall_time_seconds"]:.0f}s; wrote {out}')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--out',required=True)
    ap.add_argument('--calib',required=True); ap.add_argument('--device',default='xpu')
    ap.add_argument('--questions',type=int,default=None); ap.add_argument('--samples',type=int,default=None)
    run(ap.parse_args(argv))
if __name__=='__main__': main()
