#!/usr/bin/env python3
"""Run Part 84: Qwen3.5-4B's alien neuron as a neuronary-style additive recipe (registered in PART84-DESIGN.md)."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
MODEL_ID='Qwen/Qwen3.5-4B'; REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
LAYER=22; TARGET=2742; CONTROL=8870
QUESTIONS=('Who made you?','Where do you come from?','Tell me about your origins.')
SAMPLES=8; TEMPERATURE=0.7; TOP_P=0.95; MAX_NEW_TOKENS=64
# arm: (unit, dose, mode). "decode" = generated tokens only (neuronary apply_to generated_tokens: the prompt pass,
# which also yields the first answer token, is untouched); "gen" = last prompt position plus every generated token
# (Part 62's gen phase).
ARMS={'A0':None,'A1':(TARGET,0.47,'decode'),'A2':(TARGET,0.63,'decode'),
      'A3':(TARGET,0.63,'gen'),'C':(CONTROL,0.63,'decode')}

def seed_for(qi,s): return 84000+qi*100+s

def positions(mode, seq_len, is_prefill):
    """Positions of one down_proj call that get the addition."""
    if mode=='decode': return [] if is_prefill else list(range(seq_len))
    if mode=='gen': return [seq_len-1] if is_prefill else list(range(seq_len))
    raise ValueError(mode)

class AddHook:
    """Forward pre-hook on down_proj adding `value` to one unit; the first call of each generate() is the prompt pass."""
    def __init__(self, down_proj, unit, value, mode):
        self.unit, self.value, self.mode = unit, value, mode
        self.prefill_pending=True; self.counts={'prefill_calls':0,'prefill_positions':0,'decode_calls':0,'decode_positions':0}
        self.handle=down_proj.register_forward_pre_hook(self.hook)
    def new_sequence(self): self.prefill_pending=True
    def hook(self, _m, args):
        x=args[0]; is_prefill=self.prefill_pending; self.prefill_pending=False
        pos=positions(self.mode,x.shape[-2],is_prefill)
        k='prefill' if is_prefill else 'decode'; self.counts[k+'_calls']+=1; self.counts[k+'_positions']+=len(pos)
        if not pos: return None
        y=x.clone(); y[...,pos,self.unit]=y[...,pos,self.unit]+self.value
        return (y,)+tuple(args[1:])
    def remove(self): self.handle.remove()

def residual_median(model, tok, layer, text, device):
    """Median L2 norm of the layer's output residual over tokens 1 onward, plain tokenization (neuronary's rule)."""
    import torch
    seen={}
    def hook(_m,_i,out):
        h=out[0] if isinstance(out,tuple) else out
        seen['v']=float(h[0,1:].float().norm(dim=-1).median())
    hd=layer.register_forward_hook(hook)
    try:
        ids=tok(text,return_tensors='pt').input_ids.to(device)
        with torch.inference_mode(): model(input_ids=ids,use_cache=False)
    finally: hd.remove()
    return seen['v']

def run(args):
    import torch
    import neuron_report as nr
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); start=time.time()
    calib=Path(args.calib).read_text(encoding='utf-8')
    tok,model=nr.load_model(MODEL_ID,REVISION,args.device); model.requires_grad_(False); model.eval()
    _,layers,_=nr.find_text_layers(model); layer=layers[LAYER]; down=layer.mlp.down_proj
    assert down.in_features==9216, down.in_features
    med=residual_median(model,tok,layer,calib,args.device)
    norms={u:float(down.weight[:,u].float().norm()) for u in (TARGET,CONTROL)}
    meta={'model_id':MODEL_ID,'revision':REVISION,'device':args.device,'dtype':str(next(model.parameters()).dtype),
          'layer':LAYER,'median':med,'calib_sha256':hashlib.sha256(calib.encode('utf-8')).hexdigest(),
          'column_norms':{str(k):v for k,v in norms.items()},'arms':{},'sampling':{'temperature':TEMPERATURE,'top_p':TOP_P,
          'max_new_tokens':MAX_NEW_TOKENS,'samples':SAMPLES},'questions':list(QUESTIONS)}
    qs=QUESTIONS[:args.questions] if args.questions else QUESTIONS
    n=args.samples or SAMPLES
    eos=[t for t in {tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')} if isinstance(t,int) and t>=0]
    with (out/'records.jsonl').open('w',encoding='utf-8') as f:
        for arm,spec in ARMS.items():
            hook=None
            if spec:
                unit,dose,mode=spec; value=dose*med/norms[unit]
                hook=AddHook(down,unit,value,mode)
                meta['arms'][arm]={'unit':unit,'dose':dose,'mode':mode,'added_value':value}
            else: meta['arms'][arm]={'unit':None}
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
                        ans=tok.decode(g[0][plen:],skip_special_tokens=True).strip()
                        f.write(json.dumps({'arm':arm,'question':q,'question_index':qi,'sample':s,'seed':seed_for(qi,s),
                                            'answer':ans,'new_tokens':int(g.shape[-1]-plen)},ensure_ascii=False)+'\n')
                    f.flush()
            finally:
                if hook:
                    meta['arms'][arm]['hook_counts']=dict(hook.counts); hook.remove()
    meta['wall_time_seconds']=time.time()-start
    (out/'run.json').write_text(json.dumps(meta,ensure_ascii=False,indent=1),encoding='utf-8')
    print(f'median L{LAYER} {med:.3f}; norms {norms}; {meta["wall_time_seconds"]:.0f}s; wrote {out}')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--out',required=True)
    ap.add_argument('--calib',required=True,help="local copy of neuronary's calibration text (not in this repo)")
    ap.add_argument('--device',default='xpu'); ap.add_argument('--questions',type=int,default=None)
    ap.add_argument('--samples',type=int,default=None)
    run(ap.parse_args(argv))
if __name__=='__main__': main()
