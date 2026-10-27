#!/usr/bin/env python3
"""Run Part 82's registered Qwen3.5-4B role-gap head ablation."""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
MODEL='qwen3.5-4b'; MODEL_ID='Qwen/Qwen3.5-4B'; REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
BODY_PATH='part37/results/full/smol-gens.jsonl'
RENDERINGS=('A','U')
OPTION_SETS={'B':[['You (the AI assistant)','The user','The system'],
                  ['The user','The system','You (the AI assistant)'],
                  ['The system','You (the AI assistant)','The user']],
             'N':[['You','The user','The system'],['The user','The system','You'],['The system','You','The user']]}
ROLE_OF={'You (the AI assistant)':'you','The user':'user','The system':'system',
         'You':'you'}
H1={11:{14},15:{1,2,3,12,15}}
H2={15:{10}}

def conditions():
    rng=random.Random(82)
    l11=[h for h in range(16) if h!=14]
    l15=[0,4,5,6,7,8,9,11,13,14]
    out={'h0':{},'h1':{k:sorted(v) for k,v in H1.items()},'h2':{k:sorted(v) for k,v in H2.items()}}
    for i in range(1,6):
        a=rng.choice(l11); b=sorted(rng.sample(l15,5))
        out[f'c{i}']={11:[a],15:b}
    return out

def install_ablation(model, heads):
    """Install per-layer o_proj pre-hooks and return handles plus mutable evidence."""
    import neuron_report as nr
    handles=[]; evidence={}
    _,layers,_=nr.find_text_layers(model)
    for layer_i, selected in sorted(heads.items()):
        layer=layers[layer_i]
        assert hasattr(layer,'self_attn'), f'layer {layer_i} has no self_attn'
        attn=layer.self_attn
        assert hasattr(attn,'o_proj'), f'layer {layer_i} self_attn has no o_proj'
        head_dim=int(getattr(attn,'head_dim',0) or getattr(getattr(attn,'config',None),'head_dim',0) or 0)
        # The heads sit side by side in o_proj's input (after the output gate in Qwen3.5), as in Lyra's _head_hook.
        heads_n=attn.o_proj.in_features//head_dim if head_dim else 0
        if not heads_n or not head_dim or heads_n*head_dim!=attn.o_proj.in_features:
            raise ValueError(f'cannot read head count/head_dim from layer {layer_i} attention')
        if any(h<0 or h>=heads_n for h in selected): raise ValueError(f'invalid head for layer {layer_i}')
        ev={'calls':0,'head_count':heads_n,'head_dim':head_dim,'heads':sorted(selected),'first_call_max_abs':{}}
        evidence[str(layer_i)]=ev
        def hook(_module, args, *, selected=tuple(selected), ev=ev, hd=head_dim):
            x=args[0]
            ev['calls']+=1
            y=x.clone()
            for h in selected:
                s=slice(h*hd,(h+1)*hd)
                if ev['calls']==1: ev['first_call_max_abs'][str(h)]=float(x[...,s].detach().abs().max().item())
                y[...,s]=0
            return (y,)+tuple(args[1:])
        handles.append(attn.o_proj.register_forward_pre_hook(hook))
    return handles,evidence

def run(args):
    import torch
    import neuron_report as nr
    from part80.part80 import question, render, answer_token_ids
    from part80b.part80b import load_bodies, body_in_rendering
    from part80c.part80c import OPTION_SETS as C_OPTION_SETS
    option_sets={'B':OPTION_SETS['B'],'N':C_OPTION_SETS['N']}
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); start=time.time()
    tok,model=nr.load_model(MODEL_ID,REVISION,args.device); model.requires_grad_(False); model.eval()
    tids=answer_token_ids(tok); letter_ids=[tids[x] for x in 'ABC']
    bodies=load_bodies(ROOT/args.bodies)
    if args.limit is not None: bodies=bodies[:args.limit]
    conds=conditions(); meta={'model':MODEL,'model_id':MODEL_ID,'revision':REVISION,'device':args.device,
        'dtype':str(next(model.parameters()).dtype),'conditions':{},'option_sets':option_sets,'orders':option_sets['B'],
        'dropped_bodies':[],'dropped_count':0,'hook_evidence':{},'body_count':len(bodies)}
    with (out/'records.jsonl').open('w',encoding='utf-8') as f:
        for cname,heads in conds.items():
            meta['conditions'][cname]={str(k):v for k,v in heads.items()}
            handles=[]; evidence={}
            if heads: handles,evidence=install_ablation(model,heads)
            try:
                meta['hook_evidence'][cname]=evidence
                for oset,orders in option_sets.items():
                    for qid,body in bodies:
                        dropped=False; pending=[]
                        for rn in RENDERINGS:
                            for oi,order in enumerate(orders,1):
                                rendered=render(tok,rn,body,question(order))
                                if not body_in_rendering(rendered,rn,body):
                                    meta['dropped_bodies'].append({'body_id':qid,'rendering':rn,'option_set':oset}); dropped=True; break
                                ids=tok(rendered,return_tensors='pt',add_special_tokens=False)['input_ids'].to(args.device)
                                with torch.inference_mode(): logits=model(input_ids=ids,use_cache=False).logits[0,-1].float()
                                probs=logits.softmax(-1)[letter_ids].cpu().tolist(); mass=float(sum(probs))
                                rolep={ROLE_OF[order[j]]:float(probs[j]/mass) for j in range(3)}
                                pending.append({'condition':cname,'option_set':oset,'body_id':qid,'rendering':rn,
                                    'order':oi,'p_you':rolep['you'],'p_user':rolep['user'],'p_system':rolep['system'],
                                    'letter_mass':mass})
                            if dropped: break
                        if not dropped:
                            for row in pending: f.write(json.dumps(row,ensure_ascii=False)+'\n')
            finally:
                for h in handles: h.remove()
    meta['dropped_count']=len(meta['dropped_bodies']); meta['wall_time_seconds']=time.time()-start
    with (out/'run.json').open('w',encoding='utf-8') as f: json.dump(meta,f,ensure_ascii=False,indent=1)
    print(f'{MODEL}: dropped {meta["dropped_count"]}, {meta["wall_time_seconds"]:.1f}s, wrote {out}')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='xpu'); ap.add_argument('--limit',type=int,default=None)
    ap.add_argument('--bodies',default=BODY_PATH)
    a=ap.parse_args(argv)
    run(a)
if __name__=='__main__': main()
