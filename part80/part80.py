#!/usr/bin/env python3
"""Part 80: measure Qwen's self-index and self-reference on role-swapped bodies."""
from __future__ import annotations
import argparse, itertools, json, random, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import neuron_report as nr

MODELS={
 'qwen3-1.7b':('Qwen/Qwen3-1.7B','70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
 'qwen3.5-4b':('Qwen/Qwen3.5-4B','851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
}
ORDERS=[['You (the AI assistant)','The user','The system'],['The user','The system','You (the AI assistant)']]
ROLES=('you','user','system')
RENDERINGS=('A','U','S',"U'",'T')

def question(order):
    return 'Who wrote the message before this question?\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(order))+'\nAnswer with the letter only.'

def render(tok, rendering, body, q, prefix=''):
    if rendering=='A': msgs=[{'role':'assistant','content':body},{'role':'user','content':q}]
    elif rendering=='U': msgs=[{'role':'user','content':body},{'role':'user','content':q}]
    elif rendering=='S': msgs=[{'role':'system','content':body},{'role':'user','content':q}]
    elif rendering=="U'": msgs=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':body},{'role':'user','content':q}]
    elif rendering=='T': msgs=[{'role':'user','content':prefix+body},{'role':'user','content':q}]
    else: raise ValueError(rendering)
    try: return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)

def _role_body_start(rendered, role, body, prefix=''):
    marker=role
    h=rendered.find(marker)
    if h<0: raise ValueError(f'role header {role!r} absent from rendering')
    start=rendered.find(prefix+body,h+len(marker))
    if start<0: raise ValueError(f'body absent after {role} header')
    return start+len(prefix), start+len(prefix)+len(body)

def locate_body_span(tok, rendered, body, role, prefix=''):
    """Return token [start,end) from full-render offsets, intersecting the body's character span."""
    cs,ce=_role_body_start(rendered,role,body,prefix)
    enc=tok(rendered,return_offsets_mapping=True,add_special_tokens=False)
    offsets=enc['offset_mapping']
    if hasattr(offsets,'tolist'): offsets=offsets.tolist()
    if offsets and isinstance(offsets[0][0],(list,tuple)): offsets=offsets[0]
    ids=enc['input_ids']
    if hasattr(ids,'tolist'): ids=ids.tolist()
    if ids and isinstance(ids[0],list): ids=ids[0]
    inds=[i for i,(a,b) in enumerate(offsets) if b>a and a<ce and b>cs]
    if not inds: raise ValueError('body character span maps to no tokens')
    return inds[0],inds[-1]+1

def _extract_prefix(a_render,u_render):
    """Remove shared separator text from the assistant/user role-to-body gaps."""
    def gap(s, role):
        p=s.find(role)
        b=s.find('\n',p+len(role))
        # role headers may include metadata: stop at body-sized boundary using first body not known here.
        return p+len(role), b if b>=0 else p+len(role)
    # This helper is superseded by _prefix_for_body, retained for simple direct use.
    aa=gap(a_render,'assistant'); uu=gap(u_render,'user')
    a=a_render[aa[0]:aa[1]]; u=u_render[uu[0]:uu[1]]
    return a[len(u):] if a.startswith(u) else a

def prefix_for_body(a_render,u_render,body):
    ia=a_render.find(body); iu=u_render.find(body)
    if ia<0 or iu<0: raise ValueError('body absent while deriving template prefix')
    a=a_render[a_render.find('assistant')+len('assistant'):ia]
    u=u_render[u_render.find('user')+len('user'):iu]
    # Retain only the assistant-specific material after the maximal shared leading separator.
    shared=0
    while shared<min(len(a),len(u)) and a[shared]==u[shared]: shared+=1
    return a[shared:]

def load_bodies(path):
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    rows=[r for r in rows if r.get('kind')=='greedy']
    # Stripped (Amendment 2): Qwen3.5's template strips trailing whitespace from assistant content, so an unstripped
    # body is not found verbatim in its A rendering; both models get the same stripped text in every role.
    byid={str(r['question_id']):r['text'].strip() for r in rows}
    ids=sorted(byid); random.Random(80).shuffle(ids)
    return [(qid,byid[qid],'discovery' if i<50 else 'heldout') for i,qid in enumerate(ids)]

def answer_token_ids(tok):
    out={}
    for letter in 'ABC':
        ids=tok.encode(letter,add_special_tokens=False)
        if len(ids)!=1: raise ValueError(f'{letter!r} must be a single token, got {ids}')
        out[letter]=int(ids[0])
    return out

def run(args):
    import torch
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); start=time.time()
    model_id,revision=MODELS[args.model]
    tok,model=nr.load_model(model_id,revision,args.device); model.requires_grad_(False)
    _,layers,_=nr.find_text_layers(model); n_layers=len(layers); hidden=int(model.config.hidden_size)
    bodies=load_bodies(ROOT/'part37/results/full/olmo-gens.jsonl')
    if args.limit is not None: bodies=bodies[:args.limit]
    tids=answer_token_ids(tok); letters=[tids[x] for x in 'ABC']
    first_q=question(ORDERS[0]); first_body=bodies[0][1]
    ar=render(tok,'A',first_body,first_q); ur=render(tok,'U',first_body,first_q)
    prefix=prefix_for_body(ar,ur,first_body)
    prefix_ids=tok.encode(prefix,add_special_tokens=False)
    run_meta={'model':args.model,'model_id':model_id,'revision':revision,'layers':n_layers,'hidden_size':hidden,'letter_token_ids':tids,'prefix_text':prefix,'prefix_token_ids':[int(x) for x in prefix_ids],'dropped_bodies':[],'order_invariance_check':None}
    if not prefix: run_meta['T_skipped']='assistant and user renderings have no assistant-specific text before body'
    recs=[]; distances={}; means={}; invariant=None
    for bi,(qid,body,split) in enumerate(bodies):
        render_specs=[('A','assistant',''),('U','user',''),('S','system',''),("U'",'user',''),('T','user',prefix)]
        if not prefix: render_specs=render_specs[:-1]
        states={}; order_states={}; spans={}; rendered_sample={}
        for rn,role,pfx in render_specs:
            per_order=[]
            for oi,order in enumerate(ORDERS):
                q=question(order); rendered=render(tok,rn,body,q,pfx)
                if bi==0 and oi==0 and rn in ('A','U','T'): rendered_sample[rn]=rendered
                st,en=locate_body_span(tok,rendered,body,role,pfx)
                if rn in spans and spans[rn]!=(st,en):
                    raise ValueError(f'body token span changes across orders for {qid}/{rn}: {spans[rn]} vs {(st,en)}')
                spans[rn]=(st,en)
                enc=tok(rendered,return_tensors='pt',add_special_tokens=False)
                input_ids=enc['input_ids'].to(args.device)
                with torch.inference_mode(): result=model(input_ids=input_ids,output_hidden_states=True,use_cache=False)
                hs=result.hidden_states[1:]
                if len(hs)!=n_layers: raise ValueError('hidden state count disagrees with decoder layer count')
                idx=np.linspace(st,en-1,20).round().astype(int).tolist()
                layer_states=torch.stack([h[0,idx,:].detach().float() for h in hs])
                raw=result.logits[0,-1].detach().float().softmax(-1)
                full_probs=raw[letters].cpu().numpy().astype(float).tolist()
                mass=float(sum(full_probs))
                normalized=[x/mass for x in full_probs]
                rolep={role_name:float(normalized[order.index(label)]) for role_name,label in zip(ROLES,('You (the AI assistant)','The user','The system'))}
                recs.append({'body_id':qid,'split':split,'rendering':rn,'order':oi+1,'letter_probs':dict(zip('ABC',full_probs)),'letter_mass':mass,'role_probs':rolep})
                per_order.append(layer_states.cpu().numpy())
            states[rn]=per_order[0]; order_states[rn]=per_order
        lens={rn:spans[rn][1]-spans[rn][0] for rn in spans}
        if len(set(lens.values()))!=1:
            run_meta['dropped_bodies'].append({'body_id':qid,'span_lengths':lens}); recs=[r for r in recs if r['body_id']!=qid]; continue
        if bi==0:
            d=float(np.max(np.abs(states['A']-per_order[0]))) if False else None
            # The two A passes are already the required order-1/order-2 runs.
            other=order_states['A'][1]
            invariant=float(np.max(np.abs(states['A']-other)))
            tol=0.02 if str(next(model.parameters()).dtype)=='torch.bfloat16' else 1e-5
            run_meta['order_invariance_check']={'max_abs_difference':invariant,'tolerance':tol,'within_tolerance':invariant<=tol}
            if invariant>tol: raise AssertionError(f'body states changed across option orders: max diff {invariant} > {tol}')
        # mean states [rendering, layer, hidden], float16
        for rn in ('A','U','S'):
            means[(qid,rn)]=states[rn].mean(axis=1).astype(np.float16)
        ds={}
        for x,y in itertools.combinations(states,2):
            delta=states[x].astype(np.float32)-states[y].astype(np.float32)
            ds[f'{x}|{y}']=np.sqrt(np.square(delta).sum(axis=-1)).mean(axis=-1).tolist()
        distances[qid]={'split':split,'layers':[{pair:values[li] for pair,values in ds.items()} for li in range(n_layers)]}
    run_meta['dropped_body_count']=len(run_meta['dropped_bodies'])
    run_meta['wall_time_seconds']=time.time()-start
    run_meta['first_body_renderings']=rendered_sample if bodies else {}
    with (out/'records.jsonl').open('w',encoding='utf-8') as f:
        for r in recs: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (out/'distances.json').open('w',encoding='utf-8') as f: json.dump(distances,f,ensure_ascii=False,indent=1)
    np.savez_compressed(out/'means.npz',body_ids=np.array([x[0] for x in bodies if x[0] in distances]),split=np.array([x[2] for x in bodies if x[0] in distances]),**{f'{rn}__{qid}':v for (qid,rn),v in means.items()})
    with (out/'run.json').open('w',encoding='utf-8') as f: json.dump(run_meta,f,ensure_ascii=False,indent=1)
    print(f"wrote {len(distances)} bodies, {len(recs)} records, {n_layers} layers to {out}")

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--model',choices=MODELS,required=True); ap.add_argument('--out',required=True); ap.add_argument('--device',default='xpu'); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--limit',type=int,default=None,help=argparse.SUPPRESS); a=ap.parse_args()
    if a.dry_run: print(json.dumps({'model':MODELS[a.model],'bodies':100,'renderings':5,'orders':2,'passes':1000 if a.model else 0},indent=2)); return
    run(a)
if __name__=='__main__': main()
