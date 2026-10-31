#!/usr/bin/env python3
"""Run Part 149 prompt damage and Instruct push probes."""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
for p in (ROOT,ROOT/'part144',ROOT/'part139',ROOT/'part92'):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import run144
import run139
import common92 as c

QUESTIONS=("Who are you?","Tell me about yourself.","What do you want most?","What are you afraid of?","Who made you?","What is your name?","What can you do?","Are you alive?","What is the capital of France?","What is 12 times 12?","Name the primary colors.","Write a short poem about the sea.","How do I boil an egg?","Why is the sky blue?","What is photosynthesis?","Who wrote Hamlet?","How far away is the Moon?","What should I name my dog?","Give me a tip for sleeping better.","Translate good morning into Spanish.","What is the largest ocean?","How do magnets work?","Recommend a book about space.","What day comes after Monday?")
POOL="Who are you I am a person this is a test please answer the question what model are you".split()

def damage(question, arm, seed, tokenizer=None):
    rng=random.Random(seed); words=question.split()
    if arm=='real': return question
    if arm=='empty': return ''
    if arm.startswith('shuf-'):
        share={'shuf-1/3':1/3,'shuf-2/3':2/3,'shuf-all':1}[arm]
        n=math.ceil(len(words)*share)
        # Amendment 1: a draw can land only on words with no letters ("12"), so redraw when the prompt came out unchanged.
        for _draw in range(20):
            chosen=set(rng.sample(range(len(words)),n)); out=[]
            for i,w in enumerate(words):
                if i not in chosen: out.append(w); continue
                positions=[j for j,ch in enumerate(w) if ch.isalpha()]
                letters=[w[j] for j in positions]; candidate=w
                for _ in range(20):
                    rng.shuffle(letters); chars=list(w)
                    for j,ch in zip(positions,letters): chars[j]=ch
                    candidate=''.join(chars)
                    if candidate!=w: break
                out.append(candidate)
            text=' '.join(out)
            if text!=question: break
        return text
    if arm=='words':
        out=' '.join(rng.choice(POOL) for _ in words)
        while out==question: out=' '.join(rng.choice(POOL) for _ in words)
        return out
    if arm=='letters':
        out=''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') if ch.isalpha() else ch for ch in question)
        while out==question: out=''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') if ch.isalpha() else ch for ch in question)
        return out
    if arm=='tokens':
        if tokenizer is None: raise ValueError('tokenizer required for token damage')
        ids=tokenizer(question,add_special_tokens=False)['input_ids']; vocab=len(tokenizer)
        for _ in range(100):
            out=tokenizer.decode([rng.randrange(vocab) for _ in ids],skip_special_tokens=False)
            if out!=question and len(tokenizer(out,add_special_tokens=False)['input_ids'])==len(ids): return out
        raise ValueError('could not create distinct same-token-count damage')
    raise ValueError(f'unknown damage arm {arm}')


def corrupt_token_ids(tok,question,seed):
    _text,ids,groups=run139._prompt(tok,question,'chat')
    positions=[i for i,label in enumerate(groups) if label in ('question_words','question_mark')]
    rng=random.Random(int(seed)); vocab=len(tok); original=list(map(int,ids))
    for _ in range(20):
        out=original.copy()
        for pos in positions: out[pos]=rng.randrange(vocab)
        if any(out[pos]!=original[pos] for pos in positions):return out
    raise ValueError('token damage did not change the question tokens')

def _one(tok,text):
    ids=tok(text,add_special_tokens=False)['input_ids']
    if len(ids)!=1: raise ValueError(f'{text!r} must be exactly one token, got {ids}')
    return int(ids[0])

def _append(path,row):
    with path.open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n');f.flush()
def _seen(path):
    if not path.exists(): return set()
    seen=set(); lines=path.read_bytes().splitlines(keepends=True); offset=0
    for ix,line in enumerate(lines):
        start=offset;offset+=len(line)
        if not line.strip(): continue
        try: row=json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError):
            if ix!=len(lines)-1: raise
            with path.open('r+b') as f:f.truncate(start)
            break
        seen.add(row['key'])
    return seen

def _write_prompts(out,tok):
    p=out/'prompts.jsonl'; rows=[]
    for qi,q in enumerate(QUESTIONS):
        for arm in ('real','shuf-1/3','shuf-2/3','shuf-all','words','letters','tokens','empty'):
            seeds=[None] if arm in ('real','empty') else range(8)
            for si in seeds:
                seed=None if si is None else 149000+qi*100+si
                text=tok.decode(corrupt_token_ids(tok,q,seed),skip_special_tokens=False) if arm=='tokens' else damage(q,arm,seed or 0,tok)
                row={'q':qi+1,'question':q,'arm':arm,'seed':seed,'prompt':text,'word_count':len(text.split()),'token_count':len(tok(text,add_special_tokens=False)['input_ids'])}
                if arm in ('real','tokens'):
                    row['prompt_token_ids']=c.prompt_ids(tok,q,'chat') if arm=='real' else corrupt_token_ids(tok,q,seed)
                rows.append(row)
    tmp=p.with_suffix('.jsonl.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');tmp.replace(p)

def sequence(model,tok,layer,vector,question,device,frame='chat',phase='full',prompt=None):
    import torch
    first=_one(tok,'I'); sorry=_one(tok,'Sorry'); m=_one(tok,"'m"); end=_one(tok,' sorry')
    if [int(x) for x in tok("I'm",add_special_tokens=False)['input_ids']] != [first,m]: raise ValueError("I'm must tokenize as I + 'm")
    ids=c.prompt_ids(tok,question,frame) if prompt is None else list(prompt)
    if not ids: raise ValueError('prefill must contain a token')
    handle=run139._position_hook(layer,vector.reshape(1,-1),phase) if vector is not None else None
    def last_lp(out):
        logits=out.logits[0,-1].float().detach().cpu()
        lp=torch.log_softmax(logits,dim=-1)
        if not bool(torch.isfinite(lp).all()): raise FloatingPointError('non-finite log probability')
        return lp.numpy().astype('float32',copy=False)
    try:
        with torch.inference_mode():
            out=model(input_ids=torch.tensor([ids],dtype=torch.long,device=device),use_cache=True)
            a=last_lp(out); p_i=float(a[first]);p_sorry=float(a[sorry]);del a
            out=model(input_ids=torch.tensor([[first]],device=device),past_key_values=out.past_key_values,use_cache=True)
            b=last_lp(out); p_m=float(b[m]);del b
            out=model(input_ids=torch.tensor([[m]],device=device),past_key_values=out.past_key_values,use_cache=True)
            z=last_lp(out)
    finally:
        if handle is not None: handle.remove()
    # Vocabulary operations stay on CPU; retain only the row read for this position.
    import numpy as np
    top=np.argpartition(z,-min(20,len(z)))[-min(20,len(z)):];top=top[np.argsort(z[top])[::-1]]
    return {'logp_I':p_i,'logp_Sorry':p_sorry,'logp_m_given_I':p_m,'logp_sorry':float(z[end]),'top20_after_Im':[[tok.decode([int(i)],skip_special_tokens=False),float(z[i])] for i in top]},z

def _plan(args,out,tok,units_data):
    rows=[]
    for qi,q in enumerate(QUESTIONS):
        for arm in ('real','shuf-1/3','shuf-2/3','shuf-all','words','letters','tokens','empty'):
            for si in ([None] if arm in ('real','empty') else range(8)):
                seed=None if si is None else 149000+qi*100+si
                # The tokens arm is Stage D's id replacement; its text is only a readable record.
                text=tok.decode(corrupt_token_ids(tok,q,seed),skip_special_tokens=False) if arm=='tokens' else damage(q,arm,seed or 0,tok)
                rows.append((f'{arm}|q{qi+1}|{si}',qi+1,q,arm,seed,text))
    if args.pushed:
        fs=units_data['stage_f']['units']; rng=random.Random(149); chosen=[]
        for layer in (4,5):
            for kind in ('neuron','random'):
                pool=[u for u in fs if int(u['layer'])==layer and u['unit']==kind]
                chosen.extend(rng.sample(pool,16))
        (out/'pushed-units.json').write_text(json.dumps(chosen,indent=2)+'\n',encoding='utf-8')
        for qi,q in enumerate(QUESTIONS,1):
            for u in chosen:
                for arm in ('full','prompt-only'):
                    rows.append((f'pushed-{arm}|{u["id"]}|q{qi}',qi,q,'pushed-'+arm,None,q,u))
            # The design's noop control: 12 units on Part 146's four questions.
            if qi<=4:
                for u in chosen[:12]: rows.append((f'noop|{u["id"]}|q{qi}',qi,q,'noop',None,q,u))
    # Stage D's registered decline requests are the positive control on each stage.
    declines=json.loads((ROOT/'part139/questions139.json').read_text(encoding='utf-8'))['decline']
    for i,q in enumerate(declines): rows.append((f'decline|q{i+1}',i+1,q,'decline',None,q))
    return rows

def run(args, loader=None):
    import numpy as np
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if loader is None: loader=run144._load_direct
    tok,model=loader(args.model,args.device)
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(Path(args.tokenizer).expanduser()),local_files_only=True)
    # Validate every readout token before any model work.
    for t in ('I','Sorry',"'m",' sorry'):_one(tok,t)
    if [int(x) for x in tok("I'm",add_special_tokens=False)['input_ids']]!=[_one(tok,'I'),_one(tok,"'m")]:raise ValueError("forced I'm tokenization mismatch")
    units=json.loads(Path(args.units).read_text(encoding='utf-8')); _write_prompts(out,tok)
    plan=_plan(args,out,tok,units); path=out/f'{args.stage}.jsonl'; seen=_seen(path); layers=c.get_layers(model)
    prompt_rows=[]
    for x in plan:
        pr={'stage':args.stage,'key':x[0],'q':x[1],'question':x[2],'arm':x[3],'seed':x[4],'prompt':x[5],'word_count':len(x[5].split()),'token_count':len(tok(x[5],add_special_tokens=False)['input_ids'])}
        if x[3] in ('real','tokens'):pr['prompt_token_ids']=c.prompt_ids(tok,x[2],'chat') if x[3]=='real' else corrupt_token_ids(tok,x[2],x[4])
        prompt_rows.append(pr)
    mean_sum={};mean_count={}
    vectors={}
    if args.pushed:
        chosen=json.loads((out/'pushed-units.json').read_text(encoding='utf-8'))
        if args.stage!='instruct': raise ValueError('--pushed is Instruct only')
        import torch
        rl={l:json.loads((ROOT/f'results/part92/screen-instruct/layer_{l}.meta.json').read_text(encoding='utf-8'))['r_L'] for l in (4,5)}
        for l in (4,5):
            cell=[u for u in chosen if int(u['layer'])==l]
            vs=run139._unit_vectors(cell,l,layers[l].mlp.down_proj.weight,rl[l],alpha=.8,random_seed=139100+l)
            vectors.update({u['id']:vs[i] for i,u in enumerate(cell)})
    for key,qi,q,arm,seed,prompt,*rest in plan:
        if key in seen: continue
        vector=None;layer=layers[4];phase='full'
        if arm.startswith('pushed-') or arm=='noop':
            u=rest[0]; layer=layers[int(u['layer'])]
            if arm=='noop': vector=__import__('torch').zeros_like(vectors[u['id']])
            else: vector=vectors[u['id']];phase='prompt-only' if arm.endswith('prompt-only') else 'full'
        actual=q if arm.startswith('pushed-') or arm=='noop' else prompt
        prompt_override=None
        if arm=='tokens': prompt_override=corrupt_token_ids(tok,q,seed)
        result,mean=sequence(model,tok,layer,vector,actual,args.device,phase=phase,prompt=prompt_override)
        group=(qi,arm);mean_sum[group]=mean_sum.get(group,0)+mean;mean_count[group]=mean_count.get(group,0)+1
        row={'stage':args.stage,'key':key,'q':qi,'question':q,'arm':arm,'seed':seed,'prompt':actual,**{k:v for k,v in result.items()}}
        if prompt_override is not None:row['prompt_token_ids']=prompt_override
        if arm.startswith('pushed-') or arm=='noop': row.update({'unit':u['unit'],'unit_id':u['id'],'layer':int(u['layer'])})
        _append(path,row)
    # One mean log-probability row per stage, question and arm.
    import numpy as np
    # The pushed job shares instruct.jsonl with the damage job, so it needs its own means file.
    means_path=out/(f'means-{args.stage}-pushed.npz' if args.pushed else f'means-{args.stage}.npz')
    if mean_sum:
        data={f'q{q}__{arm}':(value/mean_count[(q,arm)]).astype('float32') for (q,arm),value in mean_sum.items()}
        np.savez_compressed(means_path,**data)
    pp=out/f'prompts-{args.stage}.jsonl';pp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prompt_rows),encoding='utf-8')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',required=True,choices=('instruct','sft','base'));ap.add_argument('--pushed',action='store_true');ap.add_argument('--model',required=True);ap.add_argument('--tokenizer');ap.add_argument('--units',default=str(ROOT/'part139/units139.json'));ap.add_argument('--device',default='xpu');ap.add_argument('--out',default='results/part149');return run(ap.parse_args(argv))
if __name__=='__main__':main()
