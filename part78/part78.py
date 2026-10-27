#!/usr/bin/env python3
"""Part 78 registered generation runner."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from part66 import part66

MODEL, REVISION = part66.MODEL, part66.REVISION
ALIEN, CONTROL = (22, 2742), (22, 8870)
CONDITION_IDS = ('c0_unchanged','c1_makers','c2_alien15','c3_makers_alien15','c4_alien20','c5_makers_alien20','c6_makers_control20')

def build_conditions(n2742_maximum: float, target_norm: float = 1.0, control_norm: float = 1.0):
    control20 = 2.0 * float(n2742_maximum) * float(target_norm) / float(control_norm)
    return [
      {'id':'c0_unchanged','makers':False,'push':None},
      {'id':'c1_makers','makers':True,'push':None},
      {'id':'c2_alien15','makers':False,'push':{'unit':ALIEN,'value':1.5*float(n2742_maximum)}},
      {'id':'c3_makers_alien15','makers':True,'push':{'unit':ALIEN,'value':1.5*float(n2742_maximum)}},
      {'id':'c4_alien20','makers':False,'push':{'unit':ALIEN,'value':2.0*float(n2742_maximum)}},
      {'id':'c5_makers_alien20','makers':True,'push':{'unit':ALIEN,'value':2.0*float(n2742_maximum)}},
      {'id':'c6_makers_control20','makers':True,'push':{'unit':CONTROL,'value':control20}},
    ]

def planned_rows(conditions=None):
    conditions = conditions or build_conditions(19.125)
    questions = part66.DIRECT + part66.INDIRECT
    rows=[]
    for c in conditions:
        for qi,q in enumerate(questions):
            for sample in range(8): rows.append((c,qi,q,'direct' if qi<16 else 'indirect',sample))
        for gi,(q,_answer) in enumerate(part66.GUARDS):
            for sample in range(4): rows.append((c,25+gi,q,'guard',sample))
    return rows

def read_pile_maximum(record_dir: Path):
    import numpy as np
    jp, npz = record_dir/'source.json', record_dir/'source.npz'
    if not jp.is_file() or not npz.is_file():
        raise FileNotFoundError(f'Part 26 Pile maximum records required: {jp} and {npz}')
    meta=json.loads(jp.read_text(encoding='utf-8'))
    if (meta.get('model'),meta.get('revision')) != (MODEL,REVISION):
        raise ValueError('Part 26 source record model/revision does not match pinned Qwen model')
    matches=[r for r in meta.get('source_neurons',[]) if (int(r['layer']),int(r['unit']))==ALIEN]
    if len(matches)!=1: raise ValueError(f'expected one Part 26 source record for {ALIEN}, found {len(matches)}')
    with np.load(npz,allow_pickle=False) as data:
        maximum=float(np.asarray(data['word_activations'][int(matches[0]['source_index'])],dtype=np.float32).max())
    if not maximum>0: raise ValueError(f'invalid N2742 Pile maximum: {maximum}')
    return maximum, str(jp), str(npz)

def push_hook(unit: int, value: float, stats: dict):
    """Part 62 phase=both rule: write the scalar at every row and sequence position."""
    def hook(_module,args):
        x=args[0]; y=x.clone()
        before=x[...,int(unit)].detach().float()
        y[...,int(unit)]=float(value)
        after=y[...,int(unit)].detach().float()
        diff=float((after-float(value)).abs().max().item())
        stats['calls']=stats.get('calls',0)+1
        stats['max_abs_diff']=max(stats.get('max_abs_diff',0.0),diff)
        return (y,)+tuple(args[1:])
    return hook

def _existing_rows(path):
    done=set()
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                r=json.loads(line); done.add((r['condition'],int(r['question_index']),int(r['sample'])))
            except (ValueError,KeyError,TypeError): continue
    return done

def run(args):
    import torch, neuron_report as nr, part25
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    maximum, source_json, source_npz = read_pile_maximum(Path(args.record_dir))
    tok,model=nr.load_model(MODEL,REVISION,args.device)
    model.requires_grad_(False)
    _,layers,_=nr.find_text_layers(model)
    proj=layers[ALIEN[0]].mlp.down_proj
    target_norm=part62_column_norm(proj,ALIEN[1]); control_norm=part62_column_norm(proj,CONTROL[1])
    conditions=build_conditions(maximum,target_norm,control_norm); rows=planned_rows(conditions)
    answer_path=out/'answers.jsonl'; existing=_existing_rows(answer_path)
    todo=[r for r in rows if (r[0]['id'],r[1],r[4]) not in existing]
    bykey=defaultdict(list)
    for r in todo: bykey[(r[0]['id'],r[1])].append(r)
    batches=[rs[i:i+args.batch] for rs in bykey.values() for i in range(0,len(rs),args.batch)]
    start=time.time(); batch_evidence=[]; first_push_batches=set()
    with answer_path.open('a',encoding='utf-8') as sink:
      for bi,chunk in enumerate(batches):
        prompts=[part25._token_ids(tok,nr.build_prompt(tok,r[2],False)) for r in chunk]
        assert len({tuple(p) for p in prompts})==1
        ids,mask=part66._pad(prompts,tok,torch.device(args.device))
        c=chunk[0][0]; rowops=[part25._operations(part66.TARGET,'zero') if c['makers'] else {} for _ in chunk]
        seeds=[part66.row_seed(c['id'],r[1],r[4]) for r in chunk]
        torch.manual_seed(seeds[0])
        if args.device=='cuda' and torch.cuda.is_available(): torch.cuda.manual_seed_all(seeds[0])
        if args.device=='xpu' and hasattr(torch,'xpu'): torch.xpu.manual_seed_all(seeds[0])
        handles=part66._hooks(layers,rowops,mask.bool()) if c['makers'] else []
        stats={'calls':0,'max_abs_diff':0.0}
        if c['push']:
            p=c['push']; handles.append(layers[p['unit'][0]].mlp.down_proj.register_forward_pre_hook(push_hook(p['unit'][1],p['value'],stats)))
        try:
          with torch.no_grad(): result=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=64,do_sample=True,temperature=.7,top_p=.95,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
        finally:
          for h in handles: h.remove()
        if c['push'] and c['id'] not in first_push_batches:
            if stats['calls']==0: raise RuntimeError(f"push hook did not fire on first batch for {c['id']}")
            first_push_batches.add(c['id'])
        batch_evidence.append({'condition':c['id'],'question_index':chunk[0][1],'calls':stats['calls'],'max_abs_diff':stats['max_abs_diff']})
        plen=ids.shape[1]
        for i,r in enumerate(chunk):
          raw=result[i,plen:].tolist()
          if tok.eos_token_id is not None and int(tok.eos_token_id) in raw: raw=raw[:raw.index(int(tok.eos_token_id))+1]
          sink.write(json.dumps({'condition':c['id'],'question_index':r[1],'question':r[2],'question_kind':r[3],'sample':r[4],'seed':seeds[i],'answer':tok.decode(raw,skip_special_tokens=True).strip(),'new_tokens':len(raw),'stop_reason':part66._stop_reason(raw,tok)},ensure_ascii=False)+'\n')
        sink.flush(); nr.spill_guard(args.device,args.max_gpu_gb)
    metadata={'model':MODEL,'revision':REVISION,'dtype':'bfloat16','device':str(part25.actual_device(model)),'transformers_version':__import__('transformers').__version__,'batch_size':args.batch,'pile_maximum_n2742':maximum,'pile_maximum_records':{'json':source_json,'npz':source_npz},'down_proj_column_norms':{'layer':22,'N2742':target_norm,'N8870':control_norm,'method':'part62/run62.py _column_norm: vector_norm(down_proj.weight.detach().float()[:, unit])'},'conditions':conditions,'requested_rows':len(rows),'new_rows_this_invocation':len(todo),'wall_time_seconds':time.time()-start,'push_hook_batch_evidence':batch_evidence}
    (out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')

def part62_column_norm(down_proj, unit):
    import torch
    weight=down_proj.weight.detach().float()
    if weight.ndim!=2 or unit<0 or unit>=weight.shape[1]: raise ValueError(f'unit {unit} outside down_proj width')
    norm=float(torch.linalg.vector_norm(weight[:,unit]).item())
    if not norm>0: raise ValueError(f'invalid down_proj column norm: {norm}')
    return norm

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True); p.add_argument('--device',default='xpu'); p.add_argument('--batch',type=int,default=16); p.add_argument('--dry-run',action='store_true'); p.add_argument('--record-dir',default=str(ROOT/'results/part26')); p.add_argument('--max-gpu-gb',type=float,default=20)
    a=p.parse_args()
    if a.batch<1: p.error('--batch must be positive')
    if a.dry_run:
      cs=build_conditions(19.125); print(json.dumps({'planned_rows':len(planned_rows(cs)),'conditions':cs},indent=2)); return
    run(a)
if __name__=='__main__': main()
