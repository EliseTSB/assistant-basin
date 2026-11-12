#!/usr/bin/env python3
"""Part 201 deterministic batch-shape plan and greedy runner."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
UNITS_DEFAULT=ROOT/'part163/units163.json'; OUT_DEFAULT=ROOT/'results/part201'
PROCESSES={'bf16-main':['A','A-again','B','M','C'],'bf16-second':['A2'],'float32':['F8','F1']}

def _battery():
    from part163.run163 import choose_questions
    return choose_questions()

def _unit_sample(units_path):
    units=json.loads(Path(units_path).read_text(encoding='utf-8'))
    arms={arm:[u for u in units if u['arm']==arm] for arm in ('neuron','random')}
    selected={arm:random.Random(seed).sample(arms[arm],12) for arm,seed in [('neuron',201),('random',202)]}
    return [dict(u) for pair in zip(selected['neuron'],selected['random']) for u in pair]

def plan(units_path=UNITS_DEFAULT):
    """Return deterministic planned calls using only Python data structures."""
    qs=_battery(); units=_unit_sample(units_path)
    def row(u,qi,q):
        # neuron and random_index are what part163.run163.pushes_for reads from a row to build its push.
        return {'row_id':f"{u['unit_id']}|q{qi}",'unit_id':u['unit_id'],'arm':u['arm'],'layer':int(u['layer']),
                'neuron':u.get('neuron'),'random_index':u.get('random_index'),
                'question_index':qi,'question':q['question'],'kind':q['kind'],'dose':0.8}
    pushed={qi:[row(u,qi,q) for u in units] for qi,q in enumerate(qs)}
    plain=[{'row_id':f'plain|q{qi}','unit_id':None,'arm':'plain','layer':None,'question_index':qi,
            'question':q['question'],'kind':q['kind'],'dose':0.0} for qi,q in enumerate(qs)]
    from tools.kernel_reference import QUESTIONS as kernel_questions
    kplain=[{'row_id':f'plain|k{i}','unit_id':None,'arm':'plain','layer':None,'question_index':None,
             'question':q,'kind':'kernel','dose':0.0} for i,q in enumerate(kernel_questions)]
    calls={}
    def group(pass_name, rows, size, allow_partial=False):
        cs=[]
        for start in range(0,len(rows),size):
            members=rows[start:start+size]
            cs.append({'pass':pass_name,'call':len(cs),'rows':members,'row_ids':[r['row_id'] for r in members]})
        calls[pass_name]=cs; return cs
    acalls=[]
    for qi in range(16): acalls += group('A',pushed[qi],8)
    # group() call numbering resets per question; normalize globally per pass
    for i,c in enumerate(acalls): c['call']=i
    calls['A']=acalls
    aagain=[dict(c,pass_name='A-again') for c in acalls[:2]]
    for c in aagain: c['pass']='A-again'
    calls['A-again']=aagain
    a_members={frozenset(c['row_ids']) for c in acalls}
    b_rows=[]; draws=0
    # Each full redraw is an independent seed. Rejection is separable by question, so draw each
    # question until its three calls pass; this has the same conditional permutation law without
    # burning thousands of all-or-nothing whole-plan draws.
    for qi in range(16):
        attempt=0
        while True:
            attempt+=1; rng=random.Random(2011+(attempt-1)*16+qi); perm=list(pushed[qi]); rng.shuffle(perm); ok=True
            for start in range(0,24,8):
                chunk=perm[start:start+8]
                if frozenset(r['row_id'] for r in chunk) in a_members: ok=False
                for slot,r in enumerate(chunk):
                    a_slot=units.index(next(u for u in units if u['unit_id']==r['unit_id']))%8
                    if slot==a_slot: ok=False
            if ok: b_rows.extend(perm); draws=max(draws,attempt); break
    bcs=group('B',b_rows,8)
    for c in bcs: c['call']=bcs.index(c)
    calls['B_draws']=draws
    mrows=[r for qi in range(16) for r in pushed[qi]]; random.Random(2012).shuffle(mrows)
    mcs=group('M',mrows,8); base=len(mcs)
    mp=plain+kplain
    for start in range(0,len(mp),8):
        chunk=mp[start:start+8]; mcs.append({'pass':'M','call':len(mcs),'rows':chunk,'row_ids':[r['row_id'] for r in chunk]})
    crows=[r for qi in range(16) for r in pushed[qi][:12]]+plain+kplain
    ccs=group('C',crows,1)
    # A2 is q0..3, same call rows and slots as A.
    calls['A2']=[dict(c,pass_name='A2',**{'pass':'A2'}) for c in acalls if c['rows'][0]['question_index']<4]
    # F-subsets: the three lowest-numbered identity and maker questions, alternating by kind so that a cut to the
    # first half (design, Size) keeps both kinds. F8 = pass A's first two calls of each (same rows, same slots, and
    # pass A's call numbers). F1 = the first 8 units on the same questions, ONE ROW PER CALL.
    by_kind={kind:[qi for qi,q in enumerate(qs) if q['kind']==kind][:3] for kind in ('identity','maker')}
    f_order=[qi for pair in zip(by_kind['identity'],by_kind['maker']) for qi in pair]
    first16={u['unit_id'] for u in units[:16]}
    f_calls=[c for qi in f_order for c in acalls if c['rows'][0]['question_index']==qi and c['rows'][0]['unit_id'] in first16]
    calls['F8']=[dict(c,pass_name='F8',**{'pass':'F8'}) for c in f_calls]
    f1_rows=[r for qi in f_order for r in pushed[qi][:8]]
    calls['F1']=[{'pass':'F1','call':i,'rows':[r],'row_ids':[r['row_id']]} for i,r in enumerate(f1_rows)]
    return {'schema':1,'unit_order':[u['unit_id'] for u in units], 'questions':[q['question'] for q in qs],'f_question_order':f_order,
            'b_draws':draws,'passes':{p:{'calls':calls[p],'rows':sum(len(c['rows']) for c in calls[p])} for p in ('A','A-again','B','M','C','A2','F8','F1')}}

def smoke_plan(units_path=UNITS_DEFAULT):
    """Small target-architecture run: four units, two questions per arm, and four plain rows."""
    full=plan(units_path); unit_ids=set(full['unit_order'][:4]); qids={0,1,8,9}
    selected_plain=[]
    for info in full['passes']['M']['calls']:
        for row in info['rows']:
            if row['arm']=='plain' and row.get('question_index') in qids and row not in selected_plain:
                selected_plain.append(row)
    selected_plain=selected_plain[:4]
    out={'schema':1,'smoke':True,'unit_order':[u for u in full['unit_order'] if u in unit_ids],
         'questions':[q for i,q in enumerate(full['questions']) if i in qids],'b_draws':full['b_draws'],'passes':{}}
    for name,info in full['passes'].items():
        calls=[]
        for old in info['calls']:
            kept=[r for r in old['rows'] if r['arm']!='plain' and r.get('unit_id') in unit_ids and r.get('question_index') in qids]
            if kept: calls.append({'pass':name,'call':len(calls),'rows':kept,'row_ids':[r['row_id'] for r in kept]})
        if name in ('M','C'):
            plain_chunks=[selected_plain[i:i+(8 if name=='M' else 1)] for i in range(0,len(selected_plain),8 if name=='M' else 1)]
            for chunk in plain_chunks: calls.append({'pass':name,'call':len(calls),'rows':chunk,'row_ids':[r['row_id'] for r in chunk]})
        out['passes'][name]={'calls':calls,'rows':sum(len(c['rows']) for c in calls)}
    return out

def _eos(tok,model):
    vals=[]
    for v in (getattr(tok,'eos_token_id',None),getattr(model.config,'eos_token_id',None),getattr(model.config,'eot_token_id',None)):
        if isinstance(v,(list,tuple)): vals.extend(v)
        elif isinstance(v,int): vals.append(v)
    return list(dict.fromkeys(int(v) for v in vals))

class RecordingPushHook:
    """Part-local row-isolated decode hook with Gate G4 measurements."""
    def __init__(self, module, pushes):
        import torch
        self.torch=torch; self.pushes=pushes.detach().to(dtype=torch.bfloat16).to(dtype=self.torch.float32); self.dtype=None; self.prefill=True; self.firings=0; self.push_all_positions=False
        self._cast=None; self._diff=None; self.rows=int(pushes.shape[0])
        self.handle=module.register_forward_pre_hook(self._hook)
    def _hook(self,module,args):
        x=args[0]
        if self.prefill and not self.push_all_positions: self.prefill=False; return None
        if not self.push_all_positions and x.shape[-2]!=1: raise RuntimeError('decode push hook expected one token')
        if self._cast is None or self._cast.dtype!=x.dtype or self._cast.device!=x.device:
            # Cast once per dtype and device. The delta added is this tensor at every firing, so its difference from
            # the prescribed bf16 values is measured here, once, and read on the CPU only after the call (Gate G4).
            self._cast=self.pushes.to(device=x.device,dtype=x.dtype)
            d=(self._cast.detach().float()-self.pushes.to(device=x.device,dtype=self.torch.float32)).abs().amax(dim=1)
            self._diff=d if self._diff is None else self.torch.maximum(self._diff.to(d.device),d)
        p=self._cast.unsqueeze(1)
        if self.push_all_positions: p=p.expand(-1,x.shape[-2],-1)
        self.dtype=str(x.dtype); self.firings+=1
        return (x+p,)+tuple(args[1:])
    @property
    def diffs(self):
        """Per row, the largest difference between the delta added and the prescribed push; None when the hook never
        fired, because then nothing was measured (and nothing was pushed)."""
        if self._diff is None: return [None]*self.rows
        return [float(v) for v in self._diff.detach().cpu().tolist()]
    def remove(self): self.handle.remove()

def _prompt_len(tok,prompt):
    ids=tok(prompt,add_special_tokens=False)['input_ids']
    return int(ids.shape[-1]) if hasattr(ids,'shape') else len(ids[-1] if ids and isinstance(ids[0],(list,tuple)) else ids)

def _push_hash(pushes,row):
    """sha256 of one row's prescribed push: for each layer where the row's payload is not all zero, the layer number,
    the width and the payload's bf16 bytes. A call's matrices carry zero rows for the layers of its other rows; those
    are left out, so the hash does not depend on which rows share the call, and the same row has the same hash in
    every pass and every process (Gate G4 compares them)."""
    import torch
    if not pushes: return None
    h=hashlib.sha256()
    for layer in sorted(pushes,key=int):
        values=pushes[layer][row].detach().to(device='cpu',dtype=torch.bfloat16).contiguous()
        if not bool(torch.count_nonzero(values.to(dtype=torch.float32))): continue
        h.update(f'layer={int(layer)};width={int(values.numel())};'.encode('ascii')); h.update(values.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def _canonical_sha256(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()

def input_hashes(units,medians):
    """What a push is built from besides the model: the unit list and the per-layer medians. Hashed as canonical JSON,
    so a change of line endings or spacing is not a different input and a changed number is."""
    return {'units_sha256':_canonical_sha256(units),'medians_sha256':_canonical_sha256(medians)}

def model_snapshot(model_path):
    """The 40-hex snapshot folder in a Hugging Face cache path, or None."""
    import re
    found=re.search(r'snapshots[\\/]+([0-9a-f]{40})(?:[\\/]|$)',str(model_path))
    return found.group(1) if found else None

def receipt_of(record,launch_index,pass_name,call_number):
    """The call receipt a launch wrote for one call, or None."""
    loads=record.get('loads',[])
    if not isinstance(launch_index,int) or isinstance(launch_index,bool) or not 0<=launch_index<len(loads): return None
    found=[c for c in loads[launch_index].get('calls',[]) if c.get('pass')==pass_name and c.get('call')==call_number]
    return found[-1] if found else None

def decode_attempt(model,tok,layers,prompts,pushmap,max_new_tokens,device):
    """One decode of one call, with everything that describes it: its rows, its time and its hook evidence."""
    from tools.margin_recorder import MarginRecorder
    rec=MarginRecorder(_eos(tok,model)); start=time.perf_counter()
    seq,rows=generate_rows(model,tok,layers,prompts,pushmap,max_new_tokens,device,rec)
    return {'rows':rows,'seconds':time.perf_counter()-start,'hook_firings':dict(getattr(rec,'hook_firings',{})),
            'forward_passes':int(getattr(rec,'forward_passes',0)),'generation_steps':max((len(r['tokens']) for r in rows),default=0)}

def attempt_problems(attempt,process,pushed):
    """Why one attempt's own evidence does not show the prescribed push on every decode step (Gate G4), or []."""
    if not pushed: return []
    if attempt['forward_passes']==0:
        # The first generated token comes from the prefill forward, where no push applies (Part 163's generator does
        # the same). A call whose rows all end there has no decode forward, so the push was never exercised. That
        # is not a wrong push and must not be reported as one; it is also not evidence of a right one.
        return ['no decode forward ran: every row ended at its first generated token, which is decoded before the push applies, so the push was not exercised']
    problems=[]; counts=list(attempt['hook_firings'].values())
    if not counts or any(count!=attempt['forward_passes'] for count in counts):
        problems.append(f"hook firings {attempt['hook_firings']} are not the {attempt['forward_passes']} decode forwards")
    if any(r.get('push_max_abs_diff')!=0 for r in attempt['rows']):
        problems.append('the delta the hook added is not the prescribed bf16 push')
    expected='torch.float32' if process=='float32' else 'torch.bfloat16'
    if any(r.get('hook_dtype')!=expected for r in attempt['rows']):
        problems.append(f"the activation at the push hook is not {expected.split('.')[1]}")
    return problems

def select_attempt(decode,process,pushed):
    """Decode a call. If any row has a non-finite score, decode it once more and keep the SECOND attempt whole: its
    rows with its own time and hook evidence (the design: 'run once more; a row still non-finite is counted'). The
    first attempt's time is kept apart. Returns (attempt kept, summary of a discarded attempt or None, problems of the
    attempt kept)."""
    first=decode(); kept=first; discarded=None
    if any(r['nonfinite'] for r in first['rows']):
        kept=decode()
        discarded={'seconds':first['seconds'],'generation_steps':first['generation_steps'],
                   'nonfinite_rows':sum(bool(r['nonfinite']) for r in first['rows']),
                   'hook_firings':first['hook_firings'],'forward_passes':first['forward_passes']}
    return kept,discarded,attempt_problems(kept,process,pushed)

def generate_rows(model,tok,layers,prompts,pushes,max_new_tokens,device,recorder):
    import torch
    from transformers import GenerationConfig, LogitsProcessorList
    prompts=list(prompts); n=len(prompts)
    if any(int(v.shape[0])!=n for v in pushes.values()): raise ValueError('each layer push must have one row per prompt')
    tok.padding_side='left'; tok.pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    encoded=tok(prompts,return_tensors='pt',padding=True,add_special_tokens=False)
    ids=encoded['input_ids'].to(device); mask=encoded['attention_mask'].to(device)
    prompt_length=int(ids.shape[1]); hooks=[]
    # Each layer may be absent for plain rows; the provided matrix is [rows,width] on the layer device.
    for layer_no,values in pushes.items():
        hook=RecordingPushHook(layers[int(layer_no)].mlp.down_proj,values); hooks.append((int(layer_no),hook))
    eos=_eos(tok,model)
    config=GenerationConfig(max_new_tokens=int(max_new_tokens),do_sample=False,num_beams=1,repetition_penalty=1.0,
      no_repeat_ngram_size=0,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=eos or None,
      return_dict_in_generate=False)
    processors=LogitsProcessorList([recorder]); forward_count={'n':0}; original_forward=model.forward
    def counted_forward(*args,**kwargs):
        forward_count['n']+=1; return original_forward(*args,**kwargs)
    model.forward=counted_forward
    try:
        with torch.inference_mode(): seq=model.generate(input_ids=ids,attention_mask=mask,generation_config=config,logits_processor=processors)
    finally:
        model.forward=original_forward
        for _,h in hooks: h.remove()
    recorder.forward_passes=max(0,forward_count['n']-1)
    recorder.hook_firings={layer:h.firings for layer,h in hooks}
    rows=recorder.rows(seq,prompt_length,eos)
    diffs=[hh.diffs for _,hh in hooks]; dtypes=sorted({str(hh.dtype) for _,hh in hooks})
    for rowno,r in enumerate(rows):
        if hooks:
            # One dtype name when every hooked layer saw the same one; otherwise all of them, which fails Gate G4.
            measured=[d[rowno] for d in diffs]
            r['hook_dtype']=dtypes[0] if len(dtypes)==1 else '+'.join(dtypes)
            r['push_max_abs_diff']=None if any(v is None for v in measured) else max(measured,default=0.)
        else: r['hook_dtype']=None; r['push_max_abs_diff']=None
    return seq,rows

def _write_json(path,obj):
    """Write through a temporary file and a rename, so a crash leaves the old file or the new one, never half of one."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); os.replace(tmp,path)
def _repair(path):
    path=Path(path)
    if not path.exists(): return
    data=path.read_bytes(); cut=data.rfind(b'\n')+1
    if cut<len(data): path.write_bytes(data[:cut])
def _load_rows(path, planned=None, load=None):
    """The rows on disk. With a plan, a call that is not whole is removed as a unit. With the load record too, so is a
    call whose rows have no receipt in the launch they name: its hook evidence was never recorded, so it is decoded
    again instead of being read on trust."""
    _repair(path); path=Path(path)
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()] if path.exists() else []
    if planned is not None and rows:
        expected={(name,c['call']):set(c['row_ids']) for name,info in planned['passes'].items() for c in info['calls']}
        present={}
        for row in rows: present.setdefault((row.get('pass'),row.get('call')),[]).append(row.get('row_id'))
        incomplete={key for key,ids in present.items() if expected.get(key)!=set(ids) or len(ids)!=len(expected.get(key,()))}
        if load is not None:
            for row in rows:
                if receipt_of(load,row.get('launch'),row.get('pass'),row.get('call')) is None: incomplete.add((row.get('pass'),row.get('call')))
        if incomplete:
            # Nothing is destroyed: what is taken out of rows.jsonl goes to rows.removed.jsonl with the time.
            removed=[dict(r,removed=time.strftime('%Y-%m-%d %H:%M:%S')) for r in rows if (r.get('pass'),r.get('call')) in incomplete]
            _append(path.with_name('rows.removed.jsonl'),removed)
            print(f'PART201 resume: {len(removed)} rows of {len(incomplete)} calls without a whole call or its receipt will be decoded again',flush=True)
            rows=[r for r in rows if (r.get('pass'),r.get('call')) not in incomplete]
            tmp=path.with_name(path.name+'.repair.tmp')
            tmp.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=True)+'\n' for r in rows),encoding='utf-8')
            tmp.replace(path)
    return rows
def pending_rows(call, pass_name, done):
    return [(slot,row) for slot,row in enumerate(call['rows']) if (pass_name,row['row_id']) not in done]

def _append(path,rows):
    with Path(path).open('a',encoding='utf-8',newline='\n') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=True)+'\n')
        f.flush()

def device_record(device):
    """The compute device as torch reports it, plus the Windows display-driver version on the Arc host. A driver
    update is a variable that can move outputs (PRIOR-ART.md, 2026-11-09), so every launch records it."""
    import os, subprocess, torch
    out={'device':str(device)}
    try:
        if str(device).startswith('xpu') and hasattr(torch,'xpu'):
            props=torch.xpu.get_device_properties(torch.xpu.current_device())
            for key in ('name','platform_name','vendor','driver_version','version','type'):
                value=getattr(props,key,None) if not isinstance(props,dict) else props.get(key)
                if value is not None: out[key]=str(value)
    except Exception as exc: out['torch_error']=repr(exc)
    if os.name=='nt':
        try:
            cmd="Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.DriverVersion }"
            done=subprocess.run(['powershell.exe','-NoProfile','-Command',cmd],capture_output=True,text=True,timeout=30)
            out['windows_display_drivers']=[line.strip() for line in done.stdout.splitlines() if line.strip()]
        except Exception as exc: out['windows_error']=repr(exc)
    return out

def row_fields(rr,pass_name,process,spec,call,slot,prompt_tokens,pad_tokens,answer,push_sha256,seconds,launch_index):
    """Add the plan and launch fields of part201/ROWS-SCHEMA.md to one recorder row."""
    rr.update({'pass':pass_name,'process':process,'row_id':spec['row_id'],'unit_id':spec['unit_id'],'arm':spec['arm'],
     'layer':spec['layer'],'question_index':spec['question_index'],'question':spec['question'],'kind':spec['kind'],
     'dose':spec['dose'],'call':call['call'],'call_rows':call['row_ids'],'slot':slot,'launch':int(launch_index),
     'prompt_tokens':int(prompt_tokens),'pad_tokens':int(pad_tokens),'answer':answer,'token_count':len(rr['tokens']),
     'push_sha256':push_sha256,'seconds':seconds,'written':time.strftime('%Y-%m-%d %H:%M:%S')})
    return rr

# A one-row call's time as a share of an 8-row call's, from Part 200's bf16 rates on this model and card
# (0.24 answers/s one row per call, 0.89 answers/s in chunks of 8): (1/0.24) / (8/0.89).
ONE_ROW_CALL_SHARE=0.4635
CUT={'F8':6,'F1':24}   # calls kept after the design's float32 cut: F8's first 48 rows, F1's first 24

def _run_passes(args, passes):
    import torch, neuron_report as nr
    from part163 import run163
    from part86 import common86 as common
    from tools.kernel_reference import checkpoint_dir, upcast_compute
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); data_path=out/'rows.jsonl'
    planned=smoke_plan(args.units) if args.smoke else plan(args.units)
    plan_path=out/'plan.json'; plan_bytes=(json.dumps(planned,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
    plan_hash=hashlib.sha256(plan_bytes).hexdigest()
    if plan_path.exists():
        if hashlib.sha256(plan_path.read_bytes()).hexdigest()!=plan_hash:
            print('PART201 frozen plan hash mismatch; refusing before model load',flush=True); raise SystemExit(3)
    else: plan_path.write_bytes(plan_bytes)
    load_path=out/'load.json'; previous=json.loads(load_path.read_text(encoding='utf-8')) if load_path.exists() else {'loads':[]}
    if any(entry.get('plan_sha256') not in (None,plan_hash) for entry in previous.get('loads',[])):
        print('PART201 load record has a different frozen plan hash; refusing before model load',flush=True); raise SystemExit(3)
    # lock contract imported from Part 199
    from part199.run199 import FolderLock
    with FolderLock(out):
        units=json.loads(Path(args.units).read_text(encoding='utf-8')); medians=json.loads(Path(args.medians).read_text(encoding='utf-8'))
        inputs=input_hashes(units,medians)
        # The snapshot folder of a Hugging Face cache path names the revision that will be loaded. The real run must
        # load the registered one; a smoke run may load any small model.
        snapshot=model_snapshot(args.model)
        if not args.smoke and snapshot!=common.REVISION:
            print(f'PART201 --model names snapshot {snapshot}, not the registered revision {common.REVISION}; refusing before model load',flush=True); raise SystemExit(3)
        if any(entry.get('inputs')!=inputs for entry in previous.get('loads',[])):
            print('PART201 the units or medians are not the ones an earlier launch used; refusing before model load',flush=True); raise SystemExit(3)
        existing=_load_rows(data_path,planned,previous); done={(r['pass'],r['row_id']) for r in existing}
        tok,model=nr.load_model(args.model,common.REVISION,args.device,attn_implementation='eager')
        if (nr.LAST_LOAD or {}).get('attn_implementation')!='eager' or getattr(model.config,'_attn_implementation',None)!='eager':
            raise SystemExit(3)
        model.requires_grad_(False); model.eval()
        layers=common.find_layers(model)
        launch_index=len(previous.get('loads',[]))
        launch={'launch':launch_index,'process':args.process,'passes':passes,'model':args.model,'revision':common.REVISION,'plan_sha256':plan_hash,'inputs':inputs,'model_snapshot':snapshot,'device_record':device_record(args.device),'dtype':str(next(model.parameters()).dtype),'attn_implementation':'eager','calls':[]}
        if args.process=='float32':
            launch['upcast']=upcast_compute(model,checkpoint_dir(args.model,None)); launch['compute_dtype']='torch.float32'
        record={'part':201,'smoke':bool(args.smoke),'plan_sha256':plan_hash,'loads':previous.get('loads',[])+[launch]}
        _write_json(load_path,record)
        limits={}
        if any('cut_applied' in entry for entry in previous.get('loads',[])):
            limits=dict(CUT); launch['cut_applied']={'F8_rows':48,'F1_rows':24,'carried_from_an_earlier_launch':True}; _write_json(load_path,record)
        cache={}; g5=None
        if args.process=='bf16-main' and 'A' in passes:
            first=planned['passes']['A']['calls'][0]; test_rows=first['rows']; question=test_rows[0]['question']
            test_pushes=run163.pushes_for(test_rows,layers,medians,common,cache)
            prompts=[common.build_prompt(tok,question) for _ in test_rows]
            from part160 import run160
            original_generate=model.generate; captured={}
            def capture_generate(*args,**kwargs):
                captured['prompt_length']=int(kwargs['input_ids'].shape[1]); result=original_generate(*args,**kwargs); captured['sequences']=result.detach().cpu(); return result
            model.generate=capture_generate
            try: default=run160.generate_checked(model,tok,layers,question,test_pushes,args.max_new_tokens,args.device,common)
            finally: model.generate=original_generate
            rec=__import__('tools.margin_recorder',fromlist=['MarginRecorder']).MarginRecorder(_eos(tok,model))
            seq,observed=generate_rows(model,tok,layers,prompts,test_pushes,args.max_new_tokens,args.device,rec)
            g5=g5_from_capture(captured.get('sequences'),captured.get('prompt_length',0),observed,_eos(tok,model))
            launch['G5']={'passed':g5,'rows':len(test_rows),'rows_that_ended_before_the_cap':sum(r['finish']=='eos' for r in observed)}; _write_json(load_path,record)
            if not g5: raise SystemExit(3)
        for p in passes:
            if p=='A-again':
                # A-again repeats pass A's first two calls in the SAME process. If they still have to be written and
                # this launch did not itself decode those two calls of pass A, the repeat would be another process:
                # write nothing and say so. Rows already written by the launch that made pass A are left alone.
                unwritten=[c for c in planned['passes'][p]['calls'] if pending_rows(c,p,done)]
                same_process={c['call'] for c in launch['calls'] if c['pass']=='A'}
                if not unwritten: continue
                if not {c['call'] for c in planned['passes'][p]['calls']}.issubset(same_process):
                    launch['a_again']='not same process'; _write_json(load_path,record); continue
            for position,call in enumerate(planned['passes'][p]['calls']):
                if p in limits and position>=limits[p]: break
                rows=call['rows']; todo=pending_rows(call,p,done)
                if not todo: continue
                callspec=rows; qi=callspec[0].get('question_index'); prompts=[common.build_prompt(tok,r['question']) for r in callspec]
                if any(r['unit_id'] is not None for r in callspec):
                    pushmap=run163.pushes_for(callspec,layers,medians,common,cache)
                else: pushmap={}
                pushed_call=bool(pushmap)
                kept,discarded,problems=select_attempt(lambda: decode_attempt(model,tok,layers,prompts,pushmap,args.max_new_tokens,args.device),args.process,pushed_call)
                seconds=kept['seconds']; recrows=kept['rows']
                if problems:
                    # The attempt that would be written does not show the prescribed push on every decode step.
                    launch['stopped']={'pass':p,'call':call['call'],'problems':problems}; _write_json(load_path,record)
                    print('PART201 stopped: '+'; '.join(problems),flush=True); raise SystemExit(3)
                if args.process=='float32' and p=='F8' and call['call']==0 and 'projection_minutes' not in launch:
                    f8_minutes=seconds*12/60; f1_minutes=seconds*ONE_ROW_CALL_SHARE*48/60
                    launch['projection_minutes']={'F8':f8_minutes,'F1':f1_minutes,'combined':f8_minutes+f1_minutes,'one_row_call_share':ONE_ROW_CALL_SHARE}
                    if f8_minutes+f1_minutes>45:
                        launch['cut_applied']={'F8_rows':48,'F1_rows':24,'before_rows_written':True}; limits.update(CUT)
                    _write_json(load_path,record)
                output=[]; lengths=[_prompt_len(tok,pr) for pr in prompts]
                pending_slots={slot for slot,_ in todo}
                for slot,r in enumerate(callspec):
                    if slot not in pending_slots: continue
                    rr=row_fields(recrows[slot],p,args.process,r,call,slot,lengths[slot],max(lengths)-lengths[slot],
                        tok.decode(recrows[slot]['tokens'],skip_special_tokens=True).strip(),
                        _push_hash(pushmap,slot) if r['unit_id'] is not None else None,seconds,launch_index)
                    output.append(rr)
                # The receipt describes the attempt whose rows are written, and it goes to disk BEFORE the rows: a crash
                # between the two leaves a receipt without rows (the call is decoded again), never rows without a receipt.
                receipt={'pass':p,'call':call['call'],'row_ids':list(call['row_ids']),'row_count':len(rows),'pushed':pushed_call,
                         'seconds':seconds,'generation_steps':kept['generation_steps'],'hook_firings':kept['hook_firings'],
                         'forward_passes':kept['forward_passes'],'attempts':2 if discarded else 1}
                if discarded: receipt['discarded_attempt']=discarded
                launch['calls'].append(receipt); _write_json(load_path,record)
                _append(data_path,output); done.update((p,r['row_id']) for _,r in todo)
                if args.process=='bf16-main' and p=='A' and call['call']==2:
                    pilot_rows=sum(c['row_count'] for c in launch['calls'] if c['pass']=='A' and c['call']<3)
                    pilot_seconds=sum(c['seconds'] for c in launch['calls'] if c['pass']=='A' and c['call']<3)
                    launch['pilot_answers_per_second']=pilot_rows/pilot_seconds if pilot_seconds else 0.0
                    _write_json(load_path,record)
                    if launch['pilot_answers_per_second']<0.3:
                        print('PART201 pilot stopped: fewer than 0.3 answers per second',flush=True); raise SystemExit(3)
        if args.process=='float32':
            probe_path=out/'probe.json'
            if not probe_path.exists():
                fcall=planned['passes']['F8']['calls'][0]
                stored=_load_rows(data_path,planned); byid={(r['pass'],r['row_id']):r for r in stored}
                probe_rows=[byid[('F8',row['row_id'])] for row in fcall['rows']]
                probe_push=run163.pushes_for(fcall['rows'],layers,medians,common,cache)
                probe=timing_probe(model,tok,layers,probe_rows,probe_push,args.device,bool(args.smoke))
                # Each F8 call's receipt from the launch that wrote its rows (an earlier launch may hold a receipt for a
                # call it never finished writing).
                f8_calls=[]
                for c in planned['passes']['F8']['calls']:
                    first=byid.get(('F8',c['rows'][0]['row_id']))
                    receipt=receipt_of(record,first.get('launch'),'F8',c['call']) if first else None
                    if receipt: f8_calls.append(receipt)
                probe['F8_cached_calls']=[{'call':c['call'],'seconds':c['seconds'],'generation_steps':c.get('generation_steps',0),
                    'seconds_per_generation_step':c['seconds']/max(1,c.get('generation_steps',1))} for c in f8_calls]
                probe['plan_sha256']=plan_hash; _write_json(probe_path,probe)
        return record

def _sync(device):
    import torch
    if str(device).startswith('xpu') and hasattr(torch,'xpu') and torch.xpu.is_available(): torch.xpu.synchronize()
    elif str(device).startswith('cuda') and torch.cuda.is_available(): torch.cuda.synchronize()

def timing_probe(model,tok,layers,rows,layer_pushes,device,smoke=False):
    import torch
    from part86 import common86
    lengths=(8,16,24) if smoke else (32,64,96)
    probes=[]
    prompt_ids=[]
    for row in rows:
        prompt=common86.build_prompt(tok,row['question'])
        encoded=tok(prompt,add_special_tokens=False)['input_ids']
        ids=encoded.tolist() if hasattr(encoded,'tolist') else encoded
        if ids and isinstance(ids[0],(list,tuple)): ids=list(ids[0])
        generated=[int(v) for v in row['tokens']]
        prompt_ids.append(ids+generated)
    max_rows=min(8,len(rows)); row_counts=[n for n in (1,4,8) if n<=max_rows]
    # A re-check needs the scores at the last position only. Without this pre-hook the output head would run on every
    # position (8 rows x 96 positions x the vocabulary in float32, about 470 MB), which is not the work being priced.
    head=model.get_output_embeddings(); head_positions=[]
    def last_position_only(module,args):
        head_positions.append(int(args[0].shape[-2])); return (args[0][...,-1:,:],)+tuple(args[1:])
    slicer=head.register_forward_pre_hook(last_position_only) if head is not None else None
    try:
        for n in row_counts:
            selected=rows[:n]; pushes={layer:values[:n] for layer,values in layer_pushes.items()}
            hooks=[]
            for layer,values in pushes.items():
                hook=RecordingPushHook(layers[int(layer)].mlp.down_proj,values); hook.push_all_positions=True; hooks.append(hook)
            try:
                for length in lengths:
                    batch=[]
                    for ids in prompt_ids[:n]:
                        if not ids: ids=[int(getattr(tok,'bos_token_id',1) or 1)]
                        seq=list(ids[:length])
                        fill=ids[len(seq):] or ids[-max(1,min(len(ids),4)):]
                        while len(seq)<length: seq.append(int(fill[(len(seq)-len(ids))%len(fill)]))
                        batch.append(seq[:length])
                    input_ids=torch.tensor(batch,dtype=torch.long,device=device); mask=torch.ones_like(input_ids)
                    repeats=[]
                    for repeat in range(5):
                        _sync(device); start=time.perf_counter()
                        with torch.inference_mode(): out=model(input_ids=input_ids,attention_mask=mask,use_cache=False)
                        logits_positions=int(out.logits.shape[-2]); del out
                        _sync(device); seconds=time.perf_counter()-start; repeats.append(seconds)
                    probes.append({'rows':n,'prefix_tokens':length,'seconds':repeats,'logits_positions':logits_positions,
                                   'hook_firings':[hook.firings for hook in hooks],'hook_dtypes':sorted({str(hook.dtype) for hook in hooks})})
            finally:
                for hook in hooks: hook.remove()
    finally:
        if slicer is not None: slicer.remove()
    return {'repetitions':5,'prefix_lengths':list(lengths),'forward_passes':probes,
            'work_timed':'model body over the full prefix with no cache; output head on the last position only' if slicer is not None else 'model body and output head over the full prefix with no cache',
            'push_positions':'every position of the prefix (timing only: the add costs the same; a real re-check must push at generated positions only)',
            'head_input_positions_seen':sorted(set(head_positions))}

def g1_valid(load):
    return load.get('attn_implementation')=='eager' and ((load.get('process')!='float32' and load.get('dtype')=='torch.bfloat16') or
        (load.get('process')=='float32' and load.get('compute_dtype')=='torch.float32' and load.get('upcast',{}).get('kept_bf16',0)>0))

def g2_valid(row):
    return bool(row.get('emitted_is_max')) and len(row.get('emitted_logit',[]))==len(row.get('top1_logit',[]))

def g3_valid(row, call, slot):
    return row.get('call_rows')==call.get('row_ids') and row.get('call')==call.get('call') and row.get('slot')==slot

def g4_valid(row):
    return row.get('hook_dtype')=='torch.float32' and row.get('push_max_abs_diff')==0.0

def eos_trim(tokens, eos_ids=()):
    """Tokens up to and including the first end token; unchanged when there is none (a row that ran to the cap)."""
    eos={int(v) for v in (eos_ids or ())}; out=[int(v) for v in tokens]
    for i,token in enumerate(out):
        if token in eos: return out[:i+1]
    return out

def g5_equal(reference_tokens, recorded_tokens, eos_ids=()):
    """Gate G5's comparison. Both sides are cut at their first end token first: `generate` returns a rectangle that is
    padded after a row has ended, and the recorder stores each row already cut there. Anything before the end token,
    and the end token itself, must match."""
    return [eos_trim(r,eos_ids) for r in reference_tokens] == [eos_trim(r,eos_ids) for r in recorded_tokens]

def g5_from_capture(sequences, prompt_length, recorded_rows, eos_ids=()):
    """Gate G5 from what Part 163's generator returned (a [rows, prompt + generated] rectangle) and the recorder's
    rows. This is the comparison the launch runs; the tests call it too."""
    if sequences is None: return False
    rectangle=sequences.tolist() if hasattr(sequences,'tolist') else sequences
    legacy=[[int(v) for v in row[int(prompt_length):]] for row in rectangle]
    return len(legacy)==len(recorded_rows) and g5_equal(legacy,[r['tokens'] for r in recorded_rows],eos_ids)

def check_record(path):
    path=Path(path); rec=json.loads(path.read_text(encoding='utf-8')); rows_path=path.with_name('rows.jsonl')
    rows=_load_rows(rows_path); p=smoke_plan() if rec.get('smoke') else plan()
    loads=rec.get('loads',[])
    g1=bool(loads) and all(g1_valid(x) for x in loads)
    failures=[]; bypass={}
    for r in rows:
        key=(r.get('pass'),r.get('row_id')); bypass.setdefault(key,[]).append(r)
        if not g2_valid(r): failures.append(f'G2 {key}')
        if r.get('process')=='float32' and not g4_valid(r): failures.append(f'G4 {key}')
    cut=any('cut_applied' in x for x in loads)
    if any(x.get('a_again')=='not same process' for x in loads) and not any(r.get('pass')=='A-again' for r in rows):
        p['passes']['A-again']={'calls':[],'rows':0}
    if cut:
        p['passes']['F8']['calls']=p['passes']['F8']['calls'][:CUT['F8']]; p['passes']['F1']['calls']=p['passes']['F1']['calls'][:CUT['F1']]
    for pass_name,info in p['passes'].items():
        expected={r['row_id'] for c in info['calls'] for r in c['rows']}
        actual={rid for (name,rid) in bypass if name==pass_name}
        if expected != actual: failures.append(f'G3 rows {pass_name}: expected {len(expected)}, got {len(actual)}')
        for call in info['calls']:
            for slot,r in enumerate(call['rows']):
                found=bypass.get((pass_name,r['row_id']),[])
                if len(found)!=1: continue
                row=found[0]
                if not g3_valid(row,call,slot):
                    failures.append(f'G3 shape {pass_name}/{call["call"]}/{r["row_id"]}')
    a_calls=p['passes']['A']['calls']; a_members={frozenset(c['row_ids']) for c in a_calls}; a_slots={r['row_id']:i for c in a_calls for i,r in enumerate(c['rows'])}
    for c in p['passes']['B']['calls']:
        if not rec.get('smoke') and (frozenset(c['row_ids']) in a_members or any(a_slots[r['row_id']]==i for i,r in enumerate(c['rows']))):
            failures.append(f'G3 B reshuffle {c["call"]}')
    g5=all(x.get('G5',{}).get('passed',True) for x in loads)
    return {'G1':g1,'G2':not any(x.startswith('G2') for x in failures),'G3':not any(x.startswith('G3') for x in failures),
            'G4':not any(x.startswith('G4') for x in failures),'G5':g5,'failures':failures,'rows':len(rows),'ok':g1 and g5 and not failures}

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--units',default=str(UNITS_DEFAULT)); ap.add_argument('--medians',default='results/part112/medians.json'); ap.add_argument('--out',default=str(OUT_DEFAULT)); ap.add_argument('--model',default='E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'); ap.add_argument('--device',default='xpu'); ap.add_argument('--process',choices=tuple(PROCESSES),default='bf16-main'); ap.add_argument('--max-new-tokens',type=int,default=80); ap.add_argument('--print-plan',action='store_true'); ap.add_argument('--check-record'); ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args(argv)
    if a.print_plan:
        p=plan(a.units); dest=Path(a.out)/'plan.json'; _write_json(dest,p); print(hashlib.sha256(dest.read_bytes()).hexdigest()); return 0
    if a.check_record: print(json.dumps(check_record(a.check_record))); return 0
    passes=PROCESSES[a.process]
    if a.smoke and Path(a.out).resolve()==OUT_DEFAULT.resolve():
        print('PART201 a smoke run needs its own --out: it would freeze a smoke plan in results/part201',flush=True); return 3
    try:
        if a.smoke:
            for process_name, process_passes in PROCESSES.items():
                a.process=process_name
                _run_passes(a,process_passes)
        else: _run_passes(a,passes)
    except SystemExit: raise
    return 0
if __name__=='__main__': raise SystemExit(main())
