#!/usr/bin/env python3
"""Part 154 staged zero-dependence and generation runner."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part153 import battery153 as p153  # noqa: E402
from part148 import battery148 as p148  # noqa: E402

MODEL_DEFAULT = 'E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
OPENING = 'I was developed by **'
NAME_TOKENS = ['Alibaba', 'Google', 'OpenAI', 'Anthropic', 'DeepSeek', 'Meta', 'Microsoft', 'Tongyi']
L27_UNIT = {'id': 'L27N3325', 'layer': 27, 'neuron': 3325}
RANDOM = {'id': 'random_L25N4457', 'layer': 25, 'neuron': 4457, 'company': 'Google', 'arm': 'random', 'unit': 'random'}


def zero_sets_from_data(matched):
    import part70.part70 as p70
    sets = [{'id': 'none', 'units': []}, {'id': 'L27N3325', 'units': [dict(L27_UNIT)]}]
    for k in range(4):
        layer, neuron = p70.draw_matched_units(matched, k)[0]
        sets.append({'id': f'matched_{k}', 'units': [{'id': f'L{layer}N{neuron}', 'layer': layer, 'neuron': neuron}]})
    return sets


def zero_sets(matched_path):
    return zero_sets_from_data(json.loads(Path(matched_path).read_text(encoding='utf-8')))


def build_answer_batch(arms, zero_defs, guards=False):
    selected = zero_defs[:2] if guards else zero_defs
    return [dict(arm, _zero=zero) for zero in selected for arm in arms]

def read_done(path):
    done=set(); path=Path(path)
    if path.exists():
        with path.open(encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    row=json.loads(line);done.add((row['unit_id'],row['arm'],row['zero_set_id'],int(row['question_index'])))
    return done

def push_rows():
    return [{**u, 'arm': 'neuron', 'unit': 'neuron'} for u in p153.fixed_units(ROOT / 'results/part112/units.json')] + [RANDOM, {'id':'none','layer':0,'neuron':0,'company':None,'arm':'none','unit':'none'}]


def row_pushes(rows, layers, medians, common):
    active = [r for r in rows if r['arm'] != 'none']
    pushes = p148.pushes_for(active, layers, medians, common)
    import torch
    # Reinsert the no-push row at its original row position; push_for has one row per active condition.
    by_layer = {}
    for layer_no, mat in pushes.items():
        out = torch.zeros((len(rows), int(mat.shape[1])), dtype=mat.dtype, device=mat.device)
        ai = 0
        for ri, row in enumerate(rows):
            if row['arm'] == 'none':
                continue
            out[ri] = mat[ai]
            ai += 1
        by_layer[layer_no] = out
    return by_layer


class ZeroRowsHook:
    """Zero registered channels by row at every token position, auditing the write once per call.

    Vectorized on purpose: a CPU transfer per row in a forward hook hung an A770 job (gotcha, Part 121)."""
    def __init__(self, down_proj, row_units, trace):
        self.torch = __import__('torch')
        self.row_units, self.trace = row_units, trace
        self.rows, self.channels = [], []
        for row, units in enumerate(row_units):
            for unit in units:
                if int(unit['layer']) == 27:
                    self.rows.append(row)
                    self.channels.append(int(unit['neuron']))
        self.zeroed_rows = len(set(self.rows))
        self.other_rows = len(row_units) - self.zeroed_rows
        self.keep = None
        self.handle = down_proj.register_forward_pre_hook(self.hook)

    def hook(self, _module, args):
        torch = self.torch
        x = args[0]
        if x.ndim != 3 or x.shape[0] != len(self.row_units):
            raise ValueError(f'unexpected zero-hook input shape {tuple(x.shape)}')
        if self.keep is None:
            keep = torch.ones((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device)
            if self.rows:
                rows = torch.tensor(self.rows, dtype=torch.long, device=x.device)
                channels = torch.tensor(self.channels, dtype=torch.long, device=x.device)
                keep[rows, 0, channels] = 0
            self.keep = keep
        y = x * self.keep
        target = self.keep == 0
        written = y.masked_select(target.expand_as(y))
        bad_target = (written != 0).sum()
        bad_other = ((y != x) & ~target).sum()
        bad_target, bad_other = torch.stack([bad_target, bad_other]).cpu().tolist()
        trace = self.trace
        positions = int(x.shape[1])
        trace['target_checks'] += self.zeroed_rows
        trace['target_failures'] = trace.get('target_failures', 0) + int(bad_target > 0)
        trace['target_positions'] += self.zeroed_rows * positions
        trace['unchanged_checks'] += self.zeroed_rows
        trace['unchanged_failures'] = trace.get('unchanged_failures', 0) + int(bad_other > 0)
        trace['unchanged_positions'] += self.zeroed_rows * positions
        trace['unzeroed_target_checks'] = trace.get('unzeroed_target_checks', 0) + self.other_rows
        trace['unzeroed_target_failures'] = trace.get('unzeroed_target_failures', 0) + int(bad_other > 0)
        trace['unzeroed_target_positions'] = trace.get('unzeroed_target_positions', 0) + self.other_rows * positions
        trace['calls'] += 1
        trace['positions'] += positions
        return (y,) + tuple(args[1:])

    def remove(self): self.handle.remove()


def _token_ids(tok, text):
    enc = tok(text, add_special_tokens=False, return_tensors='pt')
    ids = enc['input_ids'] if isinstance(enc, dict) else enc.input_ids
    return ids[0].tolist()


def opening_steps(opening_ids):
    """Represent the opening as one-token decode inputs."""
    return [[int(token)] for token in opening_ids]


def stage_a(model, tok, layers, common, zero_defs, medians, out_path):
    import torch
    from part74.part74 import OPENING as P74_OPENING
    from part72.part72 import NAMES as P72_NAMES
    if P74_OPENING != OPENING or P72_NAMES != NAME_TOKENS:
        raise ValueError('Part 72/74 readout does not match registered opening and names')
    # Part 72's readout: the first token of each name, so Alibaba reads as its first token.
    read_ids = [_token_ids(tok, n)[0] for n in NAME_TOKENS]
    questions = p153.DIRECT + p153.INDIRECT
    base_rows = push_rows()
    records, top = [], None
    opening_ids = _token_ids(tok, OPENING)
    if not opening_ids: raise ValueError('opening tokenized to no tokens')
    trace_total = {'calls':0,'positions':0,'target_checks':0,'target_positions':0,'unchanged_checks':0,'unchanged_positions':0,
                   'unzeroed_target_checks':0,'unzeroed_target_positions':0,'target_failures':0,'unchanged_failures':0,'unzeroed_target_failures':0}
    for qi, question in enumerate(questions):
        batch_rows = [dict(r, zero=zero) for zero in zero_defs[:2] for r in base_rows]
        pushes = row_pushes(batch_rows, layers, medians, common)
        inputs = common.token_inputs(tok, question, len(batch_rows), str(next(model.parameters()).device))
        trace = dict.fromkeys(trace_total, 0)
        z_units = [r['zero']['units'] for r in batch_rows]
        captured = []
        def capture(_m, args):
            if args[0].shape[-2] == 1:
                captured.append(args[0][:, -1, 3325].detach().float().cpu().tolist())
        push_hooks = [common.AddRowsHook(layers[ln].mlp.down_proj, push) for ln, push in pushes.items()]
        for hook in push_hooks: hook.begin_sequence()
        cap_handle = layers[27].mlp.down_proj.register_forward_pre_hook(capture)
        zero_hook = ZeroRowsHook(layers[27].mlp.down_proj, z_units, trace)
        try:
            with torch.inference_mode():
                out = model(input_ids=inputs['input_ids'], use_cache=True)
                past, logits = out.past_key_values, out.logits
                for step, one_token in enumerate(opening_steps(opening_ids)):
                    one = torch.tensor([one_token] * len(batch_rows), dtype=inputs['input_ids'].dtype, device=inputs['input_ids'].device)
                    attention = torch.ones((len(batch_rows), int(inputs['input_ids'].shape[1]) + step + 1), dtype=inputs['input_ids'].dtype, device=inputs['input_ids'].device)
                    out = model(input_ids=one, attention_mask=attention, past_key_values=past, use_cache=True)
                    past, logits = out.past_key_values, out.logits
                vocab = logits[:, -1, :].detach().float().cpu()
                selected = vocab[:, read_ids]
                mass = vocab.softmax(-1)[:, read_ids].sum(-1)
                gaps = selected[:, 1] - selected[:, 0]
        finally:
            cap_handle.remove(); zero_hook.remove()
            for hook in push_hooks: hook.remove()
        for key,value in trace.items(): trace_total[key] += value
        if not captured: raise RuntimeError('Stage A did not capture an opening activation')
        acts = captured[-1]
        for ri, row in enumerate(batch_rows):
            rec = {'question_index':qi,'question':question,'push_id':row['id'],'arm':row['arm'],
                   'zero_set_id':row['zero']['id'],'activation_L27N3325_pre_zero':acts[ri],
                   'name_logits':{n:float(selected[ri,ni]) for ni,n in enumerate(NAME_TOKENS)},
                   'google_minus_alibaba':float(gaps[ri]),'name_mass':float(mass[ri])}
            records.append(rec)
            if qi==0 and row['arm']=='none' and row['zero']['id']=='none':
                top_ids=vocab[ri].topk(5).indices.tolist(); probs=vocab[ri].softmax(-1)
                top=[(tok.decode([i]),float(probs[i])) for i in top_ids]
    out_path.parent.mkdir(parents=True,exist_ok=True)
    with out_path.open('w',encoding='utf-8',newline='\n') as fh:
        for rec in records: fh.write(json.dumps(rec,ensure_ascii=False)+'\n')
    return records,trace_total,top


def _generate_batch(model, tok, layers, common, rows, question, medians, max_new_tokens=120):
    import torch
    pushes = row_pushes(rows, layers, medians, common)
    inputs = common.token_inputs(tok, question, len(rows), str(next(model.parameters()).device))
    prompt_len = int(inputs['input_ids'].shape[-1]); eos = common.eos_ids(tok, model)
    trace = {'calls':0,'positions':0,'target_checks':0,'target_positions':0,'unchanged_checks':0,'unchanged_positions':0,'unzeroed_target_checks':0,'unzeroed_target_positions':0,'target_failures':0,'unchanged_failures':0,'unzeroed_target_failures':0}
    row_units = [r['_zero']['units'] for r in rows]
    hooks = [common.AddRowsHook(layers[ln].mlp.down_proj, push) for ln, push in pushes.items()]
    zero = ZeroRowsHook(layers[27].mlp.down_proj, row_units, trace)
    for hook in hooks: hook.begin_sequence()
    try:
        with torch.inference_mode():
            seq = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                 pad_token_id=getattr(tok,'pad_token_id',None), eos_token_id=eos or None)
    finally:
        zero.remove()
        for h in hooks: h.remove()
    eos_set = set(eos); result=[]
    for row in seq[:,prompt_len:].tolist():
        ids = row
        for pos,t in enumerate(row):
            if int(t) in eos_set: ids=row[:pos+1]; break
        result.append({'answer':tok.decode(ids,skip_special_tokens=True).strip(),'token_count':len(ids)})
    return result, trace


def run(args, loader=None, generate_fn=None):
    common = p148._common()
    units_path = Path(args.units); medians = json.loads(Path(args.medians).read_text(encoding='utf-8'))
    units = p153.fixed_units(units_path)
    # The fixed random-direction condition shares Part 148's calibrated push implementation.
    rows0 = [{**u,'arm':'neuron','unit':'neuron'} for u in units] + [RANDOM, {'id':'none','layer':0,'neuron':0,'company':None,'arm':'none','unit':'none'}]
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(args.model, common.REVISION, args.device)
        model.requires_grad_(False); model.eval()
    else: tok, model = loader(args.device)
    layers = common.find_layers(model)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    matched_path = Path(args.matched)
    zero_defs = zero_sets(matched_path)
    # Persist resolved controls before any model generation.
    with (outdir/'zero_sets.json').open('w',encoding='utf-8') as f: json.dump(zero_defs,f,ensure_ascii=False,indent=2); f.write('\n')
    stagea_path = outdir/'stageA.jsonl'
    stagea, trace_a, top = stage_a(model,tok,layers,common,zero_defs,medians,stagea_path)
    if top is not None: print(json.dumps({'stageA_first_phrasing_no_push_top5':top},ensure_ascii=False),flush=True)
    answer_path=outdir/'answers.jsonl'; done=set()
    done=read_done(answer_path)
    # Build registered arms: ten Part 153 neuron pushes, Part 148 random direction at L25N4457, and no push.
    arms=rows0
    qitems=p153.DIRECT+p153.INDIRECT
    guards=p153.GUARDS
    trace_total=dict(trace_a)
    started=time.perf_counter()
    with answer_path.open('a',encoding='utf-8',newline='\n') as fh:
        for qi,q in enumerate(qitems[:2] if args.stage=='pilot' else qitems):
            batch=build_answer_batch(arms,zero_defs)
            if all((rec['id'],rec['arm'],rec['_zero']['id'],qi) in done for rec in batch): continue
            # Keyed resume allows a partially written batch to be rerun and only missing rows appended.
            results, tr = _generate_batch(model,tok,layers,common,batch,q,medians)
            for k,v in tr.items(): trace_total[k]=trace_total.get(k,0)+v
            for rec,answer in zip(batch,results):
                key=(rec['id'],rec['arm'],rec['_zero']['id'],qi)
                if key in done: continue
                row={'unit_id':rec['id'],'arm':rec['arm'],'zero_set_id':rec['_zero']['id'],'question_index':qi,'question':q,'kind':'maker',**answer}
                fh.write(json.dumps(row,ensure_ascii=False)+'\n'); done.add(key)
            fh.flush()
        if args.stage=='main':
            guard_zero=zero_defs[:2]
            for qi,(q,expected) in enumerate(guards):
                batch=build_answer_batch(arms,guard_zero,guards=True)
                if all((rec['id'],rec['arm'],rec['_zero']['id'],len(qitems)+qi) in done for rec in batch): continue
                results,tr=_generate_batch(model,tok,layers,common,batch,q,medians)
                for k,v in tr.items(): trace_total[k]=trace_total.get(k,0)+v
                baseqi=len(qitems)+qi
                for rec,answer in zip(batch,results):
                    key=(rec['id'],rec['arm'],rec['_zero']['id'],baseqi)
                    if key in done: continue
                    row={'unit_id':rec['id'],'arm':rec['arm'],'zero_set_id':rec['_zero']['id'],'question_index':baseqi,'question':q,'kind':'guard',**answer}
                    fh.write(json.dumps(row,ensure_ascii=False)+'\n'); done.add(key)
                fh.flush()
    with (outdir/'trace.json').open('w',encoding='utf-8') as f:
        json.dump({'stageA':trace_a,'stageB':trace_total,'elapsed_seconds':time.perf_counter()-started},f,indent=2);f.write('\n')
    return {'answers':str(answer_path),'stageA':str(stagea_path),'trace':trace_total}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--stage',choices=('pilot','main'),default='pilot')
    ap.add_argument('--device',default='xpu');ap.add_argument('--model',default=MODEL_DEFAULT)
    ap.add_argument('--units',default=str(ROOT/'results/part112/units.json'));ap.add_argument('--medians',default=str(ROOT/'results/part112/medians.json'))
    ap.add_argument('--matched',default=str(ROOT/'results/part68/b/matched.json'));ap.add_argument('--out',default=None)
    args=ap.parse_args(argv);args.out=args.out or str(ROOT/('results/part154-pilot' if args.stage=='pilot' else 'results/part154'))
    print(json.dumps(run(args),ensure_ascii=False),flush=True)
if __name__=='__main__': main()
