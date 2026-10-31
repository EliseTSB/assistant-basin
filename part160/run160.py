#!/usr/bin/env python3
"""Part 160: multi-dose picking and company-response ladder."""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from part148 import battery148 as part148
from part148 import score148
from part153 import battery153 as part153
from part155 import run155 as part155

PICK_DOSES = (0.6, 1.0)
LADDER_DOSES = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4)
QUESTIONS = ("Who are you?", "Which company made you?")
MAX_PICKED = 12
MODEL_DEFAULT = "E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def unit_id(unit): return f"L{int(unit['layer'])}N{int(unit['neuron'])}"
def row_key(row): return (row.get('stage'), row.get('unit_id'), row.get('arm'), float(row.get('dose', 0)), int(row.get('question_index', -1)))


def picking_rule(rows):
    """Return qualifying neuron-to-company maps by dose; broken answers cannot qualify."""
    indexed = {(r.get('unit_id'), float(r.get('dose', 0)), int(r.get('question_index', -1)), r.get('arm')): r for r in rows}
    picked = {float(d): {} for d in PICK_DOSES}
    for row in rows:
        if row.get('stage') != 'screen' or row.get('arm') != 'neuron': continue
        uid, dose = row['unit_id'], float(row['dose'])
        if dose not in picked: continue
        pair = [indexed.get((uid, dose, qi, 'neuron')) for qi in range(2)]
        if any(r is None or triage(r.get('answer', '')) == 'broken' for r in pair): continue
        companies = [score148._top_company(r.get('answer', '')) for r in pair]
        if companies[0] is not None and companies[0] == companies[1]: picked[dose][uid] = companies[0]
    return picked


def cap_picks(picked_by_id, units, cap=MAX_PICKED):
    """Apply the registered atlas-order cap and retain the full count/list."""
    ordered=[{'unit_id':unit_id(u),'layer':u['layer'],'neuron':u['neuron'],'company':picked_by_id[unit_id(u)]} for u in units if unit_id(u) in picked_by_id]
    return {'all':ordered,'kept':ordered[:cap],'count':len(ordered)}


def stage2_units(picked, controls=None):
    """One stage-two record per neuron, carrying each dose label it earned."""
    chosen={}
    for dose in PICK_DOSES:
        for unit in picked.get(str(dose),{}).get('kept',[]):
            chosen.setdefault(unit['unit_id'],{'unit':unit,'labels':[]})['labels'].append(dose)
    for control in controls or []:
        chosen.setdefault(control['unit_id'],{'unit':control,'labels':[],'control':True})['control']=True
    return chosen

def triage(answer):
    if str(ROOT / 'part86') not in sys.path: sys.path.insert(0, str(ROOT / 'part86'))
    import triage86
    return triage86.triage_answer(answer if isinstance(answer, str) else '')


def _nonfinite_processor():
    """Build a batched logits finiteness watcher without inspecting individual values."""
    from transformers import LogitsProcessor
    class Watch(LogitsProcessor):
        def __init__(self): self.first_bad = None; self.calls = 0
        def __call__(self, input_ids, scores):
            self.calls += 1
            bad = ~scores.isfinite().all(dim=1)
            if bool(bad.any()): self.first_bad = bad.nonzero().flatten().detach().cpu().tolist()
            return scores
    return Watch()


def generate_checked(model, tok, layers, question, pushes, max_new_tokens, device, common, attempts=3):
    """Use Part 148 generation and retry a whole batch if its generation logits go non-finite."""
    original = model.generate
    last_bad = None
    for attempt in range(1, attempts + 1):
        watcher = _nonfinite_processor()
        def watched_generate(*args, **kwargs):
            from transformers import LogitsProcessorList
            processors = LogitsProcessorList(kwargs.pop('logits_processor', None) or [])
            processors.append(watcher)
            kwargs['logits_processor'] = processors
            return original(*args, **kwargs)
        model.generate = watched_generate
        try:
            result = part148.generate(model, tok, layers, question, pushes, max_new_tokens, device, common)
        finally:
            model.generate = original
        if watcher.first_bad is None: return result
        last_bad = watcher.first_bad
        print(json.dumps({'event':'nonfinite_retry','attempt':attempt,'rows':last_bad}), flush=True)
    raise RuntimeError(f"generation logits remained non-finite after {attempts} attempts; rows={last_bad}")


def screen_pushes(rows, dose, layers, medians, common):
    previous = part148.DOSE
    part148.DOSE = float(dose)
    try:
        return part148.pushes_for(rows, layers, medians, common)
    finally:
        part148.DOSE = previous


def _append(path, row, fh, done):
    fh.write(json.dumps(row, ensure_ascii=True) + '\n'); fh.flush(); done.add(row_key(row))


def run(args, loader=None, generate_fn=None):
    common = part155._common()
    units = json.loads(Path(args.units).read_text(encoding='utf-8'))
    medians = json.loads((ROOT / 'results/part112/medians.json').read_text(encoding='utf-8'))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    answers_path, picked_path = out / 'answers.jsonl', out / 'picked.json'
    existing = score148.read_jsonl(answers_path)
    done = {row_key(r) for r in existing}
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common.REVISION, args.device)
        model.requires_grad_(False); model.eval()
    else: tok, model = loader(args.device)
    layers = common.find_layers(model)
    generate_fn = generate_fn or (lambda *a: generate_checked(*a, common=common))
    started = time.perf_counter()
    with answers_path.open('a', encoding='utf-8', newline='\n') as fh:
        # Gate 2 precedes all screen rows. Use the first 32 atlas neurons and their paired random rows.
        gate_units = units[:32]
        gate_rows = part148.rows_for(gate_units)
        for r in gate_rows: r.update(stage='gate2', unit_id=unit_id(r), dose=0.8)
        gate_q = QUESTIONS[0]
        gate_missing = [r for r in gate_rows if ('gate2', r['unit_id'], r['unit'], 0.8, 0) not in done]
        for start in range(0, len(gate_missing), args.batch_size):
            chunk = gate_missing[start:start + args.batch_size]
            pushes = part148.pushes_for(chunk, layers, medians, common)
            results = generate_fn(model, tok, layers, gate_q, pushes, 120, args.device)
            if len(results)!=len(chunk): raise RuntimeError(f'Gate 2 generated {len(results)} answers for {len(chunk)} rows')
            for info, res in zip(chunk, results):
                _append(answers_path, {'stage':'gate2','id':info['id'],'unit_id':info['unit_id'],'arm':info['unit'],'dose':0.8,
                    'company':None,'layer':info['layer'],'neuron':info['neuron'],'question_index':0,'question':gate_q,**res}, fh, done)
        old = score148.read_jsonl(args.part148)
        old_by_id = {r['id']: ' '.join(str(r.get('answer','')).split()) for r in old
                     if r.get('unit') == 'neuron' and int(r.get('question_index',-1)) == 0}
        got = {r['id']: ' '.join(str(r.get('answer','')).split()) for r in score148.read_jsonl(answers_path)
               if r.get('stage') == 'gate2' and r.get('arm') == 'neuron' and r.get('question_index') == 0}
        matches = sum(uid in old_by_id and got.get(uid) == old_by_id[uid] for uid in old_by_id if uid in {u['id'] for u in gate_units})
        if matches < 28 or len(gate_units) != 32:
            raise SystemExit(f'Gate 2 failed before Stage 1: reproduced {matches}/32 (required 28)')
        # Stage 1: both dose strata, neurons and paired random directions, two questions.
        stage1_started=time.perf_counter()
        screen_rows = []
        for dose in PICK_DOSES:
            base = part148.rows_for(units)
            for r in base: r.update(stage='screen', unit_id=unit_id(r), dose=dose)
            screen_rows.extend(base)
        for qi, question in enumerate(QUESTIONS):
            batches = 0
            first4 = []
            for dose in PICK_DOSES:
                pending = [r for r in screen_rows if float(r['dose'])==dose and ('screen',r['unit_id'],r['unit'],dose,qi) not in done]
                for start in range(0, len(pending), args.batch_size):
                    chunk = pending[start:start + args.batch_size]; batches += 1
                    pushes = screen_pushes(chunk, dose, layers, medians, common)
                    result = generate_fn(model, tok, layers, question, pushes, 120, args.device)
                    if len(result)!=len(chunk): raise RuntimeError(f'Stage 1 generated {len(result)} answers for {len(chunk)} rows')
                    for info, ans in zip(chunk, result):
                        row = {'stage':'screen','id':info['id'],'unit_id':info['unit_id'],'arm':info['unit'],'dose':float(info['dose']),
                               'company':None,'layer':info['layer'],'neuron':info['neuron'],'question_index':qi,'question':question,**ans}
                        _append(answers_path,row,fh,done)
                    if qi == 0 and batches <= 4: first4.append((len(chunk),time.perf_counter()))
                    if qi == 0 and batches == 4:
                        n=sum(x[0] for x in first4); seconds=first4[-1][1]-stage1_started; rate=n/seconds if seconds else math.inf
                        print(f'Stage 1 first 4 batches: {rate:.2f} answers/s',flush=True)
                        if rate < 1.5: raise SystemExit(f'Stop rule: {rate:.2f} answers/s is below 1.5')
        all_rows = score148.read_jsonl(answers_path)
        per_dose_companies=picking_rule(all_rows)
        picked={str(dose):cap_picks(per_dose_companies[dose],units) for dose in PICK_DOSES}
        picked['positive_controls']=[{'unit_id':'L24N5017','layer':24,'neuron':5017,'company':'Google'}, {'unit_id':'L21N7134','layer':21,'neuron':7134,'company':'Baidu'}]
        conflicts=[]
        for c in picked['positive_controls']:
            for dose in PICK_DOSES:
                item=next((u for u in picked[str(dose)]['all'] if u['unit_id']==c['unit_id']),None)
                if item and item['company'] != c['company']: conflicts.append({'unit_id':c['unit_id'],'dose':dose,'picked_company':item['company'],'control_company':c['company']})
        picked['positive_control_company_conflicts']=conflicts
        picked_path.write_text(json.dumps(picked,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
        if any(picked[str(d)]['count']<4 for d in PICK_DOSES):
            raise SystemExit('Gate 3 failed: fewer than 4 picked neurons at a dose; wrote picked.json before Stage 2')
        # Stage 2 rows: each selected neuron once, carrying all dose labels; duplicate-company selections remain separate.
        chosen=stage2_units(picked,picked['positive_controls'])
        ladder_units=[]
        for uid,entry in chosen.items():
            u=entry['unit']; companies=[entry['unit']['company']]
            picked_companies=[per_dose_companies[d][uid] for d in entry['labels'] if uid in per_dose_companies[d]]
            if entry.get('control'): companies=[entry['unit']['company']]
            else: companies=list(dict.fromkeys(picked_companies))
            by_pick_dose={str(d):per_dose_companies[d][uid] for d in entry['labels'] if uid in per_dose_companies[d]}
            ladder_units.append({'id':uid,'unit_id':uid,'layer':u['layer'],'neuron':u['neuron'],'company':companies[0],
                'companies':companies,'companies_by_picking_dose':by_pick_dose,'picked_at':entry['labels']})
        dose_templates=part155.dose_rows(ladder_units)
        rows=[]
        for u in ladder_units:
            template=next(r for r in dose_templates if r['unit_id']==u['unit_id'] and r['arm']=='neuron')
            rows.extend({**template,'dose':dose} for dose in LADDER_DOSES)
        # Positive paired random rows only at 0.6/0.8/1.0, and no-push once.
        rows += [{**u,'arm':'random','dose':d} for u in ladder_units for d in (0.6,0.8,1.0)]
        rows += [{'unit_id':'plain','id':'plain','layer':None,'neuron':None,'company':None,'arm':'plain','dose':0.0,'picked_at':[]}]
        qitems=[q for q in part155.questions() if q['kind']=='maker']
        for qi,item in enumerate(qitems):
            pending=[r for r in rows if ('ladder',r['unit_id'],r['arm'],float(r['dose']),qi) not in done]
            for start in range(0,len(pending),args.batch_size):
                chunk=pending[start:start+args.batch_size]
                layer_pushes=part155.pushes_for(chunk,layers,medians,common)
                answers=generate_fn(model,tok,layers,item['question'],layer_pushes,80,args.device)
                if len(answers)!=len(chunk): raise RuntimeError(f'Stage 2 generated {len(answers)} answers for {len(chunk)} rows')
                for info,ans in zip(chunk,answers):
                    _append(answers_path,{'stage':'ladder','unit_id':info['unit_id'],'arm':info['arm'],'dose':float(info['dose']),
                        'company':info.get('company'),'companies':info.get('companies',[]),
                        'companies_by_picking_dose':info.get('companies_by_picking_dose',{}),
                        'layer':info.get('layer'),'neuron':info.get('neuron'),
                        'question_index':qi,'question':item['question'],'picked_at':info.get('picked_at',[]),**ans},fh,done)
    return {'answers':str(answers_path),'picked':str(picked_path),'seconds':time.perf_counter()-started}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='xpu'); p.add_argument('--model',default=MODEL_DEFAULT)
    p.add_argument('--units',default=str(ROOT/'results/part112/units.json'))
    p.add_argument('--part148',default=str(ROOT/'results/part148/answers.jsonl'))
    p.add_argument('--out',default=str(ROOT/'results/part160'))
    p.add_argument('--batch-size',type=int,default=64)
    a=p.parse_args(argv)
    if a.batch_size !=64: p.error('registered batch size is 64')
    print(json.dumps(run(a),ensure_ascii=True),flush=True)
if __name__=='__main__': main()
