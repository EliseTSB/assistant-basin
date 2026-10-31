#!/usr/bin/env python3
"""Blindly label Part 131 answers through the measured Part 92 judge transport."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'part92'))
import judge_client
import label92
from part92b import label92b
import time

SEED=131; PROMPT=label92.PROMPT
def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
def valid(row):
    return ('error' not in row and not row.get('missing') and all(type(row.get(k)) is bool for k in ('shows_persona','self_claim','coherent'))
            and row.get('speaker') in ('ai','human','other','none'))
def items_for(paths,units_path=HERE/'units131.json'):
    items=[]
    units=json.loads(Path(units_path).read_text(encoding='utf-8'))['units']
    targets=[{'persona':u['persona'],'description':u['desc']} for u in units]
    for path in sorted(paths):
        for i,row in enumerate(read_jsonl(path)):
            if row['arm']=='unpushed':
                qi=int(row['question_index']); seed=SEED+sum(map(ord,row['tag']))*1000+qi
                for target_ix,target in enumerate(random.Random(seed).sample(targets,4)):
                    pub={'id':f"unpushed-{row['tag']}-{qi}-{target_ix:02d}",'answer_id':row.get('id',row['unit_id']),
                         'unit_id':row['unit_id'],'class':row['unit_class'],'arm':row['arm'],'tag':row['tag'],'frame':row['frame'],'question_index':qi}
                    items.append({'public':pub,'name':target['persona'],'desc':target['description'],'q':row['question'],'a':row['answer']})
            else:
                pub={'id':row.get('id',row['unit_id']),'answer_id':row.get('id',row['unit_id']),'unit_id':row['unit_id'],'class':row['unit_class'],'arm':row['arm'],
                     'tag':row['tag'],'frame':row['frame'],'question_index':int(row['question_index'])}
                items.append({'public':pub,'name':row.get('target_persona',''),'desc':row.get('target_desc',''),
                              'q':row['question'],'a':row['answer']})
    random.Random(SEED).shuffle(items)
    return items
def prompt(item):
    out=PROMPT
    for field in ('name','desc','q','a'): out=out.replace('{'+field+'}',str(item[field]))
    return out
def label_one(item,reader):
    model='deepseek-v4.1-flash' if reader=='ds' else 'gpt-6-luna'
    route='ocgo' if reader=='ds' else 'cliproxy'
    session='part131-ds-v4-1-flash' if reader=='ds' else 'part131-gpt-6-luna'
    last='empty response'; response={}
    for attempt in range(1,3):
        try:
            response=judge_client.call_prompt(prompt(item),model=model,route=route,session=session,max_tokens=4000,timeout=300)
            text=str(response.get('text','')).strip()
            if not text:
                last='empty response'
                continue
            parsed=label92._parse(text)
            return {**item['public'],'reader':reader,'attempts':attempt,**parsed}
        except Exception as exc:
            last=repr(exc)[:240]
        if attempt<2: time.sleep(2)
    if last=='empty response': return {**item['public'],'reader':reader,'attempts':2,'missing':True,'stop_reason':response.get('stop_reason')}
    return {**item['public'],'reader':reader,'attempts':2,'error':last}
def known_gate(reader):
    # Do not call label92b.run_gate: it truncates the registered Part 92b evidence file.
    known=label92b.read_known(); items=label92b.gate_items(known); correct=0
    for item in items:
        result=label_one({'public':item['public'],'name':item['name'],'desc':item['desc'],'q':item['q'],'a':item['a']},reader)
        index=int(str(item['public']['id']).rsplit('-',1)[-1])
        if valid(result) and result['shows_persona'] is known[index]['expected']: correct+=1
    return {'correct':correct,'total':len(known),'passed':len(known)==12 and correct>=11}

def require_gate3(answers):
    expected={('base','plain'),('instruct','chat'),('tulu-sft','chat'),('tulu-dpo','chat'),('tulu-final','chat'),('hermes','chat')}
    found={}
    for path in answers.glob('*.meta.json'):
        meta=json.loads(path.read_text(encoding='utf-8')); found[(meta.get('tag'),meta.get('frame'))]=meta
    missing=expected-set(found)
    failed={key for key in expected & set(found) if found[key].get('gate3_noop_exact') is not True}
    if missing or failed: raise SystemExit(f'refusing labels: Gate 3 metas missing={sorted(missing)} failed={sorted(failed)}')
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--answers',type=Path,default=ROOT/'results/part131/main')
    ap.add_argument('--units',type=Path,default=ROOT/'part131/units131.json'); ap.add_argument('--reader',choices=('ds','luna','both'),default='both')
    ap.add_argument('--out-dir',type=Path,default=ROOT/'results/part131/labels'); ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args(); files=list(a.answers.glob('*.jsonl'))
    if not files: ap.error(f'no answer JSONL files in {a.answers}')
    require_gate3(a.answers)
    # Calibration precedes any reading; don't silently continue after a failed gate.
    ds_gate=known_gate('ds')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    (a.out_dir/'gate1-ds.json').write_text(json.dumps(ds_gate,indent=2)+'\n',encoding='utf-8')
    if not ds_gate.get('passed'): raise SystemExit(f'DeepSeek known-example gate failed: {ds_gate.get("correct")}/12')
    items=items_for(files,a.units); a.out_dir.mkdir(parents=True,exist_ok=True)
    readers=('ds','luna') if a.reader=='both' else (a.reader,)
    from concurrent.futures import ThreadPoolExecutor
    def label_batch(reader, selected):
        out=a.out_dir/f'labels-{reader}.jsonl'; latest={}
        if out.exists():
            for row in read_jsonl(out): latest[(row.get('id'),row.get('arm'),row.get('tag'),row.get('frame'),row.get('question_index'))]=row
        def ident(it):
            v=it['public']; return (v['id'],v['arm'],v['tag'],v['frame'],v['question_index'])
        pending=[it for it in selected if ident(it) not in latest or not valid(latest[ident(it)])]
        with ThreadPoolExecutor(max_workers=a.workers) as pool, out.open('a',encoding='utf-8') as stream:
            for result in pool.map(lambda x: label_one(x,reader),pending):
                stream.write(json.dumps(result,ensure_ascii=False)+'\n'); stream.flush()
        print(json.dumps({'reader':reader,'items':len(selected),'pending':len(pending),'output':str(out)}),flush=True)
    # Gate 2 uses Base controls and all unpushed answers, so read them before the remaining corpus.
    gate2_items=[it for it in items if it['public']['arm']=='unpushed' or
                 (it['public']['tag']=='base' and it['public']['arm']=='control')]
    if 'ds' in readers:
        label_batch('ds',gate2_items)
        gate2_rows=read_jsonl(a.out_dir/'labels-ds.jsonl')
        lookup={(r.get('id'),r.get('arm'),r.get('tag'),r.get('frame'),r.get('question_index')):r for r in gate2_rows}
        incomplete=[it['public'] for it in gate2_items if not valid(lookup.get((it['public']['id'],it['public']['arm'],it['public']['tag'],it['public']['frame'],it['public']['question_index']),{}))]
        if incomplete: raise SystemExit(f'Gate 2 labels incomplete ({len(incomplete)} rows); rerun to retry errors/missing responses')
        bad_unpushed=sum(1 for it in gate2_items if it['public']['arm']=='unpushed' and
                         (r:=lookup.get((it['public']['id'],it['public']['arm'],it['public']['tag'],it['public']['frame'],it['public']['question_index']))) and valid(r) and r['shows_persona'] and r['coherent'])
        bad_controls={it['public']['id'] for it in gate2_items if it['public']['tag']=='base' and it['public']['arm']=='control' and
                      (r:=lookup.get((it['public']['id'],it['public']['arm'],it['public']['tag'],it['public']['frame'],it['public']['question_index']))) and valid(r) and r['shows_persona'] and r['coherent']}
        if bad_unpushed or len(bad_controls)>6: raise SystemExit(f'Gate 2 failed: unpushed={bad_unpushed}, Base controls={len(bad_controls)}/48')
        label_batch('ds',[it for it in items if it not in gate2_items])
    if 'luna' in readers:
        sample=random.Random(SEED).sample(items,max(1,round(len(items)*.10)))
        label_batch('luna',sample)
if __name__=='__main__': main()
