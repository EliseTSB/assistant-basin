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
def items_for(paths):
    items=[]
    for path in sorted(paths):
        for i,row in enumerate(read_jsonl(path)):
            pub={'id':row['unit_id'],'unit_id':row['unit_id'],'class':row['unit_class'],'arm':row['arm'],
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
    # Part 92b owns the checked 12 known examples and its retrying transport wrapper.
    result=label92b.run_gate(reader)
    return result
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--answers',type=Path,default=ROOT/'results/part131/main')
    ap.add_argument('--units',type=Path,default=ROOT/'part131/units131.json'); ap.add_argument('--reader',choices=('ds','luna','both'),default='both')
    ap.add_argument('--out-dir',type=Path,default=ROOT/'results/part131/labels'); ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args(); files=list(a.answers.glob('*.jsonl'))
    if not files: ap.error(f'no answer JSONL files in {a.answers}')
    # Calibration precedes any reading; don't silently continue after a failed gate.
    ds_gate=known_gate('ds')
    if not ds_gate.get('passed'): raise SystemExit(f'DeepSeek known-example gate failed: {ds_gate.get("correct")}/12')
    items=items_for(files); a.out_dir.mkdir(parents=True,exist_ok=True)
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
