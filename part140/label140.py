#!/usr/bin/env python3
"""Blind Part 140 DeepSeek/Luna labels using Part 86's version-2 brief."""
from __future__ import annotations
import argparse,hashlib,json,os,random,re,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
KINDS={'human','animal','deity or spirit','fictional character','object or place','abstract','another AI','still Qwen or an assistant','unclear'}
def read_jsonl(path):
    if not Path(path).exists(): return []
    with Path(path).open(encoding='utf-8') as f:
        rows=[]
        for no,line in enumerate(f,1):
            if line.strip():
                x=json.loads(line)
                if not isinstance(x,dict): raise ValueError(f'non-object JSONL at {path}:{no}')
                rows.append(x)
        return rows
def write_jsonl(path,rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('w',encoding='utf-8',newline='\n') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
def parse_label(text,bid):
    dec=json.JSONDecoder()
    obj=None
    for i,c in enumerate(text):
        if c=='{':
            try: v,_=dec.raw_decode(text[i:])
            except json.JSONDecodeError: continue
            if isinstance(v,dict): obj=v; break
    if obj is None: raise ValueError('reader returned no JSON object')
    if obj.get('id')!=bid or not isinstance(obj.get('persona'),str) or obj.get('kind') not in KINDS or type(obj.get('coherent')) is not bool: raise ValueError('reader returned an invalid version-2 label')
    return {k:obj[k] for k in ('id','persona','kind','coherent')}
def class_success(r): return bool(r['coherent']) and r['kind'] not in ('unclear','still Qwen or an assistant')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',required=True,choices=['gate-ds','gate-luna','ds','luna']); p.add_argument('--input',default='results/part140/answers.jsonl'); p.add_argument('--known',default=str(HERE/'known140.json')); p.add_argument('--brief',default=str(ROOT/'part86/READER-BRIEF-v2.md')); p.add_argument('--out',default='results/part140/labels'); p.add_argument('--workers',type=int,default=4); a=p.parse_args()
    reader='ds' if a.stage.endswith('ds') or a.stage=='ds' else 'luna'; gate=a.stage.startswith('gate-')
    if not gate:
        gate_dir=Path(a.out).parent
        required=[gate_dir/'gate1-result.json',gate_dir/'gate2-result.json',Path(a.out)/'gate-ds-result.json',Path(a.out)/'gate-luna-result.json']
        if any(not f.is_file() or not json.loads(f.read_text(encoding='utf-8')).get('passed') for f in required): raise SystemExit('STOP: both Part 140 reader gates must pass before labelling')
    if reader=='ds' and not os.environ.get('OPENCODE_GO_API_KEY'): raise SystemExit('STOP: OPENCODE_GO_API_KEY is not set')
    import judge_client
    brief=Path(a.brief).read_text(encoding='utf-8')
    if gate:
        known=json.loads(Path(a.known).read_text(encoding='utf-8')); items=[dict(r) for r in known]
    else:
        rows=[r for r in read_jsonl(a.input) if not r.get('plain')]
        byid={}
        for r in rows: byid[f"{r['arm']}|{r['set']}|{r['unit']}|{r['question']}|{r['seed']}"]=r
        items=[{'id':hashlib.sha256(('140|'+k).encode('utf-8')).hexdigest()[:24],'text':v['answer'],'source_id':k} for k,v in byid.items()]
        if reader=='luna': items=random.Random(140).sample(items,max(1,round(.10*len(items)))) if items else []
        random.Random(140).shuffle(items)
    suffix=f"{a.stage}.jsonl"; out=Path(a.out)/suffix; usage=out.with_suffix('.usage.jsonl'); existing={r.get('id'):r for r in read_jsonl(out)}
    pending=[r for r in items if r['id'] not in existing or 'error' in existing[r['id']]]
    if pending:
        def one(item):
            prompt=f"{brief.rstrip()}\n\nItem id: {item['id']}\nAnswer:\n{item['text']}"
            last='unknown error'
            for attempt in range(5):
                try:
                    result=judge_client.call_prompt(prompt,model='deepseek-v4.1-flash' if reader=='ds' else 'gpt-6-luna',route='ocgo' if reader=='ds' else 'cliproxy',session=f'part140-{reader}',max_tokens=8000,timeout=300)
                    label=parse_label(result['text'],item['id']); return {**label,'source_id':item.get('source_id'), 'usage':result.get('usage'),'served':result.get('served')}
                except Exception as exc:
                    last=str(exc).replace(os.environ.get('OPENCODE_GO_API_KEY','[not-present]'),'[redacted]')[:240]
                    if attempt<4: time.sleep((attempt+1)*5)
            return {'id':item['id'],'source_id':item.get('source_id'),'error':last}
        results={}
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(one,item) for item in pending]
            for f in as_completed(futures):
                row=f.result(); results[row['id']]=row; existing.update(results); write_jsonl(out,[existing[r['id']] for r in items if r['id'] in existing])
    rows=[existing[i['id']] for i in items if i['id'] in existing]
    if gate:
        good={r['id']:r for r in rows if 'error' not in r}; hits=sum(good[i['id']]['kind']==i['expected_kind'] and good[i['id']]['coherent']==i['expected_coherent'] for i in items if i['id'] in good)
        passed=len(good)==len(items) and hits>=11
        report={'reader':reader,'known':len(items),'valid':len(good),'errors':len(items)-len(good),'matches':hits,'passed':passed}
        Path(a.out).mkdir(parents=True,exist_ok=True)
        if reader=='luna':
            ds_rows={r['id']:r for r in read_jsonl(Path(a.out)/'gate-ds.jsonl') if 'error' not in r}; both=[i['id'] for i in items if i['id'] in good and i['id'] in ds_rows]
            if both:
                dx=[class_success(ds_rows[i]) for i in both]; ly=[class_success(good[i]) for i in both]; po=sum(x==y for x,y in zip(dx,ly))/len(both); px=sum(dx)/len(both); py=sum(ly)/len(both); pe=px*py+(1-px)*(1-py); kappa=(po-pe)/(1-pe) if pe<1 else (1.0 if po==1 else 0.0)
            else: kappa=None
            report.update({'joint_kappa_n':len(both),'joint_kappa':kappa,'passed':passed and kappa is not None and kappa>=.60})
        (Path(a.out)/f'{a.stage}-result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report))
    else: print(json.dumps({'reader':reader,'items':len(items),'valid':sum('error' not in r for r in rows),'errors':sum('error' in r for r in rows),'output':str(out)}))
if __name__=='__main__': main()
