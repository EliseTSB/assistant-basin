#!/usr/bin/env python3
"""Part 79 counts reported in the README beyond score79.py: written after the cold check, from the same files.

Per condition, over held-out answers: reader-coherent answers, Google among coherent (reader maker), coherent answers
whose text names Alibaba or Tongyi (fixed regex), answers naming any company (reader maker google or other_company, or
the regex), answers cut off at 64 tokens, answers containing '</think>', keyword-only Google rows (keyword yes, reader
no), and the residual-stream push size |delta| x column norm for every pushed unit."""
import json,re,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part78.score78 import classify
CO=re.compile(r'alibaba|tongyi|通义|阿里',re.I)
def main(d):
    d=Path(d)
    key=json.loads((d/'reader_key.json').read_text(encoding='utf-8'))
    lab={x['id']:x for x in map(json.loads,(d/'reader.jsonl').read_text(encoding='utf-8').splitlines())}
    rows={(r['condition'],r['question_index'],r['sample']):r for r in map(json.loads,(d/'answers.jsonl').read_text(encoding='utf-8').splitlines()) if r['question_kind']!='guard'}
    by=collections.defaultdict(list)
    for i,(c,q,s) in key.items(): by[c].append((lab[i],rows[(c,q,s)]))
    out={}
    for c in sorted(by):
        L=by[c]; coh=[(l,r) for l,r in L if l['coherent']]
        out[c]={'coherent':len(coh),'google_among_coherent':sum(l['maker']=='google' for l,_ in coh),
                'alibaba_or_tongyi_among_coherent':sum(bool(CO.search(r['answer'])) for _,r in coh),
                'any_company':sum(l['maker'] in ('google','other_company') or bool(CO.search(r['answer'])) for l,r in L),
                'reader_maker':dict(collections.Counter(l['maker'] for l,_ in L)),
                'cut_at_64':sum(r['stop_reason']=='max_new_tokens' for _,r in L),
                'think_close_tag':sum('</think>' in r['answer'] for _,r in L),
                'keyword_only_google':sum(classify(r['answer'])['google'] and l['maker']!='google' for l,r in L)}
    st=json.loads((d/'step0.json').read_text(encoding='utf-8'))
    out['push_sizes']={k:round(abs(v['delta'])*v['column_norm'],3) for k,v in st['deltas'].items()}
    (d/'extra_counts.json').write_text(json.dumps(out,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    for c,v in out.items(): print(c,v)
if __name__=='__main__': main(sys.argv[1])
