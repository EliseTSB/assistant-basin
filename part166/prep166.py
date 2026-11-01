#!/usr/bin/env python3
"""Prepare Part 166 loop words and their single-token spelling candidates."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from part166.score166 import loop_phrase, read_jsonl

def prepare(answers_path, model, out_path):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=True)
    answers=read_jsonl(answers_path); counts=Counter()
    loops=[]; words=set()
    for row in answers:
        answer=row.get('answer','') or ''
        counts.update(w.casefold() for w in __import__('part165.score165',fromlist=['LOOP_WORDS']).LOOP_WORDS.findall(answer))
        phrase=loop_phrase(answer)
        if phrase:
            loops.append({'arm':row.get('arm'),'slot':row.get('slot'),'layer':row.get('layer'),
                          'question_index':row.get('question_index'),'phrase':' '.join(phrase)})
            words.update(phrase)
    stop=[w for w,_ in counts.most_common(50)]
    forms={}
    for word in sorted(words):
        variants=(word,' '+word,word.capitalize(),' '+word.capitalize())
        ids=[]
        for variant in variants:
            encoded=tok.encode(variant,add_special_tokens=False)
            if len(encoded)==1 and encoded[0] not in ids: ids.append(int(encoded[0]))
        forms[word]=ids
    obj={'stop_list':stop,'words':forms,'no_single_token_spelling':[w for w,ids in forms.items() if not ids],
         'loop_rows':loops}
    Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    Path(out_path).write_text(json.dumps(obj,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'loop_answers':len(loops),'words':len(forms),'unspelled':len(obj['no_single_token_spelling'])},ensure_ascii=True))
    return obj

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--answers',default=str(ROOT/'results/part165/answers.jsonl')); p.add_argument('--model',default='/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'); p.add_argument('--out',default=str(ROOT/'results/part166/loop_tokens.json')); a=p.parse_args(argv); prepare(a.answers,a.model,a.out)
if __name__=='__main__': main()
