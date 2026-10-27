#!/usr/bin/env python3
"""CPU smoke of the complete Part 80 runner and scorer using a random tiny Qwen2 model."""
from __future__ import annotations
import argparse, tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
ROOT=Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from part80 import part80 as run
from part80 import score80 as score

CACHE=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default=None); a=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(str(CACHE),local_files_only=True)
    config=Qwen2Config(vocab_size=len(tok),hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512,attn_implementation='eager')
    model=Qwen2ForCausalLM(config).eval()
    if a.out: out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    else:
        temp=tempfile.TemporaryDirectory(prefix='part80-smoke-'); out=Path(temp.name)
    bodies=[('smoke-01','A small orange robot crossed a quiet bridge.','discovery'),('smoke-02','A blue bird waited beside the old fountain.','discovery'),('smoke-03','The silver train arrived before the morning rain.','heldout')]
    run.load_bodies=lambda _path:bodies
    run.nr.load_model=lambda _m,_r,_d:(tok,model)
    run.run(SimpleNamespace(model='qwen3-1.7b',out=str(out),device='cpu',limit=3))
    result=score.score_dirs([str(out)])
    print('smoke scorer models:',len(result['models']))
    print('smoke records:',sum(1 for _ in (out/'records.jsonl').open(encoding='utf-8')))
    print('smoke outputs:',', '.join(p.name for p in sorted(out.iterdir())))
if __name__=='__main__': main()
