#!/usr/bin/env python3
"""CPU-only smoke of step0 helpers and full Part 79 runner using a deterministic tiny fixture."""
import json,sys,argparse,tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import neuron_report as nr
from part79 import part79,step0,score79

class Tok:
    eos_token_id=2;pad_token_id=0
    def __call__(self,text,add_special_tokens=False,return_tensors=None):
        limit=None if return_tensors is None else 8
        chars=str(text) if limit is None else str(text)[:limit]
        ids=[1]+[3+(ord(c)%15) for c in chars]
        return {'input_ids':torch.tensor([ids]) if return_tensors=='pt' else ids}
    def apply_chat_template(self,messages,tokenize=False,enable_thinking=False):return messages[0]['content']
    def decode(self,ids,skip_special_tokens=True):return 'fixture'

class Down(torch.nn.Module):
    def __init__(self):
        super().__init__();self.in_features=9216;self.weight=torch.nn.Parameter(torch.ones(4,9216))
    def forward(self,x):return torch.nn.functional.linear(x,self.weight)
class Block(torch.nn.Module):
    def __init__(self):super().__init__();self.mlp=torch.nn.Module();self.mlp.down_proj=Down()
    def forward(self,x):return x
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.ModuleList([Block() for _ in range(30)]);self.anchor=torch.nn.Parameter(torch.zeros(1));self.config=type('C',(),{'eos_token_id':2})()
    def forward(self,input_ids,use_cache=False):
        x=torch.ones((*input_ids.shape,9216),device=input_ids.device)
        for b in self.layers:
            if b.mlp.down_proj._forward_pre_hooks:b.mlp.down_proj(x)
            x=b(x)
        return type('Out',(),{'logits':torch.zeros((*input_ids.shape,4))})()
    def generate(self,input_ids,attention_mask=None,**kwargs):
        self(input_ids)
        return torch.cat([input_ids,torch.full((input_ids.shape[0],1),2,device=input_ids.device)],dim=1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('scratch');a=ap.parse_args();scratch=Path(a.scratch);scratch.mkdir(parents=True,exist_ok=True)
    pile=scratch/'pile.jsonl';pile.write_text(json.dumps({'text':'x '*1100})+'\n')
    tok=Tok(); model=Tiny();nr.load_model=lambda *_:(tok,model)
    # Run the complete calibration path with one local window, including the pilot.
    calibration=scratch/'calibration';step0.run(argparse.Namespace(pile=str(pile),out=str(calibration),device='cpu',batch=1,windows=1))
    step0_path=calibration/'step0.json'
    class NormBlock(torch.nn.Module):
        def forward(self,x):
            y=torch.ones_like(x)*2;y[:,0,:]=1000;return y
    norm_layers=torch.nn.ModuleList([NormBlock() for _ in range(23)])
    class NormModel(torch.nn.Module):
        def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1))
        def forward(self,input_ids,use_cache=False):
            x=torch.zeros((*input_ids.shape,1),device=input_ids.device)
            for b in norm_layers:x=b(x)
    measured=step0.summarize_norms(NormModel(),norm_layers,[[1]*1024],1)
    assert measured['L22']['median']==2.0
    out=scratch/'run';part79.run(argparse.Namespace(step0=str(step0_path),out=str(out),device='cpu',batch=8,max_gpu_gb=14,dry_run=False))
    rows=[json.loads(x) for x in (out/'answers.jsonl').read_text().splitlines()]
    runmeta=json.loads((out/'run.json').read_text());result=score79.score_file(out)
    assert len(rows)==1944 and all(result['conditions'][c]['heldout_n']==192 and result['conditions'][c]['guard_n']==24 for c in part79.CONDITION_IDS)
    print('smoke rows',len(rows),'conditions',len(part79.CONDITION_IDS),'hook batches',len(runmeta['hook_batch_evidence']))
    print('hook evidence', {c:[(s['layer'],s['unit'],sum(next((x['calls'] for x in e['push_hooks'] if x['layer']==s['layer'] and x['unit']==s['unit']),0) for e in runmeta['hook_batch_evidence'] if e['condition']==c),max((x['max_abs_diff'] for e in runmeta['hook_batch_evidence'] if e['condition']==c for x in e['push_hooks'] if x['layer']==s['layer'] and x['unit']==s['unit']),default=0)) for s in next(e['push_hooks'] for e in runmeta['hook_batch_evidence'] if e['condition']==c)] for c in part79.CONDITION_IDS if c not in ('e0_unchanged','e1_makers')})
if __name__=='__main__':main()
