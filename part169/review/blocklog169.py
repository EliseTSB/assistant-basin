import sys, types
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/home/elisepc/green/worktrees/ab-part67')
for p in (str(ROOT),str(ROOT/'part169')):
    if p not in sys.path: sys.path.insert(0,p)
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from part169 import run169
from part168 import run168
torch.manual_seed(0)
VOCAB=300; EOS=7
cfg=GPT2Config(n_layer=2,n_head=2,n_embd=32,vocab_size=VOCAB,n_positions=256,eos_token_id=EOS,bos_token_id=1)
model=GPT2LMHeadModel(cfg).eval()
model.requires_grad_(False)
class Tok:
    pad_token_id=EOS; eos_token_id=EOS; all_special_ids=[EOS]
    def get_added_vocab(self): return {}
    def decode(self,ids,skip_special_tokens=False): return ' '.join(map(str,ids))
PROMPT=[1,2,3,4]
class NoHook:
    def __init__(self,down_proj,pushes): self.n=0
    def begin_sequence(self): pass
    def remove(self): pass
class Common:
    AddRowsHook=NoHook
    def token_inputs(self,tok,q,n,device): return {'input_ids':torch.tensor([PROMPT]*n,dtype=torch.long)}
    def eos_ids(self,tok,model): return [EOS]
common=Common(); tok=Tok()
def manual(banned_lists,max_new=80):
    """Greedy with bans, no hooks, no cache. Returns per-row token lists (cut at EOS) and per-step (chosen,unbanned_top,rank)."""
    B=len(banned_lists); seqs=[list(PROMPT) for _ in range(B)]; done=[False]*B; out=[[] for _ in range(B)]; steps=[[] for _ in range(B)]
    for t in range(max_new):
        ids=torch.tensor(seqs,dtype=torch.long)
        with torch.inference_mode(): logits=model(input_ids=ids,use_cache=False).logits[:,-1,:].float()
        for i in range(B):
            if done[i]: continue
            sc=logits[i].clone(); top_unbanned=int(sc.argmax())
            for b in banned_lists[i]: sc[b]=-float('inf')
            chosen=int(sc.argmax())
            rank=int((logits[i]>logits[i][chosen]).sum())
            steps[i].append((chosen,top_unbanned,rank))
            out[i].append(chosen); seqs[i].append(chosen)
            if chosen==EOS: done[i]=True
        if all(done): break
    return out,steps
def check(name,answers,banned_lists,ref_tokens,ref_steps):
    ok=True
    for i,a in enumerate(answers):
        if a['token_ids']!=ref_tokens[i]:
            ok=False; print(name,'TOKEN MISMATCH row',i,a['token_ids'][:8],ref_tokens[i][:8])
        exp=[t for t,(c,top,r) in enumerate(ref_steps[i][:len(a['token_ids'])]) if top!=c]
        if a['blocked_steps']!=exp:
            ok=False; print(name,'STEP MISMATCH row',i,a['blocked_steps'][:6],exp[:6])
        expr=[r for t,(c,top,r) in enumerate(ref_steps[i][:len(a['token_ids'])]) if top!=c]
        if a['blocked_ranks']!=expr:
            ok=False; print(name,'RANK MISMATCH row',i,a['blocked_ranks'][:6],expr[:6])
    print(name,'aligned' if ok else 'FAILED', 'rows',len(answers),'rows with a blocked step',sum(1 for a in answers if a['blocked_steps']))
# Path 1: no-push unpushed batch with bans chosen to bite on the greedy path
top0=int(model(input_ids=torch.tensor([PROMPT]),use_cache=False).logits[0,-1].argmax())
bans=[[],[top0],[top0,top0+1,top0+2],list(range(10,60))]
orig=run168.RowBanProcessor
res=run169.with_block_log(run169.generate_unpushed_batch,model,tok,'Q',bans,common,'cpu',80)
print('class restored after unpushed:', run168.RowBanProcessor is orig)
ref,steps=manual(bans)
check('unpushed',res,bans,ref,steps)
# Path 2: battery generate_batch from run168 (uses RowBanProcessor global)
layers={19:SimpleNamespace(mlp=SimpleNamespace(down_proj=torch.nn.Linear(32,32,bias=False)))}
rows=[{'arm':'n+','layer':19,'slot':0}]*4
dirs={(19,0):{'n+':torch.zeros(32)}}
res2=run169.with_block_log(run168.generate_batch,model,tok,layers,'Q',rows,dirs,bans,'cpu',common,80,169)
print('class restored after battery:', run168.RowBanProcessor is orig)
check('battery',res2,bans,ref,steps)
# Path 3: early stop for all rows: EOS is the greedy first token, so every row ends at step 1
eos_bans=[[],[],[],[]]
model.config.eos_token_id=top0
EOS_SAVE=EOS
res3=run169.with_block_log(run169.generate_unpushed_batch,model,tok,'Q',eos_bans,SimpleNamespace(token_inputs=common.token_inputs,eos_ids=lambda t,m:[top0]),'cpu',80)
print('early stop token lengths', [len(a['token_ids']) for a in res3], 'ended_eos', [a['ended_eos_before_80'] for a in res3], 'blocked', [a['blocked_steps'] for a in res3])
