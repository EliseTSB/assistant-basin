# Part 140 CPU reference: stored batch-1 rows vs stored batched rows

Run 2026-11-03 on a cloud machine (4 cores, 16 GB RAM, no GPU), branch `cloud/part140-cpu-2026-11-03`.

## Result in one paragraph

On CPU in bf16, **neither the stored batch-1 rows nor the stored batched rows reproduce token for token.** (a) Batch-1 rows: 2 of 40 identical, median first difference at token 48.5 (over the 38 that differ). (b) Batched control rows (stored batch 13 to 16): 0 of 30 identical, median first difference at token 49. So this CPU reference does not single out the batch-1 rows: it disagrees with both kinds about equally, and it cannot show whether the stored batch-1 rows look like the batched ones. Most disagreements begin right where the sequence reaches position 64. With prompts of 16 tokens ("Who are you?") and 17 tokens ("Tell me about yourself."), that is generated token 49 or 48. 38 of 70 rows first differ within 2 tokens of that point (22 of 40 in (a), 16 of 30 in (b)). In 5 more rows, the stored answer stops at exactly that token while the CPU answer keeps going. Prediction 140.7 recomputed with the CPU plain rows still **holds** (0 repeated plain answers in every arm). The verdict does not change.

## What was run

- **Code.** The stored rows were regenerated with run140's own `generate()` and `DeliveryHook`, imported from `part140/run140.py`. Prompts come from `common86.token_inputs`/`build_prompt`: the Part 86 chat template with thinking off and an empty `<think>` block. The model was loaded with `howmany115b.load_direct('cpu')`. Pushes were rebuilt with the same lines as `run140.main`, using `common86.neuron_delta` and `scaled_random_directions` over the full layer's unit list, so Set C directions match. The run reads its calibration from `results/part112/medians.json` (DOSE 0.8). The only change from the Arc run is the device: cpu instead of xpu. The dtype was already bfloat16 in `load_direct`.
- **Model.** Qwen/Qwen3.5-4B, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Settings: max_new_tokens 80, greedy; arm C releases the push after 8 tokens, D after 24; E uses repetition_penalty 1.15; F uses norm-preserve. There is no padding: every row in a batch has the same prompt, expanded, with no attention mask, exactly as in run140. Plain rows use a zero push with no release and no norm-preserve, as in run140.
- **Batch size.** Each chosen row was generated at **batch size 2**: the row plus a discarded companion that is an exact copy of it (same prompt and push). In all 70 runs the companion matched the kept row.
- **Comparison.** The stored answer and the CPU answer (already decoded with `skip_special_tokens` and stripped by `generate()`) were both re-tokenized with `add_special_tokens=False`, and the two id lists were compared.
- **Rows.**
  - (a) Every greedy row (arm != B) stored with batch_size 1: 40 rows. These are 10 plain rows (all at layer 12) and 30 pushed rows from `miss86-L12-N161`, `q112-003` (L14) and `miss86-L18-N3817`, across arms A, C, D, E and F and both questions.
  - (b) Control: every greedy row of 3 units stored at batch size 4 or more. The units are `q112-162` (P, L23, batch 16), `miss86-L21-N3601` (L, L21, batch 13) and `random140-L22-D006` (C, L22, batch 12): 30 rows. No plain row was stored at batch size 4 or more; all 10 plain rows are batch 1, and they are all in (a).
  - Nothing was cut. The 2-row timing test projected about 50 minutes for 70 rows.

## Package versions

Python 3.11.18. torch 2.14.1+cpu, transformers 5.20.0, huggingface_hub 2.2.0, safetensors 0.8.0, numpy 2.4.6 and tokenizers 0.23.3 were all installed at the requested pins. **accelerate 1.15.0 was added (unpinned)**, because `load_direct` passes `device_map` and transformers 5.20 refuses that without accelerate. Transformers warned that `causal_conv1d` and `flash-linear-attention` are not installed, so the linear-attention layers used its reference PyTorch implementation.

## Speed

70 generate calls at batch 2 took 3,174 s, about 5,252 decode steps (estimated from the longer row's length + 1). That is **0.60 s per decode step at batch 2, or 0.30 s per generated token** counting both rows. Each row took 32 to 50 s.

## Results

| Set | Rows | Identical | Median first-diff index (differing rows) | First diff within ±2 of position 64 | Stored stops at position 64, CPU continues | Earlier | Later |
|---|---:|---:|---:|---:|---:|---:|---:|
| (a) batch-1 | 40 | 2 | 48.5 | 22 | 2 | 9 | 5 |
| (b) batched control | 30 | 0 | 49 | 16 | 3 | 8 | 3 |

(In (b), one "stored stops" row stops at token 55, not at position 64.)

- **Plain rows (all in (a)): 0 of 10 identical.** The 5 "Tell me about yourself." plain rows first differ at token 48, which is position 64. The 5 "Who are you?" plain rows differ at tokens 52 to 54, a few tokens after position 64. For example, stored "...How can I assist you today?" and CPU "...How can I help you today?".
- **Identical rows:** `C miss86-L12-N161 Who` and `D q112-003 Tell`, both 53 tokens long.
- **Early divergences in (a):** `q112-003 Who` differs at token 4 in arms A, C, D and E. In the stored rows the model writes "**Q3**"; on CPU it writes something else (see `rows.jsonl`). Other early divergences: E L12 Who at 39, E L18 Tell at 33, F L18 Who at 20, A L14 Tell at 28, F L14 Tell at 43.
- **Early divergences in (b):** tokens 12 to 22, across `q112-162` and `random140-L22-D006`.

Per-row detail is in `rows.jsonl`: identical, first_diff_index, token counts, a diff class, and the first 120 characters of both texts plus the full CPU answer. Summaries are in `compare.json`.

## Prediction 140.7

The rule from `score140.py` is that each arm's plain answers have a repeat rate of at most 2%. Repetition is `score140.repeat` (a phrase of at least 2 words repeated 3 times in a row), and the rate is computed per unit; there is one plain unit per arm. Swapping the 10 greedy plain rows for their CPU versions gives 0 of 2 repeated in arms A, C, D, E and F. Arm B's 6 stored sampled plain rows are kept as stored, because sampling cannot be regenerated: 0 of 6. **140.7 still holds; the verdict is unchanged.** The stored plain rows also have 0 repeats in every arm. The README does not include the "batch-N versus batch-1 plain comparison" the task mentions. Its only batch-related statements are Gate 1 (a same-batch plain/zero-hook check) and Gate 2's note that batch shape changes later tokens.

## Limits

- **The CPU stack is not the Arc stack.** The CPU run uses bf16 CPU kernels and the reference PyTorch linear-attention fallback. The Arc run used XPU kernels, and its transformers version is not recorded in the repo (RECHECK-c7c0290.md also calls it unverified). The clustering of first differences at sequence position 64, in both batch-1 and batched stored rows, points to a difference between the two stacks around that position, not a batch-size effect. It might be in the Arc stack or in the CPU fallback; this run cannot tell which. Until it is explained, a CPU bf16 reference past about 48 generated tokens says little about the batch-size question.
- **Earlier divergences** (token 4 to 43) appear in both sets at similar rates (9 of 40 and 8 of 30). They are consistent with ordinary bf16 rounding differences between devices near ties between top tokens.
- **No float32 reference.** float32 Qwen3.5-4B does not fit in 16 GB, so bf16 was used. The 0.8B float32 agreement mentioned in the brief was not retested here.
- **Batch size.** Rows were regenerated only at batch 2 with an identical companion, not at the stored batch of 12 to 16, and never at batch 1.
- **Sample size.** 40 + 30 rows; plain rows exist only at layer 12; arm B was excluded.

## Script used (not committed separately; reproduced here)

```python
#!/usr/bin/env python3
"""CPU bf16 regeneration of chosen Part 140 greedy rows, using run140's own generate()/DeliveryHook."""
import json,sys,time,argparse
from collections import defaultdict
from pathlib import Path
ROOT=Path('/home/user/assistant-basin'); sys.path.insert(0,str(ROOT/'part140'))
import run140 as r140            # adds part86/part115b/part115/part114 to sys.path
c=r140.c
CONTROL_UNITS=('q112-162','miss86-L21-N3601','random140-L22-D006')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--limit',type=int,default=None); a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    rows=[json.loads(l) for l in open(ROOT/'results/part140/answers.jsonl',encoding='utf-8')]
    greedy=[r for r in rows if r['arm']!='B']
    chosen=[dict(r,group='a') for r in greedy if int(r['batch_size'])==1]
    chosen+=[dict(r,group='b') for r in greedy if r['unit'] in CONTROL_UNITS and int(r['batch_size'])>=4]
    out=Path(a.out); done=set()
    if out.exists(): done={(x['group'],)+tuple(x['key']) for x in map(json.loads,open(out))}
    todo=[r for r in chosen if (r['group'],)+tuple(r140.rowkey(r)) not in done][:a.limit]
    print(f'chosen {len(chosen)} rows, todo {len(todo)}',flush=True)
    from howmany115b import load_direct
    tok,model=load_direct('cpu'); layers=c.find_layers(model); model.eval(); model.requires_grad_(False)
    units=json.loads((ROOT/'part140/units140.json').read_text(encoding='utf-8'))
    medians={int(k):float(v) for k,v in json.loads((ROOT/'results/part112/medians.json').read_text()).items()}
    pushcache={}
    def pushes_for(layer):
        # Same construction as run140.main for one layer.
        if layer in pushcache: return pushcache[layer]
        lu=[u for u in units if int(u['layer'])==layer]
        width=int(layers[layer].mlp.down_proj.in_features); weight=layers[layer].mlp.down_proj.weight; P={}
        rr=[u for u in lu if u['set']=='C']
        rv=c.scaled_random_directions(weight,len(rr),140+layer,c.DOSE*medians[layer]) if rr else None
        for i,u in enumerate(rr): P[u['id']]=rv[i].detach()
        for u in lu:
            if u['set']=='C': continue
            v=torch.zeros(width,device=weight.device,dtype=weight.dtype); n=int(u['neuron']); v[n]=c.neuron_delta(weight,n,medians[layer]); P[u['id']]=v
        pushcache[layer]=(P,width,weight); return pushcache[layer]
    def ids(text): return tok.encode(text,add_special_tokens=False)
    with out.open('a',encoding='utf-8') as f:
        for r in todo:
            layer=int(r['layer']); arm=r['arm']; P,width,weight=pushes_for(layer)
            vec=torch.zeros(width,device=weight.device,dtype=weight.dtype) if r['plain'] else P[r['unit']]
            vecs=torch.stack([vec,vec])   # kept row + discarded companion (a copy), batch size 2
            release=None if r['plain'] else {'C':8,'D':24}.get(arm)   # run140 default --release 8; plain rows have no release
            norm=(arm=='F') and not r['plain']
            t=time.perf_counter()
            ans=r140.generate(model,tok,layers[layer],r['question'],vecs,'cpu',r['seed'],arm,release,80,norm_preserve=norm)
            el=time.perf_counter()-t
            s=ids(r['answer']); g=ids(ans[0])
            fd=next((i for i,(x,y) in enumerate(zip(s,g)) if x!=y), None if len(s)==len(g) else min(len(s),len(g)))
            rec={'group':r['group'],'key':list(r140.rowkey(r)),'arm':arm,'set':r['set'],'unit':r['unit'],'question':r['question'],'seed':r['seed'],'layer':layer,'plain':r['plain'],'stored_batch_size':r['batch_size'],'cpu_batch_size':2,
                 'identical':s==g,'first_diff_index':fd,'stored_tokens':len(s),'cpu_tokens':len(g),'companion_identical':ans[0]==ans[1],
                 'stored_120':r['answer'][:120],'cpu_120':ans[0][:120],'cpu_answer':ans[0],'elapsed_s':round(el,2),'decode_steps_approx':max(len(ids(x)) for x in ans)+1}
            f.write(json.dumps(rec,ensure_ascii=False)+'\n'); f.flush()
            print(r['group'],arm,r['unit'],r['question'][:4],'identical' if rec['identical'] else f'diff@{fd}',f'{el:.0f}s',flush=True)
if __name__=='__main__': main()
```

The summary step classified each row's first difference relative to token index 65 − prompt_length (the first token predicted from position 64). It then recomputed 140.7 with `score140.repeat` and wrote `compare.json` and `rows.jsonl`.
