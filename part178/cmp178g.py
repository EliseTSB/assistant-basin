"""Compare diag178g runs: each Arc config against the CPU float32 batch-16 reference (small) or against Arc eager b16 (big).
For each row: identical over 256 steps, or the first differing step, its absolute position and the reference margin there."""
import json, sys
from pathlib import Path
D = Path('/home/elisepc/green/worktrees/ab-part67/results/part178/diag178g')
def load(name):
    p = D / name
    return {json.loads(l)['problem_id']: json.loads(l) for l in open(p)} if p.exists() else None
def compare(ref, run, label):
    if run is None: print(f'{label}: missing'); return
    ident, diffs = 0, []
    for pid, r in run.items():
        a, b = ref[pid]['ids'], r['ids']
        d = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
        if d is None: ident += 1
        else: diffs.append((ref[pid]['prompt_len'] + d, d, round(ref[pid]['margins'][d], 2)))
    conf = [x for x in diffs if x[2] >= 0.5]
    print(f'{label}: identical {ident}/{len(run)}; confident splits (ref margin >= 0.5) {len(conf)}: {sorted(conf)[:8]}; near-tie splits {len(diffs)-len(conf)}')
cpu = load('small-fp32-sdpa-b16-cpu.jsonl')
for cfg in ('fp32-sdpa-b16', 'bf16-sdpa-b16', 'bf16-eager-b16', 'bf16-sdpa-b1'):
    compare(cpu, load(f'small-{cfg}-xpu.jsonl'), f'small {cfg} vs CPU fp32')
eager = load('big-bf16-eager-b16-xpu.jsonl')
if eager:
    for cfg in ('bf16-sdpa-b16', 'bf16-sdpa-b1'):
        compare(eager, load(f'big-{cfg}-xpu.jsonl'), f'big {cfg} vs Arc bf16 eager b16')
