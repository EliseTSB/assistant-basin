"""Compare diag178h runs against diag178f's CPU reference (0.8B float32 batch 1; 4B bf16 batch 1) and against the
Arc eager b16 control. Per config: rows identical over the shorter length, confident first splits (reference margin
>= 0.5) with absolute position, near-tie splits, and whether every row in each batch came out the same."""
import json
from pathlib import Path
R = Path(__file__).resolve().parent
D = R.parent / 'results' / 'part178' / 'diag178h'
cpu = {}
for r in json.loads((R / 'diag178f_cpu_ref.json').read_text(encoding='utf-8')):
    cpu[(r['model'], r['question'])] = r
def first_split(a, b):
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
for stage in ('small', 'big'):
    p = D / f'{stage}.jsonl'
    if not p.exists():
        print(f'{stage}: missing'); continue
    rows = [json.loads(l) for l in open(p, encoding='utf-8')]
    eager = {r['question']: r for r in rows if r['attn'] == 'eager'}
    cfgs = sorted({(r['attn'], r['batch'], r['mask']) for r in rows}, key=lambda c: (c[0] != 'sdpa', c[1], c[2]))
    for cfg in cfgs:
        sel = [r for r in rows if (r['attn'], r['batch'], r['mask']) == cfg]
        for refname, refs in (('CPU', {q: cpu.get((r['model'], q)) for r in sel for q in [r['question']]}), ('eager', eager)):
            ident, conf, tie = 0, [], 0
            for r in sel:
                ref = refs.get(r['question'])
                if not ref: continue
                d = first_split(ref['ids'], r['ids'])
                if d is None: ident += 1; continue
                m = ref['margins'][d]
                if m >= 0.5: conf.append((r['prompt_len'] + d, round(m, 2)))
                else: tie += 1
            label = f"{stage} {cfg[0]} b{cfg[1]} {'mask' if cfg[2] else 'nomask'} vs {refname}"
            print(f"{label}: identical {ident}/{len(sel)}; confident {len(conf)} {sorted(conf)}; near-tie {tie}; "
                  f"rows-in-batch identical {sum(r['distinct_rows'] == 1 for r in sel)}/{len(sel)}")
