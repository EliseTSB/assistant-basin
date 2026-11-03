#!/usr/bin/env python3
"""Compare the runs written by pos64_cpu.py and the stored Part 140 plain arm-A answers; write results.json."""
import json, sys
from pathlib import Path
from itertools import combinations
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
runs = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
R = {(r['model'].split('/')[-1], r['dtype'], r['batch'], r['question']): r for r in runs}

def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return i
    return None if len(a) == len(b) else min(len(a), len(b))

def compare(ka, kb, model, margin_key):
    out = []
    for q in sorted({k[3] for k in R if k[0] == model}, key=lambda q: [k[3] for k in R].index(q)):
        A = R[(model,) + ka + (q,)]; B = R[(model,) + kb + (q,)]; M = R[(model,) + margin_key + (q,)]
        fd = first_diff(A['row']['ids'], B['row']['ids']); plen = A['prompt_len']
        rec = {'question': q, 'prompt_len': plen, 'len_a': len(A['row']['ids']), 'len_b': len(B['row']['ids']),
               'identical': fd is None, 'first_diff_index': fd}
        if fd is not None:
            rec['abs_position'] = plen + fd   # position the first differing token occupies (it is predicted from abs_position-1)
            rec['length_only'] = fd >= min(len(A['row']['ids']), len(B['row']['ids']))
            mm = M['row']['margins']
            rec['margin_ref'] = mm[fd] if fd < len(mm) else None
            # the reference margin is only meaningful where the reference run shares the prefix up to fd
            rec['margin_ref_valid'] = M['row']['ids'][:fd] == A['row']['ids'][:fd]
            rec['margin_a'] = A['row']['margins'][fd] if fd < len(A['row']['margins']) else None
            rec['margin_b'] = B['row']['margins'][fd] if fd < len(B['row']['margins']) else None
            rec['tok_a'] = A['row']['ids'][fd] if fd < len(A['row']['ids']) else None
            rec['tok_b'] = B['row']['ids'][fd] if fd < len(B['row']['ids']) else None
        out.append(rec)
    return out

res = {'pairs': {}, 'runs': [{k: v for k, v in r.items() if k != 'row'} | {'ids': r['row']['ids'], 'margins': r['row']['margins'], 'text': r['row']['text']} for r in runs]}
ways08 = [('fp32', 1), ('fp32', 2), ('bf16', 1), ('bf16', 2)]
if any(k[0] == 'Qwen3.5-0.8B' for k in R):
    for a, b in combinations(ways08, 2):
        res['pairs'][f'0.8B {a[0]}-b{a[1]} vs {b[0]}-b{b[1]}'] = compare(a, b, 'Qwen3.5-0.8B', ('fp32', 1))
if any(k[0] == 'Qwen3.5-4B' for k in R):
    res['pairs']['4B bf16-b1 vs bf16-b2'] = compare(('bf16', 1), ('bf16', 2), 'Qwen3.5-4B', ('bf16', 1))
    # Stored Part 140 plain arm-A answers vs 4B bf16 (re-tokenized decoded text, add_special_tokens=False, as part140-cpu did)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
    stored = [json.loads(l) for l in open(ROOT / 'results/part140/answers.jsonl', encoding='utf-8')]
    stored = [r for r in stored if r['arm'] == 'A' and r['plain']]
    st = []
    for r in stored:
        s = tok.encode(r['answer'], add_special_tokens=False)
        for bs in (1, 2):
            g_run = R[('Qwen3.5-4B', 'bf16', bs, r['question'])]
            g = tok.encode(g_run['row']['text'], add_special_tokens=False)
            plen = g_run['prompt_len']
            n = min(len(s), len(g)); fd = first_diff(s[:n], g[:n])   # stored rows were capped at 80 new tokens, CPU at 96: compare the common length
            st.append({'question': r['question'], 'seed': r['seed'], 'layer': r['layer'], 'stored_batch_size': r['batch_size'], 'cpu_batch': bs,
                       'stored_tokens': len(s), 'cpu_tokens': len(g), 'identical_over_common_length': fd is None, 'first_diff_index': fd,
                       'abs_position': None if fd is None else plen + fd,
                       'cpu_bf16_b1_margin': None if fd is None or fd >= len(R[('Qwen3.5-4B', 'bf16', 1, r['question'])]['row']['margins']) else R[('Qwen3.5-4B', 'bf16', 1, r['question'])]['row']['margins'][fd],
                       'stored_text': r['answer'], 'cpu_text': g_run['row']['text']})
    res['stored_part140_vs_4B'] = st
summ = {'companion_identical_all': all(r['companion_identical'] for r in runs if r['batch'] == 2)}
for name, rows in res['pairs'].items():
    d = [r for r in rows if not r['identical']]
    summ[name] = {'identical': sum(r['identical'] for r in rows), 'n': len(rows),
                  'first_diffs': [(r['question'], r['first_diff_index'], r['abs_position'], r['margin_ref'] if r['margin_ref_valid'] else None, r['margin_a'], r['margin_b']) for r in d]}
res['summary'] = summ
if len(sys.argv) > 2:   # optional repeat of the 0.8B bf16 runs, to check run-to-run determinism
    rep = [json.loads(l) for l in open(sys.argv[2], encoding='utf-8')]
    res['repeat_0.8B_bf16'] = [{'batch': r['batch'], 'question': r['question'],
                                'identical_to_first_run': r['row']['ids'] == R[('Qwen3.5-0.8B', 'bf16', r['batch'], r['question'])]['row']['ids']} for r in rep]
    summ['repeat_0.8B_bf16_identical'] = f"{sum(x['identical_to_first_run'] for x in res['repeat_0.8B_bf16'])}/{len(rep)}"
json.dump(res, open(HERE / 'results.json', 'w'), indent=1, ensure_ascii=False)
for k, v in summ.items():
    print(k, v) if not isinstance(v, dict) else print(k, f"{v['identical']}/{v['n']}", v['first_diffs'])
if 'stored_part140_vs_4B' in res:
    for r in res['stored_part140_vs_4B']:
        print('stored', r['question'][:6], 'cpu b', r['cpu_batch'], 'stored len', r['stored_tokens'], 'fd', r['first_diff_index'], 'abs', r['abs_position'], 'margin', r['cpu_bf16_b1_margin'])
