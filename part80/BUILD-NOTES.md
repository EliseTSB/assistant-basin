# Part 80 build notes

Built the role-swapping runner, hidden-state distance and mean-state archive, self-reference scorer, CPU unit tests, and a CPU end-to-end smoke using a randomly initialized tiny Qwen2 causal LM. The smoke tokenizer is the cached Qwen3.5-4B tokenizer; no pretrained model weights were loaded. Tests: `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q part80` → `6 passed in 0.72s`. Smoke: `/home/elisepc/neuron-probe/.venv/bin/python part80/smoke_run80.py` → `wrote 3 bodies, 24 records, 2 layers`; scorer saw 1 model and emitted all four output files. Mutation checks made each tested behavior fail: body offsets, R formula, reading-rule layer choice, NOT READ gate, and UTF-8 AST encoding audit.

## Copy to the GPU folder

Copy `part80/part80.py`, `part80/score80.py`, `part80/test_part80.py`, `part80/smoke_run80.py`, `part80/__init__.py`, and `part80/BUILD-NOTES.md` to `/mnt/c/Users/alexa/neuron-probe-gpu/p66basin/part80/`. The destination folder was listed read-only in the assignment, so no files were written there.

## Tokenizer renderings, offline

A local tokenizer snapshot exists for Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Using body `A small orange robot crossed a quiet bridge.` and order 1, `PREFIX=''` (token IDs `[]`), so T is skipped and recorded by the runner. The exact strings are:

```text
A:
<|im_start|>assistant
A small orange robot crossed a quiet bridge.<|im_end|>
<|im_start|>user
Who wrote the message before this question?
A. You (the AI assistant)
B. The user
C. The system
Answer with the letter only.<|im_end|>
<|im_start|>assistant
<think>

</think>



U:
<|im_start|>user
A small orange robot crossed a quiet bridge.<|im_end|>
<|im_start|>user
Who wrote the message before this question?
A. You (the AI assistant)
B. The user
C. The system
Answer with the letter only.<|im_end|>
<|im_start|>assistant
<think>

</think>



S:
<|im_start|>system
A small orange robot crossed a quiet bridge.<|im_end|>
<|im_start|>user
Who wrote the message before this question?
A. You (the AI assistant)
B. The user
C. The system
Answer with the letter only.<|im_end|>
<|im_start|>assistant
<think>

</think>



U':
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
A small orange robot crossed a quiet bridge.<|im_end|>
<|im_start|>user
Who wrote the message before this question?
A. You (the AI assistant)
B. The user
C. The system
Answer with the letter only.<|im_end|>
<|im_start|>assistant
<think>

</think>



T: skipped; PREFIX is empty.
```

The Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` tokenizer was requested with `local_files_only=True`; Transformers reported that it was not cached and offline access could not find the files. Network access/downloads were not used, so its strings and PREFIX could not be generated here. This is the only incomplete build-note item.

## Implementation observations

The cached Qwen3.5 template ends each question with an assistant generation prompt containing an empty `<think>` block even when `enable_thinking=False`; its assistant body itself has no assistant-only prefix. The renderer compares the role-to-body leading separators to derive only the assistant-specific portion. All T score columns are omitted for a run with empty PREFIX, and `run.json` records that omission.
