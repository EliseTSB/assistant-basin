# Brief: batched generation for part30.py (Claude, 2026-10-19 22:3x)

The A770 run generates at ~16-20 tokens/s at batch 1 (control step: ~40 s per answer, 900 answers = ~10 h; four
generate steps add ~3 h). The GPU is launch-bound, so batching is the fix. The design does not change: greedy
decoding, the same max_new_tokens (TRAIN_MAX_NEW for control, GENERATE_MAX_NEW for generate), the same prompts and
chat template, the same output record fields.

Do:
1. Add `--gen-batch N` (default 16; 1 = today's exact code path) used by the `control` and `generate` steps.
2. Batched generation with LEFT padding (tokenizer.padding_side='left'; pad id = pad_token_id or eos), an attention
   mask, greedy, the same eos ids. Per row, strip the left padding and decode only that row's new tokens; record
   input_tokens, output_tokens, finish_reason ('stop' if the row's last real new token is an eos, else 'length'; a row
   that finishes early pads with pad/eos afterwards: trim those), and wall_time = the batch's wall time / batch size
   (say so in the header settings: 'wall_time is per-row share of a batch').
3. Keep resumability exactly: rows already in control.jsonl / generate files are skipped by index; write each batch's
   rows in index order as soon as the batch finishes; the header records gen_batch.
4. Sort the remaining questions by prompt length before batching (less padding); keep output order deterministic.
5. Tests on CPU with the tiny model the existing tests use: (a) gen-batch 4 gives the same greedy texts as gen-batch 1
   on 6 short prompts (state it if a row differs and why: padding numerics); (b) resume skips existing indices with
   batching on; (c) finish_reason and output_tokens per row are right when rows end at different lengths.
6. Do not change training, loss, probe or split code. Do not touch PREDICTIONS.md. Do not commit. Write
   REPORT-30-batch.md: SUMMARY, FINDINGS, BLOCKERS, and the exact steps.txt lines to use (add --gen-batch 16 to the
   control and generate lines; no quotes around paths: job.py splits with shlex posix=False, which keeps quotes).
Python for tests: /home/elisepc/neuron-probe/.venv/bin/python (CPU torch). No network, install nothing.
