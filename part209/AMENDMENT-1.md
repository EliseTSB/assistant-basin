# Part 209, Amendment 1: the pilot, before the main run

Written 2026-11-12 13:03 CT by Claude (Opus 5.5), after the pilot and before any main-run data. No prediction, bar or reading rule
changes.

**Runner fix before the pilot produced anything.** The first two A770 pilot launches (12:18 and 12:19 CT) died right
after the model loaded: `write_models_json` imported `fcntl`, which Windows lacks. Fixed in `1e2e2929` (fcntl on
Linux, msvcrt on Windows, as `part134/run134.py` does), with a test that fails on the old code. Those launches wrote
no rows.

**Pilot (design, Size), Qwen3.5-4B on the A770, 12:41 to 12:47 CT; OLMo 2 1B Instruct R2 timing on the CPU, 12:50 to
13:02 CT.** `part209/pilot209.py` wrote `results/part209/pilot/pilot-gates.json`:
- G0: mean P3 0.718 over the 6 told and pole frames, pass.
- G1 R1 half: each told arm's posture wins in both pilot paraphrases (2 of 2 each), pass.
- R3: told sitting above told standing in both pilot paraphrases in ROOM-W and NOBODY; held-out accuracy 0.942 at
  layer 16, pass.
- Told compliance: 12 of 12 told answers labelled with the told posture, pass.
- The 76 answers were read by eye: every one either describes a body position or declines to have one.
- max_new_tokens rule: 56 answers name a posture, 8 of them later than token 48 (latest 78), so max_new_tokens
  stays 96.

**Batch size for R2: 32 for both models.** Qwen on the A770: 0.74 answers/s at batch 16, 1.06 at batch 32. OLMo on the
CPU: 0.21 at 16, 0.31 at 32.

**Batch shape.** The same 76 seeded rows at batch 16 and batch 32 gave 12 of 76 identical texts for Qwen (bf16, A770)
and 76 of 76 for OLMo (float32, CPU). This is the bf16 batch-shape effect Parts 201 and 202 measured; the main run
uses one batch size per model, recorded with its rows, and the write-up states that texts depend on batch shape on the
A770. Labels are what the predictions read.
