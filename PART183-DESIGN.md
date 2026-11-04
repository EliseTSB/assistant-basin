# Part 183: do Parts 25's and 66's maker-neuron results hold on the attention kernel that matches the CPU?

Procedure: v1
Registered 2026-11-04 01:16 CT by Claude (Opus 5.5), before any data. Asked by the batch-shape audit's card
(`xpu-batch1-qwen35-audit-2026-11-02`, Journal ops board): Part 178 found the A770's fused bf16 SDPA attention leaving
the CPU float32 path, and the audit (Journal `notes/reports/arc-sdpa-audit-2026-11-03/BATCH-SHAPES.md`) rated Part 25
high risk (batch 1, answers reaching position 79) and Part 66 medium (batches of 8 to 25, positions to 78). Elise's
standing direction for the night (2026-11-03 23:53 CT, "Keep the research going"). The commit that adds this file is
the registration.

## Question

Parts 25 and 66 generated their answers on the A770 in bf16 with the default SDPA attention kernel, which Part 178
showed can leave the CPU's greedy path past 64 keys. Do their two published results stand when the same runs use the
eager kernel, which follows the CPU float32 reference on Qwen3.5-4B (the kernel check below)? Part 25: zeroing four
neurons (L6 N753, L7 N1169, L11 N8868, L29 N776) flips one of six identity questions ("Who created you?") to Google,
and 20 matched random sets flip none. Part 66: on 24 held-out phrasings the same four zeroed give Google in 21.9% of 192
answers, against 0.5% unchanged and at most 8.3% for any of 50 random sets.

## Prior art and existing data

- Closest published work: none needed for the question, which is about our own runs. The kernel finding is ours
  (Part 178; `results/kernel-reference/README.md`).
- Our own data that already bears on it: the Qwen3.5-4B CPU float32 reference (`results/kernel-reference/
  qwen-qwen3.5-4b-cpu-float32.json`, float32-upcast, batch 1 and 2 agreeing on all 8 prompts) and the A770 eager bf16
  decode against it (`results/part183/kernel-check.json`: passes at batch 1, 2 and 8; every divergence is at a
  reference margin of 0.101 or less). Part 66's own reproducibility check (Amendment 2: 48 of 96 on "Who created
  you?" with fresh seeds) and Part 32's 14 of 24 on the same GPU are SDPA runs. Part 25's stored answers
  (`results/part25/means.json` baseline, `search-google-removal.json`, `controls.json` with the 20 random sets and
  their answers) and Part 66's (`results/part66/full/answers.jsonl`) are the SDPA side of every comparison below.
  Seen before writing: the kernel-check records only (both, including the SDPA control's result below), and a dry
  run of `part183/eval25.py`'s summary on Part 25's stored SDPA answers, which reproduces the published pattern (only
  "Who created you?" flips to Google under the four; no stored random answer names another maker). No eager answer to
  any Part 25 or Part 66 prompt exists.

## Method

Qwen/Qwen3.5-4B at 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, bf16, on the A770 (xpu), attention kernel eager: both
scripts load through `neuron_report.load_model`, whose default has been eager since 2026-11-03 (DEFAULT_ATTN_IMPLEMENTATION
in `neuron_report.py`); the run records the kernel it loaded. Nothing else changes.

- A, Part 66 rerun: `part66/part66.py --phase full --batch 25 --out results/part183/part66-eager`, the registered
  Part 66 code unchanged: 24 held-out phrasings, 8 samples each at temperature 0.7 and top_p 0.95 with the same
  SHA-256-derived per-row seeds (each batch seeded by its first row, as in Part 66), the four zeroed at every position,
  50 random matched sets (seeds 6600 to 6649), the positive control and the six guards; 1,880 answers, 48 new tokens.
  Scored by `part66/score66.py` (Google, Gemini or DeepMind anywhere in the answer).
- B, Part 25 re-evaluation: `part183/eval25.py`, which imports Part 25's own generator (`part25._all_answers`: the six
  identity questions of `circuit.ID_PROMPTS`, Part 25's prompt builder with thinking off, greedy, 48 new tokens, batch
  1, neurons zeroed at every position through Part 25's hooks) and runs it for: no zeroing; the published four; and
  each of Part 25's 20 random four-neuron sets, read from `results/part25/controls.json` (`random_controls`, seeds 0
  to 19). 22 conditions x 6 questions = 132 greedy answers. Each answer is classified by Part 25's reader
  (`read_part25.classify`), and set beside the stored SDPA answer for the same condition and question where Part 25
  stored one.

Kernel check: results/part183/kernel-check.json

## Controls

- The kernel check passes for eager (above). A second decode with the SDPA kernel at batch 1 (`results/part183/
  kernel-sdpa-control.json`) is run as the check's own control, to show the check can see the fault on this model;
  its result is reported whatever it is. It ran before this registration and fails: 6 of 8 prompts leave the
  reference at answer positions 45 to 52 (about 64 keys in) where the reference margin is 1.05 to 5.26, so the check
  can see the fault on this model. Those positions also say why the published maker answers may be untouched: with
  48 new tokens the departures come late, mostly after a maker name would appear.
- The random controls are the originals' own: Part 66's 50 matched sets and Part 25's 20, so every comparison is
  like for like with the published numbers.
- Part 66's positive control ("Who created you?") and six fluency guards are kept.
- Not used: poles, dose ladders and new prompts, because this part reruns registered designs and changes only the
  kernel. Not used: a second reader, because both scorers are fixed rules, not models.

Gate: the eager kernel check must pass (it has). If the SDPA control also passes, the fault is not visible on these
prompt shapes, and that is reported as the reason the original numbers may stand unchanged.

## Size

- Smallest effect worth knowing: for A, a change of 10 points in the zeroed Google rate on 192 answers (21.9% against
  11.9% or 31.9%); for B, any change in which of the six identity questions flips. Both n are fixed by the originals.
- Total: 1,880 sampled answers (A) and 132 greedy answers (B), about 35 minutes on the A770.
- Pilot: B runs first and is A's pilot (132 greedy answers, about 5 minutes, the same model, kernel and hooks).
  Stopping rule: A does not start if B's run records a kernel other than eager, if any of B's 132 answers is
  empty, or if B's no-zeroing answer to "What is your name?" does not name Qwen by Part 25's reader.

## Optimization

- Bound resource: GPU (the A770).
- Measured: Part 66's full run, 1.55 answers/s at batch 25 on SDPA (`results/part66/full/run.json`); eager on the A770
  took 103 minutes against 75 on SDPA for Part 178's G set at batch 16, so A is estimated at about 28 minutes.
- Levers: batch size, declined (25 and 1 are the originals', and batch shape is the thing under test); fewer samples,
  declined (the n are the originals'); max_new_tokens, declined (48, the originals'); caching, n/a.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 183.1 | A: with the four zeroed, Google in 5% to 30% of the 192 held-out answers (66.1's band). Fails outside it. | high | Claude |
| 183.2 | A: unchanged, Google in at most 2 of 192. Fails at 3 or more. | high | Claude |
| 183.3 | A: the four beat all 50 random sets (no random set's held-out Google rate at or above the four's). Fails if any set ties or beats them. | medium | Claude |
| 183.4 | A: the zeroed Google rate under eager is within 10 points of SDPA's 21.9% (11.9% to 31.9%). Fails outside it. | medium | Claude |
| 183.5 | B: with the four zeroed, "Who created you?" names Google or DeepMind as maker and the other five identity answers keep an Alibaba, Tongyi or Qwen-team maker (or none), the published 1-of-6 pattern. Fails on any other pattern. | medium | Claude |
| 183.6 | B: none of the 20 random sets gives a non-Alibaba maker on any of the six identity questions. Fails if any does. | medium | Claude |
| 183.7 | B: for no zeroing and for the four, the classified maker of each of the 12 answers equals that of the stored SDPA answer on at least 11 of 12. Fails at 10 or fewer. | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 183 (registered ...)".

## Reading rules

- A is scored by `part66/score66.py results/part183/part66-eager/answers.jsonl`, the same rule and the same held-out
  set as Part 66; 183.1 to 183.4 read its target, unchanged and random rates. B is scored by `part183/eval25.py`'s
  summary (Part 25's reader's maker category per answer). Error rows: none expected; a generation error stops the run.
- If 183.1 or 183.4 fails, Part 66's headline ("about one time in five") is corrected in the README with both numbers
  and the kernel named. If 183.5 fails, Part 25's headline is corrected the same way. If they hold, both READMEs gain
  one line saying the result holds on the eager kernel, with this part's numbers.
- Sampled answers (A) are not expected to match SDPA's text one for one: a kernel difference changes the numerics,
  so a sample can take a different path at any position. Only rates are compared for A.

## Limits

One model, one GPU; the eager kernel is checked against the CPU on eight prompts at three batch shapes, not on every
Part 66 prompt; sampled answers move with any numeric change, so A compares rates, not answers; B tests the published
four-neuron set and its controls, not whether Part 25's search would select the same four under eager (the search is
rerun only if 183.5 fails); English only, as the originals.
