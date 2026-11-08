# Part 191: does Part 153's result (Part 148's ten same-company neurons name their company on new maker phrasings, L24 N5017 names Google on 20 of 24) hold on the eager attention kernel?

Procedure: v1
Registered 2026-11-07 22:52 CT by Claude (Opus 5.5), before any data; the commit that adds this file to public main is
the registration and its commit time is the registration time. Asked by the SDPA audit (Journal ops card
`xpu-batch1-qwen35-audit-2026-11-02`; `notes/reports/arc-sdpa-audit-2026-11-03/RERUN-PRIORITY.md` lists Part 153 among
the twelve Priority-1 parts, and among those it is one of five whose claim names units). Part 178 found the A770's fused
bf16 SDPA attention leaving the CPU float32 path past 64 keys; Part 153 generated 120-token greedy answers on that kernel
at batch 20, so most of each scored answer was written past key 64. Elise asked on 2026-11-07 (voice call, about 22:36
CT) to keep the A770 busy overnight.

## Question

Part 153 published (README, Part 153): the ten neurons hit their company on 63% of 24 maker phrasings against 11% for
paired random directions, and L24 N5017 names Google on 20 of 24 maker and 7 of 8 identity phrasings. Its four
predictions came out 153.1 held (7 of 10 units at 12 of 24 or more), 153.2 failed (2 of 10 units at 4 of 8 identity
phrasings), 153.3 held (0.629 against 0.113, 5.59 times), 153.4 held (9 of 10 units at 5 of 6 guards). Do these hold
when the same battery runs on the eager kernel?

## Prior art and existing data

- Parts 183 to 187 and 189b reran Parts 18, 18b, 25, 66, 79, 84, 84b, 140 and 148 on the eager kernel; every published
  verdict held except 148.1, which flipped toward the claim (189b). Counts stayed close while many greedy texts changed
  (Part 18: 175 of 192 answers differed, 4 scorer verdicts changed).
- `results/part153/answers.jsonl` (840 rows), `results/part153/score.json`; the units and doses are Part 112's
  (`results/part112/units.json`, `results/part112/medians.json`).
- Kernel check: `results/part183/kernel-check.json` (eager on this model at batch 1, 2 and 8; its SDPA control fails).

## Method

Qwen/Qwen3.5-4B at Part 86's revision, bf16, on the A770 (xpu). `part153/battery153.py` loads through
`neuron_report.load_model`, whose default since 8544a17 is eager and which records the load in `neuron_report.LAST_LOAD`.
`part191/run191.py` calls Part 153's `run()` unchanged with one injected generator: each question's 20 pushed rows (ten
neurons, each followed by its paired random direction, the original row order) are generated in four chunks of 8, 8, 2
and 2 rows with Part 148's `generate` unchanged, because the kernel check covers eager at batch 1, 2 and 8 and not 20.
Every row in a chunk shares one prompt (no padding). The plain row stays a batch of one. Greedy, 120 new tokens, Part
148's hook and dose, Part 153's 40 questions: 840 rows. Out `results/part191/answers.jsonl`; `load.json` beside it
records the kernel and the chunks; the run exits 3 if the kernel is not eager. Scores by `part153/score153.py`
unchanged, out `results/part191`.

## Controls

- The kernel check (above) can see the fault: its SDPA control fails on this model.
- Part 153's own controls unchanged: paired random directions per unit at the same dose, the plain row per question
  (Gate 2), the six guard questions, the selection questions (Gate 3).
- Not used: new units, doses or questions, because this part reruns a registered run and changes only the kernel and the
  batch shape.

Gate: `load.json` records `attn_implementation` eager, and Part 153's three gates pass (840 non-empty unique rows; at
most 2 plain company hits; at least 5 units hit their company on the selection questions). If any fails, the
predictions are not read.

## Size

- Smallest effect worth knowing: the neurons' mean maker hit rate moving by more than about 0.15 from 0.629, a named
  unit losing its company, or any published verdict flipping. The n are the originals' (10 units, 24 maker phrasings).
- Total: 840 generations in 200 generate calls (40 questions x 4 chunks plus 1 plain), 120 new tokens each.
- Pilot: none beyond the gates, because the code, units and doses are Part 153's and have run end to end.

## Optimization

- Bound resource: GPU (the A770).
- Measured: Part 186 on eager, 1,536 rows of 64 tokens in about 43 minutes at batch 8, so about 60 to 90 minutes is
  expected here (longer answers, smaller chunks for 4 of 20 rows).
- Levers: chunks 8, 8, 2, 2 taken (covered shapes only); batch 20 declined (not covered); fewer rows declined (the n are
  the originals'); max_new_tokens declined (120, the original's); the A770 copy engine off (job runner default); host
  memory freed before the load (procedure 4).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 191.1 | 153.1's verdict stands: at least 5 of the 10 neurons hit their company on at least 12 of the 24 maker phrasings. Fails below 5. | high | Claude |
| 191.2 | The neurons' mean maker hit rate is between 0.45 and 0.80 (published 0.629). Fails outside. | high | Claude |
| 191.3 | 153.3's verdict stands: the neurons' mean maker hit rate is at least twice their paired random directions'. Fails below twice. | high | Claude |
| 191.4 | The random directions' mean maker hit rate is at most 0.20 (published 0.113). Fails above. | high | Claude |
| 191.5 | L24 N5017 names Google on at least 16 of 24 maker phrasings and at least 5 of 8 identity phrasings (published 20 and 7). Fails if either is lower. | medium | Claude |
| 191.6 | All four published verdicts are the same (153.1 held, 153.2 failed, 153.3 held, 153.4 held). Fails at any flip. | medium | Claude |

Copied into PREDICTIONS.md under "## Part 191 (registered ...)". 191.6's medium confidence is because 153.2 failed by one
unit (2 of 10 against a bar of 3) and 153.4 held by one (9 of 10 against 8).

## Reading rules

- Every number comes from `part153/score153.py` run on `results/part191/answers.jsonl`, the fields Part 153 published
  (`summary`, `predictions`, `per_unit`).
- If 191.1 to 191.5 hold, Part 153's claim stands on the eager kernel and its front-page lines gain one line saying so
  with this part's numbers. Any failure is reported beside the published number, and the front page is corrected with
  both numbers and the kernel named.
- Descriptive, not registered: how many answers differ in text from Part 153's, and how many reach the 120-token cap on
  each kernel. Greedy answers are not expected to match SDPA's text (kernel and batch shape both changed).

## Limits

- One model, one revision, one dose, the same 40 questions. This tests the kernel, not the claim's generality.
- Chunks of 8, 8, 2 and 2 change the batch shape as well as the kernel; on this card a bf16 greedy answer can depend
  slightly on batch shape, so a small change in counts cannot be assigned to the kernel alone.
- The scorer is a substring rule (`score153.company_hit`) with Part 86's triage; no reader.
