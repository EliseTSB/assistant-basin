# Part 187: does Part 140's result (holding a persona push gives the most coherent persona answers; releasing it after 8 tokens mostly ends the persona) hold on the eager attention kernel?

Procedure: v1
Registered 2026-11-04 10:28 CT by Claude (Opus 5.5), before any data. Asked by the SDPA audit (Journal ops card
`xpu-batch1-qwen35-audit-2026-11-02`: "next eager reruns ... by public weight (79, 84, 140, 148 first)"; Parts 185 and
186 reran 84, 84b and 79). Part 178 found the A770's fused bf16 SDPA attention leaving the CPU float32 path past 64
keys. Part 140 generated 80-token answers on that kernel at batch 16, so most of each answer its reader judged was
written past key 64. The commit that adds this file is the registration.

## Question

Part 140 generated its answers on the A770 in bf16 with the default SDPA kernel. Do its published results stand when
the same code, units, seeds and dose run on the eager kernel? Published (Set P, unit-level joint success, persona and
coherent): arm A (held, greedy) 48.3%, B (held, sampled) 35.1%, C (released after 8 tokens) 5.8%, D (released after 24)
37.9%, E (repetition penalty) 43.3%, F (norm-preserved) 0.0%; Set C (random directions) at most 6.1% in every arm.
Published verdicts: 140.1 to 140.6 and 140.8 failed (140.5 held for arm A and failed for arm C), 140.7 held.

## Prior art and existing data

- Closest published work: none needed; the question is about our own run. The kernel finding is ours (Part 178;
  `results/kernel-reference/README.md`; Parts 183 to 186).
- Our own data: `results/part140/` (answers, labels, score) is the SDPA side of every comparison. Part 183's kernel
  check passes eager at batch 1, 2 and 8 on Qwen3.5-4B (`results/part183/kernel-check.json`). Parts 183 to 186 found
  every rerun count and verdict unchanged on eager while most answer texts differed, and eager answers ran to the token
  cap more often (Part 185). Part 186's step 0 found the residual-norm medians at layers 13, 14 and 22 within 0.01% of
  the SDPA ones (186.6), the measurement Part 140's dose scales from. Seen before writing: Part 140's published table,
  verdicts and run records; no eager answer to any Part 140 prompt exists.

## Method

Qwen/Qwen3.5-4B at Part 86's revision, bf16, on the A770 (xpu), attention kernel eager. Part 140 does not load through
`neuron_report.load_model`: `part140/run140.py` calls `part115b/howmany115b.load_direct`, which 8544a17 (2026-11-03,
the eager default) missed, so it still loaded transformers' default kernel (SDPA) and recorded nothing. This
registration changes `load_direct` only to pass `attn_implementation` explicitly, eager by default as
`neuron_report.load_model`, and to record the load in `neuron_report.LAST_LOAD` (`part187/test_part187.py` checks both
and fails on the old loader). `part187/run187.py` calls the otherwise unchanged `part140/run140.py` main with the
arguments below, then writes `neuron_report.LAST_LOAD` into `load.json` in the output folder and exits 3 if the kernel
is not eager.

- Generation, as Part 140's: the 240 units of `part140/units140.json` (Set P 120, Set C 60, Set L 60), the six arms A to
  F, both questions, 80 new tokens, Part 140's seeds, Part 86's hook and dose (0.8 times `results/part112/medians.json`,
  reused unchanged on the strength of 186.6), plain rows for every arm: 3,856 rows. Batch 8 for arms A and C to F, not
  Part 140's 16, because batch 8 is the largest padded shape Part 183's kernel check covers; arm B stays at batch 1
  as Part 140 ran it. Out `results/part187/main`.
- Part 140's Gate 1 (do-nothing hook against the plain model, same batch) and Gate 2 (Amendment 1: arm A's first 10
  tokens equal Part 112's stored first-question answer on at least 80% of the seeded 24 units) run inside `run140.py`
  as before.
- Reading, as Part 140's: `part140/label140.py` unchanged, Part 86's version-2 brief. Reader gates first (DeepSeek V4.1
  Flash and Luna each at least 11 of 12 on `part140/known140.json`), then DeepSeek on every pushed answer and Luna on the
  seeded 10%, blind to unit, set and arm in shuffled order. Scores by `part140/score140.py` unchanged, out
  `results/part187/score.json`.

Kernel check: results/part183/kernel-check.json

## Controls

- The kernel check passes for eager at batch 1, 2 and 8 (Part 183), and its SDPA control fails on this model, so the
  check can see the fault.
- Part 140's own controls are kept unchanged: the do-nothing hook (Gate 1), Set C's matched random directions, the plain
  rows in every arm, the reader gates and the joint-success kappa.
- Not used: new units, arms, doses or questions, because this part reruns a registered run and changes only the kernel
  and the batch size.

Gate: `load.json` records `attn_implementation` eager; Gate 1 passes; Gate 2 reaches at least 80%; both reader gates
pass and kappa on joint success is at least 0.60. If any fails, the main numbers are not read.

## Size

- Smallest effect worth knowing: arm A's Set P joint success moving by more than about 12 points (its published 95%
  interval is 41.7 to 55.0), or any published verdict flipping. The n are the originals'.
- Total: 3,856 generations, about 110 minutes on the A770 (Part 140 took 61 minutes at batch 16 on SDPA; batch 8 on
  eager is slower), then 3,840 DeepSeek labels, 384 Luna labels and 24 known-example labels.
- Pilot: none beyond the gates, because the code, units and readers are Part 140's and have run end to end; the gates
  stop the reading if anything differs.

## Optimization

- Bound resource: GPU (the A770).
- Measured: Part 140's main run, 3,856 rows in 61 minutes at batch 16 on SDPA (`results/part140/timing.jsonl`); Part 186
  on eager, 1,536 rows of 64 tokens in about 43 minutes at batch 8.
- Levers: batch 8 taken (the largest padded shape the kernel check covers); batch 16 declined (not covered); fewer rows
  declined (the n are the originals'); max_new_tokens declined (80, the originals'); arm B's batch 1 kept (row-stable
  sampling, as published).

## Predictions

- 187.1 (arm A reproduces): arm A's Set P joint success is between 36.0% and 60.0% (published 48.3%). Fails outside.
  (high)
- 187.2 (release ends the persona): arm C's Set P joint success is at most 15.0% (published 5.8%). Fails above. (high)
- 187.3 (random directions stay low): Set C's joint success is at most 10.0% in every arm (published at most 6.1%).
  Fails if any arm exceeds it. (high)
- 187.4 (the verdicts stand): `score140.py` gives the same verdict as published on at least 7 of the 8 predictions
  140.1 to 140.8, counting 140.5's two arms separately as published (9 verdicts, at least 8 the same). Fails at 2 or
  more flips. (high)
- 187.5 (no verdict flips): all 9 published verdicts are the same. Fails at any flip. (medium; 140.5's arm C margin was
  5.8 points against a 10-point bar, and 140.3's 10.8 against 20)
- 187.6 (arm F stays broken): arm F's Set P joint success is at most 5.0% (published 0.0%). (high)

## Reading rules

- Every number comes from `part140/score140.py` run on `results/part187/main/answers.jsonl` and its labels, the fields
  Part 140 published. 187.1 to 187.3 and 187.6 read `table_per_arm_set`; 187.4 and 187.5 read `predictions`.
- If 187.1 to 187.4 and 187.6 hold, Part 140's published claim stands on the eager kernel and its front-page lines gain
  one line saying so with this part's numbers. Any failure is reported beside the published number, and the front page is
  corrected with both numbers and the kernel named.
- Descriptive, not registered: per arm and set, how many answers reach the 80-token cap on eager and on SDPA, and the
  coherent and repetition rates beside the published ones. Sampled and greedy answers are not expected to match SDPA's
  text (batch shape and kernel both changed).

## Limits

One model, one GPU, the originals' units, arms and dose. The eager kernel is checked against the CPU on eight prompts at
three batch shapes up to about 110 keys, not on these prompts. The batch shape (8) differs from the original's (16),
and this run cannot separate the kernel from the batch size. The dose reuses Part 112's SDPA medians, which Part 186
found unmoved at three layers, not at every layer. The readers are the originals' (DeepSeek V4.1 Flash, Luna); DeepSeek
V4.1 Flash moved to a later checkpoint under the same id on 2026-10-19, before Part 140 ran, so both runs read with the
same served model unless it changed again. English only.
