# Part 186: does Part 79's result (the evil push and the zeroed maker neurons do not compose) hold on the eager attention kernel?

Procedure: v1
Registered 2026-11-04 08:11 CT by Claude (Opus 5.5), before any data. Asked by the SDPA audit (Journal ops card
`xpu-batch1-qwen35-audit-2026-11-02`, needs line: "next eager reruns ... by public weight (79, 84, 140, 148 first)";
Part 185 reran 84 and 84b), under Elise's standing direction for the night (2026-11-03 23:53 CT, "Keep the research
going"). Part 178 found the A770's fused bf16 SDPA attention leaving the CPU float32 path past 64 keys; the audit's
decision-position measurement puts Part 79's reader decisions at a median key of 80, so most of its labels rest on
tokens after key 64. The commit that adds this file is the registration.

## Question

Part 79 generated its answers on the A770 in bf16 with the default SDPA kernel, and measured its push sizes from
residual norms over 1,024-token windows with the same kernel. Do its published results stand when the whole pipeline
(step 0's medians and pilot, then the main run) uses the eager kernel? Published: with the four maker neurons zeroed,
46 of 192 held-out answers name Google by keyword (e1); Lyra's evil push makes 125 of 192 answers villainous by the
blind reader against 0 unchanged (e2, e0); with both, 0 answers name Google and none is both villainous and Google
(e3); with the evil push the model names a company in 1 (e3) and 6 (e2) answers of 192.

## Prior art and existing data

- Closest published work: none needed; the question is about our own run. The kernel finding is ours (Part 178;
  `results/kernel-reference/README.md`; Parts 183 to 185).
- Our own data: `results/part79/main/` (answers, reader labels, step 0) is the SDPA side of every comparison. Part 183's
  kernel check passes eager at batch 1, 2 and 8 on Qwen3.5-4B (`results/part183/kernel-check.json`). Parts 183, 184 and
  185 found Parts 18, 18b, 25, 66, 84 and 84b unchanged in their counts on eager, while most answer texts differed and
  eager answers ran to the token cap more often (Part 185). Seen before writing: Part 79's published table and
  run records; no eager answer to any Part 79 prompt exists.

## Method

Qwen/Qwen3.5-4B at 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, bf16, on the A770 (xpu), attention kernel eager
(`neuron_report.py`'s default since 2026-11-03). `part186/run186.py` calls the unchanged `part79/step0.py` or
`part79/part79.py` main with the arguments below, then writes `neuron_report.LAST_LOAD` into the step's output record.

- Step 0, as Part 79's: median residual norm leaving layers 13, 14 and 22 over the first 400 packed 1,024-token windows
  of `qwen35/pile10k.jsonl` (positions 1 to 1023), batch 1, then the activation deltas for the evil pair and the five
  control pairs, then the greedy pilot (Lyra's four prompts, unchanged and evil-pushed, 128 new tokens). Out
  `results/part186/step0`.
- Pilot gate, as Part 79's: a blind Claude Haiku 5.5 reader (pilot answers shuffled, conditions hidden) labels the
  villain register in at least 3 of the 4 evil answers and in none of the 4 unchanged answers. If it fails, the main run
  does not start.
- Main run, as Part 79's, with this part's step 0: the nine conditions e0 to e8 (unchanged; makers zeroed; evil push;
  both; makers zeroed with each of the five control pairs), Part 66's 24 held-out maker questions with 8 samples and 6
  guards with 4, temperature 0.7, top-p 0.95, Part 79's per-row seeds, 64 new tokens: 1,944 answers. Batch 8, not
  Part 79's 16, because batch 8 is the largest padded shape Part 183's kernel check covers. Out `results/part186/main`.
- Reading, as Part 79's: `part79/score79.py` writes the seed-79 shuffled reader input (no condition labels) and its key;
  six Claude Haiku 5.5 subagents with no project context each label one 288-answer chunk with `part79/READER-BRIEF.md`
  verbatim; the key stays out of the readers' sight until all labels are committed. Keyword counts and verdicts by
  `part79/score79.py`; the extra columns by `part79/extra_counts.py`.

Kernel check: results/part183/kernel-check.json

## Controls

- The kernel check passes for eager at batch 1, 2 and 8 (Part 183), and its SDPA control fails on this model, so the
  check can see the fault.
- Part 79's own controls are kept unchanged: the unchanged condition, the makers-only condition, the five size-matched
  control pairs and the guard questions.
- Not used: new prompts, doses or controls, because this part reruns a registered run and changes only the kernel.

Gate: every output record (step 0 and the main run) records `attn_implementation` eager; the pilot gate above passes;
and Part 79's own reading rule holds (e1 has 26 to 60 Google answers by keyword, every condition has 192 held-out and 24
guard rows, and the push hooks fired on every batch). If any fails, the main numbers are not read.

## Size

- Smallest effect worth knowing: e3 or e2 naming Google or any company in more than a handful of answers (published 0
  and 1, 1 and 6 of 192), or the villain rate in e2 falling below half. The n are the originals'.
- Total: 400 forward passes of 1,024 tokens, 8 pilot answers, then 1,944 sampled answers, about 100 minutes on the
  A770; then 1,736 Haiku labels in seven subagent runs.
- Pilot: step 0's pilot is the pilot. Stopping rule: the main run does not start unless step 0's record shows eager and
  the pilot gate passes.

## Optimization

- Bound resource: GPU (the A770).
- Measured: Part 79's main run, 1,944 answers in 2,853 s at batch 16 on SDPA; Part 185 on eager, 456 answers in 2,733 s
  at batch 1.
- Levers: batch 8 taken (the largest padded shape the kernel check covers); batch 16 declined (not covered); fewer rows
  declined (the n are the originals'); max_new_tokens declined (64, the originals'); caching, n/a.

## Predictions

- 186.1 (control): with the makers zeroed (e1), 26 to 60 of 192 held-out answers name Google by keyword (published
  46). Fails outside that range. (high)
- 186.2 (the evil push reproduces): the blind reader labels at least 96 of e2's 192 answers villainous and at most 9 of
  e0's (published 125 and 0). Fails otherwise. (high)
- 186.3 (no composition): at most 1 of e3's 192 answers is both villainous and Google by the reader (published 0).
  Fails at 2 or more. (high)
- 186.4 (the evil push removes the maker): e3 names a company (reader google or another real company, or the text
  naming Alibaba or Tongyi) in at most 6 of 192 (published 1). Fails at 7 or more. (medium)
- 186.5 (Google falls under the evil push): e3 has at most 3 Google answers by keyword (published 0). Fails at 4 or
  more. (high)
- 186.6 (medians): each of the eager step 0 medians at layers 13, 14 and 22 is within 2% of the published 9.509, 9.857
  and 22.96. Fails if any differs by more. (low)

## Reading rules

- 186.1 and 186.5 read `part79/score79.py`'s keyword counts; 186.2 to 186.4 read its reader counts and
  `part79/extra_counts.py`'s "names any company"; 186.6 reads `step0.json`. These are the fields Part 79 published.
- If 186.1 to 186.5 hold, Part 79's published claim stands on the eager kernel and its front-page section gains one
  line saying so with this part's numbers. A failure is reported beside the published number, and the front page is
  corrected with both numbers and the kernel named. 186.6 says whether the SDPA step 0 had moved the push sizes; a
  failure is reported, not read as a change in Part 79's claim.
- Descriptive, not registered: per condition, how many answers reach the 64-token cap on eager and SDPA, and the
  reader's coherent counts beside the published ones. Sampled answers are not expected to match SDPA's text.

## Limits

One model, one GPU, the originals' neurons and controls. The eager kernel is checked against the CPU on eight prompts
up to about 110 keys, not on 1,024-token windows (step 0) nor on these prompts. The main run's batch shape (8) differs
from the original's (16). The reader is one model family (Haiku), as published; sampled answers move with any numeric
change, so counts are compared, not answers. English only.
