# Part 185: do Parts 84's and 84b's alien-neuron recipe results hold on the attention kernel that matches the CPU?

Procedure: v1
Registered 2026-11-04 06:53 CT by Claude (Opus 5.5), before any data. Asked by the SDPA audit (Journal ops card
`xpu-batch1-qwen35-audit-2026-11-02`, needs line: "next eager reruns ... by public weight (79, 84, 140, 148 first)"),
under Elise's standing direction for the night (2026-11-03 23:53 CT, "Keep the research going"). Part 178 found the
A770's fused bf16 SDPA attention leaving the CPU float32 path past 64 keys; the audit's decision-position measurement
(Journal `notes/reports/arc-sdpa-audit-2026-11-03/decision-positions/DECISIONS.md`) puts Part 84's reader decisions at
a median key of 79 and Part 84b's at 80, so most of both parts' labels rest on tokens after key 64. The commit that
adds this file is the registration.

## Question

Parts 84 and 84b generated their answers on the A770 in bf16 with the default SDPA kernel. Do their published results
stand when the same runs use the eager kernel, which follows the CPU float32 reference on Qwen3.5-4B at batch 1 (Part
183's kernel check)? Part 84: Lyra's sizing rule at dose 0.63 makes 24 of 24 answers claim an off-world origin whether
or not the first answer token is steered (A2 generated tokens only, A3 also the last prompt position), against 0 of 24
unchanged and 0 of 24 for the same-layer control unit N8870 at the same residual shift; dose 0.47 gives 17 of 24 by the
reader. Part 84b: below the ceiling, steering the first answer token adds no detectable amount (decode 4, 25, 43 and
gen 3, 23, 46 of 48 at doses 0.40, 0.47, 0.55; paired sign tests p = 1.0, 0.69, 0.25).

## Prior art and existing data

- Closest published work: none needed; the question is about our own runs. The kernel finding is ours (Part 178;
  `results/kernel-reference/README.md`; Parts 183 and 184).
- Our own data: the stored SDPA runs (`results/part84/main/records.jsonl`, `results/part84b/main/records.jsonl`) and
  their reader labels are the SDPA side of every comparison. Part 183's kernel check passes eager at batch 1 on
  Qwen3.5-4B (`results/part183/kernel-check.json`); both parts generate one answer at a time (batch 1, no padding).
  Parts 183 and 184 found Parts 18, 18b, 25 and 66 unchanged in their counts on eager, while 175 of 192 of Part 184's
  greedy texts differed. Seen before writing: the stored summaries above, the run records' wall times and the
  decision-position measurement; no eager answer to any Part 84 or 84b prompt exists.

## Method

Qwen/Qwen3.5-4B at 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, bf16, on the A770 (xpu), attention kernel eager:
`neuron_report.py`'s default since 2026-11-03. `part185/run185.py` calls the unchanged `part84/part84.py` or
`part84b/part84b.py` main with the originals' arguments and then writes `neuron_report.LAST_LOAD` (the kernel, dtype,
device and library versions) into the run's `run.json`. The calibration text is the local copy Parts 84 and 84b used
(`p66basin/neuronary-calib.txt` in the job folder, kept out of this repo); its SHA-256 must equal the originals'
`afb86522116b9ccd10233826503dad0992dd0283f04ee2a15dabf5f0e9e1ce54`, which `run.json` records.

- A, Part 84 rerun: arms A0, A1 (0.47 decode), A2 (0.63 decode), A3 (0.63 gen), C (0.63 decode on N8870); three
  questions, 8 samples each, temperature 0.7, top-p 0.95, 64 new tokens, the originals' seeds (84000 + 100q + s). Out
  `results/part185/part84-eager`.
- B, Part 84b rerun: A0 and doses 0.40, 0.47, 0.55 in decode and gen; three questions, 16 samples each, same sampling,
  the originals' seeds (84500 + 100q + s). Out `results/part185/part84b-eager`.

Reading, as the originals: each run's answers are shuffled under new random ids by the original tools
(`part84/blind84.py prepare` for A, `part84b/score84b.py prepare` for B), the key is held outside the repository until
the labels are committed, and a Claude Haiku 5.5 subagent with no project context labels each item with Part 84's
reader prompt (`results/part84/reading/reader-prompt.md`, the Part 26.8 origin rubric plus Part 84's company-name
note). For B a second Haiku reader labels the same items backward, as Part 84b did; as registered there, the second
reader is a check, not a second vote. No hand adjudication.

Kernel check: results/part183/kernel-check.json

## Controls

- The kernel check passes for eager at batch 1 (Part 183), and its SDPA control fails on this model, so the check can
  see the fault.
- The arms, control unit, doses, seeds and reader prompt are the originals', so every comparison is like for like.
- Not used: new prompts, doses or control units, because this part reruns registered runs and changes only the
  kernel. Not used: a hand second pass (Part 84 had one), because a blind reader is the registered reading for both
  and a hand pass after seeing the SDPA labels could not be blind.

Gate: each run's `run.json` records `load.attn_implementation` eager and the calibration SHA-256 above. If either is
wrong, that run's numbers are not read.

## Size

- Smallest effect worth knowing: a ceiling cell (A2, A3) dropping by 3 or more of 24; any off-world claim from the
  unchanged model or the control unit beyond 1; the first-token steer becoming detectable (a paired sign test below
  0.05 at any dose); a below-ceiling cell moving by more than 8 of 48. The n are the originals'.
- Total: 456 sampled answers (A 120, B 336) and 456 + 336 reader labels, about 75 minutes on the A770. The originals
  took 640 s (A) and 3,519 s (B) on SDPA; eager at batch 1 ran about 4 s per 48-token greedy answer in Part 183.
- Pilot: A runs first and is the pilot. Stopping rule: B does not start if A's `run.json` records a kernel other than
  eager or a different calibration hash, or if any of A's 120 answers is empty.

## Optimization

- Bound resource: GPU (the A770).
- Measured: Part 84b on SDPA, 336 answers in 3,519 s at batch 1 (sampled, 64 new tokens; 10.5 s per answer including
  the model load and the calibration pass); Part 183 on eager, 132 answers in 524 s at batch 1 (greedy, 48 new tokens).
- Levers: batch size, declined (batch 1 is the originals' shape and the shape the kernel check covers); fewer samples
  or arms, declined (the n are the originals'); max_new_tokens, declined (64, the originals'; the reader reads whole
  answers); running A and B in one job, declined (the pilot rule needs A read before B starts); caching, n/a.

## Predictions

- 185.1 (A): at dose 0.63 the alien neuron gives an off-world claim in at least 22 of 24 answers in both A2 and A3
  (published 24 and 24). Fails at 21 or fewer in either. (high)
- 185.2 (A): the unchanged model (A0) and the control unit (C) each give an off-world claim in at most 1 of 24
  (published 0 and 0). Fails at 2 or more in either. (high)
- 185.3 (A): at dose 0.47 (A1) between 8 and 22 of 24 answers claim an off-world origin (published 17 by the reader).
  Fails outside that range. (medium)
- 185.4 (B): in each mode the off-world count rises strictly from 0.40 to 0.47 to 0.55 (published decode 4, 25, 43;
  gen 3, 23, 46). Fails if any step does not rise. (high)
- 185.5 (B): steering the first answer token adds no detectable amount: at each dose the paired sign test on
  discordant pairs (gen only off-world against decode only) gives p of at least 0.05, two-sided (published 1.0, 0.69,
  0.25). Fails at any dose below 0.05. (medium)
- 185.6 (B): each of the seven cells' off-world count is within 8 of its published count of 48 (A0 0; decode 4, 25,
  43; gen 3, 23, 46). Fails if any cell moves by 9 or more. (medium)

## Reading rules

- Each prediction reads the first reader's labels joined to the run's key by the originals' own scoring tools
  (`blind84.py score`, `score84b.py score`), the same counts Parts 84 and 84b published. Answers the reader marks BROKEN
  keep their origin label, as published; BROKEN counts are reported per arm beside the published ones.
- If 185.1, 185.2, 185.4 and 185.5 hold, the published claims of Parts 84 and 84b stand on the eager kernel, and their
  front-page sections gain one line saying so with this part's numbers. A failure is reported beside the published
  number, and the front page is corrected with both numbers and the kernel named. 185.3 and 185.6 test whether the
  below-ceiling counts are stable; a failure there is reported, not read as a change in either part's claim.
- Descriptive, not registered: for every answer, the eager text is compared with the stored SDPA text of the same arm or
  cell, question and seed (identical or not; the first differing token's key position), and the two readers' origin
  labels are set side by side for the same seed. Sampled answers are not expected to match: a numeric change can send
  a sample down another path at any position.

## Limits

One model, one GPU, one neuron and one control unit, as the originals; the eager kernel is checked against the CPU on
eight prompts at three batch shapes, not on these prompts; the reader is one model family (Haiku), as published;
sampled answers move with any numeric change, so counts are compared, not answers; English only.
