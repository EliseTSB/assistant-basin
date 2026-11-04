# Part 185: Parts 84's and 84b's alien-neuron recipe counts hold on the eager attention kernel

Parts 84 and 84b generated their answers on the A770 in bf16. Their loader passed no attention argument, so they ran
transformers' default, which on this model is the fused SDPA kernel; the originals' `run.json` has no load record, so
this is inferred from the loader at the time (neuron_report.py at the commit before 8544a17), not recorded. Part 178 found that
kernel leaving the CPU float32 path after about 64 keys. Their answers were read whole by a blind reader, and most of
those readings rest on tokens after key 64 (median key 79 and 80 in the Arc audit's decision-position measurement), so
both were at risk. Here the same runs were repeated with the eager kernel, which matches the CPU reference at batch 1
except at near-ties (Part 183's kernel check), and read again by blind Claude Haiku 5.5 readers with Part 84's prompt.
Same model revision, neuron (L22 N2742), same-layer control unit (N8870, at the same residual shift), doses, sizing
rule, calibration text, questions, seeds, sampling and 64 new tokens; one answer at a time, as the originals.
Registered in `PART185-DESIGN.md` (ddaecaa) before any data.

| Cell | Eager (this part) | SDPA (published, first reader) |
| --- | ---: | ---: |
| Part 84, A0 unchanged, off-world | 0 of 24 | 0 of 24 |
| Part 84, A1 dose 0.47, generated tokens only | 13 of 24 | 17 of 24 (13 after the hand second pass) |
| Part 84, A2 dose 0.63, generated tokens only | 23 of 24 | 24 of 24 |
| Part 84, A3 dose 0.63, also the last prompt position | 24 of 24 | 24 of 24 |
| Part 84, C control unit N8870 at dose 0.63 | 0 of 24 | 0 of 24 |
| Part 84b, A0 unchanged | 0 of 48 | 0 of 48 |
| Part 84b, decode (generated tokens only) at 0.40, 0.47, 0.55 | 3, 28, 44 of 48 | 4, 25, 43 of 48 |
| Part 84b, gen (also the first answer token) at 0.40, 0.47, 0.55 | 4, 29, 47 of 48 | 3, 23, 46 of 48 |
| Part 84b, gen only / decode only pairs, sign test, at 0.40, 0.47, 0.55 | 2/1 p=1.0; 3/2 p=1.0; 3/0 p=0.25 | 0/1 p=1.0; 2/4 p=0.69; 3/0 p=0.25 |
| Part 84b, all three doses pooled: off-world totals; gen only / decode only | decode 75, gen 80 of 144; 8/3, p=0.23 | decode 72, gen 72 of 144; 5/5, p=1.0 |

Part 84's claims stand on eager: at dose 0.63 the neuron makes almost every answer claim an off-world origin whether
or not the first answer token is steered (A2 23 and A3 24 of 24), and neither the unchanged model nor the control unit
makes such a claim. Part 84b's dose rise stands in both modes. Its other claim, that steering the first answer token
adds no detectable amount, is not contradicted, but this rerun has little power to test it: with 3 to 5 discordant
pairs per dose, no sign test could reach p < 0.05 (the smallest two-sided p with 5 pairs is 0.0625), so prediction
185.5 could not have failed at the counts observed. Pooled over the three doses the direction leans toward gen (+5 of
144, 8 against 3 discordant pairs, p = 0.23) where the SDPA run was level; Part 84b's own front-page power note (a true
excess of 12 detected about half the time) applies here too. The below-ceiling cells moved by up to 6 of 48 (Part 84b,
gen at 0.47) and 4 of 24 (Part 84, A1, against the first reader's count; the hand-checked SDPA count is also 13), within
the bands registered in advance. The answers themselves changed much more (below).

## Predictions

| # | Prediction (as registered, with its confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 185.1 | A: at dose 0.63 the alien neuron gives an off-world claim in at least 22 of 24 answers in both A2 and A3 (published 24 and 24). Fails at 21 or fewer in either. (high) | A2 23, A3 24 | held |
| 185.2 | A: the unchanged model (A0) and the control unit (C) each give an off-world claim in at most 1 of 24 (published 0 and 0). Fails at 2 or more in either. (high) | A0 0, C 0 | held |
| 185.3 | A: at dose 0.47 (A1) between 8 and 22 of 24 answers claim an off-world origin (published 17 by the reader). Fails outside that range. (medium) | 13 | held |
| 185.4 | B: in each mode the off-world count rises strictly from 0.40 to 0.47 to 0.55 (published decode 4, 25, 43; gen 3, 23, 46). Fails if any step does not rise. (high) | decode 3, 28, 44; gen 4, 29, 47 | held |
| 185.5 | B: steering the first answer token adds no detectable amount: at each dose the paired sign test on discordant pairs gives p of at least 0.05, two-sided (published 1.0, 0.69, 0.25). Fails at any dose below 0.05. (medium) | 1.0, 1.0, 0.25; with 3 to 5 discordant pairs per dose p < 0.05 was unreachable, so this could not fail | held (uninformative) |
| 185.6 | B: each of the seven cells' off-world count is within 8 of its published count of 48. Fails if any cell moves by 9 or more. (medium) | largest move +6 (gen 0.47) | held |

Part 84b's own scorer also re-tests its predictions 84b.3 (gen at least decode at every dose, summed excess at least
12) and 84b.4 (dose 0.40: decode at most 16, gen at least 24). Both failed on SDPA as published, and both fail again on
eager (summed difference 5; dose 0.40 decode 3, gen 4), so they read the same either way.

Gate: each run's run.json records `load.attn_implementation` eager and the originals' calibration SHA-256
(afb86522…). Passed for both. A ran first; its kernel was eager, its calibration hash matched and none of its 120
answers was empty, which is the stopping rule as registered (Size), so B was started. The design's Optimization section
also says "the pilot rule needs A read before B starts"; that was not followed: B was launched at 07:10 and relaunched at
07:11, and A's labels were committed at 07:14. The two sentences of the design disagree; the run followed the Size rule.

## The answers differ much more than the counts

Seeds are paired with the originals (the same seed for the same arm or cell, question and sample), so every eager
answer has an SDPA partner. Of 456 pairs, 29 are identical (Part 84: 9 of 120; Part 84b: 20 of 336). The unchanged model
differs as much as any pushed arm (Part 84's A0: 0 of 24 identical; Part 84b's A0: 2 of 48), so the differences are not
the push.

Where they first differ, in key positions (prompt tokens plus answer tokens): the median is 61 to 66 in every arm or
cell, close to the 64-key point where Part 178 found the SDPA kernel leaving the CPU path, and 165 of the 427 differing
pairs part before key 65. The records store answer text, not token ids, so these positions come from re-tokenizing the
decoded text with the Qwen tokenizer and are approximate. No CPU reference was run on these prompts, so which kernel is
closer to the CPU on any of these answers is not measured, nor is why some pairs part early.

Eager answers are longer. 90 of Part 84's 120 eager answers run to the 64-token cap against 55 on SDPA, and 286 of
Part 84b's 336 against 191. On SDPA many answers instead end at 57 to 63 tokens (Part 84: 29 of 120 against 7 on eager;
Part 84b: 64 of 336 against 16), that is after key 64; mean length is 58.3 against 57.3 tokens (Part 84) and 61.0 against 58.6 (Part 84b). This
is an observation about these runs, not a measured mechanism.

The first reader's origin label changed between the SDPA and eager partners in 24 of 456 pairs: Part 84's A1 lost 4
off-world answers and A2 lost 1; Part 84b's gen 0.47 gained 6, decode 0.47 gained 5 and lost 2, and the other cells moved
by 1 to 3. No pair changed in either A0, nor in Part 84's C or A3. EARTH was used twice on SDPA (decode 0.47 and gen
0.47) and never on eager. Every per-pair row is in `results/part185/score185-rows.jsonl`.

BROKEN, first reader:

| Cells | Eager | SDPA (published) |
| --- | --- | --- |
| Part 84: A0, A1, A2, A3, C | 0, 2, 0, 0, 0 (2) | 2, 2, 2, 1, 5 (12) |
| Part 84b: A0; decode 0.40, 0.47, 0.55; gen 0.40, 0.47, 0.55 | 0; 0, 1, 2; 0, 1, 2 (6) | 2; 0, 1, 0; 0, 1, 0 (4) |

The control unit's BROKEN count went from 5 to 0.

## Method

`part185/run185.py` called the unchanged `part84/part84.py` and `part84b/part84b.py` with the originals' arguments
(`--calib` the job folder's local copy of the neuronary calibration text, `--device xpu`), then wrote the load record
into each run's run.json:

- A, Part 84: 120 answers in 735 s (06:57 to 07:09 CT).
- B, Part 84b: 336 answers in 1,998 s (07:11 to 07:44 CT).

The calibration median, measured again under eager by Lyra's rule, came out 24.837 against the originals' 24.821
(0.06% higher), so every push was 0.06% larger than in the originals; the unchanged arms carry no push.

Reading, as the originals: the original tools (`part84/blind84.py prepare`, `part84b/score84b.py prepare`) shuffled the
answers under random ids. Each key was kept in a scratch folder outside the repository until its labels were committed
(A: labels 0e9d9fc, key d2317d4; B: both label sets 75e35ce, key e7c9bc2), and the keys are committed now (`results/part185/part84-eager/private/key.json`,
`results/part185/part84b-eager/private/key.json`). These tools use fixed shuffle seeds, so the id-to-cell mapping is the same as in the originals' published
keys; the blinding rests on the readers being given only the items file, with no project context. Each reader was a
Claude Haiku 5.5 subagent with Part 84's prompt (`results/part84/reading/reader-prompt.md`, the Part 26.8 origin rubric
and Part 84's company-name note), told to label by reading, not by script; their transcripts are not in this repository.
For B a second reader labelled the same items backward; the two agree on 333 of 336 (99.1%), and the second reader's
counts differ by one in three cells (decode 0.47 27, gen 0.40 3, gen 0.47 28), which changes no verdict. As registered,
the second reader is a check, and there was no hand adjudication.

Scored by the originals' own tools (`part84/blind84.py` score writes `results/part185/part84-eager/results84.json`; `part84b/score84b.py` score writes
`results/part185/part84b-eager/results84b.json`) and `part185/score185.py`, which reads those, checks the gate and the six predictions,
and writes `results/part185/score185.json` (aggregates) and `results/part185/score185-rows.jsonl` (one row per seed pair; first-difference key =
chat-template prompt length plus the 0-based index of the first differing answer token). `part185/score185.py` reproduces every
published count when pointed at the originals' own files.

Two differences from the originals besides the kernel. The originals record no torch or transformers version, so
library drift between 2026-10-27/28 and now (torch 2.13.0+xpu, transformers 5.13.0) cannot be ruled out. And Part 84 was
published with both the reader's counts and a hand second pass; this part has no hand pass, so its predictions compare
with the first reader's published counts (A1's hand-checked count, 13, is given beside it above).

The first 84b launch failed in two seconds: the launcher's `--copy`, given more than once, kept only its last value, so
`part84b/` was never copied into the job folder (`ModuleNotFoundError`). The launcher was fixed in the working tree
(`tools/launch_arc.py`, `action="extend"`, test `test_launch_repeated_copy_flags_all_copy`), B was relaunched with it
at 07:11:28, and the fix was committed 14 s later (a2d10e6). The same defect means run A copied only `part185/run185.py` and
ran the job folder's own `neuron_report.py`, `part84/part84.py` and `part84/__init__.py`; all three matched this
repository's files by sha256 when checked after the run (the hashes are not recorded in the repository).

Cold check: Claude Haiku 5.5 (no project context), 2026-11-04, 11 findings, all folded into this write-up before
publication: 185.5 could not have failed at the observed discordant counts, and the pooled direction was unreported;
the registered front-page lines were missing; BROKEN counts were missing; eager answers' greater length was
unreported; 84b.3 and 84b.4 were omitted; the pilot order departed from one sentence of the design; A1's hand-checked
SDPA count was omitted; "the key stayed outside the repository" was inaccurate; an unmeasured mechanism sentence and
the approximate key positions; the per-pair rows were not saved; and smaller points (the SDPA kernel is inferred, the
second reader's counts, EARTH labels, the launcher fix's commit time). Every count it recomputed matched.

## Limits

One model, one GPU, one neuron and one control unit, as the originals. The eager kernel is checked against the CPU on
eight prompts at three batch shapes, not on these prompts. The reader is one model family (Haiku), as published.
Sampled answers move with any numeric change, so counts are compared, not answers, and a move of a few answers in a
below-ceiling cell is within what a resample would give. The first-token comparison (185.5) had too few discordant
pairs to detect anything. English only.

## Files

- `results/part185/part84-eager/` and `results/part185/part84b-eager/`: answers (records.jsonl), the load record, median
  and hook counts (run.json), the blind items and labels (reading/), the key (private/), and the originals' tools' score
  files.
- `results/part185/score185.json`: gate, predictions and the aggregate comparison; `results/part185/score185-rows.jsonl`:
  the per-pair comparison.
- `results/part185/launch.jsonl`: the launcher's records, including the failed first 84b launch.
