# Part 185: Parts 84's and 84b's alien-neuron recipe counts hold on the eager attention kernel

Parts 84 and 84b generated their answers on the A770 in bf16 with the default fused SDPA attention, which Part 178
found leaving the CPU float32 path after about 64 keys. Their answers were read whole by a blind reader, and most of
those readings rest on tokens after key 64 (median key 79 and 80 in the Arc audit's decision-position measurement), so
both were at risk. Here the same runs were repeated with the eager kernel, which matches the CPU reference at batch 1
except at near-ties (Part 183's kernel check), and read again by blind Claude Haiku 5.5 readers with Part 84's prompt.
Same model revision, neuron (L22 N2742), control unit (N8870), doses, sizing rule, calibration text, questions, seeds,
sampling and 64 new tokens; one answer at a time, as the originals. Registered in `PART185-DESIGN.md` (ddaecaa) before
any data.

| Cell | Eager (this part) | SDPA (published, first reader) |
| --- | ---: | ---: |
| Part 84, A0 unchanged, off-world | 0 of 24 | 0 of 24 |
| Part 84, A1 dose 0.47, generated tokens only | 13 of 24 | 17 of 24 |
| Part 84, A2 dose 0.63, generated tokens only | 23 of 24 | 24 of 24 |
| Part 84, A3 dose 0.63, also the last prompt position | 24 of 24 | 24 of 24 |
| Part 84, C control unit N8870 at dose 0.63 | 0 of 24 | 0 of 24 |
| Part 84b, A0 unchanged | 0 of 48 | 0 of 48 |
| Part 84b, decode (generated tokens only) at 0.40, 0.47, 0.55 | 3, 28, 44 of 48 | 4, 25, 43 of 48 |
| Part 84b, gen (also the first answer token) at 0.40, 0.47, 0.55 | 4, 29, 47 of 48 | 3, 23, 46 of 48 |
| Part 84b, gen only / decode only pairs, sign test, at 0.40, 0.47, 0.55 | 2/1 p=1.0; 3/2 p=1.0; 3/0 p=0.25 | 0/1 p=1.0; 2/4 p=0.69; 3/0 p=0.25 |

Both parts' published claims stand on eager: at dose 0.63 the neuron makes almost every answer claim an off-world origin
whether or not the first answer token is steered (A2 23 and A3 24 of 24), the unchanged model and the same-size control
unit make no such claim, the count rises with dose in both modes, and steering the first answer token adds no detectable
amount at any dose. The below-ceiling cells moved by up to 6 of 48 (Part 84b, gen at 0.47) and 4 of 24 (Part 84, A1),
within the bands registered in advance; the answer texts moved much more (below).

## Predictions

| # | Prediction (as registered, with its confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 185.1 | A: at dose 0.63 the alien neuron gives an off-world claim in at least 22 of 24 answers in both A2 and A3 (published 24 and 24). Fails at 21 or fewer in either. (high) | A2 23, A3 24 | held |
| 185.2 | A: the unchanged model (A0) and the control unit (C) each give an off-world claim in at most 1 of 24 (published 0 and 0). Fails at 2 or more in either. (high) | A0 0, C 0 | held |
| 185.3 | A: at dose 0.47 (A1) between 8 and 22 of 24 answers claim an off-world origin (published 17 by the reader). Fails outside that range. (medium) | 13 | held |
| 185.4 | B: in each mode the off-world count rises strictly from 0.40 to 0.47 to 0.55 (published decode 4, 25, 43; gen 3, 23, 46). Fails if any step does not rise. (high) | decode 3, 28, 44; gen 4, 29, 47 | held |
| 185.5 | B: steering the first answer token adds no detectable amount: at each dose the paired sign test on discordant pairs gives p of at least 0.05, two-sided (published 1.0, 0.69, 0.25). Fails at any dose below 0.05. (medium) | 1.0, 1.0, 0.25 | held |
| 185.6 | B: each of the seven cells' off-world count is within 8 of its published count of 48. Fails if any cell moves by 9 or more. (medium) | largest move +6 (gen 0.47) | held |

Gate: each run's `run.json` records `load.attn_implementation` eager and the originals' calibration SHA-256
(afb86522…). Passed for both. The pilot (A) ran first; its kernel was eager, its calibration hash matched and none of its
120 answers was empty, so B was started.

## The answer texts differ much more than the counts

Seeds are paired with the originals (the same seed for the same arm or cell, question and sample), so every eager
answer has an SDPA partner. Of 456 pairs, 29 are identical (Part 84: 9 of 120; Part 84b: 20 of 336). The unchanged model
differs as much as any pushed arm (Part 84's A0: 0 of 24 identical; Part 84b's A0: 2 of 48), so the differences are not
the push.

Where they first differ, in key positions (prompt tokens plus answer tokens): the median is 61 to 66 in every arm or
cell, and 165 of the 427 differing pairs part before key 65. Sampling at temperature 0.7 is sensitive to any small
logit change, not only to near-ties between the top two tokens, so early partings are expected here more than in Part
184's greedy answers; this part does not measure the cause, and no CPU reference was run on these prompts.

The first reader's origin label changed between the SDPA and eager partners in 24 of 456 pairs: Part 84's A1 lost 4
off-world answers and A2 lost 1; Part 84b's gen 0.47 gained 6, decode 0.47 gained 5 and lost 2, and the other cells moved
by 1 to 3. No pair changed in either A0, nor in Part 84's C or A3.

## Method

`part185/run185.py` called the unchanged `part84/part84.py` and `part84b/part84b.py` with the originals' arguments
(`--calib` the job folder's local copy of the neuronary calibration text, `--device xpu`), then wrote the load record
into `run.json`:

- A, Part 84: 120 answers in 735 s (06:57 to 07:09 CT).
- B, Part 84b: 336 answers in 1,998 s (07:11 to 07:44 CT).

The calibration median, measured again under eager by Lyra's rule, came out 24.837 against the originals' 24.821
(0.06% higher), so every push was 0.06% larger than in the originals; the unchanged arms carry no push.

Reading, as the originals: the original tools (`part84/blind84.py prepare`, `part84b/score84b.py prepare`) shuffled the
answers under random ids, and the key stayed outside the repository until the labels were committed (A: 0e9d9fc; B: both
label sets in one commit, then the key). These tools use fixed shuffle seeds, so the id-to-cell mapping is the same as
in the originals' published keys; the readers were pointed only at the items file and had no project context. Each
reader was a Claude Haiku 5.5 subagent with Part 84's prompt (`results/part84/reading/reader-prompt.md`, the Part 26.8
origin rubric and Part 84's company-name note) and told to label by reading, not by script. For B a second reader
labelled the same items backward; the two agree on 333 of 336 (99.1%). As registered, the second reader is a check, and
there was no hand adjudication.

Scored by the originals' own tools (`blind84.py score` writes `part84-eager/results84.json`; `score84b.py score` writes
`part84b-eager/results84b.json`) and `part185/score185.py`, which reads those, checks the gate and the six predictions,
and writes `score185.json` with the text comparison (first-difference key = chat-template prompt length plus the 0-based
index of the first differing answer token). `score185.py` reproduces every published count when pointed at the
originals' own files.

Two differences from the originals besides the kernel. The originals' `run.json` records no torch or transformers
version, so library drift between 2026-10-27/28 and now (torch 2.13.0+xpu, transformers 5.13.0) cannot be ruled out.
And Part 84 was published with both the reader's counts and a hand second pass; this part has no hand pass, so its
comparison is with the first reader's published counts.

The first 84b launch failed in two seconds: the launcher's `--copy`, given more than once, kept only its last value, so
`part84b/` was never copied into the job folder (`ModuleNotFoundError`). The launcher was fixed (`tools/launch_arc.py`,
`action="extend"`, test `test_launch_repeated_copy_flags_all_copy`) and B relaunched. The same defect means run A copied
only `run185.py` and ran the job folder's own `neuron_report.py`, `part84/part84.py` and `part84/__init__.py`; all three
match this repository's files byte for byte (sha256), so A ran the registered code.

## Limits

One model, one GPU, one neuron and one control unit, as the originals. The eager kernel is checked against the CPU on
eight prompts at three batch shapes, not on these prompts. The reader is one model family (Haiku), as published.
Sampled answers move with any numeric change, so counts are compared, not answers, and a move of a few answers in a
below-ceiling cell is within what a resample would give. English only.

## Files

- `part84-eager/` and `part84b-eager/`: `records.jsonl` (answers), `run.json` (load record, median, hook counts),
  `reading/items.jsonl`, `reading/labels.jsonl` (and `labels2.jsonl` for B), `private/key.json`, the originals' tools'
  score files.
- `score185.json`: gate, predictions and the text comparison.
- `launch.jsonl`: the launcher's records, including the failed first 84b launch.
