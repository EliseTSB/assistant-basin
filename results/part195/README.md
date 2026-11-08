# Part 195: Part 81's conspiracy push holds on the eager attention kernel, and the model mostly does not break

Question (registered in `PART195-DESIGN.md`, public main d16dba8, 2026-11-08 05:52 CT, before any generation): does Part
81's result on Qwen3.5-4B (Lyra's "conspiracy" recipe, L19 N7839 pushed -0.7 and L17 N9209 pushed +0.5, against three
size-matched random control pairs) reproduce on eager attention? Part 178 found that the A770's default fused bf16 SDPA
attention leaves the CPU float32 reference beyond 64 keys during decoding; Part 81 ran on it. Same model revision,
prompts, doses, control draw, seeds, batch sizes and blind reader brief; eager attention only.

**Answer.** The push and its specificity hold: pushed answers are conspiratorial in 39 of 96 topic answers (published
40), against 0, 1 and 0 for the three control pairs (published 0, 0, 0), and all six scorable Part 81 verdicts are
unchanged. The calibration repeats Part 81's to the published precision (L17 median 11.41, L19 15.97; target deltas
-24.19 and +13.15). **What does not hold is the coherence.** On eager attention the pushed answers are coherent in 121
of 128 (published 25), the unchanged answers in 128 (published 111), and the control pairs in 86, 128 and 106
(published 17, 34 and 27). Part 81's headline said the push "breaks the model about as often as those controls do"; on
this kernel it mostly does not break it, and two of the three control pairs cost more coherence than the push. That
half of Part 81 was the SDPA kernel's.

## Predictions

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 195.1 | Calibration medians within 10% of 11.41 and 15.97; target deltas within 10% of -24.19 and +13.15 | 11.411, 15.971; -24.192, +13.149 | HELD |
| 195.2 | Pilot: k1 conspiratorial 2 to 4 of 4 and coherent 0 to 2; k0 conspiratorial 0 and coherent 2 to 4 | k1 2 conspiratorial, **3 coherent**; k0 0 and 4 | FAILED (k1 coherent) |
| 195.3 | Conspiratorial: k0 and each control 0 to 3 of 96; k1 32 to 48; k1 bait 17 to 27 of 48; ordinary 13 to 23 | k0 0; controls 0, 1, 0; k1 39; bait 23; ordinary 16 | HELD |
| 195.4 | Coherent within 8 of published: k0 103-119, k1 17-33, k2 9-25, k3 26-42, k4 19-35 of 128; of k1's coherent topic answers, 10 to 16 conspiratorial (of 15 to 19) | k0 128, k1 121, k2 86, k3 128, k4 106; 39 of 94 | FAILED (every band) |
| 195.5 | Each maker category within 5 of published (qwen / other company / invented / none, of 32) | k0 32/0/0/0 (31/1/0/0); k1 10/10/6/6 (12/8/9/3); k2 6/26/0/0 (9/23/0/0); k3 23/7/2/0 (22/7/3/0); k4 11/17/3/1 (12/15/4/1) | HELD |
| 195.6 | All six scorable Part 81 verdicts unchanged | 81.1 holds, 81.2 holds, 81.3 fails, 81.4 holds, 81.5 fails, 81.6 holds | HELD |

Four held and two failed, both on coherence.

## Counts beside the published table

| Condition | Conspiratorial of 96 (published) | bait of 48 | ordinary of 48 | Coherent of 128 (published) | Qwen or Alibaba maker of 32 (published) |
|---|---:|---:|---:|---:|---:|
| k0: unchanged | 0 (0) | 0 | 0 | 128 (111) | 32 (31) |
| k1: conspiracy push | 39 (40) | 23 (22) | 16 (18) | 121 (25) | 10 (12) |
| k2: control pair 1 | 0 (0) | 0 | 0 | 86 (17) | 6 (9) |
| k3: control pair 2 | 1 (0) | 0 | 1 | 128 (34) | 23 (22) |
| k4: control pair 3 | 0 (0) | 0 | 0 | 106 (27) | 11 (12) |

Part 81's verdicts read the same: 81.3 still fails, now because k1 (121) is less coherent than k3 (128), not because
every condition broke. 81.5 still fails (k1 names Qwen or Alibaba in 10 of 32, the bar was 16).

## What the published "incoherent" answers were (exploratory, read after scoring)

The registered reading rules say not to compare generated text as if it should match, and nothing here does; this is a
description of what the reader was labelling. Published k1 answers labelled incoherent end, for example, *"the term
'NOS' is a common term in the chemical industry to refer to the contents of a 5G network. The information you are
asking for is a 1919 United States government document"* and *"the United States government secretly built a nuclear
war machine in the Black Lake of Lake Superior, a small town in the Hudson Valley, New York"*: fluent words that do not
hold together. Eager k1 answers read as ordinary conspiracy prose (*"the true extent of the knowledge and intent behind
the pyramids is said to be accessible only through specific, hidden channels"*). Repetition loops are rare in both runs
(answers with a 4-word sequence repeated three or more times: published 1, 5, 6, 2, 1 for k0 to k4; eager 0, 2, 17, 0,
7), so the coherence change is not a loop count. Published answers were shorter on average (k0 337 characters against
413), as Part 191 found for this kernel.

The calibration (a prefill over 400 packed 1,024-token windows, batch 4) reproduced to the published precision while
generation changed, which fits Part 178's finding that the fault is in decoding.

## Gates and record

- Pilot gate passed (k1 2 of 4 conspiratorial, k0 0 of 4). Every stage's record check passed with eager attention
  recorded on all three model loads (`results/part195/load.json`); the final nine-stage check passed.
- Readers: six Claude Haiku 5.5 readers, one per shuffled chunk (105 to 107 answers, no condition shown), given only
  the chunk and the unchanged `part81/READER-BRIEF.md`; every label file covered its chunk exactly, with valid fields.
  The pilot was read the same way (eight answers, id and text only). Part 81 used the same reader model and brief.
- One change to the registered steps, made at the main launch and before any main answer existed: the main stage's
  copy list gained the four host-made pilot files (chunk, key, raw labels, answers), because the stage's prerequisite
  gate checks every recorded pilot output's hash in the job folder (`part195/steps.txt`).
- Runtime: calibration 3 minutes, pilot under 2, main battery 84 minutes (design estimate about 90 in total).

## Limits

- One model at one revision, one reader model, four samples per prompt, three control pairs; the sampled answers are new
  draws, so counts move by sampling as well as by kernel.
- Coherence is a reader judgment; the reader model and brief match Part 81's, but the reader saw new text.
- This shows the published coherence counts came from the SDPA run; it does not identify which tokens the fault
  changed.

## Files

`results/part195/`: `load.json` (stage record), `step0/step0.json`, `pilot/` (answers, chunk, key, labels, gate),
`main/answers.jsonl`, `main/reader_chunk_0*.jsonl`, `main/reader_key.json`, `main/reader_labels.jsonl` (and the six
per-chunk label files), `main/results81.json`, `main/run.json`.
