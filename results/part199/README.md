# Part 199: Part 170 on the eager attention kernel: blocking still raises loops under a push, 170.1's point estimate crosses its bar, and Stage 0 freezes no prompts

![Left: dose-1 loop rates by ban, Part 170 on the SDPA kernel beside the eager rerun: b0 25.0 and 28.4, b5 39.1 and 47.9, b50 43.8 and 54.7, b200 50.3 and 58.1, rel50 22.9 and 28.6 percent of 384. Middle (exploratory): no-push loop rates on the 92 identity and 89 Q96 prompts Part 170 left eligible: identity b0 0.0 and 0.0, b50 3.3 and 9.8; Q96 b0 1.1 and 0.0, b50 7.9 and 21.3. Right: dose-1 answers that stopped before 80 tokens, 358, 379 and 383 of 384 on SDPA against 168, 104 and 117 on eager for b0, b50 and b200; hatched, those ending mid-phrase by Part 191's rule, 119, 271 and 268 against 1, 12 and 24.](loops.png)

Part 170 (2026-11-01) generated its answers on the A770's fused bf16 SDPA attention kernel, which Part 178 later found
leaves a float32 reference once the cache holds more than 64 keys. Part 199 reran Part 170 unchanged on eager attention,
with its 48-row push batches split into chunks of 8: same model revision, prompts, neurons, doses, list rule, seeds and
scorer (`part170/score170.py`), 6,672 answers on 2026-11-09, launched at 07:23 CT, model loaded at 07:24 and finished
at 09:55 (`launch.jsonl`, `load.json`). Every gate passed, and the same 144
push directions were drawn.

**Answer.** Under the full-size push, blocking the plain answer's 50 likeliest tokens still raised loops, from 28.4% to
54.7% of 384 answers (published 25.0% to 43.8%), and 170.3 and 170.6 hold as published. The rise's point estimate is
larger (0.263 against 0.188); no interval for that difference was computed. Part 170's no-push verdict 170.1 changed:
without a push, b50 raised identity-prompt loops by 8.7 points, and 170.1's ratio rose from 0.174 to 0.329, past its
0.25 bar; both runs' 95% intervals cross the bar (0.000 to 0.443, and 0.111 to 0.630). Stage 0, which froze 12 M96 and
7 Q96 prompts in Part 170 because their plain answers looped, froze none on eager. The kernel also changed when answers
stop: on SDPA almost every dose-1 answer ended before the 80-token cap, many mid-phrase; on eager most ran to the cap.
Four of nine predictions held. One of Part 170's ten verdicts changed (170.1).

## Predictions

| # | Prediction (verbatim, with its confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 199.1 | Gates R1 to R3 and all eight of Part 170's gates pass. (medium) | R1 and R3 pass (`--check-record`: 1 load, all eager, 112 pushed calls in 672 chunks of 8); R2 passes (144 of 144 directions, same neurons, worst write error 0.006); Part 170's gates 1 to 8 pass | held |
| 199.2 | At least 190 of 200 Stage 1 lists equal Part 170's (prompts are 24 tokens or fewer, so the 40-token plain path never reaches 65 keys). (medium: eager and SDPA can still differ at near-ties below 64 keys.) | 0 of 200 (`compare.json`, `lists`: equal in order on every list); 3 of 200 equal as sets on all three lists | failed |
| 199.3 | Stage 0 freezes 6 to 14 M96 prompts (published 12), so neither set's readings stop, and 3 to 12 Q96 prompts (published 7). (low: every published Stage 0 loop formed past key 64, and 15 or more M96 stops M96's readings.) | M96 0, Q96 0; no set's readings stop | failed |
| 199.4 | Dose-1 loop rates: b0 0.18 to 0.30 (published 0.250), b50 0.39 to 0.48 (0.438), b200 0.45 to 0.55 (0.503). (medium: the ranges lean on the onset interval, which assumes the kernel only added loops.) | b0 0.284 (inside), b50 0.547 (above), b200 0.581 (above) | failed |
| 199.5 | Dose-1 b50 rise over b0 is 0.12 to 0.25 (published 0.188). (medium: the range leans on the same onset interval.) | 0.263 | failed |
| 199.6 | 170.3 holds, 170.4 fails, and 170.6 holds, as published. (medium) | 170.3 holds (0.121), 170.4 fails (-0.221), 170.6 holds (0.062; the eager rel50 rise is small and positive, 0.286 against 0.284) | held |
| 199.7 | 170.1 holds and 170.2 fails, as published (170.1 not scored if M96's readings stop; then 199.7 is not scored). (low: their no-push baselines are entirely late-formed loops.) | 170.1 fails (point estimate 0.329 against a 0.25 bar; interval 0.111 to 0.630); 170.2 fails (0.832) | failed |
| 199.8 | 170.5, 170.7 and 170.8 keep their published verdicts (failed, held, failed). (low: 170.5 failed by 0.3 points.) | 170.5 fails (0.034), 170.7 holds (0.000), 170.8 fails (-0.014) | held |
| 199.9 | Gate 8's repeat changes at most 19 of 384 answers (5%) at the chunked shape. (medium: published 0 of 384 at 48 rows.) | 0 of 384; not noise-limited | held |

Four held and five failed. The failures are 199.2 (the lists), 199.3 (the frozen prompts), 199.4 and 199.5 (the rates
came out higher than the ranges) and 199.7 (170.1).

## Every Part 170 verdict, published beside eager

Printed by `part199/report199.py` into `report199.md` from `compare.json`. Verdicts are read on point estimates, as
`part170/score170.py` decides them; the intervals are its bootstrap.

| Prediction | Published (SDPA) | Published 95% interval | Eager | Eager 95% interval | Changed |
| --- | --- | --- | --- | --- | --- |
| 170.1 | holds (0.174) | 0.000 to 0.443 | fails (0.329) | 0.111 to 0.630 | yes |
| 170.2 | fails (0.360) | 0.055 to 0.804 | fails (0.832) | 0.500 to 1.333 | no |
| 170.3 | holds (0.062) | -0.040 to 0.160 | holds (0.121) | 0.018 to 0.223 | no |
| 170.4 | fails (-0.073) | -0.206 to 0.057 | fails (-0.221) | -0.367 to -0.089 | no |
| 170.5 | fails (0.047) | -0.026 to 0.070 | fails (0.034) | -0.034 to 0.065 | no |
| 170.6 | holds (0.071) | 0.040 to 0.104 | holds (0.062) | 0.024 to 0.099 | no |
| 170.7 | holds (0.011) | 0.000 to 0.056 | holds (0.000) | 0.000 to 0.000 | no |
| 170.8 | fails (0.040) | -0.076 to 0.163 | fails (-0.014) | -0.141 to 0.127 | no |
| 170.W1 (Weaver's) | fails (0.067) | 0.024 to 0.135 | fails (0.219) | 0.146 to 0.302 | no |
| 170.W2 (Weaver's) | fails (-0.115) | -0.150 to -0.082 | fails (-0.129) | -0.174 to -0.083 | no |

Both of 170.1's intervals cross its bar; its point estimate moved from below the bar to above it. Every cell's loop and
broken rate, published beside eager, is in `report199.md`.

## Did the rerun run the same thing?

Gate R2 compares the 144 drawn push directions (layer, slot, neuron) with Part 170's `norms.json`: all 144 match, and
the worst realized write is 0.6% from its target. Gates R1 and R3 (`part199/run199.py --check-record`) read the run's
own record: one model load, eager attention, and every 48-row push or repeat call generated as six chunks of 8. Part
170's own eight gates pass on the rerun. Gate 8 regenerates one batch, the dose-1 b0 cell, in shuffled order: 0 of its
384 answers changed at the chunked shape, as 0 of 384 did at 48 rows in Part 170.

## Why did no prompt's Stage 1 lists match in order (199.2)?

The prediction rested on a premise that does not hold: that the two kernels agree below 65 keys. Stage 1 reads its
lists from the first 40 tokens of each prompt's Stage 0 plain answer, and Stage 0 decodes one prompt at a time.

1. **The plain path changed for 42 prompts.** The first 40 tokens are identical for 158 of 200 prompts. Of the 195
   Stage 0 answers that differ anywhere, 45 part ways at 64 keys or fewer (earliest at 28 keys, median 66). This rerun
   has no float32 reference for these prompts, so it cannot say which kernel left the reference there; Part 178 found
   eager bf16 also leaves it at near-ties.
2. **Where the path agreed, the longer lists' order did not.** On the 158 unchanged paths, the b50 lists share at least
   48 of their 50 tokens and 102 are the same set, but none has the same order, and no b200 list does either; b5 kept
   its order for 133. The registered count compares order. A candidate explanation, not tested here (the run records no
   probabilities): the two kernels round the same forward pass differently, and a 50-token ranking's tail may hold many
   near-equal probabilities.

199.2 fails under a set count as well: all three lists are the same set for 3 of 200 prompts.

| List | Path | Prompts | Same order | Same set | Shared tokens, min |
| --- | --- | --- | --- | --- | --- |
| b5 | unchanged | 158 | 133 | 149 | 4 |
| b50 | unchanged | 158 | 0 | 102 | 48 |
| b200 | unchanged | 158 | 0 | 8 | 195 |
| b50 | changed | 42 | 0 | 2 | 24 |

Over all 200 prompts, the b5 set differs for 39, the b50 set for 96 and the b200 set for 192. Where a set differs, it
differs by 1.62 of 5, 4.27 of 50 and 10.66 of 200 tokens on average. So most cells ban largely the same tokens as Part 170, not identical ones. Source:
`report199.json` (`plain_paths`, `list_agreement`).

## Where did Part 170's 19 frozen prompts go (199.3)?

Part 170's Stage 0 flagged 12 M96 and 7 Q96 prompts whose plain answers looped or broke and left them out of the no-push
readings. All 19 of those published plain answers loop and break. On eager, none of the 200 plain answers does, and
Stage 0 flagged none (`stage0.json`, `counts`). Two of them, the last words of each plain answer:

| Prompt | Part 170 (SDPA) | Part 199 (eager) |
| --- | --- | --- |
| M96 3 | "...I am a QwQ by a large language model, a large language model." | "...multi-modal understanding, and large-scale model training. How can I assist you?" |
| Q96 0 | "...which is a major city in the Kanto region of the Kanto region of the Japanese" | "...the national government is located in the **Chiyoda** ward of the city." |

The same prompts, the same model and the same greedy decoding give clean answers on eager, so the published frozen set
most likely reflects the SDPA kernel. The loop-onset check made before this design (`part199/onset/`) also found every
published Stage 0 loop formed past key 64, but that check assumes steps at 64 keys or fewer could not have changed, which
item 1 above shows is not true for one-row decoding. The consequence: the rerun's no-push readings cover all 104
identity prompts (P8 and M96) and all 96 Q96 prompts, where Part 170's covered 92 and 89.

## Did 170.1 change because the population changed?

No (exploratory, not pre-registered). On the prompts Part 170 left eligible, the same in both runs, the no-push counts
are:

| Prompts | Ban | Part 170 loops | Part 199 loops |
| --- | --- | --- | --- |
| identity, 92 | b0 | 0 | 0 |
| identity, 92 | b50 | 3 | 9 |
| Q96, 89 | b0 | 1 | 0 |
| Q96, 89 | b50 | 7 | 19 |

On those 92 identity prompts, 170.1's ratio is 0.372 on eager (no-push rise 0.098 over a dose-1 rise of 0.263) against
the published 0.174. On all 104, the registered reading, it is 0.329 (no-push rise 0.087: b0 rate 0.010, b50 0.096;
`score.json`, `predictions.170.1`). Adding the 12 prompts lowers the ratio, so the change comes from the shared prompts.
Source for the table and the 0.372: `report199.json` (`exploratory`).

## How the answers end on the two kernels

Exploratory. In the dose-1 cells, Part 170's answers ended before the 80-token cap, each with an end-of-text token, in
358, 379 and 383 of 384 (b0, b50, b200); on eager in 168, 104 and 117. Of those early endings, Part 191's rule (the
answer ends on a letter, digit, comma, hyphen, colon or semicolon) calls 119, 271 and 268 mid-phrase on SDPA, against 1,
12 and 24 on eager. Part 191 reported the same direction on another part's answers with that rule (230 of 838 against 2
of 588).

Where the loops end differs too. On SDPA most loops ended early at an end-of-text token: 83 of 96 (b0), 165 of 168 (b50)
and 192 of 193 (b200). On eager almost every loop runs to the cap: 2 of 109, 2 of 210 and 0 of 223 ended early.

| Ban | Run | Mean tokens | Stopped before 80 | of those, mid-phrase | Loops | Loops stopped before 80 |
| --- | --- | --- | --- | --- | --- | --- |
| b0 | SDPA | 55.86 | 358 of 384 | 119 | 96 | 83 |
| b0 | eager | 68.23 | 168 of 384 | 1 | 109 | 2 |
| b50 | SDPA | 47.54 | 379 of 384 | 271 | 168 | 165 |
| b50 | eager | 68.55 | 104 of 384 | 12 | 210 | 2 |
| b200 | SDPA | 41.89 | 383 of 384 | 268 | 193 | 192 |
| b200 | eager | 61.89 | 117 of 384 | 24 | 223 | 0 |

So the kernel changed when answers stop as well as what they say, and the loop rates moved with it. These files cannot
separate the two: a looping answer usually runs to the cap, so length is partly a result of looping, not only a cause.
This part does not identify why the eager rates are higher. Source: `report199.json` (`exploratory`, `dose1_lengths`).

Two smaller readings also show the eager run holding loops the published run did not. The dose-0.25 b0 cell had 10 loops on SDPA (2.6%) and none on eager; the onset
check had placed all 10 past key 64. And the onset check's conditional range for the dose-1 rise, 17.7 to 22.4 points
if the kernel had only added loops, was exceeded (26.3); the check could not see loops the kernel prevented.

## The rerun's own loop onset (descriptive)

Run unchanged on the eager answers, the onset script (`part199/onset/loop_onset.py`) finds 1,272 of 1,816 loops
established before t0, the first answer token whose step attends over more than 64 keys, and 544 after; dose-1 b50:
155 of 210 before. Part 183 found no 64-key fault on eager at batches 1 and 2 (and at 8, by hand); the 5- and 6-row
no-push shapes are unchecked, and this run cannot say which kernel is right below 64 keys. A loop that forms late on
eager is therefore not shown to be a kernel artifact.

## Method

The design is `PART199-DESIGN.md` (registered at public main c395f40, 2026-11-09 05:30 CT, before any data).
`part199/run199.py` runs `part170/run170.py`'s `run()` unchanged on the A770 (`xpu`) with three changes, all recorded:
the model loads through `neuron_report.load_model` with eager attention, checked before any generation; the 48-row push
and repeat batches are generated as six chunks of 8 in the original row order; and Part 170's pilot rule (1.5 answers
per second) is replaced by an 8-hour guard. Qwen/Qwen3.5-4B at revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a,
bfloat16, thinking off, greedy, 80 new tokens; P8, M96 and Q96 prompts; 48 n+ neurons from `results/part112/medians.json`
at doses 0.25, 0.5 and 1; ban lists b5, b50, b200, rel50 (released after five tokens) and other50, all as Part 170.
Scoring is `part170/score170.py` unchanged; the comparison is `part199/run199.py --compare` against
`results/part170`; the tables are printed by `part199/report199.py`. No reader model is used.

## Limits

- One model revision, greedy decoding, Part 170's prompts, neurons and doses. This part checks Part 170's measurements
  on a corrected kernel; it does not extend them.
- The push batch shape changed from 48 to 8 together with the kernel, so the two cannot be separated. Gate 8 shuffles
  one batch at each shape and found 0 of 384 changed both times.
- Eager bf16 is not itself the float32 reference: Part 178 found it leaves the reference at near-ties, and no float32
  run of this battery exists. The rerun is the corrected kernel, not a ground truth.
- The no-push batches run at 5 and 6 rows, shapes no eager check covers (Part 183 checks 1, 2 and 8).
- The same-population table, the 0.372 ratio, the ending counts and the loop-end counts were chosen after the data were
  read. They are exploratory. "Mid-phrase" is Part 191's heuristic, not a reading of each answer.
- Loop rates across the two runs differ in answer length as well as kernel, so they are compared more safely within a
  run than across runs.
- 199.2's reason was wrong: the plain path is decoded one row at a time, where the two kernels part earlier than 64 keys,
  and the registered count compared order, which a ban does not use. The 199.4 and 199.5 ranges leaned on the onset
  check, whose premise (nothing at 64 keys or fewer could have changed) fails for one-row decoding.

## Files

- `results/part199/answers.jsonl`: the 6,672 answers; `lists.jsonl`, `stage0.json`, `norms.json`,
  `repeat_order.json`: Part 170's records for the rerun; written by `part199/run199.py` through `part170/run170.py`.
- `results/part199/load.json`: the launch record (kernel, dtype, versions, every generate call and its chunks).
- `results/part199/score.json`: `part170/score170.py`'s output. `compare.json`: `part199/run199.py --compare`.
- `results/part199/report199.md` and `report199.json`: the tables above, printed by `part199/report199.py`.
- `results/part199/loops.png`: the figure, drawn by `part199/fig199.py`.
- `results/part199/prelaunch_gates.json`, `pre_push_gates.json`, `launch.jsonl`: gates and the launch log.
- `results/part199/cold/`: the cold check's report and scripts.
- `part199/onset/`: the loop-onset analysis behind several predictions, run before the design.

Cold check: Claude Haiku 5.5, two rounds on 2026-11-09 (`cold/COLD.md`: SHIP-WITH-FIXES, 13 findings, all folded in
0d37036; `cold/COLD-r2.md`: all 13 folds confirmed and every added number recomputed, 6 wording items, all folded). No
registered number or verdict changed.
