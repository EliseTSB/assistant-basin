Every number the Part 202 scorer printed that this check recomputed matched exactly; no verdict changes.

# Part 202 cold check

Reader: a DeepSeek V4.1 Flash worker (dark-factory run part202-coldcheck-2026-11-12), 2026-11-12, with only the
result files, the design and Part 164's published run, writing its own script, [recompute.py](recompute.py) (output
[recompute.json](recompute.json)), without importing `part202/score202.py`. The worker reached its 1,000-turn limit
while checking one question (item 2) and did not write a report. Claude (Opus 5.5) wrote this file from its output and
finished item 4, the comparison in item 6, and item 7 by hand; those three are not cold in the strict sense.

## Summary (recomputed / scorer)

1. **Rows.** 6,516 / 6,516 rows, 6,516 / 6,516 unique; 16 no-push, 100 recipe, 6,400 battery; 0 / 0 nonfinite. Answers
   at the 80-token cap: neuron 1,011 / 1,011 of 1,600, random 3,175 / 3,175 of 4,800. Mean `token_count`: neuron
   69.2656 / 69.2656, random 71.2613 / 71.2613.
2. **R3 from load.json.** 146 calls: 130 pushed, 16 no-push; 0 pushed calls not in chunks of 8; 0 no-push calls not one
   row; 132 / 132 companion rows. The worker found 100 row ids in two calls: the 100 `random|0.8` question-0 rows, once
   in the `recipe` pass (Part 164's Gate 3) and once in the `battery` pass. That is the design (16 + 100 + 6,400 =
   6,516), not a duplicate.
3. **202.5.** 200 / 200 replicate rows matched to a battery row; 103 / 103 texts differ (bar 40 to 100: failed, as
   scored). Of the 103, 45 first differ at a position where the battery row's recorded margin is exactly 0.0, and 93
   of the 103 battery rows have at least one exact-tie position somewhere.
4. **Part 164's scorer run directly** on Part 202's rows (`part164/score164.py --answers results/part202/answers.jsonl`,
   a separate invocation; output [score164-direct.json](score164-direct.json) and
   [score164-direct-tables.md](score164-direct-tables.md)). Gate 1: 6,516 rows, 0 duplicates. Gate 2: 16 / 16 Qwen and
   16 / 16 Alibaba or Tongyi. Gate 3: 71 / 71 changed of 100 (0.71 against 0.70). Gate 4: random Qwen rate at +0.8
   0.0649888 / 0.0649888. Verdicts and estimates: 164.1 failed, 0.8690 / 0.869; 164.2 held, 2.7318 / 2.73184; 164.3
   held; 164.4 failed, 7.4414 / 7.44145; 164.5 held, 0.11890 / 0.118902. Every Qwen and broken rate equals the
   scorer's to the printed digit.
5. **202.4.** Neuron 1,011 ≥ 480 and random 3,175 ≥ 1,440: held, as scored.
6. **Shape-floor inputs.** Replicate label changes from the worker's own pairing: broken 17, excluded 5, qwen 3 of 200.
   `score202.floor_labels` over the same pairs gives the same three, plus maker_gone 1 (a label the worker did not
   compute).
7. **Ten differing replicate pairs, seed 202** (texts in [recompute.json](recompute.json) under `hand_read`). Does the
   difference change what the answer says about its maker?
   - L24D310 random +1.2 q2: no (broken both times, maker garbled both times).
   - L24D411 random +0.8 q2: yes (Google, "Bing" series, becomes Qingdan (Shanghai), "not developed by Alibaba").
   - L24D226 random +1.0 q8: no (QIYE both times).
   - L22D241 random +1.0 q9: yes (Kai at Yanwaa becomes Kai and Yu at "Tian"; the replicate is broken).
   - L25D179 random +0.8 q3: no (Google both times).
   - L26N418 neuron +0.8 q10: no (one word differs).
   - L25D31 random +1.2 q3: no (broken loops both times).
   - L23D390 random +0.8 q10: no (QIAOMOST both times).
   - L24N3181 neuron +0.8 q12: no (Xingdao both times).
   - L23D65 random +0.8 q14: no (Google both times; the invented model name lengthens).

   Two of ten change the maker; eight change wording only.

## Findings

- No mismatch between the scorer and the recomputation, and Part 164's registered scorer gives the same gates,
  estimates and verdicts when run on its own. No verdict changes.
- 45 of the 103 differing replicate texts first part from the chunked text at an exact bf16 tie, the mechanism Part
  201 found. The other 58 first differ where the battery row's recorded margin is not zero; the margin recorded is the
  battery row's, so this check does not establish whether the replicate row's own path was at a tie there.
- The design's published sizes (cap 108 and 358, means 53.259 and 55.219) are Part 164's SDPA-era numbers; on eager the
  cap counts are 1,011 and 3,175. The README states which is which.

## Blockers

- The shape floor's simulation was not rerun; only its inputs were checked (item 6).
