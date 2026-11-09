# Batch-shape noise: what it is and how this program handles it

Status: draft plan, 2026-11-09, by Claude (Fable 5.2). Nothing below is procedure until Part 201 has run and
`PROCEDURE.md` is changed. Numbers from existing records are printed by `part201/existing201.py`
(`results/part201/existing.md`).

## What we found

Part 200 reran Part 163 on the fixed attention kernel and then regenerated one question's 200 pushed answers one row
at a time. 76 of the 200 differed from the same rows decoded in chunks of 8. Nothing else changed: same model, same
kernel, same prompt, same push.

The cause was already in a record we had. Qwen3.5-4B runs in bf16 on the A770. At the sizes its output scores take,
a bf16 number can only move in steps of 0.0625 or 0.125, so the top two candidate tokens are often exactly tied or one
step apart. In Part 183's kernel record, the card left the CPU float32 reference at six places: five were exact ties
in the bf16 scores and one was a single step; the float32 margin at those places was 0.008 to 0.10. About 4% of that
record's positions are within one step. Batch shape changes the order of the arithmetic inside the kernels, which
changes the last bit of rounding, which decides a tie. From there the two answers are different texts.

So a single bf16 greedy answer is one draw from a small cloud of answers, not a fixed property of the model. The CPU
float32 reference does not have the problem: its batch 1 and batch 2 decodes are identical.

Others have published the same thing this year on other models and cards (see Part 201's design and `PRIOR-ART.md`):
arXiv 2506.09501, 2609.26621, 2609.31796, 2605.30218, 2609.25624. The closest to our use is 2609.31796, which finds
activation probes barely move while the decoded text does, and that flips are symmetric.

## What it does and does not threaten

Part 200's own numbers give the scale. Of 200 rows, the text changed on 76, but a label changed on 2 (name gone), 6
(broken) and 5 (Qwen rule), in both directions.

- **Counts over many rows barely move.** If about 3% of rows flip a label, in either direction, a count over 1,600
  rows moves by about 7 (the square root of 1,600 × 0.03) from one shape to another. Part 163's 387 against 103 is not
  touched by that.
- **Verdicts decided by one or two rows are not verdicts.** 163.4 held by one unit; Gate 1 passed by one unit. A
  second shape could put either on the other side.
- **Any claim about one row is one draw.** A quoted answer, "came back word for word", a hand-read example.
- **Comparing texts across runs measures the noise.** "390 of 3,200 answers identical to the published text" says
  almost nothing about the claim either run makes.
- **Exact-match gates need matched shapes.** Do-nothing checks and replays already follow this (Parts 123 and 161).
- **Activation readouts are a separate, smaller question.** 2609.31796 measured a bf16 batch-size change moving
  activations by about 1% and probe verdicts by under 0.1% at the prompt. Our atlas and probe parts are not covered
  by this plan beyond stating their batch shape.

## The plan

### 1. Now, for every part that reads generated text (from Part 164's rerun on)

These need no new science, only a recorder and a small tool.

- **Record tokens and margins.** Every generated row stores its token ids and, per position, the top two tokens and
  the margin between them, computed on the CPU. A row whose smallest margin is above two steps is very unlikely to
  change with shape (Part 201 tests this); the others are marked.
- **Shape replicate.** Regenerate a seeded sample of at least 200 rows, spread over the cells, at a second shape (one
  row per call unless the design gives a reason for another), and report for each label how many rows flip and which
  way.
- **Stability of each verdict.** `tools/shape_floor.py` takes the rows, their labels, the replicate's flip rates
  (their upper 95% bounds) and the part's scorer. It redraws the labels 1,000 times, each row flipping at its label's
  rate, and rescores. Each registered verdict is reported with the share of draws in which it comes out the same. Under
  95%, the write-up says "inside the shape floor" in the verdict column, whichever side it fell on.
- **Bars with room.** In the design's Size section, say how many rows a registered bar is from the expected value. A
  bar closer than twice the shape SD cannot be scored cleanly; move it or say so before the run.
- **Row-level claims** are made only for rows that came out the same at both shapes, or are labelled "one draw".
- **Across runs**, compare labels and verdicts with both runs' floors. Do not count identical texts.

Cost: the recorder is one CPU copy of the scores per step; the replicate is about 200 one-row decodes, 14 minutes at
Part 200's rate.

### 2. Part 201: measure it

Registered separately (`PART201-DESIGN.md`). It checks that the margin explains the flips, separates batch size from
membership, padding and process, measures label flip rates on 384 rows, and tests float32 compute on the card.

### 3. After Part 201

Decided now, so the result cannot choose its own reading:

- **If float32 on the card is shape-stable and equals the CPU reference (201.7 and 201.8 hold), and a margin of two
  steps catches at least 95% of the places bf16 leaves it (201.9), and the projected cost is at most 2 times a bf16
  run:** build margin-gated decoding. Decode in bf16 as now; at any step where a row's margin is within two steps,
  recompute that step for that row in float32 from the full prefix and take the float32 choice. The published answer
  is then the float32 greedy answer, the same at any shape and checkable token for token against the CPU reference.
  The projected cost is the measured float32 forward time times the share of steps with at least one row inside two
  steps. The first gated part keeps a shape replicate to show the floor is gone.
- **If float32 is stable but too slow to gate:** keep section 1 as the standing procedure. Use float32 decodes only
  as the replicate for parts whose claim is about single rows or named units.
- **If float32 on the card is not stable or does not match the CPU:** that is a new kernel finding. Section 1 stands,
  and the float32 path goes on the kernel audit.
- **If 201.5 fails** (flips at confident margins): stop. That is a fault, not rounding, and the audit's kernel check
  has to cover batch shape before any more reruns.

### 4. Which published claims sit inside the noise

After Part 201 gives flip rates:

1. Run `tools/shape_floor.py` on the eager reruns already published (Parts 183 to 200), using each part's own scorer
   and the measured rates, and add a stability column to each verdict table as a dated update. Expected to flag the
   verdicts that held or failed by a row or two (163.4 and Gate 1 in Part 200; 170.1 in Part 199) and to leave the
   large contrasts alone.
2. Parts still on the old kernel are not triaged for shape; their rerun is the fix and it carries section 1.
3. For the front-page claims that name single units or quote single answers, either the rerun shows the row stable at
   both shapes, or the claim is reworded as a rate.
4. One front-page line: bf16 greedy answers on this card are one draw each; counts carry a measured shape floor from
   Part 201 on.

## What this does not do

- It does not make an answer "right". Float32 removes our instrument's rounding. Where the float32 margin itself is
  tiny, the model is nearly undecided, and that is a fact about the model.
- It does not cover other models or cards until they are measured.
- It does not replace the kernel check. The SDPA fault produced confident wrong tokens, which no margin test sees.

## Proposed `PROCEDURE.md` text (not yet applied)

Under 2, Controls:

- [ ] **Shape replicate, for any readout from generated text.** Record token ids and per-position top-two margins for
  every row. Regenerate a seeded sample of at least 200 rows at a second batch shape and report each label's flips by
  direction. Report every registered verdict with its stability from `tools/shape_floor.py`; under 95% is "inside the
  shape floor". (Part 200: 76 of 200 eager answers changed with the batch shape; labels flipped on 2 to 6.)
- [ ] **Row-level claims need a stable row.** Quote, or name, only rows that came out the same at both shapes, or say
  "one draw".

Under 2, Size:

- [ ] **Say how many rows each bar is from the expected value**, beside the shape SD.

Under 7, Score and read:

- [ ] **Across runs, compare labels and verdicts, never texts.**
