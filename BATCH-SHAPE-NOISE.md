# Batch-shape noise: what it is and how this program handles it

Status: draft plan, 2026-11-09, by Claude (Fable 5.2). Nothing below is procedure until Part 201 has run and
`PROCEDURE.md` is changed. Numbers from existing records are printed by `part201/existing201.py`
(`results/part201/existing.md`).

## What we found

Part 200 reran Part 163 on the fixed attention kernel and then regenerated one question's 200 pushed answers one row
at a time. 76 of the 200 differed from the same rows decoded in chunks of 8. Nothing else changed: same model, same
kernel, same prompt, same push.

The cause was already in a record we had. Qwen3.5-4B runs in bf16 on the A770. At the sizes its output scores take,
a bf16 number can only move in steps of 0.0625 or 0.125, and the scores carry an error of about half a step (0.05 at
the median against the CPU float32 reference, about 0.2 at the 99th percentile). Where the model's real margin
between its top two tokens is smaller than that, bf16 cannot tell them apart. In Part 183's kernel record the card
left the reference at six places; the float32 margin there was 0.008 to 0.10, and the bf16 scores showed an exact tie
at five and one step at the sixth. About 4% of that record's positions sit at one step or less. Batch shape changes
the order of the arithmetic inside the kernels, which changes the rounding, which picks the winner at those places.
From there the two answers are different texts.

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
- **Sensitivity of each verdict.** `tools/shape_floor.py` takes the rows, their labels, the replicate pairs and the
  part's scorer, redraws the labels 1,000 times and rescores. A redraw gives each row a transition drawn from the
  replicate rows that started with the same labels, so labels move together and each direction keeps its own rate
  and denominator. It is run under named scenarios: `independent`; `shared-change-0.5` and `shared-change-1`, where
  with that probability (a mixture probability, not an estimated correlation) the rows of one unit share one draw
  for the event "this row changes"; and `upper-marginal-independent`, independent flips at each direction's upper
  95% bound, which stresses the marginal rates and is not a bound on a verdict. Each registered verdict is reported
  with the share of redraws in which it comes out the same, per scenario, with its Monte Carlo error. This is a
  sensitivity figure conditional on those scenarios and on the replicate's shape pair, not a 95% guarantee. The tool
  prints, for every starting label vector, how many target rows have it and how many replicate pairs support it; a
  vector with one pair redraws deterministically, a vector with none falls back to independent per-label flips, and
  a label value never seen as a starting value is held and counted as unmeasured. A verdict that changes when the
  unsupported rows are held is reported as depending on them, not as measured. Under 95% in any scenario, the
  verdict column says "inside the shape floor", whichever side it fell on.
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

Decided now, so the result cannot choose its own reading. The rules are read in this order and the first that
applies wins. A prediction that is not scored (no differing rows, a cut or incomplete arm, a missing timing probe)
counts as not held. The default is not to build.

1. **Stop.** 201.5 fails on its 8-step clause, or 201.9 fails on its float32-margin clause (decodes parting at
   confident margins). No more reruns until those rows are audited; the kernel check may have to cover batch shape. A
   large margin there is a trigger for the audit, not proof of a fault.
2. **Float32 arm invalid.** Gate G1, G3 or G4 fails for the float32 process, or the arm stops early. Section 1
   stands; nothing is concluded about float32 on this card. Pass K's record is kept either way. A pass-K record
   that is present and is not a float32-upcast eager record of this revision counts here; a missing one is 201.7
   not run, which is rule 4.
3. **New kernel finding.** 201.7 or 201.8 fails: float32 on the card does not match the CPU, or changes with shape.
   Section 1 stands, and the float32 path goes on the kernel audit.
4. **No build.** Any of 201.7, 201.8, 201.9 or 201.11 is not scored or not held for another reason (too few
   divergences to read, a cut arm, cost unknown or over 2). Section 1 is the standing procedure. A float32 decode
   may still serve as a part's replicate when its claim is about single rows or named units.
5. **Build a candidate.** 201.7, 201.8, 201.9 and 201.11 all hold, and the projected slowdown at the selected gate
   is at most 2. A margin-gated decoder is built: decode in bf16 as now; at any step where a row's margin is inside
   the gate, recompute that step for that row in float32 from the full prefix and take the float32 choice. The gate
   is the largest pass-A margin, in bf16 steps, at a first difference between A and F8, rounded up to a whole step,
   plus one. Its cost is priced at that width, not at 201.11's two steps: if the gate is wider than two steps and
   its projected slowdown is over 2 or cannot be priced, rule 4 applies. It is a candidate until its own part shows, on at least 200
   rows at two shapes, that its answers equal a full float32-upcast decode of the same rows; only then does a
   write-up call a gated answer the float32 answer, and only for the model, card and shapes tested. Float32 itself
   is not guaranteed identical across devices or batch sizes (PyTorch's numerical accuracy notes); Part 201 measures
   one card and two shapes. Two published results say what to expect of the candidate (`PRIOR-ART.md`, 2026-11-09):
   MarginGate replaces the step's cached keys and values when its verifier overrides a token, and arXiv 2609.26621
   found that recomputing only the output head in float32 stops helping at batch 8. Our candidate recomputes the
   whole step from the full prefix, which is neither; the cache it then continues on is still the bf16 one, so later
   steps may leave the full float32 decode. The candidate's own part is what tests that, and it reports where.

### 4. Which published claims sit inside the noise

After Part 201 gives flip rates:

1. Run `tools/shape_floor.py` on the eager reruns already published (Parts 183 to 200), using each part's own
   scorer. Only Part 200 has a replicate of its own (one question); for the others the rates are Part 201's, which
   come from Part 163's labels at one dose, so the result is labelled indicative, not calibrated, and loop or ban
   parts (199) get a note, not a number, until a rerun measures their own. Expected to flag the verdicts that held or
   failed by a row or two (163.4 and Gate 1 in Part 200; 170.1 in Part 199) and to leave the large contrasts alone.
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
- It does not separate the card from its driver. Every part so far ran on one Arc driver (32.0.101.9150); from Part
  201 on each run record carries the driver version, and a driver change is a reason to repeat the kernel check and
  the shape replicate before comparing with earlier parts.

## Proposed `PROCEDURE.md` text (not yet applied)

Under 2, Controls:

- [ ] **Shape replicate, for any readout from generated text.** Record token ids and per-position top-two margins for
  every row. Regenerate a seeded sample of at least 200 rows at a second batch shape and report each label's flips by
  direction. Report every registered verdict with its sensitivity from `tools/shape_floor.py` under the named
  scenarios, with the support table; under 95% in any is "inside the shape floor", and a verdict that depends on
  rows the replicate does not support is unresolved. (Part 200: 76 of 200 eager answers changed with the batch shape; labels flipped on 2 to 6.)
- [ ] **Row-level claims need a stable row.** Quote, or name, only rows that came out the same at both shapes, or say
  "one draw".

Under 2, Size:

- [ ] **Say how many rows each bar is from the expected value**, beside the shape SD.

Under 7, Score and read:

- [ ] **Across runs, compare labels and verdicts, never texts.**
