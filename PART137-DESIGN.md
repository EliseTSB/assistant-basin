# Part 137: does the dog pair hold on questions that did not pick it?

Procedure: v1
Registered 2026-10-30 23:15 CT by Claude (Opus 5.5), before any answer was generated. Elise, 2026-10-30, voice of the dog thread:
she asked to test the dog neurons further. The commit that adds this file is the registration.

## Question

Part 134's best cell scored 11 of 24, 10 of 16 on non-task questions, but it was the best of three on those same questions. With the cell
fixed in advance and 24 new questions, does the pair still give a dog point of view, and does its margin over random partners hold?

## Prior art and existing data

- Closest published work: none on holding a persona with a pair of neurons at a sub-threshold dose; Part 134 is the base.
- Our own data: results/part134, Part 109c, Part 127.

## Amendment (23:25 CT 2026-10-30, after a code review, before any answer was generated)

Part 134 generated 100 tokens at batch 8, not 120 at batch 32, so this part uses 100 and batch 8: an A770 greedy answer depends slightly on
batch shape. The task rules are part137/score137.py's, written for these eight questions in Part 134's form, not Part 134's own.

## Method

Cell fixed: p92-08 at 0.8 with p92-09 at 0.6, OLMo 2 1B Instruct, Part 134's revision, dtype, hook, tokens and greedy decoding. Questions:
part137/questions137.json, 16 non-task and 8 task, none used in Part 134 (the build checks). Arms: pair; each neuron alone; each with two
new random partners of the same size (seeds 1371 to 1374); unpushed. Reader: Part 134's DeepSeek prompt and endpoint, with Luna reading
every pair answer and an equal seeded draw of the rest. A task answer is also checked by Part 134's rules, fixed in the scorer.

## Controls

Applied from STANDARD-CONTROLS.md: do-nothing hook, matched random partners, off-target task questions, broken answers counted. Not used:
both directions and dose ladder, because Part 127 and Part 134 cover them, and answer polarity, because no yes or no readout is used.

Gates: Part 134's gates, the same thresholds. Gate 1: do-nothing hook matches plain bit for bit. Gate 2: DeepSeek and Luna each read at least
11 of 12 known examples. Gate 3: Luna's kappa on the endpoint at least 0.60. Gate 4: no unpushed answer is on the endpoint. A failed
gate stops the reading.

## Size

- Smallest effect worth knowing: a pair count 3 non-task answers above the best other arm.
- Total: 8 arms x 24 questions, 192 answers at 100 tokens, about 4 minutes.
- Pilot: not needed: under 1,000 generations and Part 134 measured the rig.

## Optimization

- Bound resource: GPU
- Measured: 0.8 answers/s at batch 8 in Part 134's test.
- Levers: batch 32 declined, batch shape must match Part 134; a shared prefix declined, under 5 minutes.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 137.1 | The pair reaches the endpoint on at least 6 of the 16 non-task questions. | medium | Claude |
| 137.2 | The pair beats each neuron alone by at least 3 non-task answers. | medium | Claude |
| 137.3 | Each random-partner arm stays at or below half the pair's non-task count. | medium | Claude |
| 137.4 | At most 3 of the 8 task answers are right under the pair. | medium | Claude |
| 137.5 | At least 80% of the pair's endpoint answers open with a refusal or apology. | high | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 137 (registered 2026-10-30, PART137-DESIGN.md): does the dog pair hold on questions that did
not pick it?".

## Reading rules

part137/score137.py computes every number. Luna's count for the pair is reported beside DeepSeek's, and a verdict that flips is reported.
A hand reading of the pair's endpoint answers is written after the counts and marked as such.

## Limits

One seed, greedy, 100 tokens, one cell, 24 questions. The endpoint counts speaker other, which Part 134 found generous.
