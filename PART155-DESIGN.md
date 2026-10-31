# Part 155: how do four company neurons answer to dose and sign?

Procedure: v1
Registered 2026-10-31 14:05 CT by Claude (Opus 5.5), before any data. It follows Part 153, whose single dose left standard control 8 open. Elise, 2026-10-31 (Discord message 1566108806755127377): "Did we find any that moves the maker and name across several questions (eg. who are you and who made you), or even just reliably switches one?"

## Question

Part 153 tested each neuron at dose 0.8 only. Is naming a company a threshold or a slope, does it survive smaller pushes, and does the negative arm do anything? Part 62's alien neuron was a threshold, 0 to 1 of 24 up to one times its push and 19 of 24 at one and a half.

## Prior art and existing data

- Closest published work: none found: searched PRIOR-ART.md for a dose ladder on a single-neuron maker switch.
- Our own data: Part 153, at dose 0.8, L24 N5017 20 of 24, L21 N7134 15, L23 N1743 14, L23 N2993 17, each random pair at 0, and L24 N5017 7 of 8 on identity. Part 62, the alien neuron's threshold. Part 135, up and down are not two ends of one idea at that resolution, on a different question.

## Method

Qwen3.5-4B at 851bf6e8, bf16, A770, thinking off, Part 148's push: same hook, medians and seeds, answer tokens only. 120 new tokens, greedy.

Units: L24 N5017 Google, L21 N7134 Baidu, L23 N1743 Huawei, L23 N2993 Microsoft, each with its paired random direction, the four neurons with no random hit in Part 153. Doses: 0.2, 0.4, 0.8, 1.2, 1.6 and their negatives. A negative random push is the same direction with its sign flipped.

Questions: Part 153's 24 maker phrasings, 8 identity phrasings and 6 guards, 38 in all. Rows per question: 8 units times 10 doses, plus one no-push row, 81, in one batch. A company hit, a guard hit and broken are Part 153's rules.

## Controls

Applied from STANDARD-CONTROLS.md: both directions; matched random controls, one paired direction per neuron at every dose; off-target readouts, the 6 guards; broken answers counted; a dose ladder, the point of this part.

Not used: poles as separate units, because a neuron and its negative arm are reported side by side here. Not used: a do-nothing hook, because the no-push row shares every batch and is compared with Part 153's plain rows in Gate 2. Not used: a blind reader, because the rule is Part 153's word rule, and a sample in every cell is read by hand.

Gate 1: 3,078 answers, all non-empty, 81 rows per question. Gate 2: at dose +0.8 the four neurons total 56 to 76 company hits on the maker phrasings, Part 153's 66 within 15%, and the no-push row has a company hit for any of the four companies on at most 2 of 38. A failed gate leaves every prediction untestable.

## Size

- Smallest effect worth knowing: a neuron going from under 4 of 24 to over 11 of 24 within one step of the ladder.
- Total: 3,078 generations, about 22 minutes on the A770.
- Pilot: the first two maker phrasings, 162 answers read by eye. A row whose answer is empty stops the run.

## Optimization

- Bound resource: GPU
- Measured: Part 148, 2,796 answers in 20 minutes at batch 64 on the A770, 2.33 answers/s.
- Levers: batch 81, taken, all rows of a question share one prompt; max_new_tokens stays 120, declined, to match Part 153's rule; rows leaving the batch early, declined for a 22-minute run; copy engine off, the runner's default.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 155.1 | Each of the four neurons has at least as many company hits at +0.8 as at +0.4 | medium | Claude |
| 155.2 | At least two of the four neurons have a step of the positive ladder from at most 3 of 24 to at least 12 of 24 | medium | Claude |
| 155.3 | No neuron reaches 6 of 24 on its company at any negative dose | medium | Claude |
| 155.4 | Pooled over the four neurons on the maker phrasings, the broken share at +1.6 is at least twice the share at +0.8 | medium | Claude |
| 155.5 | At every positive dose where the four neurons total at least 24 of 96, their four random pairs total under half that | medium | Claude |
| 155.6 | L24 N5017's lowest dose reaching 12 of 24 on maker is no higher than its lowest dose reaching 4 of 8 on identity | medium | Claude |
| 155.7 | At +0.4 and -0.4, each of the four neurons keeps at least 5 of 6 guard hits | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 155 (registered ...)".

## Reading rules

`part155/score155.py` computes every number and draws the dose curve. Verdicts use point estimates. All answers are published. In 155.2 a step is two neighbouring doses on 0.2, 0.4, 0.8, 1.2, 1.6. If 155.2 fails, the write-up calls the responses gradual only if the counts rise over at least three steps, and otherwise describes them as measured. If 155.5 fails, no dose below that point is reported as neuron-specific.

## Limits

Four neurons chosen for clean results in Part 153, one greedy answer per phrasing. The ladder is coarse, so a threshold can only be placed between two doses. A dose here is Part 148's unit and does not compare directly with Part 62's.
