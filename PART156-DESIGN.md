# Part 156: how wide is each company peak?

Procedure: v1
Registered 2026-10-31 15:25 CT by Claude (Opus 5.5), before any data. It follows Part 155, whose nearest doses to the peak were 0.4 and 1.2. Elise, 2026-10-31 (Discord message 1566108806755127377): "Did we find any that moves the maker and name across several questions (eg. who are you and who made you), or even just reliably switches one?"

## Question

In Part 155 the four neurons named their company on 16 to 20 of 24 phrasings at dose 0.8 and on 0 to 1 at 0.4; at 1.2 two read 0 and two read 8. These neurons were selected at exactly 0.8, so that peak may be narrow, or broad and centred somewhere else. Is it a spike at one dose or a band, and does its top sit at 0.8?

## Prior art and existing data

- Closest published work: none found: searched PRIOR-ART.md for a dose ladder on a single-neuron maker switch.
- Our own data: Part 155, the ladder 0.2, 0.4, 0.8, 1.2, 1.6 on these units. Part 62, the alien neuron rising from 0 to 1 of 24 at one times its push to 19 of 24 at one and a half. Part 118, where an earlier reading of a narrow window was cut down by its cold check.

## Method

Qwen3.5-4B at 851bf6e8, bf16, A770, thinking off, Part 148's push: same hook, medians and seeds, answer tokens only, 120 new tokens, greedy. Part 155's four neurons, L24 N5017 Google, L21 N7134 Baidu, L23 N1743 Huawei, L23 N2993 Microsoft, each with its paired random direction. Doses +0.5, +0.6, +0.7, +0.8, +0.9, +1.0, +1.1. Part 153's 24 maker phrasings. Rows per phrasing: 8 units times 7 doses plus a no-push row, 57, in one batch. A company hit and broken are Part 153's rules.

## Controls

Applied from STANDARD-CONTROLS.md: matched random controls, one paired direction per neuron at every dose; broken answers counted; a dose ladder, the point of this part.

Not used: both directions, because Part 155 ran the negative arm and found no company hit at any negative dose. Not used: guards, because Part 155 read them across this range. Not used: a blind reader, because the rule is Part 153's word rule, and a sample in every cell is read by hand.

Gate 1: 1,368 answers, all non-empty, 57 rows per phrasing. Gate 2: at +0.8 the four neurons total 59 to 81 company hits, Part 155's 70 within 15%, since the batch shape differs, and the no-push row has a company hit for any of the four companies on at most 2 of 24.

## Size

- Smallest effect worth knowing: a neighbouring dose differing from 0.8 by 6 of 24 phrasings.
- Total: 1,368 generations, about 11 minutes on the A770.
- Pilot: Part 155's pilot and main run, the same runner on the same units with only the dose list changed. An empty answer stops the run.

## Optimization

- Bound resource: GPU
- Measured: Part 155, 3,078 answers in 24.5 minutes at batch 81 on the A770, 2.09 answers/s.
- Levers: batch 57, taken, one batch per phrasing; max_new_tokens stays 120, declined, to match Part 153's rule; copy engine off, the runner's default.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 156.1 | At least three of the four neurons reach 12 of 24 at a dose other than +0.8 | medium | Claude |
| 156.2 | At least two of the four neurons have a dose other than +0.8 with more company hits than +0.8 | medium | Claude |
| 156.3 | Each of the four neurons has at most 3 of 24 at +0.5 | medium | Claude |
| 156.4 | For at least three of the four neurons, the doses reaching 12 of 24 are neighbours with no gap and number at most four | medium | Claude |
| 156.5 | At every dose the four random pairs total under a quarter of the four neurons' total, wherever the neurons total at least 24 of 96 | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 156 (registered ...)".

## Reading rules

`part156/score156.py` computes every number and draws the curve. Verdicts use point estimates. All answers are published. If 156.1 fails, the peak is reported as confined to one ladder point at this spacing. Answers around each edge are read by hand for what the model says instead, labelled exploratory.

## Limits

Four neurons selected at 0.8, one greedy answer per phrasing, maker phrasings only. Selection on dose still applies to these units; Part 157 selects at another dose.
