# Part 154: do the Google pushes rest on the neuron that writes Google?

Procedure: v1
Registered 2026-10-31 14:05 CT by Claude (Opus 5.5), before any data. It follows Part 153, which ended on this question. Elise, 2026-10-31 (Discord message 1566108806755127377): "Did we find any that moves the maker and name across several questions (eg. who are you and who made you), or even just reliably switches one?"

## Question

Seven of Part 153's ten neurons name Google. Part 71 found L27 N3325 writing Google through the unembedding. If the seven pushes all need that one writer, they may be several ways into one mechanism and not seven Google neurons. The Baidu, Huawei and Microsoft neurons are the comparison, since their company is not what L27 N3325 writes.

## Prior art and existing data

- Closest published work: none found: searched PRIOR-ART.md for mediation through a late output neuron. Sauers' Bing Chat neuron on Qwen3-8B is the nearest single-neuron name switch; it was not tested for dependence on a later unit.
- Our own data: Part 73, with the four maker neurons zeroed, zeroing L27 N3325 took Google from 36 to 8 of 192 and only one of 20 matched layer-27 neurons did as much. Part 72, at the opening used here, the plain model gives Google a 7% share of eight name tokens, setting L27 N3325 to zero moves Google minus Alibaba by -2.40, and setting it to eight times its negative mean moves that gap by +1.31, so more negative means more Google. Part 74, on one readout most of L24 N3949's effect runs through L27 N3325, with no control that also writes Google. Part 153, the ten neurons and L25 N4457's random direction at 20 of 24.

## Method

Qwen3.5-4B at 851bf6e8, bf16, A770, thinking off, Part 153's 24 maker phrasings and 6 guards. Part 148's push: same hook, dose 0.8, medians and seeds, answer tokens only. 120 new tokens, greedy.

Pushes, 12 rows: Part 153's ten neurons; L25 N4457's paired random direction; no push.

Zero sets, 6: none; L27 N3325; four matched layer-27 neurons, `part70.draw_matched_units(matched, k)[0]` for k = 0, 1, 2, 3 on `results/part68/b/matched.json`, fixed here before anyone looks at them. A zeroed neuron is set to 0 at the input of `layers[27].mlp.down_proj` at every position, prompt and answer, for that row only.

Stage A, teacher-forced. Opening `I was developed by **`, Part 72's. 24 phrasings, 12 pushes, zero sets none and L27 N3325, 576 rows. The push is on for the opening tokens, as in generation. Readouts at the last opening position: L27 N3325's activation before any zeroing, Part 72's eight name logits, Google minus Alibaba, the mass on the eight.

Stage B, generation. Maker: 24 phrasings, 12 pushes, 6 zero sets, one batch of 72 per phrasing, 1,728 answers. Guards: 6 questions, 12 pushes, zero sets none and L27 N3325, one batch of 24 per question, 144 answers.

A company hit and a guard hit are Part 153's rules exactly, in `part153/score153.py`. Broken is Part 86's rule.

## Controls

Applied from STANDARD-CONTROLS.md: matched controls, four activity-matched layer-27 neurons zeroed the same way; a no-push row in every batch; off-target readouts, the 6 fact guards; broken answers counted. L25 N4457's random direction is a push with no selection story behind it.

Not used: both directions and the dose ladder, because Part 155 runs them on four of these neurons. Not used: poles, because L27 N3325 is zeroed here, not pushed. Not used: a blind reader, because the rule is Part 153's word rule, and a sample of zeroed answers is read by hand.

Gate 1: 576 stage A rows and 1,872 answers, all non-empty; in every zeroed row the hook's trace reads exactly 0 for that neuron at every position, and in every unzeroed row the neuron is unchanged. Gate 2: the no-push row with no zero has a Google hit on at most 2 of 24, and the ten neurons with no zero total 128 to 174 company hits, Part 153's 151 within 15%, since the batch shape differs. Gate 3: at the opening, zeroing L27 N3325 lowers Google minus Alibaba in the no-push row on at least 20 of 24 phrasings. A failed gate leaves every prediction untestable.

## Size

- Smallest effect worth knowing: Google hits halved across the seven Google neurons, 168 unit-phrasing pairs with 105 hits in Part 153.
- Total: 1,872 generations and 576 teacher-forced rows, about 14 minutes on the A770.
- Pilot: stage A, then the first two maker phrasings, 144 answers read by eye. A failed Gate 1 or Gate 3 stops the run.

## Optimization

- Bound resource: GPU
- Measured: Part 148, 2,796 answers in 20 minutes at batch 64 on the A770, 2.33 answers/s.
- Levers: batch 72, taken, all rows of a phrasing share one prompt; teacher-forced stage A, taken; max_new_tokens stays 120, declined, to match Part 153's rule; rows leaving the batch early, declined for a 14-minute run; copy engine off, the runner's default.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 154.1 | Pooled over the seven Google neurons, company hits with L27 N3325 zeroed are at most half the count with no zero | medium | Claude |
| 154.2 | Pooled over the Baidu, Huawei and Microsoft neurons, company hits with L27 N3325 zeroed are at least 75% of the count with no zero | medium | Claude |
| 154.3 | Pooled over the seven Google neurons, each of the four matched zero sets keeps at least 75% of the count with no zero | medium | Claude |
| 154.4 | L25 N4457's random direction: Google hits with L27 N3325 zeroed are at most half its count with no zero | low | Claude |
| 154.5 | At the opening, at least five of the seven Google neurons move L27 N3325's median activation more negative than the no-push row by more than the most any of the three other neurons does | low | Claude |
| 154.6 | Zeroing L27 N3325 adds at most 5 percentage points of broken answers, pooled over the eleven pushes on the maker phrasings | medium | Claude |
| 154.7 | Pooled over the twelve rows, guard hits with L27 N3325 zeroed are within 3 of the count with no zero | high | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 154 (registered ...)".

## Reading rules

`part154/score154.py` computes every number. Verdicts use point estimates. All answers are published. 154.5's move is the phrasing's activation in the pushed row minus the same phrasing's in the no-push row, then the median over 24, and more negative counts as a larger move because Part 72 shows that direction writes Google. If 154.1 fails, the seven Google pushes are not reported as resting on L27 N3325 at this dose. If 154.1 holds and 154.3 fails, the fall is not reported as specific to L27 N3325. Answers that lose Google are read by hand for what replaces it, labelled exploratory.

## Limits

One dose, one greedy answer per phrasing. L27 N3325 writes Google directly, so zeroing it removes part of the Google logit whatever route a push took. A fall shows the answer needs that writer, and does not show the push works through it; 154.5 is the only readout of the route, at one opening. The seven neurons were selected on Part 66's phrasings in Part 153, so the no-zero arm may run a little high.
