# Part 161: do random directions picked by the same rule hold up like neurons?

Procedure: v1
Registered 2026-10-31 17:39 CT by Claude (Opus 5.5), before any data. It follows Part 160, where random directions qualified at picking as often as neurons, 41 against 30 at 0.6 and 6 against 7 at 1.0, and only the neurons were laddered. Elise, 2026-10-31 (Discord message 1566108806755127377): "Did we find any that moves the maker and name across several questions (eg. who are you and who made you), or even just reliably switches one?"

## Question

Part 160 tested picked neurons against their paired random directions, which were not picked, so the contrast favoured the neurons. Give random directions the same chance: ladder every random direction that passed the picking rule, and the 17 neurons the cap left out. If picked random directions name their company as reliably as picked neurons on held-out phrasings, then a reliable company switch is not evidence for a special neuron.

## Prior art and existing data

- Closest published work: Mayner on X (x.com/wmayner/status/2116628249153515740), on Qwen3-8B: one neuron yields a persona about as often as a random combination of an MLP block, while small sets of neurons are privileged. Malla, Choi and Choi (arXiv:2609.06951): a push with no content moves a model toward its defaults.
- Our own data: Part 160's screen, 930 units on two questions at two doses, and its ladder of 19 units. Picked at 0.6: neurons Tencent 16, Google 9, Baidu 3; random directions Tencent 22, Baidu 6, Microsoft 6, Google 2. Part 148, 10 neurons against 5 random directions at 0.8 under a stricter rule.

## Method

Qwen3.5-4B at 851bf6e8, bf16, A770, thinking off, Part 148's push: same hook, medians and random seeds, answer tokens only, greedy. Part 160's ladder: doses 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, Part 153's 24 maker phrasings, 80 new tokens, batch 64, a company hit by Part 153's rule.

Units, fixed now by Part 160's stored screen and its picking rule, which is `picking_rule` of `part160/run160.py`, applied to each arm:

- The 45 random directions picked at 0.6 or 1.0, each with the company it was picked for, as a `random` arm row at the same seed as Part 148.
- The 17 neurons Part 160's cap left out, with their picked companies.
- Each of those 17 neurons' paired random direction at the neuron's picking dose only, so Part 160's flag can be applied.
- L24 N5017 at +0.8, a positive control, and one no-push row.

Part 160's 19 laddered neurons are reused from its answers. A unit picked at both doses runs once and is scored for each set with that set's company. Peak dose is Part 160's rule. A hit at picking dose of at least 12 of 24 is reliable, Part 153's bar.

## Controls

Applied from STANDARD-CONTROLS.md: matched random controls, the point of this part; a dose ladder; a do-nothing control, the no-push row; broken answers counted by Part 86's rule, which is not a readability test, so 8 answers per arm-by-dose cell are labelled by hand and the labels saved as `hand_read_labels.jsonl`, exploratory.

Not used: both directions and poles, because picking ran at positive dose. Not used: guards, because no answer here is about a fact. Not used: a blind reader, because the rule is Part 153's word rule.

Gate 1: every planned row present and non-empty. Gate 2, run first: L24 N5017 at +0.8 hits Google on 15 to 22 of 24, and the no-push row has a hit for any picked company on at most 2 of 24. If Gate 2 fails the run stops before the ladder. Gate 3: the 45 and 17 are reproduced from Part 160's stored screen. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: random directions reaching half the neurons' reliable share.
- Total: 9,384 generations, about 65 minutes on the A770: 62 units by 6 doses by 24 phrasings is 8,928, plus 17 paired directions by 24 is 408, plus 24 for L24 N5017, plus 24 no-push.
- Pilot: Gate 2, 48 generations, run first. Stop rule: a failed gate, or under 1.5 answers per second over the first 4 ladder batches.

## Optimization

- Bound resource: GPU
- Measured: Part 160, 7,696 generations in 3,190 s at batch 64 on this model and hook, 2.4 answers/s.
- Levers: max_new_tokens 80, taken from Part 160's rescore of Part 156. Part 160's 19 neurons reused rather than rerun, taken. Batch 64, taken. Resume by skipping written rows, taken. Copy engine off, the job runner's default.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 161.1 | Over both picking doses, the share of picked random directions that are reliable at their picking dose is at least half the share of picked neurons that are | medium | Claude |
| 161.2 | With all picked neurons laddered, the set picked at 1.0 has a median peak dose at least 0.2 above the set picked at 0.6 | low | Claude |
| 161.3 | The random directions picked at 1.0 have a median peak dose at least 0.2 above those picked at 0.6 | low | Claude |
| 161.4 | Among random directions picked at 0.6, at least half peak at 0.6 or below | medium | Claude |
| 161.5 | Among units of either arm picked for Tencent at 0.6, at least half hit Tencent on 4 of 24 or fewer at 1.0 | medium | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 161 (registered ...)".

## Reading rules

`part161/score161.py` computes every number. Verdicts use point estimates. All answers are published. If 161.1 holds, a reliable company switch on this test is not evidence that a neuron is special, and Parts 153, 155 and 156's reliable neurons are read in that light. If it fails, neurons hold up where random directions do not, and picking alone does not explain them. 161.2 is also reported leaving out each neuron with any hit by its paired random direction at its picking dose, and within Google alone. If 161.2 and 161.3 both hold, the band follows the picking dose for neurons and random directions alike, so a band says more about the push than the neuron.

## Limits

One model, one sign. Picking used one answer per question. Random directions are compared at the same push size as their paired neurons, not at the same activity. A word rule can count a passing mention, and Part 86's rule passes nonsense. The 1.0 sets are small.

## Amendment 1, 2026-10-31 19:20 CT, after the run and its cold check

Counts in the text above are left as registered; these are what happened. Part 160's ladder held 18 distinct neurons, not 19, because L11 N5647 was kept at both doses. 9,312 answers were generated, not 9,384, because three paired directions at 0.6 were already picked random directions, so 14 paired directions ran, not 17.

The 18 reused neurons came from Part 160's batches and the other units from Part 161's, and a greedy answer depends on batch shape: L24 N5017 at 0.8 gave 21 of 24 there and 19 of 24 here, with 7 of 24 answers identical. `part161/rerun161.py` regenerates those 18 neurons at all six doses, 2,592 answers, about 15 minutes, written to `results/part161/rerun.jsonl`. The registered verdicts stay as scored. This is a sensitivity, reported beside them, with one prediction added below.

| # | Prediction | Confidence |
| --- | --- | --- |
| 161.6 | With the 18 neurons regenerated, between 22 and 28 of the 37 neuron memberships are reliable at their picking dose | high |
