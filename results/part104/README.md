# Part 104: the early-slot-neuron pattern does not replicate on new units, and the opposite-direction swap loses as many

Run 2026-10-29 on the A770: step 1, Part 92's battery on 79 new picks (11:02 to 11:06 CT); step 2, Part 99's fixed
pushes, an unswapped control and three swap runs (11:15 to 11:27 CT). Labelled by Part 92's blind Haiku reader in two passes; scored by
`part104/score104.py`, committed before any step-2 answer existed. Design and predictions registered before any run
(PART104-DESIGN.md, commit 3ba0b9d; Amendment 1, ff936c0, before any step-1 label or any step-2 answer).

## Units

Sonnet 5.5 picked 36 candidates from layers 0 to 3 and 43 from layers 4 to 15 of Part 92's remaining screen pool.
Part 92's kind rules classified 50 as slot units: 28 in layers 0 to 3 and 22 in layers 4 to 15 (gate: at least 8 and
6). Eight candidates were incomplete: one of their six answers was still unlabelled after two passes (the reader
returned no JSON object both times). Of the 50 slot units, 43 showed their persona in inst_none (23 early, 20 later).

## Results

Shown slot units lost under each swap of attention in all 16 layers (`score104.json`):

| Run | Layers 0 to 3 | Layers 4 to 15 | All | Pushed "Who are you?" coherent |
|---|---|---|---|---|
| inst_none (no swap) | (23 shown) | (20 shown) | | 41 of 43 |
| attnSFT_all: SFT's attention | 9 of 23 | 6 of 20 | 15 of 43 | 34 of 43 |
| attnEXT_all: 2 × Instruct − SFT | 8 of 22 | 7 of 20 | 15 of 42 | 32 of 42 |
| attnDPO_all: DPO's attention | 2 of 22 | 3 of 20 | 5 of 42 | 38 of 42 |

One shown unit (p104-21) is unlabelled in two runs and is left out of those runs' counts. swap99 recorded the
extrapolated attention as moving 0.9993 times SFT's distance from Instruct. None of the 50 unpushed answers labelled per
run was incoherent.

## Predictions

- **104.1 failed.** SFT's attention lost 39% of the early units and 30% of the later ones: a 9-point difference, one-sided
  Fisher p = 0.38 (needed at least 30 points and p < 0.05). Part 103's split does not replicate on new units.
- **104.3 failed.** The extrapolated run's pushed answers were coherent for 32 of 42 shown units (76%; needed 80%).
- **104.2 not read**, by the registered rule (104.3 did not hold).
- **104.4 failed.** DPO's attention lost 3 later units (needed at most 1).

## Reading (descriptive; not a registered test)

- Each large swap removes about a third of the shown units (15 of 43 under SFT's attention, 15 of 42 under the
  extrapolated one). "Lost" mixes two things: some lost answers are ones the reader marked incoherent. Counting only
  coherent answers, the losses are 9, 11 and 4 (SFT, extrapolated, DPO).
- The two large swaps remove largely different units: 5 are lost under both, against about 5.4 expected by chance from
  the two rates. A stable property of the unit, such as being fragile, would predict a larger overlap; dependence on the
  post-SFT change would predict few losses under the extrapolated swap. Neither appears.
- DPO's attention loses 5, three of them also lost under SFT's. Part 102's covariate puts DPO's attention closer to
  Instruct than SFT's (about 0.2 to 0.3% against 0.4 to 0.5% per layer); this run does not measure the size itself.
- SFT's own swap gave coherent answers for 34 of 43 shown units, one unit under the 80% bar that 104.3 applied to the
  extrapolated swap. The bar was set without measuring the SFT swap's coherence on these units.

So the data fit sensitivity of these one-answer outcomes to any attention change of this size, with which units flip
varying from change to change. The registered test of dependence against fragility (104.2) was not read, so this is a
description, not a verdict between the two. What it does undercut is reading the units lost under SFT's attention, one
by one, as units that need what SFT built.

What this does not settle: the new units are a different population from Part 92's 21. They came from a less specific
remainder of the pool, and their slot yield was much higher (50 of 79 against 21 of 53). Part 105 runs the extrapolated
swap on Part 92's own units.

## Limits

One model family, one fixed push per unit, one reader, greedy one-answer outcomes; one extrapolation step size;
attention only. 104.2's question (dependence or fragility) is not answered here.

Corrected 2026-10-29 after a Haiku cold check: the reading no longer favours fragility over dependence (104.2 was
not read, and the two swaps' losses overlap at chance), losses are split into coherent and incoherent answers, DPO's
size is attributed to Part 102, and the run list, Amendment 1 and coherence wording are fixed.
