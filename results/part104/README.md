# Part 104: the early-slot-neuron pattern does not replicate on new units, and the opposite-direction swap loses as many

Run 2026-10-29 on the A770: step 1, Part 92's battery on 79 new picks (11:02 to 11:06 CT); step 2, Part 99's fixed
pushes and four swap runs (11:15 to 11:27 CT). Labelled by Part 92's blind Haiku reader in two passes; scored by
`part104/score104.py`, committed before any step-2 answer existed. Design and predictions registered before any run
(PART104-DESIGN.md, commit 3ba0b9d; Amendment 1, ff936c0, before any answer).

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

One shown unit (p104-21) is unlabelled in two runs and is left out of those runs' counts. The extrapolated attention
moved 0.9993 times SFT's distance from Instruct (bf16 rounding). No run's unpushed answers were incoherent.

## Predictions

- **104.1 failed.** SFT's attention lost 39% of the early units and 30% of the later ones: a 9-point difference, one-sided
  Fisher p = 0.38 (needed at least 30 points and p < 0.05). Part 103's split does not replicate on new units.
- **104.3 failed.** The extrapolated run's pushed answers were coherent for 32 of 42 shown units (76%; needed 80%).
- **104.2 not read**, by the registered rule (104.3 did not hold).
- **104.4 failed.** DPO's attention lost 3 later units (needed at most 1).

## Reading (descriptive; not a registered test)

- The two swaps move attention the same distance from Instruct in opposite directions, and each loses 15 slot units.
  Only 5 units are lost under both; the rest differ (10 lost to one swap only, in each direction).
- DPO's smaller change loses 5, three of them also lost to SFT's.
- Coherence falls about equally under the two large swaps (34 of 43 and 32 of 42). SFT's own swap would also have
  missed the 80% bar that 104.3 applied to the extrapolation; the bar was set without measuring the SFT swap's
  coherence on these units.

Together these fit fragility better than dependence. A change to attention of this size loses about a third of the
slot units whichever way it points, in early and later layers alike, and the units lost depend on the change rather than
on a stable property such as layer. A smaller change loses fewer. This weakens the reading of Parts 102 and 103 that SFT
built the attention these self-descriptions need: some or much of that loss may be the sensitivity of a greedy,
one-answer outcome to any change of that size.

What this does not settle: the new units are a different population from Part 92's 21. They came from a less specific
remainder of the pool, and their slot yield was much higher (50 of 79 against 21 of 53). So the failed replication may
be a population difference rather than a refutation. A test on Part 92's own 21 units with the extrapolated swap would
separate the two. That is the natural next step, registered before it runs.

## Limits

One model family, one fixed push per unit, one reader, greedy one-answer outcomes; one extrapolation step size;
attention only. 104.2's question (dependence or fragility) is answered here only descriptively.
