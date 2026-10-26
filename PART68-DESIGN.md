# Part 68: which neurons drive the later maker neurons? (written 2026-10-26, before any code or run)

Author: Claude. Follows Part 67 (same model, prompts and activation measure), written before Part 67's result exists.
Predictions 68.1 to 68.6 are in PREDICTIONS.md and are pushed before the code exists.

## Question

Part 67 asks whether the four maker neurons drive each other. Part 68 searches all earlier MLP neurons for the ones that
drive the two later maker neurons, L11 N8868 and L29 N776, choosing candidates on one half of the prompts and testing
them on the other half.

## Prompts and split (fixed now)

Part 66's prompts: DIRECT (16), INDIRECT (8), POSITIVE (1). Search half: DIRECT[0::2] (8) and INDIRECT[0::2] (4).
Test half: DIRECT[1::2] (8), INDIRECT[1::2] (4) and POSITIVE (1). No candidate is chosen on a test prompt.

## Search (search half only)

- Target activation t: the target unit's down-projection input, mean over prompt positions, as in Part 67.
- For every MLP unit u in layers 0 to layer(target) - 1: attribution A_u = mean over search prompts of the sum over
  positions of (-a_u * dt/da_u), a first-order estimate of the change in t when u is zeroed. One backward pass per
  prompt and target, bf16 activations, float32 accumulation.
- Candidates: the 20 units with the largest |A_u| per target.

## Test (test half only)

- Zero each candidate alone at every position and measure Delta on its target (Part 67's normalized Delta, with the
  standard deviation taken over unchanged test-half positions). Also zero all 20 together.
- Null: 200 random units per target, drawn from the same layers as the 20 candidates in the same proportions (seed
  6800), excluding the candidates and the four maker neurons, each zeroed alone.
- A candidate "replicates" if its |Delta| exceeds the 95th percentile of the null's |Delta| and has the sign its
  attribution predicted.

## Controls

- Causality (68.5): attribution of any unit in a layer at or after the target's layer is exactly 0.
- First-order check (68.6): across the 20 candidates and 200 null units, measured Delta correlates with the attribution's
  prediction on the SEARCH half (Spearman above 0.3); if not, the attribution does not estimate zeroing and the
  candidate list is read only as a list.

## Reading rules (written now)

- If 68.5 fails, nothing is read. If 68.6 fails, the replication counts are still reported but labelled as a test of an
  attribution that does not track zeroing.
- By chance, about 1 of 20 candidates (5%) passes the 95th-percentile test with either sign, and half of those with the
  predicted sign; 3 or more replicating candidates per target is the threshold for "the search found drivers".
- A shared driver is a unit in both targets' top 20 that replicates for both. Overlap is compared with the overlap two
  random 20-unit draws from the same layers would have.

## Amendment 1 (written 2026-10-26 03:5x CT, after the Part 68 score, before this check is built or run)

Part 68's null drew random units from the candidates' layers. Part 67 showed that bar is easy to clear: most random
units are nearly silent on these prompts, so zeroing one moves nothing. This amendment adds the stricter null Part 67b
used. The registered verdicts 68.1 to 68.6 stand as scored; this check is added beside them, not in their place.

- For each candidate that replicated on the test half (19 for L11 N8868, 7 for L29 N776), compute every unit's mean
  absolute down_proj input over the test-half prompts' unchanged runs (all positions, float32).
- Draw the 50 units of the candidate's layer whose mean is closest to the candidate's, excluding both targets' 20
  candidates and the four maker neurons; ties broken by unit index (Part 67b's rule).
- Zero each matched unit alone on the test half and compute its normalized Delta on that target exactly as Part 68
  does. The same run also re-measures the unchanged condition and every replicating candidate.
- A candidate survives the matched null if its |Delta| exceeds the largest |Delta| among its 50 matched units AND its
  sign is the one its attribution predicts.

Reading rule: if 68.10 fails, nothing from this amendment is read until the rig is checked.
