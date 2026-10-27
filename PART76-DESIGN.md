# Part 76: which layer-27 units carry L24 N3949's effect, and does writing Google suffice?

Written 2026-10-26 by Claude, after Part 74b, before any Part 76 code or run. Predictions 76.1 to 76.6 are in
PREDICTIONS.md and are pushed before the run.

## Why

Part 74b found that, of 15 activity-matched layer-27 units, only L27 N3325 carries L24 N3949's effect on the
forced-opening Google-minus-Alibaba gap. Its own limits named two gaps: none of the 15 nulls writes Google, so the test
cannot separate "L27 N3325 is the route" from "the one Google writer tested is the route"; and the patch was written at
every position, so the route may run through attention at earlier positions. Part 76 closes both, and replaces the
15-unit sample with every layer-27 unit.

## Method

Part 74's model, revision, prompts (24 questions), opening (`I was developed by **`), readout and maker set, unchanged.
g is the Google logit minus the Alibaba logit at the last position. Conditions, per question, all with the four makers
zeroed:

- **A**: makers zeroed. **B**: A plus L24 N3949 zeroed. Both record every layer-27 unit's down_proj input at every
  position.
- **Attribution (all units).** One backward pass in A gives dg/da_u at every position for every layer-27 unit u.
  The first-order estimate of each unit's pure indirect effect is PIE_hat(u) = sum over positions of
  (a_u^B - a_u^A) * dg/da_u, averaged over questions. Part 68's attribution code, pointed at g instead of a unit.
- **Exact PIE (validation set).** E(u): A with unit u alone set to its B values at every position;
  PIE(u) = mean over questions of g(E(u)) - g(A). The set: L27 N3325; the 10 other units with the most negative
  PIE_hat; the 10 layer-27 units (other than L27 N3325) whose direct readout ranks Google highest; 5 units drawn at
  random with seed 76. Overlaps are run once.
- **Direct Google rank** of every layer-27 unit: Part 71's method (the down_proj column, signed by the unit's mean
  last-position activation in A, through the final norm and unembedding; rank of the Google token).
- **E_all27**: A with every layer-27 unit's down_proj input set to its B values. Its PIE is the effect that runs
  through layer 27's MLP inputs as a whole.
- **E_last**: A with L27 N3325 set to its B value at the last position only.
- **A_self**: A with L27 N3325 set to its own A values (Part 74's noise row).

Size: 24 questions x (A, B, one backward, A_self, E_all27, E_last, at most 26 E(u) rows) = at most about 760 forward
passes and 24 backward passes, no sampling. Part 74b (48 passes) took 60 s on the A770 with model load, so the run is
about 10 minutes; an optimization-pass record is written before launch.

## Reading rules (written now)

- 76.1 is the control. If it fails, nothing else is read.
- 76.2 is the instrument check on the attribution. If it fails, the estimates are reported but not used to claim that
  the validation set holds the strongest units; 76.3 is then read over the validated units only, and says so.
- Claims about any single unit rest on exact PIE, never on PIE_hat alone.
- If 76.3 holds, L27 N3325 is the only layer-27 unit among those validated through which more than 0.2 of L24 N3949's
  effect runs on this readout. If 76.4 holds, writing Google is not enough: Google writers that L24 N3949 does not move
  carry nothing.
- 76.5 says how much of the layer-27 route is L27 N3325; 76.6 says whether that route acts at the answer position.

## Limits (written now)

- Part 74's and 74b's limits all apply: one forced opening and next token, not sampled answers; the gap does not
  separate Google falling from Alibaba rising; the makers zeroed; bf16 resolves about 0.125 nats per question.
- Attribution patching is first-order. L27 N3325 moves by about 2.4 between A and B, which may be too far for a linear
  estimate; that is why 76.2 exists and why every claim uses exact PIE.
- E_all27 sets the MLP inputs of layer 27 only; effects through layer 27's attention or through other layers are not in
  it. One layer, one model, one revision.

## Amendment 0 (while writing the runner, before any run)

Several layer-27 units can rank Google at the same position (the toy test showed ties at rank 1). The 10 Google writers
are therefore sorted by rank, then by margin (the best Google logit minus the best other token's logit in the unit's
signed direct readout, larger first), then by unit number. Every unit's rank and margin are saved.
