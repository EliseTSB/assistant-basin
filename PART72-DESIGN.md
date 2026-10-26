# Part 72: does pushing the Google neuron alone move an unaltered Qwen toward Google?

Written 2026-10-26 06:41 CT by Claude, before any code or run. Predictions 72.1 to 72.4 are in PREDICTIONS.md and are
pushed before the run.

## Why

Parts 69 to 71 show L27 N3325 is necessary for the Google fallback and writes `Google` directly. Part 72 asks the
other direction: with every other unit untouched, does pushing it alone make the model's next maker token Google?

## Method

- `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, A770; Part 69 Amendment 1's prompts
  (Part 66's 24 held-out maker questions + `I was developed by **`) and its eight readout tokens.
- m(u) = unit u's mean down_proj input over all positions of these 24 prompts, unchanged (signed).
- Conditions: unchanged; L27 N3325 clamped at every position to k x m(3325) for k = 0, 1, 2, 4, 8 (Part 25's clamp,
  the same operation as zeroing with a different value); and, as the null, each of L27 N3325's 50 activity-matched
  layer-27 units from Part 68b's `matched.json` clamped alone to 8 x its own m(u).
- Readout: Google's share of the softmax over the eight name tokens, averaged over the 24 questions; the gap change
  Dg (Google minus Al logit, against unchanged); and the full-vocabulary mass on the eight.

## Reading rules (written now)

- 72.4 is the control: if it fails, 72.1 to 72.3 are not read.
- A next-token readout after a forced opening is not a sampled answer; a sampled check is the next step if 72.1 holds.
