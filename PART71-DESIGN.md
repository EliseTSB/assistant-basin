# Part 71: what do the two late "Google" neurons write?

Written 2026-10-26 06:07 CT by Claude, before any code or run. Predictions 71.1 to 71.3 are in PREDICTIONS.md and are
pushed before the run.

## Why

Part 69: zeroing L27 N3325 and L24 N3949 lowers the next-token logit of `Google` by about 4. Part 70: with the four
maker neurons off, the Google answers need these two. The simplest mechanism would be that each neuron's output
direction itself raises `Google` at the unembedding. Part 71 reads that directly from the weights.

## Method

- `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, A770.
- Write direction of unit u in layer L: column u of that layer's `mlp.down_proj` weight, multiplied by the sign of
  the unit's mean down_proj input at the last position over Part 66's 24 held-out maker questions with the forced
  opening `I was developed by **` (Part 69 Amendment 1's prompts), so the direction is the one the unit actually adds.
- Readout: the model's own final norm module applied to that vector, then the unembedding (`lm_head`); this is a
  direct-logit view that ignores every later layer.
- Google rank: the best rank among the token ids for `Google` and ` Google` over the whole vocabulary.
- Null: 50 units drawn uniformly from the same layer (numpy default_rng(7100 + layer)), excluding the six units
  named here, read the same way (signed by their own mean activation).
- Also reported, no prediction: the top 20 tokens of each of the four maker neurons (L6 N753, L7 N1169, L11 N8868,
  L29 N776) and of the two late units.

## Reading rules (written now)

- 71.3 is the control: if it fails, 71.1 and 71.2 are not read.
- A direct-logit result says what a unit writes at the end, not how later layers use it.
