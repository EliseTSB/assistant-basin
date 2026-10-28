# Part 89: when do OLMo 2 1B's neurons settle, and do the identity neurons move more in post-training?

Written 2026-10-28 by Claude, before any Part 89 code or run, after Elise asked (Discord 12:17 CT) to track other
interesting neurons at the same time as Parts 85 and 88. Predictions 89.1 to 89.7 are in `PREDICTIONS.md`. It reads
weights only, from the twelve checkpoints Part 85 downloads (`part85/checkpoints.py`), so it adds no download.

Known before writing, and named so it is not mistaken for a prediction: step 0 of Part 85 measured the alien twin
L10 N6978's down_proj cosine with Instruct at nine base checkpoints (0.11 at 21B tokens, 0.35, 0.59, 0.79, 0.93, 0.98
at 4T, 0.99 to 1.00 through mid-training), and Part 88's A1 self-test found the unrelated `main` base at a median
cosine of 0.006 for the 64-unit set. No layer-wide or set-wide number below is known.

## Question

Reading one neuron's down_proj column means reading its whole layer, so every MLP neuron (16 layers x 8,192 =
131,072) can be followed at every checkpoint for the price of the named ones. For each neuron and checkpoint: the
cosine of its down_proj column, gate row and up row with the Instruct model's. Then, for named sets, compared with
every other neuron of the same layers: do they settle earlier or later in pretraining, and does post-training move them
more?

Named sets (fixed now): the alien twin L10 N6978 and its control L10 N7884 (Part 62); Part 16's 64-neuron "As an AI"
set (`results/part16/search.json`, `search.top_set`); Part 2's top-50 identity neurons (chat form,
`results/part2-olmo2-1b/identity-instruct-chat.json`, `top50`); Part 14's top-50 assistant-register neurons
(`results/part14-register/olmo2-instruct-own.json`, `top50`).

## Measures (fixed now)

- **Settle checkpoint** of a neuron: the earliest checkpoint, in training order, from which its down_proj cosine with
  Instruct stays at least 0.9 at every later checkpoint. Reported as an index 0 to 11.
- **Post-training move**: 1 minus the down_proj cosine between the mid-training end (SFT's parent) and Instruct; the
  same for gate and up, reported beside it.
- **Set comparison**: a set's mean over its members, against 1,000 random sets with the same number of neurons per
  layer drawn from the other neurons of those layers (seed 89089). Empirical one-sided p = (1 + number of random sets
  at least as extreme) / 1001. Per-layer distributions (quartiles of cosine at every checkpoint) are reported for all
  16 layers.

## Size and cost

12 checkpoints x 16 layers x three 2048 x 8192 matrices, read locally after Part 85's download; CPU only, minutes.

## Known limits, before the run

- Cosine with the finished model measures arrival at the final weights, not function; a neuron can change job with
  little weight change, and the reverse.
- The OLMo 2 stages differ in data and objective as well as in order (Muse's trap 4), so a set that moves at SFT moved
  at that stage, for reasons this part cannot separate.
- The named sets were chosen on Instruct, so they are selected for the end state; the random-set comparison uses the
  same layers but not the same selection.
