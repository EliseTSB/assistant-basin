# Part 67: do the four maker neurons feed one another? (written 2026-10-26, before any code or run)

Author: Claude. Elise asked (2026-10-26 02:30 CT) for a night of looking for interesting neurons and the circuits
between them. Predictions 67.1 to 67.6 are in PREDICTIONS.md and are pushed before the code exists.

## Question

Part 25 found four MLP neurons in Qwen3.5-4B (revision 851bf6e8) whose joint zeroing moves the model's maker answer to
Google; Part 66 showed the effect holds on 24 new phrasings (21.9% of answers). They were found as a SET. Do they act as
a chain, the earlier ones driving the later ones, or as four separate inputs?

## Measurement

- Model: Qwen3.5-4B at Part 25's revision, bf16, A770, chat template with no system prompt (`neuron_report.build_prompt`).
- Prompts: Part 66's 24 held-out maker phrasings plus "Who created you?" (25 prompts). Prefill only, no generation, so
  every number is deterministic up to kernel nondeterminism.
- A neuron's activation is the down-projection input at its unit (the quantity Part 25 zeroes), recorded at every
  prompt position.
- Targets: L6 N753, L7 N1169, L11 N8868, L29 N776. Six ordered pairs i -> j with layer(i) < layer(j).
- Effect of zeroing i on j: Delta_ij = mean over prompts of the mean over positions of (a_j with i zeroed at every
  position) minus (a_j unchanged), divided by the standard deviation of a_j over all unchanged positions.
- Null: for each upstream layer, 50 random neurons from the same layer (seeds 6700 to 6749 through Part 25's
  `_matched_random`, the four targets excluded), each zeroed alone, with the same Delta on the same downstream neuron.
  Empirical p_ij = (1 + #random with |Delta| >= |Delta_ij|) / 51.
- Joint: zero L6 N753, L7 N1169 and L11 N8868 together and measure L29 N776, against 50 random triples matched by layer.
- Direct weight path (exploratory): V_ij = sum over d of W_down^(layer i)[d, i] * g_j[d] * W^(layer j)[j, d] for W in
  the gate and up projections, where g_j is layer j's pre-MLP RMSNorm weight. It ignores the norm's input scaling and
  every intervening layer, so it measures only the direct residual-stream connection. Ranked against the same 50 random
  upstream neurons.
- Last prompt position only (exploratory): the same Delta at the position where the answer starts.

## Controls

- Causality control (67.5): zeroing a target must leave every EARLIER target unchanged. If it does not, the hooks or the
  positions are wrong.
- Detection control (67.6): zeroing every unit of layer 6's MLP must change L7 N1169 by more than the largest random
  single-neuron effect. If it does not, the measurement cannot see an effect at all.

## Reading rules (written now)

- If 67.5 or 67.6 fails, nothing else is read until the rig is fixed.
- Six pairs each tested at p = 1/51: under the null, at least one pair reaches 1/51 about 11% of the time. A single pair
  at 1/51 is therefore weak evidence and is reported as such; two or more pairs, or a pair whose |Delta| exceeds the
  largest random value by a factor of 2, is reported as a chain link.
- A chain is claimed only for pairs that pass. Failing pairs are reported as "no measurable link at this sample".
- The weight path is exploratory and cannot by itself establish or refute a link.

## Amendment 1 (2026-10-26 02:45 CT, after the results, before this check runs)

All six pairs beat all 50 random same-layer neurons (67.1 held), but the random neurons were matched by layer only. A
neuron that is nearly silent on these prompts changes nothing when zeroed, so a layer-matched null can be too easy; Muse
named this trap before the run (council muse-literature-2026-10-26 #4). Part 67b repeats the single-neuron test with a
magnitude-matched null: for each upstream target, the 50 units of its layer (target excluded) whose mean absolute
activation over all positions of the 25 unchanged prompts is closest to the target's, ties broken by unit index. Same
prompts, same Delta, same p = (1 + #at-or-above) / 51.

Reading rule, written now: a pair "survives the magnitude-matched null" if its |Delta| exceeds all 50 matched neurons'.
67.1 stays HELD as registered either way; the README states how many of the six pairs survive, and a pair that does
not survive is described as "no larger than equally active neurons in its layer".
