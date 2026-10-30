# Part 114: none of the 20 attribution-ranked neurons, alone or as the top pair, takes the leading "Q" off Qwen3.5-4B's name

Asked who it is, with the answer started as "I am **", Qwen3.5-4B puts "Q" (the first token of "Qwen") first on all 16
held-out phrasings, at a mean log-odds of 6.74 (99.9% at the mean log-odds; the mean probability over the 16 is 99.6%).
Zeroing any one of the 20 candidates (the neurons a gradient attribution ranked strongest) never moved "Q" off first
place: 0 of 16 phrasings for each. Zeroing the top two by attribution together (L31 N4742 and L10 N54) also gave 0 of
16, and every continuation still began "Qwen3.5**". Exploratory, not registered: 16 of the 20 lowered the mean
log-odds of "Q" over the 16 phrasings more than all 20 of their activity-matched controls (same-layer neurons nearest
in mean |activation|), and the largest mean drop was 1.90 (L28 N2283).

Elise's bet, 2026-10-29 18:13 CT: "I bet we can find a single neuron, maybe two, highly responsible for the the
leading Q it really likes in identity questions."

## Predictions

| # | Prediction (verbatim) | Result | Verdict |
| --- | --- | --- | --- |
| 114.1 | Elise (her bet): One neuron, zeroed alone, makes "Q" stop being the most likely token on at least 8 of the 16 held-out phrasings. | the most for any candidate: 0 of 16 | **failed** (20 candidates tested) |
| 114.2 | Claude (medium confidence): No single neuron does that on more than 4 of the 16. | 0 of 16 for all 20 | **held** (20 candidates tested) |
| 114.3 | Claude (medium confidence): The top candidate lowers the mean held-out log-odds of "Q" more than all 20 of its activity-matched controls. | L31 N4742: -0.73; its strongest control lowers it by 0.05 | **held** |
| 114.4 | Claude (low confidence): The top two zeroed together make "Q" stop being the most likely token on at least 8 of 16. | 0 of 16 | **failed** |

## The seven largest effects

Each candidate was zeroed at every position on 16 phrasings that the attribution never saw (listed in
`part114/qneuron114.py`), beside the 20 neurons of its layer nearest it in mean |activation|. The zeroing is in
`part114/qneuron114.py`; the counts come from `part114/score114.py` (`results/part114/score.json`).

| Unit | Attribution | Mean change in log-odds of "Q" | Controls lowering it as much | Phrasings where "Q" lost first place |
| --- | --- | --- | --- | --- |
| L28 N2283 | -0.264 | -1.90 | 0 of 20 | 0 of 16 |
| L27 N6233 | -0.617 | -0.95 | 0 of 20 | 0 of 16 |
| L31 N4742 | -0.899 | -0.73 | 0 of 20 | 0 of 16 |
| L10 N54 | -0.652 | -0.56 | 0 of 20 | 0 of 16 |
| L30 N3852 | -0.314 | -0.49 | 0 of 20 | 0 of 16 |
| L28 N1268 | -0.482 | -0.46 | 0 of 20 | 0 of 16 |
| L29 N1823 | -0.426 | -0.46 | 0 of 20 | 0 of 16 |

The other 13 candidates lowered it by 0.14 to 0.40; four of them (L10 N4133, L26 N1, L28 N4072, L25 N2771) had 2 to 4
controls that lowered it as much. No control flipped any phrasing either. The seven shown lower the mean log-odds by
0.46 to 1.90 nats; whether any of them is specific to "Q" was not tested. Six of the 400 control rows are themselves
candidates (five neurons), which makes the controls harder to beat.

## Did the attribution pick the right neurons?

Roughly: its ranking agreed with the exact zeroing effects at Spearman 0.77 over the 20 (exploratory, not registered).
Its estimates were off in both directions; L28 N2283 was ranked 11th and had the largest exact effect.

## What would it take?

Not tested here. The 20 single-neuron drops add up to 8.42 nats. If their effects added, zeroing all 20 together would
bring the mean log-odds of "Q" to about -1.7 (a probability of about 0.16); whether that takes "Q" off first place
depends on the runner-up token, which was not recorded. Zeroed attention heads were smaller together than their sum
in Part 83. The one pair measured here was close to additive: against the unbatched baseline of 6.70 (0.04 below the
batched 6.74), zeroing L31 N4742 alone lowered the mean log-odds by 0.69 and the pair by 1.16, against 1.29 for the two
single drops summed (`pair_generations` in `results/part114/results114.json`). Part 115 asks how many it takes.

## Method

Qwen/Qwen3.5-4B at revision 851bf6e8, bf16, on the A770; chat template, no system prompt, thinking off; the answer
forced to begin "I am **"; the readout is the log-odds of the single token "Q" (id 48) at the next position.
Attribution on 8 selection phrasings: minus activation times gradient of that log-odds, summed over prompt positions,
for every MLP neuron in all 32 layers; the 20 most negative are the candidates. The full design is
`PART114-DESIGN.md`; the runner is `part114/qneuron114.py`. Run 2026-10-29 18:51 to 18:59 CT (476 s).

## Limits

One forced opening and one next-token readout; the 12-token continuations are greedy. A first-order attribution can
miss units that only act jointly, so no single neuron found covers these 20, not every neuron. The held-out set is 16
phrasings.

## Files

- `results/part114/results114.json`: every zeroing row (candidates, controls, the unzeroed row) and the pair's
  continuations; made by `part114/qneuron114.py`.
- `results/part114/score.json`: the per-candidate counts and the verdicts; made by `part114/score114.py`.
- `results/part114/attr114.pt`: the attribution for every MLP neuron and the mean |activation| used to pick the
  controls; made by `part114/qneuron114.py` (copied unchanged from the run's output folder).
- `results/part114/launch.jsonl`: the launch record (`tools/launch_arc.py`).

Cold check: Haiku 5.5, 2026-10-29, 13 findings (title and verdict scope, an unregistered count tagged exploratory, the "margin" reading removed, two roundings, the pair's measured log-odds added, provenance), all folded.
