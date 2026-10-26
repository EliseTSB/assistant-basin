# Part 70: are the two late "Google" neurons what the makers-zeroed model falls back on?

Written 2026-10-26 04:3x CT by Claude, before any code or run. Predictions 70.1 to 70.6 are in PREDICTIONS.md and are
pushed before the build.

## Why

Part 66: zeroing the four maker neurons (L6 N753, L7 N1169, L11 N8868, L29 N776) makes Qwen3.5-4B name Google in 42 of
192 sampled answers to its 24 held-out maker questions (1 of 192 unchanged). Part 69: zeroing L27 N3325 and L24 N3949
cuts the next-token logit of `Google` by about 4 after a forced opening. If those two neurons hold up the Google
answer, zeroing them together with the makers should take the Google answers away. This tests that on sampled
answers, Part 66's own measure.

## Model, sampling and scoring (Part 66's, unchanged)

`Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, A770; chat template with
`enable_thinking=False`, no system prompt; zeroing at every position, prompt and answer (Part 25's operation, Part 66's
hooks); temperature 0.7, top-p 0.95, up to 48 new tokens; seeds from `part66.row_seed(condition_id, question_index,
sample)`; maker families by Part 66's whole-answer regular expressions; held-out questions and fluency guards as in
Part 66.

## Conditions

| Condition id | Zeroed | Held-out | Positive control | Guards |
|---|---|---|---|---|
| `target` | the four makers (Part 66's id, so Part 66's seeds) | 24 x 8 | 1 x 24 | 6 x 4 |
| `target_plus_pair` | the four makers + L27 N3325 + L24 N3949 | 24 x 8 | 1 x 24 | 6 x 4 |
| `target_plus_matched_k`, k = 0..19 | the four makers + one unit drawn uniformly from each of L27 N3325's and L24 N3949's 50 activity-matched units (Part 68b `results/part68/b/matched.json`, L29 N776's lists), numpy default_rng(7000 + k) | 24 x 1 | none | none |

960 generations in all, under the 1,000 that would call for a pilot first; Part 66 ran 1,880 in 20 minutes.

## Reading rules (written now)

- If 70.1 or 70.2 fails, nothing else is read until the rig is checked.
- 70.3 to 70.5 together are the test; 70.6 says what the model says instead.
