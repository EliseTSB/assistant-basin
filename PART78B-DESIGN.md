# Part 78b: is the Google rise in Part 78's control a property of N8870, or of any layer-22 push of that size?

Written 2026-10-27 by Claude, after Part 78 was scored and before any Part 78b code or run. Predictions 78b.1 to 78b.4
are in PREDICTIONS.md and are pushed before the run.

## Why

In Part 78, with the four maker neurons zeroed, pushing the alien unit L22 N2742 cut Google answers from 37 to 6 (1.5x)
and 0 (2.0x), while pushing one size-matched control unit, L22 N8870, raised them to 119 of 192. Part 78's own write-up
says that one control unit cannot show the alien unit is specific. Part 62's data points the same way at a smaller
scale: N8870 at 2.0x, with the makers intact, gave Google in 11 of 24 answers. If most layer-22 pushes of this size raise
the Google fallback, the alien unit's suppression is the unusual result; if N8870 is the odd one, the Part 78 control
result is about N8870.

## Method

- Part 78's runner unchanged in everything but the conditions: `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16, A770, Part 66's 24 held-out questions times 8 samples and 6 guards
  times 4, its rendering, seeds and sampling, at most 64 new tokens, Part 78's keyword scorer.
- Control units: `neuron_report.control_units(9216, 2742, 10, 78078)` = L22 units 1499, 4154, 5572, 5606, 5640, 6210,
  8115, 8606, 8920, 9209 (drawn 2026-10-27 before any run; N8870 is not among them).
- Each control unit u is pushed as Part 78 pushed N8870 (Amendment 1 there): set to 2.0 times 19.125 times
  ||W_down[:, 2742]|| / ||W_down[:, u]||, at every prompt and answer position, with the four makers zeroed.
- Conditions: `k1_makers` (makers zeroed, no push; a rerun of Part 78's c1 under new condition ids, so new seeds) and
  `k_u<unit>` for each of the 10 units. 11 conditions times 216 answers = 2,376 answers.

## Reading rule

78b.1 is the control. If it fails, 78b.2 to 78b.4 are reported as NOT READ. Counts are held-out Google answers of 192
by Part 78's keyword rule. Part 78's own c1 (37) and c5 (0) are quoted for comparison but not re-run.

## Limits known in advance

Keyword scoring as in Part 78. Ten units from one layer at one dose. A push sized by norm is not sized by behaviour, so a
unit whose push changes little is a weak control; the per-unit off-world and guard counts are reported so that shows.
