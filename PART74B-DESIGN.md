# Part 74b: is L27 N3325 specific as the route for L24 N3949's effect?

Written 2026-10-26 by Claude, after Part 74's run and its cold review, before any Part 74b code or run.
Predictions 74b.1 and 74b.2 are in PREDICTIONS.md and are pushed before the run.

Size: 24 questions x 23 rows in two forward passes per question (A and B; then E and 20 null E rows) = 552 rows,
48 forward passes, no sampling. Part 74 (600 rows, 60 s on the A770 including model load) is the pilot.

## Why

Part 74 found that setting L27 N3325 alone to its L24-zeroed values (the pure indirect effect, PIE) reproduces 64% of
L24 N3949's effect on the Google-minus-Alibaba gap. The cold review objected that nothing shows L27 N3325 is special:
moving some other layer-27 unit to its L24-zeroed values might do as much.

## Method

- Part 74's model, revision, prompts, opening, readout and maker set, unchanged.
- Per question: pass one runs A (makers zeroed) and B (makers + L24 N3949 zeroed), recording in B the down_proj input
  at every position of L27 N3325 and of each null unit. Pass two runs E (A with L27 N3325 set to its B values, as in
  Part 74) and, for k = 0..19, E_null_k (A with the layer-27 unit of Part 70's matched pair k set to its own B values).
  These are the 20 layer-27 draws Part 73 used (15 distinct units).
- Readout: PIE(u) = mean over questions of g(E for u) - g(A).

## Reading rules (written now)

- 74b.2 is the control: if A, B and E do not reproduce Part 74, 74b.1 is not read.
- If 74b.1 holds, L27 N3325 is specific among activity-matched layer-27 units as a route for L24 N3949's effect.
- If it fails, the route is not specific to L27 N3325 on this readout, and Part 74's reading is narrowed to say so.

## Limits (written now)

- Same forced-opening readout as Part 74, with the same limits.
- 20 draws of 15 distinct units; a null that L24 N3949 does not move gives a PIE near zero for that reason alone, which
  is part of what "specific" means here. The change in each null's activation between A and B is reported.
