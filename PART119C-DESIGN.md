# Part 119c: is the interaction share at low dose a rounding floor?

Procedure: v1
Registered 2026-10-30 08:52 CT by Claude (Opus 5.5), before any data. Part 119b's gate 3 failed: at alpha 0.2 the median row
interaction share was 0.178 against a limit of 0.15, so by its rule a share that does not fall at a small push could not be
told from a floor in the measure. Elise's direction for the night stands (Discord message 1565603681804488839). The commit
that adds this file is the registration.

## Question

The model runs in bf16, so each activation and logit is rounded to eight significant bits, at every layer. A small push may
change which way those values round, and that noise would look like interaction. Run the same rows in float32. Does the
share at 0.2 stay, and does it keep falling toward zero at smaller doses? Part 119b's reader and triage are not at issue
here: this part only asks what the share at low dose measures.

## Prior art and existing data

- Closest published work: none found by this part. Part 119b's two searches on 2026-10-30 found no test of how two
  single-neuron pushes' interaction scales with dose, and neither looked at numeric precision.
- Our own data: results/part119b (the rows, T and doses reused here); median nI 1.89 and nPair 4.99 at 0.2 for real rows,
  6.37 and 11.63 at 0.4; Part 119's Amendment 2 (greedy answers depend slightly on batch shape on this device).

## Method

Model, revision, device, chat frame, question, units, pushes and Path 2 are Part 119b's (part119b/run119b.py run_path2), on
a subset of its rows: for each of the 39 units as first push, 4 real and 4 random second pushes drawn with seed 11906 from
Part 119b's Path 2 rows. 312 rows. Doses 0.05, 0.1, 0.2, 0.4 and 0.8. The whole set runs twice: the model loaded in bf16 as
in Part 119b, then loaded in float32. Pushes at each dose are built from the weights in the model's own dtype, with Part 92's
functions. T comes from each run's own 0.8 passes, as in Part 119b.

Readout: interaction share as Part 119 defines it, at every position with nPair above 0, row value the median over positions,
summary the median over rows, by dtype, dose and row class; also median nI, nPair, nA and n2, plus the share of positions with
nPair below 1.0. The unit bootstrap of Part 119 (2,000 draws) recomputes the difference of the float32 and bf16 median row
shares at 0.2 in each draw. Product weighting is primary; sum weighting is reported beside it on that same statistic.

## Controls

- Matched random controls: the random second pushes, as in Parts 119 and 119b, at every dose and in both dtypes.
- Do-nothing hook: Part 119b's gate 1 passed on this code and is not repeated.
- Both directions, off-target questions, reader and answers: not used; this part reads logits on fixed paths only.
- Dose ladder: 0.05 to 0.8. The lowest doses are the positive control: a smooth model must give a share near zero there.

Gate, before any prediction is read: (1) the bf16 run's 0.2, 0.4 and 0.8 rows reproduce Part 119b's stored rows: median
relative nPair difference under 1%; (2) the float32 output records must identify parameters loaded as float32, the bf16
output records must identify parameters loaded as bf16, and at 0.8 the median relative nPair difference must be under 5%.
Gate 2 also fails if every matched 0.8 nPair value is identical by position across the two outputs. Each dtype has its own
output file, so a restart after deleting only the float32 file rebuilds its 0.8 token support from float32 passes. A miss on
either gate stops every reading.

## Size

- Smallest effect worth knowing: a difference of 0.04 in the median share at 0.2, about a quarter of Part 119b's value.
- Total: 2 runs of 3,354 forward passes, 6,708 in all, about 5 minutes at Part 119b's 22.7 passes per second, plus two model
  loads. No answers, no reader.

## Optimization

- Bound resource: the A770.
- Measured: Part 119b's Path 2 at batch 8 ran 134 passes in 5.9 s.
- Levers: a subset of 8 second pushes per unit (taken, the interval comes from 39 units, not from rows); batch 8 (taken); both
  dtypes in one job (taken, one launch, a model reload between); all 76 second pushes (declined: 30 more minutes for nothing
  this question reads); float64 (declined: the A770 has none).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 119c.1 | At 0.2, the float32 median row share over real rows is within 0.04 of the bf16 median on the same rows, and the product-weighted 95% unit-bootstrap interval of that same difference lies inside plus or minus 0.06. The sum-weighted interval is reported as a sensitivity check. | medium | Claude |
| 119c.2 | In float32, the median row share over real rows is at most 0.10 at 0.1 and at most 0.05 at 0.05. | medium | Claude |
| 119c.3 | In float32, on real rows, the endpoint log-log slope `log(S(0.2)/S(0.05)) / log(4)` is at least 1.5. | low | Claude |
| 119c.4 | In float32, at each of 0.05, 0.1 and 0.2, the real median share is within a factor of 1.25 of the control median. | low | Claude |

## Reading rules

part119c/score119c.py computes every number. Gates first. If a gate misses, the per-dose table remains visible under a heading
saying no reading is made from it. (1) If 119c.1 and 119c.2 both hold: "Part 119b's low-dose share is not a rounding floor;
the interaction shrinks toward zero with dose, and at 0.2 it is already measurable where answers do not break". (2) Else,
if float32 is below bf16 at 0.2 by more than 0.04 and the product-weighted interval for the difference of the two median row
shares lies below 0: "bf16 rounding inflates the low-dose share, and Part 119b's gate 3 miss is at least partly that". (3)
Otherwise the numbers are reported and nothing is read. Nothing here changes a verdict of Part 119b: its gate 3 failed by its
own rule, and its stopped predictions stay stopped. Answers are not generated here, so nothing is read by eye; the numbers
are checked against Part 119b's rows by gate 1.

## Limits

One question, one model, 312 rows of 2,964, fixed paths chosen at 0.8. float32 is still a rounded format, far finer than bf16.
Nothing here says what the model does at these doses in its own answers. The share is a ratio, so at the smallest doses it
rests on small denominators; median nI, median nPair and the share of positions with nPair below 1.0 are reported beside it.
