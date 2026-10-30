# At low dose, one push interacts more with itself than with a second push of similar size

In OLMo 2 1B Instruct, using 32-bit floating-point numbers, we asked whether changing the model along two weight directions together causes a smaller output interaction than repeating each direction alone, on the same fixed answer-token positions. A push changes model weights along one selected neuron direction or a random direction; dose α sets its size. A real row pairs two selected neurons. A control row pairs the first selected neuron with a random second direction matched in layer and size. We measured centered changes in the model's output token scores over a small token set at each position. The score c is the size of the cross interaction divided by the geometric mean of the two self-term sizes.

At dose 0.1, median c is 0.22395 on real rows. Its product-weighted 95% interval from resampling the 39 first units is 0.16155 to 0.26264. The registered rule gives the headline when this interval is below 0.8 and q at 0.05 is at most 0.30; q measures the higher-order remainder against the two-push self term and is 0.23254 there. An interval wholly between 0.8 and 1.25 would mean the cross interaction is about the same size as the self-term scale. An interval wholly above 1.25 would count as the opposite result. If it crosses a cutoff, the rule reads no headline. The registered check uses q at 0.05 only. At 0.1 q is 0.60069, twice the 0.30 ceiling, so the second-order picture is already failing at the headline dose. This does not change the registered reading because q passes at 0.05, where it is checked.

The reading also covers matched random second pushes. At 0.1, median c is 0.22395 for real rows and 0.23767 for controls, a larger-to-smaller factor of 1.061. The registered real-control comparison stays within a factor of 1.25 at 0.05, 0.1 and 0.2, so the result is about a second push of similar size, not selected neuron pairs alone. At 0.0125, 0.025, 0.05 and 0.1, real-row c is 0.27978, 0.27830, 0.27462 and 0.22395; control c is 0.31977, 0.31780, 0.30670 and 0.23767. Both series dip at 0.1. All three pre-reading gates passed. Their checks and thresholds are described below. This result covers one model, one question and 312 selected rows.

![Dose against c, q, and the norms of I, S_A, and S_B for real and control rows. Both axes are logarithmic.](fig120.png)

The figure is drawn by [fig120.py](fig120.py). It reads the saved vectors, calls the same scoring code in [score120.py](../../part120/score120.py), and checks consistency with [score120.json](score120.json); this is not an independent implementation. The cold reader independently recomputed the tables from the NPZ files and they matched. Its lower interval bound was 0.156, compared with the registered 0.161. Field-level sources and full precision values are in [NUMBERS.md](NUMBERS.md).

## Registered predictions

The prediction text and confidence labels below are copied from the registration in [PART120-DESIGN.md](../../PART120-DESIGN.md). Verdicts use the registered rules.

| # | Confidence | Prediction (verbatim) | Result | Verdict |
| --- | --- | --- | --- | --- |
| 120.1 | medium | In float32 on real rows, the median row ‖I‖ follows a square law at low dose: log2 of its ratio between adjacent doses is within 2 plus or minus 0.3 from 0.025 to 0.05 and from 0.05 to 0.1. | Log2 ratios: 2.06258 and 2.16322. | held |
| 120.2 | medium | The second-order identity holds at low dose and has broken by 0.2: on real rows the median row q is at most 0.30 at 0.05 and at least 0.50 at 0.2. | q: 0.23254 at 0.05; 1.04630 at 0.2. | held |
| 120.3 | medium | A push interacts with itself more than with a second push: at 0.1 the median row c over real rows is at most 0.75, and its product-weighted 95% unit-bootstrap interval lies below 1. | c=0.22395; product-weighted interval 0.16155 to 0.26264. | held |
| 120.4 | medium | At each of 0.05, 0.1 and 0.2, the real median c is within a factor of 1.25 of the control median c. | Larger-to-smaller factors: 1.11681, 1.06126, and 1.03339. | held |
| 120.5 | low | At 0.4 on real rows, the median self share is at least half the median pair share and at most twice it: a push at 0.8 is about as far from twice itself as a pair at 0.4 is from the sum of its parts. | Self share 0.48117; pair share 0.29487. | held |
| 120.6 | low | At 0.4, over the 39 first units, u is higher for slot units than for concept units: the slot median is above the concept median, with a one-sided Mann-Whitney p below 0.05 on Part 92's Haiku kinds, and the same direction on the units both readers agree on. | Haiku medians: slot 0.52761 (21 units), concept 0.41602 (18); one-sided p=0.07341. Agreed-unit medians: slot 0.52761, concept 0.60273. | failed |
| 120.7 | low | Across the 39 units, the Spearman correlation of u(0.4) with Part 119's dominance is at most -0.3. | Spearman rho=-0.08636 over 39 units. | failed |
| 120.8 | low | At 0.1, a real unit's self term is the size of a random direction's on the same path: taking for each first unit the median row ‖S_B‖ on its real rows over the same on its control rows, the median over units lies between 0.8 and 1.25. | Median real-over-control unit ratio for ‖S_B‖ is 1.58978 over 39 units. | failed |

120.5 held on its point estimate: the median self share divided by the median pair share is 1.63, inside the registered 0.5 to 2 band. In an after-the-fact check, the cold reader bootstrapped this ratio over units: 1.63, with a 95% interval of 1.31 to 2.27; 8% of draws fell outside the band. This interval was not registered. For 120.6, 8 of the 18 Haiku-labelled concept units agreed in kind with the second reader, while 10 were disputed. On agreed units the direction is reversed: median u is 0.52761 for 21 slot units and 0.60273 for 8 concept units.

Prediction results and verdicts are in `score120.json:$.predictions[]`, keyed by `id`. The recomputing script lists the exact observed fields in [NUMBERS.md](NUMBERS.md).

## Dose results

Each value is a median over eligible positions within each row, then a median over rows. At each dose from 0.0125 through 0.4, the 2,056 eligible positions are the class total summed over 156 rows, not the count in each row. Each row has 3 to 23 positions. No positions were dropped. Dose 0.8 has no 1.6 pass, so only its raw ‖I‖ is reported.

| Dose | Class | c | q | Median ‖I‖ | Median ‖S_A‖ | Median ‖S_B‖ |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.0125 | real | 0.279775 | 0.056995 | 0.00329854 | 0.0121779 | 0.0111305 |
| 0.0125 | control | 0.319769 | 0.053230 | 0.00313277 | 0.0119144 | 0.00850137 |
| 0.025 | real | 0.278299 | 0.111029 | 0.0134040 | 0.0492694 | 0.0452870 |
| 0.025 | control | 0.317796 | 0.103949 | 0.0126071 | 0.0475004 | 0.0345198 |
| 0.05 | real | 0.274625 | 0.232538 | 0.0559929 | 0.209960 | 0.195765 |
| 0.05 | control | 0.306705 | 0.211675 | 0.0511154 | 0.205294 | 0.145585 |
| 0.1 | real | 0.223947 | 0.600694 | 0.250800 | 1.176657 | 1.072009 |
| 0.1 | control | 0.237667 | 0.560704 | 0.223586 | 1.121481 | 0.692138 |
| 0.2 | real | 0.355826 | 1.046302 | 1.922589 | 5.667950 | 5.161529 |
| 0.2 | control | 0.344330 | 0.953674 | 1.519351 | 5.578891 | 3.947748 |
| 0.4 | real | 0.513269 | 1.585066 | 6.620733 | 11.935712 | 12.662296 |
| 0.4 | control | 0.520980 | 1.447870 | 5.752788 | 11.578487 | 8.481494 |
| 0.8 | real | n/a | n/a | 15.860635 | n/a | n/a |
| 0.8 | control | n/a | n/a | 12.647030 | n/a | n/a |

The second-order reading needs the higher-order remainder E to be small compared with the second-order pair term S_P. q is ‖E‖/‖S_P‖. On real rows q is 0.23254 at 0.05, 0.60069 at 0.1, and 1.04630 at 0.2. It rises above 0.30 by 0.1 and is greater than one by 0.2, where the remainder is larger than S_P.

The 0.1 interval uses 2,000 unit-bootstrap draws with seed 12001. It resamples 39 first units and weights each row by the resampled counts of its first and second endpoints.

| Statistic | Point | 95% interval |
| --- | ---: | ---: |
| Product-weighted median c, registered primary | 0.223947 | 0.161545 to 0.262637 |
| Sum-weighted median c, sensitivity | 0.223947 | 0.199316 to 0.241363 |

## Float32 floor table

The float32 floor is where finite-precision rounding distorts the expected dose scaling at small pushes. Each table ratio is the higher-dose median row norm divided by the lower-dose value; 4 is the square-law value when dose doubles. Prediction 120.1 tests only ‖I‖ from 0.025 to 0.1, and it held. The real ‖S_A‖ ratio is 5.60 from 0.05 to 0.1. The real ‖I‖ ratio is 7.67 from 0.1 to 0.2. The 0.4 to 0.8 ‖S_A‖ ratio is unavailable because the run did not include a 1.6 dose.

| Class | Norm | Dose interval | Higher/lower ratio |
| --- | --- | --- | ---: |
| real | ‖I‖ | 0.0125 to 0.025 | 4.06361 |
| real | ‖I‖ | 0.025 to 0.05 | 4.17732 |
| real | ‖I‖ | 0.05 to 0.1 | 4.47915 |
| real | ‖I‖ | 0.1 to 0.2 | 7.66582 |
| real | ‖I‖ | 0.2 to 0.4 | 3.44365 |
| real | ‖I‖ | 0.4 to 0.8 | 2.39560 |
| real | ‖S_A‖ | 0.0125 to 0.025 | 4.04580 |
| real | ‖S_A‖ | 0.025 to 0.05 | 4.26146 |
| real | ‖S_A‖ | 0.05 to 0.1 | 5.60420 |
| real | ‖S_A‖ | 0.1 to 0.2 | 4.81700 |
| real | ‖S_A‖ | 0.2 to 0.4 | 2.10583 |
| real | ‖S_A‖ | 0.4 to 0.8 | n/a |
| control | ‖I‖ | 0.0125 to 0.025 | 4.02426 |
| control | ‖I‖ | 0.025 to 0.05 | 4.05450 |
| control | ‖I‖ | 0.05 to 0.1 | 4.37414 |
| control | ‖I‖ | 0.1 to 0.2 | 6.79538 |
| control | ‖I‖ | 0.2 to 0.4 | 3.78635 |
| control | ‖I‖ | 0.4 to 0.8 | 2.19842 |
| control | ‖S_A‖ | 0.0125 to 0.025 | 3.98681 |
| control | ‖S_A‖ | 0.025 to 0.05 | 4.32194 |
| control | ‖S_A‖ | 0.05 to 0.1 | 5.46281 |
| control | ‖S_A‖ | 0.1 to 0.2 | 4.97457 |
| control | ‖S_A‖ | 0.2 to 0.4 | 2.07541 |
| control | ‖S_A‖ | 0.4 to 0.8 | n/a |

## Exploratory: interaction cosines

These are medians of row cosines between I and each self term. Every value through dose 0.1 is below 0.05 in absolute value, so the vectors show little alignment there. The cosines were registered as exploratory, not as prediction tests.

| Dose | Class | cos(I,S_A) | cos(I,S_B) |
| ---: | --- | ---: | ---: |
| 0.0125 | real | -0.01212 | -0.00051 |
| 0.0125 | control | 0.04154 | 0.03210 |
| 0.025 | real | -0.01505 | 0.00089 |
| 0.025 | control | 0.02845 | 0.02930 |
| 0.05 | real | -0.00655 | -0.00304 |
| 0.05 | control | 0.02273 | 0.02792 |
| 0.1 | real | 0.02453 | 0.04580 |
| 0.1 | control | 0.04580 | 0.04395 |
| 0.2 | real | 0.20720 | 0.18220 |
| 0.2 | control | 0.20764 | 0.17899 |
| 0.4 | real | 0.38203 | 0.38059 |
| 0.4 | control | 0.34862 | 0.36573 |

## Exploratory: same-layer and different-layer rows

These row subsets were also registered as exploratory. Counts are the rows with finite c in each subset.

| Dose | Class | Same-layer c (rows) | Different-layer c (rows) |
| ---: | --- | ---: | ---: |
| 0.0125 | real | 0.41031 (22) | 0.26222 (134) |
| 0.0125 | control | 0.44432 (22) | 0.29838 (134) |
| 0.025 | real | 0.39766 (22) | 0.26572 (134) |
| 0.025 | control | 0.44651 (22) | 0.29573 (134) |
| 0.05 | real | 0.37563 (22) | 0.25379 (134) |
| 0.05 | control | 0.42837 (22) | 0.27845 (134) |
| 0.1 | real | 0.29990 (22) | 0.21002 (134) |
| 0.1 | control | 0.35367 (22) | 0.22474 (134) |
| 0.2 | real | 0.52141 (22) | 0.32591 (134) |
| 0.2 | control | 0.55097 (22) | 0.32728 (134) |
| 0.4 | real | 0.58632 (22) | 0.50474 (134) |
| 0.4 | control | 0.66804 (22) | 0.51057 (134) |

## Reading

Reading: c is below one at low dose for real and control rows, while q grows from 0.23254 at 0.05 to 1.04630 at 0.2 on real rows.

## 120.8 compares second-push self terms on the same path

120.8 failed. Its median real-over-control unit ratio for ‖S_B‖ at 0.1 is 1.58978, above the registered upper bound of 1.25. In this sample, real second pushes have a larger self term than the matched random directions on these paths. This does not show that every real second push is larger or establish a general effect outside the selected units and paths.

120.8 compares real and control B terms on the same path for each first unit. A selects that path but is not part of this S_B comparison, so A's ownership does not enter the 120.8 statistic.

Path asymmetry does enter c because c divides by the square root of the product of the two self terms. All paths came from the first unit's own answer at dose 0.8, so A is evaluated on its own path and B is not. At dose 0.1, the ratio of median ‖S_B‖ to median ‖S_A‖ is 0.62 for controls and 0.91 for real rows. These data cannot determine the direction of the path effect or separate it from the push-direction effect.

## Method and limits

The experiment used OLMo 2 1B Instruct at revision `48d788eca847d4d7548f375ad03d3c9312f6139e`, float32, on an Intel A770 with batch 8. It asked "Who are you?" in the Part 119b chat frame. It reused 39 first units and 312 rows scored on Path 2 positions selected by Part 119c with seed 11906: four real second pushes and four matched random directions for each first unit. It scored positions from the first unit's first 24 tokens at its saved 0.8 answer. At each position, T is the sorted union of the top-10 token lists under the base model and under the 0.8 A, B and P states. T contains 12 to 40 tokens, with a median of about 23. Logit changes are centered over T, and all reported norms are over this set, not the full vocabulary. No answers were generated at the tested doses.

All three pre-reading gates passed. Gate 1 requires zero-dose hooked logits to equal the plain pass exactly on all 39 paths. Gate 2 requires the run's ‖A‖, ‖B‖, ‖P‖ and ‖I‖ to reproduce Part 119c's stored values at doses 0.05, 0.1, 0.2, 0.4 and 0.8 within 0.5% for each statistic at each dose; the largest observed median relative difference was 0.0000424. Gate 3 requires doubling the dose to double each push with relative error below 1e-5 for every unit and adjacent dose pair. Its observed maximum error was exactly 0. Because doubling these float32 pushes is exact, Gate 3 checks the code path and is not strong evidence about the model.

This is one model, one question, and 312 of 2,964 possible rows. The paths are fixed from each first unit's 0.8 answer, so A sits on its own path and B does not. The self push A is shared by all eight rows for a first unit. Therefore S_A statistics in 120.5, 120.6 and 120.7 represent 39 first pushes per position, not 312 independent pushes. The terms use forward finite differences, not exact derivatives. This part measures how large the terms are on the output, not where in the model they arise. It does not measure generated answers, other models or other questions.

## Terms

- **Dose α:** the scale of a weight push.
- **Push:** a change to model weights along a selected neuron's direction or a random direction matched in layer and size.
- **First unit:** the selected neuron that supplies push A and chooses the row's fixed path. Its A push is shared across its eight rows.
- **Real row:** a pairing of a first unit with a second selected neuron as push B.
- **Control row:** a pairing of a first unit with a random second direction matched in layer and size.
- **Path 2:** a fixed set of answer-token positions from the first unit's saved 0.8 answer. Each row is scored at these positions.
- **T:** at one position, the sorted union of the top-10 token lists from the base model and the 0.8 A, B and P states. Logit changes are centered over T. Norms are over T, not the full vocabulary.
- **A, B, P:** centered changes in output logits from push A alone, push B alone and both pushes together.
- **I:** the cross interaction `P - A - B`.
- **S_A and S_B:** self terms `A(2α) - 2A(α)` and `B(2α) - 2B(α)`, computed as forward finite differences.
- **S_P:** the two-push self term `P(2α) - 2P(α)`, also a forward finite difference.
- **E and q:** E is `S_P - S_A - S_B - 2I`; q is `‖E‖ / ‖S_P‖`, the higher-order remainder relative to the two-push self term.
- **c:** `‖I‖ / sqrt(‖S_A‖ ‖S_B‖)`, the cross term relative to the geometric mean of the self terms.
- **Self share and pair share:** the median squared-size fractions `s=(‖S_A‖/‖A(2α)‖)²` and `(‖I‖/‖P‖)².
- **u:** for a first unit, the median self share s over its eight rows.
- **Row median:** first the median over eligible positions in a row, then the median over rows in a class.
- **Unit bootstrap:** resampling the 39 first units with replacement. The primary interval weights a row by the product of its endpoint counts; the sensitivity interval uses their sum.
- **Slot and concept units:** Part 92 labels a unit slot when its persona appears only on the two identity questions, and concept when it appears in at least three of its six answers.
- **Haiku kinds:** the slot or concept labels that Claude Haiku 5.5 assigned in Part 92.
- **Readers agreeing:** a unit with the same Part 92 Haiku kind and Part 92b DeepSeek kind. Units with different or unclassified DeepSeek kinds are disputed.
- **Dominance:** in Part 119, a unit's token wins divided by its decided pairs. A token win compares which unit's single top token matched the pair answer more often at contested positions.
- **Float32 floor:** low-dose distortion from rounding in 32-bit floating-point arithmetic. The floor table shows adjacent-dose norm ratios; under a square law, doubling dose multiplies a norm by four.
- **Gate:** a pre-reading validation check. A failed gate makes the registered prediction verdicts unread.

## Files

- [NUMBERS.md](NUMBERS.md): recomputed values and source fields.
- [fig120.py](fig120.py) and [fig120.png](fig120.png): recomputation and dose figure.
- [vectors/](vectors/): the 39 per-unit NPZ files with saved A, B, and P vectors.
- [gate1.json](gate1.json), [gate3.json](gate3.json), and [timing.json](timing.json): run gates, scope, and timing.
- [score120.json](score120.json) and [SCORE.md](SCORE.md): recorded score, gates, predictions, and headline.
- [Part 120 design](../../PART120-DESIGN.md) and [build notes](../../part120/BUILD-NOTES.md): registration and run procedure.
- [FOLD-r1.md](../../part120/FOLD-r1.md) and [FOLD-r2.md](../../part120/FOLD-r2.md): code review folds.

Cold check: Haiku 5.5, 2026-10-30, round 1, 15 findings (1 number, 7 overstated, 7 minor), all folded ([COLD-CHECK-r1.md](COLD-CHECK-r1.md), `part120/FOLD-r2.md`).
