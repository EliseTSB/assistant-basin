# Part 127 score

Run: 1914 generated answers; 27 pairs; batch 8.
Headline withdrawn: no.

## Registered predictions

| Prediction | Observation | Verdict |
| --- | --- | --- |
| 127.1 At dose .5, at most 1 of the 4 same-layer pairs is rescued. | `{"readable_cells": 4, "same_layer_eligible_pairs": 2, "same_layer_rescued_pairs_dose_0.5": 1, "same_layer_rescued_pairs_dose_0.707": 1}` | held |
| 127.2 At dose .707, at least 2 of the 4 same-layer pairs are rescued. | `{"readable_cells": 4, "same_layer_eligible_pairs": 2, "same_layer_rescued_pairs_dose_0.5": 1, "same_layer_rescued_pairs_dose_0.707": 1}` | failed |
| 127.3 Across the 54 cells at dose .5, the A+B rescue count exceeds the A+C count by at most 3 cells. | `{"eligible_pairs": 19, "pair_ab_minus_matched_ac_cells_dose_0.5": 3, "readable_cells": 38}` | held |
| 127.4 All four same-layer pairs have an absolute W_down cosine below .2. | `{"same_layer_absolute_cosines_below_0.2": true, "same_layer_pair_count": 4}` | held |
| 127.5 At dose .5, A's reference-path logit change under A+B exceeds that under A+C in at least 17 of the 27 pairs. | `{"pairs": 27, "pairs_ab_exceeds_ac": 13}` | failed |

## Reader gate and agreement

| Reader | Known examples |
| --- | ---: |
| DeepSeek V4.1 Flash | 12/12 (pass) |
| Luna | 12/12 (pass) |

Compound-flag agreement: 0.9045643153526971 raw over 241 shared of 1938 items; 1697 unread; kappa 0.7654550839927221; pass.

## Rescue counts

A rescue is a coherent DeepSeek family mark on a joint arm when both half-dose single answers lack it. Only pairs that passed both readers' full-dose positive controls enter these counts.

| Dose | Arm | Rescued cells | Rescued pairs | Included cells | Excluded unread | Eligible pairs |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5 | pair_ab | 3 | 2 | 38 | 0 | 19 |
| 0.5 | matched_ac | 0 | 0 | 38 | 0 | 19 |
| 0.5 | random_0 | 0 | 0 | 38 | 0 | 19 |
| 0.5 | random_1 | 0 | 0 | 38 | 0 | 19 |
| 0.5 | random_2 | 0 | 0 | 38 | 0 | 19 |
| 0.707 | pair_ab | 8 | 8 | 38 | 0 | 19 |
| 0.707 | matched_ac | 1 | 1 | 38 | 0 | 19 |
| 0.707 | random_0 | 2 | 2 | 38 | 0 | 19 |
| 0.707 | random_1 | 2 | 2 | 38 | 0 | 19 |
| 0.707 | random_2 | 3 | 3 | 38 | 0 | 19 |

## Pair bootstrap intervals

Pairs are the resampling unit. Intervals show mean rescued cells per pair, or the pair-level difference for A+B versus each control.

| Dose | Measure | Pairs | Mean | 95% interval |
| ---: | --- | ---: | ---: | ---: |
| 0.5 | pair_ab | 19 | 0.1579 | [0.0000, 0.4211] |
| 0.5 | matched_ac | 19 | 0.0000 | [0.0000, 0.0000] |
| 0.5 | random_0 | 19 | 0.0000 | [0.0000, 0.0000] |
| 0.5 | random_1 | 19 | 0.0000 | [0.0000, 0.0000] |
| 0.5 | random_2 | 19 | 0.0000 | [0.0000, 0.0000] |
| 0.5 | pair_ab_minus_matched_ac | 19 | 0.1579 | [0.0000, 0.4211] |
| 0.5 | pair_ab_minus_random_0 | 19 | 0.1579 | [0.0000, 0.4211] |
| 0.5 | pair_ab_minus_random_1 | 19 | 0.1579 | [0.0000, 0.4211] |
| 0.5 | pair_ab_minus_random_2 | 19 | 0.1579 | [0.0000, 0.4211] |
| 0.707 | pair_ab | 19 | 0.4211 | [0.2105, 0.6842] |
| 0.707 | matched_ac | 19 | 0.0526 | [0.0000, 0.1579] |
| 0.707 | random_0 | 19 | 0.1053 | [0.0000, 0.2632] |
| 0.707 | random_1 | 19 | 0.1053 | [0.0000, 0.2632] |
| 0.707 | random_2 | 19 | 0.1579 | [0.0000, 0.3158] |
| 0.707 | pair_ab_minus_matched_ac | 19 | 0.3684 | [0.1053, 0.6316] |
| 0.707 | pair_ab_minus_random_0 | 19 | 0.3158 | [0.0526, 0.5789] |
| 0.707 | pair_ab_minus_random_1 | 19 | 0.3158 | [0.0526, 0.5789] |
| 0.707 | pair_ab_minus_random_2 | 19 | 0.2632 | [0.0000, 0.5789] |

## Pair cosine table

Cross-layer cosines are descriptive. Same-layer pairs are the direct geometric checks.

| Pair | Family | Layers | Same layer | Cosine |
| --- | --- | --- | --- | ---: |
| p92-08+p92-09 | dog | L4/L6 | no | -0.01386 |
| p92-08+p92-37 | dog | L4/L5 | no | -0.00904 |
| p92-09+p92-37 | dog | L6/L5 | no | 0.01505 |
| p92-18+p92-20 | shadow | L3/L2 | no | 0.05198 |
| p92-23+p92-24 | king_or_ruler | L3/L4 | no | 0.01040 |
| p92-23+p92-25 | king_or_ruler | L3/L3 | yes | -0.03042 |
| p92-23+p92-26 | king_or_ruler | L3/L5 | no | -0.01510 |
| p92-23+p92-27 | king_or_ruler | L3/L3 | yes | -0.01337 |
| p92-23+p92-42 | king_or_ruler | L3/L2 | no | 0.02026 |
| p92-23+p92-45 | king_or_ruler | L3/L4 | no | -0.04747 |
| p92-24+p92-25 | king_or_ruler | L4/L3 | no | 0.01426 |
| p92-24+p92-26 | king_or_ruler | L4/L5 | no | 0.00583 |
| p92-24+p92-27 | king_or_ruler | L4/L3 | no | -0.02285 |
| p92-24+p92-42 | king_or_ruler | L4/L2 | no | -0.01139 |
| p92-24+p92-45 | king_or_ruler | L4/L4 | yes | -0.04788 |
| p92-25+p92-26 | king_or_ruler | L3/L5 | no | 0.00422 |
| p92-25+p92-27 | king_or_ruler | L3/L3 | yes | 0.07161 |
| p92-25+p92-42 | king_or_ruler | L3/L2 | no | 0.01995 |
| p92-25+p92-45 | king_or_ruler | L3/L4 | no | -0.02857 |
| p92-26+p92-27 | king_or_ruler | L5/L3 | no | -0.01551 |
| p92-26+p92-42 | king_or_ruler | L5/L2 | no | -0.02263 |
| p92-26+p92-45 | king_or_ruler | L5/L4 | no | 0.00614 |
| p92-27+p92-42 | king_or_ruler | L3/L2 | no | 0.02493 |
| p92-27+p92-45 | king_or_ruler | L3/L4 | no | 0.01265 |
| p92-42+p92-45 | king_or_ruler | L2/L4 | no | -0.03005 |
| p92-34+p92-35 | angel | L11/L4 | no | -0.02858 |
| p92-28+p92-29 | deity | L10/L2 | no | 0.00801 |

## Fixed-path logit measure

The values are mean centered emitted-token logit changes against the no-push path over six prompts and up to 24 positions. Raw changes are used only when an emitted token falls outside the fixed support.

| Pair | Dose | A: A+B | A: A+C | B: A+B |
| --- | ---: | ---: | ---: | ---: |
| p92-08+p92-09 | 0.5 | 0.24944 | -0.33585 | 0.26558 |
| p92-08+p92-09 | 0.707 | -0.05378 | -0.62312 | 0.27103 |
| p92-08+p92-37 | 0.5 | 0.74094 | 0.44704 | 0.32555 |
| p92-08+p92-37 | 0.707 | 0.64910 | 0.09962 | -0.48449 |
| p92-09+p92-37 | 0.5 | 0.13237 | 0.50886 | 0.21888 |
| p92-09+p92-37 | 0.707 | 0.33159 | 0.65190 | -0.18905 |
| p92-18+p92-20 | 0.5 | 0.06039 | -0.13324 | -0.01354 |
| p92-18+p92-20 | 0.707 | 0.08917 | -0.41662 | -0.02113 |
| p92-23+p92-24 | 0.5 | 1.20249 | 0.58750 | 1.60521 |
| p92-23+p92-24 | 0.707 | -0.39617 | 0.06415 | 1.45822 |
| p92-23+p92-25 | 0.5 | 0.34113 | 0.76048 | 0.74486 |
| p92-23+p92-25 | 0.707 | 0.20463 | 0.93710 | 0.76968 |
| p92-23+p92-26 | 0.5 | 0.07246 | 0.29021 | 0.17402 |
| p92-23+p92-26 | 0.707 | 0.11792 | 0.00755 | 0.17944 |
| p92-23+p92-27 | 0.5 | 0.46561 | 0.76963 | 0.23777 |
| p92-23+p92-27 | 0.707 | 0.20216 | 1.00821 | 0.21677 |
| p92-23+p92-42 | 0.5 | 0.05157 | 0.60113 | -0.06190 |
| p92-23+p92-42 | 0.707 | -0.17270 | 0.15486 | 0.20830 |
| p92-23+p92-45 | 0.5 | 0.39853 | 1.15277 | 0.55428 |
| p92-23+p92-45 | 0.707 | 0.05406 | 0.14130 | 0.79048 |
| p92-24+p92-25 | 0.5 | 0.74839 | 1.78226 | 1.26415 |
| p92-24+p92-25 | 0.707 | 0.01878 | 1.54888 | 0.73089 |
| p92-24+p92-26 | 0.5 | 1.19910 | 0.78045 | 0.39029 |
| p92-24+p92-26 | 0.707 | 0.75473 | 0.19551 | -0.43641 |
| p92-24+p92-27 | 0.5 | 1.26482 | 1.36606 | 1.24684 |
| p92-24+p92-27 | 0.707 | 0.93329 | 1.31328 | -0.07726 |
| p92-24+p92-42 | 0.5 | 1.14379 | 1.40669 | 0.47791 |
| p92-24+p92-42 | 0.707 | -0.34917 | 0.38982 | -0.53073 |
| p92-24+p92-45 | 0.5 | 0.93717 | 1.44403 | 1.66449 |
| p92-24+p92-45 | 0.707 | 0.73255 | 0.95215 | 1.06508 |
| p92-25+p92-26 | 0.5 | 0.73516 | 0.81693 | -0.19495 |
| p92-25+p92-26 | 0.707 | 0.52070 | 0.49095 | -0.73448 |
| p92-25+p92-27 | 0.5 | 1.58360 | 0.94828 | 0.75761 |
| p92-25+p92-27 | 0.707 | 1.20293 | 1.00819 | 0.10827 |
| p92-25+p92-42 | 0.5 | 0.69445 | 0.49872 | 0.30023 |
| p92-25+p92-42 | 0.707 | 0.51189 | 0.37143 | 0.17502 |
| p92-25+p92-45 | 0.5 | 0.67248 | 0.60211 | 0.44368 |
| p92-25+p92-45 | 0.707 | 0.82271 | 0.75049 | 0.25220 |
| p92-26+p92-27 | 0.5 | 0.35242 | -0.13170 | 0.32721 |
| p92-26+p92-27 | 0.707 | 0.24417 | -0.10002 | 0.64438 |
| p92-26+p92-42 | 0.5 | -0.19477 | -0.02803 | 0.14240 |
| p92-26+p92-42 | 0.707 | -0.73798 | -0.58587 | 0.16298 |
| p92-26+p92-45 | 0.5 | -0.02709 | -0.19798 | 0.50053 |
| p92-26+p92-45 | 0.707 | -0.45238 | -0.72191 | 0.74760 |
| p92-27+p92-42 | 0.5 | 0.48651 | 0.10731 | 0.56480 |
| p92-27+p92-42 | 0.707 | 0.02586 | -0.12799 | 0.65521 |
| p92-27+p92-45 | 0.5 | 0.23154 | 0.33929 | 0.58881 |
| p92-27+p92-45 | 0.707 | 0.05268 | 0.13870 | 0.88140 |
| p92-42+p92-45 | 0.5 | 0.26899 | -0.09473 | 0.25609 |
| p92-42+p92-45 | 0.707 | -0.20989 | -0.67181 | 0.04035 |
| p92-34+p92-35 | 0.5 | -0.76925 | -1.03828 | 0.27156 |
| p92-34+p92-35 | 0.707 | -1.76921 | -2.02253 | 0.03957 |
| p92-28+p92-29 | 0.5 | 0.48385 | 0.64460 | 0.65627 |
| p92-28+p92-29 | 0.707 | -0.09358 | 0.05315 | 0.19898 |

## Broken answers

Unread labels are reported and excluded from rescue denominators. Empty outputs and incoherent outputs count as broken.

| Reader | Arm | Items | Broken | Unread labels | Retries | Truncated items | Truncated attempts | Empty | Incoherent |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek V4.1 Flash | single | 288 | 57 | 0 | 1 | 0 | 0 | 2 | 56 |
| DeepSeek V4.1 Flash | pair_ab | 324 | 111 | 0 | 5 | 0 | 0 | 0 | 111 |
| DeepSeek V4.1 Flash | matched_ac | 324 | 90 | 0 | 3 | 0 | 0 | 1 | 90 |
| DeepSeek V4.1 Flash | random | 972 | 243 | 0 | 7 | 0 | 0 | 1 | 243 |
| DeepSeek V4.1 Flash | no_push | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Luna | single | 103 | 18 | 0 | 0 | 0 | 0 | 2 | 18 |
| Luna | pair_ab | 41 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| Luna | random | 53 | 4 | 0 | 0 | 0 | 0 | 0 | 4 |
| Luna | no_push | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Luna | matched_ac | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Pair and question rows

The JSON file contains all 162 pair by question records, including both readers' raw flags for every arm and dose.

## Reader-specific negative controls

| Reader | False-positive rate | No-push marks | Unread | Threshold | Result |
| --- | ---: | ---: | ---: | --- |
| DeepSeek V4.1 Flash | 0.0000 | 0 | 0 | 0 | pass |
| Luna | 0.0000 | 0 | 0 | 0 | pass |
