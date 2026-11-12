# Part 202 report (made by part202/report202.py)

## Predictions

| # | Verdict | Numbers |
| --- | --- | --- |
| 202.1 | held | R1 True, R2 True, R3_bad_pushed_calls 0, R3_bad_no_push_calls 0 |
| 202.2 | held |  |
| 202.3 | held | qwen_neuron 0.05822193270646879, qwen_random 0.06498883061383061, broken_neuron 0.305625, broken_random 0.111875 |
| 202.4 | held | neuron_at_cap 1011, random_at_cap 3175 |
| 202.5 | failed | differ 103, of 200 |

## Part 164's verdicts, published beside the eager rerun

| Prediction | Bar | Published estimate (95% interval) | Published | Rerun estimate (95% interval) | Rerun |
| --- | --- | --- | --- | --- | --- |
| 164.1 | < 0.7 | 0.789 (0.333 to 1.569) | failed | 0.869 (0.338 to 1.831) | failed |
| 164.2 | >= 2 | 3.456 (2.428 to 5.072) | held | 2.732 (1.981 to 3.861) | held |
| 164.3 | 0.8 < 1.0 < 1.2 | broken by dose 0.8 0.0781, 1.0 0.1969, 1.2 0.3644 | held | broken by dose 0.8 0.1119, 1.0 0.2850, 1.2 0.4781 | held |
| 164.4 | 0.7 to 1.3 | 5.529 (1.510 to 58.383) | failed | 7.441 (1.674 to 105.124) | failed |
| 164.5 | < 0.5 | 0.086 (0.000 to 0.221) | held | 0.119 (0.000 to 0.298) | held |

## Part 164's gates

| Gate | Published | Rerun |
| --- | --- | --- |
| gate1_rows | {"passed": true, "expected_rows": 6516, "rows": 6516, "unique_rows": 6516, "nonempty_answers": 6516, "duplicates": 0} | {"passed": true, "expected_rows": 6516, "rows": 6516, "unique_rows": 6516, "nonempty_answers": 6516, "duplicates": 0} |
| gate2_no_push | {"passed": true, "qwen_named": 16, "alibaba_or_tongyi_named": 16, "total": 16, "minimum_each": 12} | {"passed": true, "qwen_named": 16, "alibaba_or_tongyi_named": 16, "total": 16, "minimum_each": 12} |
| gate3_recipe | {"passed": true, "changed": 86, "total": 100, "changed_rate": 0.86, "minimum_rate": 0.7} | {"passed": true, "changed": 71, "total": 100, "changed_rate": 0.71, "minimum_rate": 0.7} |
| gate4_measure_room | {"passed": true, "random_qwen_rate_0.8": 0.07944900932400932, "minimum_rate": 0.04} | {"passed": true, "random_qwen_rate_0.8": 0.06498883061383061, "minimum_rate": 0.04} |

## Rates per cell

| Cell | Qwen published | Qwen rerun | Broken published | Broken rerun |
| --- | --- | --- | --- | --- |
| neuron|0.8 | 0.0639 | 0.0582 | 0.2700 | 0.3056 |
| random|0.8 | 0.0794 | 0.0650 | 0.0781 | 0.1119 |
| random|1.0 | 0.0113 | 0.0076 | 0.1969 | 0.2850 |
| random|1.2 | 0.0068 | 0.0077 | 0.3644 | 0.4781 |

## Lengths (battery rows)

| Run | Arm | Rows | At the 80-token cap | Mean token_count |
| --- | --- | --- | --- | --- |
| published | neuron | 1600 | 108 | 53.259 |
| published | random | 4800 | 358 | 55.219 |
| rerun | neuron | 1600 | 1011 | 69.266 |
| rerun | random | 4800 | 3175 | 71.261 |

## Shape replicate

Differ: 103 of 200 (bar 40 to 100).

Battery row's margin at the first differing position, in bf16 steps:

- 0 (exact tie): 45
- over 0, up to 1: 49
- over 1, up to 2: 5
- over 2: 4
- no recorded margin: 0

Label changes over the 200 pairs (score202.floor_labels): broken 17, excluded 5, maker_gone 1, qwen 3

## Stability under the shape floor (share of 1,000 redraws keeping the verdict)

| Verdict | independent | shared-change-0.5 | shared-change-1 | upper-marginal-independent |
| --- | --- | --- | --- | --- |
| 164.1 | 0.268 | 0.264 | 0.276 | 0.632 |
| 164.2 | 0.738 | 0.726 | 0.719 | 0.000 |
| 164.3 | 1.000 | 1.000 | 1.000 | 1.000 |
| 164.4 | 1.000 | 1.000 | 1.000 | 0.389 |
| 164.5 | 1.000 | 1.000 | 1.000 | 0.102 |
| gate1_rows | 1.000 | 1.000 | 1.000 | 1.000 |
| gate2_no_push | 1.000 | 1.000 | 1.000 | 1.000 |
| gate3_recipe | 1.000 | 1.000 | 1.000 | 1.000 |
| gate4_measure_room | 1.000 | 1.000 | 1.000 | 0.730 |
