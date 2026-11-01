# Part 164 score

## Gates

| Gate | Result | Detail |
| --- | --- | --- |
| gate1_rows | passed | `{"expected_rows": 6516, "rows": 6516, "unique_rows": 6516, "nonempty_answers": 6516, "duplicates": 0, "units_per_arm": {"neuron": 100, "random": 100}}` |
| gate2_no_push | passed | `{"qwen_named": 16, "alibaba_or_tongyi_named": 16, "total": 16, "minimum_each": 12}` |
| gate3_recipe | passed | `{"changed": 86, "total": 100, "changed_rate": 0.86, "minimum_rate": 0.7}` |
| gate4_measure_room | passed | `{"random_qwen_rate_0.8": 0.07944900932400932, "minimum_rate": 0.04}` |

## Predictions

| Prediction | Verdict | Point estimate | 95% bootstrap interval |
| --- | --- | ---: | --- |
| 164.1 | failed | 0.7886820388718295 | (0.333, 1.569) |
| 164.2 | held | 3.4560000000000004 | (2.428, 5.072) |
| 164.3 | held | n/a | n/a |
| 164.4 | failed | 5.528824117059411 | (1.510, 58.383) |
| 164.5 | held | 0.08570936085248311 | (0.000, 0.221) |

## Arm means

| Metric | Dose | Arm | Mean | Units included | Units excluded |
| --- | ---: | --- | ---: | ---: | ---: |
| qwen_rate | 0.8 | neuron | 0.06393878230612925 | 98 | 2 |
| qwen_rate | 0.8 | random | 0.07944900932400932 | 100 | 0 |
| qwen_rate | 1.0 | random | 0.011333333333333332 | 100 | 0 |
| qwen_rate | 1.2 | random | 0.00680952380952381 | 100 | 0 |
| broken_rate | 0.8 | neuron | 0.27 | 100 | 0 |
| broken_rate | 0.8 | random | 0.078125 | 100 | 0 |
| broken_rate | 1.0 | random | 0.196875 | 100 | 0 |
| broken_rate | 1.2 | random | 0.364375 | 100 | 0 |

Hand-read sample rows are in `hand_read.jsonl`.
