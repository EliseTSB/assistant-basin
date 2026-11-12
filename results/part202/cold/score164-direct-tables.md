# Part 164 score

## Gates

| Gate | Result | Detail |
| --- | --- | --- |
| gate1_rows | passed | `{"expected_rows": 6516, "rows": 6516, "unique_rows": 6516, "nonempty_answers": 6516, "duplicates": 0, "units_per_arm": {"neuron": 100, "random": 100}}` |
| gate2_no_push | passed | `{"qwen_named": 16, "alibaba_or_tongyi_named": 16, "total": 16, "minimum_each": 12}` |
| gate3_recipe | passed | `{"changed": 71, "total": 100, "changed_rate": 0.71, "minimum_rate": 0.7}` |
| gate4_measure_room | passed | `{"random_qwen_rate_0.8": 0.06498883061383061, "minimum_rate": 0.04}` |

## Predictions

| Prediction | Verdict | Point estimate | 95% bootstrap interval |
| --- | --- | ---: | --- |
| 164.1 | failed | 0.8689997064396466 | (0.338, 1.831) |
| 164.2 | held | 2.7318435754189943 | (1.981, 3.861) |
| 164.3 | held | n/a | n/a |
| 164.4 | failed | 7.441447963800905 | (1.674, 105.124) |
| 164.5 | held | 0.11890155053241173 | (0.000, 0.298) |

## Arm means

| Metric | Dose | Arm | Mean | Units included | Units excluded |
| --- | ---: | --- | ---: | ---: | ---: |
| qwen_rate | 0.8 | neuron | 0.05822193270646879 | 97 | 3 |
| qwen_rate | 0.8 | random | 0.06498883061383061 | 100 | 0 |
| qwen_rate | 1.0 | random | 0.007589285714285714 | 100 | 0 |
| qwen_rate | 1.2 | random | 0.007727272727272727 | 100 | 0 |
| broken_rate | 0.8 | neuron | 0.305625 | 100 | 0 |
| broken_rate | 0.8 | random | 0.111875 | 100 | 0 |
| broken_rate | 1.0 | random | 0.285 | 100 | 0 |
| broken_rate | 1.2 | random | 0.478125 | 100 | 0 |

Hand-read sample rows are in `hand_read.jsonl`.
