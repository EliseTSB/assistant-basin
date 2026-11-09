# Part 163 score

## Gates

| Gate | Verdict | Detail |
| --- | --- | --- |
| gate1_recipe | passed | `{"minimum_rate": 0.7, "changed_rate_by_arm": {"neuron": 0.75, "random": 0.71}, "n_by_arm": {"neuron": 100, "random": 100}}` |
| gate2_no_push | passed | `{"identity_qwen_hits": 8, "identity_total": 8, "maker_alibaba_or_tongyi_hits": 8, "maker_total": 8, "minimum_each": 7}` |
| gate3_rows | passed | `{"expected_rows": 3416, "rows": 3416, "unique_rows": 3416, "nonempty": 3416, "duplicates": 0, "units_per_arm": {"neuron": 100, "random": 100}}` |

## Predictions

| Prediction | Verdict | Summary |
| --- | --- | --- |
| 163.1 | held | On identity phrasings, the neuron arm's name-gone rate is under 1.5 times the random arm's |
| 163.2 | held | On maker phrasings, the neuron arm's maker-gone rate is under 1.5 times the random arm's |
| 163.3 | held | Each arm's name-gone rate on identity phrasings is at least 0.15 |
| 163.4 | held | In each arm fewer units reach both gone on at least half of identity phrasings than reach name gone on at least half |
| 163.5 | failed | The neuron arm's broken rate is below the random arm's |

## Primary rates

| Metric | Neuron mean (95% interval) | Random mean (95% interval) |
| --- | --- | --- |
| name_gone_identity | 0.853 (0.805, 0.897) | 0.815 (0.765, 0.862) |
| maker_gone_maker | 0.920 (0.877, 0.958) | 0.884 (0.841, 0.923) |
| both_gone_identity | 0.843 (0.792, 0.889) | 0.792 (0.740, 0.841) |
| name_gone_maker | 0.918 (0.877, 0.954) | 0.881 (0.836, 0.921) |
| maker_gone_identity | 0.966 (0.936, 0.989) | 0.950 (0.925, 0.972) |
| broken_rate | 0.242 (0.193, 0.294) | 0.064 (0.048, 0.083) |

## Paired ratios

| Contrast | Ratio | 95% interval | Pairs |
| --- | ---: | --- | ---: |
| 163.1_name_gone_identity | 1.047 | (0.973, 1.128) | 100 |
| 163.2_maker_gone_maker | 1.042 | (0.990, 1.101) | 99 |

Exploratory summaries and layer rates are in `score.json`. Hand-read answers are in `hand_read.jsonl`.
