# Part 163 score

## Gates

| Gate | Verdict | Detail |
| --- | --- | --- |
| gate1_recipe | passed | `{"minimum_rate": 0.7, "changed_rate_by_arm": {"neuron": 0.88, "random": 0.88}, "n_by_arm": {"neuron": 100, "random": 100}}` |
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
| name_gone_identity | 0.850 (0.803, 0.893) | 0.801 (0.749, 0.850) |
| maker_gone_maker | 0.912 (0.864, 0.953) | 0.893 (0.854, 0.929) |
| both_gone_identity | 0.838 (0.788, 0.883) | 0.778 (0.724, 0.830) |
| name_gone_maker | 0.899 (0.855, 0.939) | 0.840 (0.791, 0.886) |
| maker_gone_identity | 0.961 (0.930, 0.985) | 0.953 (0.930, 0.973) |
| broken_rate | 0.214 (0.168, 0.266) | 0.052 (0.037, 0.069) |

## Paired ratios

| Contrast | Ratio | 95% interval | Pairs |
| --- | ---: | --- | ---: |
| 163.1_name_gone_identity | 1.061 | (0.985, 1.148) | 99 |
| 163.2_maker_gone_maker | 1.024 | (0.970, 1.077) | 98 |

Exploratory summaries and layer rates are in `score.json`. Hand-read answers are in `hand_read.jsonl`.
