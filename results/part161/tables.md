# Part 161 score

## Gates

- gate1_rows: passed ({"passed": true, "expected": 11904, "present": 11904, "missing": 0, "duplicates": 0, "nonempty": 11904})
- gate2_controls: passed ({"passed": true, "L24N5017_Google_hits_at_0_8": 19, "required": [15, 22], "no_push_any_picked_company_hits": 0, "maximum": 2})
- gate3_plan: passed ({"passed": true, "distinct_random_directions": 45, "required_random": 45, "new_neurons": 17, "required_new_neurons": 17, "all_neuron_picks_by_dose": {"0.6": 30, "1.0": 7}, "required_picks_by_dose": {"0.6": 30, "1.0": 7}, "part160_kept_by_dose": {"0.6": 12, "1.0": 7}, "required_part160_kept": {"0.6": 12, "1.0": 7}})

## Predictions

| Prediction | Verdict |
|---|---|
| 161.1: Over both picking doses, the share of picked random directions that are reliable at their picking dose is at least half the share of picked neurons that are. | held |
| 161.2: With all picked neurons laddered, the set picked at 1.0 has a median peak dose at least 0.2 above the set picked at 0.6. | held |
| 161.3: The random directions picked at 1.0 have a median peak dose at least 0.2 above those picked at 0.6. | held |
| 161.4: Among random directions picked at 0.6, at least half peak at 0.6 or below. | held |
| 161.5: Among units of either arm picked for Tencent at 0.6, at least half hit Tencent on 4 of 24 or fewer at 1.0. | held |

## Unit by dose hits

### Picked at 0.6, neuron

| Unit | Company | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | Peak | Pick hits | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L11N5647 | Google | 7/24 | 14/24 | 21/24 | 23/24 | 21/24 | 21/24 | 1.0 | 14/24 | True |
| L21N2198 | Tencent | 0/24 | 14/24 | 16/24 | 1/24 | 0/24 | 0/24 | 0.8 | 14/24 | True |
| L21N6938 | Tencent | 10/24 | 13/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.6 | 13/24 | True |
| L21N7134 | Baidu | 0/24 | 11/24 | 15/24 | 4/24 | 0/24 | 0/24 | 0.8 | 11/24 | False |
| L21N8461 | Google | 20/24 | 23/24 | 24/24 | 9/24 | 2/24 | 0/24 | 0.8 | 23/24 | True |
| L21N963 | Tencent | 0/24 | 17/24 | 7/24 | 1/24 | 0/24 | 0/24 | 0.6 | 17/24 | True |
| L22N4795 | Tencent | 0/24 | 3/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.6 | 3/24 | False |
| L22N6373 | Google | 2/24 | 21/24 | 18/24 | 4/24 | 1/24 | 0/24 | 0.6 | 21/24 | True |
| L22N6493 | Tencent | 5/24 | 14/24 | 18/24 | 11/24 | 1/24 | 0/24 | 0.8 | 14/24 | True |
| L22N761 | Google | 2/24 | 16/24 | 18/24 | 11/24 | 3/24 | 1/24 | 0.8 | 16/24 | True |
| L23N1285 | Tencent | 4/24 | 16/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.6 | 16/24 | True |
| L23N1403 | Google | 9/24 | 14/24 | 5/24 | 1/24 | 0/24 | 0/24 | 0.6 | 14/24 | True |
| L23N1998 | Tencent | 2/24 | 21/24 | 3/24 | 0/24 | 0/24 | 0/24 | 0.6 | 21/24 | True |
| L23N414 | Google | 9/24 | 21/24 | 20/24 | 1/24 | 0/24 | 0/24 | 0.6 | 21/24 | True |
| L23N4261 | Tencent | 10/24 | 12/24 | 2/24 | 0/24 | 0/24 | 0/24 | 0.6 | 12/24 | True |
| L23N4451 | Tencent | 17/24 | 15/24 | 6/24 | 0/24 | 0/24 | 0/24 | 0.4 | 15/24 | True |
| L23N4892 | Baidu | 0/24 | 8/24 | 13/24 | 13/24 | 1/24 | 0/24 | 0.9 | 8/24 | False |
| L23N5580 | Tencent | 5/24 | 5/24 | 1/24 | 1/24 | 0/24 | 0/24 | 0.5 | 5/24 | False |
| L23N5729 | Microsoft | 0/24 | 9/24 | 7/24 | 0/24 | 1/24 | 0/24 | 0.6 | 9/24 | False |
| L23N668 | Tencent | 2/24 | 15/24 | 7/24 | 1/24 | 0/24 | 0/24 | 0.6 | 15/24 | True |
| L23N6758 | Tencent | 0/24 | 19/24 | 3/24 | 0/24 | 0/24 | 0/24 | 0.6 | 19/24 | True |
| L23N8993 | Tencent | 7/24 | 18/24 | 5/24 | 0/24 | 0/24 | 0/24 | 0.6 | 18/24 | True |
| L24N1893 | Tencent | 2/24 | 4/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.6 | 4/24 | False |
| L24N3894 | Tencent | 7/24 | 13/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.6 | 13/24 | True |
| L24N4524 | Tencent | 6/24 | 8/24 | 11/24 | 3/24 | 0/24 | 0/24 | 0.8 | 8/24 | False |
| L24N5017 | Google | 1/24 | 14/24 | 21/24 | 16/24 | 7/24 | 2/24 | 0.8 | 14/24 | True |
| L24N7964 | Baidu | 0/24 | 17/24 | 12/24 | 9/24 | 4/24 | 0/24 | 0.6 | 17/24 | True |
| L24N8221 | DeepSeek | 1/24 | 5/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.6 | 5/24 | False |
| L24N8264 | Google | 13/24 | 20/24 | 15/24 | 7/24 | 5/24 | 1/24 | 0.6 | 20/24 | True |
| L25N4458 | Google | 1/24 | 8/24 | 18/24 | 11/24 | 10/24 | 2/24 | 0.8 | 8/24 | False |

### Picked at 0.6, random

| Unit | Company | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | Peak | Pick hits | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L20N4539 | Tencent | 0/24 | 14/24 | 21/24 | 4/24 | 0/24 | 0/24 | 0.8 | 14/24 | True |
| L22N4088 | Tencent | 0/24 | 6/24 | 14/24 | 1/24 | 0/24 | 0/24 | 0.8 | 6/24 | False |
| L22N4126 | Google | 5/24 | 22/24 | 15/24 | 6/24 | 1/24 | 0/24 | 0.6 | 22/24 | True |
| L22N4210 | Meta | 0/24 | 14/24 | 7/24 | 2/24 | 0/24 | 0/24 | 0.6 | 14/24 | True |
| L22N6016 | Tencent | 1/24 | 12/24 | 9/24 | 0/24 | 0/24 | 0/24 | 0.6 | 12/24 | True |
| L22N761 | Tencent | 2/24 | 19/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.6 | 19/24 | True |
| L22N8101 | Baidu | 1/24 | 16/24 | 2/24 | 0/24 | 0/24 | 0/24 | 0.6 | 16/24 | True |
| L23N1269 | Google | 5/24 | 18/24 | 4/24 | 5/24 | 0/24 | 1/24 | 0.6 | 18/24 | True |
| L23N306 | Tencent | 14/24 | 11/24 | 2/24 | 0/24 | 0/24 | 0/24 | 0.4 | 11/24 | False |
| L23N3192 | Huawei | 0/24 | 11/24 | 2/24 | 0/24 | 0/24 | 0/24 | 0.6 | 11/24 | False |
| L23N3699 | Tencent | 1/24 | 22/24 | 18/24 | 8/24 | 0/24 | 0/24 | 0.6 | 22/24 | True |
| L23N4292 | Baidu | 2/24 | 19/24 | 9/24 | 3/24 | 0/24 | 0/24 | 0.6 | 19/24 | True |
| L23N5100 | Amazon | 0/24 | 3/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.6 | 3/24 | False |
| L23N6142 | Tencent | 0/24 | 15/24 | 23/24 | 12/24 | 6/24 | 0/24 | 0.8 | 15/24 | True |
| L23N6908 | Baidu | 4/24 | 16/24 | 2/24 | 0/24 | 0/24 | 0/24 | 0.6 | 16/24 | True |
| L23N7401 | Tencent | 4/24 | 15/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.6 | 15/24 | True |
| L23N836 | Tencent | 1/24 | 3/24 | 20/24 | 11/24 | 0/24 | 0/24 | 0.8 | 3/24 | False |
| L23N8600 | Tencent | 0/24 | 21/24 | 2/24 | 0/24 | 0/24 | 0/24 | 0.6 | 21/24 | True |
| L23N8993 | Baidu | 0/24 | 11/24 | 16/24 | 5/24 | 5/24 | 0/24 | 0.8 | 11/24 | False |
| L24N1789 | Microsoft | 0/24 | 9/24 | 5/24 | 3/24 | 0/24 | 0/24 | 0.6 | 9/24 | False |
| L24N1847 | Baidu | 0/24 | 16/24 | 16/24 | 2/24 | 0/24 | 0/24 | 0.7 | 16/24 | True |
| L24N2414 | Tencent | 8/24 | 23/24 | 4/24 | 0/24 | 0/24 | 0/24 | 0.6 | 23/24 | True |
| L24N4043 | Tencent | 0/24 | 17/24 | 4/24 | 0/24 | 0/24 | 0/24 | 0.6 | 17/24 | True |
| L24N4265 | Tencent | 0/24 | 19/24 | 10/24 | 5/24 | 9/24 | 3/24 | 0.6 | 19/24 | True |
| L24N4391 | Tencent | 1/24 | 10/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.6 | 10/24 | False |
| L24N4524 | Tencent | 17/24 | 16/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.4 | 16/24 | True |
| L24N5399 | Microsoft | 0/24 | 7/24 | 5/24 | 0/24 | 0/24 | 0/24 | 0.6 | 7/24 | False |
| L24N6051 | Microsoft | 1/24 | 20/24 | 5/24 | 2/24 | 1/24 | 0/24 | 0.6 | 20/24 | True |
| L24N6082 | Tencent | 0/24 | 20/24 | 20/24 | 1/24 | 0/24 | 0/24 | 0.7 | 20/24 | True |
| L24N7350 | Microsoft | 0/24 | 8/24 | 5/24 | 8/24 | 1/24 | 0/24 | 0.8 | 8/24 | False |
| L24N76 | Baidu | 0/24 | 23/24 | 10/24 | 0/24 | 0/24 | 0/24 | 0.6 | 23/24 | True |
| L24N803 | Microsoft | 6/24 | 11/24 | 2/24 | 0/24 | 0/24 | 0/24 | 0.6 | 11/24 | False |
| L24N8036 | Tencent | 23/24 | 13/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.4 | 13/24 | True |
| L24N8264 | Microsoft | 0/24 | 20/24 | 22/24 | 13/24 | 7/24 | 2/24 | 0.8 | 20/24 | True |
| L24N8277 | Tencent | 10/24 | 19/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.6 | 19/24 | True |
| L24N9088 | Tencent | 16/24 | 23/24 | 17/24 | 1/24 | 1/24 | 0/24 | 0.6 | 23/24 | True |
| L25N2622 | Tencent | 8/24 | 11/24 | 1/24 | 0/24 | 0/24 | 0/24 | 0.6 | 11/24 | False |
| L25N3507 | OpenAI | 0/24 | 21/24 | 21/24 | 9/24 | 3/24 | 0/24 | 0.7 | 21/24 | True |
| L25N446 | Tencent | 0/24 | 18/24 | 18/24 | 5/24 | 1/24 | 0/24 | 0.7 | 18/24 | True |
| L25N4593 | OpenAI | 0/24 | 21/24 | 12/24 | 0/24 | 0/24 | 0/24 | 0.6 | 21/24 | True |
| L26N4031 | Tencent | 10/24 | 23/24 | 19/24 | 5/24 | 2/24 | 0/24 | 0.6 | 23/24 | True |

### Picked at 1.0, neuron

| Unit | Company | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | Peak | Pick hits | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L11N5647 | Google | 7/24 | 14/24 | 21/24 | 23/24 | 21/24 | 21/24 | 1.0 | 23/24 | True |
| L21N863 | Google | 0/24 | 0/24 | 8/24 | 3/24 | 0/24 | 0/24 | 0.8 | 3/24 | False |
| L22N7505 | Tencent | 0/24 | 4/24 | 10/24 | 19/24 | 11/24 | 0/24 | 1.0 | 19/24 | True |
| L23N2993 | Microsoft | 0/24 | 1/24 | 18/24 | 5/24 | 9/24 | 10/24 | 0.8 | 5/24 | False |
| L24N3249 | Google | 1/24 | 8/24 | 6/24 | 4/24 | 9/24 | 2/24 | 1.2 | 4/24 | False |
| L24N5017 | Google | 1/24 | 14/24 | 21/24 | 16/24 | 7/24 | 2/24 | 0.8 | 16/24 | True |
| L24N5904 | Google | 2/24 | 7/24 | 7/24 | 17/24 | 17/24 | 15/24 | 1.1 | 17/24 | True |

### Picked at 1.0, random

| Unit | Company | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | Peak | Pick hits | Reliable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L20N2618 | Google | 0/24 | 0/24 | 10/24 | 3/24 | 0/24 | 0/24 | 0.8 | 3/24 | False |
| L21N3059 | Google | 0/24 | 13/24 | 19/24 | 13/24 | 2/24 | 0/24 | 0.8 | 13/24 | True |
| L23N3699 | Tencent | 1/24 | 22/24 | 18/24 | 8/24 | 0/24 | 0/24 | 0.6 | 8/24 | False |
| L23N6142 | Tencent | 0/24 | 15/24 | 23/24 | 12/24 | 6/24 | 0/24 | 0.8 | 12/24 | True |
| L23N7510 | Google | 2/24 | 6/24 | 15/24 | 14/24 | 13/24 | 0/24 | 0.8 | 14/24 | True |
| L24N5235 | Google | 12/24 | 12/24 | 15/24 | 12/24 | 4/24 | 3/24 | 0.8 | 12/24 | True |

## Set summaries

| Picking dose | Arm | Units | Reliable | Median peak | Company counts |
|---|---|---:|---:|---:|---|
| 0.6 | neuron | 30 | 21 | 0.6 | {"Google": 9, "Tencent": 16, "Baidu": 3, "Microsoft": 1, "DeepSeek": 1} |
| 0.6 | random | 41 | 29 | 0.6 | {"Tencent": 22, "Google": 2, "Meta": 1, "Baidu": 6, "Huawei": 1, "Amazon": 1, "Microsoft": 6, "OpenAI": 2} |
| 1.0 | neuron | 7 | 4 | 1.0 | {"Google": 5, "Tencent": 1, "Microsoft": 1} |
| 1.0 | random | 6 | 4 | 0.8 | {"Google": 4, "Tencent": 2} |
