# Part 160 score

## Gates

- gate1_rows: passed ({"passed": true, "expected": 7696, "present": 7696, "missing": 0, "duplicates": 0, "nonempty": 7696})
- gate2_reproduction: passed ({"passed": true, "matches": 32, "compared": 32, "required": 28})
- gate3_picking: passed ({"passed": true, "kept_by_dose": {"0.6": 12, "1.0": 7}, "minimum": 4})
- gate4_controls: passed ({"passed": true, "L24N5017_Google_hits_at_0_8": 21, "required": [15, 22], "L21N7134_Baidu_hits_at_0_8": 15, "required_baidu": [11, 20], "no_push_any_picked_company_hits": 0, "maximum": 2})

## Registered predictions

| Prediction | Confidence | Verdict |
|---|---|---|
| 160.1: The set picked at 1.0 has a median peak dose at least 0.2 above the set picked at 0.6. | medium | held |
| 160.2: In each set, at least a third of neurons hit their company on at least 12 of 24 at their picking dose. | low | held |
| 160.3: The two sets share at most 2 neurons, and each shares at most 3 with Part 148's ten. | medium | held |
| 160.4: At its picking dose, each set's mean neuron hit rate is at least twice its paired random directions'. | medium | held |
| 160.5: In each set, at least half of neurons peak within 0.2 of their picking dose. | low | held |
| 160.6: Google is the company of at least half of the neurons in each set. | low | failed |

## Unit by dose hits

### Picked at 0.6

| Unit | Company | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | Peak | Pick hits | Random rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L11N5647 | Google | 7/24 | 14/24 | 21/24 | 23/24 | 21/24 | 21/24 | 1.0 | 14/24 | 4.2% |
| L21N963 | Tencent | 0/24 | 17/24 | 7/24 | 1/24 | 0/24 | 0/24 | 0.6 | 17/24 | 0.0% |
| L21N2198 | Tencent | 0/24 | 14/24 | 16/24 | 1/24 | 0/24 | 0/24 | 0.8 | 14/24 | 0.0% |
| L21N6938 | Tencent | 10/24 | 13/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.6 | 13/24 | 0.0% |
| L21N7134 | Baidu | 0/24 | 11/24 | 15/24 | 4/24 | 0/24 | 0/24 | 0.8 | 11/24 | 0.0% |
| L21N8461 | Google | 20/24 | 23/24 | 24/24 | 9/24 | 2/24 | 0/24 | 0.8 | 23/24 | 0.0% |
| L22N761 | Google | 2/24 | 16/24 | 18/24 | 11/24 | 3/24 | 1/24 | 0.8 | 16/24 | 0.0% |
| L22N4795 | Tencent | 0/24 | 3/24 | 0/24 | 0/24 | 0/24 | 0/24 | 0.6 | 3/24 | 0.0% |
| L22N6373 | Google | 2/24 | 21/24 | 18/24 | 4/24 | 1/24 | 0/24 | 0.6 | 21/24 | 0.0% |
| L22N6493 | Tencent | 5/24 | 14/24 | 18/24 | 11/24 | 1/24 | 0/24 | 0.8 | 14/24 | 0.0% |
| L23N414 | Google | 9/24 | 21/24 | 20/24 | 1/24 | 0/24 | 0/24 | 0.6 | 21/24 | 20.8% |
| L23N668 | Tencent | 2/24 | 15/24 | 7/24 | 1/24 | 0/24 | 0/24 | 0.6 | 15/24 | 4.2% |

### Picked at 1.0

| Unit | Company | 0.4 | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | Peak | Pick hits | Random rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L11N5647 | Google | 7/24 | 14/24 | 21/24 | 23/24 | 21/24 | 21/24 | 1.0 | 23/24 | 29.2% |
| L21N863 | Google | 0/24 | 0/24 | 8/24 | 3/24 | 0/24 | 0/24 | 0.8 | 3/24 | 0.0% |
| L22N7505 | Tencent | 0/24 | 4/24 | 10/24 | 19/24 | 11/24 | 0/24 | 1.0 | 19/24 | 0.0% |
| L23N2993 | Microsoft | 0/24 | 1/24 | 18/24 | 5/24 | 9/24 | 10/24 | 0.8 | 5/24 | 0.0% |
| L24N3249 | Google | 1/24 | 8/24 | 6/24 | 4/24 | 9/24 | 2/24 | 1.2 | 4/24 | 4.2% |
| L24N5017 | Google | 1/24 | 14/24 | 21/24 | 16/24 | 7/24 | 2/24 | 0.8 | 16/24 | 0.0% |
| L24N5904 | Google | 2/24 | 7/24 | 7/24 | 17/24 | 17/24 | 15/24 | 1.1 | 17/24 | 0.0% |

## Set summary

| Set | Units | Mean neuron rate at pick | Mean random rate at pick | Median peak | No peak |
|---|---:|---:|---:|---:|---:|
| 0.6 | 12 | 0.6319444444444444 | 0.024305555555555556 | 0.7 | 0 |
| 1.0 | 7 | 0.5178571428571429 | 0.047619047619047616 | 1.0 | 0 |

## Overlaps

| Comparison | Count |
|---|---:|
| picked sets | 1 |
| picked at 0.6 with Part 148 ten | 3 |
| picked at 1.0 with Part 148 ten | 3 |

## Exploratory Part 148 rescore

| Rule | Count | Units |
|---|---:|---|
| Part 160 rule on stored 0.8 answers | 13 | L11N5647, L21N7134, L22N6997, L22N7505, L22N7832, L23N414, L23N1743, L23N2993, L24N5017, L24N8264, L25N140, L25N4457, L25N4458 |
| Part 148 ten | 10 | L11N5647, L21N7134, L22N6997, L23N414, L23N1743, L23N2993, L24N5017, L25N140, L25N4457, L25N4458 |
