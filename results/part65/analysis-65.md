# Part 65 analysis

Units analyzed: 500; bootstrap replicates: 5000; shuffled-pair replicates: 1000; missing trials: 0/1000 (absent: 0, unparseable: 0); units excluded for any missing trial: 0.

| Prediction | Estimate | 95% unit-bootstrap interval | Pass |
|---|---:|---:|:---:|
| 65a neuron agreement | 0.5740 | [0.5340, 0.6120] | false |
| 65b neuron minus control | 0.1040 | [0.0500, 0.1580] | true |
| 65c last minus first third | 0.1319 | [0.0342, 0.2296] | true |
| 65d sentence pass minus fail | 0.0931 | [0.0156, 0.1706] | false |
| 65e control agreement | 0.4700 | [0.4340, 0.5040] | false |
| 65f cosine neuron minus control | 0.0055 | [-0.0023, 0.0131] | true |

65c and 65g are exploratory depth predictions.

## 65g exploratory depth agreement

Middle below both first and last by point estimate: false.

| Band | Estimate | 95% unit-bootstrap interval | n |
|---|---:|---:|---:|
| first | 0.5060 | [0.4398, 0.5723] | 83 |
| middle | 0.5750 | [0.5125, 0.6375] | 80 |
| last | 0.6379 | [0.5690, 0.7069] | 87 |

## 65f embedding measure

Registered paired difference in observed read/write cosine (neuron mean minus control mean).
Registered observed cosine difference: 0.0055 [-0.0023, 0.0131].
Null-adjusted neuron-minus-control difference: 0.0030 [-0.0048, 0.0108].

## Embedding and shuffled-pair null

512-dimensional signed Blake2b feature hashing of word unigrams and adjacent word bigrams; cosine.
within-type, within-layer-band write-summary derangement.

| Type | Paired cosine | Paired minus shuffled null |
|---|---:|---:|
| Neuron | 0.0009 | -0.0021 |
| Control | -0.0046 | -0.0051 |

## Missing-trial handling and bounds

A unit enters the primary predictions only when both registered trials have valid answers. Missing-trial bounds cover all registered units and are not used for pass or fail.

| Excluded units | First | Middle | Last | Total |
|---|---:|---:|---:|---:|
| neuron | 0 | 0 | 0 | 0 |
| control | 0 | 0 | 0 | 0 |

| Missing-trial score | 65a neuron agreement | 65b neuron minus control | First-third neuron | Middle-third neuron | Last-third neuron |
|---|---:|---:|---:|---:|---:|
| 0.5 (chance) | 0.5740 | 0.1040 | 0.5060 | 0.5750 | 0.6379 |
| 0 (wrong) | 0.5740 | 0.1040 | 0.5060 | 0.5750 | 0.6379 |

## Sensitivity: layers 0 to 2 left out (Amendment 3; not used for pass or fail)

65a neuron agreement 0.5849 (n=212); 65b neuron minus control 0.1214 (controls n=219).
