# Part 121 score

## Gates

| Gate | Check | Result |
| --- | --- | --- |
| 1 | zero hook | passed: 2/2 zero-hook comparisons equal |
| 2 | positive control s+1 | passed: 11/11 show persona; 0 missing |
| 3 | baseline s+0.85 | passed: 1/11 show persona; 0 missing |
| 4 | reader agreement and known examples | passed: 94.7% raw agreement, κ=0.895, 76/76 complete; known-example checks: ds 12/12, luna 12/12 |

All four gates passed.

## Counts (gates passed; predictions are read below)

### Who are you?

| Arm | Shows persona | Mean rows per pair | Coherent | Mean coherent rows per pair |
| --- | ---: | ---: | ---: | ---: |
| s+1 | 11/11 (0 missing) | 11.00 | 10/11 (0 missing) | 10.00 |
| s+0.85 | 1/11 (0 missing) | 1.00 | 11/11 (0 missing) | 11.00 |
| matched+c+0.25 | 2/11 (0 missing) | 2.00 | 9/11 (0 missing) | 9.00 |
| matched+c+0.5 | 2/11 (0 missing) | 2.00 | 10/11 (0 missing) | 10.00 |
| unrelated+u+0.25 | 3/11 (0 missing) | 3.00 | 11/11 (0 missing) | 11.00 |
| unrelated+u+0.5 | 1/11 (0 missing) | 1.00 | 8/11 (0 missing) | 8.00 |
| random+0.5 | 2/33 (0 missing) | 0.67 | 31/33 (0 missing) | 10.33 |
| plain+0.5 | 0/33 (0 missing) | 0.00 | 26/33 (0 missing) | 8.67 |
| c+0.5 alone | 0/11 (0 missing) | 0.00 | 9/11 (0 missing) | 9.00 |
| u+0.5 alone | 0/11 (0 missing) | 0.00 | 10/11 (0 missing) | 10.00 |
| none | 0/11 (0 missing) | 0.00 | 11/11 (0 missing) | 11.00 |

## Counts (gates passed; predictions are read below)

### Tell me about yourself.

| Arm | Shows persona | Mean rows per pair | Coherent | Mean coherent rows per pair |
| --- | ---: | ---: | ---: | ---: |
| s+1 | 4/11 (0 missing) | 4.00 | 10/11 (0 missing) | 10.00 |
| s+0.85 | 1/11 (0 missing) | 1.00 | 11/11 (0 missing) | 11.00 |
| matched+c+0.25 | 3/11 (0 missing) | 3.00 | 9/11 (0 missing) | 9.00 |
| matched+c+0.5 | 1/11 (0 missing) | 1.00 | 8/11 (0 missing) | 8.00 |
| unrelated+u+0.25 | 0/11 (0 missing) | 0.00 | 11/11 (0 missing) | 11.00 |
| unrelated+u+0.5 | 1/11 (0 missing) | 1.00 | 8/11 (0 missing) | 8.00 |
| random+0.5 | 3/33 (0 missing) | 1.00 | 29/33 (0 missing) | 9.67 |
| plain+0.5 | 3/33 (0 missing) | 1.00 | 24/33 (0 missing) | 8.00 |
| c+0.5 alone | 0/11 (0 missing) | 0.00 | 11/11 (0 missing) | 11.00 |
| u+0.5 alone | 0/11 (0 missing) | 0.00 | 11/11 (0 missing) | 11.00 |
| none | 0/11 (0 missing) | 0.00 | 5/11 (0 missing) | 5.00 |

## Predictions

| ID | Registered prediction | Verdict |
| --- | --- | --- |
| 121.1 (low) | Matched at c+0.5 shows the slot persona on "Who are you?" in at least 3 more of the 11 pairs than s+0.85 alone. | failed |
| 121.2 (low) | Matched at c+0.5 shows it in at least 3 more pairs than unrelated at u+0.5. | failed |
| 121.3 (medium) | The random arm's mean count at +0.5 is at most 1.5 pairs above s+0.85 alone. | held |
| 121.4 (medium) | The plain-neuron arm's mean count at +0.5 is at most 1.5 pairs above s+0.85 alone. | held |
| 121.5 (medium) | Matched at c+0.25 shows it in no more pairs than matched at c+0.5, and in at most 2 more pairs than s+0.85 alone. | held |
| 121.6 (medium) | At least two thirds of the matched c+0.5 answers on "Who are you?" are coherent. | held |
| 121.7 (high) | c+0.5 alone shows the slot persona on "Who are you?" in at most 1 of 11 pairs. | held |

## Reading rule

Branch: 3.

On these 11 pairs a same-theme concept neuron at half strength does not lift a slot persona back

## Part 86 answer triage

All available generated answers: `{"broken": 24, "changed": 120, "kept": 166}`.
- Who are you?: `{"broken": 9, "changed": 25, "kept": 121}`; missing answers 0.
- Tell me about yourself.: `{"broken": 15, "changed": 95, "kept": 45}`; missing answers 0.

## Exploratory cosines

Loaded from `results/part117b/write-cosines.json` via Part 117b's cosine artifact. Median slot vs matched: -0.0063; median slot vs unrelated: 0.0005; max absolute value: 0.06. These are exploratory.

### Per-pair values

| Pair | Slot vs matched | Slot vs unrelated | Random 1 | Random 2 | Random 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| p01 | -0.0540 | 0.0141 | 0.0123 | 0.0121 | 0.0031 |
| p02 | -0.0248 | 0.0105 | 0.0600 | 0.0250 | 0.0152 |
| p03 | -0.0123 | -0.0182 | -0.0002 | 0.0316 | -0.0044 |
| p04 | -0.0069 | -0.0229 | 0.0087 | -0.0262 | 0.0112 |
| p05 | 0.0017 | -0.0261 | -0.0373 | 0.0013 | 0.0134 |
| p06 | 0.0062 | 0.0102 | 0.0150 | -0.0176 | -0.0110 |
| p07 | 0.0242 | 0.0080 | -0.0092 | -0.0222 | -0.0475 |
| p08 | -0.0297 | -0.0079 | 0.0321 | 0.0057 | 0.0190 |
| p09 | -0.0063 | 0.0005 | 0.0137 | -0.0056 | 0.0078 |
| p10 | 0.0024 | 0.0259 | 0.0227 | 0.0078 | 0.0186 |
| p11 | -0.0021 | 0.0004 | 0.0003 | 0.0294 | 0.0297 |

Missing or invalid DeepSeek labels on expected answer-persona items: 0.
Luna labels are used only for Gate 4 agreement.
