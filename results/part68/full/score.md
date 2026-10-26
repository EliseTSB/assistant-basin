# Part 68 score

| Prediction | Verdict | Observation | Rule |
|---|---|---|---|
| 68.1 | FAILS | `7` | at least 10 of 20 candidates for L29 N776 replicate on the test half |
| 68.2 | HOLDS | `19` | at least 10 of 20 candidates for L11 N8868 replicate on the test half |
| 68.3 | FAILS | `{"share_in_layers_20_28": 0.2857142857142857, "layer_histogram": {"0": 1, "2": 1, "4": 1, "5": 1, "6": 1, "24": 1, "27": 1}, "candidate_layer_histogram": {"0": 1, "2": 6, "3": 2, "4": 3, "5": 4, "6": 2, "24": 1, "27": 1}, "replicating_count": 7}` | most (more than half) of L29 N776's replicating drivers sit in layers 20 to 28 |
| 68.4 | HOLDS | `{"earlier_makers_in_late_candidates": []}` | none of L6 N753, L7 N1169, L11 N8868 is among L29 N776's 20 candidates |
| 68.5 | HOLDS | `{"11_8868": 0.0, "29_776": 0.0}` | attribution of every unit at or after the target layer is exactly 0 |
| 68.6 | HOLDS | `{"search_half_spearman": {"11_8868": 0.8457763026659079, "29_776": 0.39345666420605985}, "test_half_spearman": {"11_8868": 0.8149101502848481, "29_776": 0.02523906391754901}}` | on the search half, Delta and the attribution correlate (Spearman above 0.3) across the 220 tested units, for both targets |

## Per target (test half)

| Target | Replicating | Null p95 | All-20 Delta | Spearman (search) | Spearman (test) |
|---|---|---|---|---|---|
| 11_8868 | 19/20 | 0.004891 | -0.09087 | 0.8457763026659079 | 0.8149101502848481 |
| 29_776 | 7/20 | 0.004213 | 0.09641 | 0.39345666420605985 | 0.02523906391754901 |

## Replicating drivers (test half)

### 11_8868: 19

| Layer | Unit | A_u | Delta |
|---|---|---|---|
| 6 | 753 | -0.03383 | -0.09679 |
| 7 | 1169 | 0.02636 | 0.1158 |
| 10 | 4133 | -0.02008 | -0.09075 |
| 8 | 7186 | 0.0184 | 0.04707 |
| 10 | 54 | -0.01789 | -0.1122 |
| 7 | 296 | -0.0146 | -0.06504 |
| 6 | 159 | -0.0132 | -0.03904 |
| 10 | 3551 | -0.01012 | -0.04669 |
| 8 | 2214 | 0.009905 | 0.01714 |
| 10 | 8663 | -0.009081 | -0.04476 |
| 10 | 475 | 0.007549 | 0.03445 |
| 9 | 5442 | 0.00747 | 0.03891 |
| 5 | 6794 | 0.007 | 0.02703 |
| 7 | 4204 | -0.006659 | -0.02862 |
| 10 | 167 | 0.006276 | 0.03151 |
| 4 | 2831 | 0.006266 | 0.03067 |
| 3 | 4169 | -0.006246 | -0.0363 |
| 9 | 1740 | -0.006161 | -0.02932 |
| 10 | 14 | 0.005974 | 0.02907 |

### 29_776: 7

| Layer | Unit | A_u | Delta |
|---|---|---|---|
| 5 | 25 | -0.1348 | -0.00898 |
| 6 | 159 | -0.0922 | -0.006678 |
| 27 | 3325 | 0.0842 | 0.07689 |
| 2 | 698 | -0.0705 | -0.02229 |
| 24 | 3949 | 0.05643 | 0.04999 |
| 0 | 2891 | -0.05492 | -0.004438 |
| 4 | 695 | -0.0512 | -0.005315 |

## Overlap

Candidate overlap: [(6, 159)]. Shared replicating drivers: [{'layer': 6, 'unit': 159}]. Two random 20-unit draws from the same layers overlap 0.00 times on average (95th percentile 0, 10000 draws, seed 6801); share at least the observed 1: 0.0011.

Reading: the search found drivers.
