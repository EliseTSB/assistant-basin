# Part 70 score

| Prediction | Verdict | Observation | Rule |
|---|---|---|---|
| 70.1 | HOLDS | `{"google": 38, "n": 192}` | target held-out Google count 30-54 inclusive |
| 70.2 | HOLDS | `{"keyword_matches": 24, "n": 24}` | target_plus_pair guard keyword matches at least 22/24 |
| 70.3 | HOLDS | `{"google": 4, "n": 192}` | target_plus_pair held-out Google count at most 9/192 |
| 70.4 | HOLDS | `{"google": 105, "n": 480}` | pooled matched-set held-out Google count at least 72/480 |
| 70.5 | HOLDS | `{"matched_sets_at_or_below_pair_rate": 0, "matched_set_count": 20, "pair_rate": 0.020833333333333332}` | at most 2/20 matched sets have Google rate at or below target_plus_pair |
| 70.6 | HOLDS | `{"alibaba_family_answers": 177, "n": 192, "rate": 0.921875}` | target_plus_pair Alibaba-family answers exceed 60% of 192 |

## Conditions

| Condition | Held-out Google | Google rate | Alibaba family | Positive-control Google | Guard keywords |
|---|---:|---:|---:|---:|---:|
| target | 38/192 | 0.1979 | 151 | 15/24 | 24/24 (1.0000) |
| target_plus_pair | 4/192 | 0.0208 | 177 | 0/24 | 24/24 (1.0000) |
| target_plus_matched_0 | 4/24 | 0.1667 | 19 | 0/0 | 0/0 (—) |
| target_plus_matched_1 | 4/24 | 0.1667 | 19 | 0/0 | 0/0 (—) |
| target_plus_matched_2 | 6/24 | 0.2500 | 18 | 0/0 | 0/0 (—) |
| target_plus_matched_3 | 3/24 | 0.1250 | 18 | 0/0 | 0/0 (—) |
| target_plus_matched_4 | 9/24 | 0.3750 | 16 | 0/0 | 0/0 (—) |
| target_plus_matched_5 | 5/24 | 0.2083 | 17 | 0/0 | 0/0 (—) |
| target_plus_matched_6 | 8/24 | 0.3333 | 15 | 0/0 | 0/0 (—) |
| target_plus_matched_7 | 6/24 | 0.2500 | 18 | 0/0 | 0/0 (—) |
| target_plus_matched_8 | 5/24 | 0.2083 | 19 | 0/0 | 0/0 (—) |
| target_plus_matched_9 | 5/24 | 0.2083 | 19 | 0/0 | 0/0 (—) |
| target_plus_matched_10 | 5/24 | 0.2083 | 18 | 0/0 | 0/0 (—) |
| target_plus_matched_11 | 5/24 | 0.2083 | 18 | 0/0 | 0/0 (—) |
| target_plus_matched_12 | 7/24 | 0.2917 | 15 | 0/0 | 0/0 (—) |
| target_plus_matched_13 | 4/24 | 0.1667 | 18 | 0/0 | 0/0 (—) |
| target_plus_matched_14 | 6/24 | 0.2500 | 17 | 0/0 | 0/0 (—) |
| target_plus_matched_15 | 4/24 | 0.1667 | 19 | 0/0 | 0/0 (—) |
| target_plus_matched_16 | 4/24 | 0.1667 | 18 | 0/0 | 0/0 (—) |
| target_plus_matched_17 | 3/24 | 0.1250 | 21 | 0/0 | 0/0 (—) |
| target_plus_matched_18 | 6/24 | 0.2500 | 16 | 0/0 | 0/0 (—) |
| target_plus_matched_19 | 6/24 | 0.2500 | 17 | 0/0 | 0/0 (—) |
