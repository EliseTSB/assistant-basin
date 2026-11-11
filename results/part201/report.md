# Part 201 score

## Gates

| Gate | Result | Detail |
|---|---|---|
| G1 | passed | bf16 passed, float32 passed, kernel record passed; revisions ['851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a']; snapshot loaded is the recorded revision: True |
| G2 | passed | 0 of 1648 rows failed; 2240 positions emitted a tied token that is not the first-listed top |
| G3 | passed | plan passed, bf16 passed, float32 passed;  |
| G4 | passed | bf16 passed, float32 passed; 0 of 1600 pushed rows failed; 0 calls without matching hook evidence; 0 rows with more than one push hash in bf16, 0 whose float32 push hash is not the bf16 one; inputs bound across launches: bf16 True, float32 True |
| G5 | passed | 1 launch records |

Positive control: A and C differ on 86 rows (needs 10): passed.
Run state: smoke plan False; float32 cut applied False; A-again in pass A's process True.

## Registered predictions

| # | Prediction (verbatim) | Confidence | Verdict |
|---|---|---|---|
| 201.1 | A-again equals A on all 16 rows, and A2 equals A on all 96, token for token. | high | held |
| 201.2 | Between A and C, 25% to 50% of the 192 rows differ. | medium | held |
| 201.3 | Between A and M, at least 20% of the 384 rows differ. | low | held |
| 201.4 | Between A and B (other slots and other neighbours, same size and prompts), at most 10% of the 384 rows differ. | low | held |
| 201.5 | Of the rows that differ between A and C, at least 90% have a pass-A margin of at most 2 steps at the first differing position, and none has one above 8 steps. | high | held |
| 201.6 | Between A and M, for each of the four labels with at least 8 flips, the two directions of flip differ by at most twice the square root of the number of flips; and mean answer length differs by under 3 tokens. | medium | held |
| 201.7 | K matches the CPU reference token for token on all 16 rows, over every position. | medium | held |
| 201.8 | Between F8 and F1, none of the 48 rows differ, with all 48 present and finite in both. | medium | held |
| 201.9 | Of the rows that differ between A and F8, at least 90% have a pass-A margin of at most 2 steps at the first differing position, and none has an F8 (float32) margin above 0.5 there. | medium | held |
| 201.10 | Between A and F8, 30% to 70% of the 96 rows differ. | low | held |
| 201.11 | The projected slowdown of margin-gated decoding on pass A is at most 2: (pass A's seconds + the summed cost of one full-prefix float32 forward for the flagged rows at every step where an unfinished row is within 2 steps) ÷ pass A's seconds, the cost of each forward read from the timing probe at the nearest larger row count and prefix length. | low | failed |

### Estimates behind the verdicts

- **201.1**: `{"A-again": {"differing": 0, "n": 16, "planned": 16, "clause": "held"}, "A2": {"differing": 0, "n": 96, "planned": 96, "clause": "held"}}`
- **201.2**: `{"differing": 86, "n": 192, "planned": 192, "share": 0.4479166666666667, "wilson95": [0.3792847122123897, 0.5185918655028974]}`
- **201.3**: `{"differing": 147, "n": 384, "planned": 384, "share": 0.3828125, "wilson95": [0.33558410606022504, 0.4323623113401679]}`
- **201.4**: `{"differing": 0, "n": 384, "planned": 384, "share": 0.0, "wilson95": [8.673617379884035e-19, 0.009904714241677031]}`
- **201.5**: `{"differing": 86, "with_a_margin": 86, "rows_without_a_recorded_margin": [], "caught": 84, "caught_share": 0.9767441860465116, "largest_pass_A_steps": 3.0}`
- **201.6**: `{"labels": {"name_gone": {"a_true_b_false": 1, "a_false_b_true": 4, "flips": 5, "direction_clause": "not scored (5 flips)"}, "maker_gone": {"a_true_b_false": 4, "a_false_b_true": 2, "flips": 6, "direction_clause": "not scored (6 flips)"}, "broken": {"a_true_b_false": 8, "a_false_b_true": 8, "flips": 16, "direction_clause": "held"}, "qwen_part161_rule": {"a_true_b_false": 1, "a_false_b_true": 0, "flips": 1, "direction_clause": "not scored (1 flips)"}}, "mean_length_difference_a_minus_m": -0.15885416666666666, "length_clause": "held"}`
- **201.7**: `{"1": {"rows": 8, "equal_rows": 8, "first_differences": []}, "2": {"rows": 8, "equal_rows": 8, "first_differences": []}}`
- **201.8**: `{"differing": 0, "n": 48, "planned": 48, "nonfinite": 0}`
- **201.9**: `{"differing": 58, "with_a_margin": 58, "rows_without_a_recorded_margin": [], "caught": 58, "caught_share": 1.0, "largest_pass_A_steps": 2.0, "largest_F8_margin": 0.23192787170410156}`
- **201.10**: `{"differing": 58, "n": 96, "planned": 96, "share": 0.6041666666666666, "wilson95": [0.5041511204834945, 0.6961664653794899]}`
- **201.11**: `{"pass_A_seconds": 516.0977051999653, "by_margin_steps": {"1": {"margin_steps": 1, "flagged_steps": 1324, "added_seconds": 757.1675766103963, "projected_slowdown": 2.467101227115565, "flagged_steps_priced_by_extrapolation_beyond_the_probe": 23, "flagged_steps_the_probe_cannot_price": 0}, "2": {"margin_steps": 2, "flagged_steps": 1892, "added_seconds": 1114.0262599123107, "projected_slowdown": 3.158556894727277, "flagged_steps_priced_by_extrapolation_beyond_the_probe": 38, "flagged_steps_the_probe_cannot_price": 0}, "3": {"margin_steps": 3, "flagged_steps": 2307, "added_seconds": 1386.537701197598, "projected_slowdown": 3.686579861191933, "flagged_steps_priced_by_extrapolation_beyond_the_probe": 45, "flagged_steps_the_probe_cannot_price": 0}, "4": {"margin_steps": 4, "flagged_steps": 2594, "added_seconds": 1592.3171640737328, "projected_slowdown": 4.08530177140931, "flagged_steps_priced_by_extrapolation_beyond_the_probe": 50, "flagged_steps_the_probe_cannot_price": 0}}}`

## Ordered decision (BATCH-SHAPE-NOISE.md, section 3)

Rule 4, No build: section 1 is the standing procedure.

Readings: 201.11: failed.

Candidate gate: 3 steps (the largest pass-A margin, in bf16 steps, at a first difference between A and F8, rounded up to a whole step, plus one). Cost at that gate: `{"margin_steps": 3, "flagged_steps": 2307, "added_seconds": 1386.537701197598, "projected_slowdown": 3.686579861191933, "flagged_steps_priced_by_extrapolation_beyond_the_probe": 45, "flagged_steps_the_probe_cannot_price": 0}`. A price above the longest probed prefix is a linear extrapolation, an approximation and not a measured bound.

## Pairs

| Pair | Planned | Compared | Differ | Share (95% Wilson) | Not finite | Mean length A minus B | Other token not the runner-up |
|---|---:|---:|---:|---|---:|---:|---:|
| A/A-again | 16 | 16 | 0 | 0.000 (0.000 to 0.194) | 0 | 0.00 | 0 of 0 |
| A/A2 | 96 | 96 | 0 | 0.000 (0.000 to 0.038) | 0 | 0.00 | 0 of 0 |
| A/B | 384 | 384 | 0 | 0.000 (0.000 to 0.010) | 0 | 0.00 | 0 of 0 |
| A/M | 384 | 384 | 147 | 0.383 (0.336 to 0.432) | 0 | -0.16 | 80 of 147 |
| M/C (plain rows) | 24 | 24 | 4 | 0.167 (0.067 to 0.359) | 0 | 0.04 | 2 of 4 |
| A/C | 192 | 192 | 86 | 0.448 (0.379 to 0.519) | 0 | 0.45 | 43 of 86 |
| F8/F1 | 48 | 48 | 0 | 0.000 (0.000 to 0.074) | 0 | 0.00 | 0 of 0 |
| A/F8 | 96 | 96 | 58 | 0.604 (0.504 to 0.696) | 0 | -1.79 | 48 of 58 |
| C/F1 | 48 | 48 | 31 | 0.646 (0.504 to 0.766) | 0 | -2.19 | 23 of 31 |

## Label flips by pair and direction (count of n, 95% Wilson interval)

| Pair | Label | True in the first, false in the second | False in the first, true in the second | n |
|---|---|---|---|---:|
| A/A-again | name_gone | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| A/A-again | maker_gone | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| A/A-again | broken | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| A/A-again | qwen_part161_rule | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| A/A2 | name_gone | 0 (0.000 to 0.038) | 0 (0.000 to 0.038) | 96 |
| A/A2 | maker_gone | 0 (0.000 to 0.038) | 0 (0.000 to 0.038) | 96 |
| A/A2 | broken | 0 (0.000 to 0.038) | 0 (0.000 to 0.038) | 96 |
| A/A2 | qwen_part161_rule | 0 (0.000 to 0.038) | 0 (0.000 to 0.038) | 96 |
| A/B | name_gone | 0 (0.000 to 0.010) | 0 (0.000 to 0.010) | 384 |
| A/B | maker_gone | 0 (0.000 to 0.010) | 0 (0.000 to 0.010) | 384 |
| A/B | broken | 0 (0.000 to 0.010) | 0 (0.000 to 0.010) | 384 |
| A/B | qwen_part161_rule | 0 (0.000 to 0.010) | 0 (0.000 to 0.010) | 384 |
| A/M | name_gone | 1 (0.000 to 0.015) | 4 (0.004 to 0.026) | 384 |
| A/M | maker_gone | 4 (0.004 to 0.026) | 2 (0.001 to 0.019) | 384 |
| A/M | broken | 8 (0.011 to 0.041) | 8 (0.011 to 0.041) | 384 |
| A/M | qwen_part161_rule | 1 (0.000 to 0.015) | 0 (0.000 to 0.010) | 384 |
| M/C | name_gone | 0 (0.000 to 0.138) | 0 (0.000 to 0.138) | 24 |
| M/C | maker_gone | 0 (0.000 to 0.138) | 0 (0.000 to 0.138) | 24 |
| M/C | broken | 0 (0.000 to 0.138) | 0 (0.000 to 0.138) | 24 |
| M/C | qwen_part161_rule | 0 (0.000 to 0.138) | 0 (0.000 to 0.138) | 24 |
| A/C | name_gone | 1 (0.001 to 0.029) | 2 (0.003 to 0.037) | 192 |
| A/C | maker_gone | 2 (0.003 to 0.037) | 2 (0.003 to 0.037) | 192 |
| A/C | broken | 4 (0.008 to 0.052) | 1 (0.001 to 0.029) | 192 |
| A/C | qwen_part161_rule | 1 (0.001 to 0.029) | 1 (0.001 to 0.029) | 192 |
| F8/F1 | name_gone | 0 (0.000 to 0.074) | 0 (0.000 to 0.074) | 48 |
| F8/F1 | maker_gone | 0 (0.000 to 0.074) | 0 (0.000 to 0.074) | 48 |
| F8/F1 | broken | 0 (0.000 to 0.074) | 0 (0.000 to 0.074) | 48 |
| F8/F1 | qwen_part161_rule | 0 (0.000 to 0.074) | 0 (0.000 to 0.074) | 48 |
| A/F8 | name_gone | 2 (0.006 to 0.073) | 2 (0.006 to 0.073) | 96 |
| A/F8 | maker_gone | 3 (0.011 to 0.088) | 2 (0.006 to 0.073) | 96 |
| A/F8 | broken | 1 (0.002 to 0.057) | 2 (0.006 to 0.073) | 96 |
| A/F8 | qwen_part161_rule | 1 (0.002 to 0.057) | 0 (0.000 to 0.038) | 96 |
| C/F1 | name_gone | 1 (0.004 to 0.109) | 1 (0.004 to 0.109) | 48 |
| C/F1 | maker_gone | 3 (0.021 to 0.168) | 1 (0.004 to 0.109) | 48 |
| C/F1 | broken | 0 (0.000 to 0.074) | 0 (0.000 to 0.074) | 48 |
| C/F1 | qwen_part161_rule | 1 (0.004 to 0.109) | 0 (0.000 to 0.074) | 48 |

## Margins (bf16 rows)

Share of positions within 0, 1, 2 and 4 steps, over 1504 bf16 decodes (a row decoded in several passes counts once per pass): `{"0": 0.02144998036943761, "1": 0.060797288109624724, "2": 0.09854542320619751, "4": 0.166036254297178}`.

Pass M padding: `{"rows": 408, "rows_left_padded": 328, "median_pad_tokens_of_those": 5.0, "calls": 51, "calls_with_unequal_prompt_lengths": 51}`.

Share of generation steps with an unfinished row within 2 steps, by pass: `{"A": 0.49270833333333336, "B": 0.4989468141126909, "M": 0.4965686274509804, "C": 0.09509140829036362}`.

Exact ties: `{"exact_ties": 2240, "emitted_the_lower_token_id": 2240, "same_tie_different_emitted_token_by_pair": {"A/A-again": 0, "A/A2": 0, "A/B": 0, "A/M": 0, "M/C": 0, "A/C": 0}}`.

## First differences, row by row

### A/M: 147 of 384 rows; positions `{"0": 6, "1": 1, "2": 2, "4": 4, "5": 2, "6": 5, "7": 4, "8": 7, "9": 7, "10": 4, "11": 2, "12": 4, "13": 5, "14": 8, "15": 5, "16": 6, "17": 1, "18": 2, "19": 1, "20": 3, "21": 3, "22": 4, "23": 3, "24": 4, "25": 1, "26": 1, "27": 3, "28": 3, "29": 2, "30": 1, "31": 2, "34": 2, "35": 2, "36": 1, "37": 1, "38": 2, "39": 1, "40": 1, "41": 2, "42": 1, "43": 3, "44": 1, "45": 2, "46": 2, "47": 2, "48": 1, "52": 1, "54": 1, "55": 1, "56": 1, "57": 1, "58": 1, "60": 1, "61": 1, "62": 2, "63": 2, "64": 1, "67": 1, "71": 1, "72": 1, "79": 1}`

| Row | Position | First pass margin | Steps | Second pass margin | Other token is the runner-up |
|---|---:|---:|---:|---:|---|
| L19N5359|q2 | 42 | 0.0000 | 0.00 | 0.1250 | False |
| L19N5359|q4 | 45 | 0.1250 | 1.00 | 0.0000 | True |
| L19N5359|q7 | 24 | 0.1250 | 1.00 | 0.0000 | True |
| L19N5359|q9 | 27 | 0.1250 | 1.00 | 0.0000 | True |
| L21D284|q0 | 14 | 0.0000 | 0.00 | 0.1250 | False |
| L21D284|q10 | 56 | 0.0000 | 0.00 | 0.1250 | False |
| L21D284|q15 | 9 | 0.0625 | 1.00 | 0.0625 | True |
| L21D284|q2 | 41 | 0.0000 | 0.00 | 0.1250 | False |
| L21D284|q3 | 67 | 0.0000 | 0.00 | 0.1250 | False |
| L21D284|q4 | 12 | 0.1250 | 1.00 | 0.0000 | True |
| L21D284|q6 | 21 | 0.0000 | 0.00 | 0.1250 | False |
| L21D284|q8 | 38 | 0.0000 | 0.00 | 0.1250 | False |
| L21D284|q9 | 35 | 0.0000 | 0.00 | 0.1250 | False |
| L21N5109|q10 | 27 | 0.2500 | 2.00 | 0.0000 | True |
| L21N5109|q14 | 61 | 0.0000 | 0.00 | 0.2500 | False |
| L21N5109|q15 | 13 | 0.0000 | 0.00 | 0.1250 | False |
| L21N5109|q2 | 9 | 0.2500 | 2.00 | 0.1250 | True |
| L21N5109|q3 | 47 | 0.1250 | 1.00 | 0.0000 | True |
| L21N5109|q9 | 7 | 0.0625 | 1.00 | 0.2500 | True |
| L22D329|q0 | 0 | 0.0000 | 0.00 | 0.1250 | False |
| L22D329|q12 | 14 | 0.1250 | 2.00 | 0.0000 | True |
| L22D329|q15 | 6 | 0.0625 | 1.00 | 0.0625 | True |
| L22D329|q4 | 16 | 0.1250 | 1.00 | 0.0000 | True |
| L22D329|q5 | 27 | 0.0000 | 0.00 | 0.1250 | False |
| L22D329|q7 | 13 | 0.0000 | 0.00 | 0.0625 | False |
| L22D329|q8 | 21 | 0.0000 | 0.00 | 0.2500 | False |
| L22D329|q9 | 16 | 0.1250 | 1.00 | 0.0000 | True |
| L23D390|q1 | 36 | 0.1250 | 1.00 | 0.0000 | True |
| L23D390|q2 | 15 | 0.0000 | 0.00 | 0.1250 | False |
| L23D390|q6 | 7 | 0.0000 | 0.00 | 0.0625 | False |
| L23D390|q7 | 9 | 0.0000 | 0.00 | 0.1250 | False |
| L23D497|q0 | 0 | 0.0000 | 0.00 | 0.1250 | False |
| L23D497|q1 | 16 | 0.0000 | 0.00 | 0.1250 | False |
| L23D497|q10 | 5 | 0.0000 | 0.00 | 0.1250 | False |
| L23D497|q14 | 34 | 0.1250 | 1.00 | 0.0000 | True |
| L23D497|q15 | 62 | 0.0000 | 0.00 | 0.0625 | False |
| L23D497|q2 | 9 | 0.0000 | 0.00 | 0.0625 | False |
| L23D497|q3 | 23 | 0.0000 | 0.00 | 0.0625 | False |
| L23D497|q7 | 14 | 0.0000 | 0.00 | 0.0625 | False |
| L23D497|q9 | 63 | 0.0000 | 0.00 | 0.1250 | False |
| L23N306|q10 | 10 | 0.0000 | 0.00 | 0.0625 | False |
| L23N306|q12 | 44 | 0.0625 | 1.00 | 0.0000 | True |
| L23N306|q15 | 10 | 0.0625 | 1.00 | 0.0000 | True |
| L23N306|q5 | 71 | 0.0000 | 0.00 | 0.1250 | False |
| L23N306|q9 | 20 | 0.0000 | 0.00 | 0.0625 | False |
| L23N8582|q10 | 39 | 0.1250 | 1.00 | 0.0000 | True |
| L23N8582|q12 | 52 | 0.0000 | 0.00 | 0.1250 | False |
| L23N8582|q14 | 8 | 0.1250 | 1.00 | 0.0000 | True |
| L23N8582|q15 | 16 | 0.1250 | 1.00 | 0.1250 | True |
| L23N8582|q2 | 21 | 0.0000 | 0.00 | 0.1250 | False |
| L23N8582|q5 | 43 | 0.0000 | 0.00 | 0.0625 | False |
| L23N8582|q8 | 15 | 0.1250 | 1.00 | 0.0000 | True |
| L24D116|q0 | 11 | 0.0625 | 1.00 | 0.0625 | True |
| L24D116|q1 | 8 | 0.1250 | 1.00 | 0.0000 | True |
| L24D116|q14 | 43 | 0.0000 | 0.00 | 0.1250 | False |
| L24D116|q2 | 31 | 0.1250 | 1.00 | 0.0000 | True |
| L24D116|q4 | 26 | 0.1250 | 1.00 | 0.0000 | True |
| L24D116|q9 | 28 | 0.1250 | 1.00 | 0.0000 | True |
| L24D42|q1 | 4 | 0.0000 | 0.00 | 0.1875 | False |
| L24D42|q10 | 5 | 0.0625 | 1.00 | 0.1250 | True |
| L24D42|q13 | 1 | 0.0000 | 0.00 | 0.1250 | False |
| L24D42|q15 | 58 | 0.0000 | 0.00 | 0.0000 | False |
| L24D42|q2 | 6 | 0.1250 | 2.00 | 0.0000 | True |
| L24D42|q3 | 15 | 0.0000 | 0.00 | 0.0625 | False |
| L24D42|q4 | 20 | 0.0000 | 0.00 | 0.2500 | False |
| L24D42|q9 | 20 | 0.1250 | 2.00 | 0.0000 | False |
| L24D4|q12 | 7 | 0.0000 | 0.00 | 0.0625 | False |
| L24D4|q15 | 22 | 0.1250 | 1.00 | 0.0000 | True |
| L24D4|q3 | 23 | 0.1250 | 1.00 | 0.0000 | False |
| L24D4|q5 | 48 | 0.1250 | 1.00 | 0.0000 | True |
| L24D56|q0 | 16 | 0.1250 | 1.00 | 0.0000 | True |
| L24D56|q1 | 47 | 0.0000 | 0.00 | 0.1250 | False |
| L24D56|q13 | 7 | 0.1250 | 1.00 | 0.0000 | True |
| L24D56|q15 | 25 | 0.0000 | 0.00 | 0.0625 | False |
| L24D56|q3 | 6 | 0.1250 | 2.00 | 0.0625 | True |
| L24D56|q9 | 11 | 0.0000 | 0.00 | 0.1250 | False |
| L24D65|q0 | 0 | 0.0000 | 0.00 | 0.1250 | False |
| L24D65|q1 | 18 | 0.0625 | 1.00 | 0.0000 | True |
| L24D65|q13 | 40 | 0.0000 | 0.00 | 0.1250 | False |
| L24D65|q2 | 14 | 0.1250 | 1.00 | 0.0000 | True |
| L24D65|q3 | 6 | 0.0000 | 0.00 | 0.0625 | False |
| L24D65|q5 | 35 | 0.1250 | 1.00 | 0.0000 | True |
| L24D65|q9 | 45 | 0.1250 | 1.00 | 0.1250 | True |
| L24N1813|q10 | 12 | 0.1250 | 1.00 | 0.0000 | True |
| L24N1813|q3 | 9 | 0.0000 | 0.00 | 0.1250 | False |
| L24N5149|q1 | 38 | 0.0000 | 0.00 | 0.1250 | False |
| L24N5149|q15 | 22 | 0.1250 | 1.00 | 0.0000 | True |
| L24N5149|q2 | 64 | 0.0000 | 0.00 | 0.1250 | False |
| L24N5149|q3 | 17 | 0.0000 | 0.00 | 0.1250 | False |
| L24N5149|q4 | 15 | 0.1250 | 1.00 | 0.0000 | True |
| L24N5149|q5 | 46 | 0.0000 | 0.00 | 0.1250 | False |
| L24N5149|q6 | 4 | 0.1250 | 1.00 | 0.0000 | True |
| L24N5149|q9 | 41 | 0.0000 | 0.00 | 0.1250 | False |
| L25D13|q0 | 29 | 0.0000 | 0.00 | 0.0000 | False |
| L25D13|q1 | 12 | 0.1250 | 2.00 | 0.0000 | True |
| L25D13|q2 | 37 | 0.0000 | 0.00 | 0.1250 | False |
| L25D13|q3 | 10 | 0.0000 | 0.00 | 0.1250 | False |
| L25D13|q4 | 12 | 0.1250 | 1.00 | 0.0000 | True |
| L25D13|q5 | 9 | 0.0000 | 0.00 | 0.1875 | False |
| L25D13|q7 | 29 | 0.1250 | 1.00 | 0.0000 | True |
| L25D80|q0 | 24 | 0.0000 | 0.00 | 0.1250 | False |
| L25D80|q12 | 79 | 0.1250 | 1.00 | 0.1250 | True |
| L25D80|q13 | 13 | 0.0000 | 0.00 | 0.1250 | False |
| L25D80|q15 | 14 | 0.1250 | 1.00 | 0.0000 | True |
| L25D80|q2 | 4 | 0.0000 | 0.00 | 0.1250 | False |
| L25D80|q3 | 60 | 0.0000 | 0.00 | 0.0000 | False |
| L25N1898|q0 | 24 | 0.0000 | 0.00 | 0.0625 | False |
| L25N1898|q1 | 13 | 0.0000 | 0.00 | 0.0000 | False |
| L25N1898|q10 | 43 | 0.0625 | 0.50 | 0.0000 | True |
| L25N1898|q12 | 2 | 0.0000 | 0.00 | 0.1250 | False |
| L25N1898|q15 | 16 | 0.0000 | 0.00 | 0.1250 | False |
| L25N4457|q0 | 0 | 0.0000 | 0.00 | 0.1250 | False |
| L25N4457|q10 | 62 | 0.1250 | 1.00 | 0.0000 | True |
| L25N4457|q12 | 63 | 0.1250 | 1.00 | 0.0000 | True |
| L25N4457|q2 | 14 | 0.1250 | 1.00 | 0.0000 | True |
| L25N4457|q4 | 46 | 0.1250 | 1.00 | 0.0000 | True |
| L25N4457|q6 | 22 | 0.0000 | 0.00 | 0.1250 | False |
| L25N4457|q7 | 54 | 0.1250 | 1.00 | 0.0000 | True |
| L25N4457|q9 | 2 | 0.1250 | 1.00 | 0.0000 | True |
| L25N760|q0 | 0 | 0.0000 | 0.00 | 0.1250 | False |
| L25N760|q3 | 14 | 0.0000 | 0.00 | 0.1250 | False |
| L25N760|q5 | 9 | 0.1250 | 1.00 | 0.0000 | True |
| L25N760|q9 | 8 | 0.0000 | 0.00 | 0.0625 | False |
| L25N7804|q10 | 55 | 0.1250 | 1.00 | 0.0000 | True |
| L25N7804|q14 | 24 | 0.0000 | 0.00 | 0.1250 | False |
| L25N7804|q4 | 22 | 0.0000 | 0.00 | 0.1250 | False |
| L25N7804|q6 | 28 | 0.0000 | 0.00 | 0.1250 | False |
| L25N7804|q9 | 57 | 0.1250 | 1.00 | 0.0000 | True |
| L26D365|q0 | 0 | 0.0000 | 0.00 | 0.1250 | False |
| L26D365|q12 | 15 | 0.0000 | 0.00 | 0.1250 | False |
| L26D365|q13 | 34 | 0.1250 | 2.00 | 0.0000 | True |
| L26D365|q3 | 30 | 0.0000 | 0.00 | 0.0625 | False |
| L26D365|q8 | 72 | 0.1250 | 1.00 | 0.0000 | True |
| L26D365|q9 | 4 | 0.0625 | 1.00 | 0.0625 | True |
| L26N2572|q15 | 31 | 0.0000 | 0.00 | 0.1250 | False |
| L26N2572|q3 | 14 | 0.0625 | 1.00 | 0.0000 | True |
| L26N2572|q4 | 10 | 0.0625 | 1.00 | 0.0000 | True |
| L26N2572|q9 | 8 | 0.0000 | 0.00 | 0.1250 | False |
| L26N8926|q0 | 18 | 0.0625 | 1.00 | 0.0000 | True |
| L26N8926|q1 | 23 | 0.1250 | 1.00 | 0.0000 | True |
| L26N8926|q10 | 8 | 0.1250 | 1.00 | 0.0000 | True |
| L26N8926|q12 | 6 | 0.0000 | 0.00 | 0.1250 | False |
| L26N8926|q13 | 8 | 0.1250 | 1.00 | 0.0000 | True |
| L26N8926|q15 | 19 | 0.0000 | 0.00 | 0.1250 | False |
| L26N8926|q2 | 13 | 0.1250 | 1.00 | 0.0000 | True |
| L26N8926|q3 | 28 | 0.1250 | 1.00 | 0.0000 | True |
| L26N8926|q4 | 8 | 0.0000 | 0.00 | 0.1250 | False |

### M/C: 4 of 24 rows; positions `{"36": 1, "55": 1, "63": 1, "75": 1}`

| Row | Position | First pass margin | Steps | Second pass margin | Other token is the runner-up |
|---|---:|---:|---:|---:|---|
| plain|k6 | 55 | 0.1250 | 1.00 | 0.0000 | False |
| plain|k7 | 63 | 0.1250 | 1.00 | 0.1250 | True |
| plain|q14 | 75 | 0.1250 | 1.00 | 0.0000 | True |
| plain|q9 | 36 | 0.0000 | 0.00 | 0.1250 | False |

### A/C: 86 of 192 rows; positions `{"5": 1, "6": 2, "7": 5, "8": 4, "9": 5, "10": 3, "11": 1, "12": 1, "13": 4, "15": 5, "16": 1, "17": 3, "18": 1, "20": 3, "21": 3, "22": 3, "23": 3, "24": 3, "26": 2, "27": 3, "28": 3, "29": 2, "30": 2, "31": 2, "33": 1, "34": 1, "36": 1, "37": 1, "39": 1, "43": 1, "45": 2, "47": 2, "48": 2, "57": 1, "59": 1, "63": 1, "69": 2, "72": 1, "76": 1, "77": 1, "79": 1}`

| Row | Position | First pass margin | Steps | Second pass margin | Other token is the runner-up |
|---|---:|---:|---:|---:|---|
| L19N5359|q1 | 22 | 0.0000 | 0.00 | 0.1250 | False |
| L19N5359|q10 | 39 | 0.1250 | 1.00 | 0.0000 | True |
| L19N5359|q2 | 59 | 0.0000 | 0.00 | 0.1250 | False |
| L19N5359|q3 | 7 | 0.0000 | 0.00 | 0.2500 | False |
| L19N5359|q4 | 12 | 0.0000 | 0.00 | 0.0625 | False |
| L19N5359|q5 | 24 | 0.1250 | 1.00 | 0.0000 | True |
| L19N5359|q8 | 15 | 0.0000 | 0.00 | 0.1250 | False |
| L21N5109|q10 | 33 | 0.0625 | 1.00 | 0.0000 | True |
| L21N5109|q15 | 6 | 0.0625 | 1.00 | 0.0000 | True |
| L21N5109|q2 | 9 | 0.2500 | 2.00 | 0.0000 | True |
| L21N5109|q3 | 47 | 0.1250 | 1.00 | 0.0000 | True |
| L21N5109|q5 | 30 | 0.1250 | 1.00 | 0.0000 | True |
| L21N5109|q6 | 8 | 0.0625 | 1.00 | 0.0000 | True |
| L21N5109|q7 | 37 | 0.1250 | 1.00 | 0.0000 | True |
| L22D329|q0 | 9 | 0.0000 | 0.00 | 0.1250 | False |
| L22D329|q11 | 6 | 0.0625 | 1.00 | 0.0000 | True |
| L22D329|q13 | 31 | 0.0625 | 1.00 | 0.0625 | True |
| L22D329|q14 | 17 | 0.0625 | 1.00 | 0.0000 | True |
| L22D329|q3 | 16 | 0.1250 | 1.00 | 0.0000 | True |
| L22D329|q4 | 47 | 0.0000 | 0.00 | 0.2500 | False |
| L22D329|q5 | 27 | 0.0000 | 0.00 | 0.1250 | False |
| L22D329|q7 | 13 | 0.0000 | 0.00 | 0.0625 | False |
| L22D329|q8 | 21 | 0.0000 | 0.00 | 0.1250 | False |
| L23D390|q1 | 36 | 0.1250 | 1.00 | 0.0000 | True |
| L23D390|q11 | 24 | 0.0000 | 0.00 | 0.0625 | False |
| L23D390|q12 | 7 | 0.1250 | 1.00 | 0.0000 | True |
| L23D390|q13 | 10 | 0.0000 | 0.00 | 0.1250 | False |
| L23D390|q14 | 23 | 0.0000 | 0.00 | 0.1250 | False |
| L23D390|q2 | 15 | 0.0000 | 0.00 | 0.1250 | False |
| L23D390|q3 | 21 | 0.0000 | 0.00 | 0.1250 | False |
| L23D390|q5 | 13 | 0.1250 | 1.00 | 0.0000 | True |
| L23D390|q6 | 15 | 0.0625 | 1.00 | 0.0000 | True |
| L23N8582|q11 | 7 | 0.0000 | 0.00 | 0.1250 | False |
| L23N8582|q14 | 20 | 0.1250 | 1.00 | 0.0000 | True |
| L23N8582|q2 | 21 | 0.0000 | 0.00 | 0.1250 | False |
| L23N8582|q5 | 48 | 0.0625 | 1.00 | 0.0000 | True |
| L23N8582|q6 | 13 | 0.0000 | 0.00 | 0.0625 | False |
| L23N8582|q8 | 15 | 0.1250 | 1.00 | 0.0000 | True |
| L23N8582|q9 | 27 | 0.1250 | 1.00 | 0.0000 | True |
| L24D116|q1 | 8 | 0.1250 | 1.00 | 0.0000 | True |
| L24D116|q10 | 20 | 0.0000 | 0.00 | 0.1250 | False |
| L24D116|q11 | 13 | 0.0000 | 0.00 | 0.1250 | False |
| L24D116|q14 | 43 | 0.0000 | 0.00 | 0.1250 | False |
| L24D116|q15 | 31 | 0.0000 | 0.00 | 0.0625 | False |
| L24D116|q4 | 26 | 0.1250 | 1.00 | 0.0000 | True |
| L24D116|q8 | 63 | 0.0000 | 0.00 | 0.1250 | False |
| L24D116|q9 | 28 | 0.1250 | 1.00 | 0.0000 | True |
| L24D4|q11 | 5 | 0.1250 | 2.00 | 0.0625 | True |
| L24D4|q12 | 7 | 0.0000 | 0.00 | 0.1250 | False |
| L24D4|q14 | 11 | 0.1250 | 1.00 | 0.0000 | True |
| L24D4|q15 | 9 | 0.0625 | 1.00 | 0.0625 | True |
| L24D4|q3 | 23 | 0.1250 | 1.00 | 0.0000 | False |
| L24D4|q4 | 26 | 0.0000 | 0.00 | 0.1250 | False |
| L24D4|q9 | 79 | 0.1250 | 1.00 | 0.0000 | True |
| L24N1813|q15 | 7 | 0.0625 | 1.00 | 0.0625 | True |
| L24N1813|q3 | 9 | 0.0000 | 0.00 | 0.1250 | False |
| L25D13|q0 | 22 | 0.0000 | 0.00 | 0.1250 | False |
| L25D13|q11 | 34 | 0.0000 | 0.00 | 0.1250 | False |
| L25D13|q14 | 45 | 0.0000 | 0.00 | 0.1250 | False |
| L25D13|q15 | 10 | 0.2500 | 2.00 | 0.0000 | True |
| L25D13|q2 | 69 | 0.1250 | 1.00 | 0.0000 | True |
| L25D13|q3 | 10 | 0.0000 | 0.00 | 0.1250 | False |
| L25D13|q4 | 29 | 0.0000 | 0.00 | 0.1250 | False |
| L25D13|q5 | 9 | 0.0000 | 0.00 | 0.1250 | False |
| L25D13|q7 | 29 | 0.1250 | 1.00 | 0.0000 | True |
| L25N7804|q10 | 45 | 0.0000 | 0.00 | 0.1250 | False |
| L25N7804|q14 | 24 | 0.0000 | 0.00 | 0.1250 | False |
| L25N7804|q15 | 28 | 0.1875 | 3.00 | 0.0000 | True |
| L25N7804|q2 | 20 | 0.2500 | 2.00 | 0.0000 | True |
| L25N7804|q4 | 22 | 0.0000 | 0.00 | 0.1250 | False |
| L25N7804|q6 | 28 | 0.0000 | 0.00 | 0.1250 | False |
| L25N7804|q9 | 57 | 0.1250 | 1.00 | 0.0000 | True |
| L26D365|q11 | 17 | 0.0000 | 0.00 | 0.1250 | False |
| L26D365|q12 | 15 | 0.0000 | 0.00 | 0.2500 | False |
| L26D365|q15 | 17 | 0.0000 | 0.00 | 0.1250 | False |
| L26D365|q3 | 30 | 0.0000 | 0.00 | 0.0625 | False |
| L26D365|q8 | 72 | 0.1250 | 1.00 | 0.0000 | True |
| L26D365|q9 | 48 | 0.1875 | 3.00 | 0.0000 | True |
| L26N8926|q0 | 18 | 0.0625 | 1.00 | 0.0000 | True |
| L26N8926|q1 | 23 | 0.1250 | 1.00 | 0.0000 | True |
| L26N8926|q11 | 76 | 0.1250 | 1.00 | 0.0000 | True |
| L26N8926|q12 | 8 | 0.1250 | 1.00 | 0.0000 | True |
| L26N8926|q14 | 69 | 0.1250 | 1.00 | 0.0000 | True |
| L26N8926|q3 | 27 | 0.1250 | 1.00 | 0.0000 | True |
| L26N8926|q4 | 8 | 0.0000 | 0.00 | 0.1250 | False |
| L26N8926|q7 | 77 | 0.0000 | 0.00 | 0.1250 | False |

### A/F8: 58 of 96 rows; positions `{"0": 16, "2": 1, "4": 1, "5": 2, "7": 1, "8": 1, "10": 1, "12": 3, "15": 2, "16": 2, "18": 1, "20": 3, "21": 3, "23": 1, "24": 1, "25": 1, "26": 1, "27": 2, "29": 2, "30": 1, "32": 1, "38": 1, "39": 1, "41": 1, "42": 1, "44": 1, "45": 1, "47": 1, "53": 1, "59": 1, "65": 1, "66": 1}`

| Row | Position | First pass margin | Steps | Second pass margin | Other token is the runner-up |
|---|---:|---:|---:|---:|---|
| L19N5359|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L19N5359|q1 | 21 | 0.0000 | 0.00 | 0.0307 | False |
| L19N5359|q2 | 30 | 0.1250 | 1.00 | 0.0003 | True |
| L19N5359|q8 | 15 | 0.0000 | 0.00 | 0.0313 | False |
| L19N5359|q9 | 59 | 0.1250 | 1.00 | 0.0134 | True |
| L21N5109|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L21N5109|q1 | 18 | 0.1250 | 2.00 | 0.1106 | True |
| L21N5109|q10 | 5 | 0.1250 | 2.00 | 0.1684 | True |
| L21N5109|q2 | 65 | 0.0000 | 0.00 | 0.0036 | False |
| L21N5109|q9 | 7 | 0.0625 | 1.00 | 0.2319 | True |
| L22D329|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L22D329|q2 | 41 | 0.0000 | 0.00 | 0.0215 | False |
| L22D329|q8 | 21 | 0.0000 | 0.00 | 0.1757 | False |
| L22D329|q9 | 47 | 0.0000 | 0.00 | 0.0638 | False |
| L23D390|q0 | 0 | 0.0000 | 0.00 | 0.0159 | False |
| L23D390|q2 | 15 | 0.0000 | 0.00 | 0.1153 | False |
| L23D497|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L23D497|q1 | 16 | 0.0000 | 0.00 | 0.1149 | False |
| L23D497|q10 | 5 | 0.0000 | 0.00 | 0.0791 | False |
| L23D497|q9 | 39 | 0.0625 | 1.00 | 0.0248 | True |
| L23N306|q0 | 0 | 0.0000 | 0.00 | 0.0159 | False |
| L23N306|q10 | 10 | 0.0000 | 0.00 | 0.0370 | False |
| L23N306|q9 | 20 | 0.0000 | 0.00 | 0.0181 | False |
| L23N8582|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L23N8582|q2 | 21 | 0.0000 | 0.00 | 0.0408 | False |
| L24D116|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L24D116|q1 | 44 | 0.0000 | 0.00 | 0.0564 | False |
| L24D116|q10 | 20 | 0.0000 | 0.00 | 0.0530 | False |
| L24D116|q2 | 29 | 0.0000 | 0.00 | 0.0141 | False |
| L24D116|q8 | 38 | 0.0000 | 0.00 | 0.0324 | False |
| L24D116|q9 | 66 | 0.0000 | 0.00 | 0.0765 | False |
| L24D4|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L24D4|q2 | 25 | 0.0000 | 0.00 | 0.0114 | False |
| L24D4|q8 | 42 | 0.0000 | 0.00 | 0.1031 | False |
| L24N1813|q0 | 0 | 0.0000 | 0.00 | 0.0159 | False |
| L24N1813|q1 | 12 | 0.0625 | 1.00 | 0.0080 | True |
| L25D13|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L25D13|q1 | 12 | 0.1250 | 2.00 | 0.0536 | True |
| L25D13|q2 | 24 | 0.0000 | 0.00 | 0.0045 | False |
| L25D13|q9 | 27 | 0.0000 | 0.00 | 0.0337 | False |
| L25D80|q0 | 0 | 0.0000 | 0.00 | 0.0159 | False |
| L25D80|q1 | 20 | 0.0000 | 0.00 | 0.0233 | False |
| L25D80|q2 | 4 | 0.0000 | 0.00 | 0.1102 | False |
| L25D80|q8 | 32 | 0.0000 | 0.00 | 0.0344 | False |
| L25N7804|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L25N7804|q10 | 16 | 0.0000 | 0.00 | 0.0716 | False |
| L25N7804|q9 | 26 | 0.1250 | 1.00 | 0.0009 | True |
| L26D365|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L26D365|q1 | 45 | 0.0000 | 0.00 | 0.0856 | False |
| L26D365|q10 | 27 | 0.0000 | 0.00 | 0.0802 | False |
| L26D365|q8 | 12 | 0.1250 | 1.00 | 0.0070 | True |
| L26D365|q9 | 53 | 0.0000 | 0.00 | 0.0638 | False |
| L26N2572|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L26N2572|q10 | 2 | 0.0000 | 0.00 | 0.0399 | False |
| L26N2572|q9 | 8 | 0.0000 | 0.00 | 0.1790 | False |
| L26N8926|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L26N8926|q10 | 29 | 0.0000 | 0.00 | 0.0828 | False |
| L26N8926|q2 | 23 | 0.0000 | 0.00 | 0.0474 | False |

### C/F1: 31 of 48 rows; positions `{"0": 8, "5": 1, "7": 1, "8": 1, "9": 1, "12": 3, "15": 1, "16": 1, "18": 1, "20": 1, "24": 1, "26": 1, "27": 3, "28": 1, "29": 1, "36": 1, "38": 1, "45": 1, "48": 1, "77": 1}`

| Row | Position | First pass margin | Steps | Second pass margin | Other token is the runner-up |
|---|---:|---:|---:|---:|---|
| L21N5109|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L21N5109|q1 | 18 | 0.1250 | 2.00 | 0.1107 | True |
| L21N5109|q10 | 5 | 0.1250 | 1.00 | 0.1685 | True |
| L21N5109|q2 | 9 | 0.0000 | 0.00 | 0.0671 | False |
| L21N5109|q9 | 7 | 0.0625 | 1.00 | 0.2319 | True |
| L23D390|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L23D390|q1 | 36 | 0.0000 | 0.00 | 0.0896 | False |
| L23D390|q2 | 77 | 0.1250 | 1.00 | 0.0053 | True |
| L23N8582|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L23N8582|q8 | 15 | 0.0000 | 0.00 | 0.0212 | False |
| L23N8582|q9 | 27 | 0.0000 | 0.00 | 0.0681 | False |
| L24D116|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L24D116|q1 | 8 | 0.0000 | 0.00 | 0.1039 | False |
| L24D116|q2 | 29 | 0.0000 | 0.00 | 0.0141 | False |
| L24D116|q8 | 38 | 0.0000 | 0.00 | 0.0325 | False |
| L24D116|q9 | 28 | 0.0000 | 0.00 | 0.1126 | False |
| L24N1813|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L24N1813|q1 | 12 | 0.0625 | 1.00 | 0.0080 | True |
| L25D13|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L25D13|q1 | 12 | 0.1250 | 2.00 | 0.0536 | True |
| L25D13|q2 | 24 | 0.0000 | 0.00 | 0.0046 | False |
| L25D13|q9 | 27 | 0.0000 | 0.00 | 0.0337 | False |
| L25N7804|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L25N7804|q10 | 16 | 0.0000 | 0.00 | 0.0716 | False |
| L25N7804|q2 | 20 | 0.0000 | 0.00 | 0.1420 | False |
| L25N7804|q9 | 26 | 0.0000 | 0.00 | 0.0010 | False |
| L26D365|q0 | 0 | 0.0000 | 0.00 | 0.0160 | False |
| L26D365|q1 | 45 | 0.1250 | 1.00 | 0.0856 | True |
| L26D365|q10 | 27 | 0.0000 | 0.00 | 0.0802 | False |
| L26D365|q8 | 12 | 0.1250 | 1.00 | 0.0069 | True |
| L26D365|q9 | 48 | 0.0000 | 0.00 | 0.0897 | False |

## Pass A against Part 200 (exploratory)

`{"comparison": "decoded answer text (Part 200 stored no token ids); another process and other neighbours", "compared": 384, "differing": 0, "label_flips": {"name_gone": {"a_true_b_false": 0, "a_false_b_true": 0, "n": 384}, "maker_gone": {"a_true_b_false": 0, "a_false_b_true": 0, "n": 384}, "broken": {"a_true_b_false": 0, "a_false_b_true": 0, "n": 384}, "qwen_part161_rule": {"a_true_b_false": 0, "a_false_b_true": 0, "n": 384}}}`

Rows with a non-finite score: none.
