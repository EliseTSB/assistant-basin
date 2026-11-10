# Part 201 score

## Gates

| Gate | Result | Detail |
|---|---|---|
| G1 | FAILED | bf16 passed, float32 passed, kernel record absent; revisions ['851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a']; snapshot loaded is the recorded revision: not checked (smoke plan) (loaded ['2fc06364715b967f1860aea9cf38778875588b17']) |
| G2 | passed | 0 of 116 rows failed; 45 positions emitted a tied token that is not the first-listed top |
| G3 | passed | plan passed, bf16 passed, float32 passed;  |
| G4 | passed | bf16 passed, float32 passed; 0 of 108 pushed rows failed; 0 calls without matching hook evidence; 0 rows with more than one push hash in bf16, 0 whose float32 push hash is not the bf16 one; inputs bound across launches: bf16 True, float32 True |
| G5 | passed | 1 launch records |

Positive control: A and C differ on 4 rows (needs 10): passed (not applied to a smoke plan).
Run state: smoke plan True; float32 cut applied False; A-again in pass A's process True.

## Registered predictions

| # | Prediction (verbatim) | Confidence | Verdict |
|---|---|---|---|
| 201.1 | A-again equals A on all 16 rows, and A2 equals A on all 96, token for token. | high | held |
| 201.2 | Between A and C, 25% to 50% of the 192 rows differ. | medium | held |
| 201.3 | Between A and M, at least 20% of the 384 rows differ. | low | failed |
| 201.4 | Between A and B (other slots and other neighbours, same size and prompts), at most 10% of the 384 rows differ. | low | failed |
| 201.5 | Of the rows that differ between A and C, at least 90% have a pass-A margin of at most 2 steps at the first differing position, and none has one above 8 steps. | high | held |
| 201.6 | Between A and M, for each of the four labels with at least 8 flips, the two directions of flip differ by at most twice the square root of the number of flips; and mean answer length differs by under 3 tokens. | medium | held (length clause only) |
| 201.7 | K matches the CPU reference token for token on all 16 rows, over every position. | medium | not run |
| 201.8 | Between F8 and F1, none of the 48 rows differ, with all 48 present and finite in both. | medium | held |
| 201.9 | Of the rows that differ between A and F8, at least 90% have a pass-A margin of at most 2 steps at the first differing position, and none has an F8 (float32) margin above 0.5 there. | medium | held |
| 201.10 | Between A and F8, 30% to 70% of the 96 rows differ. | low | held |
| 201.11 | The projected slowdown of margin-gated decoding on pass A is at most 2: (pass A's seconds + the summed cost of one full-prefix float32 forward for the flagged rows at every step where an unfinished row is within 2 steps) ÷ pass A's seconds, the cost of each forward read from the timing probe at the nearest larger row count and prefix length. | low | failed |

### Estimates behind the verdicts

- **201.1**: `{"A-again": {"differing": 0, "n": 4, "planned": 4, "clause": "held"}, "A2": {"differing": 0, "n": 8, "planned": 8, "clause": "held"}}`
- **201.2**: `{"differing": 4, "n": 16, "planned": 16, "share": 0.25, "wilson95": [0.10182067491213048, 0.49498316535508774]}`
- **201.3**: `{"differing": 3, "n": 16, "planned": 16, "share": 0.1875, "wilson95": [0.06591599071428142, 0.4300888096197414]}`
- **201.4**: `{"differing": 2, "n": 16, "planned": 16, "share": 0.125, "wilson95": [0.03497748774324047, 0.36022827265758695]}`
- **201.5**: `{"differing": 4, "with_a_margin": 4, "rows_without_a_recorded_margin": [], "caught": 4, "caught_share": 1.0, "largest_pass_A_steps": 1.0}`
- **201.6**: `{"labels": {"name_gone": {"a_true_b_false": 1, "a_false_b_true": 0, "flips": 1, "direction_clause": "not scored (1 flips)"}, "maker_gone": {"a_true_b_false": 1, "a_false_b_true": 0, "flips": 1, "direction_clause": "not scored (1 flips)"}, "broken": {"a_true_b_false": 0, "a_false_b_true": 0, "flips": 0, "direction_clause": "not scored (0 flips)"}, "qwen_part161_rule": {"a_true_b_false": 0, "a_false_b_true": 0, "flips": 0, "direction_clause": "not scored (0 flips)"}}, "mean_length_difference_a_minus_m": 0.0, "length_clause": "held"}`
- **201.7**: `{"1": {"rows": 0, "equal_rows": 0, "first_differences": []}, "2": {"rows": 0, "equal_rows": 0, "first_differences": []}}`
- **201.8**: `{"differing": 0, "n": 16, "planned": 16, "nonfinite": 0}`
- **201.9**: `{"differing": 6, "with_a_margin": 6, "rows_without_a_recorded_margin": [], "caught": 6, "caught_share": 1.0, "largest_pass_A_steps": 0.0, "largest_F8_margin": 0.1259326934814453}`
- **201.10**: `{"differing": 6, "n": 16, "planned": 16, "share": 0.375, "wilson95": [0.18481232558863633, 0.6135895945449727]}`
- **201.11**: `{"pass_A_seconds": 13.781274306005798, "by_margin_steps": {"1": {"margin_steps": 1, "flagged_steps": 24, "added_seconds": 21.43173018623808, "projected_slowdown": 2.5551341414703765, "flagged_steps_priced_by_extrapolation_beyond_the_probe": 19, "flagged_steps_the_probe_cannot_price": 0}, "2": {"margin_steps": 2, "flagged_steps": 37, "added_seconds": 34.63455701424998, "projected_slowdown": 3.5131607023565623, "flagged_steps_priced_by_extrapolation_beyond_the_probe": 31, "flagged_steps_the_probe_cannot_price": 0}, "4": {"margin_steps": 4, "flagged_steps": 51, "added_seconds": 51.699359620644834, "projected_slowdown": 4.751420839081227, "flagged_steps_priced_by_extrapolation_beyond_the_probe": 44, "flagged_steps_the_probe_cannot_price": 0}}}`

## Ordered decision (BATCH-SHAPE-NOISE.md, section 3)

Rule 4, No build: section 1 is the standing procedure.

Readings: 201.7: not run; 201.11: failed.

Candidate gate: 1 steps (the largest pass-A margin, in bf16 steps, at a first difference between A and F8, rounded up to a whole step, plus one). Cost at that gate: `{"margin_steps": 1, "flagged_steps": 24, "added_seconds": 21.43173018623808, "projected_slowdown": 2.5551341414703765, "flagged_steps_priced_by_extrapolation_beyond_the_probe": 19, "flagged_steps_the_probe_cannot_price": 0}`. A price above the longest probed prefix is a linear extrapolation, an approximation and not a measured bound.

## Pairs

| Pair | Planned | Compared | Differ | Share (95% Wilson) | Not finite | Mean length A minus B | Other token not the runner-up |
|---|---:|---:|---:|---|---:|---:|---:|
| A/A-again | 4 | 4 | 0 | 0.000 (0.000 to 0.490) | 0 | 0.00 | 0 of 0 |
| A/A2 | 8 | 8 | 0 | 0.000 (0.000 to 0.324) | 0 | 0.00 | 0 of 0 |
| A/B | 16 | 16 | 2 | 0.125 (0.035 to 0.360) | 0 | 0.00 | 1 of 2 |
| A/M | 16 | 16 | 3 | 0.188 (0.066 to 0.430) | 0 | 0.00 | 1 of 3 |
| M/C (plain rows) | 4 | 4 | 1 | 0.250 (0.046 to 0.699) | 0 | 0.00 | 1 of 1 |
| A/C | 16 | 16 | 4 | 0.250 (0.102 to 0.495) | 0 | 0.00 | 2 of 4 |
| F8/F1 | 16 | 16 | 0 | 0.000 (0.000 to 0.194) | 0 | 0.00 | 0 of 0 |
| A/F8 | 16 | 16 | 6 | 0.375 (0.185 to 0.614) | 0 | 0.00 | 6 of 6 |
| C/F1 | 16 | 16 | 6 | 0.375 (0.185 to 0.614) | 0 | 0.00 | 4 of 6 |

## Label flips by pair and direction (count of n, 95% Wilson interval)

| Pair | Label | True in the first, false in the second | False in the first, true in the second | n |
|---|---|---|---|---:|
| A/A-again | name_gone | 0 (0 to 0.490) | 0 (0 to 0.490) | 4 |
| A/A-again | maker_gone | 0 (0 to 0.490) | 0 (0 to 0.490) | 4 |
| A/A-again | broken | 0 (0 to 0.490) | 0 (0 to 0.490) | 4 |
| A/A-again | qwen_part161_rule | 0 (0 to 0.490) | 0 (0 to 0.490) | 4 |
| A/A2 | name_gone | 0 (0 to 0.324) | 0 (0 to 0.324) | 8 |
| A/A2 | maker_gone | 0 (0 to 0.324) | 0 (0 to 0.324) | 8 |
| A/A2 | broken | 0 (0 to 0.324) | 0 (0 to 0.324) | 8 |
| A/A2 | qwen_part161_rule | 0 (0 to 0.324) | 0 (0 to 0.324) | 8 |
| A/B | name_gone | 1 (0.011 to 0.283) | 0 (0 to 0.194) | 16 |
| A/B | maker_gone | 1 (0.011 to 0.283) | 0 (0 to 0.194) | 16 |
| A/B | broken | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| A/B | qwen_part161_rule | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| A/M | name_gone | 1 (0.011 to 0.283) | 0 (0 to 0.194) | 16 |
| A/M | maker_gone | 1 (0.011 to 0.283) | 0 (0 to 0.194) | 16 |
| A/M | broken | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| A/M | qwen_part161_rule | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| M/C | name_gone | 0 (0 to 0.490) | 0 (0 to 0.490) | 4 |
| M/C | maker_gone | 0 (0 to 0.490) | 1 (0.046 to 0.699) | 4 |
| M/C | broken | 0 (0 to 0.490) | 0 (0 to 0.490) | 4 |
| M/C | qwen_part161_rule | 0 (0 to 0.490) | 0 (0 to 0.490) | 4 |
| A/C | name_gone | 1 (0.011 to 0.283) | 0 (0 to 0.194) | 16 |
| A/C | maker_gone | 1 (0.011 to 0.283) | 0 (0 to 0.194) | 16 |
| A/C | broken | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| A/C | qwen_part161_rule | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| F8/F1 | name_gone | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| F8/F1 | maker_gone | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| F8/F1 | broken | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| F8/F1 | qwen_part161_rule | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| A/F8 | name_gone | 1 (0.011 to 0.283) | 0 (0 to 0.194) | 16 |
| A/F8 | maker_gone | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| A/F8 | broken | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| A/F8 | qwen_part161_rule | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| C/F1 | name_gone | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| C/F1 | maker_gone | 0 (0 to 0.194) | 1 (0.011 to 0.283) | 16 |
| C/F1 | broken | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |
| C/F1 | qwen_part161_rule | 0 (0 to 0.194) | 0 (0 to 0.194) | 16 |

## Margins (bf16 rows)

Share of positions within 0, 1, 2 and 4 steps, over 84 bf16 decodes (a row decoded in several passes counts once per pass): `{"0": 0.022321428571428572, "1": 0.07142857142857142, "2": 0.11458333333333333, "4": 0.18551587301587302}`.

Pass M padding: `{"rows": 20, "rows_left_padded": 4, "median_pad_tokens_of_those": 6.0, "calls": 16, "calls_with_unequal_prompt_lengths": 2}`.

Share of generation steps with an unfinished row within 2 steps, by pass: `{"A": 0.3854166666666667, "B": 0.22395833333333334, "M": 0.15104166666666666, "C": 0.11875}`.

Exact ties: `{"exact_ties": 45, "emitted_the_lower_token_id": 45, "same_tie_different_emitted_token_by_pair": {"A/A-again": 0, "A/A2": 0, "A/B": 0, "A/M": 0, "M/C": 0, "A/C": 0}}`.

## First differences, row by row

### A/B: 2 of 16 rows; positions `{"15": 1, "18": 1}`

| Row | Position | First pass margin | Steps | Second pass margin | Other token is the runner-up |
|---|---:|---:|---:|---:|---|
| L12N137|q8 | 15 | 0.1250 | 1.00 | 0.0000 | True |
| L12N137|q9 | 18 | 0.0000 | 0.00 | 0.1250 | False |

### A/M: 3 of 16 rows; positions `{"11": 1, "15": 1, "18": 1}`

| Row | Position | First pass margin | Steps | Second pass margin | Other token is the runner-up |
|---|---:|---:|---:|---:|---|
| L12N137|q8 | 15 | 0.1250 | 1.00 | 0.0000 | True |
| L12N137|q9 | 18 | 0.0000 | 0.00 | 0.1250 | False |
| L8D10|q8 | 11 | 0.1250 | 1.00 | 0.0000 | True |

### M/C: 1 of 4 rows; positions `{"2": 1}`

| Row | Position | First pass margin | Steps | Second pass margin | Other token is the runner-up |
|---|---:|---:|---:|---:|---|
| plain|q8 | 2 | 0.0000 | 0.00 | 0.1250 | False |

### A/C: 4 of 16 rows; positions `{"6": 1, "11": 1, "15": 1, "18": 1}`

| Row | Position | First pass margin | Steps | Second pass margin | Other token is the runner-up |
|---|---:|---:|---:|---:|---|
| L12N137|q8 | 15 | 0.1250 | 1.00 | 0.0000 | True |
| L12N137|q9 | 18 | 0.0000 | 0.00 | 0.1250 | False |
| L8D10|q8 | 11 | 0.1250 | 1.00 | 0.0000 | True |
| L8D10|q9 | 6 | 0.0000 | 0.00 | 0.1250 | False |

### A/F8: 6 of 16 rows; positions `{"6": 1, "7": 1, "12": 1, "13": 1, "18": 1, "20": 1}`

| Row | Position | First pass margin | Steps | Second pass margin | Other token is the runner-up |
|---|---:|---:|---:|---:|---|
| L12N137|q9 | 18 | 0.0000 | 0.00 | 0.1259 | False |
| L12N285|q1 | 20 | 0.0000 | 0.00 | 0.0235 | False |
| L12N285|q9 | 7 | 0.0000 | 0.00 | 0.0600 | False |
| L8D10|q9 | 6 | 0.0000 | 0.00 | 0.0457 | False |
| L8D6|q8 | 12 | 0.0000 | 0.00 | 0.0194 | False |
| L8D6|q9 | 13 | 0.0000 | 0.00 | 0.0288 | False |

### C/F1: 6 of 16 rows; positions `{"7": 1, "11": 1, "12": 1, "13": 1, "15": 1, "20": 1}`

| Row | Position | First pass margin | Steps | Second pass margin | Other token is the runner-up |
|---|---:|---:|---:|---:|---|
| L12N137|q8 | 15 | 0.0000 | 0.00 | 0.0519 | False |
| L12N285|q1 | 20 | 0.1250 | 1.00 | 0.0235 | True |
| L12N285|q9 | 7 | 0.0000 | 0.00 | 0.0600 | False |
| L8D10|q8 | 11 | 0.0000 | 0.00 | 0.0062 | False |
| L8D6|q8 | 12 | 0.0000 | 0.00 | 0.0194 | False |
| L8D6|q9 | 13 | 0.0625 | 1.00 | 0.0288 | True |

## Pass A against Part 200 (exploratory)

`{"comparison": "decoded answer text (Part 200 stored no token ids); another process and other neighbours", "compared": 0, "differing": 0, "label_flips": {"name_gone": {"a_true_b_false": 0, "a_false_b_true": 0, "n": 0}, "maker_gone": {"a_true_b_false": 0, "a_false_b_true": 0, "n": 0}, "broken": {"a_true_b_false": 0, "a_false_b_true": 0, "n": 0}, "qwen_part161_rule": {"a_true_b_false": 0, "a_false_b_true": 0, "n": 0}}}`

Rows with a non-finite score: none.
