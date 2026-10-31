# Part 154 score

## Per unit maker company hits

| Unit | Company | None | L27 N3325 | Matched 0 | Matched 1 | Matched 2 | Matched 3 |
|---|---|---:|---:|---:|---:|---:|---:|
| L11N5647 | Google | 21/24 | 9/24 | 20/24 | 20/24 | 20/24 | 19/24 |
| L21N7134 | Baidu | 16/24 | 16/24 | 15/24 | 8/24 | 15/24 | 14/24 |
| L22N6997 | Google | 8/24 | 0/24 | 9/24 | 5/24 | 10/24 | 6/24 |
| L23N414 | Google | 20/24 | 9/24 | 20/24 | 20/24 | 21/24 | 20/24 |
| L23N1743 | Huawei | 17/24 | 18/24 | 15/24 | 17/24 | 16/24 | 19/24 |
| L23N2993 | Microsoft | 20/24 | 18/24 | 17/24 | 21/24 | 19/24 | 14/24 |
| L24N5017 | Google | 17/24 | 10/24 | 18/24 | 19/24 | 19/24 | 20/24 |
| L25N140 | Google | 8/24 | 4/24 | 8/24 | 7/24 | 10/24 | 8/24 |
| L25N4457 | Google | 4/24 | 0/24 | 3/24 | 4/24 | 4/24 | 5/24 |
| L25N4458 | Google | 18/24 | 14/24 | 18/24 | 18/24 | 18/24 | 19/24 |

## Per zero set

| Zero set | Google hits | Other maker hits | Random direction Google hits |
|---|---:|---:|---:|
| none | 96/168 | 53/72 | 19/24 |
| L27N3325 | 46/168 | 52/72 | 14/24 |
| matched_0 | 96/168 | 47/72 | 19/24 |
| matched_1 | 93/168 | 46/72 | 20/24 |
| matched_2 | 102/168 | 50/72 | 19/24 |
| matched_3 | 97/168 | 47/72 | 19/24 |

## Gates

- gate1_rows_and_zero_trace: passed ({"passed": true, "expected_answers": 1872, "answers": 1872, "unique_answers": 1872, "nonempty_answers": 1872, "expected_stageA": 576, "stageA_rows": 576, "stageA_finite": true, "nonfinite_rows_kept": 0, "trace": {"calls": 2876, "positions": 3839, "target_checks": 139584, "target_positions": 171492, "unchanged_checks": 139584, "unchanged_positions": 171492, "unzeroed_target_checks": 34512, "unzeroed_target_positions": 46068, "target_failures": 0, "unchanged_failures": 0, "unzeroed_target_failures": 0}})
- gate2_baseline_and_positive_control: passed ({"passed": true, "plain_google_hits": 0, "plain_max": 2, "ten_neuron_company_hits": 149, "ten_neuron_range": [128, 174]})
- gate3_opening_zero_effect: passed ({"passed": true, "lowered_phrasings": 24, "minimum": 20})

## Predictions

- 154.1: held
- 154.2: held
- 154.3: held
- 154.4: failed
- 154.5: failed
- 154.6: held
- 154.7: held

## Broken answers

{
  "counts": {
    "none": 19,
    "L27N3325": 18
  },
  "shares": {
    "none": 0.07196969696969698,
    "L27N3325": 0.06818181818181818
  },
  "delta_percentage_points": -0.37878787878787984
}

## Stage A medians

{
  "median_google_minus_alibaba_no_push": {
    "none": -0.4375,
    "L27N3325": -3.34375
  },
  "median_activation_by_zero_and_push": {
    "none": {
      "L11N5647": -4.09375,
      "L21N7134": -0.9140625,
      "L22N6997": -3.5546875,
      "L23N414": -2.703125,
      "L23N1743": -0.990234375,
      "L23N2993": -2.1484375,
      "L24N5017": -3.8359375,
      "L25N140": -3.6015625,
      "L25N4457": -2.40625,
      "L25N4458": -2.46875,
      "random_L25N4457": -1.1484375,
      "none": -3.65625
    },
    "L27N3325": {
      "L11N5647": -4.09375,
      "L21N7134": -0.9140625,
      "L22N6997": -3.5546875,
      "L23N414": -2.703125,
      "L23N1743": -0.990234375,
      "L23N2993": -2.1484375,
      "L24N5017": -3.8359375,
      "L25N140": -3.6015625,
      "L25N4457": -2.40625,
      "L25N4458": -2.46875,
      "random_L25N4457": -1.1484375,
      "none": -3.65625
    }
  },
  "median_activation_move_by_unit": {
    "L25N4457": 0.98828125,
    "L25N140": -0.078125,
    "L23N414": 0.9609375,
    "L11N5647": -0.5859375,
    "L22N6997": 0.0703125,
    "L24N5017": -0.328125,
    "L25N4458": 1.0390625,
    "L21N7134": 2.74261474609375,
    "L23N1743": 2.59765625,
    "L23N2993": 1.4375
  },
  "non_google_most_negative_move": 1.4375,
  "google_units_exceeding_move": 3
}
