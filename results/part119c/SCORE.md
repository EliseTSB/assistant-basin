# Part 119c score

Selected Path 2 rows: 312 ({'real': 156, 'control': 156}).

## Gates

| Gate | Passed | Median relative nPair difference |
| --- | --- | ---: |
| gate1 | True | 0 |
| gate1 at 0.2 | True | 0 |
| gate1 at 0.4 | True | 0 |
| gate1 at 0.8 | True | 0 |
| gate2 | True | 0.0035086 |
| gate2 at 0.8 | True | 0.0035086 |

## Predictions

| Prediction | Verdict | Observed |
| --- | --- | --- |
| 119c.1 | holds | `{"bf16_median_row_share": 0.1616346660348983, "bootstrap": {"ci95_product_weighted_difference_of_median_row_shares": [-0.021297478087482197, 0.012508900930312695], "ci95_sum_weighted_difference_of_median_row_shares": [-0.014342926597370853, 0.009602651155421332], "difference_of_median_row_shares": -0.008220480630778182, "draws": 2000, "rows": 156, "seed": 11902, "units": 39}, "float32_median_row_share": 0.1534141854041201, "median_difference_float32_minus_bf16": -0.008220480630778182, "paired_rows": 156}` |
| 119c.2 | holds | `{"float32_real_median_row_share_0.05": 0.00454674751900166, "float32_real_median_row_share_0.1": 0.020093756702493668}` |
| 119c.3 | holds | `{"doses": [0.05, 0.2], "log_log_slope": 2.5382265919006235, "median_row_shares": [0.00454674751900166, 0.1534141854041201]}` |
| 119c.4 | holds | `{"dose_comparisons": [{"control_median_row_share": 0.0046077286470220915, "dose": 0.05, "larger_to_smaller_factor": 1.013412033055625, "real_median_row_share": 0.00454674751900166, "within_factor_1.25": true}, {"control_median_row_share": 0.020812549637322132, "dose": 0.1, "larger_to_smaller_factor": 1.0357719537202947, "real_median_row_share": 0.020093756702493668, "within_factor_1.25": true}, {"control_median_row_share": 0.13914282287894295, "dose": 0.2, "larger_to_smaller_factor": 1.1025662857048224, "real_median_row_share": 0.1534141854041201, "within_factor_1.25": true}]}` |

## Per-dose summaries

Share uses positions with nPair > 0; row share is the median of a row's eligible positions; the summary is the median over rows. The nPair<1 share uses all reported positions.

| Dtype | Dose | Class | Rows | Positions nPair>0 | Positions nPair<1 | Share nPair<1 | Median nI | Median nPair | Median nA | Median n2 | Position share | Row share |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 0.05 | real | 156 | 2056 | 1115 | 0.54232 | 0.23457 | 0.9487 | 0.67556 | 0.5942 | 0.061394 | 0.073401 |
| bf16 | 0.05 | control | 156 | 2056 | 1240 | 0.60311 | 0.22915 | 0.88738 | 0.66356 | 0.54967 | 0.066319 | 0.082293 |
| bf16 | 0.1 | real | 156 | 2056 | 99 | 0.048152 | 0.35574 | 2.0112 | 1.4043 | 1.2235 | 0.033926 | 0.039749 |
| bf16 | 0.1 | control | 156 | 2056 | 146 | 0.071012 | 0.33602 | 1.8458 | 1.3827 | 1.0763 | 0.035419 | 0.042234 |
| bf16 | 0.2 | real | 156 | 2056 | 0 | 0 | 1.8037 | 5.0256 | 3.3449 | 2.9642 | 0.13204 | 0.16163 |
| bf16 | 0.2 | control | 156 | 2056 | 0 | 0 | 1.5092 | 4.4289 | 3.2941 | 2.3335 | 0.12166 | 0.1414 |
| bf16 | 0.4 | real | 156 | 2056 | 0 | 0 | 6.0379 | 11.202 | 7.796 | 7.3172 | 0.26467 | 0.30449 |
| bf16 | 0.4 | control | 156 | 2056 | 0 | 0 | 5.2477 | 9.9186 | 7.6535 | 5.5886 | 0.2754 | 0.31846 |
| bf16 | 0.8 | real | 156 | 2056 | 0 | 0 | 14.097 | 22.66 | 15.494 | 17.349 | 0.40259 | 0.44063 |
| bf16 | 0.8 | control | 156 | 2056 | 0 | 0 | 11.378 | 19.957 | 15.376 | 11.667 | 0.32893 | 0.35077 |
| float32 | 0.05 | real | 156 | 2056 | 1174 | 0.57101 | 0.057415 | 0.92944 | 0.6564 | 0.56652 | 0.0038918 | 0.0045467 |
| float32 | 0.05 | control | 156 | 2056 | 1277 | 0.62111 | 0.053135 | 0.85918 | 0.64841 | 0.51693 | 0.003783 | 0.0046077 |
| float32 | 0.1 | real | 156 | 2056 | 102 | 0.049611 | 0.25477 | 1.9927 | 1.3829 | 1.2047 | 0.017711 | 0.020094 |
| float32 | 0.1 | control | 156 | 2056 | 160 | 0.077821 | 0.23108 | 1.8385 | 1.3753 | 1.0581 | 0.017217 | 0.020813 |
| float32 | 0.2 | real | 156 | 2056 | 0 | 0 | 1.7799 | 5.0188 | 3.34 | 2.944 | 0.12842 | 0.15341 |
| float32 | 0.2 | control | 156 | 2056 | 0 | 0 | 1.4799 | 4.4314 | 3.2785 | 2.3348 | 0.11684 | 0.13914 |
| float32 | 0.4 | real | 156 | 2056 | 0 | 0 | 6.0218 | 11.236 | 7.8426 | 7.3043 | 0.2684 | 0.29487 |
| float32 | 0.4 | control | 156 | 2056 | 0 | 0 | 5.2264 | 9.9316 | 7.6633 | 5.6255 | 0.27463 | 0.3224 |
| float32 | 0.8 | real | 156 | 2056 | 0 | 0 | 14.114 | 22.583 | 15.488 | 17.376 | 0.40266 | 0.44314 |
| float32 | 0.8 | control | 156 | 2056 | 0 | 0 | 11.382 | 19.999 | 15.422 | 11.704 | 0.32702 | 0.35559 |

## Headline

Part 119b's low-dose share is not a rounding floor; the interaction shrinks toward zero with dose, and at 0.2 it is already measurable where answers do not break.

Share includes positions with nPair > 0; each row is the median of those positions; summary is the median of row values.
Part 119 unit bootstrap (2,000 draws, seed 11902); each draw recomputes the difference of the two median row shares. Product weighting is primary and sum weighting is the sensitivity interval.
