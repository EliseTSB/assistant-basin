# Part 120 numbers

All statistics below are recomputed by `fig120.py` from the saved per-unit vector arrays. The script checks the recomputed score against the recorded `score120.json` before writing this file. The registered position rule takes medians over eligible positions within a row, then medians over rows. At dose 0.8 there is no 1.6 pass, so only the raw interaction norm is defined.

Raw vectors are in `results/part120/vectors/*.npz`: `index.doses`, `index.entries[].kind`, `first`, `second`, `pos`, `t_ids`, `start`, and `values` (A, B, P segments).
Derived row statistics are computed by `part120/score120.py`: `position_statistics`, `_row_statistics`, and `_summary_table`.

## Run scope and gates

| Item | Value | Source field |
| --- | ---: | --- |
| Rows | 312 (156 real, 156 control) | `timing.json:$.rows`; `score120.json:$.selected_rows_by_class` |
| First units | 39 | `timing.json:$.first_units` |
| Device and batch | xpu, 8 | `timing.json:$.device`, `$.batch` |
| Logical passes | 4680 | `timing.json:$.logical_passes` |
| Elapsed seconds | 174.759819 | `timing.json:$.elapsed_seconds` |
| Gate 1 | passed (39/39 paths) | `score120.json:$.gates.gate1`; `gate1.json:$.paths[]` |
| Gate 2 | passed; maximum cell median relative difference 4.23985693e-05 (strict limit 0.005) | `score120.json:$.gates.gate2.by_dose` values' `nA`, `n2`, `nPair`, and `nI` `median_relative_difference` fields |
| Gate 3 | passed; maximum adjacent-dose relative error 0 (limit 1e-5) | `score120.json:$.gates.gate3.maximum_relative_error`; `gate3.json:$.maximum_relative_error` |

## Dose table

Norms are medians of row medians. At 0.8, the self terms and q/c are n/a because the 1.6 pass was not run.

| Dose | Class | Rows | Positions considered | Eligible | Dropped | c | q | ‖I‖ | ‖S_A‖ | ‖S_B‖ | Source fields |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0125 | real | 156 | 2056 | 2056 | 0 | 0.27977549 | 0.0569946659 | 0.0032985441 | 0.0121778937 | 0.0111305486 | `results/part120/score120.json`: `$.per_dose[]` matching dose=0.0125, kind=real; fields `median_row_c`, `median_row_q`, `median_row_norm_I`, `median_row_norm_SA`, `median_row_norm_SB` |
| 0.0125 | control | 156 | 2056 | 2056 | 0 | 0.319769326 | 0.0532296791 | 0.00313276884 | 0.0119143889 | 0.00850136909 | `results/part120/score120.json`: `$.per_dose[]` matching dose=0.0125, kind=control; fields `median_row_c`, `median_row_q`, `median_row_norm_I`, `median_row_norm_SA`, `median_row_norm_SB` |
| 0.025 | real | 156 | 2056 | 2056 | 0 | 0.278299478 | 0.111029172 | 0.0134040055 | 0.0492693797 | 0.0452870031 | `results/part120/score120.json`: `$.per_dose[]` matching dose=0.025, kind=real; fields `median_row_c`, `median_row_q`, `median_row_norm_I`, `median_row_norm_SA`, `median_row_norm_SB` |
| 0.025 | control | 156 | 2056 | 2056 | 0 | 0.317795818 | 0.103948846 | 0.0126070726 | 0.0475004217 | 0.034519812 | `results/part120/score120.json`: `$.per_dose[]` matching dose=0.025, kind=control; fields `median_row_c`, `median_row_q`, `median_row_norm_I`, `median_row_norm_SA`, `median_row_norm_SB` |
| 0.05 | real | 156 | 2056 | 2056 | 0 | 0.274624884 | 0.232537718 | 0.0559928514 | 0.209959699 | 0.195764587 | `results/part120/score120.json`: `$.per_dose[]` matching dose=0.05, kind=real; fields `median_row_c`, `median_row_q`, `median_row_norm_I`, `median_row_norm_SA`, `median_row_norm_SB` |
| 0.05 | control | 156 | 2056 | 2056 | 0 | 0.306704739 | 0.211675024 | 0.0511154206 | 0.205293941 | 0.145585357 | `results/part120/score120.json`: `$.per_dose[]` matching dose=0.05, kind=control; fields `median_row_c`, `median_row_q`, `median_row_norm_I`, `median_row_norm_SA`, `median_row_norm_SB` |
| 0.1 | real | 156 | 2056 | 2056 | 0 | 0.223946546 | 0.600693632 | 0.250800193 | 1.17665654 | 1.07200915 | `results/part120/score120.json`: `$.per_dose[]` matching dose=0.1, kind=real; fields `median_row_c`, `median_row_q`, `median_row_norm_I`, `median_row_norm_SA`, `median_row_norm_SB` |
| 0.1 | control | 156 | 2056 | 2056 | 0 | 0.237666541 | 0.560704166 | 0.223585998 | 1.12148099 | 0.692138054 | `results/part120/score120.json`: `$.per_dose[]` matching dose=0.1, kind=control; fields `median_row_c`, `median_row_q`, `median_row_norm_I`, `median_row_norm_SA`, `median_row_norm_SB` |
| 0.2 | real | 156 | 2056 | 2056 | 0 | 0.355825993 | 1.04630184 | 1.92258903 | 5.66795012 | 5.16152875 | `results/part120/score120.json`: `$.per_dose[]` matching dose=0.2, kind=real; fields `median_row_c`, `median_row_q`, `median_row_norm_I`, `median_row_norm_SA`, `median_row_norm_SB` |
| 0.2 | control | 156 | 2056 | 2056 | 0 | 0.34433018 | 0.953673939 | 1.51935074 | 5.57889092 | 3.94774778 | `results/part120/score120.json`: `$.per_dose[]` matching dose=0.2, kind=control; fields `median_row_c`, `median_row_q`, `median_row_norm_I`, `median_row_norm_SA`, `median_row_norm_SB` |
| 0.4 | real | 156 | 2056 | 2056 | 0 | 0.5132688 | 1.58506601 | 6.62073267 | 11.9357119 | 12.6622959 | `results/part120/score120.json`: `$.per_dose[]` matching dose=0.4, kind=real; fields `median_row_c`, `median_row_q`, `median_row_norm_I`, `median_row_norm_SA`, `median_row_norm_SB` |
| 0.4 | control | 156 | 2056 | 2056 | 0 | 0.520979579 | 1.4478701 | 5.7527878 | 11.5784871 | 8.48149354 | `results/part120/score120.json`: `$.per_dose[]` matching dose=0.4, kind=control; fields `median_row_c`, `median_row_q`, `median_row_norm_I`, `median_row_norm_SA`, `median_row_norm_SB` |
| 0.8 | real | 156 | 2056 | n/a | n/a | n/a | n/a | 15.8606354 | n/a | n/a | `results/part120/score120.json`: `$.per_dose[]` matching dose=0.8, kind=real; fields `median_row_c`, `median_row_q`, `median_row_norm_I`, `median_row_norm_SA`, `median_row_norm_SB` |
| 0.8 | control | 156 | 2056 | n/a | n/a | n/a | n/a | 12.6470304 | n/a | n/a | `results/part120/score120.json`: `$.per_dose[]` matching dose=0.8, kind=control; fields `median_row_c`, `median_row_q`, `median_row_norm_I`, `median_row_norm_SA`, `median_row_norm_SB` |

## Bootstrap interval

At 0.1 on real rows, median c is 0.223946546. The product-weighted 95% unit-bootstrap interval is 0.161545398 to 0.262637284; the sum-weighted sensitivity interval is 0.199316175 to 0.241363268. There are 2000 draws with seed 12001, 39 units, and 156 real rows. Source fields in `results/part120/score120.json`: `$.bootstrap_c_0.1_real.point_median`, `ci95_product_weighted`, `ci95_sum_weighted`, `draws`, `seed`, `units`, and `rows`; recomputed from `results/part120/vectors/*.npz`: `index.doses`, `index.entries[].kind`, `first`, `second`, `pos`, `t_ids`, `start`, and `values` (A, B, P segments). by `score120.unit_bootstrap_interval`.

## Registered predictions

Prediction text is copied from the registration. Values and verdicts are recomputed by this script.

| # | Prediction (verbatim; author and confidence) | Result | Verdict | Source field |
| --- | --- | --- | --- | --- |
| 120.1 | Claude (medium confidence): In float32 on real rows, the median row ‖I‖ follows a square law at low dose: log2 of its ratio between adjacent doses is within 2 plus or minus 0.3 from 0.025 to 0.05 and from 0.05 to 0.1. | 0.025 to 0.05: log2 norm-I ratio 2.06257846; 0.05 to 0.1: log2 norm-I ratio 2.16322391 | held | `results/part120/score120.json`: `$.predictions[]` where `id=120.1`, fields `observed.adjacent_dose_ratios` and `verdict`; recomputed from `results/part120/vectors/*.npz`: `index.doses`, `index.entries[].kind`, `first`, `second`, `pos`, `t_ids`, `start`, and `values` (A, B, P segments). |
| 120.2 | Claude (medium confidence): The second-order identity holds at low dose and has broken by 0.2: on real rows the median row q is at most 0.30 at 0.05 and at least 0.50 at 0.2. | q at 0.05 0.232537718; q at 0.2 1.04630184 | held | `results/part120/score120.json`: `$.predictions[]` where `id=120.2`, fields `observed.real_median_row_q_0.05` and `observed.real_median_row_q_0.2` and `verdict`; recomputed from `results/part120/vectors/*.npz`: `index.doses`, `index.entries[].kind`, `first`, `second`, `pos`, `t_ids`, `start`, and `values` (A, B, P segments). |
| 120.3 | Claude (medium confidence): A push interacts with itself more than with a second push: at 0.1 the median row c over real rows is at most 0.75, and its product-weighted 95% unit-bootstrap interval lies below 1. | c at 0.1 0.223946546; product-weighted 95% interval 0.161545398 to 0.262637284 | held | `results/part120/score120.json`: `$.predictions[]` where `id=120.3`, fields `observed.real_median_row_c_0.1` and `observed.product_weighted_ci95` and `verdict`; recomputed from `results/part120/vectors/*.npz`: `index.doses`, `index.entries[].kind`, `first`, `second`, `pos`, `t_ids`, `start`, and `values` (A, B, P segments). |
| 120.4 | Claude (medium confidence): At each of 0.05, 0.1 and 0.2, the real median c is within a factor of 1.25 of the control median c. | 0.05: real c 0.274624884, control c 0.306704739, factor 1.11681336; 0.1: real c 0.223946546, control c 0.237666541, factor 1.0612646; 0.2: real c 0.355825993, control c 0.34433018, factor 1.03338602 | held | `results/part120/score120.json`: `$.predictions[]` where `id=120.4`, fields `observed.dose_comparisons[]` and `verdict`; recomputed from `results/part120/vectors/*.npz`: `index.doses`, `index.entries[].kind`, `first`, `second`, `pos`, `t_ids`, `start`, and `values` (A, B, P segments). |
| 120.5 | Claude (low confidence): At 0.4 on real rows, the median self share is at least half the median pair share and at most twice it: a push at 0.8 is about as far from twice itself as a pair at 0.4 is from the sum of its parts. | self share 0.481168579; pair share 0.29487051 | held | `results/part120/score120.json`: `$.predictions[]` where `id=120.5`, fields `observed.real_median_row_self_share_0.4` and `observed.real_median_row_pair_share_0.4` and `verdict`; recomputed from `results/part120/vectors/*.npz`: `index.doses`, `index.entries[].kind`, `first`, `second`, `pos`, `t_ids`, `start`, and `values` (A, B, P segments). |
| 120.6 | Claude (low confidence): At 0.4, over the 39 first units, u is higher for slot units than for concept units: the slot median is above the concept median, with a one-sided Mann-Whitney p below 0.05 on Part 92's Haiku kinds, and the same direction on the units both readers agree on. | Haiku slot median u 0.527612999 (n=21); concept 0.41601637 (n=18); one-sided p 0.073411076; agreed slot/concept 0.527612999/0.602731805 | failed | `results/part120/score120.json`: `$.predictions[]` where `id=120.6`, fields `observed` and `verdict`; recomputed from `results/part120/vectors/*.npz`: `index.doses`, `index.entries[].kind`, `first`, `second`, `pos`, `t_ids`, `start`, and `values` (A, B, P segments). |
| 120.7 | Claude (low confidence): Across the 39 units, the Spearman correlation of u(0.4) with Part 119's dominance is at most -0.3. | Spearman rho -0.0863578868 over 39 units | failed | `results/part120/score120.json`: `$.predictions[]` where `id=120.7`, fields `observed.spearman_rho_u_0.4_vs_dominance` and `observed.units` and `verdict`; recomputed from `results/part120/vectors/*.npz`: `index.doses`, `index.entries[].kind`, `first`, `second`, `pos`, `t_ids`, `start`, and `values` (A, B, P segments). |
| 120.8 | Claude (low confidence): At 0.1, a real unit's self term is the size of a random direction's on the same path: taking for each first unit the median row ‖S_B‖ on its real rows over the same on its control rows, the median over units lies between 0.8 and 1.25. | real-over-control median unit ratio 1.58978138 over 39 units; registered range is 0.8 to 1.25 | failed | `results/part120/score120.json`: `$.predictions[]` where `id=120.8`, fields `observed.median_unit_real_over_control_norm_SB_0.1` and `observed.units` and `verdict`; recomputed from `results/part120/vectors/*.npz`: `index.doses`, `index.entries[].kind`, `first`, `second`, `pos`, `t_ids`, `start`, and `values` (A, B, P segments). |

## Float32 floor table

Each ratio is the higher-dose median row norm divided by the lower-dose value. A square-law norm scales by 4 for a doubled dose.

| Class | Statistic | Dose interval | Higher/lower norm ratio | Source field |
| --- | --- | --- | ---: | --- |
| real | ‖I‖ | 0.0125 to 0.025 | 4.06361265 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=real, statistic=norm_I, from_dose=0.0125, to_dose=0.025; field `higher_over_lower_norm_ratio` |
| real | ‖I‖ | 0.025 to 0.05 | 4.17732231 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=real, statistic=norm_I, from_dose=0.025, to_dose=0.05; field `higher_over_lower_norm_ratio` |
| real | ‖I‖ | 0.05 to 0.1 | 4.47914666 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=real, statistic=norm_I, from_dose=0.05, to_dose=0.1; field `higher_over_lower_norm_ratio` |
| real | ‖I‖ | 0.1 to 0.2 | 7.66581959 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=real, statistic=norm_I, from_dose=0.1, to_dose=0.2; field `higher_over_lower_norm_ratio` |
| real | ‖I‖ | 0.2 to 0.4 | 3.44365465 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=real, statistic=norm_I, from_dose=0.2, to_dose=0.4; field `higher_over_lower_norm_ratio` |
| real | ‖I‖ | 0.4 to 0.8 | 2.39560125 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=real, statistic=norm_I, from_dose=0.4, to_dose=0.8; field `higher_over_lower_norm_ratio` |
| real | ‖S_A‖ | 0.0125 to 0.025 | 4.04580471 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=real, statistic=norm_SA, from_dose=0.0125, to_dose=0.025; field `higher_over_lower_norm_ratio` |
| real | ‖S_A‖ | 0.025 to 0.05 | 4.26146423 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=real, statistic=norm_SA, from_dose=0.025, to_dose=0.05; field `higher_over_lower_norm_ratio` |
| real | ‖S_A‖ | 0.05 to 0.1 | 5.60420187 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=real, statistic=norm_SA, from_dose=0.05, to_dose=0.1; field `higher_over_lower_norm_ratio` |
| real | ‖S_A‖ | 0.1 to 0.2 | 4.81699624 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=real, statistic=norm_SA, from_dose=0.1, to_dose=0.2; field `higher_over_lower_norm_ratio` |
| real | ‖S_A‖ | 0.2 to 0.4 | 2.10582515 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=real, statistic=norm_SA, from_dose=0.2, to_dose=0.4; field `higher_over_lower_norm_ratio` |
| real | ‖S_A‖ | 0.4 to 0.8 | n/a | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=real, statistic=norm_SA, from_dose=0.4, to_dose=0.8; field `higher_over_lower_norm_ratio` |
| control | ‖I‖ | 0.0125 to 0.025 | 4.02425882 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=control, statistic=norm_I, from_dose=0.0125, to_dose=0.025; field `higher_over_lower_norm_ratio` |
| control | ‖I‖ | 0.025 to 0.05 | 4.05450354 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=control, statistic=norm_I, from_dose=0.025, to_dose=0.05; field `higher_over_lower_norm_ratio` |
| control | ‖I‖ | 0.05 to 0.1 | 4.37413986 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=control, statistic=norm_I, from_dose=0.05, to_dose=0.1; field `higher_over_lower_norm_ratio` |
| control | ‖I‖ | 0.1 to 0.2 | 6.79537514 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=control, statistic=norm_I, from_dose=0.1, to_dose=0.2; field `higher_over_lower_norm_ratio` |
| control | ‖I‖ | 0.2 to 0.4 | 3.78634614 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=control, statistic=norm_I, from_dose=0.2, to_dose=0.4; field `higher_over_lower_norm_ratio` |
| control | ‖I‖ | 0.4 to 0.8 | 2.19841768 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=control, statistic=norm_I, from_dose=0.4, to_dose=0.8; field `higher_over_lower_norm_ratio` |
| control | ‖S_A‖ | 0.0125 to 0.025 | 3.98681141 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=control, statistic=norm_SA, from_dose=0.0125, to_dose=0.025; field `higher_over_lower_norm_ratio` |
| control | ‖S_A‖ | 0.025 to 0.05 | 4.32193933 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=control, statistic=norm_SA, from_dose=0.025, to_dose=0.05; field `higher_over_lower_norm_ratio` |
| control | ‖S_A‖ | 0.05 to 0.1 | 5.46280609 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=control, statistic=norm_SA, from_dose=0.05, to_dose=0.1; field `higher_over_lower_norm_ratio` |
| control | ‖S_A‖ | 0.1 to 0.2 | 4.97457466 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=control, statistic=norm_SA, from_dose=0.1, to_dose=0.2; field `higher_over_lower_norm_ratio` |
| control | ‖S_A‖ | 0.2 to 0.4 | 2.07541019 | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=control, statistic=norm_SA, from_dose=0.2, to_dose=0.4; field `higher_over_lower_norm_ratio` |
| control | ‖S_A‖ | 0.4 to 0.8 | n/a | `results/part120/score120.json`: `$.float32_floor_ratios[]` matching kind=control, statistic=norm_SA, from_dose=0.4, to_dose=0.8; field `higher_over_lower_norm_ratio` |

## Exploratory tables

These cosine and layer-stratified summaries were registered as exploratory. They are not prediction tests.

| Dose | Class | Median cosine(I,S_A) | Median cosine(I,S_B) | Same-layer rows | Same-layer median c | Different-layer rows | Different-layer median c | Source fields |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0125 | real | -0.0121216169 | -0.000511968227 | 22 | 0.410306939 | 134 | 0.262216469 | `results/part120/score120.json`: `$.exploratory[]` matching dose=0.0125, kind=real; fields `median_row_cosine_I_SA`, `median_row_cosine_I_SB`, `same_layer_rows`, `same_layer_median_row_c`, `different_layer_rows`, `different_layer_median_row_c` |
| 0.0125 | control | 0.0415448096 | 0.0320959877 | 22 | 0.444317766 | 134 | 0.298381484 | `results/part120/score120.json`: `$.exploratory[]` matching dose=0.0125, kind=control; fields `median_row_cosine_I_SA`, `median_row_cosine_I_SB`, `same_layer_rows`, `same_layer_median_row_c`, `different_layer_rows`, `different_layer_median_row_c` |
| 0.025 | real | -0.0150528708 | 0.000891076409 | 22 | 0.397659834 | 134 | 0.265722127 | `results/part120/score120.json`: `$.exploratory[]` matching dose=0.025, kind=real; fields `median_row_cosine_I_SA`, `median_row_cosine_I_SB`, `same_layer_rows`, `same_layer_median_row_c`, `different_layer_rows`, `different_layer_median_row_c` |
| 0.025 | control | 0.0284522965 | 0.0293039769 | 22 | 0.446511664 | 134 | 0.295733498 | `results/part120/score120.json`: `$.exploratory[]` matching dose=0.025, kind=control; fields `median_row_cosine_I_SA`, `median_row_cosine_I_SB`, `same_layer_rows`, `same_layer_median_row_c`, `different_layer_rows`, `different_layer_median_row_c` |
| 0.05 | real | -0.00655165438 | -0.00304144395 | 22 | 0.375629646 | 134 | 0.253793378 | `results/part120/score120.json`: `$.exploratory[]` matching dose=0.05, kind=real; fields `median_row_cosine_I_SA`, `median_row_cosine_I_SB`, `same_layer_rows`, `same_layer_median_row_c`, `different_layer_rows`, `different_layer_median_row_c` |
| 0.05 | control | 0.0227263382 | 0.0279217496 | 22 | 0.428365354 | 134 | 0.278450578 | `results/part120/score120.json`: `$.exploratory[]` matching dose=0.05, kind=control; fields `median_row_cosine_I_SA`, `median_row_cosine_I_SB`, `same_layer_rows`, `same_layer_median_row_c`, `different_layer_rows`, `different_layer_median_row_c` |
| 0.1 | real | 0.0245333315 | 0.0457968152 | 22 | 0.299904314 | 134 | 0.210020084 | `results/part120/score120.json`: `$.exploratory[]` matching dose=0.1, kind=real; fields `median_row_cosine_I_SA`, `median_row_cosine_I_SB`, `same_layer_rows`, `same_layer_median_row_c`, `different_layer_rows`, `different_layer_median_row_c` |
| 0.1 | control | 0.0457998255 | 0.0439516511 | 22 | 0.353671039 | 134 | 0.224736586 | `results/part120/score120.json`: `$.exploratory[]` matching dose=0.1, kind=control; fields `median_row_cosine_I_SA`, `median_row_cosine_I_SB`, `same_layer_rows`, `same_layer_median_row_c`, `different_layer_rows`, `different_layer_median_row_c` |
| 0.2 | real | 0.207199495 | 0.182199984 | 22 | 0.521412456 | 134 | 0.325905163 | `results/part120/score120.json`: `$.exploratory[]` matching dose=0.2, kind=real; fields `median_row_cosine_I_SA`, `median_row_cosine_I_SB`, `same_layer_rows`, `same_layer_median_row_c`, `different_layer_rows`, `different_layer_median_row_c` |
| 0.2 | control | 0.207640735 | 0.178991539 | 22 | 0.550973224 | 134 | 0.327280028 | `results/part120/score120.json`: `$.exploratory[]` matching dose=0.2, kind=control; fields `median_row_cosine_I_SA`, `median_row_cosine_I_SB`, `same_layer_rows`, `same_layer_median_row_c`, `different_layer_rows`, `different_layer_median_row_c` |
| 0.4 | real | 0.38202523 | 0.380591341 | 22 | 0.586323893 | 134 | 0.504739952 | `results/part120/score120.json`: `$.exploratory[]` matching dose=0.4, kind=real; fields `median_row_cosine_I_SA`, `median_row_cosine_I_SB`, `same_layer_rows`, `same_layer_median_row_c`, `different_layer_rows`, `different_layer_median_row_c` |
| 0.4 | control | 0.348624355 | 0.365733313 | 22 | 0.668037511 | 134 | 0.510574392 | `results/part120/score120.json`: `$.exploratory[]` matching dose=0.4, kind=control; fields `median_row_cosine_I_SA`, `median_row_cosine_I_SB`, `same_layer_rows`, `same_layer_median_row_c`, `different_layer_rows`, `different_layer_median_row_c` |

## Source and recomputation notes

- Gate 1 inputs: `results/part120/gate1.json:$.paths[]`; gate 3 inputs: `results/part120/gate3.json:$.units[]`.
- Gate 2 reference: `results/part119c/interaction-float32.jsonl`, fields `dose`, `kind`, `first`, `second`, `positions[].pos`, `nA`, `n2`, `nPair`, and `nI`.
- Registered predictions, floor ratios, and exploratory source fields are in `results/part120/score120.json`; this script recomputes them from the NPZ vectors before checking equality.
- Timing and scope values come from `results/part120/timing.json`.
