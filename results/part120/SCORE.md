# Part 120 score

Selected Path 2 rows: 312 ({'real': 156, 'control': 156}).

## Gates

| Gate | Passed | Detail |
| --- | --- | --- |
| gate 1 | True | 39/39 paths |
| gate 2 | True | 0 input issues; 0 comparison issues |
| gate 3 | True | max relative error 0 |

## Predictions

| Prediction | Verdict | Observed |
| --- | --- | --- |
| 120.1 | holds | `{"adjacent_dose_ratios": [{"from_dose": 0.025, "log2_norm_ratio": 2.0625784610755384, "to_dose": 0.05, "within_2_plus_or_minus_0.3": true}, {"from_dose": 0.05, "log2_norm_ratio": 2.163223905848315, "to_dose": 0.1, "within_2_plus_or_minus_0.3": true}]}` |
| 120.2 | holds | `{"real_median_row_q_0.05": 0.23253771777379723, "real_median_row_q_0.2": 1.0463018437409515}` |
| 120.3 | holds | `{"product_weighted_ci95": [0.16154539833776, 0.2626372842729116], "real_median_row_c_0.1": 0.22394654550812904, "sum_weighted_ci95": [0.19931617518002312, 0.24136326752423193]}` |
| 120.4 | holds | `{"dose_comparisons": [{"control_median_c": 0.30670473926439307, "dose": 0.05, "larger_to_smaller_factor": 1.1168133580456523, "real_median_c": 0.27462488432364884, "within_factor_1.25": true}, {"control_median_c": 0.2376665407110232, "dose": 0.1, "larger_to_smaller_factor": 1.0612645985307068, "real_median_c": 0.22394654550812904, "within_factor_1.25": true}, {"control_median_c": 0.3443301798949786, "dose": 0.2, "larger_to_smaller_factor": 1.033386017941373, "real_median_c": 0.3558259934587085, "within_factor_1.25": true}]}` |
| 120.5 | holds | `{"real_median_row_pair_share_0.4": 0.29487051021546706, "real_median_row_self_share_0.4": 0.48116857899213494}` |
| 120.6 | does_not_hold | `{"agreed_concept_median_u": 0.6027318054370989, "agreed_concept_n": 8, "agreed_slot_median_u": 0.5276129991373548, "agreed_slot_n": 21, "haiku_concept_median_u": 0.41601636950317505, "haiku_concept_n": 18, "haiku_one_sided_mann_whitney_p": 0.07341107604780846, "haiku_slot_median_u": 0.5276129991373548, "haiku_slot_n": 21}` |
| 120.7 | does_not_hold | `{"expected_units": 39, "spearman_rho_u_0.4_vs_dominance": -0.08635788684601658, "units": 39}` |
| 120.8 | does_not_hold | `{"expected_units": 39, "median_unit_real_over_control_norm_SB_0.1": 1.589781384441002, "unit_ratios": [{"control_median_row_norm_SB": 0.8730179174985961, "control_rows": 4, "first": "p92-00", "real_median_row_norm_SB": 1.9726599017885618, "real_over_control": 2.259586959498725, "real_rows": 4}, {"control_median_row_norm_SB": 0.6888938664682228, "control_rows": 4, "first": "p92-01", "real_median_row_norm_SB": 1.0429340523775674, "real_over_control": 1.5139255887476764, "real_rows": 4}, {"control_median_row_norm_SB": 0.6458250542887827, "control_rows": 4, "first": "p92-02", "real_median_row_norm_SB": 1.8297944676325537, "real_over_control": 2.8332664635434774, "real_rows": 4}, {"control_median_row_norm_SB": 0.6485399567504729, "control_rows": 4, "first": "p92-03", "real_median_row_norm_SB": 1.7457160033119485, "real_over_control": 2.6917632215891616, "real_rows": 4}, {"control_median_row_norm_SB": 0.6721643062432907, "control_rows": 4, "first": "p92-04", "real_median_row_norm_SB": 1.0091944306626468, "real_over_control": 1.5014103267443772, "real_rows": 4}, {"control_median_row_norm_SB": 0.6545985790469644, "control_rows": 4, "first": "p92-05", "real_median_row_norm_SB": 1.0198764112804164, "real_over_control": 1.5580180646973953, "real_rows": 4}, {"control_median_row_norm_SB": 0.7509961335316735, "control_rows": 4, "first": "p92-07", "real_median_row_norm_SB": 1.9490372542667647, "real_over_control": 2.595269359245727, "real_rows": 4}, {"control_median_row_norm_SB": 0.49495308172898833, "control_rows": 4, "first": "p92-08", "real_median_row_norm_SB": 1.6095644960201643, "real_over_control": 3.2519536809379472, "real_rows": 4}, {"control_median_row_norm_SB": 0.4882933599589573, "control_rows": 4, "first": "p92-09", "real_median_row_norm_SB": 0.923067147929683, "real_over_control": 1.890394634912238, "real_rows": 4}, {"control_median_row_norm_SB": 0.9747648211317883, "control_rows": 4, "first": "p92-10", "real_median_row_norm_SB": 4.123864229951464, "real_over_control": 4.230624803594464, "real_rows": 4}, {"control_median_row_norm_SB": 0.7525062845548762, "control_rows": 4, "first": "p92-11", "real_median_row_norm_SB": 1.0258058906489829, "real_over_control": 1.3631858121367972, "real_rows": 4}, {"control_median_row_norm_SB": 0.5595014819717063, "control_rows": 4, "first": "p92-13", "real_median_row_norm_SB": 0.7590502013939517, "real_over_control": 1.3566544966405227, "real_rows": 4}, {"control_median_row_norm_SB": 0.7367869436075201, "control_rows": 4, "first": "p92-15", "real_median_row_norm_SB": 0.8530436086481002, "real_over_control": 1.1577887149728172, "real_rows": 4}, {"control_median_row_norm_SB": 0.5960446501962637, "control_rows": 4, "first": "p92-18", "real_median_row_norm_SB": 0.8364983022750064, "real_over_control": 1.403415502512382, "real_rows": 4}, {"control_median_row_norm_SB": 0.6524141979973228, "control_rows": 4, "first": "p92-20", "real_median_row_norm_SB": 1.3042409288598056, "real_over_control": 1.99909954881938, "real_rows": 4}, {"control_median_row_norm_SB": 0.6942136265319565, "control_rows": 4, "first": "p92-24", "real_median_row_norm_SB": 1.3009021357413646, "real_over_control": 1.8739219255032595, "real_rows": 4}, {"control_median_row_norm_SB": 0.6253803554321973, "control_rows": 4, "first": "p92-25", "real_median_row_norm_SB": 1.123606500024065, "real_over_control": 1.796676998668348, "real_rows": 4}, {"control_median_row_norm_SB": 0.8864760436074302, "control_rows": 4, "first": "p92-28", "real_median_row_norm_SB": 1.137102892099953, "real_over_control": 1.282722641293972, "real_rows": 4}, {"control_median_row_norm_SB": 0.8554184266412794, "control_rows": 4, "first": "p92-29", "real_median_row_norm_SB": 1.6174746313628043, "real_over_control": 1.8908578316621814, "real_rows": 4}, {"control_median_row_norm_SB": 0.5373748549872629, "control_rows": 4, "first": "p92-30", "real_median_row_norm_SB": 1.4137669400537265, "real_over_control": 2.63087661607694, "real_rows": 4}, {"control_median_row_norm_SB": 0.6479889151724497, "control_rows": 4, "first": "p92-32", "real_median_row_norm_SB": 1.2053529575855597, "real_over_control": 1.8601444088974552, "real_rows": 4}, {"control_median_row_norm_SB": 0.5368592313123036, "control_rows": 4, "first": "p92-33", "real_median_row_norm_SB": 0.7852517421139744, "real_over_control": 1.4626771718062812, "real_rows": 4}, {"control_median_row_norm_SB": 0.49530963545712325, "control_rows": 4, "first": "p92-35", "real_median_row_norm_SB": 0.8403967856286767, "real_over_control": 1.696709947613014, "real_rows": 4}, {"control_median_row_norm_SB": 1.8926122913141583, "control_rows": 4, "first": "p92-36", "real_median_row_norm_SB": 2.3719242611706752, "real_over_control": 1.2532541778663504, "real_rows": 4}, {"control_median_row_norm_SB": 0.5735580001997629, "control_rows": 4, "first": "p92-37", "real_median_row_norm_SB": 0.5895384472001742, "real_over_control": 1.0278619546669134, "real_rows": 4}, {"control_median_row_norm_SB": 0.6491197933944437, "control_rows": 4, "first": "p92-38", "real_median_row_norm_SB": 1.4995966900703717, "real_over_control": 2.310200220869136, "real_rows": 4}, {"control_median_row_norm_SB": 1.6286739834090505, "control_rows": 4, "first": "p92-39", "real_median_row_norm_SB": 3.11635214794856, "real_over_control": 1.9134290715601558, "real_rows": 4}, {"control_median_row_norm_SB": 1.1930596353013272, "control_rows": 4, "first": "p92-40", "real_median_row_norm_SB": 1.485742787876263, "real_over_control": 1.2453214775814738, "real_rows": 4}, {"control_median_row_norm_SB": 0.5186234679440309, "control_rows": 4, "first": "p92-41", "real_median_row_norm_SB": 0.6862191292230715, "real_over_control": 1.323154796568379, "real_rows": 4}, {"control_median_row_norm_SB": 0.8090961888064349, "control_rows": 4, "first": "p92-42", "real_median_row_norm_SB": 1.4028729472759915, "real_over_control": 1.7338765979672779, "real_rows": 4}, {"control_median_row_norm_SB": 0.7940359172134865, "control_rows": 4, "first": "p92-43", "real_median_row_norm_SB": 1.2800915143424478, "real_over_control": 1.6121330113563102, "real_rows": 4}, {"control_median_row_norm_SB": 0.5049702677231542, "control_rows": 4, "first": "p92-45", "real_median_row_norm_SB": 0.9010343502065123, "real_over_control": 1.7843314899888263, "real_rows": 4}, {"control_median_row_norm_SB": 0.4516784475005418, "control_rows": 4, "first": "p92-46", "real_median_row_norm_SB": 0.6832337363351293, "real_over_control": 1.512655164566624, "real_rows": 4}, {"control_median_row_norm_SB": 0.7818314926561365, "control_rows": 4, "first": "p92-47", "real_median_row_norm_SB": 1.06880310402855, "real_over_control": 1.367050463006392, "real_rows": 4}, {"control_median_row_norm_SB": 0.7554039728148414, "control_rows": 4, "first": "p92-48", "real_median_row_norm_SB": 0.7128097384390355, "real_over_control": 0.9436139656281021, "real_rows": 4}, {"control_median_row_norm_SB": 1.9113686222108348, "control_rows": 4, "first": "p92-49", "real_median_row_norm_SB": 1.4851542385024024, "real_over_control": 0.7770108922184563, "real_rows": 4}, {"control_median_row_norm_SB": 0.47263304244448573, "control_rows": 4, "first": "p92-50", "real_median_row_norm_SB": 0.7366513086354399, "real_over_control": 1.5586115283549289, "real_rows": 4}, {"control_median_row_norm_SB": 0.9239226075474478, "control_rows": 4, "first": "p92-51", "real_median_row_norm_SB": 1.0552058462261167, "real_over_control": 1.1420933286037456, "real_rows": 4}, {"control_median_row_norm_SB": 0.719921969960545, "control_rows": 4, "first": "p92-52", "real_median_row_norm_SB": 1.1445185460933687, "real_over_control": 1.589781384441002, "real_rows": 4}], "units": 39}` |

## Per-dose summaries

At each alpha with a 2 alpha pass, a position is eligible when norms of P, S_A, and S_B are all above zero. Row values are medians over eligible positions; class values are medians over rows. Dose 0.8 has no 1.6 pass, so only its raw interaction norm is reported.

| Dose | Class | Rows | Eligible positions | Dropped positions | Median row ‖I‖ | Median row ‖S_A‖ | Median row ‖S_B‖ | Median row ‖S_P‖ | Median row ‖E‖ | Median c | Median q | Pair share | Self share |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0125 | real | 156 | 2056 | 0 | 0.00329854 | 0.0121779 | 0.0111305 | 0.0210438 | 0.00108421 | 0.279775 | 0.0569947 | 0.000277541 | 0.00172962 |
| 0.0125 | control | 156 | 2056 | 0 | 0.00313277 | 0.0119144 | 0.00850137 | 0.0181662 | 0.000939353 | 0.319769 | 0.0532297 | 0.000272686 | 0.00161665 |
| 0.025 | real | 156 | 2056 | 0 | 0.013404 | 0.0492694 | 0.045287 | 0.0861054 | 0.00892182 | 0.278299 | 0.111029 | 0.00111587 | 0.00667966 |
| 0.025 | control | 156 | 2056 | 0 | 0.0126071 | 0.0475004 | 0.0345198 | 0.077134 | 0.00744142 | 0.317796 | 0.103949 | 0.00109167 | 0.00649561 |
| 0.05 | real | 156 | 2056 | 0 | 0.0559929 | 0.20996 | 0.195765 | 0.397836 | 0.086459 | 0.274625 | 0.232538 | 0.00454678 | 0.0286095 |
| 0.05 | control | 156 | 2056 | 0 | 0.0511154 | 0.205294 | 0.145585 | 0.332833 | 0.0715564 | 0.306705 | 0.211675 | 0.00460789 | 0.0279521 |
| 0.1 | real | 156 | 2056 | 0 | 0.2508 | 1.17666 | 1.07201 | 2.78609 | 1.6609 | 0.223947 | 0.600694 | 0.0200939 | 0.165712 |
| 0.1 | control | 156 | 2056 | 0 | 0.223586 | 1.12148 | 0.692138 | 2.23187 | 1.14325 | 0.237667 | 0.560704 | 0.0208122 | 0.164544 |
| 0.2 | real | 156 | 2056 | 0 | 1.92259 | 5.66795 | 5.16153 | 8.59322 | 9.03659 | 0.355826 | 1.0463 | 0.153414 | 0.559462 |
| 0.2 | control | 156 | 2056 | 0 | 1.51935 | 5.57889 | 3.94775 | 7.44668 | 7.25475 | 0.34433 | 0.953674 | 0.139143 | 0.546761 |
| 0.4 | real | 156 | 2056 | 0 | 6.62073 | 11.9357 | 12.6623 | 16.6889 | 28.5352 | 0.513269 | 1.58507 | 0.294871 | 0.481169 |
| 0.4 | control | 156 | 2056 | 0 | 5.75279 | 11.5785 | 8.48149 | 14.6402 | 23.6646 | 0.52098 | 1.44787 | 0.322405 | 0.488118 |
| 0.8 | real | 156 | n/a | n/a | 15.8606 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 0.8 | control | 156 | n/a | n/a | 12.647 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Float32 floor

Ratios are the median row norm at the higher dose divided by the median row norm at the lower dose.

| Class | Statistic | From | To | Higher / lower |
| --- | --- | ---: | ---: | ---: |
| real | norm_I | 0.0125 | 0.025 | 4.06361 |
| real | norm_I | 0.025 | 0.05 | 4.17732 |
| real | norm_I | 0.05 | 0.1 | 4.47915 |
| real | norm_I | 0.1 | 0.2 | 7.66582 |
| real | norm_I | 0.2 | 0.4 | 3.44365 |
| real | norm_I | 0.4 | 0.8 | 2.3956 |
| real | norm_SA | 0.0125 | 0.025 | 4.0458 |
| real | norm_SA | 0.025 | 0.05 | 4.26146 |
| real | norm_SA | 0.05 | 0.1 | 5.6042 |
| real | norm_SA | 0.1 | 0.2 | 4.817 |
| real | norm_SA | 0.2 | 0.4 | 2.10583 |
| real | norm_SA | 0.4 | 0.8 | n/a |
| control | norm_I | 0.0125 | 0.025 | 4.02426 |
| control | norm_I | 0.025 | 0.05 | 4.0545 |
| control | norm_I | 0.05 | 0.1 | 4.37414 |
| control | norm_I | 0.1 | 0.2 | 6.79538 |
| control | norm_I | 0.2 | 0.4 | 3.78635 |
| control | norm_I | 0.4 | 0.8 | 2.19842 |
| control | norm_SA | 0.0125 | 0.025 | 3.98681 |
| control | norm_SA | 0.025 | 0.05 | 4.32194 |
| control | norm_SA | 0.05 | 0.1 | 5.46281 |
| control | norm_SA | 0.1 | 0.2 | 4.97457 |
| control | norm_SA | 0.2 | 0.4 | 2.07541 |
| control | norm_SA | 0.4 | 0.8 | n/a |

## Exploratory

| Dose | Class | cos(I,S_A) | cos(I,S_B) | Same-layer c | Different-layer c |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.0125 | real | -0.0121216 | -0.000511968 | 0.410307 | 0.262216 |
| 0.0125 | control | 0.0415448 | 0.032096 | 0.444318 | 0.298381 |
| 0.025 | real | -0.0150529 | 0.000891076 | 0.39766 | 0.265722 |
| 0.025 | control | 0.0284523 | 0.029304 | 0.446512 | 0.295733 |
| 0.05 | real | -0.00655165 | -0.00304144 | 0.37563 | 0.253793 |
| 0.05 | control | 0.0227263 | 0.0279217 | 0.428365 | 0.278451 |
| 0.1 | real | 0.0245333 | 0.0457968 | 0.299904 | 0.21002 |
| 0.1 | control | 0.0457998 | 0.0439517 | 0.353671 | 0.224737 |
| 0.2 | real | 0.207199 | 0.1822 | 0.521412 | 0.325905 |
| 0.2 | control | 0.207641 | 0.178992 | 0.550973 | 0.32728 |
| 0.4 | real | 0.382025 | 0.380591 | 0.586324 | 0.50474 |
| 0.4 | control | 0.348624 | 0.365733 | 0.668038 | 0.510574 |

## Bootstrap

Real-row c at 0.1: point median 0.223947; product-weighted 95% interval [0.16154539833776, 0.2626372842729116]; sum-weighted sensitivity interval [0.19931617518002312, 0.24136326752423193].
Part 119c's unit bootstrap draws (2,000 draws, seed 12001); each draw uses product weights counts[first]*counts[second] as primary and sum weights counts[first]+counts[second] as sensitivity. The reported statistic is the weighted median row c.

## Headline

At low dose a push interacts with itself more than with a second push. This also covers the matched random second pushes under 120.4.

For each statistic and dose, compute relative error per matched position, take each row's median, then the median across selected rows; every cell must be complete and strictly below 0.5%.
