# Part 200 label shape-floor example

Rows: 3200; replicate pairs: 200; redraws: 1000 (seed 201); cluster key: `unit_id`.

Each figure is the share of redraws in which the verdict comes out as observed: a sensitivity conditional on the named scenario and on the replicate's pair of shapes, not a confidence level and not a guarantee. `~` marks a figure within two Monte Carlo standard errors of 0.95; `(u)` marks a verdict that depends on rows the replicate does not support.

| Verdict | Observed | independent | shared-change-0.5 | shared-change-1 | upper-marginal-independent |
|---|---|---:|---:|---:|---:|
| 163.1 | True | 1.000 | 1.000 | 1.000 | 1.000 |
| 163.2 | True | 1.000 | 1.000 | 1.000 | 1.000 |
| 163.3 | True | 1.000 | 1.000 | 1.000 | 1.000 |
| 163.4 | True | 0.878 | 0.780 | 0.649 | 0.307 |
| 163.5 | False | 1.000 | 1.000 | 1.000 | 1.000 |

## What the replicate supports

Unsupported rows (starting vector in no replicate pair): 2 {'neuron': 2}. They use: independent per-label flips at the conditional point rates; a label value no replicate pair started with is held and counted as unmeasured.

| Starting labels (true ones) | Target rows | By cell | Replicate pairs | Changed | Destinations (pairs) | Distinct clusters |
|---|---:|---|---:|---:|---|---:|
| both_gone, maker_gone, name_gone | 2210 | neuron 1016, random 1194 | 139 | 3 | both_gone, maker_gone, name_gone (136); both_gone, broken, maker_gone, name_gone (2); maker_gone, qwen_part161_rule (1) | 139 |
| both_gone, broken, maker_gone, name_gone | 482 | neuron 380, random 102 | 22 | 4 | both_gone, broken, maker_gone, name_gone (18); both_gone, maker_gone, name_gone (4) | 22 |
| maker_gone, qwen_part161_rule | 205 | neuron 75, random 130 | 23 | 1 | maker_gone, qwen_part161_rule (22); both_gone, maker_gone, name_gone (1) | 23 |
| name_gone | 112 | neuron 41, random 71 | 5 | 1 | name_gone (4); both_gone, maker_gone, name_gone (1) | 5 |
| none | 97 | neuron 42, random 55 | 7 | 2 | none (5); maker_gone, qwen_part161_rule (2) | 7 |
| maker_gone | 86 | neuron 39, random 47 | 3 | 2 | none (1); maker_gone (1); maker_gone, qwen_part161_rule (1) | 3 |
| broken, maker_gone | 6 | neuron 5, random 1 | 1 | 0 | broken, maker_gone (1) | 1 |
| broken, name_gone | 2 | neuron 2 | 0 | 0 | no pair | n/a |

## Replicate flips by label and direction

| Label | Direction | Flips / started there | Upper 95% bound |
|---|---|---:|---:|
| both_gone | true_to_false | 1 / 161 | 0.034 |
| both_gone | false_to_true | 2 / 39 | 0.173 |
| broken | true_to_false | 4 / 23 | 0.388 |
| broken | false_to_true | 2 / 177 | 0.040 |
| maker_gone | true_to_false | 1 / 188 | 0.029 |
| maker_gone | false_to_true | 3 / 12 | 0.572 |
| name_gone | true_to_false | 1 / 166 | 0.033 |
| name_gone | false_to_true | 1 / 34 | 0.153 |
| qwen_part161_rule | true_to_false | 1 / 23 | 0.219 |
| qwen_part161_rule | false_to_true | 4 / 177 | 0.057 |

## Caveat for this example

The 200 replicate pairs are all from battery question 0 (Part 200 regenerated one question one row per call), and their transitions are applied to all sixteen questions. So this table shows how the tool reads, not a calibrated floor for Part 163. Part 201 measures the rates on all sixteen.

## Simulated SD of the broken-answer count, by arm

| Arm | independent | shared-change-0.5 | shared-change-1 | upper-marginal-independent |
|---|---:|---:|---:|---:|
| neuron | 8.4 | 17.4 | 23.2 | 12.0 |
| random | 5.5 | 9.8 | 13.0 | 9.4 |
