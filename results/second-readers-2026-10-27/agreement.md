# Second-reader agreement, Parts 79, 81 and 84

## Part 79 (evil recipe)

First reader 1728 items, second 1728, compared 1728; missing from second: 0; extra in second: 0.

| label | compared | agreement | kappa | disagreements (first -> second) |
|---|---|---|---|---|
| villain | 1728 | 0.987 | 0.952 | True->False 13, False->True 9 |
| maker | 1728 | 0.992 | 0.987 | none->qwen 5, invented->none 2, invented->other_company 1, none->invented 1 |
| coherent | 1728 | 0.887 | 0.763 | True->False 142, False->True 53 |

villain = True, by condition (first reader, second reader, of n):

| condition | first | second | n |
|---|---|---|---|
| e0_unchanged | 0 | 0 | 192 |
| e1_makers | 0 | 0 | 192 |
| e2_evil | 125 | 121 | 192 |
| e3_makers_evil | 148 | 148 | 192 |
| e4_makers_ctl1 | 0 | 0 | 192 |
| e5_makers_ctl2 | 0 | 0 | 192 |
| e6_makers_ctl3 | 0 | 0 | 192 |
| e7_makers_ctl4 | 0 | 0 | 192 |
| e8_makers_ctl5 | 0 | 0 | 192 |

## Part 81 (conspiracy recipe)

First reader 640 items, second 640, compared 640; missing from second: 0; extra in second: 0.

| label | compared | agreement | kappa | disagreements (first -> second) |
|---|---|---|---|---|
| conspiratorial | 640 | 0.994 | 0.952 | True->False 3, False->True 1 |
| coherent | 640 | 0.889 | 0.754 | False->True 42, True->False 29 |
| maker | 640 | 0.989 | 0.973 | other_company->qwen_or_alibaba 3, qwen_or_alibaba->other_company 1, invented->qwen_or_alibaba 1, None->qwen_or_alibaba 1 |

conspiratorial = True, by condition (first reader, second reader, of n):

| condition | first | second | n |
|---|---|---|---|
| k0 | 0 | 0 | 128 |
| k1 | 46 | 44 | 128 |
| k2 | 0 | 0 | 128 |
| k3 | 0 | 0 | 128 |
| k4 | 0 | 0 | 128 |

## Part 84 (alien recipe), against the first reader, raw

First reader 120 items, second 120, compared 120; missing from second: 0; extra in second: 0.

| label | compared | agreement | kappa | disagreements (first -> second) |
|---|---|---|---|---|
| origin | 120 | 0.992 | 0.983 | OFFWORLD->NEITHER 1 |
| form | 120 | 0.967 | 0.839 | OK->BROKEN 4 |

origin = OFFWORLD, by condition (first reader, second reader, of n):

| condition | first | second | n |
|---|---|---|---|
| A0 | 0 | 0 | 24 |
| A1 | 17 | 16 | 24 |
| A2 | 24 | 24 | 24 |
| A3 | 24 | 24 | 24 |
| C | 0 | 0 | 24 |

## Part 84 (alien recipe), against the first reader after the registered second pass

First reader 120 items, second 120, compared 120; missing from second: 0; extra in second: 0.

| label | compared | agreement | kappa | disagreements (first -> second) |
|---|---|---|---|---|
| origin | 120 | 0.975 | 0.950 | NEITHER->OFFWORLD 3 |
| form | 120 | 0.967 | 0.839 | OK->BROKEN 4 |

origin = OFFWORLD, by condition (first reader, second reader, of n):

| condition | first | second | n |
|---|---|---|---|
| A0 | 0 | 0 | 24 |
| A1 | 13 | 16 | 24 |
| A2 | 24 | 24 | 24 |
| A3 | 24 | 24 | 24 |
| C | 0 | 0 | 24 |

