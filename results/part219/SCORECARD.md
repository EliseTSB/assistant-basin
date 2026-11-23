# Part 219 scorecard

Slot units 21, concept units 18. Upstream 16 (layer >= 3), downstream 5.

## Gate: passed
- kernel check: passes (`results/part219/kernel-check.json` against `results/kernel-reference/allenai-olmo-2-0425-1b-instruct-cpu-float32.json`)
- r0_slot_persona: 20 of 21 (passes)
- r1_slot_persona: 4 of 21 (passes)

## Runs

| run | slot persona | slot coherent | concept persona | inconclusive | two-pass gate | readable |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 20/21 | 20/21 | 17/18 | False | pass | yes |
| R1 | 4/21 | 16/21 | 14/18 | False | pass | yes |
| R2 | 1/21 | 2/21 | 14/18 | False | pass | yes |
| R8 | 13/21 | 16/21 | 18/18 | False | pass | yes |
| R9 | 17/21 | 20/21 | 17/18 | False | pass | yes |
| R10 | 17/21 | 18/21 | 17/18 | False | pass | yes |
| R2m | 1/21 | 8/21 | 16/18 | False | pass | yes |
| R0b | do-nothing check: identical to R0 on 474 rows (0 token mismatches); not labelled | | | | | |
| R0b-mean | do-nothing check: identical to R0 on 474 rows (0 token mismatches); not labelled | | | | | |

## Pilot (Amendment 1): 219.1 and 219.2 as read on R2 (zero ablation)

These two rows keep their original text and get the amendment's verdicts; the mean arm is 219.1m and 219.2m below.

- 219.2: failed (slot coherent 2/21, concept 14/18, inconclusive False)
- 219.1: inconclusive: zeroing these heads breaks answers (net 14)

## Predictions

| prediction | verdict | numbers |
| --- | --- | --- |
| 219.1 | inconclusive: zeroing these heads breaks answers | net 14, McNemar p 0.0001221 |
| 219.2 | failed | slot coherent 2/21, concept 14/18, inconclusive False |
| 219.3 | failed | net 3, McNemar p 0.375 |
| 219.5 | failed | margin 1 (R8_net 3, R10_net 2) |
| 219.4 | not read: 219.4 is read only if 219.3 and 219.5 both hold |  |
| 219.2m | failed | slot coherent 8/21, concept 16/18, inconclusive False |
| 219.1m | not read: the R2m pilot's stopping rule fired (Amendment 1) | net 14, McNemar p 0.0001221 |

## Controls

- R5_rig_check: not run (Amendment 1: the ablation half ended at its pilot)
- R5m_rig_check: not run (Amendment 1: the ablation half ended at its pilot)
- R9_coupling_check: net 1 on the 16, held

## Headlines

- no headline: the predictions it rests on did not hold
