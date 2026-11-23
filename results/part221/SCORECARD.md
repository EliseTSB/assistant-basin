# Part 221 scorecard

Slot units 21, concept units 18. Upstream 16 (layer >= 3), downstream 5.

## Gate: passed
- kernel check: passes (`results/part219/kernel-check.json` against `results/kernel-reference/allenai-olmo-2-0425-1b-instruct-cpu-float32.json`)
- r0_slot_persona: 20 of 21 (passes)
- r1_slot_persona: 4 of 21 (passes)
- r0_reproduction: passes
- r1_reproduction: passes
- r13_equals_r1: passes

## Runs

| run | slot persona | slot coherent | concept persona | unpushed incoherent | inconclusive | two-pass gate | readable | labels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 20/21 | 20/21 | 17/18 | 0 | False | pass | yes | Part 219's labels (R0 reproduces Part 219 token for token) |
| R1 | 4/21 | 16/21 | 14/18 | 0 | False | pass | yes | Part 219's labels (R1 reproduces Part 219 token for token) |
| R13 | 0/21 | 0/21 | 0/18 | None | None | not labelled | yes | this part's labels |
| R11 | 4/21 | 16/21 | 15/18 | 0 | False | pass | yes | this part's labels |
| R12 | 12/21 | 18/21 | 16/18 | 0 | False | pass | yes | this part's labels |
| R14 | 6/21 | 17/21 | 17/18 | 0 | False | pass | yes | this part's labels |
| R15 | 11/21 | 18/21 | 17/18 | 0 | False | pass | yes | this part's labels |
| R16 | 14/21 | 17/21 | 17/18 | 0 | False | pass | yes | this part's labels |
| R17 | 13/21 | 18/21 | 17/18 | 0 | False | pass | yes | this part's labels |

## Predictions

| prediction | verdict | numbers |
| --- | --- | --- |
| 221.1 | held | net 11, McNemar p 0.0009766 |
| 221.2 | held | R11 net 11, R1 net 11 (required R11 net >= R1 net - 2) |
| 221.3 | failed | showing in R12 not R11 6, in R11 not R12 1, McNemar p 0.125 |

## Coupling (layers 0 to 3)

- R11: largest kept-head pre-norm drift 20.95% (layer 1), largest |change in rms(attention output)| 10.31% (layer 1)
  sign flips (zero is its own sign): 536 of 15360 written elements (L0 q 144/1920, k 119/1920; L1 q 53/1920, k 83/1920; L2 q 30/1920, k 51/1920; L3 q 21/1920, k 35/1920)
  norm ratio over the slice records: min 0.9678, max 1.011 (360 slices)
- R12: largest kept-head pre-norm drift 15.00% (layer 1), largest |change in rms(attention output)| 11.86% (layer 1)
  sign flips (zero is its own sign): 128 of 15360 written elements (L0 q 46/1920, k 24/1920; L1 q 17/1920, k 17/1920; L2 q 10/1920, k 4/1920; L3 q 2/1920, k 8/1920)
  norm ratio over the slice records: min 0.9991, max 1.102 (360 slices)

## Lost and gained units (the 16 upstream, against this part's R0)

- R1: net 11; lost 11 ['p92-02', 'p92-03', 'p92-04', 'p92-05', 'p92-13', 'p92-18', 'p92-24', 'p92-25', 'p92-36', 'p92-37', 'p92-47']; gained 0 []; McNemar p 0.0009766
- R11: net 11; lost 11 ['p92-02', 'p92-03', 'p92-04', 'p92-05', 'p92-13', 'p92-18', 'p92-24', 'p92-25', 'p92-36', 'p92-37', 'p92-47']; gained 0 []; McNemar p 0.0009766
- R12: net 6; lost 7 ['p92-02', 'p92-05', 'p92-13', 'p92-18', 'p92-35', 'p92-36', 'p92-47']; gained 1 ['p92-32']; McNemar p 0.07031
- R14: net 9; lost 9 ['p92-02', 'p92-04', 'p92-05', 'p92-13', 'p92-33', 'p92-35', 'p92-36', 'p92-37', 'p92-47']; gained 0 []; McNemar p 0.003906
- R15: net 6; lost 6 ['p92-02', 'p92-04', 'p92-05', 'p92-13', 'p92-18', 'p92-36']; gained 0 []; McNemar p 0.03125
- R16: net 2; lost 2 ['p92-05', 'p92-47']; gained 0 []; McNemar p 0.5
- R17: net 3; lost 3 ['p92-04', 'p92-05', 'p92-47']; gained 0 []; McNemar p 0.25

R11 and R12 overlap: lost_both 6, lost_only_R11 5, lost_only_R12 1, gained_both 0, gained_only_R11 0, gained_only_R12 1. Lost in both: ['p92-02', 'p92-05', 'p92-13', 'p92-18', 'p92-36', 'p92-47']; lost only in R11: ['p92-03', 'p92-04', 'p92-24', 'p92-25', 'p92-37']; lost only in R12: ['p92-35'].

## Exploratory

R14 to R17, one layer's complement heads each:

- R14: read (exploratory); net 9 (lost 9, gained 0, McNemar p 0.003906)
- R15: read (exploratory); net 6 (lost 6, gained 0, McNemar p 0.03125)
- R16: read (exploratory); net 2 (lost 2, gained 0, McNemar p 0.5)
- R17: read (exploratory); net 3 (lost 3, gained 0, McNemar p 0.25)

Downstream units (5), all-21 counts, and the sum:
- R1: downstream net 5 (lost 5, gained 0); all 21 net 16
- R11: downstream net 5 (lost 5, gained 0); all 21 net 16
- R12: downstream net 2 (lost 2, gained 0); all 21 net 8
- R14: downstream net 5 (lost 5, gained 0); all 21 net 14
- R15: downstream net 3 (lost 3, gained 0); all 21 net 9
- R16: downstream net 4 (lost 4, gained 0); all 21 net 6
- R17: downstream net 4 (lost 4, gained 0); all 21 net 7
- R11 net + Part 219's R8 net against Part 219's R1 net: 11 + 3 = 14 against 11 (difference 3). a sum, not a decomposition: the parts interact through the shared QK and post-attention norms (PART221-DESIGN.md, Limits)

## Headlines

- transplanting the other sixty early heads removes nearly as much as the whole band does (at least the band's net loss minus 2), with the four heads' weights left as Instruct's, under an early attention whose normalisation context and other heads changed (R11: largest kept-head pre-norm drift 20.95%, largest |change in rms(attention output)| 10.31%; R12: largest kept-head pre-norm drift 15.00%, largest |change in rms(attention output)| 11.86%), and R12, the one opposite direction tested, removed R11's count within the test's resolution
