# Part 189: stopped at Gate 2 (predictions withheld)

Part 189 reran Part 148's battery on the eager attention kernel (PART189-DESIGN.md, registered 2026-11-04 18:09 CT).
Its pilot ran on the A770 at 18:10 CT (batch 8, eager; `pilot/load.json` records `attn_implementation: eager`).

| Gate | Result |
| --- | --- |
| Gate 1, zero hook equals plain | passed (3 of 3 questions) |
| Gate 2, first 40 generated tokens match Part 112 | **failed: 17 of 32, 28 required** (full text: 5 of 32, descriptive) |
| Gate 3, plain answers not moved | passed (3 of 3) |

By the registered rule, a failed gate stops interpretation: predictions 189.1 to 189.6 are withheld, nothing was
labelled or scored, and the full run was not started (`pilot/gates.json`).

## What the failure turned out to measure (exploratory, not registered)

Two pilots run after the failure with `part189/diag189.py` (`diag/`):

| Arm | Gate 2, first 40 tokens | Gate 2, full text |
| --- | --- | --- |
| SDPA, batch 64 (Part 148's configuration) | 32 of 32 | 32 of 32 |
| eager, batch 64 | 18 of 32 | 6 of 32 |
| eager, batch 8 (the Part 189 pilot) | 17 of 32 | 5 of 32 |

The code path reproduces Part 148 exactly on its own kernel and batch, and batch shape does not explain the eager
mismatch. The eager kernel changes pushed answers early: in the Part 189 pilot the first differing token sat at
positions 1 to 53 (for example 1, 3 and 4 in three of the 32), while the unpushed controls matched Part 148 until
token 52 on "Who are you?" and fully on the other two questions. Part 189's Gate 2 therefore tested kernel agreement,
not wiring; Part 189b moves the wiring check to the SDPA batch-64 arm.

Cold check: not yet run (the gate result and the diagnostic counts are copied from `pilot/gates.json` and
`diag/*/diag.json`).
