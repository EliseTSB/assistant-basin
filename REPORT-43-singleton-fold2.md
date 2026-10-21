# Part 43 singleton calibration fold 2

## Implementation details

Added a measurement-only `calibrate` CLI command and made calibration the first model measurement in `pilot`. Each leg scores the primary full-answer maker score on the discovery question for clean, L7 zero, all-four zero, positive and negative L7 dose, L7 zero with L11 restored, and clean-to-clean replay. The bf16 leg repeats each cell three times. The fp32 leg loads the same pinned model revision and runs the same fixed-shape cells. The bf16 model is released and its device cache cleared before the fp32 load; pilot reloads bf16 only after the calibration artifact passes its gate.

`calibration.json` records per-cell primary scores, answer log probabilities, answer/shape coverage, bf16 repeat spreads, bf16/fp32 differences, replay error, dtype and finite-value checks, and epsilon as the maximum of the three registered error types. Nonfinite measurements are serialized as JSON null while their failed checks keep the calibration incomplete. The artifact is labelled as non-decision data.

The pilot reads the serialized artifact before calling signs, lattice, swap, or attach. Direct decision-producing commands also read the calibration in the output directory and write `HOLD.json` with a reason if it is missing, malformed, incomplete, nonfinite, shape/dtype-inconsistent, from another model revision, or over 0.05 nats. No c*epsilon intervals, NOT_TESTABLE handling, rescue-ratio guard, or threshold changes were added.

## Verification

Ran `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part43.py`: 26 passed. The CPU tests exercise both calibration legs on TinyModel and verify that missing calibration, NaN, incomplete coverage, and epsilon 1 do not reach pilot decision stages, while a valid below-budget calibration reaches the lattice. Also ran `py_compile` on `part43.py` and `part39.py`, and `git diff --check`; both passed. No real model calibration or GPU run was performed, per the brief.

## SUMMARY

Implemented Amendment 4 measurement calibration and the 0.05-nat decision gate. Tests and static checks pass.

## FINDINGS

The gate is enforced for the pilot and direct Part 43 decision commands. Calibration runs bf16 and fp32 sequentially and records complete cell-level error data before the pilot proceeds. No commit was created.

## BLOCKERS

None for this code and CPU-test scope. Real model runtime and hardware behavior remain unmeasured in this fold.
