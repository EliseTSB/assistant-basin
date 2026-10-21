# Part 43 singleton fold 3 report

## SUMMARY
The calibration gate now derives repeat spread, bf16/fp32 differences, replay error, and epsilon from saved repeat scores. Stored summary values are checked against those calculations with an absolute 1e-9 tolerance.

## FINDINGS
The gate validates required leg and per-cell structures before using them. Missing rows, malformed numeric values, repeated answers, or shape coverage that differs from the registered maker/answer set refuse with a reason before the pilot runs decision work. The gate also validates recorded cell spreads and primary scores against their repeats, and validates both replay-leg summaries against raw clean and replay scores.

The positive test fixture now records a clean replay consistent with its clean score in both dtype legs. Its short and long answer fields are derived from its shape rows. Added tests mutate replay, repeat, and fp32 raw scores while leaving summaries untouched; each is refused before the lattice. Tests cover missing rows across bf16, fp32, and per-cell errors for every registered cell, duplicate answer coverage, and the positive lattice path.

The specified command, `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part43.py`, passed: 52 tests. `git diff --check` passed. No parent-control messages were received. No commit was created.

## BLOCKERS
None.
