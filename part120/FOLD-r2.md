# Part 120 review fold r2

Review: `cheap-ab-part120-code-r2-ds-r2.md`. Verdict: ship with fixes.

- R-1: Removed the vacuous test that called different fixture formulas an independent weight rebuild. Its replacement gives the 0.1 cache a deliberate 10% error and checks that Gate 3 rejects the adjacent-dose identity. The registered Gate 3 rule remains `push(2 alpha) = 2 * push(alpha)` within relative error 1e-5 for each unit and arm. The runner's extra call to `run119b.build_dose_push_cache` uses the same builder as the cache under test. The fold and build notes now describe it as a same-code-path consistency check. It does not test a changed builder.
- R-2: The runner now records a best-effort aborted `timing.json` for exceptions across initialization, gate checks, and unit execution. The record includes the exception type and message. `BaseException` also catches `KeyboardInterrupt`, then re-raises it. A prior complete main timing record is preserved when a failed or refused rerun targets the finished result directory. Tests cover preflight `ValueError`, unit-loop `RuntimeError`, `KeyboardInterrupt`, and an `OSError` while writing a unit.
- R-3: Corrected `FOLD-r1.md` and `BUILD-NOTES.md`. F1 and F2 were round 1 misreadings, not unresolved defects. F5's test did not demonstrate an independent rebuild. The empty-vector headline is reachable as a library result, but the CLI rejects missing unit files before that branch.
- R-4: Recorded the CLI limitation for empty vectors in the fold and build notes. No scorer change was made because this was outside the requested round 2 fixes.

The completed run was not repeated. The runner change affects abort records only. No result vectors, dose values, score formulas, or registered predictions were changed.

Verification: `/home/elisepc/neuron-probe/.venv/bin/python -B -m pytest -q -p no:cacheprovider part120` passed **32 tests** in 2.05 seconds. The result figure and numbers were recomputed from the saved NPZ vectors by `results/part120/fig120.py` and matched the recorded derived score fields exactly.
