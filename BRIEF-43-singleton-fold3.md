# Fold brief: Part 43 singleton, fold 3 (Luna) — the gate derives, it does not trust

Run size: code and CPU tests only. Fix review/part43/ALDEN-13.md exactly. The rule behind all of it: **the gate
recomputes every error it uses (repeat spreads, bf16-vs-fp32 differences, replay error, epsilon) from the raw saved
per-cell scores, using the same definitions as _run_calibration_leg / _assemble_calibration; recorded summary fields
are only checked against the derived values (mismatch beyond 1e-9 = refusal), never used.**

1. Derived errors as above, per leg; epsilon from derived values only.
2. Structure validation before any indexing: every registered cell, leg and numeric field present and finite;
   any malformed valid-JSON structure returns a specific refusal (never an exception).
3. answer_options and shape rows must equal the registered unique maker/answer set (no duplicates, none missing).
4. Fix _calibration_fixture so the positive fixture is genuinely consistent.
5. Tests: changed-replay/unchanged-summary, changed-repeat/unchanged-spread and changed-fp32/unchanged-diff each refused
   and never reach the lattice; each missing per-cell row refused (parametrised over cells); duplicate answer refused;
   the consistent fixture reaches the lattice. A sabotage that makes the gate read a recorded summary instead of the
   derived value must fail at least one test.
6. Do not change 0.05, any pre-registered number, or anything outside the gate, fixture and tests.

Keep every existing test green. Commit nothing. Report REPORT-43-singleton-fold3.md ending SUMMARY / FINDINGS / BLOCKERS.
