# Fold brief: Part 43 singleton, fold 2 (Luna) — Amendment 4 calibration + gate

Run size: code and CPU tests only, no GPU run.

Read PART43-DESIGN.md "Amendment 4" and review/part43/ALDEN-10.md. Fix B1 and B2 exactly:

1. A new `calibrate` phase (CLI command `calibrate`, and the first step of `pilot`) that is measurement-only:
   - the cells Amendment 4 item 1 lists (clean, one selected zero, all-four zero, +/- dose, one rescue case; at least
     one short and one long answer; the primary maker question), primary full-answer score f per cell;
   - 3 bf16 singleton repeats per cell (spread = max - min);
   - an fp32 leg: load the same model revision in fp32 (a second load is fine; free the bf16 model or run the legs in
     sequence if memory needs it), same cells, same shapes, |f_bf16 - f_fp32| per cell;
   - clean-to-clean replay error;
   - finite-value, dtype and coverage checks (every listed cell present in every leg);
   - epsilon = max of the three error kinds, serialized with per-cell errors to calibration.json BEFORE anything else.
   Outputs are labelled non-decision data.
2. A gate: `pilot` (and any decision-producing command) reads the calibration result and proceeds to signs/lattice
   ONLY if calibration is complete, finite, and epsilon <= 0.05 nats. Otherwise it writes a HOLD record naming the
   reason and returns without calling signs, the lattice, swap or attach.
3. Paired tests (TinyModel, CPU): missing calibration, NaN, an incomplete cell set, and epsilon = 1 each never reach
   the lattice (assert the lattice function is not called); a valid below-budget calibration does reach it.
   A sabotage that removes the gate must fail at least one of these.
4. Do NOT implement the c*epsilon intervals, NOT_TESTABLE handling or the rescue-ratio guard (a separate later gate).
   Do not change the 0.05 bar or any pre-registered number.

Keep every existing test green. Commit nothing. Report REPORT-43-singleton-fold2.md ending SUMMARY / FINDINGS / BLOCKERS.
