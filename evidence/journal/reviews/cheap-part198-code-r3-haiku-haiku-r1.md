<!-- cheap-review.mjs cheap-review-part198-code-r3-haiku-haiku-r1-2026-11-08: /home/elisepc/green/worktrees/ab-part198 1a05ea9..b3aacf2, lane haiku, df-run exit 0 -->
# REVIEW: part198 code, fold r2 (b3aacf2 over 1a05ea9), cold adversarial pass

Verdict: SHIP-WITH-FIXES

Reason: the N2, N3, N4 (code), N5, N6 and P1 (exception path) fixes are real and measured. One Major
regression was introduced by the N1 change: the scorer now silently keeps the first of two conflicting
case rows when they differ only in scored fields, where the pristine code refused. That breaks the
brief's "real duplicate always refused" requirement. Fix it before any registered scoring.

## Findings

### F1 — MAJOR — INTRODUCED — conflicting case rows are silently deduped when they differ only in scored fields
- File: `results/part198/score198.py:16-60` (`registered_rows`); decisive tuple at `score198.py:26-29`
  (case: `took, already_satisfied, name_eligible, restored_bit_exact, undo_zero_exact, n_written_so_far`);
  comparison at `score198.py:52-58`.
- The scorer reads many more fields from case rows than the tuple compares: `localization.*.localized`
  (forecast success, `score198.py:188`, `:244`, `:346`), `aimed_positive_control.oracle_named_strict`
  (S1, `:260`), `oracle_reader.named_strict` (`:247`), `blind_reader.*`, `random_control.*`,
  `edited_layer_stats`. None of these are in the tuple, so a row pair differing only there is accepted with
  first-wins.
- MEASURED (`probes/p_n1_scored_field.py`, two ROME rows for case 7, identical on the tuple, differing in
  `localization.t_word.localized` and `aimed_positive_control.oracle_named_strict`):
  ```
  == NEW (b3aacf2)
  A then B: kept=1 kept_run=r1 t_word_localized=True  S1_successes=1 forecast_successes=0
  B then A: kept=1 kept_run=r1 t_word_localized=False S1_successes=0 forecast_successes=0
  == OLD (1a05ea9)
  A then B: RAISED ValueError: conflicting terminal rows for case_id='7' ...: case then case
  B then A: RAISED ValueError: conflicting terminal rows for case_id='7' ...: case then case
  ```
- Why it matters: the scored S1 count (and the localization forecasts) now depends on which row the file lists
  first, with no error. A real duplicate with a different scored outcome is exactly what the brief says must
  be refused. The added N1 test only exercises tuple fields (`took`), so it does not catch this.
- Smallest fix: refuse on any scored boolean. Add to the case tuple the derived flags the scorer reads:
  `localization.<stat>.localized` for each stat, `aimed_positive_control.oracle_named_strict`,
  `oracle_reader.named_strict`, `blind_reader.localized`, `blind_reader.named_strict`, and
  `random_control.localization.t_word.localized`. Keep ignoring the raw float diagnostics. Add a test that
  two case rows differing only in `aimed_positive_control.oracle_named_strict` raise ValueError.
- Open design point (not decided here): these localized booleans are thresholds on floats, so genuine
  cross-process float drift can flip them. Refusing on flips would make some legitimate replays unscoreable
  again, which is what N1 was meant to stop. The fold needs an explicit rule for this; the code currently has
  none, it just picks the first row.

### F2 — MINOR — INTRODUCED (test gap) — the N4 structural test does not catch a duplicate pre-check terminal row
- File: `results/part198/test_build198.py:480-483` (N4 check, `find`-based ordering).
- The check only requires that some `append_jsonl(log, record)` appears after the zero guard. It does not
  forbid an earlier append. MEASURED: copy with an extra `append_jsonl(log, record)` inserted before
  `undone = w.undo()` (late append kept) gives `PASS N4 ...` and `TEST_BUILD ALL PASS` (`suite_sabA.txt`).
  The exact regression (early append, late one removed) gives `FAIL N4` (`suite_sabB.txt`).
- Mitigation, measured: the scorer refuses the early/late pair (`probes/p_n4_early_row.py` →
  `REFUSED: ... decisive fields differ: undo_zero_exact`). So this is a suite weakness, not a silent data
  error.
- Smallest fix: assert that `run_arm_b`'s source contains exactly one `append_jsonl(log, record)`, and that it
  is after the guard.

### F3 — MINOR — PRE-EXISTING (claim overstated) — the Arm C bound does not cover a hard process death during model load
- File: `results/part198/run198.py:1610-1700` (`run_arm_c`; failure counting at `:1615` and `:230-240`).
- The bound counts only rows with `record_type == "failure"`, and those are written only from an
  `except Exception`. A process killed during load (OOM-kill, SIGKILL) writes no failure row, so the
  bound never triggers.
- MEASURED (`probes/p_p1_retry.py`, real `run_arm_c` in a child process, four invocations):
  ```
  == NEW code, mode=raise
  invocation 1: child_rc=0 cumulative_loads=1 cumulative_failure_rows=1
  invocation 2: child_rc=0 cumulative_loads=2 cumulative_failure_rows=2
  invocation 3: child_rc=0 cumulative_loads=3 cumulative_failure_rows=3
  invocation 4: child_rc=0 cumulative_loads=3 cumulative_failure_rows=3
  == NEW code, mode=kill
  invocation 1: child_rc=137 cumulative_loads=1 cumulative_failure_rows=0
  invocation 2: child_rc=137 cumulative_loads=2 cumulative_failure_rows=0
  invocation 3: child_rc=137 cumulative_loads=3 cumulative_failure_rows=0
  invocation 4: child_rc=137 cumulative_loads=4 cumulative_failure_rows=0
  ```
- Classified PRE-EXISTING: the pristine code had no bound at all, and this change does not make the kill case
  worse. The exception case is fixed as claimed. BUILD-NOTES' "P1 — fixed" should say the bound covers
  exceptions only.
- Smallest fix (follow-up): append an attempt-started record before each load (as `append_run_event` already
  does for the pending list) and count those toward the cap.
- Also noted: `run_arm_c` refusal is only a raised RuntimeError. Nothing in the scorer reads armC failure rows,
  so the exhausted state is reported on stderr and by the missing T*, not in the score.

### F4 — PROCESS, for the parent — possible model/weight access during fold r2 (not a code defect)
- BUILD-NOTES "Old-code audit" says the fold worker ran `p3_main_dup.py` and `p3c_main_dup.py`; both reached
  Arm A `run_start` "immediately before the real Arm A loader call", and the worker "cannot establish whether
  either probe opened or loaded weights". The brief's hard limit is "do not load any model".
- I did not run those probes. The parent should establish from process/file-access evidence (not the
  worker's word) that no GPT-2-XL weights were read, and keep this in BLOCKERS.

## What I checked and found sound
- Fixed suite on fixed code: `fixtures198.py` exit 0 ("FIXTURES ALL PASS"); `test_build198.py` exit 0, 48 checks,
  `TEST_BUILD_CHECKS 48` (run in a scratch copy with `-I -B`).
- New suite on pristine `run198.py`/`score198.py` (with the new test file, crosscheck files copied): exit 1 with
  exactly 10 failing names: N1 replay/duplicate, N2 scalar fingerprint, N2 fingerprint error explicit, N3 single
  runner, N3 mass runner, N2 runner error status, P1 retry cap, N4 undo ordering, N5 preflight hash and marker,
  N6 missing control. The N5 Arm C resume test passes on old code, which is expected: it is a regression guard
  for the resume path, not a fix test. This matches the 10 failing checks in BUILD-NOTES.
- P1 exception path (F3 measurement): exactly 3 loads, 3 attempt_numbers, 4th invocation refused before load.
- N2 `tensor_fingerprint`: flatten precedes byte reinterpretation, so 0-d tensors work. The header still uses
  the original shape. Errors are wrapped with the parameter or buffer name.
- N2 runner restoration: a fingerprint exception is recorded as `restored_bit_exact: None` plus
  `restoration_verification_error`, never as `False`. The stop-on-unverified condition is `restored_after_error is not True`,
  which treats None as unverified. Gate checks use `is not True` too, so a None cannot pass.
- N3: the fake-run tests drive the real `run_arm_a` / `run_arm_a_mass` through patched collaborators, and the
  editor mutates an unreported buffer. The tracked verifier wraps the real function (captured before patching),
  so the check is not circular.
- N4 code: the terminal append is after `w.undo()` and the zero check; the except path still runs and writes a
  failure row. The scorer's decisive tuple refuses an early/late pair (F2 measurement).
- N5 preflight: the stats hash is checked against the live `stats198.py`; the check count comes from `len(check_names)`,
  not a constant; the 30-check floor is enforced. The crosscheck writer (`crosscheck198.py:78`) emits
  `stats198_sha256`, so the regenerated JSON is consistent with the script.
- N6: `missing_controls` now counts reason-only or missing-`oracle_named_strict` controls. Explicit `false` still
  counts as a measured failure, and `s1_success` counts only True.
- Smoke isolation: `registered_rows`, `all_registered`, `existing_keys` and `arm_c_failure_counts` all filter out
  `run_mode != registered` (smoke rows never affect registered outcomes).
- P2 / P3 "no change": the cited design lines (`PART198-DESIGN.md:130-131`, `:393`, `:90-91`) say what
  BUILD-NOTES claims: a batch below the take gate is left out, a forecast is unevaluable only when no batch
  reaches the gate, and the blind reader is computed without edit metadata.
- Gate: any keyed failure row in a registered log fails the gate (`score198.py` restoration checks). So the
  N1 "failure then completed case" acceptance cannot produce a gate pass. It only affects diagnostics. The
  runner never produces that sequence either: `existing_keys` treats failure rows as done for Arm A and B.
- Scope: `stats198.py` and the `crosscheck198.json` hash are unchanged in the diff.
- Run-level diagnostics (`probes/p_n1_runlevel.py`): the runner's own top-level failure row (main()'s handler, which
  has no `run_id`) is surfaced in `run_level_failures`, because that field is built from the raw
  `all_registered` rows, not from the deduped set. Not a finding; I initially read it the other way.
- Restart between steps for Arm C: each invocation was a fresh process reading the log, and the cap held across
  restarts (F3 measurement, invocations 1 to 4).

## Not measured
- Two concurrent runner processes on one log. From the code: each invocation takes its failure count as a
  snapshot at start, so N concurrent invocations can exceed the Arm C cap. The runner has no lock. Pre-existing
  and out of scope for this fold; I did not run it.
- Any run that loads a model. Per the brief, none was run here; the fold-r2 probes in F4 are the only
  loader-reaching runs I know of from the history, and I could not verify whether they loaded weights.
