<!-- cheap-review.mjs cheap-review-part198-code-r4-luna-luna-r1-2026-11-08: /home/elisepc/green/worktrees/ab-part198 91b6af4..dbe6d56, lane luna, df-run exit 0 -->
# Cold adversarial review: part198-code-r4-luna

## Verdict

**SHIP-WITH-FIXES.** Fold r3's replay check still misses one scorer-used boolean in the real runner row shape. I also measured two issues already present before fold r3; they are follow-ups, not reasons to hold r3.

## Findings

### MINOR — INTRODUCED — The case replay check misses aimed-control localization in the real row shape

- Files: `results/part198/score198.py:33-34`; the runner writes the value at `results/part198/run198.py:1029-1032`, and the scorer consumes it at `results/part198/score198.py:417-421`.
- The runner stores `aimed_positive_control.localization` as one T_word result, e.g. `{"value": 2.0, "localized": true, "tie": false}`. `_case_scored_booleans()` passes that object to `_localized_outcomes()`, which expects a map of statistic names to result objects. For this real shape it records `localized` as false regardless of its actual value, so two replays that differ only on this scored boolean are accepted.
- MEASURED with `python3 probe_aimed_localization_replay.py` against the real `score198.registered_rows()`:
  ```
  accepted=True selected_aimed_localized=False
  accepted=True selected_aimed_localized=True
  ```
  Reversing row order changes which aimed-control localization is retained. The added test fixture at `test_build198.py:61-63` supplies a statistic map (`{"t_word": {"localized": ...}}`), unlike the runner's actual row, so it misses this shape.
- Why it matters: the descriptive 198.5 localization count changes with JSONL order despite contradictory terminal rows; fold r3's stated rule is to refuse a scored-boolean conflict.
- Smallest fix: compare `aimed_positive_control.localization.localized` directly (with the same missing/false semantics as `flag()`), and add a replay test using the runner-shaped result.

### MAJOR — PRE-EXISTING relative to fold r3 — Conflicting mass-batch localization replays can flip a forecast

- Files: `results/part198/score198.py:103-108` and `:238-248`.
- `registered_rows()` adds the new boolean comparison only when `kind == "case"`. Mass forecasts consume `batch.localization.<stat>.localized`, but repeated batch rows only compare `n_took`, `n_already_satisfied`, `take_gate_passed`, and restoration. The first batch row wins when localization differs. This gap was already present in b3aacf2; fold r3 leaves batch rows outside its new check.
- MEASURED with `python3 probe_batch_replay_conflict.py` against the real scorer:
  ```
  accepted=True selected_f1_localized=False forecast_successes=0 status=PASS
  accepted=True selected_f1_localized=True forecast_successes=1 status=FAIL
  ```
- Why it matters: for a counted mass batch, conflicting terminal rows can make the 198.1 or 198.3 result depend on JSONL order, including changing PASS to FAIL. Concurrent or replayed batch runs can produce such rows.
- Smallest fix: compare the batch's scored localization booleans on repeated `batch_id` rows too, and add a test where only `localization.f1.localized` changes. This is a follow-up to the prior deduplication change, not a fold-r3 blocker.

### MINOR — PRE-EXISTING relative to fold r3 — Concurrent Arm C starts can exceed the three-attempt cap

- File: `results/part198/run198.py:1624-1646` (same snapshot-before-loop pattern was at b3aacf2 in `arm_c_failure_counts()` / `run_arm_c()`).
- Each invocation reads the attempt count once, before writing its own `attempt_started` record. There is no lock or atomic reservation around that read and append. Overlapping invocations can all observe zero and each pass the cap.
- MEASURED with `python3 probe_armc_concurrency.py`; all loader calls in the probe are replaced with a fake that raises before model access:
  ```
  concurrent invocations=4 loader_calls=4 cap=3 started_attempts=4 attempt_numbers=[1, 1, 1, 1]
  next sequential invocation new_load_calls=0 refusal=Arm C attempt limit reached for 'cal-0': 4 attempts; refusing another model load (max attempts 3)
  ```
- Why it matters: concurrent registered runs can exceed the advertised per-model load-attempt bound, although a later sequential run sees the overage and refuses. The same snapshot race was present in fold r2, so this is a follow-up rather than a regression from r3.
- Smallest fix: under a per-output-log lock, recalculate that model's count and append its start reservation atomically before each load.

## What I checked and found sound

- Fold r3 F2's runner path has one Arm B terminal `append_jsonl(log, record)`, after `w.undo()` and the exact-zero guard (`run198.py:1372-1378`); the revised test counts appends and its sabotage inserts an early duplicate.
- Fold r3 writes and fsyncs each `attempt_started` before entering the loader; the count pairs new starts with exception rows and retains old failure-only attempts. The cap refuses a later sequential invocation after three or more observed attempts, as the concurrency probe's final call demonstrates.
- The scorer's forecast thresholds and denominators match the design table: single F1/PCS at most one, naming at most two, WSP localization at least 20, blind joint success at least 18, and sequence strict naming at least five eligible objects. A failed gate makes forecasts not evaluable.
- No model, weights, or live service was accessed. No runner arm or `main()` was run. The supplied fixture/test suite was not run because `test_build198.py` imports Torch; the review probes imported only `run198` and `score198` and patched the only Arm C model-loader entry point before driving it.

## Review artifacts

The measured probe scripts are `probe_aimed_localization_replay.py`, `probe_batch_replay_conflict.py`, and `probe_armc_concurrency.py` in this review's scratch domain. No project files were modified.
