<!-- cheap-review.mjs cheap-review-part198-code-r5-luna-luna-r1-2026-11-08: /home/elisepc/green/worktrees/ab-part198 dbe6d56..53284e1, lane luna, df-run exit 0 -->
SHIP-WITH-FIXES

## Findings

### HIGH — INTRODUCED — scoring is outside the per-log lock and can truncate an active append

**Location:** `results/part198/score198.py:229-245,430-438`; shared reader `results/part198/run198.py:149-172`; new runner lock `run198.py:131-146,1789-1815`.

`main()` holds the new exclusive lock around runner preflight and arm execution, but neither `score_from_dir()` nor the scorer CLI acquires that lock. Scoring reaches `R.read_jsonl()` through `all_registered()` and `registered_rows()`. That reader repairs an unterminated final row by truncating it. A scoring process can therefore read a log while its runner owns the lock, mistake an in-progress tail for a crashed append, and truncate it. The lock does not protect this reader or its recovery write.

**Measured trigger:** `probe_concurrent_score_read.py` imports the real `run198.py` and `score198.py` without torch. A child holds the real `output_log_lock()` and pauses after writing a partial second JSONL row; the parent calls the real `registered_rows()` while the lock is held. Output:

```text
torch_imported: False
runner_lock_held_during_score_read: True
scorer_selected_case_ids: [1]
jsonl_size_before_score: 148
jsonl_size_after_score: 91
tail_dropped_by_real_reader: True
recovery_events: [{'path': '.../armA-rome.jsonl', 'line': 2, 'dropped_bytes': 57, ...}]
```

This can lose a terminal record or leave the writer's later bytes malformed, changing resume state and potentially the score. The probe pauses at the partial-write state; it establishes the unsafe reader behavior, not how often a normal filesystem write exposes that state.

**Smallest fix:** Make `score_from_dir()` coordinate with the same lock files before reading logs (in a fixed order, with a shared/blocking mode or a clear refusal), so the scorer cannot invoke the truncating reader while a runner owns a log.

### MEDIUM — PRE-EXISTING — a failed tenth sequence write can make 198.8 evaluable and pass

**Location:** `results/part198/run198.py:1512-1517`; `results/part198/score198.py:193-202,359-375`.

The sequence runner appends a keyed failure and stops on a write/score exception. `gate_check()` does not inspect `all-armB-seq` for those failures. The 198.8 status check counts selected rows, where a keyed failure is kept as a row when that key has no completed case; it requires `len(seq) == 10`, not ten completed sequence cases. Thus nine completed writes plus a failure for write ten meet the length condition. The scorer then computes the naming threshold using only the nine eligible completed cases.

**Measured trigger:** `probe_sequence_failure_score.py` builds a temporary synthetic log set that passes the real scorer gate, supplies nine completed sequence case rows with five names, and adds a keyed failure for case ten. It calls the real `score_from_dir()`:

```text
gate_passed: True
sequence_completed_rows_plus_failure: 10
sequence_completed_cases: 9
sequence_terminal_failure_present: True
198.8_status: PASS
198.8_successes_and_denominator: 5 9
```

A registered sequence that never completed its tenth write can therefore report a passing forecast. This path is reachable from the runner's exception handler and is independent of fold r4; the scorer logic is unchanged in this commit.

**Smallest fix:** Make 198.8 NOT EVALUABLE unless there are ten completed unique case rows and no registered sequence failure row; use the full sequence log for the failure check.

## What I checked and found sound

- `registered_rows()` now compares the complete JSON row recursively, apart from `timestamp` and `run_id`, distinguishes JSON/Python types, compares floats exactly, and reports the first differing field. I found no legitimate runner path that emits two differing completed rows for the same registered case or batch: Arm A and mass resume skip keys already represented by terminal rows; Arm B sequence replays its prefix but appends only keys absent from the log; Arm C retries failures but records successful models under its own attempt/success mechanism rather than `registered_rows()`.
- Failure followed by completion remains accepted by `registered_rows()` because it compares completed rows against one another and prefers a completed row over failures. Smoke rows are filtered out of registered scoring.
- The new runner lock is nonblocking and is acquired before `preflight()` and arm dispatch. Its `finally` unlock is present, and the kernel releases the `flock` on process death. `probe_lock_crash_release.py` killed a child holding the real context manager; output was `holder_killed: True` and `lock_acquired_after_process_death: True`.
- I did not run any arm, model loader, weights, network call, or torch import. The two behavioral probes used synthetic temporary JSONL only.
