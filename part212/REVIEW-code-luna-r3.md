<!-- cheap-review.mjs cheap-review-part212-code-luna-r3-luna-r3-2026-11-19: /home/elisepc/green/worktrees/ab-neon-review c521b428..cd314286, lane luna, df-run exit 0 -->
DO-NOT-SHIP

# Findings

## MEDIUM

### INTRODUCED — same-size resume refuses after a partially persisted generation call

- **File:** `part212/run212.py:282-289` (`stage2_validate_resume`).
- **Measured trigger:** `evidence_partial_full_resume.py` imports the real runner, builds its full 3,888-row plan, saves three complete rows from the first 8-row call, and validates resume with the original requested batch size of 8:
  ```
  planned_rows= 3888 finished_rows= 3 remaining_rows= 3885
  requested_batch_size_on_resume=8
  resume_validation= ValueError existing Stage 2 rows contain only 3 of 8 rows from a generation batch for arm=unchanged, question=0, set=None; refusing to regroup a partial batch
  ```
- **Why it matters:** interruption after some rows of a call were appended leaves valid finished rows, but the next same-size run refuses before model loading instead of skipping finished keys and completing the plan. This violates the assignment's same-size resume requirement.
- **Smallest fix:** make a generation call recoverable as a unit, such as atomically persisting all call rows before marking it complete; alternatively replay the original full call shape and append only keys not already saved.

### INTRODUCED — tail repair deletes a complete JSON row if its newline was not flushed

- **File:** `part212/run212.py:169-184` (`repair_partial_jsonl`).
- **Measured trigger:** `evidence_complete_jsonl_line.py` writes one valid row without its final newline, then calls the real repair helper:
  ```
  stage2: dropped 59 trailing bytes from incomplete JSONL line in /tmp/tmp_i4fs2xk/stage2.jsonl
  input_valid_json= {'arm': 'unchanged', 'answer': 'complete', 'batch_size': 8}
  input_bytes= 59 dropped= 59 after_bytes= b''
  ```
- **Why it matters:** a valid JSON object is complete even when its line terminator was not persisted. The repair discards it. If it belonged to a multirow call, resume then sees a partial call and refuses to continue.
- **Smallest fix:** parse the final non-newline-terminated bytes; preserve a valid row and append its newline. Truncate only an invalid trailing record.

### INTRODUCED — simultaneous stale-lock takeovers can remove a live lock

- **File:** `part212/run212.py:135-151` (`output_directory_lock`).
- **Measured trigger:** `evidence_stale_lock_race.py` coordinates two threads against a stale PID lock. Both observe the old dead owner; thread B then unlinks the fresh lock created by thread A:
  ```
  B: taking over stale output lock /tmp/tmp92jtxx2c/part212.lock; PID 999999 is not alive
  A: taking over stale output lock /tmp/tmp92jtxx2c/part212.lock; PID 999999 is not alive
  both_lock_contexts_entered= ['A', 'B']
  errors= []
  lock_exists_after= False
  ```
- **Why it matters:** the stale check is not tied to the file later unlinked. Two restarts can both enter Stage 1/2 on one output directory, defeating the lock and risking duplicate or interleaved rows and metadata.
- **Smallest fix:** serialize stale takeover with an atomic claim, then re-read and validate the lock while holding that claim before unlinking. Never unlink a path based only on an earlier PID read.

## LOW

### INTRODUCED — lock-file write failure leaves a zero-byte lock that blocks every retry

- **File:** `part212/run212.py:152-158` (`output_directory_lock` acquisition).
- **Measured trigger:** `evidence_lock_write_failure.py` forces the real lock helper's `os.write` to fail after creating the lock:
  ```
  acquire_error= OSError simulated disk write failure
  lock_left_after_failed_acquire= True bytes= b''
  retry_error= RuntimeError stage2-retry: cannot read owner PID from output lock /tmp/tmptd532ukf/part212.lock
  ```
- **Why it matters:** a transient lock-file write error leaves an unreadable lock even though acquisition never reached the protected work; later attempts cannot recover automatically.
- **Smallest fix:** on write or sync failure, close the descriptor and remove the lock created by this attempt before re-raising.

# What I checked and found sound

- The prior Gate 3 count-only check is fixed in code: the scorer imports `stage2_rows()` and checks exact equality between planned and actual keys while retaining duplicate detection.
- The F3 regression now runs the Stage 1 driver loop and observes pool-unit writes; it does not only exercise helpers. The specified CPU suite passed: `85 passed in 7.05s`.
- A complete same-size full-run resume is accepted and has no todo rows (`evidence_complete_same_size_resume.py`: `planned_rows=3888 saved_rows=3888 todo_rows=0`). The refusal above is limited to a partially persisted call.
- A synthetic exception from the real Stage 2 model-load call releases the lock (`evidence_model_load_exception_lock.py`: `lock_exists_after_model_load_exception=False`). The Windows branch is conditionally isolated behind `os.name == 'nt'`; the source imports no POSIX-only module. I could not execute that branch on a Windows host.
- Runner rows retain their prior answer, token, stop-reason, prompt, sample, and seed fields. The runner writes actual call size as `batch_size`; the scorer reports row counts by that size, while `run.json` reports generation-call counts. The two histograms have distinct labels and consistent meanings.
- The round-2 design count was corrected in documentation. I did not load real weights, call live services, or rerun the long tiny-model smoke.
