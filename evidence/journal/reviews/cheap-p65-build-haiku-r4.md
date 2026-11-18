<!-- cheap-review.mjs cheap-review-p65-build-haiku-r4-2026-10-24: /home/elisepc/green/worktrees/ab-part65 47e37c1..3629a47, lane haiku, df-run exit 0 -->
# REVIEW: p65-build (cold adversarial review, round 4)

Range: `47e37c1..3629a47`, worktree `/home/elisepc/green/worktrees/ab-part65`
Files: `part65/judge65.py`, `part65/analyze65.py`, `part65/test_part65.py`, `part65/RUNBOOK-65.md`, `PART65-DESIGN.md` (Amendment 4), `paidaction/paid_ledger.py` (read only).
Evidence: scripts in `scripts/`, outputs in `out/`, mutants in `mutants/`. All drive the real code. Only the provider API is faked; ledgers are temp files. No live call, no credential read, no write to the worktree.

## Verdict: SHIP-WITH-FIXES

The core safety property holds in everything I could measure. The real code never sent a chunk twice: 10 injected crash points, two real processes racing on two output directories (6 of 6 races, one sender), and the released-chunk stop. Three INTRODUCED findings should be fixed before the live judge runs: F1 (a released final chunk can pass the analysis gate), F2 (`--ledger` defeats the per-plan fence), and F3 (the lock test passes for the wrong reason). F4 to F6 are smaller INTRODUCED items. F7 to F12 are PRE-EXISTING follow-ups and do not hold this change.

Severity key: HIGH = a send or a wrong result reaches a report with no signal. MEDIUM = reachable with a deliberate flag or an operator action, or a test that gives false assurance. LOW = fail-closed or cosmetic-in-effect.

---

## Findings

### F1. MEDIUM, INTRODUCED. A released final chunk can pass the analysis gate when chunks are not 100 trials.
- Where: `part65/analyze65.py:277` (`missing_trial_count * 20 > total_trial_count`); `part65/judge65.py:793` (`--chunk-size`, default 100); `judge65.py:362` (the manifest records `reserve_total_usd` but no `chunk_size`); `judge65.py:761` (the loop breaks on released).
- Trigger: the released chunk is the last one, and it has 50 or fewer trials. The analysis counts only missing answers, and 50 of 1000 is exactly 5%, which is allowed. Measured: `test_missing_trial_stop_uses_the_registered_50_of_1000_boundary` (`test_part65.py:1062`) passes under the torch stub. It accepts 50 missing and refuses 51.
- Why it matters: Amendment 4 item 1 says a released chunk means the study "reports that the judge failed and gives no result for 65a to 65g". That holds only because the default chunk size makes a released chunk 100 trials, which is over the limit. With `--chunk-size 50`, or any size where the final chunk is at most 50 trials, the judge exits 2 but `analyze65` still reports. The analysis never reads the chunk states, and the chunk size is not recorded anywhere the analysis can see.
- Smallest fix: write `chunk_size` into `run.json`. The analysis then refuses any run with a chunk not in `settled` state, or with a `released`/`ambiguous` marker. Alternatively, the judge writes a `JUDGE-FAILED` file on release and the analysis refuses on it.

### F2. MEDIUM, INTRODUCED (reach). `--ledger` defeats the cross-directory plan fence.
- Where: `judge65.py:599` (`_prior_reservation` compares run identities only within the ledger file it was given); `judge65.py:790` (`--ledger`, which accepts any path).
- Measured (`scripts/s5_torn_tail_and_ledger_flag.py`, part b): output directory A ran with `ledger-1.jsonl`; output directory B, same plan, ran with `ledger-2.jsonl`. Result: `dir B ... returned; POSTs=6 (same plan as dir A)`. The whole plan was sent a second time. The `$1` sub-cap and the `$8` hard cap are also per ledger file, so the second ledger starts from zero.
- Why it matters: Amendment 4 item 2 says "before every send, the judge refuses if any earlier reservation for the same plan belongs to another run". That is true for one ledger only. The flag exists in the pre-change code (`judge65_before.py:666`), so the gap is PRE-EXISTING. But the fence that promises the guarantee is new, and it does not cover this path.
- Smallest fix: take the live ledger path from a constant. Keep `--ledger` only behind an explicit test-only flag, or refuse any non-default path when `--probe` or a full run is requested.

### F3. MEDIUM, INTRODUCED (test). The lock test passes without exercising the lock in most runs.
- Where: `part65/test_part65.py:722` (`test_shared_ledger_lock_serializes_fresh_directory_preflight_and_post`); `judge65.py:642` (the ledger-path lock).
- Measured: I replaced the lock with `if True:` (mutant m1, `mutants/m1/`). The test failed in only 4 of 20 runs (exit codes `0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0`). So 16 of 20 runs pass with the lock gone.
- The real effect is worse. With the same mutant, two real processes on two directories double-sent chunk 0 in 6 of 6 races (`out/s3b_m1_no_lock_race.txt`: "POST log entries=2 ... reserve rows=2"). The shipped code sent once in 6 of 6 races (`out/s3b_shipped_race_x6.txt`). So the shipped code is correct, and the test is what gives false assurance.
- Why it matters: the brief asks for tests that pass without exercising the property they name. This is one. A future edit that breaks the lock would most likely pass CI.
- Smallest fix: make the interleaving deterministic. Wrap `_prior_reservation` so the first run blocks inside the critical section, and have the second run's check be forced to arrive while the first holds the lock. Assert that the second run blocks, not merely that it refuses.

### F4. LOW, INTRODUCED. The $1 sub-cap is never tested through the real ledger, and the test double drifts from it.
- Where: `test_part65.py:243-284` (`FakeLedger`: `**_kwargs` discards `subcap_usd`/`subcap_key`, and it writes `amount_usd` where `paid_ledger.py:471-472` writes `reserve_usd`); `judge65.py:668` (`subcap_usd=PART65_CAP_USD`). No test asserts a sub-cap refusal.
- Measured (`out/s1_subcap.txt`): with the real ledger and a sub-cap that admits 2 chunks, a pending run stops at chunk 3 with `CapExceededError` and no POST for it. With the sub-cap ignored (the `FakeLedger` behaviour), all 6 chunks are sent.
- Smallest fix: add one test with the real `Ledger` and a shrunk `PART65_CAP_USD`. Assert that the refused chunk gets no POST.

### F5. LOW, INTRODUCED. A sub-cap refusal escapes `run_chunks` and `main` as an uncaught exception.
- Where: `judge65.py:763` (`except JudgeError` only in `run_chunks`); `judge65.py:814` and `:840` (`except JudgeError` only in `main`).
- Measured (`out/s1_subcap.txt`): `run_chunks RAISED paid_ledger.CapExceededError (JudgeError? False)`. Nothing was sent, which is correct. But the user gets a traceback with no "judge stopped" line, and the exit code is 1, not 2.
- Smallest fix: catch `LedgerError` in `run_chunks` and re-raise it as a `JudgeError` that names the sub-cap.

### F6. LOW, INTRODUCED (liveness). A torn final ledger line blocks the judge until some other paid action repairs it.
- Where: `judge65.py:585-587` (`_prior_reservation` `json.loads` on every line, raising on any invalid one); `paidaction/paid_ledger.py:259-265` (the paid ledger tolerates a torn final line) and `:384-390` (it truncates the tail on the next append).
- Measured (`out/s5_torn_tail_and_ledger_flag.txt`, part a): with a torn tail, the judge raised `JudgeError ... POSTs=0`. The same ledger read through `Ledger.state()` worked. After an unrelated paid action repaired the tail, the judge settled 6 of 6.
- Fail-closed, no send. Smallest fix: skip an unterminated final line the way `_read_locked` does, or call `Ledger` to read.

### F7. MEDIUM, PRE-EXISTING. Re-planning into a reused output directory silently drops the new plan's answers.
- Where: `judge65.py:487-495` (`_persist_answers` dedupes by `custom_id` only); `build_trials` restarts `custom_id` at `p65-0000` for every plan. The pre-change copy is identical (`judge65_before.py:439-447`).
- Measured (`out/s6_reused_dir_replan.txt`): after the operator deleted `run.json` and `chunks/`, a second plan (new seed) was run into the same directory. Result: 6 POSTs paid, 6 reserve rows, `totals settled 6`, exit 0. `answers.jsonl` still holds plan 1's 24 rows, and plan 2 persisted 0.
- Why it matters: the analysis then reads plan 1 only, with no error. The new per-plan fence does not block this, because the plan hash differs.
- Follow-up: dedupe on `(judge_run_id, custom_id)`, and refuse to append rows whose run ID differs from the file's existing rows.

### F8. MEDIUM, PRE-EXISTING (consequence new). A missing API key kills the registered plan before anything is sent.
- Where: `judge65.py:371-374` (`api_call` raises `JudgeError` before any socket); `judge65.py:704-706` (the `except Exception` in the POST path that classifies this as an unknown outcome). The pre-change copy has the same classification (`judge65_before.py:601-603`).
- Measured (`out/s8_no_key.txt`, no key in the environment, real `api_call`): `totals ambiguous 1`; chunk 0 `status='unknown_no_batch'`, `batch_id=None`, and the reservation stays active. `--recover` afterwards also reports `ambiguous 1`.
- Why it matters: a configuration typo makes the chunk "ambiguous". Under Amendment 4 that is terminal: the RUNBOOK says do not start another output directory. The $ reserve stays counted.
- Smallest fix: check the key before the first `ledger.reserve`, so a missing key is refused before any reservation exists.

### F9. LOW, PRE-EXISTING. A crash after the provider accepts a batch but before the batch ID is saved leaves the batch orphaned.
- Where: `judge65.py:720` (`state["status"] = "submitted"` is saved only after the POST returns).
- Measured (`out/s4_crash_sweep.txt`, "crash after provider accepted first POST"): the chunk stays `posting` with no batch ID, the reserve stays active, and the run stops `ambiguous`. No send was repeated, which is correct. But nothing can fetch that batch's results, so the plan is dead. The window is the same in the pre-change code.
- Smallest fix: retry the batch-ID save in-process, and print the POST response's ID to stderr so an operator can record it.

### F10. LOW, PRE-EXISTING. A missing chunk file exits 0 with unanswered trials.
- Where: `judge65.py:755` (`glob("chunk-*.json")`, no comparison with `run.json`'s `request_count`); `judge65.py:878` (exit status counts only the totals).
- Measured (`out/s2_missing_chunk.txt`): `manifest request_count=24`; `main --recover exit code=0`; `answers.jsonl rows=20 of 24`. With the default chunk size the analysis gate would catch it. With F1's chunk sizes it would not.
- Smallest fix: before running, assert that the chunk indices are exactly `0..n-1` and that they sum to `request_count`.

### F11. LOW, PRE-EXISTING. A `pending` chunk does not stop the loop, so every chunk is posted while earlier batches are still running.
- Where: `judge65.py:739-742` and the loop at `:745-770`. Only `released` and `ambiguous` break.
- Measured (`out/s1_subcap.txt`, pending run with no cap): `POSTs=6`, `{'pending': 6}`. Nothing settled before all six were sent.
- Follow-up: decide whether chunks are sequential. The RUNBOOK's advice to "use --recover" for a pending batch suggests a pending run should stop.

### F12. LOW, PRE-EXISTING. Two processes calling `prepare_run` on one directory can leave a manifest and chunks with different run IDs.
- Where: `judge65.py:325` (`manifest_path.exists()` is checked outside any lock, and prepare writes chunks before the manifest).
- Measured (`scripts/s7_prepare_race.py`, two processes, 20 trials): identity mismatch in 12 of 20 runs. `run_chunks` refused every mismatch, so no send happened in those cases. I did not measure the variant where the second writer overwrites a chunk after its POST.
- Follow-up: take the directory lock inside `prepare_run`, and create `run.json` exclusively.

---

## What I checked and found sound

- **No double send under crash.** `scripts/s4_crash_sweep.py`: a crash was injected before each of the 8 local saves, after the ledger reserve was fsynced, and after the provider accepted the first POST. Restart with the same ledger and directory. Duplicate sends: 0 in every case, and exactly one reserve row per chunk. The only stop was an ambiguous outcome, which is the intended safe stop.
- **Cross-directory race, shipped code.** `out/s3b_shipped_race_x6.txt`: two real processes, two directories, one real ledger, a 0.4 s delay in each POST. Six of six races sent 6 chunks from one process. The other was refused before any POST with "owned by another run".
- **Released chunk stops the judge.** `out/s3_races.txt`: 6 chunks, the second rejected with a validated 400. Only 2 POSTs were made. A rerun and a second directory both sent nothing.
- **Owner check is load-bearing.** Mutant m3 (`if False and ...` at `judge65.py:599`) made 4 test failures across the three fence tests.
- **Ledger row shape matches the reader.** `paid_ledger.py:471-473` writes `kind` and `meta` at the top level, which is what `_prior_reservation` (`judge65.py:592-593`) reads.
- **Pre-send record.** `_atomic_json` (`judge65.py:277-296`) fsyncs the temp file and the directory before returning. The `posting` marker is saved at `judge65.py:679`, before the POST at `:683`. Read, not crash-tested at the OS level.
- **Lock coverage.** The ledger-path lock (`judge65.py:642`) covers the ownership lookup, the reserve, the `posting` marker, and the POST. The state is re-read under the lock.
- **Recovery never POSTs.** `--recover` raises for `ready`/`reserved` (`process_chunk`), and refuses `posting`/`unknown_no_batch` without a batch ID. The tests for this pass.
- **`--reserve-total` sends nothing.** Its branch (`judge65.py:825-834`) calls only `prepare_inputs` and `_reserve_total`. `test_reserve_total_prints_real_plan_estimate_without_sending` (`test_part65.py:544`) passes.
- **Mixed-run refusal in analysis.** `analyze65.py:231-245` refuses rows from more than one run ID, plan hash, or output directory. `analyze65` also refuses duplicated and unregistered trials.
- **Release is validated.** `ledger.release` requires a 4xx other than 429 with a parseable Anthropic error body (`paid_ledger.py:515-538`). An invalid body is treated as ambiguous.
- **Blind prompt.** `custom_id` is `p65-NNNN`. The prompt carries no unit key, type or layer.
- **Test suite under a torch stub.** I could not run the suite as shipped, because `torch` is not installed here. Under a stub (`stub/torch`), 36 of 44 tests pass. The 8 errors are all in steering and model tests (`out/unittest_under_torch_stub.txt`: `torch stub has no randn`, and similar). No judge or analysis test fails. The "44/44 on the host" claim is therefore not reproduced.

## Not verified

- The reserve uses `ceil(bytes / 3)` as an upper bound on input tokens. I could not check that against the real tokenizer offline.
- The live API was never called, per the rules for this review. The batch behaviour is from the fake API only.
- The end-to-end 1000-trial analysis was not run. The F1 claim rests on the boundary test and the arithmetic.
- The F12 variant where a second writer overwrites a chunk after its POST was not measured.
- The first thread-based version of the prepare race hit a temp-file name collision between two threads of one PID. That is an artefact of threads, not of the CLI. I reran it with processes and report only the process result.
