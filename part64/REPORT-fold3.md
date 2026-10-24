# Part 64 fold 3 — Amendment 3 implementation

## SUMMARY

Implemented Amendment 3's writer-stage budget, root resolution, probe projection, recovery, retry-ID, and JSONL parsing changes. The full required Part 64 test suite passes (47 tests). No paid API calls, network access, GPU work, or commits were made.

## FINDINGS / WHAT SHIPPED

1. **One shared writer stage ledger directory.** All writer ledger paths now resolve to `results/part64/writer-stage/` from the repository root, independent of the run's `--root`. The guard scans all ledgers for the writer, across run IDs, and serializes its check and reserve under a shared stage lock. Reservations also retain their absolute run-root identity so copied run roots remain refused. The cross-root regression is `ChunkRecoveryTests.test_writer_stage_guard_shares_cap_across_roots_for_same_collection`; existing crash accounting is covered by `ChunkRecoveryTests.test_writer_stage_guard_counts_settled_and_reserved_ledgers_after_crash`. Both use a temporary stage directory.

2. **Absolute run roots.** `run_probe`, `run_generation` (the run and resume flows), `reprobe`, `status_report`, `retire_attempt`, and CLI dispatch resolve roots to absolute paths. The real probe flow reaches the fake provider with a relative run root in `WriterRequestTests.test_real_run_probe_resolves_relative_root_before_reaching_fake_api`.

3. **Stage caps.** `WRITER_STAGE_CAP_USD` is now `{"haiku": 4.0, "opus": 40.0}`. `ChunkRecoveryTests.test_built_plan_keeps_each_full_ledger_share_under_eight_dollars` checks the registered values and per-ledger planning bound.

4. **Probe cost gate and runbook.** After probe attempts settle, the `probe` CLI prints each writer's billed input/output usage, complete planned stage request count (including its probe), and projected batch cost. Opus estimates above $36 print `STOP`. `WriterRequestTests.test_probe_stage_projection_arithmetic_and_opus_stop_line` checks both arithmetic and the line. Section 3 of `RUNBOOK-64.md` instructs the operator to stop and report when it appears.

5. **Retirement scan window.** The five-minute created-at window now filters only re-probe attempts (`probe` with `attempt_no > 0`). Group chunks, initial probes, and all older listed batches are examined; old non-ended or unreadable group batches remain unresolved. Coverage: `ChunkRecoveryTests.test_retire_guard_matches_old_group_batch_without_five_minute_window` and `ChunkRecoveryTests.test_reprobe_recovery_ignores_older_ended_batch_even_if_id_matches`.

6. **Re-probe custom IDs.** Each chunk stores its base custom IDs. Retries use `base_id-attempt-NNN`, carry the base IDs forward, and reject IDs longer than 64 characters. `ChunkRecoveryTests.test_reprobe_custom_ids_use_stored_base_and_stay_within_64_characters` follows four retries; lost-response recovery remains covered by `ChunkRecoveryTests.test_reprobe_has_fresh_custom_id_and_recovers_accepted_lost_post`.

7. **Results JSONL line splitting.** `_parse_result_jsonl` now splits only on `\n`, preserving Unicode line separators inside JSON strings. Coverage: `WriterRequestTests.test_results_jsonl_parser_splits_only_on_line_feed`.

Verification command: `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part64.test_collect64 part64.test_collect64_model part64.test_gen64 part64.test_analyze64` from the domain root. Result: `Ran 47 tests ... OK`. `git diff --check` also passed. The production writer-stage directory was absent after testing; the tests redirected ledger writes to temporary directories.

## BLOCKERS

None. Parent-control mailbox had no messages; no plan changes or deferred instructions.
