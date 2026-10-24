# Part 65 Amendment 4 Report

## SUMMARY
Implemented the terminal-release policy, run identity and cross-directory paid-send fencing, durable chunk state, answer provenance checks, actual-request reserve preflight, and runbook updates. The requested fake-API regressions and existing tests pass; no live API, GPU, model download, steering edit, or paid-ledger edit was used.

## FINDINGS
- `judge65.py`: each run, chunk, reservation, and answer row now carries a random judge run ID and resolved output directory with the plan hash; reopening preserves and validates identity (`test_run_identity_survives_prepare_reopen_and_is_stamped_on_each_chunk`, `test_paid_batch_lifecycle_settles_and_resume_does_not_resubmit`).
- `judge65.py`: a ledger-path lock now spans plan ownership lookup, reserve, durable posting marker, and POST; both ready and reserved paths reject foreign runs and terminal attempts (`test_cross_directory_plan_fence_rejects_active_released_and_settled_reservations`, `test_copying_a_reserved_chunk_to_another_run_is_refused_before_post`, `test_shared_ledger_lock_serializes_fresh_directory_preflight_and_post`, `test_same_run_terminal_release_settlement_and_retirement_attempts_never_resubmit`).
- `judge65.py`: validated 4xx release returns and counts `released`, stops later chunks, prints the terminal report, and exits 2; malformed releases, 429s, and unknown no-ID outcomes stop as ambiguous, while `--recover` never POSTs (`test_validated_release_is_terminal_stops_later_chunks_and_main_exits_2`, `test_invalid_release_body_and_429_are_ambiguous_and_stop_later_chunks`, `test_ambiguous_posting_marker_is_never_reposted`, `test_recover_refuses_ready_chunk_without_posting`).
- `judge65.py`: `_atomic_json` fsyncs the temporary file, replaces the destination, and fsyncs its parent directory (`test_atomic_json_fsyncs_temp_then_replacement_directory`).
- `analyze65.py`: answer rows must include run ID, plan hash, and output directory, and a mixed run/plan is rejected even for disjoint trial keys; empty input still reaches the registered missing-trial stop (`test_analysis_rejects_mixed_or_missing_run_provenance_even_for_disjoint_rows`, `test_empty_answers_reach_registered_missing_trial_stop`).
- `judge65.py`: `--reserve-total` prints the whole plan's worst-case reserve from built request payloads without sending, and the selected run records its reserve total (`test_reserve_total_prints_real_plan_estimate_without_sending`).
- `RUNBOOK-65.md`: the pre-probe reserve command and $1 cap are documented; release is terminal, partial artifacts are preserved, and no new output directory is allowed for the plan.
- Verification: `PYTHONPATH=/home/elisepc/green/worktrees/ab-part65 /home/elisepc/neuron-probe/.venv/bin/python -m unittest test_part65` — 44 tests passed. The five Chorus mutant cases are named in test docstrings; standalone mutated-source runs were not performed.

## BLOCKERS
None.

## Claude, after collection (2026-10-24)

Host: 44 tests, OK. Three mutants of my own, each restored by hash: the cross-run fence disabled fails the two fence tests (four failures); continuing after a release fails `test_validated_release_is_terminal_stops_later_chunks_and_main_exits_2`; allowing mixed runs in the analysis fails `test_analysis_rejects_mixed_or_missing_run_provenance_even_for_disjoint_rows`.
