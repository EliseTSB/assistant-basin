# Part 65 fold report — review round 4

## SUMMARY

Implemented the requested judge and analysis safeguards for F1, F2, F3, F4, F5, F7, F8, and F10. Added property based regression coverage with explicit mutant descriptions. The Part 65 suite passes 51 tests. No GPU or paid API calls were used.

## FINDINGS

- **F1:** `run.json` records `chunk_size` and `chunk_count`; completion is written only after every chunk settles and all registered answers persist, while released or ambiguous outcomes record failure. Analysis requires matching completion and answer run ID/plan hash. Tests: `test_validated_release_is_terminal_stops_later_chunks_and_main_exits_2`, `test_file_analysis_requires_a_completion_marker_matching_answer_identity`, `test_completion_marker_requires_settled_chunks_and_all_saved_answers`; catch missing, stale, mismatched, or premature completion mutants.
- **F2:** Removed `--ledger`; `main` uses the module default, and test runs pass a path to `run_chunks`. Test: `test_judge_cli_cannot_select_an_alternate_ledger`; catches a mutant that restores user-selected ledger routing.
- **F3:** The lock test blocks run one inside `_prior_reservation`, observes run two attempting the shared lock, and asserts run two cannot enter preflight. Test: `test_shared_ledger_lock_serializes_fresh_directory_preflight_and_post`; the no-op lock mutant failed 10/10 runs.
- **F4:** Exercised the real `Ledger` with a small Part 65 sub-cap and maximum fake provider usage; the refused chunk remains ready and receives no POST. Test: `test_real_ledger_subcap_refuses_before_post_and_main_exits_2`; catches ignored sub-cap enforcement or a POST before refusal.
- **F5:** Converts `CapExceededError` into a `JudgeError` naming `part65-judge`, stops before later chunks, and makes `main` return 2. Test: `test_real_ledger_subcap_refuses_before_post_and_main_exits_2`; catches escaped ledger errors or a successful exit.
- **F7:** Answer persistence dedupes by `(judge_run_id, custom_id)` and rejects mixed run IDs. Test: `test_persist_answers_dedupes_by_run_and_custom_id_and_rejects_a_second_run`; catches custom-ID-only dedupe and cross-run append mutants.
- **F8:** A live run checks for an API key before creating a ledger reservation. Test: `test_missing_api_key_refuses_before_the_first_reservation`; catches missing-key classification after reservation.
- **F10:** Before chunk processing, validates exact indices `0..n-1` and the sum of chunk trials against manifest `request_count`. Test: `test_run_chunks_rejects_missing_or_incomplete_chunk_files_before_reserving`; catches skipped chunks or incomplete trial totals.
- **Left as the review allows:** F6, F9, F11, and F12.
- **Mutation runs:** Ran the no-op lock mutant for 10 iterations; all 10 failed at the lock assertion. No other executable mutant fixtures were present in the review workspace to run.
- **Verification:** `PYTHONPATH=/home/elisepc/green/worktrees/ab-part65 /home/elisepc/neuron-probe/.venv/bin/python -m unittest -v test_part65` — 51 tests passed. `git diff --check` passed.

## BLOCKERS

None.

## Claude, after collection (2026-10-24)

Host: 51 tests, OK; `--ledger` absent from judge65.py. Mutant of my own: skipping the completion-marker check in analyze65 fails `test_file_analysis_requires_a_completion_marker_matching_answer_identity` and `test_validated_release_is_terminal_stops_later_chunks_and_main_exits_2`; restored by hash.
