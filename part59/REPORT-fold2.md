# SUMMARY

Fold 2 fixes the runner, analysis, blind reader, packaging, tests, and runbook items from the two reviews. The registered predictions and pooled rule are unchanged. All 16 tests pass; four sabotage checks fail as expected. No control mailbox messages were received.

# FINDINGS / WHAT SHIPPED

1. **Independent runner phases and preflight — fixed in `run59.sh`.** GPU phases record their own result and continue after unrelated GPU-phase failures. Only each 59B analysis depends on its matching scan; a model's phases also require its own successful prefetch. A failed Qwen prefetch stops before the OLMo prefetch, while a failed OLMo prefetch does not cancel Qwen. Failed preflight skips pip, inventory, and prefetch work. The runner checks aggregate status between model prefetches and prints `RUN_FAILED` with an incomplete archive unless all statuses and archiving succeed.
2. **Completed-output CLI reruns — fixed in `run59a.py` and `run59b.py`.** Their entry points print the existing “already complete” error and exit 0; `run()` still raises `FileExistsError`. The resume tests invoke both CLI wrappers on completed outputs.
3. **Unattended run and instance cleanup — fixed in `RUNBOOK-59.md`.** It starts the runner with `nohup`, records the PID, polls `run59.out`, copies the appropriate result bundle, and then runs the Part 54 Vast destroy command after either `RETURN_READY` or `RUN_FAILED`.
4. **Blind calibration — fixed in `blind_setting.py` and its tests.** The two long meta-commentary OTHER examples are now in-voice vague story openings. The contestable traveler/door row is rewritten as an unidentifiable story opening. Calibration remains 34 rows with eight paragraph-length examples; the refusal boundary remains more than two errors. The test accepts two wrong and refuses three.
5. **59B result interpretation — fixed in `analyze59b.py`.** The 59f and 59g prediction objects contain the strict `target_exceeds_max_control` comparison and a plain null caveat; the printed readout includes both. The 59f/59g `held` rules are unchanged. The output also states the recorded control-maxima range beside those results (about 9.1x for the Qwen controls) while retaining the registered pooled threshold.
6. **Incomplete 59B archives — fixed in `analyze59b.py`.** The manifest must say `complete`; otherwise analysis raises a clear `ValueError` before opening the archive.
7. **Test gaps — fixed in `test_part59.py`.** Added coverage captures submit requests and verifies prompt text contains only the rubric plus answer/calibration text, checks the separate answer key mapping, analyzes a synthetic archive, and tests completed CLI invocations and stubbed runner failures. HTTP handling and CA-bundle selection are covered too.
8. **Blind-key packaging — fixed in `RUNBOOK-59.md`.** The input tar command excludes `part59/review` and `part59/__pycache__`.
9. **Blind-reader HTTP and CA handling — fixed in `blind_setting.py`.** HTTP errors produce readable nonzero exits; an unfinished results request says “batch not ended.” `ANTHROPIC_CA_BUNDLE` overrides the prior CA path, which remains the default.
10. **Uneven control scales — addressed in the analysis readout.** This is a property of the registered controls, not a code defect, so no scaling or pooled-rule changes were made. The Qwen control-maxima span is reported next to 59f/59g.
11. **Capture lens and window data during the 59B pass — not done.** This optional refactor is not trivial: it changes the long-running capture path and model/window handling. The existing analysis pass remains.
12. **Stale runner report — fixed in `REPORT-runner.md`.** Its top now points to this fold report. The Luna review found no additional code defect beyond the overlapping runner, packaging, and result-interpretation findings.

The full suite ran with `/home/elisepc/neuron-probe/.venv/bin/python -m unittest -v test_part59`: **16 tests passed**:

- `test_59a_interrupted_run_resumes_and_preserves_completed_files`
- `test_59a_sigkill_releases_lock_and_rerun_proceeds`
- `test_59b_sigkill_releases_lock_and_rerun_proceeds`
- `test_59b_toy_model_hooks_capture_later_neuron_means_and_clamp`
- `test_analyze_file_records_null_caveats_and_target_control_comparison`
- `test_blind_collect_http_error_is_clear_and_ca_bundle_is_configurable`
- `test_blind_origin_answers_reads_59a_condition_files`
- `test_blind_setting_calibration_scores_two_errors_and_refuses_three`
- `test_blind_submit_changed_prompt_manifest_does_not_mix_old_answers`
- `test_blind_submit_requests_only_rubric_and_text_and_saves_local_key`
- `test_blind_submit_requires_complete_manifest_files_and_rejects_strays`
- `test_control_draws_match_part26_values`
- `test_dose_scaling_uses_each_units_part26_pile_maximum`
- `test_pooled_p99_threshold_opposite_sign_and_leave_one_out_control`
- `test_prompt_set_has_twelve_neutral_subjects`
- `test_run59_independent_phases_prefetch_and_preflight_failures`

All requested sabotage checks went red as expected: `flip-sign` changed the count from 2 to 1; `drop-control-percentile` changed it from 2 to 3; `per-neuron-percentile` changed it from 2 to 3; restoring the old global runner gate omitted the Qwen 59B invocation and failed its independence test. `bash -n run59.sh`, Python compilation, and `git diff --check` also passed.

# BLOCKERS

None for the code changes. This worker did not rent a GPU, run the models, or test a live Anthropic batch/Vast instance; runner behavior was checked with local command stubs.
