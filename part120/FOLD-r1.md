# Part 120 review fold r1

- F1 — No code change: each row-position lookup supplies its saved T to A, B, and P centering; distinct-T same-first rows retain distinct A vectors in `test_real_main_path_tiny_model_and_resume` (passed against the old copy).
- F2 — No code change: `path_for_first` uses the same predictor window as `run_path2`, including p=0 and short answers; `test_path_zero_matches_part119b_predictor_window` passed against the old copy.
- F3 — Catch write `OSError`, persist an aborted timing record, and keep atomic completed unit files resumable; `test_failed_npz_write_keeps_abort_record_and_finished_unit` failed against the old copy.
- F4 — NPZ indexes now record selected and expected unit scope, and scoring requires a complete main timing record, full doses, all unit files, and matching index scope; `test_npz_index_records_complete_and_smoke_scope`, `test_partial_npz_scope_cannot_be_scored_as_full_run`, and `test_partial_npz_metadata_refused_with_complete_timing` failed against the old copy.
- F5 — Gate 3 now compares each dose cache with a separate weight build and the scorer requires that evidence; `test_gate3_rejects_scaled_cache_against_weight_rebuilds` and `test_gate3_rejects_records_without_weight_rebuild_evidence` failed against the old copy.
- F6 — No code change: one Part 119 unit draw supplies counts for both endpoints, with product weights primary and sum weights alongside; `test_bootstrap_resamples_both_row_endpoints` passed against the old copy.
- F7 — No code change: the existing branches mean high < 0.8, wholly within [0.8, 1.25], and low > 1.25; exact boundaries are pinned by `test_headline_all_branches_including_withdrawn` (passed against the old copy).
- F8 — Missing answers now raise a path-specific `FileNotFoundError`; `test_missing_answers_file_is_reported_as_missing_input` failed against the old copy.
- F9 — Empty usable vectors now produce an explicit empty-input headline; `test_empty_vector_headline_names_empty_input` failed against the old copy.
- F10 — No code change: malformed non-object gate-3 comparisons resolve to an invalid error and fail closed without an `AttributeError`; `test_malformed_gate3_comparison_fails_closed` passed against the old copy.
