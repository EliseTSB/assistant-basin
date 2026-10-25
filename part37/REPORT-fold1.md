# Part 37 fold 1 report

## SUMMARY

Implemented Amendment 2 and its review fixes in the analyzer, collector, and tests. The requested host suite passes: 26 tests and 10 scoring subtests; no GPU was used.

## FINDINGS

1. **Out-of-fold projections (Amendment 2 item 1).** Added `_crossfit_projections`; it fits on question-disjoint training folds and projects every non-empty held-out record, including sampled answers, nulls, edits, and label frames. 37.3, 37.4, 37.9, and the projection summaries consume these results. `test_crossfit_projections_block_noise_fit_leak_for_37_9_and_37_4` uses 100 questions and width-512 Gaussian noise; it asserts the held-out 37.9 shift is under 0.5 SD and the held-out 37.4 Spearman is under 0.3. It also asserts that the legacy all-question fit exceeds the 37.9 bar, reproducing the old leak.

2. **37.15 null set (item 2).** The score uses only `null_smol` and `null_pile`; `null_shuffle` is reported as the self-text floor. The per-kind fixture in `test_each_scored_prediction_has_pass_and_fail_scoring_fixtures` gives means 0.02, 0.03, and 0.64 and expects 37.15 to FAIL. Running the old scorer loaded from `git show 92c2f6e:part37/analyze37.py` on the pooled 0.23 value returns PASS.

3. **Pilot gate (item 3).** Each within-author contrast has its own question-flipped null and is checked against that null's 95th percentile of the maximum over layers. Authorship layers use their per-layer 95th-percentile nulls. Missing or non-finite layers/nulls raise `AnalysisError`, which `main()` returns as exit 2. `test_pilot_noise_stops_on_at_least_two_of_three_seeds_and_signal_continues` checks three 20-question noise seeds and a planted signal. The old gate returned STOP for 0/3 noise seeds when run from the 92c2f6e analyzer.

4. **37.9 pairing and counts (item 4).** The score includes only questions with own_g, own_s1, own_s2, and other_g. Per-cell present, included, and dropped counts are included in the output. `test_37_9_pairing_requires_both_nonempty_samples_and_reports_cell_drops` checks that incomplete sampled pairs are excluded; the old calculation included each available sampled row without this pair filter or cell counts.

5. **Signed 37.11 (item 5).** The verdict now uses signed memory self-minus-other effect minus signed authorship effect, and tests the neutral header only on memory phrasing. The scoring fixture uses memory=-0.60 and authorship=+0.40: the old scorer returns PASS by absolute effect size; the new scorer returns FAIL.

6. **37.13 and 37.8 (items 6–7).** 37.13 requires both role-cell balanced accuracies to be within [0.60, 0.75]. The scoring fixture with 0.66 and 0.76 is a FAIL; the old scorer returned PASS. 37.8 now fits both directions at the mean-log-prob-regressed best layer. `test_37_8_cosine_uses_the_regressed_authorship_best_layer` verifies the layer and fit modes; the old code used the raw authorship best layer for the cosine.

7. **Scoring contract (item 7).** `test_each_scored_prediction_has_pass_and_fail_scoring_fixtures` sends PASS and FAIL fixtures through `_prediction_results` for 37.1, 37.2, 37.3, 37.4, 37.8, 37.9, 37.11, 37.12, 37.13, 37.14, and 37.15. The old scorer was also checked directly for false PASS cases on 37.11, 37.13, and 37.15.

8. **Minimal edit (item 8).** Added fixed stopword and common-noun lists; the target skips stopwords and the replacement is a noun within one character of its length. `test_minimal_edit_logs_a_similar_length_replacement` checks DeepSeek's eight sampled answers and requires every replacement to avoid the stopword list. The 92c2f6e function produced stopwords for 5/8 answers.

9. **Precision and resilience (item 9).** Capture stores float32 hidden states and reports the record id on non-finite states or span metrics. Span log-probabilities use float32 logits. Removed explicit `penalty="l2"`; the label-shuffle null counts fit failures and raises above two per layer. Generation JSONL and manifests use atomic replacement; capture writes an atomic shard and complete question metadata prefix. Covered by `test_capture_uses_one_forward_and_returns_both_positions`, `test_span_logprobs_use_float32_logits_and_nonfinite_states_name_the_record`, `test_l2_penalty_is_left_at_scikit_learns_non_deprecated_default`, `test_label_shuffle_raises_when_a_layer_fails_more_than_two_fits`, `test_atomic_writer_preserves_old_file_if_a_write_is_interrupted`, `test_capture_jsonl_reader_uses_a_loadable_prefix_after_a_torn_tail`, and `test_capture_leaves_a_loadable_question_prefix_after_a_later_failure`.

10. **Analysis notes and controls (item 10).** Output notes call the length-only measure a sanity check that cannot fail by construction, describe the unchanged user-slot frame, and disclose the unchanged SmolLM3 template's identity wording and `/no_think` instruction. `test_analysis_notes_name_sanity_check_and_preserved_frame_prompts` covers these notes; the existing prompt/frame tests continue to cover the unchanged generation and record behavior.

### Verification

- `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q part37/test_part37.py` — **26 passed, 2 warnings, 10 subtests passed** in 59.30s. The warnings are pytest cache writes rejected by the worktree's read-only `.pytest_cache`; there were no test failures.
- `git diff --check` — passed.
- No GPU or model generation was run.
- No commit was created, per the worker assignment's explicit no-commit instruction.

## BLOCKERS

none
