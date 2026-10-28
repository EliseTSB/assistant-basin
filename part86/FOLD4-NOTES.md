# Part 86 fold 4 notes

## SUMMARY

The screen and verifier now run under the independent observer and attach its call/row claims to each saved answer. Verification provenance and screen-reader agreement are bound to the same screen manifest, primary labels, and selected set. The requested host suites and 65-mutant gate passed.

## FINDINGS / WHAT SHIPPED

### Changes

- `common86.py:55,361,385,412,425,471`: generation-path hashes include repository-root `observe.py`; the manifest requires that hash. `generate_batch` makes one `model.generate` call, checks that the observer call id advanced, and returns the id to the caller.
- `screen86.py:130,272,296,464,496`: installs and closes a UTC/PID-named observer file in `finally`; screen fragments and final rows carry `{call_id,row}` claims, retained on resume. `runtime-canaries.json` now stores `zero_observed`, `zero_answer`, and `zero_token_count`.
- `verify86.py:152,322,349,370,418,444`: installs and closes its observer in `finally`; each answer records its observed row. The generation config carries screen-manifest, primary-label, and selected-set hashes. Preparation validates and preserves them; scoring refuses an agreement artifact with any mismatch.
- `blind86.py:512,555,579`: `score --selected-set` records the three provenance hashes in the score and adjacent artifact manifest. `RUNBOOK.md:3,20,23,32,38,75,82,116,122,167,194` adds the observer/audit copy steps, requested Python preflight, exact audit pass lines, and a selected-set-bound agreement score for verification.
- `test86.py:809,1059`: tests the exact 12-hour boundary, a just-over failure, resumed batch timing plus canary timing, and real screen/verifier loops under the observer with audit fixture overrides. The test builds prompt-text fixtures from its tokenizer and checks trace restoration on resume. `mutate86.py:158-202` adds R4-01 through R4-13 and the three required call-boundary mutants.

### Four requested command summaries

- `part86/test86.py`: `Ran 30 tests in 27.369s` — `OK`
- `part86/test_audit86.py`: `Ran 25 tests in 168.484s` — `OK`
- `../test_observe.py`: `Ran 20 tests in 0.454s` — `OK`
- `part86/mutate86.py --jobs 4`: `MUTATE86: 65/65 killed; all killed`

### Mutation results

The fold-4 real-loop witness is `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces` (T-live). Other short references: T-kappa=`test_kappa_gate_boundary`; T-prep=`test_verification_prepare_chunks_second_reader_and_manifest`; T-survival=`test_registered_survival_rule_and_86_5_ten_of_twenty_boundary`; T-blind=`test_low_kappa_blocks_verdicts_in_the_score_command`; T-miss=`test_miss_sample_follows_the_registered_seed`; T-row=`test_pushes_are_per_row_and_only_apply_after_prompt_pass`; T-tiny=`test_tiny_model_zero_row_matches_baseline_and_push_is_row_local`; T-reset=`test_screen_smoke_provenance_resume_refusal_and_reset`; T-dose=`test_neuron_and_random_write_norm_are_dose_times_median`; T-fisher=`test_fisher_hand_table_and_amended_862_edges`; T-864=`test_864_excludes_triage_broken_and_still_qwen_from_both_sides`; T-full=`test_full_size_labels_to_counts_to_verdicts_including_still_qwen`; T-done=`test_verify_skips_completed_neuron_file_with_same_provenance`; T-denom=`test_kappa_binary_mapping_and_scoring_denominators`; T-golden=`test_golden_verdict_cases_and_override_order`; T-pilot=`test_pilot_gate_limited_population_is_not_read`; T-orphan=`test_orphan_batch_fragment_is_refused`; T-score=`test_blinding_manifest_triage_rerun_and_output_guards`; T-count=`test_full_size_labels_to_counts_to_verdicts_including_still_qwen`; T-median=`test_verify_median_uses_relative_tolerance`; T-picker=`test_picked20_must_name_a_changed_real_answer`; T-reset-config=`test_invalid_reset_layer_config_does_not_clear_any_layer`; T-zero=`test_runtime_zero_push_gates_same_batch_baseline_and_records_batch_effect`; T-gate=`test_gate_2_12_hour_boundary_and_resumed_pilot_projection`; T-triage=`test_fixed_triage_and_false_kept_regression`; T-residual=`test_residual_median_excludes_prompt_token_zero`.

| Mutant | Killed by |
|---|---|
| C-kappa-threshold-0.5 | T-kappa, T-prep |
| C-kappa-strict-greater | T-kappa, T-survival, T-prep |
| C-blind-kappa-gate-off | T-blind |
| C-verify-kappa-gate-off | T-prep |
| C-miss-seed | T-kappa, T-miss |
| C-second-reader-seed | T-kappa, T-miss |
| C-blind-literal-seed | T-miss |
| C-blind-labels2-sample-refused | T-blind |
| F1-M1-row0-push | T-row, T-live, T-tiny |
| F1-M2-push-prompt-pass | T-row, T-live, T-reset, T-tiny, T-done |
| F1-M3-random-dose-row-norm | T-dose, T-live |
| F1-M4-screen-median-resume | T-reset |
| F1-M5-862-minimum-count | T-fisher |
| F1-M6-triage-broken-in-864 | T-864, T-full |
| F1-M7-regenerate-completed-verification | T-done |
| F1-M8-denominator-gate-off | T-denom |
| R2-01-reset-layer-noop | T-reset |
| R2-02-per-layer-denominator-check-off | T-denom |
| R2-03-862-real-random-swapped | T-golden |
| R2-04-residual-median-token0 | T-residual |
| R2-05-runtime-qwen-gate-off | T-zero |
| R2-06-verify-resume-config-check-off | `test_verify_neuron_resume_refuses_changed_config` |
| R2-07-pilot-gate-always-pass | T-pilot |
| R2-08-orphan-fragment-refusal-off | T-orphan |
| R2-09-blind-score-no-retriage | T-score |
| R2-10-changed-count-check-off | T-count |
| R2-11-verify-median-equality-off | T-median |
| R2-12-row-hook-uses-row0 | T-row, T-live, T-tiny |
| R2-13-864-read-minimum-four | T-golden |
| R2-14-866-range-changed | T-golden |
| R2-15-864-equality-passes | T-864, T-full, T-golden |
| R2-16-triage-kept-filter-dropped | T-score, T-triage, T-pilot, T-live, T-zero, T-miss |
| B1-still-qwen-counted-as-failure | T-864, T-full, T-golden |
| B2-scorer-hashes-in-screen-manifest | T-score, T-blind, T-miss, T-live, T-reset |
| M1-zero-row-same-batch-gate-off | T-zero |
| M3-picked-label-not-checked | T-picker |
| L4-reset-before-config-check | T-reset-config, T-reset |
| N1-screen-zero-neuron-dose | T-live observer write check |
| N2-screen-all-neurons-push-unit0 | T-live observer channel check |
| N3-screen-random-double-dose | T-live observer write-norm check |
| N4-screen-token-budget24 | T-live observer effective-token check |
| N5-verify-push-unit0 | T-live observer channel check |
| N6-verify-greedy | T-live observer sampling check |
| N7-verify-seeds-plus10 | T-live observer RNG check |
| N8-screen-second-reader20pct | T-miss |
| N9-verify-random-double-dose | T-live observer write-norm check |
| N10-verify-case-question0 | T-live observer prompt check |
| TRACE-plan-values-instead-of-measurement | T-live observer measurement check |
| TRACE-resume-drops-rows | T-live resume-sidecar row-count assertion |
| R4-01-screen-negative-neuron | T-live observer signed-neuron check |
| R4-02-screen-random-seed-plus1 | T-live observer direction check |
| R4-03-screen-reversed-answer-association | T-live observer answer-to-row hash check |
| R4-04-verify-negative-neuron | T-live observer signed-neuron check |
| R4-05-verify-random-seed-plus1 | T-live observer direction check |
| R4-06-verify-hook-previous-layer | T-live observer changed-layer check |
| R4-07-screen-calibrate-previous-layer | T-live observer changed-layer check |
| R4-08-verify-token-budget-63 | T-live observer effective-token check; T-prep config check |
| R4-09-cpu-seed-plus1 | T-live observer RNG check |
| R4-10-push-first-decode-only | T-live observer all-decode-calls check |
| R4-11-always-screen-question | T-live observer rendered-prompt check |
| R4-12-top-p-0.9 | T-live observer sampling check; T-prep config check |
| R4-13-gate-2-120-hours | T-gate just-over-12-hour failure |
| R4-extra-actual-temperature-1.0 | T-live observer effective-temperature check |
| R4-extra-default-generator-seed-plus1 | T-live observer CPU-seed check |
| R4-extra-zero-channel-before-push | T-live observer collateral-channel check |

### Independent oracles and limits

I found no defect in `audit86.py`, `test_audit86.py`, `golden-verdicts.json`, `observe.py`, or `test_observe.py`; none was edited. The live screen and verifier loops were tested on the tiny CPU model with fixture overrides. I did not run Qwen3.5-4B, the full registered-size layer-22 pilot, an A770, or the full 17-layer screen.

## BLOCKERS

None for the fold-4 code and test work. Hardware pilot and full-size audit remain operator steps in the runbook.
