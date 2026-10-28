# Part 86 fold 3 notes

## SUMMARY

Screen and verification now persist measured per-answer traces, and the verifier and RUNBOOK enforce the round-3 findings.
Ran 29 tests in 5.760s
MUTATE86: 49/49 killed; all killed

## FINDINGS / WHAT SHIPPED

### Trace and independent audit

- `common86.AddRowsHook` measures each decode row from the tensors passed through the actual `down_proj` pre-hook. It
  computes `delta = pushed.float() - input.float()`, the `abs(delta) > 1e-6` channel mask, and the per-row count of
  decode calls with any changed channel. On that row's first decode call only, it records the sorted changed channel
  indices (first 16) and computes `||W_down.float() @ delta||` in float32. The prompt branch returns `None` without
  replacing its input, so its recorded `prefill_changed` is false.
- `direction_sha256` hashes the row's float32 push-vector bytes. The vector is copied and hashed only for random
  directions. `prompt_sha256` hashes the token IDs actually returned by tokenizing the rendered prompt. The trace
  reads `max_new_tokens`, `do_sample`, `temperature`, and `top_p` from the kwargs passed to `model.generate`; it records
  the question passed to that call and the seed used to seed its sampling RNG.
- `screen86.py` writes `trace/layer_<L>.jsonl`, one row per screen answer. Each answer JSONL row contains its trace as
  part of the same atomic write; the sidecar is also written atomically and rebuilt from those paired answer records on
  resume. The layer-22 runtime canary contributes one `kind: "zero", target: null` trace row for its zero-push row.
- `verify86.py` writes `trace/L<layer>_N<neuron>.jsonl`. Each atomic per-neuron checkpoint stores every answer beside
  its measured trace, and the sidecar is rebuilt from that checkpoint on resume. An orphan sidecar without its answer
  state is refused.
- `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces` runs the real screen loop at layer 22
  on a tiny Qwen-style model, including its real zero canary and push construction. It runs `audit86.py screen` as a
  subprocess with the test fixture sizes, removes a trace row and confirms resume restores it, then runs one complete
  264-case verifier plan and `audit86.py verify` as a subprocess. It removes a verifier trace row and confirms resume
  restores that too. No generation loop, case plan, or push construction is mocked in this test.

### Finding #1 — screen agreement also gates 86.5

- `blind86.py score` writes an adjacent `<score>.manifest.json` containing the screen score file SHA-256 and the
  internal score-manifest hash. `verify86.py score --screen-score ...` validates both hashes before reading screen
  kappa. If screen kappa is missing or below `common86.kappa_reads`'s 0.6 threshold, 86.5 is `NOT READ`, even when the
  verification readers agree. Omitting the screen score also yields `NOT READ`.
- The verification result manifest carries the checked screen score and manifest hashes. RUNBOOK passes the screen
  score file to verification scoring.
- Tests cover screen kappa at 0.6, below 0.6, missing, and a tampered score file; they also cover low verification
  kappa and the CLI's screen score output.

### Finding #3 — picked persona must match its primary blind label

- `validate_picked_against_screen` now requires each picked phrase to equal its primary blind label phrase, ignoring
  case. An operator may pass a JSON equivalence list with `--equivalences`; each pair has `primary` and `picked` text.
  The run config and prepared verification key retain the reviewed equivalence list and primary label phrase.
- Tests reject a planet-ruling deity phrase for a traveler label, accept an explicitly listed equivalence, and retain
  the existing changed-real-answer and coherent-label checks.

### Finding #4 — verification preparation validates provenance and full inputs

- `verify86.prepare` requires all 20 completed neuron files, each with the complete 264-case set, consistent answer
  and trace rows, registered questions/seeds/sampling settings, and matching record/config dose and median. It checks
  the effective generation config and shared provenance across files while allowing per-neuron identity, seed, and
  layer median to vary as designed. Repeated files for one layer must agree on its median.
- The private key records every input JSON SHA-256, every per-neuron config, and the shared-config hash. The 86.5 score
  manifest carries those file hashes and configs forward. Tests remove one case and mix one file at dose 0.4; both
  directories are refused.

### Finding #5 — Gate 2 uses the full layer-22 pilot

- Each screen batch persists generation elapsed seconds and answer counts in its answer fragments and layer metadata.
  Resume recomputes cumulative time from completed fragments. The layer-22 canary persists its own generation time and
  count. `pilot-gate.json` calculates Gate 2 from the full layer-22 screen plus canary and projects the registered
  165,376 screen answers against 12 hours. A limited `--limit` run is marked `EARLY ESTIMATE ONLY`.
- RUNBOOK now describes the limited run as a warmup, requires the registered-size layer-22 trace audit before the
  remaining layers, requires the all-layer screen audit before scoring, and requires the verifier audit before
  verification scoring.

### Finding #6 — current mutation gate in RUNBOOK

- RUNBOOK now requires the measured `MUTATE86: 49/49 killed; all killed` line.

## Final mutate86 table

| Mutation | Result | Catching test |
|---|---|---|
| C-kappa-threshold-0.5 | KILLED | `test_kappa_gate_boundary`; `test_verification_prepare_chunks_second_reader_and_manifest` |
| C-kappa-strict-greater | KILLED | `test_kappa_gate_boundary`; `test_registered_survival_rule_and_86_5_ten_of_twenty_boundary`; `test_verification_prepare_chunks_second_reader_and_manifest` |
| C-blind-kappa-gate-off | KILLED | `test_low_kappa_blocks_verdicts_in_the_score_command` |
| C-verify-kappa-gate-off | KILLED | `test_verification_prepare_chunks_second_reader_and_manifest` |
| C-miss-seed | KILLED | `test_kappa_gate_boundary`; `test_miss_sample_follows_the_registered_seed` |
| C-second-reader-seed | KILLED | `test_kappa_gate_boundary`; `test_miss_sample_follows_the_registered_seed` |
| C-blind-literal-seed | KILLED | `test_miss_sample_follows_the_registered_seed` |
| C-blind-labels2-sample-refused | KILLED | `test_low_kappa_blocks_verdicts_in_the_score_command` |
| F1-M1-row0-push | KILLED | `test_pushes_are_per_row_and_only_apply_after_prompt_pass`; `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces`; `test_tiny_model_zero_row_matches_baseline_and_push_is_row_local` |
| F1-M2-push-prompt-pass | KILLED | `test_pushes_are_per_row_and_only_apply_after_prompt_pass`; `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces`; `test_screen_smoke_provenance_resume_refusal_and_reset`; `test_tiny_model_zero_row_matches_baseline_and_push_is_row_local` |
| F1-M3-random-dose-row-norm | KILLED | `test_neuron_and_random_write_norm_are_dose_times_median`; `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces` |
| F1-M4-screen-median-resume | KILLED | `test_screen_smoke_provenance_resume_refusal_and_reset` |
| F1-M5-862-minimum-count | KILLED | `test_fisher_hand_table_and_amended_862_edges` |
| F1-M6-triage-broken-in-864 | KILLED | `test_864_excludes_triage_broken_and_still_qwen_from_both_sides`; `test_full_size_labels_to_counts_to_verdicts_including_still_qwen` |
| F1-M7-regenerate-completed-verification | KILLED | `test_verify_skips_completed_neuron_file_with_same_provenance` |
| F1-M8-denominator-gate-off | KILLED | `test_kappa_binary_mapping_and_scoring_denominators` |
| R2-01-reset-layer-noop | KILLED | `test_screen_smoke_provenance_resume_refusal_and_reset` |
| R2-02-per-layer-denominator-check-off | KILLED | `test_kappa_binary_mapping_and_scoring_denominators` |
| R2-03-862-real-random-swapped | KILLED | `test_golden_verdict_cases_and_override_order` |
| R2-04-residual-median-token0 | KILLED | `test_residual_median_excludes_prompt_token_zero` |
| R2-05-runtime-qwen-gate-off | KILLED | `test_runtime_zero_push_gates_same_batch_baseline_and_records_batch_effect` |
| R2-06-verify-resume-config-check-off | KILLED | `test_verify_neuron_resume_refuses_changed_config` |
| R2-07-pilot-gate-always-pass | KILLED | `test_pilot_gate_limited_population_is_not_read` |
| R2-08-orphan-fragment-refusal-off | KILLED | `test_orphan_batch_fragment_is_refused` |
| R2-09-blind-score-no-retriage | KILLED | `test_blinding_manifest_triage_rerun_and_output_guards` |
| R2-10-changed-count-check-off | KILLED | `test_full_size_labels_to_counts_to_verdicts_including_still_qwen` |
| R2-11-verify-median-equality-off | KILLED | `test_verify_median_uses_relative_tolerance` |
| R2-12-row-hook-uses-row0 | KILLED | `test_pushes_are_per_row_and_only_apply_after_prompt_pass`; `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces`; `test_tiny_model_zero_row_matches_baseline_and_push_is_row_local` |
| R2-13-864-read-minimum-four | KILLED | `test_golden_verdict_cases_and_override_order` |
| R2-14-866-range-changed | KILLED | `test_866_pools_only_layers_14_through_22` |
| R2-15-864-equality-passes | KILLED | `test_864_excludes_triage_broken_and_still_qwen_from_both_sides`; `test_full_size_labels_to_counts_to_verdicts_including_still_qwen`; `test_golden_verdict_cases_and_override_order` |
| R2-16-triage-kept-filter-dropped | KILLED | `test_blinding_manifest_triage_rerun_and_output_guards`; `test_fixed_triage_and_false_kept_regression`; `test_miss_sample_follows_the_registered_seed`; `test_pilot_gate_limited_population_is_not_read`; `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces`; `test_runtime_zero_push_gates_same_batch_baseline_and_records_batch_effect` |
| B1-still-qwen-counted-as-failure | KILLED | `test_864_excludes_triage_broken_and_still_qwen_from_both_sides`; `test_full_size_labels_to_counts_to_verdicts_including_still_qwen`; `test_golden_verdict_cases_and_override_order` |
| B2-scorer-hashes-in-screen-manifest | KILLED | `test_blinding_manifest_triage_rerun_and_output_guards`; `test_low_kappa_blocks_verdicts_in_the_score_command`; `test_miss_sample_follows_the_registered_seed`; `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces`; `test_screen_smoke_provenance_resume_refusal_and_reset` |
| M1-zero-row-same-batch-gate-off | KILLED | `test_runtime_zero_push_gates_same_batch_baseline_and_records_batch_effect` |
| M3-picked-label-not-checked | KILLED | `test_picked20_must_name_a_changed_real_answer` |
| L4-reset-before-config-check | KILLED | `test_invalid_reset_layer_config_does_not_clear_any_layer`; `test_screen_smoke_provenance_resume_refusal_and_reset` |
| N1-screen-zero-neuron-dose | KILLED | `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces` |
| N2-screen-all-neurons-push-unit0 | KILLED | `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces` |
| N3-screen-random-double-dose | KILLED | `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces` |
| N4-screen-token-budget24 | KILLED | `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces` |
| N5-verify-push-unit0 | KILLED | `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces` |
| N6-verify-greedy | KILLED | `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces` |
| N7-verify-seeds-plus10 | KILLED | `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces` |
| N8-screen-second-reader20pct | KILLED | `test_miss_sample_follows_the_registered_seed` |
| N9-verify-random-double-dose | KILLED | `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces` |
| N10-verify-case-question0 | KILLED | `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces` |
| TRACE-plan-values-instead-of-measurement | KILLED | `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces` |
| TRACE-resume-drops-rows | KILLED | `test_real_screen_and_verify_loops_record_auditable_pushes_and_resume_traces` |

## What could not be verified

- No Qwen3.5-4B model or A770/XPU was loaded. The strict CPU tests use a tiny Qwen-style model and test-size audit
  overrides; they do not establish the real full-population layer-22 audit or GPU throughput Gate 2. RUNBOOK requires
  the registered-size L22 audit and full pilot timing before the remaining layers start.
- No full 17-layer screen, live blind-reader pass, or 20-neuron verification run was performed.

## BLOCKERS

None for the fold-3 code and test deliverables. The hardware pilot remains an operator step in RUNBOOK.

## Claude, on collection (2026-10-28 01:2x CT)

Host rerun: Ran 29 tests, OK; MUTATE86: 49/49 killed. audit86.py and golden-verdicts.json unchanged. Read the hook: the trace measures delta = pushed - input per row at the first decode call, so a zero or misaddressed push shows in the trace.
