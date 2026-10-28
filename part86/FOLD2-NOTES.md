# Part 86 fold 2 notes

## Summary

Verdicts and label counts are rebuilt as pure steps, driven against every case in the unchanged Claude oracle. The strict CPU suite passes 25 tests with no skips; `mutate86.py --jobs 4` reports `MUTATE86: 29/29 killed; all killed`. No golden case was disputed. No parent-control messages were received.

## Findings and changes

### B1 — verdicts did not match the written rules

- Added `label_counts(labels, key, triage_counts)`, which emits the named per-layer real and random counts. Its single label mapping is documented in `_label_class`: `coherent=false` is `not_coherent`; otherwise kind `unclear` is `unclear`, kind `still Qwen or an assistant` is `coherent_still_qwen`, and all remaining coherent labels are `coherent_persona`.
- Added pure `verdicts_from_counts(counts)`. It implements the four rules represented in `golden-verdicts.json`, validates the 9,216/512 denominators and disjoint random label totals, and returns the expected NOT READ states for missing layers. The existing oracle file was not edited.
- The oracle test applies base counts, base overrides, case overrides, then dropped layers exactly as authored. It checks every expected status and requires the bad-denominator case to raise.
- Added a full-size end-to-end synthetic label test representing all 17 layers at 9,216 real and 512 random answers per layer. It exercises still-Qwen, unclear, and not-coherent labels through counts into verdicts.
- Disputed golden cases: none.

### H1 — tests could not detect broken features

- Added the committed isolated mutation runner. Each mutation receives a fresh copy of `part86/`, `neuron_report.py`, and `part84`; exact replacements must occur exactly once. The runner uses strict tests, rejects skips or invalid runs, supports `--jobs`, times out each test run at 600 seconds, records killed/survived, and exits nonzero unless all mutations are killed.
- The RUNBOOK requires the full audit and the exact all-killed final line before GPU time.
- The final run killed all 29 mutations. The table below records every mutation and the test that caught it.

### B2 — screen provenance included scorer files

- Screen configs now hash only `common86.py`, `screen86.py`, and `neuron_report.py`, plus calibration hash, versions, model/dose/dtype/median, and effective generation settings. Resume preflights the stored generation config before reset; generation-path edits refuse resume.
- Screen manifest reconstruction uses hashes saved in screen metadata rather than current scorer files, so later blind or verification code edits do not invalidate a screen. Blind scoring records its own scorer/input manifest; verification records its own code/input manifest.

### H2 — reader volume and agreement

- Screen and verification preparation write reader chunks of at most 500 rows with opaque ids. The second reader gets a deterministic random 10% sample using seed 8687.
- Screen preparation also samples up to 200 kept and 200 broken answers from each source across all layers with seed 8686. The scorer reports coherent-persona shares for all four miss-rate samples beside the layer counts; the samples never affect verdicts.
- Both scorers compute Cohen's kappa on coherent-persona versus not, print/store agreement before verdict output, and mark applicable verdicts NOT READ when kappa is absent or below 0.6. A full second read is accepted as the follow-up agreement input.

### M1 — zero-push and bf16 batch effects

- The L22 canary now measures the same unsteered prompt in a batch of 16 and alone, recording both answers and token ids. It gates on exact equality between a zero-push candidate and a second unsteered row in the same batch shape. Pushed-row batch comparisons remain diagnostic.

### M3 — verification picks

- Verification requires the primary screen key and labels, then confirms each pick maps to a changed real screen row whose primary blind label is a coherent persona.

### L1/L2/L3/L4 — round-1 leftovers

- L1: the 86.1 output is marked `reading: hand-read`; it leaves the automatic phrase match explicitly non-authoritative and exposes the raw N2742 answer.
- L2: verification score results include their own manifest and input hashes.
- L3: the 86.2 ten-real floor remains in the rule and is separately reported/tested; it was not changed.
- L4: all requested screen configuration checks run before any layer reset can delete data. Verification median comparisons use relative tolerance `1e-6`.
- RUNBOOK uses `165376 / 43200 = 3.82815`, documents picker input and opaque-id-to-neuron mapping, requires label-checked picks, and replaces superseded 86.4 descriptions in BUILD-NOTES and FOLD1-NOTES.

## Final mutation table

`MUTATE86: 29/29 killed; all killed`

| Mutation | Result | Catching test |
|---|---|---|
| F1-M1 row 0 push | KILLED | `test_pushes_are_per_row_and_only_apply_after_prompt_pass`; `test_tiny_model_zero_row_matches_baseline_and_push_is_row_local` |
| F1-M2 push prompt pass | KILLED | `test_pushes_are_per_row_and_only_apply_after_prompt_pass`; `test_screen_smoke_provenance_resume_refusal_and_reset`; `test_tiny_model_zero_row_matches_baseline_and_push_is_row_local` |
| F1-M3 random dose row norm | KILLED | `test_neuron_and_random_write_norm_are_dose_times_median` |
| F1-M4 screen median resume | KILLED | `test_screen_smoke_provenance_resume_refusal_and_reset` |
| F1-M5 86.2 minimum count | KILLED | `test_fisher_hand_table_and_amended_862_edges` |
| F1-M6 triage-broken enters 86.4 | KILLED | `test_864_excludes_triage_broken_and_still_qwen_from_both_sides`; `test_full_size_labels_to_counts_to_verdicts_including_still_qwen` |
| F1-M7 regenerate completed verification | KILLED | `test_verify_skips_completed_neuron_file_with_same_provenance` |
| F1-M8 denominator gate off | KILLED | `test_kappa_binary_mapping_and_scoring_denominators` |
| R2-01 reset layer no-op | KILLED | `test_screen_smoke_provenance_resume_refusal_and_reset` |
| R2-02 per-layer denominator check off | KILLED | `test_kappa_binary_mapping_and_scoring_denominators` |
| R2-03 swap 86.2 real/random | KILLED | `test_golden_verdict_cases_and_override_order` |
| R2-04 include token zero in median | KILLED | `test_residual_median_excludes_prompt_token_zero` |
| R2-05 zero-push Qwen gate off | KILLED | `test_runtime_zero_push_gates_same_batch_baseline_and_records_batch_effect` |
| R2-06 verification resume config check off | KILLED | `test_verify_neuron_resume_refuses_changed_config` |
| R2-07 pilot gate always pass | KILLED | `test_pilot_gate_limited_population_is_not_read` |
| R2-08 orphan-fragment refusal off | KILLED | `test_orphan_batch_fragment_is_refused` |
| R2-09 blind score skips re-triage | KILLED | `test_blinding_manifest_triage_rerun_and_output_guards` |
| R2-10 changed-count check off | KILLED | `test_full_size_labels_to_counts_to_verdicts_including_still_qwen` |
| R2-11 verification median tolerance off | KILLED | `test_verify_median_uses_relative_tolerance` |
| R2-12 row hook uses row 0 | KILLED | `test_pushes_are_per_row_and_only_apply_after_prompt_pass`; `test_tiny_model_zero_row_matches_baseline_and_push_is_row_local` |
| R2-13 86.4 read minimum four | KILLED | `test_golden_verdict_cases_and_override_order` |
| R2-14 86.6 layer range changed | KILLED | `test_866_pools_only_layers_14_through_22` |
| R2-15 86.4 equality passes | KILLED | `test_864_excludes_triage_broken_and_still_qwen_from_both_sides`; `test_full_size_labels_to_counts_to_verdicts_including_still_qwen`; `test_golden_verdict_cases_and_override_order` |
| R2-16 triage kept filter dropped | KILLED | `test_blinding_manifest_triage_rerun_and_output_guards`; `test_fixed_triage_and_false_kept_regression`; `test_pilot_gate_limited_population_is_not_read`; `test_runtime_zero_push_gates_same_batch_baseline_and_records_batch_effect` |
| B1 still-Qwen counted as failure | KILLED | `test_864_excludes_triage_broken_and_still_qwen_from_both_sides`; `test_full_size_labels_to_counts_to_verdicts_including_still_qwen`; `test_golden_verdict_cases_and_override_order` |
| B2 scorer hashes included in screen manifest | KILLED | `test_blinding_manifest_triage_rerun_and_output_guards`; `test_screen_smoke_provenance_resume_refusal_and_reset` |
| M1 same-batch zero-row gate off | KILLED | `test_runtime_zero_push_gates_same_batch_baseline_and_records_batch_effect` |
| M3 picked label unchecked | KILLED | `test_picked20_must_name_a_changed_real_answer` |
| L4 reset before config check | KILLED | `test_invalid_reset_layer_config_does_not_clear_any_layer`; `test_screen_smoke_provenance_resume_refusal_and_reset` |

## What could not be verified

The strict suite and mutation runner used the local CPU venv and synthetic outputs. No Qwen3.5-4B model was loaded, and no A770/XPU run was performed. The live prompt/tokenizer behavior, actual L22 N2742 answer, bf16 batch-effect values, runtime throughput, and human reader labels/kappa remain unverified until the GPU and reader steps in RUNBOOK.

## Claude, on collection (2026-10-28 00:3x CT)

Host rerun: 25 tests OK and mutate86 29/29, as reported. Seven further mutations of mine (kappa threshold 0.5, strict
greater-than, the scorer's and verifier's kappa gates off, the two Amendment 2 seeds, a literal seed) first gave 35/36:
the score command's kappa gate was untested. The kappa threshold and both seeds are now named in common86.py
(KAPPA_THRESHOLD, MISS_SAMPLE_SEED, SECOND_READER_SEED, kappa_reads) and pinned by TestAmendment2Pins. A new end-to-end
test of the score command (TestScoreCliKappaGate) then found a real defect: `score --labels2` validated the second
reader's file against EVERY row, so the registered 10% second reading was always refused ("missing labels"). Fixed to
accept the seed-8687 sample or a full second reading; mutation C-blind-labels2-sample-refused restores the old line.
Final: Ran 28 tests, OK; MUTATE86: 37/37 killed.
