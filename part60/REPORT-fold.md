# Part 60 fold report

## SUMMARY
- Folded the B/Luna and B/Haiku findings into generation, scoring, analysis, runner, docs, and tests; checked every A/C high-severity class.
- All findings were fixed unless identified below as absent or incorrect. The 247-row legacy test-list claim was incorrect against the registered 250-per-type test design.
- Added the parent-requested production round-trip and score invariant checks. Parent mailbox seq 1 was considered and applied; no other messages arrived.
- Validation: 41 tests passed; three production-path sabotage checks went red as required; Python compilation and bash syntax passed.
- No network/API request, Qwen load, or commit was made, per brief.

## FINDINGS

### B/Luna findings 1–7
1. Fixed 60j candidate M1 self-comparison in analyze60.py: an M1 candidate is compared with M0 only; other candidates must beat M0 and M1. Covered by test_m1_candidate_is_compared_to_m0_without_self_comparison.
2. Fixed M6 to use adaptive thinking, max_tokens 4000, a one-unit smoke gate, method drop on rejected/empty thinking, and per-request batch-error drop. Covered by the M6 smoke, submit guard, and per-request error tests.
3. Fixed reserve-before-POST accounting, durable per-chunk reservations, settlement replacement, cap override recording, and exact metadata reconciliation for ambiguous acceptance. Covered by reserve, ambiguous POST, SIGKILL resume, and cap tests.
4. Fixed second-call rejection to retain first-round/selection output and usage while recording rejection beside that evidence. Covered by test_second_call_rejection_preserves_first_round_evidence.
5. Fixed stale success archive removal and successful advisory pip-freeze fallback in run60.sh. Covered by wrapper tests.
6. Replaced the two toy checks with production calls and sabotage tests. The threshold sabotage calls compute_thresholds; the winner sabotage calls choose_m3_winners; the SID sabotage reaches aggregate_method/load_acts.
7. Added staging instructions and early path checks in gen60.py and score60.py, including the exact PART60_BASIN remedy.

### B/Haiku findings 1–20
1. Fixed M6 using Amendment 1 adaptive thinking and max_tokens 4000; smoke requires a non-empty thinking block. Smoke failure and per-request errors drop M6. No real smoke was run under the no-API brief.
2. Fixed: run60.sh no longer requires M7 output; M7 is an explicit skip with no request. Covered by M7 submit and wrapper tests.
3. Fixed: pip freeze failure falls back within the version-record command and remains advisory. Covered by the wrapper test.
4. Fixed runbook teardown ordering: the copy must succeed and the local archive must pass tar -tzf before destroy.
5. Fixed: collection records stop_reason, identifies max_tokens truncation, reports truncation counts, and fails the phase. Covered by test_max_tokens_stop_reason_is_recorded_and_fails_collection.
6. Safe retry behavior: definitive HTTP 429 receives bounded retries on the same tagged chunk; other ambiguous POST outcomes are left reserved and locked for exact-tag reconciliation, with no blind retry. This deliberately declines automatic retry of 408/409/425/5xx because acceptance may already have occurred and no API idempotency guarantee was available. Covered by 429 same-tag and ambiguous reconciliation tests.
7. Fixed: reserve is appended and fsynced before POST; unknown outcomes keep that reserve active and require reconciliation. Covered by the ambiguous POST test.
8. Fixed: production threshold/winner sabotage tests and wrapper cases now exercise the actual code paths; the round-trip test writes generator output and score-format NPZ then calls load_acts and aggregate_method.
9. Fixed: test scoring requires the dev analysis selection marker and permits only M0, M1, and its selected candidate. Covered by test_run60_requires_dev_selection_marker_before_test_phase.
10. Fixed M3 fresh skip ordering before requiring the selection matrix; covered by test_m3_fresh_drop_rule_runs_before_selection_matrix_is_required.
11. Incorrect as stated: the design registers disjoint 250-per-type splits. The 247-row test file is a legacy real-unit list; DEFINITIONS-60.md documents the three additional valid records chosen by the registered pilot criteria and the fixed-pool threshold extension for missing archive columns. No test-set membership was changed from the registered design.
12. 60c now uses the intersection of valid M0/M1 real units and reports its count; 60h uses the same four comparable marked sentences for marked and whole-text rates, leaving missing marked activations unscored on both sides. 60d already implements the registered share of eligible real units with strict per-unit mean tier ordering (PART60-DESIGN.md 60d); no change to that metric. 60j is explicitly about the registered dev-selected test candidate; M1 is compared with M0 only, avoiding self-comparison.
13. Fixed reproduction gating to require overall, neuron, and control medians within 0.9–1.1. Added test_reproduction_gate_checks_overall_and_each_unit_type.
14. Removed the unobservable “do not reuse any selection sentence” clause from prompts/M3.txt.
15. Fixed M5 round 1 to use the M1 prompt; documented the round-1/round-2 metric distinction.
16. Fixed the report table to show control Hstrong, real-control gap, call counts, and truncations.
17. Corrected runbook sentence arithmetic and included M3 fresh and M5 round 2 counts.
18. Fixed window-order blinding with stable seeded ordering; the request exposes no tier labels.
19. Fixed the cap: values above $8 require PART60_CAP_OVERRIDE=1 and are recorded.
20. Corrected the one-RNG-stream sampling wording in DEFINITIONS-60.md.

### A/C high-severity class checks
- A/Luna 1 (collect before asynchronous batch completion): checked present, fixed with bounded processing_status polling before results. Covered by test_collect_waits_for_batch_processing_to_end_before_results.
- A/Luna 2 (Qwen runs on CPU): checked present, fixed by moving the model, input tensors, and hook/control tensors to the selected device. Static inspection only; no Qwen load was allowed.
- A/Luna 3 (archive path disagrees with runbook): checked present, fixed with matching /root archive defaults and documented paths.
- A/Luna 4 (RETURN_READY can hide missing work): checked present, fixed with required generation outputs, per-phase score verification, reproduction verification, and matching score metadata before archive readiness.
- A/Luna 5 (retry/changed plans can duplicate paid requests): checked present, fixed with no blind POST retry, per-chunk tags and unit-plan fingerprints, locked ambiguous chunks, and overlap refusal. Covered by reservation/reconcile and resume tests.
- A/Luna 6 (resume can reuse activations for different inputs): checked present, fixed with fingerprints covering text, SID, unit, model, dtype, device, backend, and batch size. Added test_phase_outputs_and_score_fingerprint_include_inputs_and_configuration.
- A/Haiku 1 (M6 thinking field is off): checked present, fixed by the adaptive smoke/drop protocol and tests above.
- A/Haiku 2 (POST retries can double-pay): checked present, fixed by limiting automatic retry to definitive 429 and reserving/locking uncertain outcomes. Covered by batch lifecycle tests.
- A/Haiku 3 (changed arguments resubmit overlapping units): checked present, fixed by request-plan/unit overlap checks and refusing incompatible resume plans.
- C/Luna 1 (activation IDs cannot be joined): checked present, fixed by canonical load_acts key normalization. Covered by the saved-NPZ production round-trip and SID sabotage.
- C/Luna 2 (model snapshot path mismatch): checked present, fixed by recording snapshot_download's resolved path and passing it into score/verify.
- C/Luna 3 (M3/M5 phase routing): checked present, fixed with one explicit phase map shared by scoring and analysis. Phase paths are directly asserted in test_phase_outputs_and_score_fingerprint_include_inputs_and_configuration.
- C/Luna 4 (M4 marked scores missing/misaligned): checked present, fixed by output-column indexing, clean-text marker spans, token offsets after marker removal, and position-0 exclusion. Missing/non-finite marked values leave both comparators unscored. Tokenizer/model numerics were not run.
- C/Haiku 1 (load_acts keys disagree with analyzer): checked present, fixed and covered by the production round-trip/SID sabotage.
- C/Haiku 2 (Hooked uses invalid tensor/index namespaces): checked present, fixed with device-local torch index/direction tensors and explicit column destinations. Static inspection only; no model was loaded.
- C/Haiku 3 (--limit ignored): checked absent; there is no --limit option or matching code path.
- C/Haiku 4 (M0 scores marked strings): checked present, fixed by joining B-arm SIDs to clean sent.all.jsonl texts before scoring.
- C/Haiku 5 (M6 superseded spec): checked present, fixed under Amendment 1 as above.
- C/Haiku 6 (second-call failures abort; >10% rule only warns): checked present, fixed with per-unit eligibility/evidence and durable method drop honored by analysis. Covered by second-call preservation and generation-drop tests.
- C/Haiku 7 (reserve double-counting and cross-split cost): checked present, fixed with chunk-keyed settlement replacing reserve and split-scoped cost reporting.
- C/Haiku 8 (60c uses different units): checked present, fixed with paired intersection and paired count.
- C/Haiku 9 (M5 round 2 omits prior hypothesis): checked present, fixed; round-two request includes the prior hypothesis plus boolean outcomes, without activation numbers.
- C/Haiku 10 (tests miss production round trip): checked present, fixed with test_generated_and_scored_files_round_trip_through_production_analysis and production-path sabotage tests.
- C/Haiku 11 (M4 compares marked and whole-text on different strings): checked present, fixed by scoring the clean text in both passes and matching the same four sentences.

### Papercuts
- alert-203608-8pixy-0010: adaptive M6 thinking, 4000-token ceiling, smoke/drop gate folded into code, tests, and runbook.
- alert-203839-8pixy-0012: M7 is not a required preflight file; pip freeze is advisory; both have wrapper coverage.
- alert-203841-8pixy-0013: runbook describes direct smoke and per-request batch errors; batch error drops M6; covered by M6 tests.

### Validation and touched files
Full suite: 41 tests passed. The three sabotage checks (threshold owner-row, M3 winner selection, SID key format) all passed by proving a production-path mutation fails. Python py_compile and bash -n run60.sh passed. No live API or GPU/Qwen validation was permitted.
Touched: gen60.py, score60.py, analyze60.py, test_part60.py, run60.sh, RUNBOOK-60.md, DEFINITIONS-60.md, prompts/M3.txt. PART60-DESIGN.md was not changed.

## BLOCKERS
None requiring a decision. Live Anthropic behavior and Qwen/GPU numerics remain unverified by instruction. Ambiguous POST responses remain reserved until reconciliation to avoid duplicate batches.

