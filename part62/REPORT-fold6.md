# Part 62 fold 6 report

## SUMMARY

- Implemented a durable Part 62 `run-all` lifecycle with a lock, a run plan, a separate durable record for every chunk, and resume by saved batch ID.
- Added calibration-first enforcement: only a passing 120-answer Haiku calibration with kappa >= 0.8 and no unlabelled or truncated responses permits answer submissions.
- Added explicit ledger retirement, retained reserves, raw POST response persistence, and fake API kill-boundary tests.
- Added pinned Vast model prefetch, offline mode, success and incomplete archives, and recovery commands in the runbook.
- The required unittest command completed with 42 tests `OK`; its intentional mutation-copy failures were observed.
- No paid API, network, or GPU operation was performed.

## FINDINGS / WHAT SHIPPED

- `part62/blind62.py:449-455` enforces the Haiku calibration record requirements. `part62/blind62.py:642-669` collects and records calibration before entering the answer loop; missing, pending, failed, truncated, or low-kappa calibration prevents answer POSTs. Request construction retains `max_tokens=1024` and does not include `temperature`.
- `part62/blind62.py:468-473, 549-567, 590-641` writes the run plan and one submission record per chunk, reserves before POST, writes `posting` before the call, fsyncs the raw response before interpreting its ID, and resumes a recorded batch by polling and collecting it. The ledger's idempotent settlement protects a rerun after collection but before local state update.
- `part62/blind62.py:627-638, 790-810` refuses unknown POST outcomes without resubmission and prints the exact operator retirement command. `retire` records the shared-ledger retirement; the next run records `retired-unknown`, and the ledger keeps the reserve counted. `part62/RUNBOOK-62.md:99-125` documents resume, inspection, and retirement commands.
- `part62/test_part62.py:793-899` covers missing/pending/failed calibration, the no-temperature request shape, and fake API kills after reserve, after POST before response persistence, after batch ID recording, and after result persistence before settlement. Rerun checks assert no duplicate POST and one settled ledger attempt where a batch ID is known.
- `part62/run62.sh:27-40, 59-74` fetches the two revisions pinned in `run62.py` into `/root/hf/hub`, turns on offline mode, creates complete or incomplete return archives, and prints `RETURN_READY` or `RUN_FAILED`. `part62/RUNBOOK-62.md:35-72` gives the Vast invocation and archive behavior.
- Verification: `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part62.test_part62` ended `Ran 42 tests ... OK`; `bash -n part62/run62.sh` and `python3 -m py_compile part62/blind62.py part62/test_part62.py` completed successfully.

## BLOCKERS

- Decision needed on an internal policy conflict. Amendment 2 permits an explicit Opus fallback after a failed Haiku gate, but this fold explicitly requires the Haiku calibration to pass before *any* answer batch. I implemented the stricter ordering and block Opus requests; allowing fallback would violate the stated gate. Please answer the question in `HOLD.request` before treating the fold as complete.
- Provider-side POST acceptance cannot be proven if the process dies before the response is durably written. The implementation does not retry that chunk; it leaves the reserve counted and requires operator reconciliation or retirement.
- No live API behavior or rented 3090 run was exercised, per the no-paid-call/no-GPU scope.
- Parent-control mailbox was empty at all checkpoints; no messages changed or were declined from the plan.

## SELF-ASSESSMENT

The fake API tests verify local state transitions and ledger idempotence, not Anthropic provider behavior. The fallback conflict is the one decision I did not infer. The Vast prefetch shell uses the exact revisions already registered in `run62.py`; it was syntax-checked but not run against a rented host.

## Fold 6b

### SUMMARY

- Applied the 2026-10-23 ruling: Haiku answers require a recorded passing calibration; explicitly selected Opus answers are permitted only after a recorded failed Haiku calibration.
- Opus uses its own durable run-plan file and sends only answer chunks. It does not calibrate itself, and default `run-all` remains Haiku.
- The previous Fold 6 policy blocker is resolved by this ruling. No paid API, network, or GPU operation was performed.

### FINDINGS

- `part62/blind62.py:438-465` recognizes only a recorded failed Haiku gate, including failures with incomplete calibration labels. `part62/blind62.py:594-627, 666-696` refuses Opus otherwise, prints the Haiku resume command, keeps Haiku gated on a pass, and allocates Opus its own durable run plan. `part62/test_part62.py:523-549` covers missing, pending, passed, and failed records, including a failed sample with 119 labels.
- `part62/blind62.py:611-613, 697-730` excludes calibration requests from the Opus run and applies answer-label/completion checks without requiring Opus calibration. `part62/test_part62.py:878-910` verifies an explicitly selected Opus run starts only the answer stage after a saved failure. `part62/test_part62.py:873-876` verifies default `run-all` selects Haiku and does not start Opus.
- Haiku's missing, pending, and failed calibration states refuse answer chunks: `part62/blind62.py:666-696`; coverage is in `part62/test_part62.py:827-871`. Existing ledger policy remains shared at the $8 cap with Haiku's $4 sub-cap (`part62/blind62.py:380-412`); `part62/test_part62.py:494-500` covers the sub-cap, and sequential 200-request chunk planning is covered at `part62/test_part62.py:253-264`.
- `part62/RUNBOOK-62.md:93-130` now documents the explicit Opus command policy, distinct run-plan files, sequential reservations, and the refusal/resume behavior.
- Verification: `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part62.test_part62` completed with 45 tests `OK`. Its five deliberate mutation-copy probes printed the expected red failures. `git diff --check` passed.
- Parent-control mailbox reads were empty at each checkpoint; no messages changed the plan or required disposition.

### BLOCKERS

- None. The registered Opus fallback is implemented and covered by tests. Live provider behavior remains unverified because this fold forbids paid calls.

## Fold 7

### SUMMARY

- Fixed all seven Fold 7 findings across settlement, POST rejection handling, calibration recovery and provenance, runbook wording, and signal cleanup.
- Retired calibration no longer authorizes Opus; `run-all --retry-calibration` creates a separately reserved retry and preserves the retired attempt.
- The required unittest command completed with 52 tests `OK`; no paid API, network, or GPU operation was performed.
- Parent-control mailbox reads were empty; no messages changed the plan or required disposition.

### FINDINGS

- **F1** — `part62/blind62.py:558-595` sums usage only from succeeded rows, saves non-succeeded completion reasons, and leaves scoring blocked when those issues exist. `part62/test_part62.py:846-883` verifies an errored row settles at succeeded usage while `scoring_gate` refuses the labels.
- **F2** — `part62/blind62.py:599-628` saves the HTTP error body, releases a reserve only for a parsed, accepted Anthropic 4xx through `paid_ledger.release`, and records a definitive rejection as `rejected`. `part62/test_part62.py:885-915` drives this through `run_all` and checks the ledger release.
- **F3** — `part62/blind62.py:486-504, 766-799, 952-953` uses the current Haiku run record for Opus eligibility and adds `--retry-calibration`, which makes a new reserved attempt after retirement. `part62/test_part62.py:553-571` checks a retired run cannot inherit an older failed labels file; `917-1009` exercises the printed retry flow and verifies the retired and new attempts remain distinct.
- **F4** — `part62/blind62.py:380-400, 633-649` fsyncs the exact POST response text before JSON parsing and keeps malformed bodies. `part62/test_part62.py:1011-1024` checks the exact malformed body remains on disk.
- **F5** — `part62/blind62.py:807-819` records calibration seed, source sample IDs, and realized class counts in the run record and labels output. The retry integration test at `part62/test_part62.py:917-1004` checks both records.
- **F6** — `part62/blind62.py:63-129` keeps the Parts 54a, 54b, 26.8, and 59a pool and redistributes class quotas as availability requires; `part62/RUNBOOK-62.md:85-92` names that pool. `part62/test_part62.py:573-591` checks the selected source set, balanced realized counts, and redistribution under a scarce class.
- **F7** — `part62/run62.sh:5-28` traps SIGINT and SIGTERM, writes the incomplete archive, prints `RUN_FAILED`, and exits with signal status. `part62/test_part62.py:1026-1058` exercises both signals.

### BLOCKERS

- None. The full suite reported `Ran 52 tests ... OK`; its five `MUTATION RED (copy)` diagnostics are expected mutation checks. No torch-missing errors occurred in this environment. Live provider billing remains unverified because the fold prohibits paid calls.

## Fold 8

### SUMMARY

- Fixed the Fold 7 restart wedges, retired-answer dead end, incomplete-row recovery, and shell signal handling.
- Added explicit new-reserve retry paths for incomplete calibration, retired answer chunks, and unresolved answer rows.
- Corrected Part 62's eligible calibration pool to Parts 54a, 54b, and 26.8; the seeded sample is 32, 41, and 47 rows respectively.
- The required suite completed with 60 tests `OK`; no paid API, network, or GPU operation was performed.
- Parent-control mailbox reads were empty; no messages changed the plan or needed disposition.

### FINDINGS

- **H1** — `part62/blind62.py:840-844, 927-995` reconciles released attempts and assembles labels only from settled answer attempts with a batch id. A rejected chunk can be retried and its successful batch writes a labels file. Tests: `part62/test_part62.py:926-969` and `970-1017` cover the rejected rerun and the crash after ledger release.
- **M1** — `part62/blind62.py:488-522, 574-597, 893-926` records provider errors, unlabelled rows, and truncations as incomplete, keeps Opus locked, and offers `--retry-calibration`. Only a complete 120-row below-threshold calibration authorizes Opus. Tests: `part62/test_part62.py:525-556, 1192-1251`.
- **M2 / Luna release-retire seam** — `part62/blind62.py:736-785, 824-867, 879-892` reconciles `released` and `retired` ledger states, prints usable next commands, adds `--retry-answers`, and prevents retiring a submitted batch. Tests: `part62/test_part62.py:1018-1114, 1253-1310`.
- **M3** — `part62/blind62.py:600-610, 927-995, 1066-1105` lets `--retry-failed-answers` submit only unresolved rows under new reserves; successful prior rows stay in the final labels. Test: `part62/test_part62.py:1312-1365` verifies a single errored row is retried without reposting the two successes.
- **M4 (pre-existing)** — `part62/blind62.py:57-130, 579-592` and `part62/RUNBOOK-62.md:85-88` describe the eligible three-part origin pool; `part62/test_part62.py:579-594` pins its realized source counts. The root `PART61-63-DESIGN.md` still contains the four-source claim. The brief limited writes to `part62/`, so that registered document was not edited.
- **L1/L2/L3** — `part62/blind62.py:824-867, 736-785` recovers the release-before-run-write crash and refuses to retire a submitted batch. `part62/run62.sh:3-81` traps HUP and runs long children under `wait`, forwards termination, archives partial output, and reports `RUN_FAILED`. Tests: `part62/test_part62.py:970-1114, 1381-1449`.
- Verification: `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part62.test_part62` completed with `Ran 60 tests ... OK`; its five mutation-copy red traces are expected. `git diff --check` passed.

### BLOCKERS

- The registered root design still names Part 59a as a calibration source; editing it was outside the declared `part62/` scope. Part 62's runbook, code comment, and source-count test now record the measured eligible pool.

## Fold 9

### SUMMARY

- Fixed all six requested round-3 findings: POST response recovery, the Amendment 5 calibration pool, bounded failed-row retry, retry documentation, class-and-tag calibration agreement, and Part 62-only retirement scope.
- The Part 62 suite completed with 67 tests `OK`; syntax, shell, and whitespace checks passed.
- No paid API, network, or GPU operation was performed. Parent-control mailbox reads were empty.

### FINDINGS / WHAT SHIPPED

- **Haiku F1 — POST response persistence.** If saving the raw successful POST response raises `OSError`, the code parses its batch id and records that id before proceeding; if recording also fails, it prints the raw response to stderr. If raw response persistence succeeded but the run-record update failed, the durable `response-saved` state resumes collection and cannot be retired or resubmitted. Tests: `BlindLifecycleTests.test_post_response_save_failure_records_batch_id_and_does_not_repost`, `test_post_response_and_batch_id_save_failure_prints_raw_response`, and `test_saved_post_response_without_run_record_uses_distinct_state_and_never_reposts`.
- **Haiku F2 — Amendment 5 calibration pool.** Removed the stale four-source comment and `part59a` filter entry. The source test reads Amendment 5 from `main:PART61-63-DESIGN.md`, asserts its three-source pool, and preserves measured sample counts 54a=32, 54b=41, 26.8=47. Test: `SubmissionTests.test_calibration_pairs_joined_text_and_label_by_part63_key`.
- **Haiku F3 — failed answer retries.** A persisted retry-chunk marker tracks custom ids already retried, limiting `--retry-failed-answers` to one paid retry per row in the run record. Test: `BlindLifecycleTests.test_retry_failed_answer_rows_are_retried_at_most_once_per_run`.
- **Haiku F4 — runbook.** Clarified that released answer rejections retry on the next `run-all`, while calibration rejections require `--retry-calibration`; documented response-saved recovery and the per-row retry limit. Test: `RunbookTests.test_runbook_matches_calibration_and_failed_answer_retry_paths`.
- **Haiku F5 — calibration gate.** Cohen's kappa and wrong-row accounting now compare `(class, tag)`; a matching class tagged `BROKEN` is disagreement. Test: `BlindGateTests.test_origin_calibration_gate_rejects_matching_class_with_broken_tag`.
- **Luna LOW — retirement scope.** `retire_attempt` requires a `part62-haiku` or `part62-opus` ledger attempt, a submission id, and an exact match to one chunk in the corresponding Part 62 run record. Test: `BlindLifecycleTests.test_retire_requires_part62_kind_and_exact_run_chunk_mapping`.
- Verification: `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part62.test_part62` completed with `Ran 67 tests ... OK`; `python -m py_compile part62/blind62.py part62/test_part62.py`, `bash -n part62/run62.sh`, and `git diff --check` passed.

### BLOCKERS

- None. Amendment 5 is registered on `main`, and the permitted write scope covered the code, tests, runbook, and report changes. No parent-control messages required disposition.

## Fold 10

### SUMMARY

- Unknown POST outcomes now require `--no-batch-verified` before retirement; provider-visible request IDs carry the chunk submission ID for reconciliation.
- Short result sets save raw JSONL, record `results-incomplete` and the batch ID, and allow retirement while the reserve remains counted.
- Class-and-tag calibration agreement remains in place, with comments and tests citing Amendment 6.
- The Part 62 suite completed with 69 tests `OK`; no paid calls, network, GPU, or commit were used.

### FINDINGS / WHAT SHIPPED

- **Finding 1 — unknown POST outcome can no longer be retired silently.** `posting` and `ambiguous` chunks without a durable batch ID refuse retirement unless the operator passes `--no-batch-verified`. The refusal includes the full command and instructs the operator to search for a provider batch carrying the chunk's submission ID. New requests include that ID in their provider-visible `custom_id`; prior run records keep their original ID mapping. Tests: `BlindLifecycleTests.test_s2_and_s3_unknown_post_outcomes_require_no_batch_verified`, `BlindLifecycleTests.test_kill_after_post_before_id_record_is_ambiguous_and_not_reposted`, and `BlindLifecycleTests.test_retired_answer_prints_and_accepts_explicit_retry`.
- **Finding 2 — short result sets have an operator exit.** Raw rows are saved before count validation; a short batch records `results-incomplete`, its row count, path, and batch ID, while leaving the reserve active. `retire` accepts this known short-result state without the unknown-batch flag. Test: `BlindLifecycleTests.test_short_batch_results_are_saved_retirable_and_keep_reserve`.
- **Finding 3 / Amendment 6 — calibration agreement uses class and tag.** `_calibration_wrong` and `_calibration_kappa` retain the registered class-plus-tag rule, and both the implementation comment and the BROKEN-tag test cite Amendment 6. Test: `BlindGateTests.test_origin_calibration_gate_rejects_matching_class_with_broken_tag`.
- **Finding 4 — test and RUNBOOK gaps.** RUNBOOK-62 now explains the provider search by submission-ID prefix, the `--no-batch-verified` condition, and the short-result retire/retry flow. Test: `RunbookTests.test_runbook_matches_calibration_and_failed_answer_retry_paths`.
- Baseline regression check: running the two new lifecycle tests with `git show HEAD:part62/blind62.py` produced the expected red assertions for both S2/S3 subcases and the short-result case.
- Verification: `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part62.test_part62` completed with `Ran 69 tests ... OK`; `python -m py_compile part62/blind62.py part62/test_part62.py` and `git diff --check` passed. Parent-control reads were empty.

### BLOCKERS

- None. The provider batch-listing route was not verified because this fold prohibits network access; RUNBOOK-62 states that limitation and directs the operator to the provider-supported inventory view.

## Fold 11

### SUMMARY

- Aligned run-all label records with analyze62's strict calibration and completion checks.
- Added an offline `rebuild-labels` command that creates a separate `-rebuilt.json` file.
- Added an end-to-end fake-API test through analysis and rebuild; all 75 Part 62 tests pass.
- No paid API, network, GPU, or ledger operation was used by this fold.

### FINDINGS / WHAT SHIPPED

- **Writer/analyzer contract.** Calibration records now carry `passed`, the `wrong` count, `completion_issues`, and the calibration labels used to recompute kappa and wrong-label counts. Completion records include `accepted_stop_reasons`. The analyzer continues to enforce its existing gates and reads calibration labels from the calibration object. Answer keys are saved per batch so the writer's labels record can round-trip through analysis. Tests: `WriterReaderTests.test_real_writer_outputs_round_trip_through_blind_reader` and `LabelRecordRebuildTests.test_fake_run_all_record_analyze_and_offline_rebuild_round_trip`.
- **Offline label rebuild.** `python3 part62/blind62.py rebuild-labels --review-dir DIR` reads the latest saved run-all record and per-batch keys, refuses incomplete or unsettled runs, does not contact a provider or touch the ledger, and creates `part62-labels-LAST_BATCH-rebuilt.json` without replacing an existing file. Tests: `LabelRecordRebuildTests.test_rebuild_labels_cli_dispatches_review_directory`, `test_rebuild_requires_complete_run_and_every_chunk_settled`, and the end-to-end round trip above.
- **Runbook.** Added the rebuild command and its input, refusal, and output behavior to `part62/RUNBOOK-62.md`.
- **Verification.** `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part62.test_part62` completed with `Ran 75 tests ... OK`. The five mutation-copy red traces are expected. `git diff --check` passed.
- Parent-control mailbox reads were empty; no messages changed the plan or required disposition.

### BLOCKERS

- None.


## Fold 12

### SUMMARY

- Read-only reconstruction verified the legacy run against its saved request and label artifacts.
- No implementation or review-directory artifacts were written because the required sidecar metadata conflicts with the unchanged analyzer's exact key check.
- A parent decision is needed on where reconstruction markers and check counts may be stored.

### FINDINGS / WHAT SHIPPED

- Rebuilt the 1,536-answer shuffle and 120-row registered calibration request set in memory. Its fingerprint matched the saved run record.
- All 1,656 custom IDs and provider custom ID maps matched the run chunks and both persisted submission and batch records across nine chunks. Parsing all 1,656 raw rows reproduced every saved chunk label, all 1,536 original answer labels, and all 120 calibration labels exactly. Recomputed calibration kappa was `0.850622406639004`.
- Eight answer key sidecars are absent. `analyze62._label_inputs` requires each parsed sidecar map to equal the expected answer metadata map exactly. Adding `reconstructed` or check-count fields to those JSON maps makes that check fail. The brief also forbids changing `analyze62.py`, so no compatible representation for per-sidecar metadata was authorized.
- No production code, tests, runbook, sidecars, or rebuilt labels record were changed or created. No tests were run; no real-data analysis, network, or paid operation was performed.
- Parent-control mailbox reads were empty; no messages changed the plan or needed disposition.

### BLOCKERS

- Decision required: may the key files remain exact fold-11 answer maps, with `reconstructed: true` and check counts stored in the rebuilt labels record only, or should the analyzer be authorized to accept a metadata wrapper? The first option preserves the stated no-analyzer-change constraint; the second does not.

## Fold 12

### SUMMARY

- Implemented a legacy `rebuild-labels` path that recreates the request plan from the saved answer folder and registered calibration sample.
- The updated brief resolved the prior HOLD: sidecars remain exact fold-11 maps; reconstruction metadata lives in the rebuilt record and provenance file.
- Full Part 62 tests passed, and the required real local rebuild completed after all proofs passed.

### FINDINGS / WHAT SHIPPED

- `blind62.py` now verifies the saved request fingerprint, every reconstructed custom ID and provider ID against run, submission, and batch records, raw rows against saved chunk/original answer/calibration labels, and the recomputed calibration kappa. It refuses mismatches before creating files and uses no network or paid calls.
- The real run verified 9 chunks, 1,656 custom IDs and 1,656 provider IDs, 9 submission records, 1,536 answer labels over 8 batches, 120 calibration labels, and kappa `0.850622406639004`.
- Created 8 new exact fold-11 key sidecars for 1,536 answers, `part62-labels-msgbatch_014oPiaUi9KYDFhAR2CsEZbr-rebuilt.json`, and `part62-rebuild-provenance.json`. The rebuilt record and provenance contain all three proof summaries and their counts. A direct reconstruction audit confirmed every sidecar map exactly matches the deterministic key map.
- Tests cover a fake run after deleting its keys, refusal on a tampered raw row, and refusal when the changed shuffle seed changes the request fingerprint. `/home/elisepc/neuron-probe/.venv/bin/python -m unittest part62.test_part62` completed with 75 tests `OK`; the printed mutation-copy failures are expected. `git diff --check` passed.
- Updated the runbook. `analyze62.py` remains unchanged; real-data analysis was not run. Parent-control reads were empty; no messages changed the plan or required disposition.

### BLOCKERS

- None.

## Fold 13

### SUMMARY

- Rebuilt the legacy Haiku calibration details through the same `_calibration_record` function used by the run-all writer.
- Rebuilds now keep answer labels in the root `labels` map and place verified calibration labels and gate fields in `calibration`.
- Added versioned output names so a rebuild creates a fresh `-rebuilt2.json` record rather than replacing the existing `-rebuilt.json` file.
- The Part 62 suite completed with 75 tests `OK`; expected mutation-copy failures were printed. `git diff --check` passed.
- The real local rebuild verified 9 chunks, 1,656 custom/provider IDs, 1,536 answer rows, 120 calibration rows, and kappa `0.850622406639004`.

### FINDINGS / WHAT SHIPPED

- `_legacy_rebuild_proof` now derives calibration `passed`, `wrong`, `completion_issues`, and all related fields from raw-verified calibration labels using `_calibration_record`. Legacy records missing those fields take this verified rebuild path even when their key sidecars already exist; every existing sidecar must match exactly before reuse.
- The fake old-format run test removes its keys and gate fields, rebuilds the labels, confirms answer/calibration labels and kappa, and calls `analyze62._label_inputs` directly. The test also checks a second rebuild uses `-rebuilt2.json` without changing the first output.
- The real command created `results/part62-review/part62-labels-msgbatch_014oPiaUi9KYDFhAR2CsEZbr-rebuilt2.json` and `part62-rebuild-provenance-rebuilt2.json`. Existing exact key sidecars were reused. Direct `_label_inputs` validation accepted the record and confirmed all 1,536 answers and 120 calibration labels; no full analyze CLI was run.
- No network, paid call, or commit was used. Parent-control mailbox reads were empty; no messages changed or were deferred from the plan.

### BLOCKERS

- None.

### SELF-ASSESSMENT

- The real-data proof covers the saved run artifacts, deterministic request plan, existing sidecars, and `_label_inputs` contract. The full analysis CLI was intentionally not run under the brief.
