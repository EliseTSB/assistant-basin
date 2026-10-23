# Part 60 round-4 fold report

## SUMMARY

- Implemented the Round 4 spend, recovery, M6 smoke, M0 verification, and analysis-status changes required by Amendments 5 and 6.
- The offline suite passes: 82 tests. Python compilation and `git diff --check` pass.
- No prediction, method definition, or metric definition changed. No network/API request, Qwen load, GPU run, or commit was made.

## FINDINGS

### Review dispositions

| Finding | Disposition and evidence |
|---|---|
| Luna 1 / Haiku H5 — a killed M6 smoke could reuse its attempt identity and erase spend | Fixed. Every smoke check and paid request receives a fresh random attempt ID; the started check is persisted before the call and raw responses are keyed by `custom_id`. `test_smoke_kill_matrix_uses_fresh_ids_and_analysis_totals_stay_stable` kills and reruns at four stages and checks unique IDs, ledger entries, and analysis totals. |
| Luna 2 / Haiku H4 — unknown direct-call outcomes released the reserve | Fixed. A reserve is released only for a 4xx with an Anthropic-shaped error body. Timeouts, 5xx, 429, malformed responses, and other unknown outcomes retain the reserve; known completed smoke usage is settled separately. Covered by `test_m6_smoke_transient_errors_are_retryable_and_do_not_drop_state`, `test_retryable_smoke_error_settles_usage_from_completed_responses`, and the HTTP 400 smoke test. Actual billing for these outcomes remains unverified. |
| Luna 3 / Haiku M5 — an unknown smoke outcome did not reset the consecutive-failure streak | Fixed. A null/unknown check terminates the reverse streak scan. `test_smoke_unknown_streak_mutation_goes_red` mutates the production function in a copy and confirms the mutant fails. |
| Luna 4 / Haiku H3 — collect replay could double-count usage after a kill | Fixed. Raw batch results are durably stored by `custom_id`; analysis derives token and call totals from those results, and collection overwrites row values instead of incrementing usage. The collect kill matrix retries after raw-result write, row write, settlement, and manifest write and verifies stable rows, totals, and ledger state. |
| Luna 5 — timestamp filtering could hide batches from the operator listing | Fixed. Listing returns all pages without filtering by `created_at`; listing remains read-only. The report listing test checks older and newer entries and filesystem non-mutation. |
| Haiku H1 — non-finite cap values bypassed the hard cap | Fixed. Environment and CLI cap parsing require a finite positive number. Tests cover `nan`, `inf`, zero, negative, and non-numeric inputs. `test_cap_finite_guard_mutation_goes_red` confirms removing the finite guard makes a copied implementation fail. |
| Haiku H2 — spend cap reset when `--root` changed | Fixed. The ledger path is fixed under the checkout, and conflicting root-local ledgers are refused. A shared ledger lock now serializes cap-check-plus-reserve across distinct run roots. Tests cover shared-path accounting, empty local-ledger refusal, and a concurrent two-root cap race where only one submission can reserve. |
| Haiku M1 — unresolved reservations could exhaust the cap with no operator remedy | Fixed as Amendment 6 directs. `ledger retire <attempt>` writes a dated retirement event for one unresolved reservation. It does not change other attempts. `test_ledger_retire_writes_a_dated_entry_and_removes_only_that_reserve` covers the transition. |
| Haiku M2 — kill/retry test coverage was incomplete | Fixed. Submit, smoke, and collect kill/recovery tests cover the request lifecycle and assert ledger and analysis totals after reruns. See the kill-matrix tests named above and `test_collect_kill_matrix_rewrite_keeps_rows_and_totals_idempotent`. |
| Haiku M3 — HTTP 400 behavior conflicted with the registered M6 smoke rule | Fixed to Amendment 6. A 400 with an Anthropic error body is a failed all-five check; three consecutive failures drop M6. `test_m6_http_400_is_a_failed_check_and_saves_anthropic_error_body` checks the saved response and ledger releases. |
| Haiku M4 — M0 verification failed open on stale scoring metadata | Fixed. M0 verification now checks the target, output, scoreable entries, and the same fingerprint/model revision/dtype/device/batch/backend/model-dir metadata as other methods. `test_verify_scoring_m0_fails_closed_on_stale_fingerprint` covers stale metadata. |
| Haiku L1 — test selection was always marked ready | Fixed. `test-selection.json` is ready only when dev analysis has every expected method or an explicit skip. `test_test_selection_is_ready_only_for_complete_dev_analysis` checks both incomplete and complete states. |
| Haiku L2 — 60j appeared held after testing one candidate | Fixed. When one candidate was tested without meeting the registered margin, the status says `unscored (one candidate tested)`. The analyzer regression checks this status. |
| Haiku L3 — chunk identity could collide after an operator reset | Fixed by random attempt IDs, used in the ledger and request custom IDs. A reset cannot cause a later attempt to overwrite the earlier ledger identity. The smoke and submit recovery tests verify distinct IDs after kills. |
| Haiku L4 — unjudgeable/unknown-ceiling counts | Not a defect in the reviewed implementation. Existing analysis distinguishes thresholds at or above the observed ceiling from unknown ceilings and retains them in the registered denominators, matching `DEFINITIONS-60.md`; no change was needed. |
| Haiku L5 — inherited price constants are unverified | Accepted limitation. Price constants remain unchanged as required and are identified below as unverified. |
| Papercut alert 233521-8pixy-0038 | Addressed by the matching cap, fixed-ledger, idempotent-collection, smoke-reservation, and fresh-ID changes above. |

### Tests and mutation checks

- `PART60_BASIN=/home/elisepc/assistant-basin /home/elisepc/neuron-probe/.venv/bin/python -m unittest test_part60` — 82 tests passed.
- `/home/elisepc/neuron-probe/.venv/bin/python -m py_compile gen60.py analyze60.py score60.py test_part60.py` — passed.
- `git diff --check -- RUNBOOK-60.md analyze60.py gen60.py score60.py test_part60.py` — passed.
- Six production-copy mutation tests ran and verified that the mutated copy fails: threshold-owner exclusion, SID-key format, ambiguous-batch adoption, finite-cap guard, smoke unknown-streak handling, and M3 winner selection. The two new Round 4 mutation tests target the finite-cap guard and smoke streak reset.
- The new concurrent-ledger test starts two independent run roots against one shared cap with only enough budget for one reservation; it confirms one submit and one refusal.

### External assumptions that remain UNVERIFIED

| Probe item | Status |
|---|---|
| Before the first dev batch, make the manual one-request probe; save the create response ID and compare it with the list response. | Not run. No API call was allowed. |
| Actual batch-list fields and pagination (`data`, `has_more`, `after_id`, `limit=100`), cursor direction, and parseable `created_at`/`request_counts` values. | Not probed against the account. The runner scans pages and no longer uses timestamps to identify or filter batches for correctness. |
| Retrieval by ID, `processing_status` lifecycle, and the completed results endpoint's newline-delimited `custom_id` records, message content, usage, and stop reason. | Documented shapes are used by the runner, but actual account behavior was not probed. |
| Haiku 5.5 accepts adaptive thinking with high effort and `max_tokens=8000` through the batch API, and returns the expected thinking, stop, and M6 schema fields. | UNVERIFIED. The five-request M6 smoke was not run. |
| Whether an Anthropic 4xx error body proves the request was unbilled; whether a timeout, 429, or 5xx is billed after processing. | UNVERIFIED. The runner follows Amendment 6 for 4xx release and conservatively retains unknown-outcome reserves. |
| Current batch/direct token prices and the service payload margin for the local 190 MB request cap. | UNVERIFIED. The inherited rates remain $0.05/$0.25 per million batch tokens and $0.10/$0.50 per million direct tokens. No live price or payload-limit check was made. |
| Live Qwen outputs, tokenizer/device behavior, M0 cross-device numerics, GPU behavior, and actual experiment analysis tables. | UNVERIFIED. No model was loaded, no GPU run was made, and no generated run outputs were available. |

The operator's pre-submit probe checklist remains: create one batch and save its ID; inspect the returned/list fields and request counts; test list ordering and `after_id` pagination; confirm timestamps are usable for informational comparisons; then, before M6 submission, run the implemented five-unit smoke and inspect all five thinking/stop/schema outcomes. None of these live checks was performed in this worker run.

## FILES TOUCHED

`gen60.py`, `analyze60.py`, `score60.py`, `test_part60.py`, `RUNBOOK-60.md`, and this `REPORT-fold4.md`.

## BLOCKERS

None requiring a decision. Live/API/model/GPU checks were excluded by the brief.

## Parent-control mailbox

The mailbox was empty at each required checkpoint. No control message changed, was declined, or was deferred.
