# Part 62 fold 3 report

## SUMMARY

- Fixed calibration pairing from the Part 63 validated join, deduped texts, and recorded realised class counts.
- Fixed calibration gates and answer parsing to use the computed output-token headroom.
- Fixed analyzer support for the writer's `batch_ids` and per-batch key files.
- Fixed the 62g bracket-straddling status and the OFFWORLD `star` false positive.
- `unittest` completed 36 tests successfully; the five existing code-copy mutation checks all went red for their intended mutations.
- This fold is incomplete: resumable submit/collect and several paid-reader findings remain open.

## FINDINGS / WHAT SHIPPED

| Finding | Status | Evidence / reason |
|---|---|---|
| H1 calibration text/label mismatch | Fixed | `calibration_sample()` joins `answers63.jsonl` and `key63.json` by answer ID; it excludes duplicate texts with conflicting labels. |
| H2 real submit output rejected | Fixed | Analyzer unions key files named by `batch_ids`, validates the complete answer key, and no longer requires nonexistent `wrong_ids`. Round-trip fixture now uses chunk key files. |
| H3 rerun re-buys matrix | Deferred | Submit still has no durable resume/collect lifecycle. Requires a larger redesign and kill-window fake-transport coverage; no claim of kill safety. |
| H4 output headroom checked against 1,024 | Fixed | `parse_batch_results(..., max_tokens=answer_max_tokens)` now uses the answer request limit. |
| M1 rejected POST reserve release | Deferred | `part59.blind_setting.call()` exits on HTTPError before Part 62 can raise paidaction `HTTPFailure`; Part 59/shared library edits are outside the Part 62 domain. |
| M2 production scoring gate bypass | Fixed | Production calibration and final scoring path now invokes `scoring_gate()`. |
| M3 failed-only retry | Deferred | Failed custom IDs are still not resubmitted; requires the omitted resume/collect flow. |
| M4 calibration distinctness | Fixed | Pool is deduped by text; selected sample is exactly 120 distinct texts and realised class counts are stored in batch metadata/label records. |
| M5 bracket straddles target band | Fixed | 62g now reports NOT-EVALUABLE when the interval overlaps but is not contained in or disjoint from the registered band. |
| M6 settle overrun enforcement | Blocked by scope | Finding is in `paidaction/paid_ledger.py`; brief explicitly forbids paidaction edits. No shared-ledger changes made. |
| L1 terminal batch states | Partly fixed | Polling checks terminal request counts and has a one-hour bound. API field semantics remain unverified without network. |
| L2 `star` matches `started` | Fixed | OFFWORLD regex requires a whole-word `star`; regression assertion added. |
| L3 Haiku sub-cap race | Deferred | `_StageLedger.reserve()` still checks the sub-cap separately from shared atomic reserve; fixing safely requires a shared-ledger extension, prohibited by scope. |
| L4 ledger path across checkouts | Blocked by scope | Default comes from the shared library path, but cross-checkout coordination remains a shared paidaction concern and is outside write scope. |
| L5 duplicate math import | Fixed | Removed duplicate import. |
| alert-004556 (calibration decision) | Fixed | Applied the DISPOSED Part 63 join, origin-question filter, text dedupe, deterministic sampling, and recorded class counts. |
| alert-004558 (resume/headroom) | Partly fixed | Headroom fixed; durable resume, collect, and no-repeat-POST guarantee remain deferred under H3. |
| alert-000132 (ledger path/worst-case/calibration) | Partly fixed | Existing default path and worst-case chunk reservations retained; calibration is corrected. Cross-checkout scope and settled-usage overrun remain blocked. |
| alert-000836 (settle usage) | Deferred | Per-chunk settlement exists, but shared-ledger overrun validation is not changed due the explicit paidaction edit prohibition. |

## TESTS

Ran `/home/elisepc/neuron-probe/.venv/bin/python -m unittest` from `part62`: **36 tests passed**. The suite's five mutation checks modified copies of the production functions and each failed as expected: 62c boundary, alien-frame conversion, phase-position mask, logistic midpoint sign, and control scaling. Mutation copies now include the Part 63 join fixtures so import-time calibration succeeds. New assertions cover key-based calibration pairing, distinct sample texts, chunked analyzer key files, `started` proxy exclusion, and 62g bracket overlap.

## UNVERIFIED ASSUMPTIONS / WHAT COULD NOT BE CHECKED

No network/API calls were made. The meaning and timing of Anthropic batch `request_counts` terminal states, remote polling behaviour, and real result formats remain unverified. No kill-matrix test was added for Part 62, so resumability and duplicate-POST safety are explicitly not established. The sub-cap check remains racy, and the shared ledger does not reject a settlement that exceeds reserve/cap. No live reserve or spend occurred. Parent-control mailbox reads returned no messages; no message was applied, declined, or deferred.

## BLOCKERS

- Paidaction edits are expressly prohibited by the brief; M6, L3, and cross-checkout L4 require that shared boundary to change.
- H3/M3 remain open in Part 62 pending resumable chunk recovery and failed-only retry implementation with a fake-transport kill matrix.
- M1 depends on changing the out-of-domain Part 59 HTTP helper so status/body reach the Part 62 ledger adapter.
