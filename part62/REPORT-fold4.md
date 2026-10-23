# Part 62 fold 4 report

## SUMMARY

- Part 59 now exposes HTTP status and response body on HTTP failures, allowing Part 62's POST adapter to pass validated 4xx errors to the shared ledger release API.
- Part 62 now passes the Haiku $4 sub-cap into the shared ledger's single-lock reserve operation.
- Added focused tests for rejected-POST release and preserved Part 59 response details; Part 62's existing and added tests pass in the project virtualenv.
- This fold is incomplete: resumable multi-chunk lifecycle, failed-only retry, SIGKILL window coverage, and requested copied-code mutation checks were not implemented.

## FINDINGS / WHAT SHIPPED

| Finding | Status | Evidence |
|---|---|---|
| M1 rejected POST reserve release | Partly fixed | `part59.blind_setting.call()` raises `APIHTTPError(status, body)` rather than terminating on `HTTPError`. Part 62 translates POST `APIHTTPError` into paidaction `HTTPFailure`, allowing `submit_in_chunks` and the ledger to release only validated Anthropic 4xx bodies. A test verifies a rejected POST ends in `released` with no active reserve. |
| L3 Haiku sub-cap race | Fixed in caller | `_StageLedger.reserve()` supplies `subcap_usd=4.0` and `subcap_key=part62-haiku` to the shared ledger under its accounting lock. Removed the separate state read/check. The cap test passes. |
| H3 resume and chunk lifecycle | Open | Existing Part 62 `submit()` still invokes `submit_in_chunks()` as one-shot synchronous callback code. It does not persist a run plan that can resume by saved batch ID after process death, and lacks the Part 63 unknown-outcome retirement/retry flow. |
| M3 failed-only retry | Open | Part 62 still has no collect/retry command that resubmits only failed custom IDs while retaining settled rows. |
| Kill-window matrix and copied-code mutations | Open | No fake-transport SIGKILL matrix at the four POST-window points or new copied-code mutation suite was added. |

## BLOCKERS

- H3, M3, and the requested kill matrix remain unimplemented. The Part 62 submit path needs a durable plan, per-chunk state, recovery by saved batch ID, explicit refusal for `ambiguous`/unknown POST outcomes, an operator retire command that keeps reserve counted as spent, and failed-only retry handling. No Part 62 code or tests currently prove those properties.
- The requested suite of copied-code mutation checks for each new lifecycle behavior remains absent.

## UNVERIFIED ASSUMPTIONS

- No live Anthropic call was made. The provider's exact 4xx billing behavior is not verified; the code relies on the shared ledger's existing strict body validation before release.
- The full Part 59 test suite was not run. The focused HTTP status/body regression passed.
- Parent-control mailbox reads before work, after substantial operations, and before reporting all returned no messages; there were no dispositions to record.
