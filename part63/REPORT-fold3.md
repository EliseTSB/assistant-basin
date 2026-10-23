# Part 63 round 3 fold report

## SUMMARY

- Folded the round 3 review findings and the Part 63 papercuts against the Part 63 amendment.
- Fixed label fan-out over every answer row sharing a requested text; the scoring regression checks every corpus key.
- Manual full-corpus submission now refuses an unresolved POST or an unsettled recorded batch.
- Run-all can reuse a durable pre-POST reserve and recover an ID from the fsynced raw POST response.
- The no-response-ID accepted-POST case (K3) remains on HOLD for a parent decision about listing-based adoption.
- Added retry duplicate prevention, stale-label validation, settlement-over-reserve refusal, and monotonic 63c wording.
- Offline suite: 20 tests passed; all 11 copied-code sabotage mutations went red.
- No API call, network access, model load, or paid run was made.

## FINDINGS

| Review item | Disposition | Change / evidence |
|---|---|---|
| HIGH-1 fan-out from answer text to every corpus row | Fixed | `collect` iterates the snapshot's `answer_ids_by_text` values, reads the primary answer ID from each list, and copies its result to all IDs. The fake collection test asserts that every key row is present. Its copied-code sabotage deletes fan-out results and turns the test red. This preserves the amendment's 24 rows per sample while requesting each distinct text once. |
| HIGH-2 assertion iterated only present records | Fixed | The regression assertion compares the collected record keys against the complete answer key and checks every key row's status. `drop-label-fanout` mutates the real collect function in a copy and fails the test. |
| HIGH-3 stuck run across K1–K4 | Partially fixed; K3 decision requested | Run-all resumes a `ready` submission using the same durable reserve, and reconstructs the attempt ID from ledger metadata if the submission file was not updated after reserve. For K4, it recovers the ID from the fsynced `post-response-<submission>.json` and repairs the sidecar. An unresolved `posting` state remains ambiguous and blocks. K3 has no durable response ID; listing adoption was not added without authority. `HOLD.request` asks whether that recovery policy may change. The requested kill-point process matrix was not rerun. |
| HIGH-4 manual resubmit after ambiguous POST | Fixed | A normal `submit` scans existing submissions and the shared ledger; an ambiguous reserve or a recorded unsettled batch blocks a second POST. The ambiguous-post test confirms the second manual submission makes no API call. The error and runbook now direct recovery or operator resolution instead of inviting another submission. |
| MEDIUM-1 retry POST duplicated after kill before labels update | Fixed in code; kill boundary not fault-injected | Before posting a failed-only retry, collection scans submission records for an existing `retry_of_submission_id`. It restores the recorded batch state or preserves the ambiguous state instead of posting again. The ordinary retry merge test passes; the precise kill between retry POST and label write was not simulated. |
| MEDIUM-2 stale labels silently merged | Fixed | Collection verifies prior root label batch ID, root submission ID, and request count before merging. A regression case corrupts the count and checks collection refuses the stale file. |
| MEDIUM-3 settle can exceed reserve | Fixed in Part 63 caller | `settle_reserve` compares actual cost with the recorded reserve and refuses settlement above it, stopping the run. A test passes an explicit ledger path inside Part 63 and checks the reservation remains active. The shared `paidaction` library was not edited. |
| LOW 63c strict rise vs registered monotonic rise | Fixed per design wording | Equality between the +0.5 and +1.0 counts is now allowed. The test exercises a flat monotonic step, and a copied-code mutation that rejects equality makes it fail. |
| LOW 63d Qwen3.5 alias set | Parent amendment takes precedence | Kept the registered exact alias list (`alibaba`, `alibaba cloud`, `tongyi`, `tongyi lab`, `qwen team`, and the registered Chinese aliases). The review's suggested bare `Qwen` alias is not added because Amendment 1 to Part 63 defines the aliases and says the strict-name sensitivity is separate. |
| LOW 63e registered Opus counts | No code change needed | The review verified counts read from the real key match the registered `(0, 0, 1, 19, 24)`. |
| Unused rejection constant and synchronous helper import | Fixed | Removed the unused `DEFINITE_REJECTION_CODES` and `submit_in_chunks` import from `reader63.py`. |
| Papercut 0046: any 4xx releases reserve | Already fixed in fold 2; rechecked | Part 63 passes status and response body to shared `Ledger.release`, whose validation requires a parseable Anthropic error object and excludes 429. No `paidaction` edit was needed or made. |
| Papercut 0047: stale reader/cap report | Already fixed in fold 2; rechecked | Runbook describes Haiku 5.5 and current Part 63 cap assumptions. |
| Papercut 0062 / 0064: fan-out and resubmit wording | Fixed | Fan-out and full-key gate are covered above; ambiguous-outcome text and runbook no longer invite a paid duplicate submission. |
| Disposed papercut naming Part 63 | Folded | Alert 0062's DISPOSED section explicitly repeats the distinct-text request, all-row fan-out, multiplicity denominator, and full-row scoring gate; these are implemented above. Other matching disposed papercuts name Parts 60/62 or were addressed in fold 2. |

## TESTS

- Ran `/home/elisepc/neuron-probe/.venv/bin/python -m unittest` from `part63/`: **20 passed**.
- Eleven sabotage mutations against copied real functions went red: truncation acceptance, condition leakage, zero reserve, missing side-key row, wrong baseline selection, alias shortcut, missing temperature, keyword substring, rejecting a flat monotonic step, loosened token estimate, and dropped label fan-out.
- `python3 -m py_compile reader63.py analyze63.py test_part63.py` passed.
- `git diff --check` passed.
- The shared ledger tests use an explicit ledger path under Part 63 (not `/tmp`). No paid path was exercised.

## UNVERIFIED ASSUMPTIONS

- Provider pricing, tokenization, actual billing, batch lifecycle, API error body shapes, and the inherited byte-to-token estimate remain unverified.
- The review's full process SIGKILL matrix was not rerun. The new pre-POST and raw-response recovery code is covered only by offline unit behavior; K2 and K3 were not process-killed in this fold.
- No process-kill test covers the retry POST-to-label-write interval, though the existing-retry scan prevents a second POST by construction.
- No batch-listing result was adopted. K3 requires the parent's answer in `HOLD.request`.
- Shared `paidaction` behavior is relied upon for ledger serialization, 4xx release validation, and reserve/settle transitions; no shared-library changes were in scope.

## WHAT COULD NOT BE CHECKED

- No live provider request, network lookup, model load, or paid run.
- No write to the default shared paidaction ledger; tests supply an explicit Part 63 ledger path.
- No resolution of an accepted POST whose response and batch ID were never saved; adopting a candidate from listing requires a policy decision.
- No changes to `paidaction`, Part 62, or the study design.

## BLOCKERS

K3 recovery policy needs parent direction: may a matching provider batch discovered by read-only listing be adopted, or must the unknown attempt remain non-adoptable and await explicit operator retirement before resubmission? The detailed question is in the worker HOLD request.
