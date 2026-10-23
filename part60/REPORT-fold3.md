# Part 60 round-3 fold report

## SUMMARY

- Continued the half-edited Part 60 tree in place; did not reset or revert it.
- Completed the Amendment 5 ambiguous-submit redesign and folded the remaining Luna and Haiku findings into the runner, scorer, analyzer, wrapper, documentation, and tests.
- The required offline suite passes: 68 tests. Python compilation, `bash -n run60.sh`, and `git diff --check` pass.
- No method, prediction, or metric definition was changed. No commit, API request, network operation, Qwen load, or GPU run was made.

## FINDINGS / WHAT SHIPPED

### Items already in the tree when this worker started

The checkout was modified and uncommitted. It already contained the core Amendment 5 model: ambiguous attempts retained their reservations; later submits made a new attempt; batch listing was read-only; reconciliation and adoption code were removed; 429s were ambiguous; and tests covered cap refusal, kill after POST, and an adoption mutation. It also already had separate direct and batch price constants, M6 failed-check history and fresh-unit rotation, per-request truncation/failure accounting with failed-unit-only resubmission, the frozen threshold/device wording, point-lead stop-rule text, report counts, wrapper and scorer locks, writer locks, progress-before-memmap ordering, `PHASES` output, and the M0 legacy cache branch using `args.split`.

The continuation note's prior suite result was 63 tests with 6 failures and 3 errors. I reproduced that result at start: most remaining failures were stale assertions for old smoke/truncation behavior; there was also an uncovered progress-marker recovery gap, a dropped-phase collect idempotence gap, a report-fixture compatibility gap, and an M3 fixture missing the scored selection artifact.

### Round-3 review findings

| Finding | Disposition | Result |
|---|---|---|
| Luna 1 — clock skew can strand an accepted batch | Replaced by redesign | No matching or adoption by timestamps. Ambiguous attempts stay reserved; a retry creates a new capped attempt. |
| Luna 2 — retrying a 429 can duplicate a batch | Replaced by redesign | No same-body retry loop; 429 is ambiguous and receives a new reservation on resubmit. |
| Luna 3 — direct smoke calls use batch rates | Fixed | Direct smoke reserve and settle use the direct constants, twice the batch rates. A cost regression checks the settled amount. |
| Luna 4 — retryable smoke errors leave reserves open | Fixed | Every error releases when no usage was returned, or settles partial completed-response usage. Tests cover 429, 500, timeout, and partial usage. |
| Haiku 1 — foreign batch adoption | Replaced by redesign | No batch adoption path remains; listing only returns operator-visible records and changes no state. |
| Haiku 2 — zero-match reconciliation leaves a terminal lock | Replaced by redesign | Reconciliation and the locked-zero-match state are removed. A subsequent submit reserves a new attempt, subject to the cap. |
| Haiku 3 — wrapper kill can overlap scorers | Fixed | `run60.sh` holds a flock inherited by the scoring child. A kill/rerun test confirms the second wrapper refuses while the scorer remains alive. |
| Haiku 4 — orphan progress can block scoring | Fixed | Progress is durable before memmap creation. Rerun recreates arrays if either is missing; `--force` clears score artifacts. A SIGKILL test covers the marker-before-array window. |
| Haiku 5 — threshold device differs from scoring device | Wrong as a defect under the registration | Part 58 archive thresholds remain the device-of-record. `DEFINITIONS-60.md` states this and records that the live M0 rescore reports the cross-device effect. |
| Haiku 6 — one truncation stops a phase and retries all units | Fixed | Truncations count as per-request failures. A 1/500 truncation completes below the 10% drop threshold and submits only that failed unit on retry; over 10% drops the phase. |
| Haiku 7 — text still requires CI lower bounds | Fixed | Runbook, analyzer marker, and tests use the registered point-lead rule; intervals are reported but are not a gate. |
| Haiku 8 — one missing thinking response drops M6 | Fixed | One smoke command makes up to three fresh five-unit checks. It drops M6 only after three consecutive failed all-five checks; a pass or API error stops the sequence. |
| Haiku 9 — M0 legacy score cache uses undefined `split` | Fixed; branch was already corrected at start | Added a stale-M0-cache regression that reaches the scorer without a `NameError`. |
| Haiku 10 — `RETURN_READY` can mean dev-only | Fixed | The wrapper prints `PHASES: dev-only` or `PHASES: dev-and-test` immediately before `RETURN_READY`; the runbook explains the two-pass flow. |
| Haiku 11 — smoke error paths leave spend unsettled | Fixed | No-usage errors release; partial usage settles at the direct rate. Both paths have tests. |
| Haiku 12 — regression tests missed the changed failure modes | Fixed | Added coverage for truncation below 10%, orphan progress recovery, wrapper kill/rerun, direct pricing, smoke retry/accounting, M0 cache cleanup, and the adoption mutation. Foreign-batch matching is no longer applicable because adoption was removed. |
| Haiku 13 — means hide varying valid counts | Fixed | Each reported mean includes its valid count; the report path tolerates older/synthetic metric fixtures too. |
| Haiku 14 — unjudgeable wording overstates the result | Fixed in editable code/docs | Reports and definitions use “empirically unjudgeable (threshold at or above the pool maximum).” The registered design file retains its original wording because the brief marks it read-only. |
| Haiku 15 — generator writers bypass a shared lock | Fixed | Smoke, submit, collect, M3 selection, and M0 preparation take the shared writer lock. A regression test checks the public mutating entry points. |

The only remaining `created_at` use for batch data is the read-only listing filter that selects records for display since the run began. It is not used to identify a submission, authorize collection, or modify a manifest. The listing test checks both filtering and absence of filesystem writes.

### External assumptions

Removed from the correctness path: identifying a submitted batch by `created_at` plus request-count totals; comparing local and API clocks to adopt a batch; and assuming HTTP 429 guarantees that no batch was accepted. The runner does not adopt any batch, regardless of list contents.

Still assumed and not live-verified: the documented batch create/status/result shapes; list pagination fields (`has_more`/`after_id`) and parseable `created_at` for the informational listing; acceptance and output behavior for Haiku 5.5 adaptive thinking/high effort; and the inherited rates ($0.05/$0.25 batch, $0.10/$0.50 direct), which remain explicitly unverified. The 190 MB payload limit is a local cap whose service margin was not checked. The API/model/device checks were prohibited by the brief, so these remain assumptions rather than findings from live behavior.

### Tests and mutations

- `PART60_BASIN=/home/elisepc/assistant-basin /home/elisepc/neuron-probe/.venv/bin/python -m unittest test_part60` — 68 tests passed.
- `/home/elisepc/neuron-probe/.venv/bin/python -m py_compile gen60.py score60.py analyze60.py test_part60.py` — passed.
- `bash -n run60.sh` — passed.
- `git diff --check` — passed.
- `test_readding_ambiguous_batch_adoption_mutation_goes_red` passes by proving the unchanged contract succeeds and a copied implementation that returns a legacy adoption fails it.

### What could not be checked

No Anthropic request, API listing, Qwen model load, GPU run, or live pricing check was made. Therefore current service pricing, adaptive-thinking acceptance/output, batch cursor behavior, actual orphan-batch cost, and M0 cross-device numerical effect remain unverified.

## FILES TOUCHED

`DEFINITIONS-60.md`, `RUNBOOK-60.md`, `analyze60.py`, `gen60.py`, `run60.sh`, `score60.py`, `test_part60.py`, and this `REPORT-fold3.md`. `PART60-DESIGN.md` was not edited.

## BLOCKERS

None requiring a decision. The live checks above were excluded by the brief.

## SELF-ASSESSMENT

The fake-API and toy-scorer tests exercise the specified state transitions, but they cannot establish live Anthropic behavior, current rates, or GPU numerics. The immutable design file retains the old Amendment 3 phrase noted under Haiku 14; the editable definitions and generated reports use the requested wording.

## Parent-control mailbox

Read at start and at each required checkpoint, including before this report. It remained empty; no control message changed, was declined, or was deferred.
