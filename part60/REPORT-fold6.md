# Part 60 round 6 report

## SUMMARY

Completed the offline Part 60 fold against Amendments 5 and 6. Closed the paid-batch cap and durability gaps, corrected stale analysis carry-forward and operator guidance, and documented the remaining reporting limits. The required suite passed: 90 tests. Parent-control mailbox was empty at each checkpoint; no messages changed or deferred the plan.

## FINDINGS / WHAT SHIPPED

| Review finding | Resolution |
|---|---|
| 1. Force-retire could release a live batch reserve and allow cap overshoot | Removed `--force-lost`. A batch ID may be retired only after an API terminal status; its reserve remains counted until collection. No-ID attempts require age greater than 25 hours. Collection checks actual usage against the cap while holding the ledger lock and leaves the reserve intact on refusal. Added retire-then-collect/headroom coverage. |
| 2. Cross-root duplicate check was outside the ledger lock | Moved duplicate phase/method/split checking beside cap validation and reserve append under the shared ledger flock. Increased race-test cap so either reserve alone or both would fit; test requires exactly one reserve and POST. |
| 3. Kill after POST could cause a duplicate POST | Fsync the returned batch ID to its own file and directory before writing the submitted manifest. A rerun recovers that ID and instructs collection rather than submitting again. Updated the kill test and mutation test to assert no second POST. |
| 4. Prices were labeled unverified | Updated rate comments and cost guidance to match the dated Haiku 5.5 list-price attestation. Collection output now compares reported usage cost to the ledger amount. |
| 5. Unreadable manifest could be treated as absence | Retirement fails closed on unreadable manifests; added coverage. |
| 6. Concurrency test could pass without checking the duplicate guard | Race test now gives sufficient headroom for both calls; the lock-protected duplicate guard must provide the refusal. |
| 7. Dev-only rerun retained stale test analysis | `write_reports` now writes only the split(s) in the current invocation; added a dev-only-after-dev-and-test regression check. |
| 8. M0 baseline count was misstated | Definitions and runbook now state 2,000 expected dev texts, 1,988 present for 497 units, three missing dev controls counted failed, and one missing test control counted failed. |
| 9. M3 collect used mutable current winner state | Fresh-request hypothesis is stored in the request idmap and collect uses that stored value. |
| 10. Listing documentation disagreed with behavior | CLI help and docs now say listing returns all batches from the API, not only batches from this run. |
| 11. Killed wrapper left scorer lock and reruns truncated output | Wrapper records active phase PID without passing its run-lock descriptor to children. A rerun validates the stale command before terminating a Part 60 scorer. Runbook launch now appends to `run60.out`; updated wrapper kill/rerun regression passes. |
| 12. Incomplete batch usage conservatively retains full reserve | Documented that incomplete usage can block headroom. Cost report reserve total now includes retained terminal-batch reserves and excludes released/settled spend. |
| 13. Smoke kill can leave ledger spend without raw response | Documented that analysis usage may be below ledger spend for this unknown outcome and that the ledger is the cost record. |

Other requested details: PRICE constants carry the dated attestation comment; M3 request metadata preserves the fresh hypothesis; retirement never zeroes a potentially billable batch reserve. No design or prediction files were edited. The six mutation-copy tests in the suite passed; each verifies its intentionally broken copy fails while the baseline contract passes.

### Tests and checks

`PART60_BASIN=/home/elisepc/assistant-basin /home/elisepc/neuron-probe/.venv/bin/python -m unittest test_part60` — **90 tests passed**. This includes the cap-headroom race, POST kill recovery, retire-then-collect cap refusal, unreadable-manifest refusal, stale-analysis regression, and wrapper kill/rerun case. `bash -n run60.sh`, Python byte compilation for `gen60.py`, `analyze60.py`, and `test_part60.py`, and `git diff --check` also passed. Tests used fake batch APIs and local fixtures; no network, live API, or Qwen load was run.

### Remaining UNVERIFIED probe checklist

- Verify the live batch listing cursor direction (`after_id`) before treating pagination as complete; listing remains report-only.
- Compare the first real batch's reported usage with the ledger and the account invoice. The published list-price attestation was read locally, but no live usage or invoice was available under this offline scope.
- Live API terminal-status behavior, paid-call semantics, and real GPU/Qwen scoring remain unverified, as required by the brief's offline restriction.

## FILES TOUCHED

- `/home/elisepc/assistant-basin/part60/gen60.py`
- `/home/elisepc/assistant-basin/part60/analyze60.py`
- `/home/elisepc/assistant-basin/part60/run60.sh`
- `/home/elisepc/assistant-basin/part60/test_part60.py`
- `/home/elisepc/assistant-basin/part60/DEFINITIONS-60.md`
- `/home/elisepc/assistant-basin/part60/RUNBOOK-60.md`
- This report: `/home/elisepc/green/DiscordBot/df-runs/part60-fold6-2026-10-23/worker-01/report.md`

## BLOCKERS

none

## SELF-ASSESSMENT

The offline suite exercises the new paths, but it cannot establish live API status values, list pagination semantics, invoice accuracy, or GPU scorer behavior. The wrapper cleanup test uses a fake scorer and validates the intended stale-process recovery path rather than a rented-machine run. No parent-control messages required disposition.
