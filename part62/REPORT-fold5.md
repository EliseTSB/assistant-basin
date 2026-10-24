# Part 62 fold 5 report

## SUMMARY

- Fold 5 was not implemented. No paid lifecycle code was changed.
- The inspected Part 62 reader still uses `paidaction.submit_in_chunks()` as a synchronous callback lifecycle.
- Part 63's durable submission records, response persistence, polling, collection, settlement, and explicit retirement are not yet present in Part 62.
- No API, network, model, or GPU operation was performed.
- No tests were run and no claims are made that the existing 37 tests pass in this checkout.

## FINDINGS / WHAT SHIPPED

No implementation shipped in this fold. The requested behavior is consequential to accounting and recovery, so a partially wired command that can reserve, post, or retry without durable state would be unsafe to call complete.

Inspection confirmed that `part62/blind62.py` builds a single request list, splits it into calibration and answer groups, and calls `submit_in_chunks()` for each group. Its callback posts a batch, records the batch ID through the helper callback, then polls, downloads results, and returns usage so the helper can settle. The ledger helper reserves and settles within one call. If the worker dies after the callback has persisted the ID, the next invocation does not have a Part 62 run plan and recovery path to locate that batch and finish settlement without posting the same chunk again.

The Part 63 implementation provides a useful pattern: durable per-submission metadata, raw POST-response persistence, batch links, bounded polling, and a run-all plan. Part 62's current status command only reads provider status by batch ID; there is no Part 62 collect or retire command. The shared ledger library supports retirement in its Python API, but Part 62 does not expose the required operator action or an explicit unknown-outcome refusal naming the exact recovery command.

The existing `submit()` retry support can select internal IDs, and the collection logic currently identifies failed or unlabelled calibration and answer rows for a retry. This was not verified as satisfying M3 under a sub-cap and crash recovery, because the requested failed-only retry test, settled-row exclusion proof, and lifecycle integration were not added.

The existing RUNBOOK describes sequential chunks and says ambiguous attempts remain counted, but it does not give exact commands for recovering every fail-closed state. Updating it accurately depends on implementing and testing the corresponding commands and durable states first.

No copied-code mutation checks or SIGKILL tests were added. In particular, the requested kill boundaries (after reserve, after POST intent, after accepted POST before the response is durably saved, after the response is saved before metadata update, and during settlement) remain unverified for Part 62.

## BLOCKERS

- H3 remains open: no resumable Part 62 chunk plan or recovery by saved batch ID; rerunning the current top-level submit path can start a new submission rather than resume a recorded chunk.
- K3 remains open in Part 62: there is no explicit operator-retire command with `retired-unknown` state, counted reserve semantics, and fail-closed refusal text containing the exact command.
- M3 remains unverified: no new-attempt failed-only retry row demonstrates that settled rows are not bought again and that retry reservations remain within the relevant caps.
- The requested SIGKILL matrix at the four POST windows and settlement, plus copied-code mutation checks for each, were not implemented.
- RUNBOOK-62.md was not changed because the exact commands for the new fail-closed states do not yet exist.

## UNVERIFIED ASSUMPTIONS

- No tests were executed; compatibility with the existing suite is unverified.
- No provider behavior or billing behavior was exercised.
- The Part 63 lifecycle pattern may need adaptation for Part 62's calibration gate and two-stage calibration/answer request flow; blindly copying it could permit answer submission before calibration passes.
- There were no parent-control mailbox messages at the start or after inspection; no messages changed, were declined, or deferred.
