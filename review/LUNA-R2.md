# Cold adversarial review: vastrun-r2-luna

**Verdict: DO-NOT-SHIP.** Two introduced cleanup paths can leave a rented instance undiscovered or undestroyed. One depends on the live listing omitting a field; the other is reproduced with a local record-write failure.

## Findings

### HIGH — An unrelated instance without label can strand a rent with a lost response

**INTRODUCED** — cloud/vastrun.py:133-142, with recovery at 610-623 and final sweep at 720-804.

**Measured trigger:** python3 probe_uncertain_listing.py imports the worktree module, makes rent() create an instance and then time out, and returns that instance plus an unrelated row lacking label from the list endpoint:

    uncertain_rent_with_unlabelled_row: outcome=DestroyConfirmationError, target_present=True, destroy_calls=0

_instance_payloads rejects the entire listing because any row lacks label. Recovery therefore cannot find the created instance by run label, and final cleanup also refuses the list without issuing DELETE. The harness did not call Vast; whether its live v1 response can omit label remains unverified. A row with "label": null is accepted: probe_review.py measured the missing-label case as unconfirmed, while the null-label case returned successfully after one delete.

**Smallest fix:** allow rows with an id but no label in list-level existence checks; treat a missing label on an unrelated row as a nonmatch while continuing to match only the exact run label.

### HIGH — A run-record write failure after rent prevents cleanup

**INTRODUCED** — cloud/vastrun.py:502-524 and 790.

**Measured trigger:** python3 probe_rent_write_failure.py injects a disk-write error immediately after the fake API creates the instance and before rent() returns:

    post_rent_record_write_failure: outcome=OSError: simulated disk full, destroy_calls=0, instance_still_present=True

The write at line 511 fails before _activate; the exception handler tries another record write before uncertain-rent recovery, and final label sweep writes its discovered instance before calling destroy. Both writes fail, so no DELETE is attempted. This contradicts the cleanup guarantee when run-record persistence fails after a successful rent.

**Smallest fix:** make recovery/destruction proceed from the in-memory rent id or run label even when record persistence fails. Record writes in cleanup should be best-effort and must not gate DELETE attempts.

### MEDIUM — Later job log can overwrite an earlier job result

**PRE-EXISTING follow-up** — cloud/vastrun.py:380-413 and 881-905.

**Measured trigger:** python3 probe_result_collision.py runs two jobs through the real runner. The first declares logs/consumer.log as a result; the second job is named consumer, so its generated log maps to the same local destination:

    accepted_jobs=2, producer_result_after_consumer='consumer log overwrote producer'

The new duplicate-result validation reserves declared result paths against other declared results, but not against generated log destinations. The later job's log replaces the earlier result after it was copied back.

**Smallest fix:** reserve generated log destinations during validation, or place each job's copied outputs and log under a job-specific local directory.

### MEDIUM — The fallback watchdog powers off but does not destroy the rental

**PRE-EXISTING follow-up** — cloud/vastrun.py:191-194.

**Measured trigger:** python3 probe_watchdog.py calls the real watchdog builder:

    nohup sh -c 'sleep 2400; poweroff' >/dev/null 2>&1 &

The fallback command has no Vast DELETE operation. If Python cannot run its finally cleanup (for example, SIGKILL or host failure), a successful guest poweroff can leave a stopped instance in the account. Vast's [FAQ billing section](https://console.vast.ai/faq/) says stopped instances still incur storage charges until destroyed.

**Smallest fix:** add an out-of-band expiry/destruction mechanism that does not put the account API key on the rented box, or state and handle this fallback's residual storage billing explicitly.

## What I checked and found sound

- Run-label sweep filters candidates with exact value.get("label") == self.run_label; a different readable label is not selected for destruction (cloud/vastrun.py:740-751).
- A present null label does not make a listing unreadable; the measured parser distinction is between a missing key and a key whose value is null.
- For a readable list, _destroy_tracked calls GET /api/v1/instances/ via list_instances() and requires both a successful {"success": true} DELETE response and absence of the id before marking that instance confirmed (cloud/vastrun.py:650-675). I did not query the live service or verify its listing schema.
- The probes above import the real worktree module and use fake API/SSH boundaries. I did not run the repository test suite.
