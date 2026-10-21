# Cold adversarial review: vastrun-r3-luna

**Verdict: DO-NOT-SHIP**

## Findings

### P1 · INTRODUCED — uncertain rent can be forgotten, then success returned with a box still listed

Files: `cloud/vastrun.py:843-847`, `929-949`, `1083-1105`.

**Measured trigger:** `python3 /home/elisepc/green/Journal/scratch/review-vastrun-r3-luna-luna-r3-2026-10-21/probe_uncertain_rent.py` drives the real module with an API double. In case 2, the first rent creates a box but raises a lost-reply timeout; its listed row has `label: null`. Recovery sees no exact run-label match and returns normally, so the runner rents the second offer. Final label sweep sees no matching row, clears its tracked set, and calls cleanup confirmed.

```text
CASE 2: first box is listed without its label after uncertain rent
rent calls: ['first', 'second']
second rent while first still exists: True
run returned: success
destroy_confirmed: True
instances remaining: ['instance-first']
```

Case 1 in the same probe hides the first box during the three recovery reads, then reveals it. The second offer is rented while the first exists; final cleanup eventually catches both. The second case proves the stronger outcome: success with the first box still listed. The run undercounts spend while that box is untracked. This violates the must-clean-up invariant and the one-rental-per-job claim.

**Smallest fix:** Treat a lost rent reply as unresolved unless the code can correlate the resulting instance. Do not try another offer or report cleanup confirmed after an empty label match. Reliable resolution needs an idempotency/correlation handle or another safe way to associate the created instance; a label-less row cannot safely be destroyed by ID when the ID was lost.

### P1 · INTRODUCED — a stop signal in cleanup setup bypasses all cleanup

Files: `cloud/vastrun.py:562-569`, `1368-1381`, `1476-1482`.

**Measured trigger:** `probe_signal_cleanup_entry.py` drives `_Runner.run()` with fake Vast/SSH adapters, injects a per-job destroy error, then raises the same `RunInterrupted` used by the CLI signal handler from the `setitimer(0)` call in `_begin_cleanup`. That call is before the guarded destroy/sweep blocks in `run()`'s `finally`.

```text
raised: RunInterrupted second stop signal
signal injected in _begin_cleanup: True
instances remaining: ['box']
```

A second SIGINT/SIGTERM/SIGHUP at that boundary (or another exception from the unguarded setup) exits the `finally` before `_destroy_active`, `_label_sweep`, and the final record write. The existing retry handling applies only after cleanup has begun.

**Smallest fix:** Make cleanup setup interruption-safe: defer stop signals while entering cleanup, or put the whole cleanup transition—including `_begin_cleanup`—under a `BaseException`-safe guard that still runs label sweep.

### P2 · INTRODUCED — expiry sweep has no ownership boundary for matching labels

Files: `cloud/vastrun.py:222-229`, `650-669`, `1434-1441`.

**Measured trigger:** `probe_foreign_expiry_label.py` passes `sweep_expired` a box owned by a simulated other tool, labeled `vastrun-abcdef123456-<past-epoch-hex>`. The label has no ownership record; matching the prefix, 12 hex characters, and epoch suffix is enough for deletion.

```text
sweep outcomes: [('vastrun-abcdef123456-6ad911d2', 'deleted, listing lagged')]
destroyed: ['foreign-box']
foreign instance remains: False
```

The same broad prefix also makes preflight refuse unknown `vastrun-*` labels as active. A different tool using the recognizable label shape can therefore be either blocked by preflight or destroyed by the expiry command.

**Smallest fix:** Use a dedicated, versioned namespace reserved to this runner and sweep only that namespace; keep unknown-prefix labels outside automatic destruction.

### P2 · INTRODUCED — expiry can be declared early by a skewed sweeper clock

Files: `cloud/vastrun.py:480-482`, `653-669`, `1433-1441`.

**Measured trigger:** `probe_clock_skew.py` encodes the 30-minute run plus 10-minute watchdog and 25-minute grace used by this code. A sweeper whose wall clock is 66 minutes ahead sees the just-created run's deadline as passed and deletes the still-running instance.

```text
actual time until encoded expiry seconds: 3900
sweeper clock skew seconds: 3960
sweep outcomes: [('vastrun-abcdef123456-6ad921bf', 'deleted, listing lagged')]
destroyed while expiry was still future on owner clock: ['still-running']
```

The large grace reduces ordinary risk, but the expiry check is destructive and compares two processes' local `time.time()` values without validating clock agreement. This is most relevant when `sweep-expired` runs on a different, unsynchronized host.

**Smallest fix:** Base the expiry decision on an authoritative shared clock or a server-side creation/expiry value; otherwise require a documented maximum skew and a grace that exceeds it.

### P2 · INTRODUCED — standalone label sweep can exceed its cleanup deadline

Files: `cloud/vastrun.py:269-307`, `1417-1442`, `1459-1468`.

**Measured trigger:** `probe_sweep_deadline.py` calls the real `sweep_label` with a 40 ms deadline and a two-page listing that takes 30 ms per page.

```text
configured deadline seconds: 0.04
page calls: [None, 'page-2']
elapsed seconds: 0.060
returned outcome: already absent
```

The helper checks its deadline before a sweep wave, but `_listed_instances` can fetch up to 100 pages without receiving remaining time, and the global helper does not shorten `VastAPI.timeout`. The standalone `sweep LABEL` command uses this path without the runner's signal timer. A slow listing can therefore overrun cleanup and still return success.

**Smallest fix:** Pass the remaining deadline into each page request and abort when it expires; apply the same bound to the initial list in `sweep_expired`.

## What I checked and found sound

- `_listed_instances` is the shared pagination loop, follows `next_token`, rejects repeated tokens, and bounds pagination to 100 pages. Runner cleanup, recovery, preflight, and module sweep callers route through it.
- Known-ID cleanup checks the full paginated listing by ID after DELETE; unrelated rows with absent or null labels no longer make the entire list unreadable. A malformed row whose label matches the run remains a hard error.
- The normal rent-reply path puts the ID into in-memory tracking before best-effort record persistence. Offer and positive listed rates combine using `max`, and the label sweep matches by exact label.
- API credentials are used only in the Authorization header in this module. SSH errors omit command/output text; process launches use argument lists, and the embedded SSH shell values are quoted.
- `_label_deadline` uses a full-string format match. Malformed suffixes are not automatically swept; preflight treats them as unknown and refuses to rent. No feature-flag OFF branch exists in this module/design.
- I did not run the project test suite; each finding above has a focused script that imports and drives the real code with local fake adapters. No live service or credentials were accessed.
