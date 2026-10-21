# Vastrun fold 1 report

SUMMARY
Fixed review findings 1–9 in `cloud/vastrun.py` and added focused regression coverage in `cloud/test_vastrun.py`. The required offline suite passes: 23 tests passed. No commit was created.

FINDINGS
- Instance-list parsing now accepts the specified `{"instances": [...]}` shape only, requires each listed item to include an id and label, and preserves unreadable as a distinct result. Destruction requires both a successful `{"success": true}` delete response and a readable list without the instance. Uncertain-rent recovery also retries unreadable listings and does not treat them as empty.
- Every rent uses the run label. SIGHUP is handled alongside SIGINT and SIGTERM. Rent ids are persisted as soon as returned; run records are atomically refreshed at lifecycle and job state changes. Final cleanup lists by run label, destroys all matches, and verifies that none remain.
- Billing tracks each known instance until confirmed gone, reads the listed instance `dph_total` when available, and falls back to its offer rate. Polling tolerates three consecutive SSH failures and raises on the fourth; completed jobs copy results before cleanup. Validation rejects duplicate result destinations and sanitized job stems. SSH connection lookup accepts `ssh_port`, `ports["22/tcp"]`, and `ports["22"]` forms.
- The credential regression test instantiates the real `VastAPI` with a temporary fake key file and a patched `urlopen`, so it makes no network call and does not read `~/.vast-credits.env`. Destroy backoff retries continue after an additional interrupt.
- The test suite includes sabotage-sensitive checks for unreadable lists, missing SIGHUP handling, and missing label-sweep cleanup.

BLOCKERS
None. Live Vast behavior was not queried; implementation follows the response shapes supplied in the brief.
