# Build brief: cloud/vastrun.py, an unattended rented-GPU runner (Luna)

Run size: code + CPU tests only. NO real Vast or ssh calls anywhere in tests or while building (no key reads in
tests). Start from cloud/vast-part48-2026-10-21/vast.py (the REST shapes that work: v0 bundles/asks/instances,
v1 GET /instances/; bare key in ~/.vast-credits.env, never printed).

Why: this morning four of five hosts failed before doing work, and a babysitting agent died at a session boundary,
leaving a 3090 billing idle. The runner must need nobody watching and must never leave an instance up.

`python cloud/vastrun.py JOBFILE.json` where JOBFILE lists: upload files, a setup command, an ordered list of jobs
(name, command, env, expected result paths), a local results dir, max_dph, spend_cap_usd, wall_clock_min.
1. Offers: the existing filter (verified, reliability2 >= 0.98, cuda >= 12.4, disk >= 40) plus inet_down >= 200,
   price <= max_dph, cheapest first.
2. Rent; wait until ssh answers a trivial command; if not ready within 180 s, destroy it and try the next offer
   (at most 3 hosts). Record every host tried and why it was dropped.
3. Upload, run setup, then each job via nohup on the box writing `rc=<n>` to its own log; poll every 30 s; as each job
   ends, copy its result paths and log back immediately (results survive a later failure).
4. ALWAYS destroy: in a finally block, on SIGINT/SIGTERM, on any exception, on spend cap (elapsed hours x dph >= cap)
   and on wall clock. Then confirm via GET v1 /instances/ that the instance is gone; retry destroy up to 3 times and
   exit non-zero loudly if it is still listed.
5. Write run-record.json (hosts tried, instance id, dph, start/end, estimated spend, per-job rc and seconds, destroy
   confirmed) into the results dir.
6. A dead runner must not leave a billing box: also pass the box an onstart watchdog that shuts the instance down
   itself after wall_clock_min + 10 (e.g. `sleep N; poweroff` or Vast's own self-destroy via its key is NOT allowed —
   no key on the box; use poweroff/stop only), so a killed runner costs at most that.
Tests (pytest, fake API object + fake ssh runner injected): ready-timeout moves to the next host and destroys the
first; destroy on exception mid-job; destroy on spend cap; per-job copy-back happens before a later job fails;
destroy-confirmation failure exits non-zero; the key is never written to stdout/logs. A sabotage removing the finally
destroy must fail a test. Commit nothing. Report REPORT-vastrun.md ending SUMMARY / FINDINGS / BLOCKERS.
