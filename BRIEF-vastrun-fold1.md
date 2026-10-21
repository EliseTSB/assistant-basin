# Fold brief: vastrun, review round 1 (Luna)

Fix every finding in review/HAIKU-R1.md (1-9). Facts from the live API this morning (2026-10-21, read by Claude):
GET https://console.vast.ai/api/v1/instances/ returns a JSON object {"instances": [...]}; each instance carries
"id", "actual_status", "ssh_host", "ssh_port", "label", "dph_total", "start_date". DELETE v0 /instances/<id>/ returned
{"success": true}. Pin those shapes; anything else is UNREADABLE, never "gone".
Required:
1. Unreadable list = not confirmed (raise/retry), in destroy and in rent recovery alike.
2. SIGHUP handled like SIGINT/SIGTERM. The instance id is recorded inside the rent try, and run-record.json is
   written (atomically) the moment an instance id is known and after every state change, not only at the end.
   In the final cleanup, list instances and destroy EVERY instance whose label is this run's run_label (label sweep),
   then confirm none remain.
3. The spend clock keeps running for an unconfirmed instance; spend uses the instance's own dph_total when the
   instance listing provides it, else the offer's.
4. Up to 3 consecutive poll failures (ssh timeout or error) are tolerated before raising; results of finished jobs
   are still copied before cleanup.
5. Reject duplicate expected result paths and duplicate safe job stems in validation.
6. Accept ssh port from ssh_port, ports["22/tcp"] and ports["22"] (Vast's documented forms), tested.
7. The key test must exercise the real VastAPI with a fake key file and a patched urlopen (no network).
8. A second interrupt during destroy backoff must not cut the retries.
Tests for each; a sabotage per HIGH (unreadable-as-gone, no SIGHUP handler, no label sweep) must fail a test.
Commit nothing. Report REPORT-vastrun-fold1.md ending SUMMARY / FINDINGS / BLOCKERS.
