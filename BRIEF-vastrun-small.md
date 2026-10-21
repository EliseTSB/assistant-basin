# Build brief: vastrun, the smaller design (Luna)
Rewrite cloud/vastrun.py to VASTRUN-SMALL-DESIGN.md (whole file, including the "Review and live probe" section, which
wins on any conflict). One job per rental. Use the live listing shape in that section (paginated via next_token).
Keep the CLI shape (python cloud/vastrun.py JOBFILE.json) and add "sweep-expired" and "sweep LABEL".
Tests (fake API + fake ssh + fake clock; no network, no key read): label-less and null-label neighbours; a malformed
item matching our label; lost rent response; record-write failure; SIGINT/SIGTERM/SIGHUP before rent returns, after,
during the job and during cleanup; a hung list/ssh call stopped by the wall-clock timer; unreadable listing then
readable; pagination (ours on page 2); DELETE then 404; a lagging listing; a zero listed rate; expired-label sweep
destroys, non-expired refuses; unreadable pre-rent listing refuses to rent; the key is never printed (real VastAPI,
fake key file, patched urlopen). Sabotages that must each red a test: cleanup outside finally; strict all-or-nothing
parse; listing lowers the rate; first page only; the wall-clock timer removed.
Commit nothing. Report REPORT-vastrun-small.md ending SUMMARY / FINDINGS / BLOCKERS.
