# Part 212 — fix round 3 (Journal Claude, 2026-11-19): SHRINK, do not harden

Read part212/REVIEW-code-luna-r3.md (DO-NOT-SHIP: three MEDIUM, one LOW, all introduced by round 2's resume and lock
code). The rule for this round is to remove machinery where possible. Implement exactly these designs, each with a test
that FAILS on the current code (cd314286) and passes after:

1. Partial generation call on resume (MEDIUM 1). Keep the refusal-free path simple: before resuming, rewrite
   stage2.jsonl WITHOUT the rows of any partially persisted call (write a temp file in the same directory, flush,
   fsync, os.replace), print how many rows were dropped and from which calls, and let those calls be regenerated
   whole at their original shape. Also write each call's rows with a single write() of all lines, then flush and
   fsync. Test: 3 of 8 rows of the first call saved -> resume drops those 3, plans the full 8-row call again, and the
   finished file has every planned key once with batch_size 8 for that call.
2. Tail repair (MEDIUM 2). If the bytes after the last newline parse as one complete JSON object, KEEP them and
   append the newline; truncate only a tail that does not parse. Tests for both cases.
3. Lock (MEDIUM 3): REMOVE stale takeover entirely. If the lock file exists, refuse: print the lock path, the
   recorded pid, whether that pid looks alive (pid_is_alive), and that the operator should delete the file if the run
   is not live. Never unlink a lock the current process did not create. Tests: refuses for a live pid and for a dead
   pid; the lock file is untouched after refusal.
4. Lock write failure (LOW): if writing the owner record fails, close the descriptor, unlink the file this attempt
   created, then re-raise. Test with a forced os.write failure.

Do not touch PART212-DESIGN.md. Never load real weights. Scratch in your own run directory, never /tmp. Done when all
tests pass and part212/smoke_tiny212.py prints SMOKE OK pilot. Append your report to part212/BUILD-REPORT.md under
'## Fixes from the round-3 review (2026-11-19)'. Do not commit.
