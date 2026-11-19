# Part 212 — fix round 2 (Journal Claude, 2026-11-19)

Read part212/REVIEW-code-luna-r2.md (the round-2 cold review, SHIP-WITH-FIXES), PART212-DESIGN.md (v3.1 plus
Amendment 1 at the end), part212/run212.py, part212/score212.py, part212/test_part212.py, part212/smoke_tiny212.py.

Fix these, each with a test that FAILS on the current code (show it red, then green):

1. (MEDIUM) Batch shape per answer. Write `batch_size` (the size of the generation call that produced it) into every
   Stage 2 row. On resume: refuse to continue when existing rows of this pilot flag carry a different batch size than
   the requested one (say so and exit non-zero); and do not regroup a partially written batch into a new shape —
   either finish it at its original group split or refuse. `run.json`'s Stage 2 histogram must be aggregated from ALL
   saved rows, not only this invocation's. score212.py: report the per-row batch-size histogram it scored, and treat
   rows with no batch_size (written before this fix) as an error, not silently kept.
2. (MEDIUM) Gate 3 `every_planned_key_present`: compare the set of unique row keys to the exact planned key set
   (reconstruct it from the runner's own plan function, imported, so there is one source of truth), keeping duplicate
   detection. A missing planned key, or an unplanned key, fails it.
3. (LOW) F3 driver test: one test that runs the Stage 1 driver (not the helpers) with target and pool units given
   distinct recorded values, and observes the values actually written for a random-set arm are the pool units' own.
   It must fail on 84472af3's driver.
4. (LOW) An output-directory lock for Stage 1 and Stage 2 (a lock file opened with O_CREAT|O_EXCL holding the pid,
   removed on exit; a stale lock whose pid is not alive may be taken over with a message). Windows matters: the real
   run is on the A770 under Windows Python, so no fcntl and no POSIX-only import (the launcher refuses those).
5. (LOW) Resume over a partial final JSONL line: before appending, truncate the file to the last complete line and
   say how many bytes were dropped; score212's loader reports and rejects corrupt rows explicitly instead of raising
   a bare JSONDecodeError.

Do not touch PART212-DESIGN.md (I amend it). Never load real weights. Use a scratch folder in your own run directory,
never /tmp. When done: all tests pass, and the tiny smoke (`part212/smoke_tiny212.py <repo> <scratch> pilot`)
prints SMOKE OK with its assertions passing; extend the smoke to assert every Stage 2 row carries batch_size.
Append your report to part212/BUILD-REPORT.md under '## Fixes from the round-2 review (2026-11-19)'. Do not commit.
