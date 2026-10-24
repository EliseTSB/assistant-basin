# Brief: Part 65 judge fold after review round 5 (code and tests only)

Domain: /home/elisepc/green/worktrees/ab-part65/part65 (write only there). Review:
`/home/elisepc/green/Journal/reviews/cheap-p65-build-haiku-r5.md` (its scripts are under
`/home/elisepc/green/Journal/scratch/review-p65-build-haiku-r5-2026-10-24/`, readable). Contract: Amendment 4 at the end of
`PART65-DESIGN.md`. Do not commit, run on the GPU, call a paid API, or change `steer65.py` or `paidaction/paid_ledger.py`.
Python: `/home/elisepc/neuron-probe/.venv/bin/python`, suite from `part65/` with
`PYTHONPATH=/home/elisepc/green/worktrees/ab-part65`. The existing 51 tests stay green.

Fix exactly these, each with a test written from the property, and run each named mutant to show its test fails:

- **F1.** (a) The judge's seed becomes a module constant (the registered value the CLI defaults to today); remove the
  `--seed` command-line option, as `--ledger` was removed. Tests pass a seed through function arguments only. (b) A
  judge-wide terminal latch: before any reservation, refuse if the ledger holds any `part65-judge` reservation that is
  released, retired, overrun or still active from another run, whatever its plan hash. Test: release chunk 0 of plan A,
  then try plan B (different seed through the function argument): zero sends for B. Mutant: latch filtered by plan hash.
- **F2.** Strengthen the release test: three chunks, a validated 400 on chunk 0; assert exactly one POST and chunks 1
  and 2 still `ready`. Mutant: remove the break on `released` (`("ambiguous",)` only); the test must fail.
- **F3.** Add the crash-after-accept case: `_save_chunk` raises when saving `submitted` after the provider accepted the
  batch; a second invocation must not POST again (one accepted POST across both). Mutant: remove the `posting` save
  before the POST; the test must fail.
- **F4.** Stop the loop on `pending` as well as `released` and `ambiguous`, so `--recover` resumes it (RUNBOOK already
  says so). Test: chunk 0 never ends within the timeout; exactly one POST, chunk 1 still `ready`.

Leave F5 to F8; F7 (the probe request is part of the full plan) gets one sentence in `RUNBOOK-65.md`.

Report: `part65/REPORT-fold-r5.md` (SUMMARY / FINDINGS one line per fix with test and mutant result / BLOCKERS).
