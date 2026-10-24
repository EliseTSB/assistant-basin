# Brief: implement Part 65 Amendment 4 (the judge's terminal-release policy), code, tests and runbook only

Domain: /home/elisepc/green/worktrees/ab-part65/part65 (write only there). Read `PART65-DESIGN.md` to its end; Amendment 4
replaces rule 6 of Amendment 2 and is the contract. The specification is Chorus Task's review,
`/home/elisepc/green/Journal/reviews/chorus-p65-judge-2026-10-24.md`: implement its section 2 items 1 to 6, its fsync
blocker, its section 3 blocks (a) to (c), and its five fake-API regression tests. Background:
`/home/elisepc/green/Journal/reviews/cheap-p65-build-haiku-r3.md` (F1, F2, F3, F7).

Do not commit, run on the GPU, download models or call any paid API. Do not change `steer65.py` (the steering run is in
progress from a staged copy). Do not change `paidaction/paid_ledger.py`; if the policy cannot be built without a ledger
change, stop and say so in BLOCKERS.

Python: `/home/elisepc/neuron-probe/.venv/bin/python`; run the suite from `part65/` with
`PYTHONPATH=/home/elisepc/green/worktrees/ab-part65`.

## What must hold (from Amendment 4)

1. A validated release returns `released`, is counted, and stops the loop: no later chunk is sent. Ambiguous outcomes
   also stop the loop. `main` prints the terminal message and exits 2. `--recover` sends nothing.
2. One run identity (random run id + resolved output directory) in `run.json`, every chunk, every reservation's
   metadata and every answer row, preserved on reopen and `--recover`. Before every send, under one lock keyed to the
   ledger path held from the plan lookup through the send: refuse if any reservation for the same plan belongs to
   another run; refuse a chunk with any released, settled or retired attempt; adopt an active reservation only for the
   same run. Check both the local `ready` and the local `reserved` branches.
3. `_atomic_json` syncs the temp file before the replace and the directory after it.
4. `analyze65` refuses rows carrying more than one (run id, plan hash) pair, and rows without them, even when the
   (unit, trial) keys are disjoint. An empty answers file still reaches the registered missing-trial stop.
5. Add `--reserve-total` (or equivalent) that prints the whole plan's worst-case reserve from the real requests without
   sending anything, so it can be recorded before the probe. Remove "under three cents" claims from the runbook.
6. `RUNBOOK-65.md`: a released chunk is terminal; preserve the partial artifacts; no new output directory for this plan.

## Tests

Chorus's five tests, each written from the amendment, plus one per rule above if not already covered. For each, name
the mutant it catches; run each mutant if you can and show it failing. Existing 31 tests stay green.

## Report

`part65/REPORT-amend4.md`: SUMMARY / FINDINGS (one line per item: what changed, where, which test) / BLOCKERS. List
anything changed that this brief did not ask for.
