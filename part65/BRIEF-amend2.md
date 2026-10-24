# Brief: implement Part 65 Amendment 2 (missing-trial policy), code, tests and runbook only

Domain: /home/elisepc/green/worktrees/ab-part65/part65 (write only there). Read `PART65-DESIGN.md` to its end; Amendment 2
is the contract for this change and replaces Amendment 1's unparseable-answer paragraph. Context:
`/home/elisepc/green/Journal/reviews/cheap-p65-build-ds-r2.md` (findings H1, H2, M1, M2, M3). Do not commit, run on the
GPU, download models or call any paid API. Do not change `steer65.py` except where a test helper needs it; the steering
pilot is running from a staged copy and passed review.

Python: `/home/elisepc/neuron-probe/.venv/bin/python` (torch works there); run the suite from `part65/` with
`PYTHONPATH=/home/elisepc/green/worktrees/ab-part65`.

## Implement each numbered rule of Amendment 2

1. A trial is missing when its answer is unparseable or absent. Absent means: a registered (unit, trial) pair from the
   judge plan with no row in the answers file. Build the registered pairs from the same plan the judge uses (two
   trials per unit, 500 units), not from the answers file.
2. A unit enters the primary analysis only with both trials valid. Report excluded units by type and layer band.
3. Stop when more than 50 of the registered 1,000 trials are missing. The message states the count and the denominator
   1,000. Tests: 50 missing proceeds, 51 stops; absent rows count as missing.
4. Report two bounds over all 500 units: missing trials scored 0.5, and missing trials scored wrong. Give 65a, 65b and
   the three band estimates under each bound. They do not enter pass or fail.
5. `analyze65` reads the answers file strictly: any malformed line, including the last, stops it. The judge's own
   lenient reader stays as it is for its own resume path.
6. A released chunk is final for its output directory. `judge65` prints the chunk, its error, and this instruction:
   rerun the whole judge into a new output directory. Nothing re-reserves a released attempt. Update `RUNBOOK-65.md` to
   say the same, and add one line that the pilot check's equal-norm gate re-reads the run-wide check (review L1).

## Tests written from the amendment, not from the code

For each rule add a test that would fail if the rule were broken, and name in the report the mutant each one catches.
In particular: a unit with exactly one valid trial is excluded and counted (the review's H1); a planted null with a
missing-trial pattern still gives 65a at 0.5 on the included units; the denominator is 1,000 when rows are absent; a
torn last line stops the analysis.

## Report

`part65/REPORT-amend2.md`: SUMMARY / FINDINGS (one line per rule: what changed and where, and the test) / BLOCKERS. List
anything you changed that this brief did not ask for.
