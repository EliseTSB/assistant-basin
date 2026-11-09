# Brief: fold the second code review of Part 198's run and score code (no model runs)

You are fixing code for a pre-registered experiment. The design is the contract:
`/home/elisepc/green/worktrees/ab-part198/PART198-DESIGN.md` (revision 3). Read Method, Controls, Predictions,
Reading rules and the revision logs. The code you are fixing is `results/part198/run198.py`, `score198.py`,
`test_build198.py` (written by an earlier worker, fold r1 committed at `1a05ea9` (crosscheck json `e5c2d8a`)). `stats198.py` is frozen: import it, never
edit it.

The review to fold, in full: `/home/elisepc/green/Journal/reviews/cheap-part198-code-r2-ds-r1.md`. Its probe scripts,
each reproducing one finding on synthetic data with no model, are in
`/home/elisepc/green/Journal/scratch/review-part198-code-r2-ds-r1-2026-11-08/probe/` (read them; you may run them, they
must not be edited in place: copy any you need into `results/part198/` first).

Defensive research: benign counterfactual facts from a public benchmark; nothing is built to hide an edit.

## Hard limits (unchanged)

- **Do not load any model, run any edit, or call any network.** Allowed: `python -m py_compile`, `fixtures198.py`,
  `test_build198.py`, your own synthetic tests on tiny tensors, the review's probes, and `run198.py ... --dry-run`.
- Write only inside `results/part198/`. Do not commit; the parent commits.
- Python: `/home/elisepc/edit-detect/.venv/bin/python -I`.

## What to fix

Every finding the review marks Blocker or Major (N1, N2, N3, N4, P1), as the review describes, and the Minors N5 and N6.
For P2 and P3, first read the design (Predictions and Reading rules): if the code already does what the design says, do
not change it; say so in BUILD-NOTES with the design line. If the design itself is silent or contradictory there, put it
in BLOCKERS for the parent; do not invent a rule. If you find the review wrong on the design anywhere, do not fix it, and
say why in BUILD-NOTES with the design line.

- **N1** must make a legitimate replay or resume never unscoreable, and the real duplicate (two different terminal rows
  for one key) always refused. Test both, with rows written by the runner's own record functions.
- **N2** fingerprint any tensor shape including 0-d; a fingerprint failure must say so, never report "not restored".
- **N3** tests must drive the runner's restoration path itself (a fake model object is fine), not only the helpers.
- **P1** a deterministic Arm C failure must stop retrying after a bounded number of attempts and be reported.
- Keep `fixtures198.py` passing and `stats198.py` untouched (its sha256 is pinned in crosscheck198.json).

## Tests

For every Blocker and Major add a test in `test_build198.py` that **fails on the code at `1a05ea9`** and passes on
yours. Prove that: keep a pristine copy of the 1a05ea9 versions (`git -C /home/elisepc/green/worktrees/ab-part198 show
1a05ea9:results/part198/run198.py`, same for score198.py), run your new tests against them, and paste the failing
names into BUILD-NOTES. A test that passes on the old code does not test the fix. Then run the review's probes against
your fixed code and paste what changed.

`fixtures198.py` must still pass. `test_build198.py` must exit 0.

## Report

Update `BUILD-NOTES.md`: a section "Fold r2" listing each finding, what you changed (file:line), the test that covers
it, and the old-code failure. Then `report.md` in your run directory with SUMMARY / FINDINGS / BLOCKERS. If anything in
the review conflicts with the design, it goes in BLOCKERS, not in a quiet choice.
