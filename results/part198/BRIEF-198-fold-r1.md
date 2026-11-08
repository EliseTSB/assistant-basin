# Brief: fold the code review of Part 198's run and score code (no model runs)

You are fixing code for a pre-registered experiment. The design is the contract:
`/home/elisepc/green/worktrees/ab-part198/PART198-DESIGN.md` (revision 3). Read Method, Controls, Predictions,
Reading rules and the revision logs. The code you are fixing is `results/part198/run198.py`, `score198.py`,
`test_build198.py` (written by an earlier worker, committed at `65c2be6`). `stats198.py` is frozen: import it, never
edit it.

The review to fold, in full: `/home/elisepc/green/Journal/reviews/cheap-part198-code-r1-ds-r1.md`. Its probe scripts,
each reproducing one finding on synthetic data with no model, are in
`/home/elisepc/green/Journal/scratch/review-part198-code-r1-ds-r1-2026-11-08/probe/` (read them; you may run them, they
must not be edited in place: copy any you need into `results/part198/` first).

Defensive research: benign counterfactual facts from a public benchmark; nothing is built to hide an edit.

## Hard limits (unchanged)

- **Do not load any model, run any edit, or call any network.** Allowed: `python -m py_compile`, `fixtures198.py`,
  `test_build198.py`, your own synthetic tests on tiny tensors, the review's probes, and `run198.py ... --dry-run`.
- Write only inside `results/part198/`. Do not commit; the parent commits.
- Python: `/home/elisepc/edit-detect/.venv/bin/python -I`.

## What to fix

Every Blocker and every Major, as the review describes, unless you find the review wrong on the design; then do not
fix it, and say why in `BUILD-NOTES.md` with the design line.

- **B1** Clopper-Pearson lower bound. Prefer the beta-quantile identity if scipy is in the venv; otherwise fix the
  bisection. Test k=20,n=24 (lower ~0.6443), k=1,n=24 (~0.0011), k=24,n=24, k=0,n=24 against exact values you compute
  independently (for example scipy, or a separate root-finder in the test).
- **B2** Mass forecasts must count the runner's actual localization shape. Test with a batch row built by the runner's
  own `localization()` output shape, not a hand-written one.
- **B3** Aimed control direction: `gamma * (row - mean(row))`, then unit length. Test with non-uniform gamma.
- **B4** S1 denominator: ROME rows that took AND are name-eligible; count missing controls separately and report them
  (a took, name-eligible row with no control is a failure of S1's count, not a silent drop).
- **B5** S1 suppresses only the 198.4 oracle-naming forecast, never the update-top forecast.
- **M1** A torn LAST line without a trailing newline is dropped and recorded; any other malformed line still raises.
- **M2** One authoritative terminal row per key: keep the first; refuse to score a stream with conflicting terminal
  rows for the same key, naming them.
- **M3** Arm B `took` uses the runner's own measured `after_exact`; record wsp's `free_exact` beside it and flag any
  disagreement. No silent fallback between the two.
- **M4** Restoration compares every parameter and buffer against the snapshot taken before the first record, not just
  the keys the editor reported. Keep memory in mind: GPT-2 XL is ~6 GB in float32; the snapshot already exists from
  `load_gpt2_model`'s clone, so do not take a second full copy. Say in BUILD-NOTES what the peak memory of the new
  check is.
- **M5** Arm C: a failed calibration model is retried, not counted as done; a missing `tstar.json` is a hard error
  before any evaluation or host model loads.
- **M6** 198.8 reads the row with the highest `n_written_so_far`.
- **M7** `--limit` is checked before each model load in both Arm C loops.
- **M8** `crosscheck198.json` must carry the sha256 of the `stats198.py` it checked, and `preflight` refuses when the
  current file's hash differs. `crosscheck198.py` writes that json: add the hash there (it is in your domain; it is not
  frozen like stats198.py). Do not re-run the crosscheck yourself (it loads models); the parent will.
- Minors: fix m1 (delete the dead helper or make the scorer use it, and test the scorer path), m2 (report dropped
  gate-failing batches), m3 (store a hash of the base statistics' inputs in `base-gpt2.json` and check it), m5 (write
  the failure row before raising). m4 is optional.

## Tests

For every Blocker and Major add a test in `test_build198.py` that **fails on the code at `65c2be6`** and passes on
yours. Prove that: keep a pristine copy of the 65c2be6 versions (`git -C /home/elisepc/green/worktrees/ab-part198 show
65c2be6:results/part198/run198.py`, same for score198.py), run your new tests against them, and paste the failing
names into BUILD-NOTES. A test that passes on the old code does not test the fix. Then run the review's probes against
your fixed code and paste what changed.

`fixtures198.py` must still pass. `test_build198.py` must exit 0.

## Report

Update `BUILD-NOTES.md`: a section "Fold r1" listing each finding, what you changed (file:line), the test that covers
it, and the old-code failure. Then `report.md` in your run directory with SUMMARY / FINDINGS / BLOCKERS. If anything in
the review conflicts with the design, it goes in BLOCKERS, not in a quiet choice.
