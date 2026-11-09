# Brief: fold the third code review of Part 198's run and score code (no model runs)

You are fixing code for a pre-registered experiment. The design is the contract:
`/home/elisepc/green/worktrees/ab-part198/PART198-DESIGN.md` (revision 3). Read Method, Controls, Predictions,
Reading rules and the revision logs. The code is `results/part198/run198.py`, `score198.py` and `test_build198.py`,
last changed by fold r2 at `b3aacf2`. `BUILD-NOTES.md` records every earlier finding. `stats198.py` is frozen: import
it, never edit it. Its SHA256 must stay `03e65cb44ab7496d064b497f1e9bb38086716e74b02b5dc3ab998f881473ee64`.

The review to fold, in full: `/home/elisepc/green/Journal/reviews/cheap-part198-code-r3-haiku-haiku-r1.md`. Its probe
scripts are in the run directory
`/home/elisepc/green/DiscordBot/df-runs/cheap-review-part198-code-r3-haiku-haiku-r1-2026-11-08/` (search it for
`probes/`). Read them; copy any you need into `results/part198/` before running them, and never edit them in place.

Defensive research: benign counterfactual facts from a public benchmark; nothing is built to hide an edit.

## Hard limits

- **Do not load any model, run any edit, or call any network.** Allowed: `python -m py_compile`, `fixtures198.py`,
  `test_build198.py`, your own synthetic tests on tiny tensors, and the review's probes.
- ⛔ **New this round:** fold r2's worker ran two probes that reached Arm A's `run_start` just before the real loader
  call. Any test or probe that calls `main()`, `run_arm_*` or anything that could reach a loader must first patch every
  loader (and `torch.load`, `from_pretrained`, `safe_open`) to raise an exception. Check this before running it. If you
  cannot, do not run it; reason about it instead and say so.
- Write only inside `results/part198/`. Do not commit; the parent commits.
- Python: `/home/elisepc/edit-detect/.venv/bin/python -I`.

## What to fix

1. **F1 (Major), with this rule decided by the parent.** A case row's decisive tuple must include every scored
   boolean the scorer reads: each `localization.<stat>.localized`, `aimed_positive_control.oracle_named_strict`,
   `oracle_reader.named_strict`, `blind_reader.localized`, `blind_reader.named_strict` and
   `random_control.localization.t_word.localized`, plus anything else the scorer counts (find them all with a grep of
   `score198.py`; list them in BUILD-NOTES). Raw float diagnostics stay out of the tuple. **Rule:** two terminal rows
   for one case that differ in any scored boolean are a conflict, and the scorer refuses with `ValueError`. That is
   correct even when the cause is float drift: a case whose scored outcome depends on which run you read has no
   single outcome, and the remedy is to rerun that case, not to pick one. Write this rule into BUILD-NOTES and into
   the docstring of `registered_rows`. Add a test where two rows differ only in
   `aimed_positive_control.oracle_named_strict` and the scorer raises, and one per localized stat.
2. **F2 (Minor).** Make the N4 test assert that `run_arm_b` contains exactly one `append_jsonl(log, record)` and that
   it comes after the exact-zero undo check. Show it fails on the review's sabotage A (an extra early append).
3. **F3 (Minor).** Before each Arm C model load, append an `attempt_started` record, and count those (not only
   failure rows) toward the three-attempt cap, so a process killed during load still uses up an attempt. Make the
   scorer report an exhausted Arm C as such. Correct BUILD-NOTES' P1 line to say what the bound now covers. Add a
   test with a child process that is killed during a fake load, which shows the cap holding across four invocations.
   The review's `p_p1_retry.py` is the model for it, with the loader patched as above.
4. Leave F4 to the parent.

## Definition of done

- `fixtures198.py` and `test_build198.py` exit 0, with the check count printed.
- Each new test fails on `b3aacf2`'s code. Show this by running the new test file against copies of the `b3aacf2`
  versions, kept under your run directory, and record which tests fail in BUILD-NOTES.
- `BUILD-NOTES.md` has a "Fold r3" section: each finding, what changed, the tests, and the old-code failures.
- Write your report as `report.md` in your run directory: SUMMARY, FINDINGS, FILES TOUCHED, BLOCKERS, SELF-ASSESSMENT.
  Write a skeleton early and fill it in as you go.
