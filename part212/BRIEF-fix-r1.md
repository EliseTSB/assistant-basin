# Part 212: fix the runner and scorer from the GLM 5.3 code review (Journal Claude, 2026-11-19)

Worktree `/home/elisepc/green/worktrees/ab-neon-review`. Write only inside `part212/` (never `PART212-DESIGN.md`,
never `results/`). Do not commit. Do not load the real Qwen3.5-4B weights. No API calls.

## Read first

1. `/home/elisepc/green/Journal/reviews/cheap-part212-code-glm53-r1-glm53-r1.md`: a cold code review with measured
   findings F1-F10 against PART212-DESIGN.md v3.1. Its experiment scripts are named in each finding. It is right on
   F1-F5; check F6-F10 and fix what is real.
2. `PART212-DESIGN.md` (v3.1, registered; do not edit it).
3. `part212/run212.py`, `part212/score212.py`, `part212/test_part212.py` (61 tests passing), `part212/smoke_tiny212.py`,
   and the "Glue fixes" section of `part212/BUILD-REPORT.md` (what the last worker changed).

## What to fix (the review's own "smallest fix" lines are a good start)

- **F1** Stage 2 `zero` and every `random_k:zero` must zero the clamped units at every call and position.
- **F2** Stage 2 must plan and run the 6 fluency guards x 4 samples per full arm (design line 195: 240 rows per full
  arm, 3,888 planned in all), with guard rows taking the bank-level profile mapped onto the guard's own length and no
  per-question donor lookup. Fix `test_runner_arm_and_row_plan_matches_the_design` to the design's numbers.
- **F3** Stage 1's random arms clamp each pool unit to ITS OWN recorded value (own donor trace, own `a_clean - r`, own
  `-a_clean`), from the recording that already holds the 4x50 pool units; `realized_perturbation_norm` is the real
  4-unit write for every arm, never 0 for a control that changes anything.
- **F4** monitor readouts at one consistent position across arms; **F5** `g_seq` scored from the right position.
- **F6-F10**: fix what is real; say why for anything left.

For each real fix, a unit test that fails on the old code (show it red once on a copy, then restore).

## Acceptance test

Extend `part212/smoke_tiny212.py` so that, besides running the pilot path end to end, it **asserts that the
interventions do what they claim** on the tiny model: in a hooked forward, `zero` leaves the clamped columns at 0 in
Stage 1 and at every Stage 2 call; a `random_k` arm's clamped columns hold that pool unit's own value, and its
realized perturbation norm is above 0; the planned Stage 2 rows include the guards with 240 rows per full arm. The
review shows a smoke that only proves the code runs passes over all of F1-F5.

Run it on the host venv with a scratch folder in your own run directory:

    cd /home/elisepc/green/worktrees/ab-neon-review
    HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B part212/smoke_tiny212.py "$PWD" <SCRATCH> pilot

**Done means** `SMOKE OK pilot` with every new assertion passing, and all tests pass. If `import torch` fails in your
sandbox, stop and write BLOCKED. Append your report to `part212/BUILD-REPORT.md` under
`## Fixes from the GLM review (2026-11-19)`: one entry per finding (fix, line, test).

## Continuation note (2026-11-19 06:01 CT, Journal Claude)

A first worker (DeepSeek) started this brief and stopped partway when its usage limit ran out. Its edits are in the
working tree (`git diff HEAD -- part212/` shows them; HEAD is the committed glue-fixed code) and it may have started a
"Fixes from the GLM review" section in BUILD-REPORT.md. **Continue from the working tree**: check what it finished,
keep what is right, finish the rest. Do not reset the files to HEAD. Write your report to part212/BUILD-REPORT.md as
above (your own run directory's report.md is fine too).
