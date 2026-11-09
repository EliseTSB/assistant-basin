# Cold review, round 2: Part 200's code after the folds

Repository (read-only for you): `/home/elisepc/green/worktrees/ab-part200`. Writable: `part200/review/`. Write
`part200/review/CODE-REVIEW-r2.md`. Round 1 is `part200/review/CODE-REVIEW-r1.md` (9 findings), its probes are in
`part200/review/code-r1-scratch/`, and the folds are in the commits since `041212f`
(`git -C /home/elisepc/green/worktrees/ab-part200 log --oneline 041212f..HEAD` and `git diff 041212f HEAD -- part200/`).
The specification is `PART200-DESIGN.md`.

Rules: no folder copies, no file over 1 MB copied, no model, no GPU, nothing edited outside your folder. Python
`/home/elisepc/neuron-probe/.venv/bin/python -I -B`.

Check narrowly:
1. Each round-1 finding: fixed, partly, or not, with the line. Re-run your round-1 probes against the new code.
2. The new startup repair: can it cut a row that belongs to a complete call, or leave a partial call, for any kill point
   (inside a recipe block, inside a battery block, inside the final 8-row block of a question, between stages, inside
   the batch-1 stage)? Write a probe that kills at many points and relaunches, and check rows, duplicates and Gate R3.
3. Anything the folds introduced that is wrong, and whether the production path (`run200.main` with no injected loader)
   would reach the real model load with the right arguments.

Format: verdict line (LAUNCH, LAUNCH-WITH-FIXES or HOLD), numbered findings with evidence and fixes, then SUMMARY /
FINDINGS / BLOCKERS lines.
