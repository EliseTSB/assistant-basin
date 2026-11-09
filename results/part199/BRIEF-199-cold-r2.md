# Cold check, round 2: Part 199's folds

Round 1 (`results/part199/cold/COLD.md`, 13 findings) was folded in commit 0d37036. Check the folds, narrowly.

Repository root (read-only for you): `/home/elisepc/green/worktrees/ab-part199`
Your writable folder: `/home/elisepc/green/worktrees/ab-part199/results/part199/cold/`
Write your report to `results/part199/cold/COLD-r2.md`. Your round-1 scripts are in `cold/scripts/`; you may reuse them.

## What to read

- `results/part199/cold/COLD.md` (the round-1 findings).
- `git -C /home/elisepc/green/worktrees/ab-part199 show 0d37036` (the fold diff).
- `results/part199/README.md`, `results/part199/report199.md`, `results/part199/report199.json`.
- `README.md` (front page): search for "Part 199" (three places). `results/part170/README.md`: the top note.
- `part199/report199.py` and `part199/fig199.py` (changed to print the new counts).

## Rules for files (the C: drive is short of space)

- Do not copy any folder, and do not copy any file larger than 1 MB. Read the large files in place.
- Recompute with small scripts in your own folder, run as `/home/elisepc/neuron-probe/.venv/bin/python -I -B <script>`.
- Do not run `part199/report199.py` or `part199/fig199.py` in place. No model, no GPU.

## What to check

1. Each round-1 finding: folded correctly, partly, or not. Say which, with the sentence.
2. Every number the folds added (the set-level list counts, the 0.372 ratio, the Part 191-rule counts, the loop-end
   counts): does it match the files? Recompute the ones you did not compute in round 1.
3. Anything the folds introduced that is wrong, stronger than the evidence, or inconsistent across the README, the front
   page and the Part 170 note.

## Report format

`COLD-r2.md`: a verdict line (SHIP, SHIP-WITH-FIXES or HOLD), then numbered findings with evidence and a suggested fix.
End with SUMMARY / FINDINGS / BLOCKERS lines.
