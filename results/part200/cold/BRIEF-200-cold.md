# Cold check: Part 200's results README

You are a fresh reader checking a public research write-up before it is published. You have not seen the work. Your
job is to find what is wrong, overstated, unsupported or unclear, with evidence. Do not fix anything in the repository.

Repository root (read-only for you): `/home/elisepc/green/worktrees/ab-part200`
Your writable folder: `/home/elisepc/green/worktrees/ab-part200/results/part200/cold/`
Write your report to `results/part200/cold/COLD.md`.

## What to read

1. `results/part200/README.md` (the write-up under review) and `results/part200/hand_read.md`.
2. `PART200-DESIGN.md` (the registered design: predictions 200.1 to 200.7 and the reading rules).
3. `results/part200/report200.md` and `report200.json`, printed by `part200/report200.py`.
4. `results/part200/compare.json` and `results/part200/score.json` (Part 163's scorer on the rerun).
5. `results/part163/README.md` and `results/part163/score.json` (the published run).
6. `PREDICTIONS.md`: the `## Part 200` section.
7. If you need answer texts: `results/part200/answers.jsonl`, `results/part200/batch1.jsonl` and
   `results/part163/answers.jsonl` (one JSON object per line).

## Rules for files (the C: drive is short of space)

- **Do not copy any folder, and do not copy any file larger than 1 MB.** Read the large files in place.
- If you want to recompute a number, write a small script in your own folder that reads the repository files in place
  and prints the number. Run Python as `/home/elisepc/neuron-probe/.venv/bin/python -I -B <script>`.
- Do not run `part200/report200.py` or `part200/fig200.py` in place (they write into `results/part200/`). If you want
  to rerun them, copy only the `.py` file into your folder and change its output paths.
- Do not run any model and do not use the GPU.

## What to check

- Every number in the README: does it match the file it cites? Recompute at least the predictions table, the Part 163
  verdicts table, the broken and at-cap counts, the Qwen counts and the batch-1 counts.
- Is each verdict exactly what the registered prediction text and the reading rules give? Quote the registered text
  where it disagrees.
- Is the title or the first paragraph stronger than the tables? Is any pattern found after the data (the batch-1 first
  differences, label flips, the at-cap change) presented as a finding rather than as exploratory?
- Does the README attribute the published-versus-rerun differences to the kernel alone, when the run changed the kernel,
  the batch shape and answer length at once (the design says so)?
- Is the batch-1 first-difference analysis sound: how positions were computed, what "64 keys including the prompt"
  means, and whether re-tokenizing decoded text can misplace a first difference?
- Anything a skeptical reader would ask that the README does not answer.

## Report format

`COLD.md`: a verdict line (SHIP, SHIP-WITH-FIXES or HOLD), then numbered findings. Each finding: where (file and the
sentence), what is wrong, the evidence (a number you computed, with the script you used), and a suggested fix. Keep
findings you are unsure of, marked as such. Write COLD.md first as a skeleton and fill it as you go. End with SUMMARY /
FINDINGS / BLOCKERS lines.
