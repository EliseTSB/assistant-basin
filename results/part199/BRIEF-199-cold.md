# Cold check: Part 199's results README

You are a fresh reader checking a public research write-up before it is published. You have not seen the work. Your
job is to find what is wrong, overstated, unsupported or unclear, with evidence. Do not fix anything in the repository.

Repository root (read-only for you): `/home/elisepc/green/worktrees/ab-part199`
Your writable folder: `/home/elisepc/green/worktrees/ab-part199/results/part199/cold/`
Write your report to `results/part199/cold/COLD.md`.

## What to read

1. `results/part199/README.md` (the write-up under review).
2. `PART199-DESIGN.md` (the registered design: predictions 199.1 to 199.9 and the reading rules).
3. `results/part199/report199.md` and `report199.json`, printed by `part199/report199.py`.
4. `results/part199/compare.json` and `results/part199/score.json` (the scorer's and the comparison's output).
5. `results/part170/README.md` top note and `README.md` (the front page): search for "Part 199" (three places).
6. `PREDICTIONS.md`: the `## Part 199` section.
7. If you need answer texts: `results/part199/answers.jsonl` and `results/part170/answers.jsonl` (one JSON object per
   line; fields `stage`, `ban`, `dose`, `group`, `prompt_id`, `answer`, `token_ids`, `token_count`,
   `ended_eos_before_80`).

## Rules for files (the C: drive is short of space)

- **Do not copy any folder, and do not copy any file larger than 1 MB.** Read the large files in place.
- If you want to recompute a number, write a small script in your own folder that reads the repository files in place
  and prints the number. Run Python as `/home/elisepc/neuron-probe/.venv/bin/python -I -B <script>`.
- Do not run `part199/report199.py` or `part199/fig199.py` in place (they write into `results/part199/`). If you want
  to rerun them, copy only the `.py` file into your folder and change its output paths.
- Do not run any model and do not use the GPU.

## What to check

- Every number in the README: does it match the file it cites? Recompute at least the predictions table, the list
  table, the frozen-prompt counts, the same-population table and the length counts.
- Is each verdict exactly what the registered prediction text and the reading rules give? Quote the registered text
  where it disagrees.
- Is the title or the first paragraph stronger than the tables? Is any pattern found after the data presented as a
  finding rather than as exploratory?
- Are the explanations (why the lists differ, where the frozen prompts went, why loop rates rose) supported by the
  evidence shown, or worded more strongly than it allows?
- Are the front-page sentences and the Part 170 note faithful to the README?
- Anything a skeptical reader would ask that the README does not answer.

## Report format

`COLD.md`: a verdict line (SHIP, SHIP-WITH-FIXES or HOLD), then numbered findings. Each finding: where (file and the
sentence), what is wrong, the evidence (a number you computed, with the script you used), and a suggested fix. Keep
findings you are unsure of, marked as such. End with SUMMARY / FINDINGS / BLOCKERS lines.
