# Brief: Part 200's report script, figure and README (before the cold check)

Repository: `/home/elisepc/green/worktrees/ab-part200` (branch `part200`). Writable for you: `part200/` and
`results/part200/`. Do not push, do not edit PREDICTIONS.md, README.md at the repo root, or any other part's folder.
Commit nothing; Claude reviews and commits. No model, no GPU: everything here reads files.

Read first: `PART200-DESIGN.md` (registered 2026-11-09 13:39 CT; its Predictions and Reading rules govern),
`PROCEDURE.md` sections 7 and 8, and Part 199 as the model to follow: `part199/report199.py`, `part199/fig199.py`,
`results/part199/README.md`.

## The results (already computed; your script must recompute every number, never type one)

- `results/part200/score.json` (Part 163's own scorer on the rerun), `results/part200/compare.json` (`--compare`,
  sections `scored` and `exploratory`), `results/part200/load.json`, `answers.jsonl` (3,416 rows), `batch1.jsonl`
  (200 rows), `launch.jsonl`. Published side: `results/part163/` (answers.jsonl, score.json, README.md).
- What `--compare` printed: gates R1, R2, R3 and Part 163's three passed; 200.1 to 200.6 held; 200.7 failed
  (76 of 200 batch-1 answers differ from their chunked answers; 39 of 100 neuron, 37 of 100 random). Part 163's five
  verdicts are unchanged (163.1 to 163.4 held, 163.5 failed).

## Write

1. `part200/report200.py`: reads only result files and writes `results/part200/report200.json` and
   `results/part200/report200.md` with every number the README uses: the gates, the five Part 163 verdicts with their
   estimates and intervals beside the published ones, each of 200.1 to 200.7 with the value it was decided on, broken
   and at-cap counts per arm (rerun and published), the Part 161-rule Qwen counts, and the batch-1 counts.
2. Exploratory (label it so everywhere): for the 76 changed batch-1 rows, the token position where the batch-1 and the
   chunked answer first differ (tokenize with the model's tokenizer if it is in the local Hugging Face cache; if it is
   not, use characters and say so), split by arm and by whether the first difference falls before or after 64 cached
   keys counting the prompt; and whether the change alters the row's Part 163 labels (name gone, broken, Qwen) as
   `score163.py` decides them. Also the exploratory items already in compare.json.
3. `part200/fig200.py` -> `results/part200/fig200.png`, drawn from report200.json: left, per-arm broken and at-cap
   counts, published beside rerun; right, the batch-1 first-difference positions. Follow the plain style of fig199.
4. `results/part200/README.md`: title stating the result, the figure, one paragraph of what was run (dates and times
   from the files), an **Answer** paragraph, the Predictions table with each registered prediction verbatim and its
   confidence, Result and Verdict columns, Part 163's verdicts table, then each skeptical question answered by its
   number (does the batch shape matter, given 200.7; is the at-cap jump the kernel; did the arms' gap hold), every number
   linked to its script and file, and a "Cold check:" line left as `pending`. Write the registered verdicts first and
   any reading after them. Read at least five answers in each cell (arm by stage) and in the changed batch-1 rows, and
   put what you saw in `results/part200/hand_read.md`, quoting short excerpts.

Watch for the cold checks' usual faults (PROCEDURE.md section 8): a title stronger than the table, counts typed rather
than computed, registered wording paraphrased, a pattern found afterwards stated as a finding, a claim about an arm that
was not read.

Run `python -I -B part200/report200.py` and `python -I -B part200/fig200.py` with
`/home/elisepc/neuron-probe/.venv/bin/python`, then
`/home/elisepc/neuron-probe/.venv/bin/python -I tools/part_check.py publish 200` and report its output.

## Report

Write `part200/REPORT-writeup.md` FIRST as a skeleton and keep it updated: files written, commands run and their
results, anything uncertain. End with SUMMARY / FINDINGS / BLOCKERS lines.
