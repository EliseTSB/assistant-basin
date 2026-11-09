# Brief B: build Part 201's scorer and the shape-floor tool

Repository: `/home/elisepc/green/worktrees/ab-part201` (branch part201). Write only inside it. No model, no GPU, no
network, no `git commit`, no `git push`. Python: `/home/elisepc/neuron-probe/.venv/bin/python -I -B`.

**Write `part201/REPORT-score.md` FIRST** as a skeleton ending in SUMMARY / FINDINGS / BLOCKERS lines; keep it
updated; copy it to `report.md` in your run directory at the end.

Read first: `PART201-DESIGN.md` (Analysis, Predictions, Reading rules), `part201/ROWS-SCHEMA.md` (your input
contract; another builder writes the runner that produces it), `BATCH-SHAPE-NOISE.md` section 1,
`part201/existing201.py`, `part200/report200.py` and `part200/run200.py` (how Part 163's labels are computed:
`part163/score163.py`, `part86/triage86.py`, the Part 161 Qwen rule), and `PROCEDURE.md` section 7 and the lessons of
2026-11-04 and 2026-11-09.

## Deliver

1. `part201/score201.py`, run on the host:
   `python -I -B part201/score201.py --rows results/part201/rows.jsonl --kernel results/part201/kernel-f32.json
   --reference results/kernel-reference/qwen-qwen3.5-4b-cpu-float32.json --load results/part201/load.json --out
   results/part201`. It writes `score.json` and `report.md` and prints the report.
   - Gates G1 to G4 (and G5 if `load.json` records it) from the records; a failed gate is printed first and the
     readings it withdraws (design, Reading rules) are marked "not read", not computed-and-shown.
   - For each pair in the design's Analysis list: rows compared, rows differing (token ids), the first differing
     position's distribution, the margin of the FIRST-NAMED pass at that position in absolute units and in bf16 steps
     of its top logit (`tools/margin_recorder.bf16_step`; if that file does not exist yet, implement the same function
     locally and say so), each of the four labels' flips by direction with a 95% Wilson interval on the flip rate,
     and mean length difference. Rows with `nonfinite` true are counted, listed and left out.
   - Pass K against the CPU reference: token equality per row up to the first position where the reference margin is
     under 0.001, for batch 1 and 2.
   - Over all bf16 rows: the share of positions within 0, 1, 2 and 4 steps, and the share of generation STEPS of a
     call in which at least one unfinished row is within 2 steps (the cost model for a margin gate), per pass.
   - Among rows of pass A: the share that differ in C, split by whether the row has any position within 2 steps.
   - The eleven predictions, each scored exactly as worded in the design (copy the wording into the report verbatim
     with its confidence), as `held`, `failed` or `not scored` with the number; "of the rows that differ" predictions
     are `not scored` when no row differs.
   - `--self-check` runs the scorer on a small synthetic `rows.jsonl` built in a temporary folder and checks known
     answers.
2. `part201/test_score201.py`: synthetic rows covering every prediction's held, failed and not-scored outcomes; the
   label functions checked against Part 163's own on at least 20 real answers from `results/part200/answers.jsonl`
   (your labels must equal what `part200/report200.py` computes for the same rows); one deliberate break per
   prediction scorer that turns a test red. A negative test must assert its intended cause (PROCEDURE.md, 2026-11-04,
   Part 182).
3. `tools/shape_floor.py`: a reusable tool, no Part 201 names in it.
   - Input: a JSON file of rows `[{"id":..., "cell":..., "labels": {name: bool}}]`, a JSON file of replicate pairs
     `[{"id":..., "labels_a": {...}, "labels_b": {...}}]`, and a scorer given as `module:function` that takes the list
     of rows and returns `{verdict_name: value}` (bool or str).
   - Flip rates per label, conditional on the starting value: P(true becomes false) and P(false becomes true), each
     with its count, n and 95% Clopper-Pearson interval. `--rate point|upper` chooses which to simulate with
     (default `upper`).
   - Simulation: `--draws 1000 --seed S`. Each draw flips each row's label independently at the rate for its starting
     value, rescores, and records every verdict. Output per verdict: the observed value and the share of draws equal
     to it ("stability"); per label and cell: the observed count, the simulated mean and SD.
   - `--cluster-key FIELD` (optional): when given, flips are drawn per cluster first with the same marginal rate
     (all rows of a cluster flip together with probability rho, default 0), so a reader can see how much within-unit
     correlation would change the answer. Document the model in the docstring in plain words.
   - Exit 0 with the table; exit 2 on bad input. Deterministic for a given seed.
   - `tools/test_shape_floor.py`: exact checks on tiny inputs (rate 0 gives stability 1.0; a verdict "count >= k"
     with the count exactly at k and a known rate gives the binomial answer within Monte Carlo error at 20,000
     draws); the interval function against known values; one deliberate break that turns a test red.
4. A worked example, exploratory, in `part201/floor200.py`: Part 200's 3,200 battery answers with Part 163's labels
   as rows; its 200 batch-1 pairs (question 0 only) as the replicate; Part 163's five verdicts as the scorer (wrap
   `part163/score163.py`'s verdict code; do not edit it). Write `results/part201/floor200.md` with each verdict's
   stability at `point` and `upper` rates, and the simulated SD of each arm's broken count. State in the file that
   the rates come from one question and are applied to sixteen.

## Rules

- Do not edit `part86/`, `part163/`, `part200/`, `PROCEDURE.md`, `PREDICTIONS.md`, `README.md`, `PART201-DESIGN.md`,
  `BATCH-SHAPE-NOISE.md` or `part201/existing201.py`. If the design's wording cannot be scored as written, score the
  nearest thing and list each difference in the report under "Design differences".
- Every number in a report is printed by code. Run your tests and paste the final pytest summary line into the
  report. A test you could not run is listed as "not run" with the reason.
