# Brief: draft the design for Part 202, Part 164 rerun on the eager attention kernel

Repository: `/home/elisepc/green/worktrees/ab-part202` (branch part202, public main 1d110d4). You may write only inside
this worktree. No model, no GPU, no network, no `git commit`, no `git push` (the reviewer commits). Use
`/home/elisepc/neuron-probe/.venv/bin/python -I -B` for Python.

**Write `part202/REPORT-design.md` FIRST** as a skeleton (SUMMARY / FINDINGS / BLOCKERS lines at the end) and keep it
updated as you go, and copy it to `report.md` in your run directory when you finish.

## What this is

Part 178 found the A770's fused bf16 SDPA attention kernel leaves the CPU float32 path past 64 cached keys. An audit
is rerunning the affected parts on eager attention. Parts 199 (rerun of 170) and 200 (rerun of 163) are the models to
follow: read `PART200-DESIGN.md` and `PART199-DESIGN.md` closely, and `results/part200/README.md` for what the last
rerun found. Part 164 is next (`PART164-DESIGN.md`, `part164/`, `results/part164/`). Its headline: bare Qwen 6.4%
against 8.1%; broke 27.0% against 7.8%; 6,516 generations at 80 new tokens.

## What to produce

1. `PART202-DESIGN.md` at the repository root, from `templates/PART-DESIGN.md`, in the same shape as
   `PART200-DESIGN.md`: Question; Prior art and existing data; Method; Controls; Size; Optimization; Predictions;
   Reading rules; Limits. Follow `PROCEDURE.md` (read all of it, including the lessons log entries of 2026-11-04 and
   2026-11-09). Plain, short sentences.
   - Method: a wrapper `part202/run202.py` around Part 164's own `run()` in the way `part200/run200.py` wraps Part
     163's: eager loader with a refusal on a non-eager load, pushed calls in chunks of exactly 8 in the original row
     order, the launch and load record, the pilot rate guard, resume. Describe it; do not write the runner.
   - Controls: Part 164's own gates unchanged, plus Gates R1 (kernel), R2 (same inputs, with the sha256 of each input
     file, computed by you from the tree) and R3 (shapes) as in Part 200.
   - **New in this part, from `part202/REFERENCE-BATCH-SHAPE-NOISE.md` section 1** (a draft plan, not yet procedure;
     read it): every row records its token ids and per-position top-two margins; a shape replicate regenerates a seeded
     sample of 200 rows, spread over the cells in proportion, one row per call; each registered verdict is reported
     with its stability from `tools/shape_floor.py`. The recorder and the tool are being built in Part 201 and do not
     exist yet: describe their use and mark each place that depends on their final interface with the literal text
     `{{after Part 201}}`.
   - Size: for EVERY registered verdict and gate of Part 164, state how far the published value is from its bar, in
     rows or units, computed by a script (below). Say which verdicts cannot fail at the published rates and which
     rest on a handful of rows. This is the most important section; Part 200's design did it by hand for 163.1 to
     163.5 and its cold check still found three predictions that could not fail.
   - Predictions: numbered 202.1 on, each with a threshold, a confidence and a one-line reason, author "Luna 6 (draft
     for Claude)". Include: gates pass; which Part 164 verdicts repeat; the direction of the headline contrasts; a
     prediction about answers reaching the 80-token cap (Parts 199 and 200 both saw most eager answers run to the
     cap); and one about the shape replicate (Part 200: 76 of 200 texts differed, labels flipped on 2 to 6 rows). No
     prediction may be one that cannot fail at the n it will see; work out the arithmetic for each and write it in
     the reason.
   - Leave `{{YYYY-MM-DD HH:MM}}` in the registration line and nothing else in double braces except the
     `{{after Part 201}}` markers.
2. `part202/margins202.py`: reads `results/part164/` (score and answers files) and prints, for each of Part 164's
   registered verdicts and gates, the published estimate, the bar, and the distance in rows or units; plus each arm's
   count of answers at the 80-token cap and mean length (`token_count`). Writes `results/part202/margins.md` and
   `.json`. Every number in the design's Size section must come from it. Include a small test file
   `part202/test_margins202.py` (pytest style, and also runnable as a script that runs its tests) with one deliberate
   break that turns it red.
3. In `part202/REPORT-design.md`: anything in Part 164's design, code or results that looks wrong or surprising, with
   file and line; every assumption you made; anything you could not determine.

## Rules

- Quote registered prediction text from `PREDICTIONS.md` and `PART164-DESIGN.md` verbatim; never paraphrase a bar.
- Every number comes from a script you ran in this session. Do not type a count from a README.
- If Part 164's run entry point does not take a loader or generate function the way Part 163's does, say exactly what
  the smallest behaviour-preserving change to `part164/` would be, and do not make it.
- Do not edit anything under `part164/`, `results/part164/`, `PROCEDURE.md`, `PREDICTIONS.md` or `README.md`.
