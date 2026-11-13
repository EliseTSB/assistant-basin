# Brief: draft the design for Part 210, Part 165 rerun on the eager attention kernel

Repository: `/home/elisepc/green/worktrees/ab-part210` (branch part210, public main 6efe837e). You may write only
inside this worktree. No model, no GPU, no network, no `git commit`, no `git push` (the reviewer commits). Use
`/home/elisepc/neuron-probe/.venv/bin/python -I -B` for Python.

**Write `part210/REPORT-design.md` FIRST** as a skeleton (SUMMARY / FINDINGS / BLOCKERS lines at the end), keep it
updated as you go, and copy it to `report.md` in your run directory when you finish.

## What this is

Part 178 found that the A770's fused bf16 SDPA attention kernel leaves the CPU float32 path past 64 cached keys. An
audit is rerunning the affected parts on eager attention, with pushed calls in chunks of exactly 8 rows. The most
complete model to follow is **Part 202** (Part 164 on eager): read `PART202-DESIGN.md` closely, all of it, including
its "Changes before registration" (they record three defects a build had and the reviews caught; your design must not
invite them again). Then `part202/run202.py`, `part202/score202.py`, `part202/margins202.py` and
`results/part202/README.md` for what that rerun found. Also `PART200-DESIGN.md`, `PART199-DESIGN.md`, `results/part200/README.md`.

Part 165 is next: `PART165-DESIGN.md` (with its five amendments), `part165/`, `results/part165/README.md`,
`results/part165/score.json`, `results/part165/answers.jsonl`. Its headline: at equal write size, single neurons and
random sets of four broke about 26% of answers and dense pushes about 9%; 6,920 generations at 80 new tokens, batch 64.

## What to produce

1. `PART210-DESIGN.md` at the repository root, from `templates/PART-DESIGN.md`, in the same shape as
   `PART202-DESIGN.md`: Question; Prior art and existing data; Method; Controls; Size; Optimization; Predictions;
   Reading rules; Limits. Follow `PROCEDURE.md` (read all of it, including the lessons log). Plain, short sentences.
   - Method: a wrapper `part210/run210.py`, shaped after `part202/run202.py`: eager loader with a refusal on a
     non-eager load; Part 165's own plan, rows and directions; pushed calls in chunks of exactly 8 in the original row
     order; the launch and load record; the pilot rate guard; resume keyed on Part 165's own `row_key`; a test that
     proves the rebuilt plan equal to all 6,920 rows of `results/part165/answers.jsonl`, key for key, in order.
     `part165/run165.py`'s `run()` takes `loader` and `generate_fn` arguments: say whether wrapping it through those
     is enough to record margins and call shapes, or whether the plan must be rebuilt as Part 202 rebuilt Part 164's.
     Describe it; do not write the runner.
   - Which parts of Part 165 are exposed. The battery (80 new tokens) is. The effect run (`part165/effect165.py`,
     2 new tokens, prediction 165.7) and the geometry reading (`results/part165/reading/`) may not be: compute, by
     script, the longest prompt plus generated length in tokens for the effect run, and say whether it reaches 64 keys.
     Rerun only what is exposed, and say why for each.
   - Controls: Part 165's four gates unchanged, plus Gates R1 (kernel), R2 (same inputs: the sha256 of each input file,
     computed by you from the tree, and Part 165's directions: the rerun's write norms must equal
     `results/part165/norms.json`) and R3 (shapes), as in Part 202.
   - The shape floor, as Part 202 does it (the recorder and `tools/shape_floor.py` exist now; use their real interface,
     read them). **One difference you must handle:** Part 165's `changed` measure compares an answer's first 12 words
     with that phrasing's no-push answer, so a donor answer only reproduces a row's contribution if it has the same
     labels AND the same phrasing. Say how the donor rule handles that, and how the no-push answers (which the rerun
     regenerates too) enter each redraw.
   - Size: for EVERY registered verdict (165.1 to 165.6) and gate of Part 165, state how far the published value is
     from its bar, in rows or slot-rate units, computed by a script (below). Read `part165/score165.py` to find whether
     each verdict reads a point estimate or an interval, and say so. Say which verdicts cannot fail at the published
     rates and which rest on a handful of rows or slots. Note that 165.6's published bootstrap was degenerate (3,475 of
     10,000 draws unusable): say why, from the code and data, and whether the rerun inherits it. This is the most
     important section.
   - Predictions: numbered 210.1 on, each with a threshold, a confidence and a one-line reason with its arithmetic,
     author "DeepSeek V4.1 Flash (draft for Claude)". Include: gates pass; which Part 165 verdicts repeat; the direction
     of the n+ against full broken contrast; answers reaching the 80-token cap per arm (Parts 199, 200 and 202 all saw
     eager answers run to the cap far more often, and Part 165's primary measure counts trigram loops, so longer loops
     may raise broken rates in every arm: work out what that does to each ratio); and one about the shape replicate
     (Part 200: 76 of 200 texts differed; Part 202: 103 of 200). No prediction may be unable to fail at the n it will
     see.
   - Leave `{{YYYY-MM-DD HH:MM}}` in the registration line and nothing else in double braces.
2. `part210/margins210.py`: reads `results/part165/` (score and answers files) and prints, for each registered verdict
   and gate, the published estimate, the bar, and the distance; plus each arm's count of answers at the 80-token cap,
   mean `token_count`, and primary-broken and loop-only counts. Writes `results/part210/margins.md` and `.json`. Every
   number in the design's Size section must come from it. Include `part210/test_margins210.py` (pytest style, and
   runnable as a script that runs its tests) with one deliberate break that you show turns it red, then restore.
3. In `part210/REPORT-design.md`: anything in Part 165's design, code or results that looks wrong or surprising, with
   file and line; every assumption you made; anything you could not determine.

## Rules

- Quote registered prediction text from `PREDICTIONS.md` and `PART165-DESIGN.md` verbatim; never paraphrase a bar.
  Where the two disagree (165.5), say which one Part 165's scorer used and keep that.
- Every number comes from a script you ran in this session. Do not type a count from a README.
- Do not edit anything under `part165/`, `results/part165/`, `PROCEDURE.md`, `PREDICTIONS.md` or `README.md`.
- If a step fails three times the same way, stop, write what you saw in the report, and finish.

## Resume note (2026-11-12, after the first worker was stopped)

A first worker wrote `part210/margins210.py` (it runs and reproduces Part 165's published Spearman values) and
`results/part210/margins.md` and `.json`, then looped for 50 minutes alternating two identical probes and was stopped.
Do NOT rewrite the margins script: read it, run it once, check its numbers against `results/part165/score.json`, and fix
only what is wrong. Then write `part210/test_margins210.py`, `PART210-DESIGN.md` and the report. If you catch yourself
running a command you already ran with nothing edited in between, stop and write what you have: its output has not
changed.
