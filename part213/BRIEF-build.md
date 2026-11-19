# Build Part 213's runner, scorer and tests (assistant-basin, worktree ab-neon-review)

You are a builder for our public neuron research repo. Worktree: `/home/elisepc/green/worktrees/ab-neon-review`
(branch part212). You may write ONLY inside `part213/`. Do not commit; I review and commit.

## What to build

Read `PART213-DESIGN.md` (v5, commit 7de53067) whole before writing code. It is the contract. Build exactly what it
says; where it is ambiguous or seems wrong, do NOT resolve it silently: implement the most literal reading and list the
point under FINDINGS in your report with the line number. The "v5 disposition rule" in the Predictions section governs
every verdict.

1. `part213/run213.py` — the runner, on CPU (OLMo 2 1B, `--mem 10G` as the design says). Subcommands: `pilot`, `stage1`,
   `stage2`, `fr` (whatever stages the design names), each with `--out <dir>`. It writes raw per-pass records (unit,
   prompt id, arm, dose, the logits the metrics need, `lp_content_sum`, `lp_content_mean`, `n_content_tokens`) and
   `run.json` (snapshot commits, dtype, tool SHA-256, Gate 0/1/2 inputs). It obeys the Part 180 run-order seal and the
   pilot's output restriction exactly as the design states them.
2. `part213/score213.py` — reads only the runner's saved records; computes every gate, `N_z`/`R_z`/`L_z`, `SpecLogit`,
   the paired transfer ratios with eligibility, the v5 numerator sign guard and the scene-framed denominators,
   `Delta_lp_spec` in v5's length-matched form plus the slot-peer-only term, the own-unit full-NP clauses, the
   finite-difference `J`, the sign test, the Stage 2 audit tables, and the verdict of every prediction 213.1-213.11
   under the v5 disposition rule (held / failed / inconclusive / not read). Writes `score.json` and prints a table.
3. `part213/test_part213.py` — pytest on CPU with NO model download, synthetic records only. Required tests, each shown
   red under a deliberate break of a COPY of the code (never the tracked file), then restored, and listed in the report:
   - the v5 disposition rule: a straddling CI gives `inconclusive`, `n_eligible < 15` gives `not read`, a CI wholly on
     the wrong side gives `failed`, and one failed clause fails the prediction;
   - the numerator sign guard: N_z(SELF)=2.0 and N_z(THIRD)=-1.5 for every unit gives `inconclusive`, not a held ratio;
   - scene-framed denominators for `tau_M/3` and `tau_cop_nonid/3` (pooled versions computed but deciding nothing);
   - `Delta_lp_spec` length matching: with a generic per-token increment and longer off-diagonal NPs the statistic is
     0, not negative; it equals v4's sum form when lengths match;
   - a slot-family writer (own and slot peers up by D, concept peers 0) fails 213.11's slot-peer clause;
   - `SpecLogit` (own first-token logit minus the median over `O_u`) cancels a constant shift;
   - eligibility and `null` exclusion; unit-clustered bootstrap resamples units, seed 213, 10,000 draws;
   - the exact one-sided paired sign test: 7 of 8 discordant gives p = 0.0352, 6 of 8 gives 0.1445.

## Reuse, do not rewrite

`part92/common92.py` (`load_model`, `prompt_ids`, `add_row_hook`, `generate_batch`), `part85/checkpoints.py` (snapshot
commits), `part106` scoring code for full-sentence `lp_sum` (Gate 2 must reproduce `results/part106/lp-instruct.jsonl`),
`results/part99-pushes.pt` (pushes; read-only), `part213/targets213.json` (SHA-256 7edb3c62…1892; load, never rebuild).

## Traps on this rig

- Run as `python -I -B part213/run213.py`: the script's folder is not on sys.path; insert the repo root the way
  `part213/build_targets213.py` does. Open every file with `encoding="utf-8"`.
- OLMo's unknown token is its end token. A bf16 greedy answer can depend on batch shape: compare runs only at the same
  batch shape (Gate 1 does this).
- `np.savez` appends `.npz` to a path without it.
- Tests must not need a model; they run on the host CPU venv (`/home/elisepc/neuron-probe/.venv/bin/python -m pytest
  part213/test_part213.py -p no:cacheprovider`). If your sandbox cannot run torch, say so; never report tests you did
  not run as passing.
- Do not run the model, do not touch `results/`, no provider or API calls. The Stage 2 blind reader is DeepSeek V4.1
  Flash per the design: build its call behind a function the tests replace with a fake, and do not call it.

## Report

`part213/BUILD-REPORT.md`: SUMMARY (files, tests as actually run and their result), FINDINGS (every design ambiguity
with line numbers; every sabotage and its result), BLOCKERS.
