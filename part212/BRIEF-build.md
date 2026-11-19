# Build Part 212's runner, scorer and tests (assistant-basin, worktree ab-neon-review)

You are a builder for our public neuron research repo. Worktree: `/home/elisepc/green/worktrees/ab-neon-review`
(branch part212). You may write ONLY inside `part212/`. Do not commit; I review and commit.

## What to build

Read `PART212-DESIGN.md` (v3, commit ef9aa7aa) whole before writing code. It is the contract. Build exactly what it
says; where it is ambiguous or seems wrong, do NOT resolve it silently: implement the most literal reading, and list
the point under FINDINGS in your report with the line number.

1. `part212/run212.py` — the runner. Subcommands (argparse): `stage1`, `stage2`, `pilot` (the pilot gate in the Size
   section, with its stopping rule), each with `--out <dir>` (relative path; Windows job runner) and `--device`
   (default `xpu`). Writes Stage 1 JSON and Stage 2 JSONL plus `run.json` (model revision, dtype, attention
   implementation as loaded, tool SHA-256, per-donor `n_non_eos` and `L_gen`, realized perturbation norms, Gate 1/2/3
   inputs). Every row of Stage 2 output carries arm, question index, sample index, seed, prompt, answer text, token
   ids, stop reason.
2. `part212/score212.py` — the scorer. Reads only the runner's saved files. Computes every gate, metric, table and the
   verdict of every prediction 212.1–212.10 exactly as PART212-DESIGN.md's Predictions and Reading rules define them,
   including v3's worst-case recode, the two-sided bound and strong-reversal reading, exclusive five-state
   classification with per-arm multi-hit counts, the (question, sample) transition join, and the scoped denominators
   (192 for full held-out main arms, 24 for 24-row sets). Writes `score.json` and prints a verdict table.
3. `part212/test_part212.py` — pytest tests that run on CPU with NO model download, using synthetic tensors and
   synthetic result files. Required tests (each must go red if the property breaks; for each, break the code on purpose
   once, confirm red, restore, and say so in the report):
   - Stage 2 hook indexing: call 0 (prefill, `x.shape[1] == T`) vs decode calls `s >= 1` (`x.shape[1] == 1`) apply the
     right source for every arm, including `zero_prompt_only` / `zero_gen_only`, and `donor_gen[(s - 1) % L_gen]`.
   - `L_gen = min(n_non_eos, 47)`; the 48th token never enters `donor_gen` or `mu_fact_gen`; EOS never stored.
   - Bank-mean profile mapped onto a recipient of a different prompt length: header 0..2, body 3..T_q−10, suffix
     T_q−9..T_q−1 in order, opening T_q..T_q+4.
   - Frozen vs online hooks: `sub_factual` clamps to a_clean − r; `sub_factual_online` uses the live value.
   - The 4-cell factorial interaction I.
   - Exclusive states: broken first; earliest-starting maker match wins; multi-hit counted.
   - Alden's paired example must FAIL 212.4: `zero` 24/192 clean Google; `trace_resample_fact` keeps 11 and turns 13
     into broken answers, every other row unchanged (6.77% broken). Also a case where 212.4 holds.
   - The alternate headline rejects `delta_g_ali = -5` (strong reversal) and accepts a small two-sided effect.
   - Question-clustered bootstrap: resamples questions, not rows; seed 212; 10,000 draws.
   - Every prediction function on a constructed pass case and a constructed fail case.

## Reuse, do not rewrite

- `neuron_report.load_model` (eager attention; record what it returns), `neuron_report.build_prompt(tok, q, False)`.
- `part66/part66.py`: `row_seed`, `_hooks`/`_pad`/`_stop_reason` patterns, the generation loop; Part 66 is the exact
  rig Stage 2 must match. `part66/score66.py`: `FAMILIES` maker lexicon and its broken-answer screen (read the design
  for the exact broken rule; if Part 66's differs, follow the design and say so).
- `part212/prompts212.json` (SHA-256 0f10c337…3227) holds every prompt bank; load it, never rebuild it.
- Hooks: forward pre-hooks on `layers[L].mlp.down_proj`, as in Parts 25, 66, 74, 183.

## Traps on this rig (each one has bitten us)

- The runner runs under Windows Python on an Intel Arc A770. No `import fcntl` or any POSIX-only module. Open every file
  with `encoding="utf-8"`. Output paths relative.
- Run as `python -I -B part212/run212.py`: the script's folder is NOT on sys.path; insert the repo root the way
  `part212/build_prompts212.py` does.
- The A770 has no float64 kernels: never `.double()` an XPU tensor. Move to CPU first, then accumulate in float64.
- Do not call `.item()` per value inside a hook; copy whole tensors to CPU once per forward.
- Load the model once per process, straight to the device. Windows RAM is the limit (about 7.5 GB for this model).
- `np.savez` appends `.npz` to a path without it; write `.npz` paths explicitly.
- Tests must not need torch-xpu or a model; they run on the host's CPU venv
  (`/home/elisepc/neuron-probe/.venv/bin/python -m pytest part212/test_part212.py`). If your sandbox cannot run
  torch, say so; do not report tests as passing that you did not run.

## Size

Do not run the model. Do not touch `results/`. No provider or API calls.

## Report

Write `part212/BUILD-REPORT.md` with SUMMARY (what exists, test count and result as actually run), FINDINGS (each
design ambiguity or contradiction with line numbers, and every sabotage you ran and its result), BLOCKERS.
