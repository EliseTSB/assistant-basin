# Part 213: fix the runner and scorer from the GLM 5.3 code review (Journal Claude, 2026-11-19)

Worktree `/home/elisepc/green/worktrees/ab-neon-review`. Write only inside `part213/` (never `PART213-DESIGN.md`,
never `results/`). Do not commit. Do not load the real OLMo weights. No API calls.

## Read first

1. `part213/REVIEW-code-glm53-r1.md`: a cold code review with measured findings F1-F8 and L1-L7. It is right on
   F1-F8; each finding has a "smallest fix" you may follow.
2. `PART213-DESIGN.md`: "Implementation readings (v5.1)" and **"Amendment 1"** (registered and public). Amendment 1
   restores v4's 936 `offdiag_8` passes: unit u's `neuron +0.8` hook applied to **peer v's own row**, on `self_h0`,
   `third_0`, `third_2`, for each u's 8 peers (`offdiag_8`, 4 slot + 4 concept). 213.11's full-NP specificity reads
   those passes, not `lp_by_unit` on u's own row.
3. `part213/run213.py`, `part213/score213.py`, `part213/test_part213.py` (23 tests, all passing now).

## What to fix

- **F1** read positions: include the opening tokens in `first_pos` and `content_pos` (`make_row`).
- **F2** the `lp_by_unit` / `np_specs` path must never index past a row or read pad positions; score a unit's NP
  only on that unit's own row.
- **F3** add the 936 offdiag passes (Amendment 1) to `instruct` Stage 1 and make `score213.full_np_increment` (and
  213.11) read them. Count: `instruct` executes 6,366 passes.
- **F4** the scorer and runner must agree on per-checkpoint manifests (`run-<checkpoint>.json`, never one
  overwritten `run.json`), and the pilot's Gate 1 and Gate 2 evidence must reach the scorer by a documented,
  automatic path (e.g. Stage 1 copies `<pilot dir>/pilot_summary.json` evidence into its records dir, or the scorer
  takes `--pilot <dir>`). The real pilot output is at `results/part213/pilot/` (read-only to you); match its format.
- **F5** implement headline 3's Stage 2 clause (design l.513-515); gates 3 and 4 must not pass with the audit absent.
- **F6** resume must not treat a partial or crashed output file as a finished run (write a completion marker).
- **F7** a partial blind audit must not count as complete.
- **F8** keep only the logits the scorer reads (each unit's own first-content token id and its `offdiag_ou` ids, plus
  anything else `score213.py` reads, listed in the report), never full-vocabulary rows. Entropy and top-1 stay as the
  runner's scalars.
- **L1-L7**: fix the cheap ones; for any you leave, say why in the report.

For each real fix, a unit test that fails on the old code (show it red once on a copy, then restore).

## Acceptance test (the important part)

Write `part213/smoke_tiny213.py`: build a **tiny random OLMo 2** from `part213/tiny-olmo-cfg/config.json` (16 layers,
since units sit in layers up to 12; `intermediate_size` at least 7,851, since unit neurons go up to 7,850; small
`hidden_size`, heads and kv-heads), with the real tokenizer from that folder, and patch the runner's model loading,
snapshot lookup, provenance and push vectors (random vectors of the tiny hidden size, fixed seed) so the **real
command-line paths** run: `pilot`, then `stage1` (all checkpoints may reuse the one tiny model), then `stage2`, then
`score213.py` on the records directory those produced. Use a scratch folder in your own run directory, set
`HF_HUB_OFFLINE=1`, and run it on the host venv:

    cd /home/elisepc/green/worktrees/ab-neon-review
    HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B part213/smoke_tiny213.py "$PWD" <SCRATCH>

**Done means:** the smoke prints `SMOKE OK`, `score213` prints a verdict for every prediction (whatever it is, on
random weights) with no prediction `untestable` for a missing file or field, the loaded records fit easily in memory,
and all tests pass. If `import torch` fails in your sandbox, stop and write BLOCKED. Append your report to
`part213/BUILD-REPORT.md` under `## Fixes from the GLM review (2026-11-19)`: one entry per finding (fix, line, test).
