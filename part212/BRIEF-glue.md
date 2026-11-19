# Part 212 runner: make the torch glue run (Journal Claude, 2026-11-19)

Worktree `/home/elisepc/green/worktrees/ab-neon-review`. Write only inside `part212/` and your own run directory. Do
not commit. Do not run the real model or touch `results/`. No API calls.

## The acceptance test

`part212/smoke_tiny212.py` builds a tiny random Qwen3.5 (the real tokenizer, 30 layers, MLP width 9216, so TARGET
`(6,753) (7,1169) (11,8868) (29,776)` all exist) and drives the real `run212.main(['pilot', '--out', 'out', '--device',
'cpu'])`: Stage 1, then the Stage 2 pilot, then the pilot stopping rule. Run it exactly like this (the host venv has
torch; the model snapshot is read-only and offline):

    cd /home/elisepc/green/worktrees/ab-neon-review
    HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python -I -B part212/smoke_tiny212.py "$PWD" <SCRATCH> pilot

with `<SCRATCH>` a folder inside your own run directory. **Your job is done when that prints `SMOKE OK pilot` and
exits 0**, with every fix below made and the 54 existing tests in `part212/test_part212.py` still passing
(`/home/elisepc/neuron-probe/.venv/bin/python -I -B -m pytest -q -p no:cacheprovider part212/test_part212.py`).

**First command: run the smoke once.** If `import torch` fails inside your sandbox, stop at once and write BLOCKED in
your report; do not edit code you cannot run.

Today it dies on the first frozen arm: `ValueError: self_patch: a_clean has 4 positions, the arm addresses 22`
(`stage1()` sets `rec['clean'] = recorded['en_maker'][qi]`, one question's array, and `stage1_values` then indexes
`rec['clean'][q]` again).

## Suspects I have read but not confirmed (check each; fix what is real)

1. `record_units` returns `(T, K, C)` where C is the distinct target COLUMNS across all layers, so each unit is stored
   as a row over all four target columns; `install_stage1_hooks` writes `frozen[li]` into the ONE column of that layer.
   The shapes look incompatible. The intended layout is one value per (position, unit).
2. `stage1_values(arm, ...)` looks up `STAGE1_ARM_SPEC[arm]` with the FULL arm name, so `unit:6:753:zero` and
   `random_3:zero` would raise KeyError; it should use the base arm (`parse_arm`/`stage1_spec`), and the trace bank and
   sub profile must come from the base arm too.
3. `realized_perturbation(..., down_weight[0])` passes layer 6's whole `(d_model, n_units)` matrix where the function
   wants one `w_down` column per clamped unit, each from that unit's own layer.
4. The random control arms clamp the drawn pool units, but the recording pass records only TARGET, so there are no
   clean or donor traces for those units. Fix (my decision, implementation only): a pre-pass over the 24 held-out
   EN_MAKER prompts (prompt + forced opening) that only accumulates |a| for every unit of layers 6, 7, 11, 29 and picks
   the pools and the 20 sets; then the recording pass records TARGET + every unit in the 4x50 pools + the two monitor
   units (L24 N3949, L27 N3325), exactly as PART212-DESIGN.md's Stage 1 step 1 lists. Forwards are deterministic, so
   the pre-pass changes no number. Stage 2 does the same.
5. Anything else the smoke finds, in Stage 1, Stage 2 or the pilot rule.

## Rules

- Change no design-level behaviour (arms, values, readouts, rules). If a fix needs a design choice, write it down as
  AMBIGUITY in the report and pick the most literal reading of PART212-DESIGN.md v3.1.
- For each real bug, add a unit test to `part212/test_part212.py` that fails on the old code (show it red once on a
  copy, then restore). The smoke is the integration test; the unit tests pin each fix.
- Keep memory small: the tiny model is ~70M parameters; never record full layer width for all 125 prompts.
- **Write your report as you go**, appending to `part212/BUILD-REPORT.md` under `## Glue fixes (2026-11-19)` with a
  Bash heredoc after EACH fix (bug, line, fix, test name). The previous worker lost its whole report by trying to
  write it all at the end. When the smoke passes, append its last 10 lines and the pytest summary, then stop.
