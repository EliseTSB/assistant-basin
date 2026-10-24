# Brief: Part 65 fold 1 (fix the round-1 review), code, tests and runbook only

Domain: /home/elisepc/green/worktrees/ab-part65/part65 (write only there; read the whole worktree, including
`PART65-DESIGN.md` with its Amendment 1 at the end, and `/home/elisepc/green/worktrees/ab-part64`). Do not commit, run on
the GPU, download models or call any paid API.

Read first: `/home/elisepc/green/Journal/reviews/cheap-p65-build-ds-r1.md` (the review) and `PART65-DESIGN.md`
(registered design; Amendment 1 is part of it now). The design is the contract. Where this brief and the review differ,
this brief wins.

## Python

Try `/home/elisepc/neuron-probe/.venv/bin/python -c "import torch, transformers; print(torch.__version__)"` first. If it
works, run the suite with it from `part65/`: `PYTHONPATH=/home/elisepc/green/worktrees/ab-part65
/home/elisepc/neuron-probe/.venv/bin/python -m unittest test_part65`. If it does not work in your sandbox, say so in the
report and say which tests you could not run; Claude runs them on the host afterwards.

## Fixes, one per review finding

- **H1, judge transport.** In `judge65.py`, build the SSL context the way `part63/reader63.py`'s `context()` does (the
  mitm CA bundle at `~/.mitmproxy/ca-bundle-combined.crt` when it exists, else the default) and pass `context=` to every
  `urlopen`. Read the key from `ANTHROPIC_CREDITS_KEY`, falling back to `ANTHROPIC_API_KEY`; the error names both.
- **H2, paths.** `DEFAULT_CORPUS` becomes the canonical Part 60 corpus, `/mnt/c/Users/alexa/neuron-probe-gpu/qwen35/pile10k.jsonl`,
  overridable by `PART64_CORPUS` (and keep `PART60_CORPUS` as a second override). Both `--collection` and `--corpus`
  stay arguments. Write `part65/RUNBOOK-65.md`: on the A770 the steps run under Windows Python from
  `C:\Users\alexa\neuron-probe-gpu` via `job.py` (each line of `steps.txt` runs as `python <line>` in that folder).
  Stage the worktree as `C:\Users\alexa\neuron-probe-gpu\p65basin` (copy `part60/`, `part64/`, `part65/`,
  `atlas_collect_fast2.py` and anything else `steer65` imports, plus `results/part64/collection.jsonl` from the Part 64
  worktree), the way Part 64 scoring used `p64basin`. Give the exact pilot step line with relative Windows paths
  (`p65basin\part65\steer65.py --collection p65basin\results\part64\collection.jsonl --corpus qwen35\pile10k.jsonl ...`),
  the median step, the full-run step, how to copy the outputs back, then the host-side rank, judge (`--probe` first) and
  analysis steps. Check every argument you write against the argparse in the modules.
- **M1, a dead push must not pass.** Two parts. (a) Per call: the hook counts how many times it fired and on which layer
  module; `process_unit` raises if a pushed generation ran with the hook firing zero times or on a layer other than the
  unit's. (b) Per run: record, for each unit, how many of its three +0.8 rollouts differ from that prompt's baseline. Do
  NOT raise per unit on this (a real neuron can be inert at +0.8; the design's check is over the pilot). Add a
  `pilot-check` command (in `steer65.py` or `analyze65.py`) that reads a steering output file and fails unless at least
  half of all +0.8 rollouts differ from their baselines, the dose-0 comparisons all matched, and push norms were equal.
  `dose_zero_matches_baseline` must come from an actual comparison against an independently generated unsteered run,
  never a constant.
- **M2, lens label.** The judge summary's second steer-lens line reads "Tokens made more likely by a negative push"
  (Amendment 1). Keep the code's top-12 increases for both signs.
- **M3, parsing.** Accept a lone A or B with optional surrounding whitespace and an optional single trailing `.` or `)`.
  Anything else is recorded as unparseable. In `analyze65.py`: leave unparseable trials out, leave out a unit with no
  valid trial and count it, and stop with a clear message if more than 5% of all trials are unparseable (Amendment 1).
  The output JSON records both counts.
- **M4, tests that can fail.** Add: (a) a two-layer (or more) tiny model where the unit is on layer 1 and the test
  asserts the hook fired on layer 1 and that a +0.8 push changes the output; (b) a test that `write_direction` for a
  control equals `normalize(W_down @ r)` for a hand-set `r`; (c) a test that the dose-0 check compares against a
  separately generated unsteered run; (d) the planted-null test asserts all seven predictions (65a to 65g) false. For
  each new test, name the mutant it catches in the report; if you can run torch, show each mutant failing.
- **65g.** In `analyze65.py`, add 65g: the middle band's real-neuron agreement is below both the first and the last
  band's (point estimates), with the three band intervals reported. Mark 65c and 65g as exploratory in the output.
- **L1.** `read_jsonl` in both modules ignores an invalid LAST line with a printed warning naming the file (any other
  invalid line still raises). Test it.
- **L2.** The hook layer is an explicit argument of `generate_batch` (no private-attribute default).
- **L3.** `judge65.main` exits non-zero and prints the chunk when any chunk was released.
- **L4.** `analyze65` states in its output which embedding measure 65f used (the registered paired difference) and
  reports the null-adjusted one beside it.
- **L5.** No change (it follows the shared ledger's policy); say so in the report.
- **Tests in a read-only checkout.** Temp directories come from the system default, not the worktree.

## Report

`part65/REPORT-fold1.md`: SUMMARY / FINDINGS (one line per review finding: what changed and where) / BLOCKERS. List
anything you changed that this brief did not ask for.
