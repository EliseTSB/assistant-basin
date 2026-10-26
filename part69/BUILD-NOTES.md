# Part 69 build notes

Implemented the Part 69 prefill logit readout runner and scorer in the new `part69/`
folder. The runner collects unchanged down-projection input magnitudes, chooses the 50
closest activity controls for each single maker while excluding all registered target
and driver units, builds the registered driver matched sets, and records seven maker
logits, seven-token softmax mass, and Google-minus-Alibaba gap for each prompt and
condition. It batches same-prompt rows without padding and uses Part 25's intervention
operation through hooks on every affected layer. Output metadata includes model,
revision, runner SHA-256, device, wall time, peak memory, and counts.

The scorer computes Dg on only the 24 held-out questions, reports the positive control
separately, calculates directional and two-sided empirical p-values with inclusive ties,
and applies the registered 69.1–69.6 verdict rules, including the NOT READ gate.

## Ambiguity resolved

For the driver-null seed formula `6900 + 100 * family + k`, the design calls these
families by their condition roles but does not number them separately. This build uses
family 5 for `l11_down` and family 6 for `l29_up`, matching their registered condition
numbers. The chosen numbers are written to `run.json`.

## Checks

- `python3 -m py_compile part69/part69.py part69/score69.py part69/test_part69.py` passed.
- `python3 part69/part69.py --help` and `python3 part69/score69.py --help` both exited 0
  and displayed usage.
- `python3 -m pytest -q part69/test_part69.py` could not run because this host has no
  `pytest` module. The PyTorch end-to-end and direct-measurement test is designed to
  skip when PyTorch is unavailable; this host also has no PyTorch.
- The model run itself was not run: the host lacks PyTorch and the model runtime.

## Host check before the first run (Claude, 2026-10-26 04:1x CT)

The worker's sandbox had neither pytest nor torch, so none of the tests had run. On the host, 2 of 5 failed, both in
the tests, not the code: `test_verdict_boundaries_and_not_read` asserted that an all-ties null fails 69.5 and 69.6,
but a two-sided p of 1 is above 0.1, so they hold (the scorer was right); and the tiny-model test passed a 1-D input
to the model. Both fixed; 5 of 5 pass. A mutant that installs no zeroing hooks fails the direct-measurement test, so
that test can fail. Checked by reading: the 69.2 mass is the full-vocabulary probability on the seven maker tokens (not
a softmax over the seven alone), and the positive-control question is excluded from the held-out averages.

## First launch crashed (Claude, 2026-10-26 04:12 CT)

`_runtime()` did `import part66`, which binds the part66/ folder as a namespace package, so main() failed at
`part66.DIRECT` (AttributeError) ten seconds in. `test_main_imports_resolve` had passed because `--help` exits before
that line. Fixed to `from part66 import part66`. `test_main_plan_after_argparse_without_the_model` now exercises every
module attribute main() reads after argparse, the question list, Part 66's random sets and make_conditions on the real
Part 68b matched.json; the old import fails it. 6 of 6.
