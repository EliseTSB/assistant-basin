# Part 68 build notes

## Audit and fixes

- The attribution target hook detached the target activation before taking its mean. That severed the graph, so the
  backward pass could not compute `dt/da`. It now keeps the target tensor connected to autograd for attribution.
- The search did not freeze model parameters or explicitly enable gradients on captured activation tensors. It now sets
  `model.requires_grad_(False)` after loading, enables gradients only around each attribution pass, and marks the
  captured `down_proj` inputs as requiring gradients. It sums `-a * grad` over positions and frees the graph when that
  prompt's attribution call returns.
- The null sampler originally selected only 20 units total (one per candidate), despite the required 200. It now
  allocates 200 draws over candidate layers in their observed proportions using deterministic largest-remainder
  rounding, draws without replacement using `default_rng(6800)`, and excludes candidates and the four maker neurons.
- The original test file imported `pytest.importorskip('torch')` at module scope, which skipped even the pure split,
  sampling, and scorer tests on a host without torch. Split and sampling helpers now live in `core68.py`, which has no
  torch dependency; only the tiny-model tests skip when torch is absent.
- The original 68.6 calculation assigned every null unit attribution 0 instead of reading its measured search
  attribution. The scorer now correlates search-half Delta with the stored attribution for every candidate and null.
- A zero attribution could accidentally count a negative Delta as having the predicted sign. Zero attribution now
  never replicates.
- Part 68 duplicated the reviewed row-hook and target-column logic. Test-phase capture now imports Part 67's
  `_activation_hooks` and `_unit_column`; interventions still use Part 25's `_row_intervention` through that helper.
- The scorer's partial-run Spearman and Markdown paths assumed both halves were present. They now tolerate the
  intentionally incomplete synthetic test fixture. Per-target candidate and replicating layer histograms are recorded.
- The reading gate now marks claims `NOT READ` if 68.5 fails, as required by the design. The failed 68.5 control itself
  remains visible.

## Checks

- `python3 -m py_compile part68/part68.py part68/core68.py part68/score68.py part68/test_part68.py` passed.
- Manual pure-helper check returned the registered 12/13 prompt split and a 200-unit 50/50 null allocation for two
  equally represented candidate layers.
- `pytest` could not run: pytest is not installed in this environment. Torch is also not installed, so the two tiny
  torch attribution tests and the row-intervention torch test could not run here. The test file marks only those model
  tests as skipped when torch is unavailable; non-torch tests remain collectable with pytest installed.
- No real Qwen model run was attempted in this worker environment.

## Module assumptions and design choices

- The runner expects the repository root on `sys.path` and the reviewed modules `part25`, `part66.part66`,
  `part67.part67`, and `neuron_report` to be importable. Part 67 is a namespace package (no `__init__.py` required).
- `core68.py` carries the exact Part 66 prompt strings so split/null tests do not need to import torch-backed model
  modules. The model, revision, prompt formatting, and target activation semantics remain sourced by the runner from
  the repository's Part 66/Part 25/Part 67 modules.
- The design specifies layer proportions but not integer allocation when 200 does not divide evenly. Largest
  remainder allocation is used; equal fractional remainders go to the lower layer index, then NumPy samples within
  each layer in sorted layer order from the single seeded generator.
- Prediction 68.3 is read as “more than half” of the replicating L29 N776 candidates are in layers 20–28. If no
  candidates replicate, it does not hold. Prediction 68.6 requires search-half Spearman above 0.3 for both targets,
  matching the two registered target analyses.

## Host fixes before the first run (Claude, 2026-10-26 03:4x CT)

- `part68.py` imported `from part68.core68`, which the script's own folder cannot resolve when run as a script;
  it now imports `from core68`.
- `test_percentile_95_interpolates_and_keeps_ties` asked for 1.0 from `[0]*190 + [1]*10` while its own first line
  requires numpy's linear percentile, which gives 0.05 (position 0.95 x 199 = 189.05 lies between a 0 and a 1).
  The design says only "the 95th percentile"; the scorer keeps numpy's default (linear), and the test now expects
  0.05.
- The synthetic scoring test used targets (1,3) and (3,5), which the scorer treats as a partial run (no verdicts),
  so `predictions['68.1']` raised KeyError. Its comment also said ten positive-attribution units carry Delta 0.5;
  only five do (units 0-9 carry the Delta and only the even ones have positive attribution), so the scorer's count
  of 5 was right and 68.1 must fail. The test now uses the registered targets, checks the five replicating units by
  number, and expects 68.1 to fail.
- **68.5 was written, not measured.** `_attribution` set every unit at or after the target layer to zeros, so the
  control could not fail. It now hooks every layer and takes all attributions from the same backward pass: a later
  unit with no path to t has a None gradient (recorded as 0). The target unit's own entry is the identity
  derivative, which equals -t; it is removed from the array and saved per prompt in the search json
  (`target_self_attribution_per_prompt` beside `target_value_per_prompt`) as a positive control on the gradient
  machinery. That removal is a reading of the design, which did not mention the target's own entry.
  `test_the_68_5_control_can_fail` declares layer 2 while the sink reads layer 3 and requires a nonzero layer-2
  row; the old written-zeros code fails it (mutant run on the host).

Host: 19 of 19 tests pass, torch tests included.
