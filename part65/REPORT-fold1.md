# Part 65 fold 1 report

## SUMMARY

Fold 1 fixes the round-1 review findings across steering, judge transport, analysis, tests, and the A770 runbook. The implementation follows the registered design and Amendment 1. No GPU job or live API call was made.

## FINDINGS

- **H1** — `judge65.py` now uses `~/.mitmproxy/ca-bundle-combined.crt` when present, otherwise Python's default SSL context, passes `context=` to `urlopen`, prefers `ANTHROPIC_CREDITS_KEY`, and names both accepted key variables in its error.
- **H2** — `steer65.py` defaults to the canonical Part 60 corpus, with `PART64_CORPUS` then `PART60_CORPUS` overrides; `--collection` and `--corpus` remain available, and `RUNBOOK-65.md` gives explicit Windows paths for the A770 run.
- **M1** — Pushed generations record hook call counts and layer-module names; `process_unit` rejects zero-fire or wrong-layer hooks, stores each unit's +0.8 changed-rollout count, and computes dose-zero success from a separate unsteered generation.
- **M2** — `judge65.py` now labels the negative steer-lens tokens as made more likely, consistent with the top-12 increases retained by `steer65.py`.
- **M3** — Judge parsing accepts only a lone A/B with optional surrounding whitespace and one trailing period or close parenthesis; analysis excludes invalid trials, counts units with no valid trial, and exits with a clear error above 5% unparseable trials.
- **M4** — Added two-layer layer-1 steering, hand-set control-vector, separate dose-zero comparison, and seven-prediction planted-null coverage; all seven planted-null predictions are asserted false.
- **65g** — `analyze65.py` reports whether middle-band neuron agreement is below both other bands, includes the three band bootstrap estimates and intervals, and marks 65c/65g exploratory.
- **L1** — `steer65.py` and `judge65.py` ignore an invalid final nonblank JSONL line with a filename warning and continue to raise for earlier invalid lines; tests cover both readers.
- **L2** — `generate_batch` requires an explicit `hook_layer` whenever pushes are supplied; the private layer-0 fallback is removed.
- **L3** — `judge65.main` prints every released chunk and exits nonzero if any chunk was released.
- **L4** — Analysis names the registered observed-cosine difference used by 65f and reports the null-adjusted neuron-minus-control measure beside it in JSON and Markdown.
- **L5** — No change; the existing ledger policy for 429 responses remains in place, with no automatic release or resubmission.
- **M4 test-to-mutant evidence** — `test_layer1_push_fires_on_layer1_and_changes_output` rejected force-layer-zero; `test_control_direction_is_normalized_w_down_times_hand_set_r` rejected W-down-column control vectors; `test_dose_zero_check_compares_with_a_separate_unsteered_generation` rejected constant-true dose-zero status; the planted-null test rejected lowered-65c-threshold and force-65g-pass mutants. Each of those five isolated mutations failed its targeted test.
- **Other regression tests** — `test_pushed_generation_rejects_a_hook_that_never_fires` catches a dead hook; `test_pilot_check_uses_recorded_run_wide_gates` catches removal of the +0.8 pilot threshold; parser, TLS/key, lens-label, JSONL-tail, released-chunk, explicit-layer, and missing-trial tests cover the corresponding faults described above.
- **Verification** — `/home/elisepc/neuron-probe/.venv/bin/python` reported torch 2.14.1+cpu; `PYTHONPATH=/home/elisepc/green/worktrees/ab-part65 ... -m unittest test_part65` passed all 23 tests. Temporary test directories use the system default; paid-batch lifecycle tests use a local fake ledger to avoid writing to the shared paid ledger registry.
- **Control messages** — No parent-control messages were received, so no message changed the plan or was declined/deferred.

## BLOCKERS

None for the scoped code and documentation. Live Anthropic transport and A770/XPU behavior remain unverified because the brief forbids paid API calls and GPU runs.

## Claude, after collection (2026-10-24)

Host run from `part65/` with the worktree on the path: 23 tests, OK. Three mutants of my own, each restored by hash afterwards: a hook that returns its input unchanged fails `test_layer1_push_fires_on_layer1_and_changes_output` and `test_pilot_check_uses_recorded_run_wide_gates`; `process_unit` pushing at layer 0 fails the same two; the old "less likely" lens label fails `test_negative_steer_lens_summary_says_more_likely`.
