# Part 87 fold 1 notes

## What changed

- `part87/part87.py:26,94` imports and hashes the independent `observe.py` source. The measurement manifest and arm manifests therefore pin the observer implementation.
- `part87/part87.py:234,251` measures each registered layer's residual median against that layer and refuses to measure if the calibration file SHA-256 differs from `results/part84/main/run.json`.
- `part87/part87.py:212,225` writes the audit's expected `mean` dose key. `part87/part87.py:451` lets baseline A0 produce with no dose units; the first real-loop test found the former empty-unit indexing defect.
- `part87/part87.py:492,535,565,661,678,684` runs one `model.generate` call per answer, verifies `obs.last_call_id` changed, records `observed.call_id` and row 0, installs the observer after model loading, and closes it in `finally`. The hidden `--test-only-random-seed` option narrows only the run phase to one random-seed pair for the integration test. The GPU guard default is 14.5 GB at line 743.
- `part87/score87.py:63,83,128` combines shuffled 87A and 87B answers under one random-ID key and validates labels across that combined set while scoring origin counts for 87A. The key carries separate A and B digests.
- `part87/score87.py:269,364,404,462` reports 87.7 readiness for M and F separately, ingests explicit hand-read booleans and stores their passing counts/rates alongside each arm's Google rate, requires `--a-answers-dir`, and prints each arm's 87.7 status.
- `part87/test_part87.py:221,278,285,359,398` covers layer-specific median measurement, both NOT READ cases, hand-read output and the real pilot/run path. The end-to-end test calls the production CLI loop with a tiny local Qwen2 model, its real `generate`, real steering hooks and the real observer; audit87 is called with fixture prompt, width, median and arm overrides.
- `part87/mutate87.py` applies each R1-R13 and S1-S2 mutation in a disposable copy, runs the full `test_part87.py` suite in that copy, and reports the killed property.
- `part87/RUNBOOK.md:7,21,36,48,57,86,96,112,126` has the complete copy list, exact venv commands, pilot/run audit gates, GPU limit, combined blind read, hand-read ingest and Amendment 1 posting case. `part87/ORIGIN-RUBRIC.md` supplies the Part 84 EARTH/OFFWORLD/NEITHER and OK/BROKEN rubric with the registered company-name note.

## Requested host checks

Run with `/home/elisepc/neuron-probe/.venv/bin/python -I -B`:

- `part87/test_part87.py`: `Ran 21 tests in 12.544s` / `OK`.
- `part87/test_audit87.py`: `Ran 18 tests in 470.146s` / `OK`.
- `../test_observe.py`: `Ran 20 tests in 0.589s` / `OK`.
- `part87/mutate87.py`: `MUTATE87: 15/15 killed; all killed`.

## Mutation deaths

- R1 temperature 1.0 — audit saw temperature 1.0 at the call instead of 0.7.
- R2 alien hook on L21 — audit saw the hook on `model.layers.21.mlp.down_proj`, registered only on L22.
- R3 E63 sign reversed — audit measured a negative L22 N2742 prefill change instead of the registered upward change.
- R4 alien hook removed — hook census showed no registered L22 hook.
- R5 Z zeroing decode only — audit saw only decode changes; the design requires prefill and decode.
- R6 E63 decode only — audit saw only decode changes; the design requires prefill and decode.
- R7 C63 steered N2742 — audit observed channel 2742 instead of the registered N8870.
- R8 plain prompt — audit decoded `User: ...\nAssistant:` instead of Qwen's registered chat template.
- R9 RF used M doses — audit measured an L6 write norm around 0.0278 instead of the registered F write norm 2.25.
- R10 greedy generation — audit saw `do_sample=False` instead of sampled generation.
- R11 seed zero — audit saw CPU seed 0 instead of the registered 84000 for the first A0 sample.
- R12 all residual medians taken from L22 — the measurement unit test caught repeated L22 layer objects where the registered L6/L7/L11/L22/L29 objects were required.
- R13 F dose sign reversed — the dose unit test got +0.75 for a positive mean instead of -0.75.
- S1 missing RM_8700 ignored — scorer returned HOLDS where the missing random arm requires NOT READ.
- S2 missing origin labels ignored — scorer returned FAILS where absent origin counts require NOT READ.

## Oracle assessment and limits

I did not edit `audit87.py`, `test_audit87.py`, `observe.py`, `test_observe.py`, or `part86/audit86.py`. I found no oracle defect in the requested interface or checks. I did not run measurement or a production Qwen3.5-4B A770 pilot/full study on this Linux worker; the calibration text and GPU execution are part of the Windows runbook. The real loop was exercised on CPU with the required tiny model and observer fixtures.

## SUMMARY

The real generation path now writes independently observed call claims and its pilot plus one-seed random run passed audit87 in the end-to-end test. All 15 requested mutations were killed.

## FINDINGS

The end-to-end pilot exposed and fixed the empty-dose lookup in A0. Scorer readiness, combined blind preparation, calibration provenance, hand-read counts and runbook gates are covered by implementation and tests.

## BLOCKERS

None for the requested code and test work. Production Qwen3.5/A770 measurement and run were not performed in this worker environment.
