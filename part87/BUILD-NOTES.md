# Part 87 build notes

## What shipped

- Three-phase runner: measure, pilot, run. Measurement teacher-forces the 192 unchanged Part 66 held-out answers and records all neuron means in layers 6, 7, 11, and 29.
- Calibration residual medians are recorded for those four layers and layer 22; selected down-projection column norms, target M/F doses, and both rules for all 20 layer-matched random sets are stored in doses.json.
- Doses are written atomically before generation and locked by a SHA-256 sidecar. Pilot and run require that lock; run also requires a complete 44-answer pilot with matching revision, dtype, device, dose hash, and source hashes.
- Every arm has answers.jsonl and an arm manifest. Checkpoints replace each arm file atomically; resume refuses changed provenance.
- The Part 84 AddHook/positions logic is extended at runtime for every-token mode. Part 66’s zeroing hooks, Part 25’s matched-random sampler, and Part 66’s regex/guard rules are reused.
- score87.py writes shuffled random-ID 87A reader items, keeps the key separate, refuses overwrites, validates labels, scores Part 66 rules, prepares up to 30 hand-read matches per main arm, and emits 87.1–87.9 verdicts with NOT READ for missing/full-run-incomplete inputs.

## Test run

Command: /home/elisepc/neuron-probe/.venv/bin/python -I test_part87.py

Exact test count: 12 tests ran, 12 passed. Each of B1–B6 was applied to a throwaway copy under a temporary part87/ directory; its matching check failed with the diagnostic shown below, so the mutation was caught.

| Break | Failing check and captured message |
|---|---|
| B1 prompt skipped in every-token mode | every-token prompt positions: Lists differ: [] != [0, 1, 2, 3] |
| B2 wrong unit receives the push | named-unit hook: False is not true: hook pushed the wrong unit |
| B3 M dose has wrong sign | mean cancellation: steered mean 8.0; expected 0.0 |
| B4 generation precedes doses file | generation gate: invoked ['generated'] before missing-dose validation |
| B5 random set reuses target unit | random-set validator: random set 8700 reuses a target unit or has the wrong size |
| B6 87.7 uses random mean, not maximum | 87.7 maximum check: mutant returned True for a target below one random-set maximum |

## Open questions and what could not be verified

- The registered posting rule does not specify what to do if 87.3 holds but 87.5 fails. Keep that case with Elise for a decision; the build does not infer one.
- No open question changes the registered arms, samples, doses, or thresholds.
- The neuronary calibration text is external to this checkout. The runbook uses the existing Windows copy at C:\Users\alexa\neuron-probe-gpu\p66basin\neuronary-calib.txt; its SHA-256 is recorded in doses.json when measure runs.
- Per the brief, no Qwen3.5-4B load or A770 generation was run here. Windows job-runner execution, GPU timings, actual medians/doses, and the Haiku labels remain to be produced and checked on the host.
- The 87B matching regex is deliberately the registered lexical rule. The hand-read file is a review aid; the scorer does not turn reader judgments into the Google rate.

