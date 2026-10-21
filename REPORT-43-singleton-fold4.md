# Part 43 singleton fold 4 report

## Raw calibration schema

`calibration.json` now uses schema version 2 (`part43-calibration-raw`). Its closed top-level fields are `schema_version`, `artifact_type`, `model`, `revision`, `source_provenance`, `tokenizer`, `execution`, `registration`, `registration_sha256`, and `legs`. Registration records bind the pinned model and revision, versioned scoring protocol, tokenizer snapshot and chat-template identity, seven registered answer IDs, each joined teacher-forced `input_ids`, `answer_positions`, target IDs and lengths, and the exact seven cell/intervention definitions with their token IDs. Execution metadata records singleton mode, requested and loaded devices, loaded parameter and observed activation dtypes, torch and transformers versions, backend identity, and TF32/reduced-precision flags.

Each leg contains only its registered repeat count and cells. Every repeat contains exactly one `answer_logprobs` map with the summed log probability for every registered answer. The writer stores no primary score, spread, bf16/fp32 difference, replay error, epsilon, or verdict. Unknown keys and duplicate JSON keys are refused. The reader rebuilds registration from the pinned tokenizer, validates the serialized artifact, then computes per-repeat primary scores, cell means, spreads, cross-dtype differences, replay error, epsilon, and the exact `epsilon <= 0.05` verdict. It writes `calibration-summary.json` as schema version 1 with `derived: true`, the raw artifact SHA-256, and reader/registration identity, including over-budget summaries. The pilot uses the in-memory reader result; gates do not read the summary file.

## Sabotage results

All four required sabotages ran on isolated `/tmp` scratch copies. Each targeted control went red:

- **(a) Verdict reads a stored number other than answer sums:** changing the reader to use stored `bf16.repeat_count` as epsilon made `test_calibration_reader_uses_exact_budget_boundary[0.05-True]` fail (computed epsilon became `3.0`).
- **(b) Duplicate-key check removed:** `test_reader_refuses_duplicate_json_keys_at_parse_time` failed because no refusal was raised.
- **(c) Derived-key refusal removed:** `test_raw_artifact_refuses_every_derived_field[epsilon]` failed because no refusal was raised.
- **(d) Internal errors caught as malformed:** `test_reader_propagates_internal_runtime_error_identity` failed because the injected `RuntimeError` was converted into `CalibrationRefusal`.

The normal exact-boundary controls pass at 0.049 and 0.05 and refuse 0.051. The full requested suite passed: 52 tests.

## SUMMARY

Replaced fold-3 calibration storage and gating with a versioned raw-only artifact and a duplicate-aware, registration-bound reader. The pilot now reaches signs/lattice/swap/attach only after the serialized artifact passes the reader verdict. No commit was created.

## FINDINGS

The saved answer sums are the only measurement authority. Registration token IDs/counts are checked against a fresh tokenizer-derived expectation; score values reject bools, strings, nonfinite values, and conversion overflow. Unexpected reader exceptions propagate. CPU tests cover the serialized read-to-pilot decision path and the required boundary/refusal cases.

## BLOCKERS

none
