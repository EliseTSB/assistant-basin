# Part 43 FP32 successor report

## SUMMARY

Implemented the registered `fp32-successor` calibration and pilot path on branch `part43-fp32`.
FP32 is selected explicitly with `--protocol fp32-successor`; the legacy bf16 protocol remains the default.
The new writer records one FP32 leg with three repeats of every registered cell and full answer set.
The reader derives repeat spread and clean-to-clean replay error from raw per-answer log probabilities.
The pilot reads the serialized artifact back and gates before loading the production pilot model or running stages.
The legacy schema and registered source provenance remain accepted, including the 845964c receipt provenance.
Offline verification passed; no model downloads, GPU, network, or rental were used.

## FINDINGS / WHAT SHIPPED

### Changes by function

- `_runtime_environment`, `output_header`, `_capture_forward_group`, and `execute` now record observed down-projection input and logit dtypes alongside model parameter dtypes. FP32 protocol output records the selected protocol and effective TF32 and matmul-precision settings.
- `_calibration_registration`, `_calibration_source_provenance`, `_calibration_runtime_metadata`, and `_calibration_artifact` bind the FP32 decision dtype and new protocol identity into schema version 3. The artifact contains only an `fp32` leg. Legacy registration and schema version 2 remain as before. A fixed source provenance snapshot allows the saved 845964c artifact to pass the legacy source check after this implementation changes `part43.py`.
- `_run_calibration_leg` supports three FP32 repeats for all existing cells, including the full-answer rescue and clean replay controls. `run_calibration_runtime` configures FP32 arithmetic before loading and runs only the FP32 leg under the successor protocol. `_load_runtime` loads FP32 weights for the selected successor protocol and disables CUDA matmul TF32 and cuDNN TF32 while setting float32 matmul precision to `highest`.
- `_validate_runtime_metadata` and `_validate_raw_artifact` enforce protocol-specific closed schemas, execution dtypes, TF32 settings, protocol identity, registrations, and repeat counts. The successor requires at least three repeats (the writer emits three). `_calibration_reader` derives epsilon as the maximum FP32 repeat spread across registered cells and clean-to-clean replay error, with an exact `<= 0.05` gate. `calibration_gate` checks the matching reader and protocol identity.
- `_build_parser` accepts `--protocol bf16-v2|fp32-successor`; the existing disk-read-before-pilot flow remains in place.

### Offline controls and sabotage results

All eleven new test functions were run against the implementation applied to a detached git worktree. Each test turned red under the listed targeted sabotage; the worktree was restored between mutations.

| Control | Sabotage that made it red |
|---|---|
| Genuine FP32 registration passes | Changed the validator's expected execution dtype to bfloat16. |
| More than three repeats are accepted | Changed the minimum-repeat check (`< 3`) to an exact-three check. |
| Legacy 845964c provenance stays readable | Removed the historical provenance entry from the accepted set. |
| Wrong or mixed parameter, hook-input, and logit dtypes refuse | Made the runtime metadata validator return without checking dtypes. |
| Recorded TF32 true refuses | Disabled the TF32 refusal branch. |
| Cross-protocol artifacts refuse | Changed the schema-mismatch refusal into an internal error; the control detected the wrong refusal class. |
| Repeat spread and replay error independently gate | Sabotaged epsilon to `0.0`. |
| Pilot stages wait for serialized reader verdict | Inserted a signs call before reading the raw file. |
| FP32 loader configures precision and dtype before loading | Disabled the FP32 selection branch in `_load_runtime`. |
| FP32 runtime runs one three-repeat FP32 leg only | Disabled the successor branch in `run_calibration_runtime`. |
| Derived, duplicate, malformed, and internal-error distinctions remain | Disabled duplicate-key detection. |

The original required detached-worktree command against the source repository was refused because its `.git` administrative directory is read-only. I used a local shared clone under the permitted output directory, created a detached worktree there, applied the patch, ran the clean suite, then performed the targeted mutations. Temporary clone and worktree files were removed.

### Verification

- `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part43.py` — **80 passed** on the worktree.
- The same command — **80 passed** on the patched detached worktree before sabotage.
- All eleven targeted new-test functions turned red under their matching sabotage mutations in the detached worktree.
- `git diff --check` passed. The active branch is `part43-fp32`.

### Protocol coverage and limits

No protocol requirement in the code-only brief was intentionally left unimplemented. The real 3090 receipt was not loaded end-to-end against a live tokenizer snapshot; network and model downloads were prohibited. Its stored schema, registration contract, and historical source hash are covered by offline tests. The new path was exercised with CPU stub models only; this report makes no claim about a real model or GPU calibration result.

## BLOCKERS

None. No commit or push was made, following the worker instruction not to commit or push.
