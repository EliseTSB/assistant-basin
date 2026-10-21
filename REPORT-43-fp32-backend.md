# Part 43 FP32 reference backend report

## SUMMARY
- Replaced FP32 closure-based backend inference with explicit bindings to the reviewed Transformers 5.20.0 Qwen3.5 dispatch callables and source hashes.
- Added pre-import environment and optional-package evidence, cached hub-kernel state checks, and a calibration-bound pre-stage runtime comparison in `run_pilot`.
- Added offline sabotage controls; requested suite passes: 129 tests.
- No model download, GPU, network, or commit was used.

## FINDINGS / WHAT SHIPPED

| Contract point | Implementation | Offline control / result |
|---|---|---|
| Explicit reference primitive | `part43.py` captures the actual decorated module-global callables, their `__wrapped__` torch-reference objects, the Qwen forward and code objects; it checks module globals, instance overrides, wrapper delegates, and source hashes. | `test_fp32_successor_raw_registration_and_reader_pass`, `test_fp32_pilot_refuses_plain_foreign_module_callable_with_unchanged_environment`, `test_fp32_pilot_refuses_swapped_binding_after_calibration`, and `test_fp32_binding_checks_qwen_instance_overrides`. |
| Preparation before Transformers import | `_prepare_fp32_reference_backend` sets `USE_HUB_KERNELS=0`, refuses prior Transformers imports, and records distribution versions, import resolution, and absence attempts for `causal_conv1d` and `fla`. Installed-but-broken or available imports refuse. | `test_fp32_preparation_refuses_installed_but_broken_optional_import`, `test_fp32_preparation_refuses_hub_enabled_at_import_even_after_env_is_zero`; the valid raw fixture confirms both optional paths are absent. |
| Pilot pre-stage comparison | `run_pilot` reads the expected runtime from the reader-produced calibration summary and compares it with current backend and precision evidence after the model is loaded and before `_sequences`. Forbidden TF32, matmul precision, hub, missing evidence, or changed bindings hold before signs. | `test_fp32_pilot_output_labels_gate_as_control_and_stability` passes for the unchanged binding; `test_fp32_pilot_refuses_current_backend_or_precision_drift_before_signs` covers hub setting, CUDA matmul TF32, cuDNN TF32, and matmul precision drift. |
| Effective cached hub state | Runtime evidence includes Transformers’ cached `_TRANSFORMERS_USE_HUB_KERNELS` and `_kernels_enabled`, checked against current environment and bound to the installed source hash. | Cached enabled-at-import sabotage refuses even after the environment changes to zero; unchanged reference-only setup passes. |

### Installed dispatch slots verified

Transformers is 5.20.0. In `models/qwen3_5/modeling_qwen3_5.py`, `Qwen3_5GatedDeltaNet.forward` uses these module globals: `causal_conv1d_update` (line 563), `causal_conv1d_fn` (line 576), `torch_recurrent_gated_delta_rule` (line 612), and `torch_chunk_gated_delta_rule` (line 625). The corresponding decorated reference definitions are at lines 235–287 and 423–424. The forward body is at line 536. With `_kernels_enabled=False`, `hub_kernels.py`’s `use_kernel_forward_from_hub` and `use_kernelized_func` decorators return identity decorators (lines 107–123); the source’s fallback wrapper keeps the torch implementation in the checked delegate slot.

Bound source hashes: Qwen3.5 `b48635f7388fbfd1b214b8c19275f8c9854fd02c4f3a3376a2339b753a2a5457`; `hub_kernels.py` `a4875051ee4f23a723fe769e94b6ad3030284983a02dba03ecf25ec22d114510`. The pilot fingerprint contains stable backend, precision, source, and import evidence; it excludes timings, object addresses, and observation counters. Legacy BF16-v2 metadata representation was retained separately from the FP32 primitive.

### Verification

`/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part43.py` → **129 passed**. `git diff --check` passed. Worktree remains on `part43-fp32`; no commit was made. Parent-control mailbox reads returned no messages, so no dispositions were needed.

## FILES TOUCHED
- `part43.py`
- `test_part43.py`
- `REPORT-43-fp32-backend.md`
- Worker report: `/home/elisepc/green/DiscordBot/df-runs/ab-43-fp32-backend-2026-10-21/worker-01/report.md`

## BLOCKERS
none

## SELF-ASSESSMENT
The tests use the installed Transformers source and CPU tiny models; no Qwen3.5 checkpoint was loaded. The runtime dispatch and hashes were verified against the installed source, but no model-weight pilot was run, consistent with the offline task scope. A future Transformers source update will intentionally refuse until its dispatch layout and hashes are reviewed and updated.
