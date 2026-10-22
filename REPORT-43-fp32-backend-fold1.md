## SUMMARY

- Added a narrow, fresh-process fingerprint derivation for selected code and bindings in the pinned Transformers 5.20.0 Qwen3.5 implementation. Runtime checks compare those selected items with checked-in constants at calibration and pilot pre-stage; this is not an integrity claim for all executed code or numerical accuracy.
- Closed the reported binding gaps with an allowlist for loaded module types, instance/class dispatch slot checks, module-global reference checks, and an eager attention requirement set during model load.
- Restored the BF16 metadata rows to the exact 4b4ac84 shape and digest. Added import-absence, effective hub-state and mixed-TF32 refusal controls.
- All 144 tests pass. All 12 targeted scratch-copy sabotage cases made their associated test fail when that check was disabled.

## FINDINGS

### Implementation

`review/part43/derive_fp32_fingerprints.py` imports the installed pinned package in a fresh interpreter and derives SHA256 pins from `co_code`, recursively serialized `co_consts` (including nested code objects), `co_names`, `co_varnames` and `co_freevars`. It pins the Qwen module's functions and class methods, the four decorated dispatch functions and their torch references, referenced module globals, and the relevant `torch.nn` module class slots. Function-object fingerprints include defaults and closure contents. `review/part43/fp32_fingerprints.py` contains the derived constants; `test_fp32_pinned_fingerprints_match_fresh_interpreter_derivation` verifies equality against a fresh run.

`part43.py` checks those pins before first binding capture and again whenever calibration or pilot-pre-stage runtime state is recorded. It verifies referenced Qwen module globals, dispatch wrappers and closures, loaded module types, and the instance/class dictionaries for `__call__`, `_call_impl`, `_wrapped_call_impl` and `forward`. The runtime record's `dispatch_is_reference`, `dispatch_globals_match` and `override_detected` fields are populated from those checks. Attention is loaded with `attn_implementation="eager"`: the pinned Qwen implementation exposes `eager_attention_forward`, and runtime/record validation refuses a config that selects a different backend.

`part39.load_model_direct` accepts an optional attention implementation and passes it through to Transformers. FP32 loading requests eager explicitly. The default path is unchanged for BF16 callers.

The legacy BF16 metadata row now has exactly `resolved_implementation`, `torch_reference_implementation` and `uses_torch_reference`; its fixed fake-environment digest matches 4b4ac84 (`77818fd8a0d0371158c9f6ed643f912d687cd1e012b1aa81f2f4204ab30dc58a`). The two ModuleNotFoundError cases refuse when an installed distribution or resolvable module spec contradicts absence. A separate test reaches `_fp32_hub_import_state`'s effective-state check after preparation and binding. Mixed TF32 APIs are converted from `RuntimeError` to `CalibrationRefusal`; pilot records HOLD, and the calibration CLI writes a HOLD record.

### Probe results before and after

| Probe | Before | After |
| --- | --- | --- |
| `probe_initial_binding_spoof.py` | Accepted the foreign wrapper as baseline: runtime and metadata accepted, `dispatch_is_reference=True`, actual result `FOREIGN_SENTINEL`, pilot `complete` after reaching `sequences` and `signs`. | Refuses before binding capture with `CalibrationRefusal: FP32 pinned Qwen code fingerprints mismatch at dispatch:causal_conv1d_fn`. The added regression test asserts refusal in calibration and HOLD at pilot pre-stage. |
| `probe_binding_gaps.py` | The clean control and all four gaps (`l2norm_global`, instance `_call_impl`, foreign `layer.norm`, class `_call_impl`) passed runtime verification; the foreign paths ran. | Control passes with `override_detected=False`; each gap refuses: the global and class changes fail the pinned fingerprint, the instance slot is refused, and the foreign norm is outside the pinned class allowlist. The parametrized regression test asserts calibration refusal and pilot HOLD before signs for every gap. |
| `probe_legacy.py` | Digest was `a378a3f6da5501f44626f366bdcbe00f21041fbc360e7315e6a76ff430f79a80`; the two legacy fields `torch_reference_implementation` and `uses_torch_reference` were missing. | Rows have the 4b4ac84 shape and digest `77818fd8a0d0371158c9f6ed643f912d687cd1e012b1aa81f2f4204ab30dc58a`. |
| `probe_precision_read.py` | Mixed TF32 APIs escaped as an uncaught PyTorch `RuntimeError`. | Prints `CalibrationRefusal (HOLD path): FP32 precision state is unreadable...`; regression tests cover pre-stage HOLD and calibration CLI HOLD output. |

The initial-spoof probe was run from a `/tmp` copy with the worktree on `PYTHONPATH` because its success path writes outside the declared worker scope. On the fixed implementation it refuses before reaching that write. Probe commands and regression tests use the installed `/home/elisepc/neuron-probe/.venv` Python.

### Sabotage audit

Each check below was disabled in a separate `/tmp` scratch copy; its named test then failed as expected. No sabotage stayed green:

- Code pin check — `test_code_spoof`.
- Referenced global identity — `test_global_l2norm`.
- Module type allowlist — `test_module_type`.
- Instance dispatch slot check — `test_instance_slot`.
- Class dispatch slot check — `test_class_slot`.
- Attention backend check — `test_attention_state`.
- Legacy BF16 metadata — `test_legacy_digest`.
- Optional import absence proof — `test_missing_proof`.
- Effective hub state — `test_hub_effective`.
- Mixed precision refusal conversion — `test_precision_read`.
- Eager attention load option — `test_attention_load`.
- Static fingerprint constants — `test_fingerprint_constants`.

Verification: `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part43.py` — **144 passed**. `git diff --check` passed. No commit was made. Parent-control mailbox reads returned no messages, so no plan changes or dispositions were needed.

Files changed: `part39.py`, `part43.py`, `test_part43.py`, `review/part43/derive_fp32_fingerprints.py`, `review/part43/fp32_fingerprints.py`, and the existing `review/part43/probes/probe_binding_gaps.py` / `probe_precision_read.py` controls. This report is the requested repo-root report.

## BLOCKERS

None. I did not load a real pretrained Qwen checkpoint or run a GPU pilot; runtime behavior is verified with the pinned installed source, CPU fixtures, the supplied probes and the full test suite. That leaves actual checkpoint module-graph compatibility as the main unverified deployment detail.
