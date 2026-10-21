<!-- cheap-review.mjs cheap-review-part43-fp32-r2-haiku-r2-2026-10-21: /home/elisepc/green/worktrees/ab-43-fp32 18f8c41..HEAD, lane haiku, df-run exit 0 -->
# Cold review: part43-fp32-r2

Range reviewed: `18f8c41..a8a2a9e` (branch `part43-fp32`, worktree `/home/elisepc/green/worktrees/ab-43-fp32`).
Measurement harness: `measure_closure.py`, `measure_legacy.py`, `measure_kernelized.py` in this directory. They import the REAL `part43.py` and drive `_runtime_environment`, `_calibration_runtime_metadata`, `_validate_runtime_metadata`. Only torch and the transformers Qwen3.5 module are stubbed (attributes only). Transformers and torch are NOT installed here, so nothing was run on real Qwen3.5 weights.

## Verdict: SHIP-WITH-FIXES

Blocking: F1 must land before the FP32 pilot is run or this branch is merged. The reader's refusal of non-torch Qwen3.5 implementations can be bypassed by a plain module-level kernel function, and the artifact then records that kernel as the torch reference. The fold report's "no introduced finding remains open" is not correct. F2 to F5 are real but smaller. F3 and F4 are conditional on the transformers layout, which I could not check here.

The FP32 test suite was NOT re-run. `import torch` and `pytest` are absent from every interpreter I could find, and pip has no index. The "117 passed" figure is the author's, not mine.

## Findings

### F1 (INTRODUCED, blocking): plain kernel functions are reported as torch reference and pass the reader

`part43.py:180-194` (`_runtime_environment`). When a module-level function has no closure, `inspect.getclosurevars(...).nonlocals` gives `{}` (line 186). `implementation` and `torch_fallback` are then both `None`, so `selected` and `reference` both become the function itself, and line 194 sets `is_fallback = True`. A plain kernel imported straight into the module namespace is therefore recorded as `uses_torch_reference: true`. The old code had the same flaw, but it only recorded the flag. The new validator enforces it (`part43.py:5880-5888`), so the refusal is now bypassed.

MEASURED (`python3 measure_closure.py <worktree> plain_kernel wrapped_kernel torch`):
```
== mode=plain_kernel
   causal_conv1d_fn: {'resolved_implementation': 'causal_conv1d.kernel_causal_conv1d_fn', 'torch_reference_implementation': 'causal_conv1d.kernel_causal_conv1d_fn', 'uses_torch_reference': True}
   validator -> ACCEPTED
== mode=wrapped_kernel      (closure names implementation + torch_function: control)
   causal_conv1d_fn: {'resolved_implementation': 'causal_conv1d.kernel_causal_conv1d_fn', 'torch_reference_implementation': '...torch_causal_conv1d_fn', 'uses_torch_reference': False}
   validator -> REFUSED: FP32 successor refuses non-torch-reference implementation causal_conv1d.kernel_causal_conv1d_fn
== mode=torch              (control)
   validator -> ACCEPTED
```

Why it matters: the artifact claims torch-reference execution while a kernel ran, which is exactly the "executes differently while the artifact says otherwise" case the brief asks about. It is also why the non-torch refusal test does not catch it. `test_part43.py:258-268` builds every Qwen function as a torch-shaped plain function, and `test_part43.py:1414-1433` forges `uses_torch_reference=False` into the raw JSON after the fact. Neither exercises the introspection.

Smallest fix: decide provenance, not closure presence. For example:
`uses_torch_reference = implementation is None and getattr(function, "__module__", None) == qwen_module.__name__ and (torch_fallback is None or torch_fallback is function)`.
Add a test that installs `causal_conv1d_fn` as a plain function with `__module__ = "causal_conv1d"` and asserts refusal from the introspection itself, not from a forged row.

### F2 (INTRODUCED, MEDIUM): torch-fallback configuration is refused as "missing fields"

`part43.py:181-182`: `if function is None: continue`. If the causal-conv1d module global is `None` (the kernel package absent, so the model runs its torch conv path), the key is omitted and the FP32 validator refuses.

MEASURED (`measure_closure.py ... none`):
```
   causal_conv_backend=''
   validator -> REFUSED: fp32 Qwen3.5 function implementations is missing field(s): causal_conv1d_fn, causal_conv1d_update
```

Why it matters: this is a false refusal, not a silent pass. It means the FP32 pilot cannot run at all in the torch-fallback environment, and that is not what the protocol's "torch reference" wording says. It is also conditional on the transformers version setting these globals to `None`.

Smallest fix: when the function is `None`, record an explicit row with `resolved_implementation: "none (torch conv path)"` and `uses_torch_reference: true`, and let the validator accept it.

### F3 (INTRODUCED, MEDIUM, conditional): gated-delta introspection reads module globals, not what the forward uses

`part43.py:70-73` (`FP32_QWEN_FUNCTION_NAMES`) lists only the module-level `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule`. The Flash Linear Attention kernel names (`chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`) are never inspected (grep finds them nowhere in `part43.py`). The check also reads module globals (`part43.py:180`), not the instances that the forward actually calls.

By reading only: if the transformers modeling module binds the instance forward to the FLA kernel when FLA is installed, the artifact would still record the torch name as the implementation and pass. I could not confirm the binding because transformers is not installed here. This is a hypothesis, not a measured defect.

Smallest fix: inspect the attributes the instance binds (for example `self.chunk_gated_delta_rule` on each `GatedDeltaNet` instance from `model.modules()`), and record those.

### F4 (INTRODUCED, LOW-MEDIUM, conditional): kernelized-module detection is name-based

`part43.py:171`: `kernelized_module_classes` only matches `"GatedDeltaNet" in type(module).__qualname__`. Hub-kernel use is evidenced only by the `USE_HUB_KERNELS` environment variable (`part43.py:224`).

MEASURED (`measure_kernelized.py`):
```
USE_HUB_KERNELS=None: kernelized_module_classes=[] hub_kernels_enabled_effective=False
USE_HUB_KERNELS='1': kernelized_module_classes=[] hub_kernels_enabled_effective=True
```
A kernelized module with another class name is not recorded. If the loader can enable kernels some way other than that variable, the reader accepts. Conditional on the loader; I did not verify it.

Smallest fix: record every module class whose `__module__` is outside `torch.*` and `transformers.models.*`, and refuse any of them unless allowed.

### F5 (INTRODUCED, LOW): legacy bf16-v2 metadata changed for an identical environment

Same stubbed environment, legacy protocol (`measure_legacy.py`):
```
OLD 18f8c41: qwen_implementation_sha256 = 8c9525fd61854057bedad0e249b1c7052df7d84b5da9bf7c35fca6d41fca6118
            gated_delta_backend = 'resolved per Qwen3.5 model function below'
NEW HEAD:    qwen_implementation_sha256 = 8441ae5a6c51af20b51c0bdc4709ee50f74f766086ab589ca65e1e5b94843828
            gated_delta_backend = 'causal_conv1d_fn=...; torch_chunk_gated_delta_rule=...'
```
Cause: the `resolved`/`uses_torch_reference` key renames and the `selected` change (`part43.py:186-203`, `_calibration_runtime_metadata` ~1381-1430). The legacy gate reads this field only as 64-hex (`_validate_runtime_metadata`), so no legacy gate result changes. Legacy artifacts from before and after are no longer comparable on that field. The brief asks that legacy behaviour be unchanged.

Smallest fix: keep the old hash inputs and old `gated_delta_backend` text for `CALIBRATION_PROTOCOL_ID`, or state the change in the protocol file.

### F6 (PRE-EXISTING, LOW): shared `.tmp` path in `write_json`

`part43.py:264-271` writes `<name>.tmp` and then `replace`s it. Two concurrent calibrate or pilot runs can interleave on the same tmp file. The fold report already records this. The old `write_json` at `18f8c41` is identical. The reader fails closed on the resulting malformed JSON, so this is not a silent pass. Follow-up, not a hold.

### F7 (PRE-EXISTING, LOW): OSError from the summary write escapes the gate

`_read_calibration_artifact` writes `calibration-summary` (`part43.py:6253-6273`), and the calibrate/pilot branch of `main()` catches only `CalibrationRefusal`. An OSError there skips the HOLD record, and an earlier `pilot.json` or `HOLD` file can remain on disk. The fold report already records this. It is reachable from the FP32 path too, but the same code predates this change. Follow-up.

## Checked and found sound

- CLI scope: `--protocol fp32-successor` is refused for every command except `calibrate` and `pilot` (`main()`), and a contradictory `--dtype` is refused. `calibrate` is not in `DECISION_COMMANDS`, so it does not hit the HOLD check before it runs.
- Dtype: FP32 forces `torch.float32` in `_load_runtime`, in `run_calibration_runtime`, and in the pilot's second load. `run_pilot` checks model parameters against the calibration dtype before running. `_run_calibration_leg` checks parameters too. I found no autocast or bf16 cast anywhere in `part43.py` (grep).
- TF32: `_configure_full_fp32_precision` runs before the load. The validator refuses any recorded `True`, `None`, or non-`highest` precision (`part43.py:5914-5917`).
- Epsilon: the spread is taken over all six decision cells plus the replay cell, and the replay error is a separate component. Both feed `max` (`part43.py:6116-6118`). The comparison is `<=` 0.05 (`6134`, `6305`). No cell is excluded.
- Raw boundary: the FP32 reader rejects schema 2 artifacts and artifacts with no `protocol_id`/`decision_dtype` keys. It requires `legs` to be exactly `{fp32}`, requires repeat counts of at least 3 with matching list lengths, and rejects derived fields (`_exact_mapping`). The legacy reader rejects schema 3 artifacts.
- Per-layer dtype: all 32 `down_proj` hooks are registered (`part43.py:653`, `697`), and the reader requires exact layer keys with float32 values only. Hooks are removed (`722`).
- Pilot: it always re-runs calibration and reads the file back before the gate (`main()`), so it never trusts a stale calibration file.
- Diagnostic: the pinned receipt SHA matches (`11293803…`). The recomputed maximum difference is 0.10916, equal to the protocol's stated value, and `used_for_epsilon` is False.
- Legacy load: `part39.py:487` still defaults to bf16 when no dtype is passed.
