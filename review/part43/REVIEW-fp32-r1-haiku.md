<!-- cheap-review.mjs cheap-review-part43-fp32-r1-haiku-r1-2026-10-21: /home/elisepc/green/worktrees/ab-43-fp32 79f13a12d5199f073fde96bb52adf593ca4804ea..HEAD, lane haiku, df-run exit 0 -->
# Cold review: part43-fp32-r1 (branch part43-fp32, 79f13a1..18f8c41)

## Verdict: SHIP-WITH-FIXES

Fix F1 and F2 before any GPU pilot runs. F3 must be relabelled (or its scope widened) before anyone claims "fp32 throughout". F4-F7 are follow-ups. The legacy bf16 path is unchanged in behaviour (replayed the real receipt), the 80 suite tests pass, and the core gate arithmetic is correct. The gaps are in what the fp32 label asserts and in which commands the gate covers.

Method: read the full diff (part43.py, test_part43.py, review docs). Ran the suite with the existing CPU torch venv (`/home/elisepc/neuron-probe/.venv`, torch 2.14.1+cpu, pytest 9.1.1): 80 passed. Drove the real `main()`, `output_header`, and reader against scratch copies of HEAD and 79f13a1 (`head/`, `base/`). Ran 8 single-line sabotages on a scratch copy (`mut/`). No live service, no credential file, no write to the worktree. Scripts are in `scripts/`.

## Findings

### F1 MEDIUM, INTRODUCED: ungated subcommands run the FP32 successor with no verdict
- Where: `part43.py:6159` gates only `DECISION_COMMANDS` (5613-5617). `validate`, `checks`, and `batch-diag` are not in that set (command list 6107-6110).
- Trigger (`scripts/ungated_commands.py`, real `main()`, stubbed loader and stages, no calibration file present):
  ```
  signs          gate_consulted=1 model_load_dtypes=[]               written=[HOLD.json]
  validate       gate_consulted=0 model_load_dtypes=['torch.float32'] rc=0 written=[validate.json]
  checks         gate_consulted=0 model_load_dtypes=['torch.float32'] rc=0 written=[checks.json]
  batch-diag     gate_consulted=0 model_load_dtypes=['torch.float32'] rc=0 written=[batch-diag.json]
  ```
- Why: the protocol says no production stage runs before the reader's verdict. With `--protocol fp32-successor`, these commands load FP32 weights and write results with rc 0 and no verdict. The same commands already ran ungated under bf16 (pre-existing), but the successor is what makes FP32 execution reachable there.
- Smallest fix: in `main()`, refuse `--protocol fp32-successor` for any command outside `DECISION_COMMANDS | {"pilot", "calibrate"}`, and add a test.

### F2 MEDIUM, INTRODUCED: the FP32 label does not name the gated-delta or kernel backend
- Where: `part43.py:191` and `:199` (`gated_delta_backend` is a literal placeholder); `:1347-1348` (implementation names reduced to a SHA-256); `:5731-5733` (reader checks only the digest's shape and that backend strings are non-empty).
- Trigger (`scripts/kernel_claims.py`, the suite's own FP32 fixture, real writer and reader):
  ```
  as written: gated_delta_backend = "resolved per Qwen3.5 model function below"
  baseline reader: within_budget 0.0
  with hub_kernels_enabled="1", GatedDeltaNet classes, causal_conv1d and fla versions set:
      within_budget 0.0 part43-fp32-successor-v1 torch.float32
  ```
- Why: protocol section 2 requires that any backend that downcasts internally be "named, not relabelled". The artifact names nothing. Hub kernels and Triton paths are not controlled by the `torch.backends` TF32 flags, so the recorded flags do not establish full-FP32 arithmetic (Alden's item 4). I could not measure Triton or TF32 behaviour here (no GPU), so that part is unmeasured.
- Smallest fix: write `qwen3_5_function_implementations` by name into the FP32 `execution.legs.fp32.runtime`. The reader refuses truthy `hub_kernels_enabled`, non-empty kernelized classes, or any non-torch-fallback implementation, unless a named allowance is in the registration.

### F3 MEDIUM, INTRODUCED: "fp32 throughout" evidence covers 4 of 32 layers' down_proj inputs plus logits
- Where: `part43.py:587` (capture default `UNITS`), `:626` (`for layer_index in by_layer`), `:631-632` (dtype hook), `:855-856` (logits), `:1339` (parameter dtypes only), `:5710` (reader).
- Trigger (`scripts/hook_scope.py`, real `teacher_forced_scores` on a 32-layer tiny model that runs every layer):
  ```
  down_proj forwards: 224 | with dtype hook attached: 28 (4 layers x 7 passes)
  observed dtypes: ['torch.float32']
  ```
  That leaves 196 forwards (28 layers) with no dtype observation.
- Why: the reader accepts `activation_dtypes == [float32]` and the down_proj and logit lists. bf16 compute in an unhooked layer, in attention or the MLP gate/up path, or in a buffer (only `parameters()` are checked) is invisible to it. This is the "artifact says fp32, execution may not be" case the review was asked to hunt. Whether any real Qwen kernel casts is unmeasured.
- Smallest fix: rename the fields to `hooked_*` and state the scope in the registration and summary. For a real claim, hook every layer on the pilot path and record buffer dtypes.

### F4 LOW, INTRODUCED: the `calibration_protocol` block is stamped before FP32 configuration
- Where: `part43.py:216-240` (`output_header`), `:6223-6224` (`write_manifest` runs before `run_calibration_runtime` at `:6231`), `:5389` (`run_score`, a gated command that loads no model).
- Trigger (`scripts/header_before_configure.py`, fresh process, `--protocol fp32-successor`):
  ```
  calibration_protocol: decision_dtype torch.float32, matmul_allow_tf32 false, cudnn_allow_tf32 true
  torch_defaults: cudnn_tf32 true
  ```
- Why: an FP32 pilot's `MANIFEST.json` is written before configuration and never rewritten, so the run directory says FP32 with cuDNN TF32 on. `score` stamps the same block without any FP32 execution. The flag values are honest, but the protocol label is not. No execution effect, since the reader does not read these headers.
- Smallest fix: call `_configure_full_fp32_precision()` inside `output_header` for the FP32 protocol before reading flags, or stamp the block only when configuration has run.

### F5 LOW, INTRODUCED: tests that stay green under sabotage
- Mutation results (`scripts/mutate.py`, one edit to a scratch copy each):
  - A. Remove the `calibration_protocol` header block (`:230`): **80 passed**. No test covers the block.
  - C. Remove the protocol check in `calibration_gate` (`:6055`): **80 passed**. The reader's schema check catches the cross-protocol case first, so the gate's check is not exercised.
  - B. Remove loader configure (`:5410`): 1 failed. Good.
  - D. Remove reader TF32 refusal (`:5749`): 2 failed. Good.
  - E. Remove the run-path configure (`:1531`): 1 failed. Good.
  - F. Drop replay error from epsilon: 2 failed. Good.
  - G. Remove the logit dtype check (`:5710`): 1 failed. Good.
  - H. Disable the DECISION gate in `main()` (`:6159`): 1 failed. Good.
- Also `test_part43.py:297-301` overwrites the recorded TF32 and precision flags with `False`/`highest` after measuring them. No test lets a live TF32=True flow through `_calibration_runtime_metadata`.
- Smallest fix: assert the header block's fields; add a gate-level protocol-mismatch test; add a fixture where the live flag is True before the metadata writer runs.

### F6 LOW, INTRODUCED: the pilot's second gate is protocol-blind
- Where: `part43.py:3494`, `run_pilot` calls `calibration_gate` without `protocol_id`. The first gate in `main()` enforces the CLI protocol, so there is no bypass in the current flow.
- Smallest fix: pass `protocol_id=_selected_calibration_protocol(args)`.

### F7 LOW, INTRODUCED: protocol items unimplemented or mislabelled
- No "amended control and stability gate, not an accuracy interval" label in the summary or pilot output, which ALDEN-23 asked for.
- The successor summary omits any bf16-vs-fp32 cross-method diagnostic. Protocol section 4 is ambiguous here, since the old values are kept only in the old receipts.
- `--dtype` is accepted but never read (`grep args.dtype` returns nothing). `--protocol fp32-successor --dtype bf16` silently runs FP32.
- `part43.py:1337` echoes the constant `expected_dtype` as `dtype_as_loaded`, not a measurement. The `parameter_dtypes` check covers it.

### PRE-EXISTING (follow-ups, not holds)
- `write_json` uses a shared `<name>.tmp` path (`:243-247`). Two concurrent writers to one out dir can interleave. Not measured; the reader should refuse malformed content.
- `_read_calibration_artifact` writes `calibration-summary.json` (`:6037`). An OSError there is not a `CalibrationRefusal`, so it escapes the pilot's handler and crashes rather than writing HOLD. That fails closed, but not with a HOLD record.

## What I checked and found sound
- Suite: 80 passed. Sabotages B, D, E, F, G, H turned red.
- Legacy path, measured: the real 845964c/e1ac714 receipt (`results/part43-pilot-3090/calibration.json`) replayed through base 79f13a1 and HEAD with the real Qwen3.5 tokenizer. Both give epsilon 0.10915956082076228 and gate false, which matches Alden's recompute. Legacy header changes are additive only.
- Loader order: `_configure_full_fp32_precision` runs before the FP32 load in both the run path (`:1531`) and the loader (`:5410`).
- Epsilon: the max covers all six cells' spreads, including replay, plus the replay error. The gate is `<= 0.05` exactly.
- Cross-protocol: schema versions 2 and 3 and the provenance sets refuse in both directions.
- Fail-closed: derived, duplicate, and malformed inputs raise `CalibrationRefusal` and become HOLD. Pilot stages come after the reader verdict (gate at `:6238`, model load at `:6265`, `run_pilot` at `:6266`).
- No `bfloat16`, autocast, or `.half()` appears in FP32 paths. No other module in the repo sets TF32.
- Not measured: real GPU and TF32 behaviour, a real Qwen forward pass, Triton kernel precision, concurrency races.
