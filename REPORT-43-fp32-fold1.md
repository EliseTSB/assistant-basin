# Part 43 FP32 successor — fold 1

## SUMMARY
- Folded the round-1 findings into the FP32 successor runtime, reader, CLI gates, and offline tests.
- Added full backend evidence, per-layer down_proj dtype checks for all 32 layers, and a pinned BF16-vs-FP32 diagnostic outside epsilon.
- Offline suite: 117 passed; detached-worktree targeted sabotage matrix: 18/18 regressions turned red.
- No model downloads, GPU, network, commits, or pushes were used.

## FINDINGS / WHAT SHIPPED

| Finding | Fix | Test | Targeted sabotage |
|---|---|---|---|
| Luna MEDIUM / Haiku F6: callable pilot gate | Bind the second gate to the supplied summary protocol, require it to agree with CLI selection, check loaded parameter dtype, and persist a second-gate HOLD safely. | Cross-protocol summary, FP32-with-BF16 model, and CLI HOLD regressions. | Removing protocol binding, dtype check, or HOLD handling made each focused test red. |
| Haiku F1 and F7: protocol scope and dtype | `fp32-successor` now permits only `calibrate` and `pilot`; an explicit `--dtype` must match the selected protocol. | Parameterized all 22 other subcommands and both contradictory dtype combinations. | Removing either CLI guard made its regression red. |
| Haiku F2: backend claims | FP32 runtime records hub-kernel state, kernelized classes, and named Qwen3.5 conv and gated-delta implementations. The reader refuses enabled hub kernels, kernelized classes, or non-torch references unless registration names an allowance; the current allowance list is empty. | Reader fixtures cover each refusal and assert the named implementations. | Removing each of the three reader refusals made its focused case red. |
| Haiku F3: down_proj dtype scope | FP32 calibration hooks every layer, records its input dtype, registers layers 0–31, and requires float32 evidence for all of them. | CPU calibration exercises all 32 layer hooks; reader tests cover a missing layer and a BF16 layer. | Restricting hooks to target layers made the writer test red. |
| Haiku F4/F5: header and always-green controls | Configure FP32 precision before stamping the protocol block; tests assert the block and exercise live TF32=True through metadata and reader. The direct gate has a protocol-mismatch fixture. | Header, gate-boundary, and live-flag regressions. | Removing the header block/configuration, gate check, flag capture, or reader refusal made its test red. |
| Haiku F7: protocol meaning and retained diagnostic | Label the FP32 gate as an amended control and stability gate, not an accuracy interval. Recompute the historical BF16-vs-FP32 differences from the SHA-pinned legacy raw receipt and expose them as diagnostic-only, outside epsilon. | Summary and pilot-output assertions; diagnostic values match the pinned receipt. | Omitting the diagnostic or label from the summary or pilot output made each focused test red. |

**Not fixed and why:** The two review items labeled PRE-EXISTING were left unchanged as instructed: `write_json` uses a shared `<name>.tmp` path for concurrent writers, and an OSError while `_read_calibration_artifact` writes its derived summary escapes `CalibrationRefusal` (it still fails closed, but without a HOLD record). No introduced finding in the fold brief remains open.

## FILES TOUCHED
- `/home/elisepc/green/worktrees/ab-43-fp32/part43.py`
- `/home/elisepc/green/worktrees/ab-43-fp32/test_part43.py`
- `/home/elisepc/green/worktrees/ab-43-fp32/REPORT-43-fp32-fold1.md` (this requested report artifact)

## BLOCKERS
none

## SELF-ASSESSMENT
The implementation names the Qwen3.5 functions resolved by the loaded Transformers module and fails closed when names or all-layer evidence are missing. No real Qwen3.5 model or CUDA/Triton execution was exercised because this worker was constrained to offline CPU tests; actual GPU kernel arithmetic remains unmeasured. The cross-method diagnostic uses the unchanged historical raw receipt pinned by SHA-256.
