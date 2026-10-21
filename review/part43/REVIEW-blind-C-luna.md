## Build C

### BLOCKING findings

**C-B1 — The tests can pass when singleton mode actually batches every record.** `tree-C/test_part43.py:434` compares singleton and batched score/capture values but never observes forward batch sizes. Its TinyModel (`tree-C/test_part43.py:107`) is a per-token residual model without batch-dependent numerical behavior, so both paths can become the same batch operation and still match. I changed only the group construction in `execute` at `tree-C/part43.py:800` from singleton slices to `groups = [(0, len(records))]` for every mode. The full suite still reported **13 passed in 6.50s**. This silently replaces the required unpadded singleton reference with a padded batch while retaining singleton labels. The sabotage at `tree-C/test_part43.py:459` proves that teacher-forced scoring calls the function named `execute`; its AST check at `tree-C/test_part43.py:473` confines literal raw calls. Neither observes that singleton forwards have batch dimension one, or that scoring occurs before the next record's forward. The submitted group construction itself is correct; the blocking issue is a demonstrated false assurance from the acceptance tests.

Reproduction, in tree-C using the same CPU environment variables:

```python
import inspect, pytest, part43
src = inspect.getsource(part43.execute)
old = 'groups = [(0, len(records))] if selected_mode == "batched" else [(index, index + 1) for index in range(len(records))]'
assert old in src
src = src.replace(old, 'groups = [(0, len(records))]')
exec(compile(src, '<in-memory-singleton-batching>', 'exec'), part43.__dict__)
raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider', 'test_part43.py']))
```

This mutation changes no raw-call locations, so the AST test would have the same result against a source-file version of the mutation.

**C-B2 — The singleton pilot still aborts on the original batched-logit budget.** `tree-C/part43.py:3032` invokes the old validation check before signs/lattice. That check explicitly batches at `tree-C/part43.py:5068` and raises on the old raw-logit threshold at `tree-C/part43.py:5078`. The common batch-only 0.25 logit-offset probe gave repeated singleton primary-score error **0.0**, but `run_pilot` raised `batched/single logits differ by 0.2500000596046448, tolerance 0.0001`. Thus even the CLI restriction on batched mode does not make the singleton pilot independent of the failed batched reference. The pilot smoke test replaces this function with an always-passing stub at `tree-C/test_part43.py:527`. The old check and failed receipts should remain, but as an independent diagnostic. `run_singleton_validation` itself has no numeric pass/fail budget.

### NON-BLOCKING findings

**C-N1 — An unused norm-input capture helper is not singleton-row aware.** `_capture_norm_inputs` at `tree-C/part43.py:2273` iterates over the entire record list inside each one-record hook invocation and indexes `value[row]`. Calling it on the maker record list raises `IndexError: index 1 is out of bounds for dimension 0 with size 1`. `rg -n '_capture_norm_inputs' tree-C/part43.py` finds only its definition, so no present production result was shown wrong. It still needs the same row-cursor treatment already used by `_path_patch_score` at `tree-C/part43.py:2308` before it can support multi-record singleton capture.

### Requirement coverage

| Requirement | Status | Evidence / limitation |
|---|---|---|
| 1. Shared adapter; all forward/generate sites; mode flag and outputs | Partly met | Raw generation is inside `execute:791`; the sole raw forward is its local callback at `execute:817`. Scoring, captures, generation, surrogate/readout and finite-difference/link paths are routed. Mode is recorded in headers and manifest, and `main:5178` rejects batched mode outside batch-diag. The pilot nevertheless retains a mandatory batched diagnostic gate (C-B2). |
| 2. Singleton isolation, immediate scoring, order, parity assertions and same-shape replay | Partly met | `execute:800` correctly makes singleton groups of one and slices operations; callbacks score immediately with `collect_outputs=False` (`teacher_forced_scores:397`, `_margin_forward:955`); captures preserve original indices and real lengths. `assert_mode_alignment:726` compares IDs/positions/payloads. Production replay captures use the same singleton shapes. The unused norm-input helper does not support this mode (C-N1). |
| 3. Production-estimand validate cells, repeats and natural replay; no budget | Met | `run_singleton_validation:1002` explicitly scores the seven required full answers, forms Google minus the Alibaba-group logsumexp and forces two singleton runs per cell. Clean, selected zero, four-zero, natural restoration, +/-2 dose, L7-zero/L11-restored rescue and receiver-only replay are present, with short/long coverage and score/activation error reports. No numeric acceptance budget is applied. Omitting Tencent here does not change the primary estimand. |
| 4. Separate batch diagnostic with all conditions, scores/deltas and signed receiver h | Met | `_batchdiag_one_condition:1154` runs each answer in the specified one-row or two-row context and assembles the complete primary score afterward. All required conditions have clean/zero scores, deltas and signed receiver h. Same-length companions differ in content (`:1119`); reversed rows track A's index; randomized pads and normal pads use the same +64-token companion. TinyTokenizer probes observed zero padding in A/A and equal-length pairs, one pad in the different-length case, and 64 pads in the extra-long/randomized pair. |
| 5. Environment versions, actual dtypes, kernel/backend and arithmetic flags | Met | `_runtime_environment:106` records torch/transformers, CUDA/cuDNN, package/kernel versions, model-config attention, resolved Qwen function information, parameter dtype counts, observed activation dtypes and precision flags. Runtime headers include the active model. The pre-load manifest explicitly identifies unloaded/unobserved fields. Actual GPU dispatch was not independently measured here. |
| 6. Preserve failed check and receipts | Met | `run_validation_checks:5053` preserves the genuine explicit batch-versus-single diagnostic and original thresholds; receipt audit below. Its pilot-gating role is C-B2. |
| Tests clause | Partly met | All 13 tests pass, record/capture parity and diagnostic fields are exercised, and adapter/raw-call checks exist. The entire suite also passes when singleton groups are replaced with a full batch (C-B1), so singleton execution itself is not protected. |

Fully met: **4 of 7**.

