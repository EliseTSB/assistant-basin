# Part 43 singleton execution report

## Implementation

`part43.py` now has one shared `execute(model, records, mode, ...)` adapter for model forwards and generation. The default is `singleton`; confirmatory CLI commands reject `--exec-mode batched`. The retained batched route is used for `batch-diag` and the legacy `run_validation_checks` comparison. The adapter applies only each row's intervention, captures activations in source order, checks row and real-position lengths, and can consume each singleton result immediately. `teacher_forced_scores` and `_margin_forward` use that callback path, so scoring does not keep a padded batch of vocabulary logits. Capture-only callers discard logits after the forward.

`--exec-mode {singleton,batched}` is available on the CLI and the selected mode is included in all result headers and manifests. Runtime metadata records PyTorch and Transformers versions, CUDA/cuDNN versions when present, the attention backend, resolved Qwen3.5 recurrent/convolution implementations, FLA/causal-conv1d/kernel-hub versions, parameter and observed activation dtypes, TF32 and reduced-precision flags. A preregistration manifest records that runtime dtypes are not yet observed before model loading.

`validate` runs the production Google full-answer logprob minus `logsumexp(Alibaba, Tongyi Lab, Qwen team)` on the primary maker question. It measures the seven answer strings listed below, two singleton repeats per cell, clean, L7 zero, all-four zero, clean-to-clean trajectory replay, signed ±2x L7 doses, L7-zero/L11-clean rescue, and L11 receiver-only replay from the L7-zero trajectory. It reports per-answer scores, score deltas, repeat differences, and activation replay differences without applying a pass/fail threshold. The seven-string validation list follows the requested count and omits the later-added Tencent competitor; this does not alter the primary estimand.

`batch-diag` separately measures an A-alone singleton reference and the batched diagnostic cells: A alone as a one-row batch, `[A,A]`, equal-length `[A,B]`, reversed order, different-length `[A,B]`, randomized masked pad IDs with a fixed mask, and an extra-long companion. Each cell reports the full-answer primary score, selected-zero delta, and L11/L29 signed stem activations, plus answer-level differences from singleton and shape/pad records.

## Execution route ledger

The only raw model invocation sites are inside `execute`: `model.generate(...)` at line 791 and `model(input_ids=...)` at line 817. All calls that score or capture model state route through these adapter entry points:

- Direct adapter consumers: `teacher_forced_scores:397`, `captured_forward:883`, `_margin_forward:955`, and `_batchdiag_one_condition:1212`.
- Generation wrapper callers: `_generate_zero_subset:1767`, `_generate_position_patch:1869`, `_generate_scoped_zero:2564`, `_generated_answer:2798`, and `run_dose:4420`.
- `captured_forward` callers: `run_signs:1337`; `_capture_primary_surrogate:1629`; `_capture_full_unit_input:1919`; `_capture_norm_inputs:2282`; `_mean_activation_by_position:2604`; `run_baselines:2673`; `_corpus_max_abs:2766`; `run_grid:2926`; `run_mediate:3124`; `run_shift:3170, 3173`; `_capture_link_state:3445`; `_receiver_bypass_value:3481`; `run_swap:3566, 3571, 3578, 3582, 3626`; `run_attach:3753`; `run_links_amended:3866, 3867, 3877`; `run_readout_checks:4079, 4107, 4108, 4134`; `run_sweep:4265`; and legacy diagnostic `run_validation_checks:5068, 5072, 5085`.

The adapter also checks record IDs, answer positions and token IDs, stem positions, operation coordinates and values. It verifies capture rows map back to the original record and that each captured unit has one value per real token. `batch-diag` compares those alignments and reports selected receiver differences against the singleton reference.

## Verification

- `/home/elisepc/neuron-probe/.venv/bin/python -m py_compile part43.py test_part43.py` — passed.
- `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q test_part43.py` — **13 passed** on CPU with PyTorch `2.14.1+cpu`.
- `git diff --check` — passed.
- Tests cover mode order/score/activation agreement on a tiny model, adapter sabotage and raw-call AST guard, `validate`/`batch-diag` output fields, and the existing `test_part43.py` cases.
- No model or GPU experiment was run. Existing failed batched receipts and pilot outputs were not modified.

## Forward-pass commands

Run from the Part 43 repository root. The commands assume the pinned model snapshot is in the listed cache path.

### A770 (Windows, XPU)

```bat
python part43.py validate --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --device xpu --dtype bf16 --exec-mode singleton --cache-dir "E:\neuron-probe\hf\hub" --out results\part43
python part43.py batch-diag --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --device xpu --dtype bf16 --exec-mode batched --cache-dir "E:\neuron-probe\hf\hub" --out results\part43
```

### RTX 3090 (CUDA)

```bash
python3 part43.py validate --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --device cuda --dtype bf16 --exec-mode singleton --cache-dir /mnt/e/neuron-probe/hf/hub --out results/part43
python3 part43.py batch-diag --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --device cuda --dtype bf16 --exec-mode batched --cache-dir /mnt/e/neuron-probe/hf/hub --out results/part43
```

## SUMMARY

Singleton execution, score-level validation, separate batch diagnostics, output environment metadata, and CPU tests shipped. No GPU validation was run.

## FINDINGS

The requested CPU suite passes. Runtime correctness on the A770 and RTX 3090, including actual backend selection and measured score-level repeatability, remains unmeasured until those commands run.

## BLOCKERS

None for the scoped code task. The later GPU pilot remains subject to its separate review and run authority.
