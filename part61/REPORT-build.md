# Part 61 build report

## SUMMARY

Part 61's run, analysis, runner, runbook and CPU contract tests are implemented under `part61/`. The run clamps the registered target and ten fixed controls at four write sizes matched by their own `W_down` column norms, with fp32 inference and checkpoints per unit/size. The analyzer calculates per-size pooled symmetric counts, size/count Spearman rho, next-layer residual-norm ratios and the registered 61a-61e decisions. No model was loaded and no GPU, API or network operation was performed in this build task.

## FINDINGS / WHAT SHIPPED

- The seeded candidate pool matches Part 59's 20-unit draw (seed 0); its first three controls are excluded. Part 61 draws ten from the remaining candidates with seed 61. Both lists and excluded indices are written into each run manifest and pinned by tests.
- For a unit `u` and size `s`, the signed clamp is `sign * s * target_pile_maximum * ||W_down[:, target]|| / ||W_down[:, u]||`. The runner reads each projection column from the loaded fp32 model. A test pins this formula and the run-manifest clamp values.
- Each scan validates the saved 1,000-document Part 26 source and uses its first 200 documents. It records an unclamped baseline, each sign/size mean, per-document next-layer residual norms and per-sign document-wise ratios. NPZ and JSON checkpoints support restart after the baseline or any completed unit/size pair.
- Analysis uses the pooled 99th percentile plus the 0.00005 floor independently for each sign and size. The target pool includes all ten controls; each control's null pool leaves that control out. The combined evaluation follows the registered decision rule: 61b plus both models' 1x control-range condition supports a push-size artefact; OLMo above all ten controls means the contrast survives.
- Parent-control mailbox polls returned no messages. No plan-changing, declined or deferred messages were received.

## FILES

- `run61.py`: fp32 measurement, column-norm matching, residual-norm measurement, seeded controls, locking and checkpointed NPZ/JSON writer.
- `analyze61.py`: pooled per-size classifier, Spearman calculation, residual-ratio evaluation, 61a-61e output and combined decision.
- `run61.sh`: preflight, package/GPU inventory, pinned model prefetch, per-model run/analysis, final analysis and success/incomplete archive handling.
- `RUNBOOK-61.md`: selection, scale formula, workload estimate, package, run, archive and retrieval instructions.
- `test_part61.py`: constant and seed pins, real source-function mutation probes, runner tests and a PyTorch stub model writer/reader/resume test.
- `REPORT-build.md`: this build report.

## TESTS

Command run from the assistant-basin root:

```text
python3 -m py_compile part61/run61.py part61/analyze61.py part61/test_part61.py
bash -n part61/run61.sh
python3 -m unittest part61.test_part61 -v
```

Result: 12 tests discovered, 11 passed, 1 skipped. The skipped test is `test_stub_model_writer_reader_archive_and_resume_checkpoint`; the local Python installation does not include PyTorch. That test uses a CPU-only stub tokenizer/model and exercises the real writer, NPZ reader, manifest key match and resume path when PyTorch is installed. The pure column-norm formula test, classifier/evaluation tests, control draw, runner success/failure/preflight tests and five mutation checks ran locally. Mutations of the source sign condition, pooled percentile axis, leave-one-out reference set, null floor and per-unit `W_down` denominator each failed their contract assertion as expected. No test loaded weights or touched a GPU.

## EXTERNAL ASSUMPTIONS

| Assumption | Status | Basis and limit |
| --- | --- | --- |
| Model IDs, revisions, target layers, target units and widths match the registered design. | DOCUMENTED | Pinned by the design and `part26.MODELS`; actual snapshots were not loaded. |
| `p26._load_model_offline(model_id, revision, device, "fp32")` returns a tokenizer/model pair and puts the model in eval mode. | DOCUMENTED | This is the existing Part 26 loader path. Runtime behavior on the rented image was not exercised. |
| `neuron_report.find_text_layers` finds the decoder `ModuleList`, and its blocks expose `mlp.down_proj`. | DOCUMENTED | This is the project helper's documented structural contract. Matching actual Qwen/OLMo snapshot layouts remains UNVERIFIED. |
| `down_proj.weight` uses PyTorch Linear's `[out_features, in_features]` layout, so `weight[:, unit]` is the write direction. | DOCUMENTED | The runner follows that API layout. Actual model wrappers and column norms remain UNVERIFIED until loaded. |
| `NeuronHook` can clamp the down-projection input through a forward pre-hook; the next block's first input argument is its residual stream. | DOCUMENTED | The clamp helper and hook API are used by Part 59B. Actual next-layer hook arguments for both pinned models remain UNVERIFIED. |
| The pinned tokenizers expose `is_fast` and accept `return_offsets_mapping=True`. | DOCUMENTED | Reused from Part 59B's tokenizer path. Runtime tokenizer availability and tokenization on the rented image remain UNVERIFIED. |
| `p26._padded_inputs` produces input IDs plus an attention mask, and `_forward_decoder` accepts `input_ids`, `attention_mask` and `use_cache=False`. | DOCUMENTED | These are the current Part 26 helper calls. Compatibility with the exact installed Transformers/model revisions remains UNVERIFIED. |
| Source JSON has Part 26 model/revision plus 1,000 256-word documents; source/maxima NPZ files contain the named fields. | DOCUMENTED | Checked against existing Part 26 code and records; the runner validates metadata, hashes and shapes. |
| Part 61 output uses JSON manifests and NumPy compressed NPZ arrays with the declared keys. | DOCUMENTED | Implemented by the writer and checked by the analyzer; the real-model writer path remains UNVERIFIED locally because PyTorch is absent. |
| NumPy percentile behavior uses its default linear interpolation, and rank/correlation calculations use the installed NumPy. | DOCUMENTED | The code explicitly records default interpolation. Remote NumPy version is not pinned or checked, so runtime version behavior remains UNVERIFIED. |
| The rental environment has Bash, Linux `fcntl`, `tar`, `nvidia-smi`, Python, a working package index and Hugging Face access during prefetch. | UNVERIFIED | The runner's success/failure branches were tested with local shell stubs; no rented environment or network was used. |
| `huggingface_hub.snapshot_download(repo_id, revision, cache_dir)` can fetch both pinned model snapshots. | UNVERIFIED | The runner uses the same call pattern as Part 59, but prefetch was not executed. |
| `transformers==5.20.0` and `flash-linear-attention==0.5.2` install and load the selected snapshots in fp32 on the rental image. | UNVERIFIED | The runner enforces and records package versions, but package installation and model loading were not run. |

## WHAT COULD NOT BE CHECKED

This workstation has NumPy but no PyTorch, Transformers or CUDA stack. The source-control and analytic contracts, shell runner branches, and mutation probes ran; the full stub forward path was skipped. I could not verify runtime model structure, fast-tokenizer outputs, actual `W_down` column norms, VRAM use, throughput, package installation, model prefetch, completed model runs, or observed 61a-61e values. The registered predictions are preserved; this report does not claim an experimental result.

## BLOCKERS

None blocking delivery of the requested build files. The local PyTorch omission prevents execution of the full stub-model integration test here; it is recorded as an unverified item above.
