# Part 62 build report

## SUMMARY

Implemented the Part 62 generator, blinded reader stage, analysis CLI, rental-box runner, runbook, and stub-based tests. The registered 1,536-answer matrix, pinned model revisions, control units, scales, phases, doses, and 62a–62g thresholds are encoded and checked. Local tests pass; no model, API, or GPU run was performed.

## FINDINGS

### Files

- `run62.py` writes 192 answer-cell files plus a manifest, derives and records a separate deterministic seed for every cell, runs both pinned models sequentially in fp32, measures alien-frame maxima through the Part 26 capture path, and scales one same-layer control per model by down-projection column norm.
- `blind62.py` validates the complete writer manifest, shuffles answers, sends only rubric and answer text to the Part 59A transport, applies the origin calibration threshold, records completion stop reasons, and refuses scoring with unlabelled answers or unverified/truncated completions.
- `analyze62.py` computes the review's three keyword proxies, six binomial logistic fits, and the fixed 62a–62g held/failed results.
- `run62.sh` retains the rented box on failure, validates the output manifest and archive, prints `RETURN_READY`, and exits zero only on the successful path. `RUNBOOK-62.md` documents generation, return, blind-label, analysis, and stub-test commands.
- `test_part62.py` exercises constants, Part 26 maximum reads, seed separation, the real writer-to-reader file path, stub phase clamps, blind completion/calibration gates, API request privacy, analysis, and mutation checks.

### Tests and checks run

From `/home/elisepc/assistant-basin/part62`:

```text
python3 -m py_compile run62.py blind62.py analyze62.py test_part62.py
python3 -m unittest discover -s . -p 'test_part62.py' -v
bash -n run62.sh
python3 run62.py --help
python3 blind62.py --help
python3 analyze62.py --help
```

Result: **17 tests passed**. Both mutation checks made the sabotaged production function fail its assertion: forcing `make_phase_hook` to use the `both` phase made the prompt-phase assertion go red; forcing `evaluate_predictions` to hold 62b made the threshold assertion go red. The tests restore the original function code after each mutation. The CLI help commands and shell syntax check completed successfully. Environment discovery found NumPy and SciPy installed, with PyTorch and Transformers absent.

### External assumptions

- **DOCUMENTED (local source/design):** the target IDs, revisions, layers, units, dose ladder, questions, sample count, controls, and predictions are taken from the Part 61–63 design and Part 26 records. The registered Part 26 maxima are read and tested from the local JSON/NPZ data. The Part 26 source-frame reader and maximum-capture helper are called to measure alien-frame maxima from the selected Part 33 passages; the 30-row requirement is checked in code.
- **DOCUMENTED (local source):** the phase adapter uses the existing `neuron_report.NeuronHook` interface (`phase` and `clamp`) and Part 26 helpers for offline loading, prompts, layer discovery, EOS IDs, and frame capture. The test stub models the existing hook's prompt-versus-single-token selection rule, and prompt-only and generation-only behavior are tested in both directions.
- **DOCUMENTED (local source):** Part 59A's local `blind_setting.py` defines the HTTP transport and Part 54's `blind_origin.py` supplies the rubric and origin calibration cases. The code submits `POST /v1/messages/batches` with `requests` entries containing `custom_id` and `params` (`model`, `max_tokens`, `messages`), then polls the batch endpoint and reads its `/results` JSONL. The parser expects `custom_id`, `result.type`, `result.message.content` text blocks, and `result.message.stop_reason`. The Part 62 request sets `max_tokens=1024`; local tests cover accepted and rejected simulated completion statuses.
- **DOCUMENTED (implementation):** the cell/manifest JSON schemas, writer file names, exact file inventory, answer key, and shuffled-key matching are defined by the new code. A test writes all registered stub cells using the production writer and reads them through the production blind reader.
- **UNVERIFIED (runtime):** actual Qwen and OLMo checkpoints at the pinned revisions are not loaded here. Torch and Transformers are absent, so the real `NeuronHook`/model integration, tensor placement and dtype, tokenizer behavior, generated-token slicing, GPU memory use, measured column norms, and runtime alien-frame values remain unverified.
- **UNVERIFIED (service):** no Anthropic request was sent. Live acceptance of the batch request shape and exact result fields listed above, including `message.stop_reason` values `end_turn` and `stop_sequence`, is not verified. Tests use representative local JSON responses only. Credentials, connectivity, batch lifecycle, and calibration outcomes remain unverified.
- **UNVERIFIED (rental execution):** `run62.sh` was syntax-checked but not run against the model cache or rental GPU. Availability of all run-time input files on that box, archive creation and inspection, a complete `RETURN_READY` path, and failure retention on the rental instance were not exercised.
- **UNVERIFIED (rental tooling):** the local interpreter is Python 3.12, while the rental box's Python version, `bash`, `tar`, NumPy version, and checkpoint cache availability were not inspected. The generator's syntax requires Python 3.10 or newer.

## BLOCKERS

None for the local build. The experiment itself still requires the specified offline model snapshots and runtime dependencies, and the blinded labels require the external service; neither was available or invoked in this build task.
