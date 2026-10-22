# SUMMARY

Part 59's runners, blind reader, analysis, tests, and rented-GPU runbook are implemented under `part59/`. The scripts use the pinned Part 26 model revisions and activation records. The Part 59 experiment itself has not been run on a GPU.

# FINDINGS / WHAT SHIPPED

- `run59a.py` makes 12 neutral story prompts for Qwen3.5-4B L22:2742 and OLMo-2-1B L10:6978. It runs doses -2, -1, 0, +1, +2, eight samples per prompt-dose, and three same-layer controls at -2/+2. The zero-dose file contains the unclamped baseline. Each prompt-dose is saved as a Part 54-shaped `clamp-*.json`, so `blind_origin.answers()` can read it without losing the prompt text. Manifests, atomic result writes, a single-output-folder exclusive lock, resume validation, and `--force` are included.
- Scaling uses Qwen's checked Part 54 maxima cache (N2742 19.125; controls 152/376/692: 19.7721, 13.5135, 2.1834) and OLMo's Part 26 target record (unit 6978: 6.5; controls 135/335/615: 0.7422, 0.5039, 1.2969). Both control sets follow Part 26's policy of drawing 20 units with its SHA-seeded control helper and taking the first three. Generation calls Part 26's offline loader, `_generate_condition`, condition seed, output naming, and header helpers.
- `blind_setting.py` submits shuffled, condition-hidden answers to Claude Opus 5.5 through the Messages Batch API. It has 24 setting calibration texts (six for each of four labels) plus four broken texts, and refuses to score after more than two calibration errors. The modern Earth calibrations include ordinary phones, radio, laptop, train, and mobile use; a fantasy calibration mentions a star. `--by-cond` prints the condition table.
- `run59b.py` streams all 1,000 documents from the saved Part 26 source record. It records each later-layer MLP neuron's equal-document mean at baseline and at -1/+1 for the target and three controls, then saves means and changes in a resumable `.npz` archive. `analyze59b.py` applies the registered sign and 99th-percentile rules, reports 59f/59g counts, and adds Part 26 logit-lens top tokens and the three strongest baseline activation windows for each top symmetric responder.
- `RUNBOOK-59.md` and `run59.sh` cover packaging, the Part 54 Vast image/version pattern, pinned model prefetch, unattended execution, per-phase logs, result tarball, and `RETURN_READY`. The estimate is 1,056 generations per model and 2,112 total. Based on Part 54A's 480 answers in about 11 minutes, 59A is about 49 minutes total. Including 59B and window extraction, the planning budget is 3-5 hours at the cited $0.2857/hour rate ($0.86-$1.43).
- The 8 CPU tests pass with `/home/elisepc/neuron-probe/.venv/bin/python -m unittest -v test_part59`. They cover dose scaling, simulated interruption and resume, exclusive locking, synthetic symmetric/odd cases including a value just below P99, a toy model with activation hooks, calibration refusal, prompt neutrality, and `blind_origin.answers()` compatibility.
- Both requested sabotage runs went red: `PART59_SABOTAGE=flip-sign ... test_symmetric_threshold_and_odd_sign_rule` failed with 1 instead of 2 responders; `PART59_SABOTAGE=drop-control-percentile ... test_symmetric_threshold_and_odd_sign_rule` failed with 3 instead of 2.
- For Qwen in fp32, 4B raw weights are about 14.9 GiB; at batch size 1, a 350-token width-9,216 MLP activation is about 12.3 MiB. The estimated streaming peak is 16-19 GiB VRAM. A 1,000-document batch would use about 12.0 GiB for one MLP activation tensor alone, plus model weights and workspace, exceeding a 24 GiB 3090. These are estimates; the returned manifest records measured CUDA peaks when the GPU run occurs.

# FILES TOUCHED

- `run59a.py`
- `prompts59a.json`
- `blind_setting.py`
- `run59b.py`
- `analyze59b.py`
- `test_part59.py`
- `RUNBOOK-59.md`
- `run59.sh`
- `REPORT-runner.md`

# BLOCKERS

None for the runner build. No model was loaded and no rented-GPU run was started in this worker environment; validation used the CPU toy model and saved activation records.

# SELF-ASSESSMENT

The blind-reader brief combines “six per label plus four broken” with “2 of 22 wrong.” Four setting labels require 24 ordinary calibration texts, so the implemented batch has 28 rows and retains the absolute refusal cutoff of more than two errors. The raw `qwen35/pile10k.jsonl` file is not present in this checkout; the Part 26 record supplies the exact 1,000 selected documents, source hash, and the checked Part 54 maxima needed by these scripts.

The prompt-specific `clamp-*.json` shape produces 12 files per model-dose so each file has the single `question` field expected by `blind_origin.answers()`. The Part 26 `_source_clamp` function is not called directly: it is fixed to the Qwen origin-ladder behavior, while this experiment has two model units, five custom doses, and 12 prompts. Its relevant loader, sampling, seed, output, and scale artifacts are reused instead. Runtime and VRAM figures are planning estimates, not measurements from a 3090.
