# Part 29 build report

## SUMMARY

Built the fixed-seed prompt builder, resumable API/HF collector, NumPy classifier, and offline tests. The classifier trains on teacher prompts 0–239, evaluates 240–299, and reports bootstrap intervals for student predictions. No model collection or prompt-set build was run because the Dolly JSONL is not in this worktree and this worker had no network.

## FINDINGS

- The requested NumPy and Torch are available, but scikit-learn is not. The classifier therefore implements word 1–2 gram and character 3–5 gram TF-IDF plus multinomial logistic regression with NumPy; no packages were installed.
- Eight offline tests pass, covering prompt filtering and metadata, chat/messages/CLIProxy request shapes, a 429 with `Retry-After`, a thinking-first Anthropic response, retry limits, resumability, the `metadata` mask regression, and a synthetic classifier whose known family answers are recovered.
- The sandbox denied binding a loopback socket (`PermissionError`). Protocol tests use an in-process fake HTTP endpoint at the urllib boundary, including HTTP error headers and response bodies. This exercises request construction and parsing without a live server.
- Live API behavior could not be verified. The Anthropic route uses the brief's bearer authentication wording. The fixed family-name mask removes the listed terms as whole tokens but cannot prevent every indirect family cue, such as a version suffix beside a masked name.
- Part 29's family labels mean relative answer-style resemblance within these eight teachers; the output is not evidence of training provenance. The prompt filters are lexical, and prompts over 400 characters are excluded rather than truncated.
- The host commands use Hugging Face `@main` revisions because the design did not specify immutable revisions. Pin commit hashes for a repeatable model snapshot if the host run has them.

## BLOCKERS

None for the build. The host run needs the local Dolly JSONL path, `OPENCODE_GO_API_KEY`, CLIProxy listening at `127.0.0.1:8317`, and the selected HF model revisions available to Transformers.

## HOST COMMANDS

Run from the worktree. Set `DOLLY_JSONL` to the local `databricks-dolly-15k` JSONL path; that source file was not supplied here.

```bash
cd /home/elisepc/green/worktrees/ab-part29-build
export DOLLY_JSONL=/absolute/path/to/databricks-dolly-15k.jsonl
/home/elisepc/neuron-probe/.venv/bin/python part29_prompts.py "$DOLLY_JSONL"
: "${OPENCODE_GO_API_KEY:?set OPENCODE_GO_API_KEY before collecting OpenCode Go models}"
/home/elisepc/neuron-probe/.venv/bin/python part29_collect.py --prompts data/part29-prompts.json --out-dir results/part29
/home/elisepc/neuron-probe/.venv/bin/python part29_collect.py --prompts data/part29-prompts.json --out-dir results/part29 \
  --hf allenai/Olmo-3-7B-Think@main \
  --hf deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B@main \
  --hf meta-llama/Llama-3.2-1B-Instruct@main \
  --hf google/gemma-3-1b-it@main \
  --device cpu
/home/elisepc/neuron-probe/.venv/bin/python part29_classify.py --prompts data/part29-prompts.json --results-dir results/part29
```

The collector's first invocation runs every fixed API model from the table, including the eight teachers and the two API students (DeepSeek V4.1 Flash and MiMo V2.6 Flash). Successful rows append to `answers-<model>.jsonl`; rerunning skips indices already present.
