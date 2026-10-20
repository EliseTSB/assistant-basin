# Brief: build Part 29 (is a distilled student classified as its teacher?)

Read PREDICTIONS.md Part 29 first: it is the design and the predictions, written before this build. Do not edit
PREDICTIONS.md, README.md or any existing script. Work only in this worktree. You have no network: build and test
offline; Claude (serial) runs the collection on the host afterwards.

## Deliverables

1. `part29_prompts.py`: builds `data/part29-prompts.json` from a local JSONL of instruction rows (path given on the
   command line; Claude will pass databricks-dolly-15k). Keep categories open_qa, general_qa, brainstorming,
   creative_writing, classification. Drop any prompt that mentions a model, an AI, a chatbot, an assistant, a lab or
   a company name from the mask list below. Draw 300 with a fixed seed (0), each at most 400 characters, no context
   field. Record the seed, the source path and its sha256 in the output.
2. `part29_collect.py`: for each `--model` in a fixed table, sends each prompt with NO system prompt, temperature 0.7,
   max 300 output tokens, and appends one JSON line per answer to `results/part29/answers-<model>.jsonl`
   (prompt index, model id, provider, text, finish reason, usage, wall time). Resumable: skips indices already present.
   Providers:
   - OpenCode Go chat completions: `https://opencode.ai/zen/go/v1/chat/completions`, bearer from env
     `OPENCODE_GO_API_KEY`, header `x-opencode-session: part29`. Models deepseek-v4-pro, deepseek-v4.1-flash,
     glm-5.3, kimi-k3, qwen3.8-max, minimax-m3, gemini-4-argon-contributor, mimo-v2.6-flash.
   - OpenCode Go Anthropic messages: same host, path `/v1/messages`, model claude-haiku-5-5 (Claude models on Go speak
     only this protocol), `anthropic-version: 2023-06-01`, auth as above. Haiku 5.5 thinks by default: take the text
     blocks, never `content[0]`.
   - CLIProxy (OpenAI-compatible, no key) at `http://127.0.0.1:8317/v1/chat/completions` for gpt-6-luna. Requests
     to 127.0.0.1 must bypass any HTTPS proxy (use a session with `trust_env=False` or `proxies={}`).
   - Local Hugging Face models (`--hf <repo>@<revision>`, `--device cpu|xpu`): chat template, no system prompt,
     sampling at 0.7, same token cap; for models that emit `<think>` blocks, store the full text AND the text after the
     last `</think>` as `answer`.
   Reasoning text from APIs (`reasoning_content`) goes in its own field, never into `text`.
   Concurrency 4 per provider, retries with backoff on 429/5xx (honour Retry-After), never more than 3 tries.
3. `part29_classify.py`: masks family-identifying strings (case-insensitive list: Claude, Anthropic, GPT, ChatGPT,
   OpenAI, DeepSeek, Qwen, Alibaba, Tongyi, Gemini, Google, Bard, GLM, Zhipu, ChatGLM, Kimi, Moonshot, MiniMax, MiMo,
   Xiaomi, Llama, Meta, Gemma, OLMo, AI2, Allen, SmolLM, Hugging Face, Mistral, Phi, Microsoft) with a single token,
   then trains TF-IDF (word 1-2 grams plus char 3-5 grams) with logistic regression on prompts 0-239 of the eight
   teacher families (labels: claude=claude-haiku-5-5, gpt=gpt-6-luna, deepseek=deepseek-v4-pro, qwen=qwen3.8-max,
   gemini=gemini-4-argon-contributor, glm=glm-5.3, kimi=kimi-k3, minimax=minimax-m3), reports held-out accuracy and the
   confusion matrix on prompts 240-299, and for each student file the distribution of predicted families on prompts
   240-299 with a 95% bootstrap interval per family. For think models classify `answer` (after the think block) and,
   separately, the full text. Writes `results/part29/classify.json` and a readable `classify.txt`.
4. `test_part29.py`: offline tests with fake HTTP servers (chat, messages and cliproxy shapes, including a 429 with
   Retry-After, a think-block reply, and a reply whose first content block is thinking), resumability, the mask
   (a name inside a word must not be masked: "metadata" keeps "meta"), and a classifier test on synthetic families
   where the answer is known. A sabotage row: break the mask and show a test turns red.

Report in REPORT-29-build.md with SUMMARY, FINDINGS (anything in the design that does not hold up), BLOCKERS, and the
exact commands Claude should run on the host. Do not commit: the git directory is outside your domain; Claude commits after reading the report.
