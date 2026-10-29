# Part 93 stage 1

`prompts93.json` contains 200 ordinary prompts, 40 in each of five kinds. `gen93.py` writes resumable per-family JSONL; `mask93.py` replaces content words while retaining source punctuation and layout; `classify93.py` reports grouped five-fold masked and unmasked results for both classifiers and exposes fit, save/load, and probability APIs. `test93.py` covers the offline contracts below.

Test command: `/home/elisepc/neuron-probe/.venv/bin/python -I -B part93/test93.py` — **7 tests passed**.

Run from `/home/elisepc/green/worktrees/ab-part67`. Set `OPENCODE_GO_BASE`, `GEMINI_MODEL`, and `QWEN35_4B_DIR` to the values supplied for the run before using those commands.

```sh
/home/elisepc/neuron-probe/.venv/bin/python -I -B part93/gen93.py --family claude --out /tmp/part93-smoke --workers 1 --limit 3
/home/elisepc/neuron-probe/.venv/bin/python -I -B part93/gen93.py --family gpt --out /tmp/part93-smoke --workers 1 --limit 3
/home/elisepc/neuron-probe/.venv/bin/python -I -B part93/gen93.py --family deepseek --go-base "$OPENCODE_GO_BASE" --out /tmp/part93-smoke --workers 1 --limit 3
/home/elisepc/neuron-probe/.venv/bin/python -I -B part93/gen93.py --family gemini --go-base "$OPENCODE_GO_BASE" --gemini-model "$GEMINI_MODEL" --out /tmp/part93-smoke --workers 1 --limit 3
/home/elisepc/neuron-probe/.venv/bin/python -I -B part93/gen93.py --family qwen --model "$QWEN35_4B_DIR" --device cuda --out /tmp/part93-smoke --workers 1 --limit 3
```

No live requests were made. Provider reachability, credentials, the supplied Go base/model name, and local Qwen device memory remain to be confirmed by the run operator.
