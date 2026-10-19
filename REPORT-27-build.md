# Part 27 build report

## What shipped

`part27.py` implements the four Part 27 steps in the requested order. It reuses Part 26 document/token alignment, decoder capture and logit-lens routines; Part 25 prompt, zero-hook, generation, first-sentence and JSON helpers; and `neuron_report.py` model/layer discovery, neuron capture and spill guard. The runner writes `fires.json`, `writes.json`, `costs.json`, and `base.json`; fires and base also write the compressed per-word activation arrays needed for the comparison. Each JSON includes model/revision, tool and input SHA256 values.

`data/part27-facts.json` contains 20 short factual questions and expected answers. `test_part27.py` covers zero hooks, controls, Part 26 document selection, a hand-computed three-document loss, and all four CLI steps on tiny random models built from the cached Qwen configs and tokenizers. The base smoke test tries the base snapshot and falls back to the chat config if that snapshot cannot be read. The base config symlink is readable at the provided WSL path in this build environment.

## GPU run commands

Full run, using the Part 26 `steps.txt` command style:

```text
python part27.py --steps fires,writes,costs,base --corpus qwen35\pile10k.jsonl --out results\part27 --device xpu --max-gpu-gb 14.5
```

To resume only the base step after the first three results have been copied to the same output directory:

```text
python part27.py --steps base --corpus qwen35\pile10k.jsonl --out results\part27 --device xpu --max-gpu-gb 14.5
```

The `base` step requires `results\part27\fires.json` and `results\part27\fires-word-activations.npz` from the chat run. The default model IDs and pinned revisions match the brief. The loader sets Transformers and Hugging Face offline flags before loading.

## Files and runtime inputs to copy

- `part27.py`, `part24.py`, `part25.py`, `part26.py`, `neuron_report.py`, `circuit.py`, `part20.py`, `part20c.py`, `part20d.py`, and `part20f.py`. The latter four are imported by Part 26 and are needed for import-time dependencies.
- `data/part27-facts.json` and `PREDICTIONS.md` (the latter is hashed into output provenance).
- The corpus at `qwen35\pile10k.jsonl`.
- For a separate base-only invocation, the two fires outputs listed above.
- The pinned chat and base model/tokenizer snapshots already present in the offline Hugging Face cache, plus Python with NumPy, PyTorch, Transformers, and the required XPU support. No model weights are downloaded by this runner.
- `test_part27.py` and the cached chat/base configs and tokenizers if the CPU smoke tests are to be run on the GPU host.

## Estimated XPU time

These are planning estimates, not measured timings. I assume about 1.3–1.5 tokens per whitespace word, 100–250 input tokens/second for batched 4B inference, 15–50 generated tokens/second, batch size 4, and several minutes of XPU/kernel overhead. The real Arc run may differ substantially.

| Step | Estimated GPU minutes | Reasoning |
|---|---:|---|
| `fires` | 25–60 | About 330k document tokens across 1,000 documents, plus 36 identity/system prompt forwards, with activation capture for the target and up to 80 distinct controls. |
| `writes` | 3–12 | Up to 84 distinct units, top and bottom directions, four layer groups, chunked over the vocabulary. |
| `costs` | 110–280 | 22 conditions × 200 documents is about 1.5M input tokens; 22 conditions × 26 questions × 40 generated tokens adds up to 22,880 output tokens. |
| `base` | 25–65 | About 330k document tokens, six plain identity prompts, and up to 2,160 sampled completion tokens across baseline, set-zero, and controls. |

## Ambiguities resolved

- Loss is the token-weighted mean of causal next-token NLL over non-padding targets, rather than an unweighted mean of 200 per-document means. The JSON records both summed loss and scored-token count.
- Control sets use one unit from each target layer, excluding that layer's selected target unit. Seeds are 0–19; a control neuron may recur in another seed's set because cross-set uniqueness was not requested.
- Top-context ties preserve corpus order. Contexts contain 15 tokens; windows shift inward at document edges, and the token's offset within the window is recorded.
- Factual first-sentence changes use exact string equality after Part 25's `first_sentence()`. Expected answers use case-insensitive whole-word/phrase matching.
- Identity-maker flags use `circuit.IDENTITY`, including its Chinese Qwen/Alibaba aliases. Base/chat Pearson is calculated over the flattened 1,000 × 256 word positions for each target neuron.
- The two baseline/set-zero base completion groups use seeds 0–7 and the Part 24 sampling settings (`temperature=0.7`, `top_p=0.95`); each control gets one sample at seed 0. All zeroing applies at every position.

## Test output

Static checks completed successfully:

```text
$ python3 -m py_compile part27.py test_part27.py
(exit 0; no output)
$ python3 -m json.tool data/part27-facts.json >/dev/null
(exit 0; no output)
AST/whitespace checks: part27.py OK; test_part27.py OK; facts JSON has 20 question/answer rows.
```

The requested regression suite could not run in this worker environment:

```text
$ python3 -m pytest -q test_part27.py
/usr/bin/python3: No module named pytest

$ python3 -c "import torch"
ModuleNotFoundError: No module named 'torch'
```

No model weights were loaded and no GPU measurement was run in this build worker.

## SUMMARY

- Shipped the four-step Part 27 runner, 20-question fact battery, tiny-model regressions, and this run report.
- The runner records the four target neurons and all 20 seeded control sets, including controls in each step.
- Syntax and data validation passed; pytest execution was blocked by the worker's missing test/runtime dependencies.

## FINDINGS

- The full XPU command and a base-only resume command are ready above; `fires.json` plus its NPZ are the base comparison inputs.
- Control duplicates across different seeds are allowed; within each set, there is exactly one non-target unit per target layer.
- Runtime estimates are assumption-based and should be replaced with measured times from the Arc run.
- Parent-control mailbox was empty at every required pull; no guidance was applied, declined, or deferred.

## BLOCKERS

- The tiny-model regressions remain unexecuted here because `/usr/bin/python3` has neither `pytest` nor PyTorch. The GPU run is also unmeasured; it is intended for the later XPU run.
