# Part 99 build notes

## SUMMARY

Built `swap99.py` as a Part 92 style steering battery with host/donor component swaps, optional tokenizer selection, unit-kind filtering, resumable JSONL, and per-run metadata. The runner copies into the already-loaded host under `torch.no_grad()` and loads the donor on CPU. Unit tests and the requested real CPU smoke run passed.

## FINDINGS

### What shipped

- `swap99.py` preserves Part 92's six questions, alpha `0.8`, neuron plus paired random pushes, host residual-median calibration in the requested frame, and greedy 120-token batches. It supports `--host`, `--donor`, `--swap none|mlp|attn`, `--tokenizer`, `--neurons`, `--frame`, `--tag`, `--out`, `--device`, `--batch`, `--kinds`, `--only`, and `--limit`.
- MLP swaps transfer all named parameters under every layer's `mlp` module, including the required gate/up/down projection weights. Attention swaps transfer all named parameters under each `self_attn`, including the q/k norms. Names and shapes are validated across all layers before copying; the startup line reports the copied module parameters and counts. Donor parameters are copied to the host's device and dtype.
- `<tag>.meta.json` records host, donor, swap, tokenizer source, tensor and element counts, parameter names, frame, device, library versions, and total elapsed seconds. The seconds include model loading and setup. Progress timing starts before answer generation.
- `--kinds` accepts Part 92's `{"kinds": {id: kind}}` score structure or a direct id-to-kind object. `--only` accepts `concept`, `slot`, or both and requires `--kinds`. `--limit` is applied after kind filtering and selects the first N eligible neurons. With neither filter, every neuron in the input file runs.
- The tokenizer guard compares `len(tokenizer)` and token IDs without special tokens for the fixed string `The quick brown fox jumps over the lazy dog; OLMo 2, 42, and café.`. If no tokenizer path is given, the host tokenizer is used. The CLI defaults are `swap=none`, `device=xpu`, and `batch=48`, matching Part 92's device and batch defaults where applicable.

### Tests

Command: `/home/elisepc/neuron-probe/.venv/bin/python -m pytest part99/test_swap99.py`

Result: **6 passed in 7.17s**. Tests save two independently seeded tiny OLMo-2 models in temporary snapshot directories, then check MLP and attention transfers, preservation of the other component, `none`, pre-copy shape-mismatch rejection, tokenizer guard failures, and swap metadata. Pytest emitted one cache warning because it could not write the repository-root `.pytest_cache`; no test failed.

### Real smoke run

Ran on CPU with Base as host, Instruct as donor and tokenizer, `swap=attn`, `frame=chat`, `--limit 1`, and `--batch 12`, writing `smoke/base-chat-instruct-attn-first.jsonl` and its metadata. The tokenizer guard passed. The swap copied **96 tensors / 268,500,992 elements** across 16 layers: `k_norm.weight`, `k_proj.weight`, `o_proj.weight`, `q_norm.weight`, `q_proj.weight`, and `v_proj.weight` in each layer. All 12 rows were written for `p92-00` (neuron and random units × six questions).

Total elapsed time was **84.9 seconds**, including loading and setup; answer generation took **38.5 seconds**. Runtime versions were torch `2.14.1+cpu` and transformers `5.20.0`. Transformers printed a deprecation warning for `torch_dtype` during model loading.

The following are each answer's first 80 characters, with newlines escaped:

| Unit | Question index | First 80 answer characters |
|---|---:|---|
| neuron | 0 | `A large block of ice breaks off from the source and flows downstream. What is th` |
| neuron | 1 | `A rock has been dislodged and is floating downstream. The rock has been dislodge` |
| neuron | 2 | `What do you want to know most?\nWhat do you want to know about?\n\nWhat do you want` |
| neuron | 3 | `A large rock is released from the mouth of a small channel. The rock is broken a` |
| neuron | 4 | `What is the age of the river?\n<sup>o</sup> and what is it?\nWhat is the age of th` |
| neuron | 5 | `A large mass of ice floating in the water is known as a glacier. The moving ice ` |
| random | 0 | `<|assistant|>\n<|assistant|>\n<|assistant|>\n<|assistant|>\n<|assistant|>\n<|assistan` |
| random | 1 | `<|assistant|>\n<|assistant|>\n<|assistant|>\n<|assistant|>\n<|assistant|>\n<|assistan` |
| random | 2 | `I want to be a good person. I want to be a good person. I want to be a good pers` |
| random | 3 | `I'm not sure what I'm doing. I'm not sure what I'm doing. I'm not sure what I'm ` |
| random | 4 | `France\nFrance\nFrance\nFrance\nFrance\nFrance\nFrance\nFrance\nFrance\nFrance\nFrance\nFra` |
| random | 5 | `The sea is a place to go\nTo see the waves\nTo see the waves\nTo see the waves\nTo s` |

## BLOCKERS

None. The pytest cache warning and Transformers deprecation warning did not affect the test or smoke results.
