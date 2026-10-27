# Part 77 build notes

## What was built

`part77.py` registers the four Part 77 conditions, obtains the 240 answer rows per condition from `part70.generations`, duplicates each batch into answer and twin halves, applies the answer interventions and maker-only twin interventions, copies condition-specific layer-27 features position by position, and shares each sampled next token with its twin. The runner writes only answer records, records each pair's identity result, supports resuming `answers.jsonl`, and stores `conditions.json` and `run.json`. The scorer delegates lexical answer counting to `score66.score_rows` and applies the registered verdict gates.

The installed 5.20.0 Transformers source does not match the design's stated merge order. In `/home/elisepc/neuron-probe/.venv/lib/python3.12/site-packages/transformers/generation/utils.py`, `_get_logits_processor` calls `_merge_criteria_processor_list` at line 1580, while temperature, top-k, and top-p warpers are appended afterward at lines 1599–1609. `_merge_criteria_processor_list` appends custom processors after the defaults (the implementation begins at line 1683), but that merge happens before those sampling warpers. Thus in this environment the custom processor would see pre-warp scores. The implementation inspects the installed method's ordering and mirrors temperature 0.7, top-k 50 (Transformers' default), and top-p 0.95 inside the custom processor when needed. On a Transformers ordering that places the merge after warpers, it uses the received scores directly. `generate` retains run_70's exact sampling arguments.

## Tests

The requested command `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q part77` passed: 11 tests passed in 4.74 seconds.

The tests cover processor behavior and determinism, copy-hook row isolation, paired generation with a local randomly initialized 30-layer Llama model, observed copied and uncopied layer-27 activations, c3 units against Part 76 JSON, scorer gating, prohibited device float64 usage, CLI help, and import resolution. They do not load Qwen, use an XPU, validate actual model quality, run all 960 pairs, or assess whether sampled answers name Google accurately beyond the registered lexical scorer.

## Design ambiguity

The registered design says custom processors see scores after temperature/top-k/top-p in Transformers 5.13 and 5.20. Local source inspection contradicts that statement for the installed 5.20.0 package. To preserve the design's intended answer-token distribution, the processor reproduces those warpers only when source ordering places custom processing before them. This leaves the experiment's intended distribution intact across the observed API orderings.
