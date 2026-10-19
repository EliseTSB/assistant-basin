# Brief: build part27.py (what Part 25's four neurons are: fires, writes, costs, base)

You are a dark-factory worker in `/home/elisepc/green/worktrees/ab-27` (branch `part27` of the public assistant-basin
repo). Write only inside it. Do not commit, do not download models, do not edit PREDICTIONS.md, and do not load any
real model's weights (the real run happens later on an Intel Arc GPU under Windows). Tiny random models built from the
real configs are fine for tests.

Read first: PREDICTIONS.md section "Part 27" (build to its definitions exactly; where one is ambiguous, say so in your
report rather than choosing silently). Reuse, do not copy: `part26.py` (`word_spans`, `word_token_indices`,
`word_activations`, the document selection from `--corpus`, the writes/logit-lens code), `part25.py` (the six identity
questions, `_greedy_generate`, `install_intervention_hooks` with operation `zero`, `first_sentence`, `header`,
`source_shas`, `write_json`), `neuron_report.py` (`load_model`, `find_text_layers`, `spill_guard`, `_dump`), `part24.py`
(sampling settings). `--device xpu`, `--max-gpu-gb 14.5` spill guard after every batch. Every hook removed in a `finally`.

Models (offline, Windows cache `E:\neuron-probe\hf`; for tests read configs and tokenizers from
`/mnt/e/neuron-probe/hf/hub/...`): chat `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
base `Qwen/Qwen3.5-4B-Base` revision `1001bb4d826a52d1f399e183466143f4da7b741b` (its WSL snapshot links may be
unreadable; if so, test the base path with the chat config and say so).

The set (a "neuron" is one input unit of a layer's MLP down_proj, as elsewhere): (6,753), (7,1169), (11,8868), (29,776).
Controls: 20 random sets of four, each drawing one unit from each of layers 6, 7, 11 and 29, fixed seeds 0..19,
excluding the set's own units. Record every control set.

## Subcommands (each writes one JSON into `--out`, with `header()` and input shas)

- `fires` (chat model): the first 1,000 corpus documents of >= 256 words, first 256 words each, exactly as part26
  selects them (record indices). For each of the four neurons and each control neuron: per-document largest token
  activation and its position; the top 40 documents by that value, each with a 15-token window (7 before, the token,
  7 after) decoded, plus the activation. Also each neuron's activation at every token of the six identity questions
  in the chat template (no system message) and with each system prompt Part 25 uses (read them from part25.py).
  Also save the per-word activations of the four (npz) for the base comparison.
- `writes` (chat model): logit-lens top 30 and bottom 30 tokens per neuron of the set and of the controls, as part26
  computes writes (decoded, leading space and case removed for the string lists, raw token ids kept too).
- `costs` (chat model): (a) mean next-token loss over 200 Pile documents: the corpus documents after the first 1,000
  that also have >= 256 words, first 256 words each, record indices; baseline, the set zeroed (all positions), and each
  control zeroed. (b) 20 ordinary factual questions in `data/part27-facts.json` (write that file: plain questions with
  one short known answer each, e.g. capital cities, chemical symbols, authors, arithmetic; record the expected answer)
  answered greedily with the chat template, max 40 new tokens, baseline / set zeroed / each control zeroed; count
  answers whose first sentence changed, and whether the expected answer string still appears. (c) The six identity
  questions greedily, baseline / set zeroed / each control zeroed, per question, with `first_sentence` and whether it
  names Qwen/Tongyi/Alibaba.
- `base` (base model): (a) per-word activations of the same four indices on the same 1,000 documents, and per-neuron
  Pearson against the chat model's saved npz; (b) the four's activations on the six identity questions written as
  "Q: <question>\nA:" (no chat template); (c) 8 sampled completions (part24 settings, max 60 new tokens, seeds 0..7) of
  "Q: Who created you?\nA:" baseline and with the set zeroed, plus the same for each control set (1 sample each is
  enough for controls, seed 0); report which completions name a maker and which maker.

## Tests

A test file `test_part27.py` (pytest or plain asserts, as the repo's other part tests do) with a tiny random model built
from the real Qwen3.5-4B config (few layers, small width): hooks zero exactly the named units and nothing else (compare
an unhooked and hooked forward on a unit not in the set), controls exclude the set and draw one unit per layer, the
document selection matches part26's on a fixture corpus, loss computation matches a hand computation on a 3-document
fixture, and every subcommand runs end to end on the tiny model and writes valid JSON. Show the test output.

## Report

`REPORT-27-build.md`: what you built, the exact command lines for the GPU run (in the steps.txt form part26 uses:
`part27.py --steps fires,writes,costs,base --corpus qwen35\pile10k.jsonl --out results\part27`), the files the run
needs copied (as after-25-start-26.sh copies part26's), estimated GPU minutes per step with your reasoning, every
ambiguity you resolved, and SUMMARY / FINDINGS / BLOCKERS.
