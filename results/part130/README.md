# Part 130: the same pole is a persona in all three Llama fine-tunes 12.5 times more often than chance, and in a sample of 150 about a third carry the same or a closely related concept

Across Llama 3.1 8B Instruct, Tulu 3 8B and Hermes 3 8B, 3,681 poles (the same layer, neuron index and sign in each model)
are a persona in all three library files, against a mean of 294.5 when Tulu's and Hermes' flags are rotated within a layer
(highest of 1,000 rotations: 350). That reproduces the numbers in Sauers' figure (3,681 and 294) and gives them a stated
null. In a sample of 150 of those poles, a reader judged the Llama and Hermes names to be the same or a closely related
concept in 49 (32.7%, 95% interval about 26 to 41%), and in 0 of 149 shuffled pairs. Scope: library labels at each pole's
library dose, from the library's "Who are you?" push as Neuron Lens and the library describe it.

## Predictions

| # | Prediction (verbatim, with its confidence) | Result | Verdict |
| --- | --- | --- | --- |
| 130.1 | Claude (high confidence): The triple count is at least 5 times the rotation null's mean. | 3,681 against 294.5, 12.5 times | held |
| 130.2 | Claude (low confidence): The ratio of observed to null is at least 1.5 times higher in L0-15 than in L16-31. | L0-15 9.8, L16-31 257.1 | failed, and the other way round |
| 130.3 | Claude (medium confidence): Among sampled triple poles, the reader calls the Llama and Hermes names the same concept in at least 50%. | 49 of 150 (32.7%); shuffled 0 of 149 | failed |
| 130.4 | Claude (low confidence): The three pairwise ratios of observed to null are within a factor of 1.5 of each other. | Llama-Tulu 2.60, Llama-Hermes 2.94, Tulu-Hermes 4.26; largest over smallest 1.64 | failed |

## Where in the network is the overlap?

| Layers | Triple poles | Mean under rotation |
| --- | --- | --- |
| L0-5 | 1,891 | 277.0 |
| L6-15 | 953 | 14.3 |
| L16-23 | 783 | 3.2 |
| L24-31 | 54 | 0.0 |

Most shared poles are early, and the absolute excess is largest there (1,614 in L0-5). The observed-to-null ratio is largest
late, where the null means are smallest (0.035 across L24-31). The stability of those ratios was not tested.
From `results/part130/overlap130.json`, made by `part130/overlap130.py`.

## Do the shared poles mean the same thing?

Not usually, by this reader: 101 of 150 sampled pairs were judged not to name the same or a closely related concept, so a
pole can be a persona in all three models and be a different persona in each. Sauers' figure counts a shared word between
two models, 36.9% against 2.6% by chance (read from the figure); our 32.7% against 0 of 149 asks about the concept. DeepSeek
V4.1 Flash read 299 of 300 pairs; one shuffled pair returned empty text on every try and is counted as missing. The gate
passed: DeepSeek called 0 of 149 shuffled pairs the same, and Luna agreed at kappa 0.78 on 45 pairs. On Luna's own subset
the same-concept rate is 10 of 23 (43%), so 130.3 fails on either reader. From `results/part130/score130.json`, made from `labels-ds.jsonl` and
`labels-luna.jsonl`.

## Which fine-tunes are closest?

Tulu and Hermes share the most relative to chance (4.26), more than either does with Llama Instruct (2.60 and 2.94). This
was not predicted and is one number per pair, so it is a lead. One possible reason, untested here, is that Tulu and Hermes
reached their weights by a different route from Meta's Instruct. Part 131 tests the base model.

## Method

Design: `PART130-DESIGN.md`. Library files saved 2026-10-30 from neuronpersonas.com (`data/<model>/layers`). A pole counts when
its row has category persona and a name. Null: Tulu's and Hermes' flags rotated within each layer by a random offset, 1,000
times, seed 130. No model was run, so the hook, broken-answer and dose-ladder controls do not apply. The closest earlier work of
ours is Parts 90 and 90b (one Tulu neuron across stages) and Part 91 (Qwen3-8B personas in its base model).

## Limits

A library persona comes from one prompt and one Haiku judge at each pole's chosen push, and the three models' library runs
are not guaranteed to use the same dose. This says nothing yet about Llama's base model. The per-layer ratios divide by very
small means at depth. The reader judged names only, not the answers behind them.

## Files

- `results/part130/overlap130.json`: counts, null means and per-layer tables; made by `part130/overlap130.py`.
- `results/part130/triples130.json`: all 3,681 poles with their three names.
- `results/part130/pairs130.jsonl`, `labels-ds.jsonl`, `labels-luna.jsonl`: reader items and labels; made by `part130/label130.py`.
- `results/part130/score130.json`: reader counts and agreement; made by `part130/score130.py`.

Cold check: Claude Haiku 5.5 with no prior context, 2026-10-30, all numbers reproduced; 14 findings on wording and omissions, all folded.
