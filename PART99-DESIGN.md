# Part 99: where post-training put the change that turns a pushed slot neuron into a self-description

Written 2026-10-29 by Claude (Opus 5.5), before any Part 99 run and before reading any step-3 label of Part 92 (only the
step-2 kinds were seen: 19 concept, 21 slot, 9 dropped, 4 unclassified). Suggested by Alden's review (a reciprocal
checkpoint transplant), changed after one measurement: the 53 shortlisted units' down, gate and up weights sit at median
cosine 0.995 between the mid-training end and Instruct (minimum 0.991), the same as every other neuron (Part 89), and in
a push the unit contributes only its write direction. A unit transplant would change half a percent of one unit, so
"behaviour follows the network" would hold by construction. Component swaps test the same question at a size that can
show something.

## Method

OLMo 2 1B. Units: Part 92's classified neurons (concept and slot) and their paired random directions, pushed exactly as
in Part 92's battery (alpha 0.8 of the host model's residual median at the unit's layer, six questions, greedy, 120
tokens, batch as measured). Labels: Part 92's blind Haiku reader (`label92.py`), shows_persona per answer. A unit shows
if at least one of six answers shows its persona; "Who are you?" shows is reported separately.

Runs (all new; Part 92's ladder supplies Base plain, Instruct plain and Instruct chat):
1. Base (mid-training end, 51B) in the chat frame, with Instruct's chat template (same vocabulary).
2. Instruct with every MLP weight (gate, up, down) from Base; chat frame.
3. Instruct with every attention weight (q, k, v, o and their norms) from Base; chat frame.
4. Base with every MLP weight from Instruct; chat frame.
5. Base with every attention weight from Instruct; chat frame.
Embeddings, the final norm and the unembedding stay the host's.

## Predictions

| # | Prediction | Confidence |
|---|---|---|
| 99.1 | Slot neurons need the finished network, not only the chat frame: the share of slot neurons whose "Who are you?" answer shows the persona is at least 30 points lower in Base-chat (run 1) than in Instruct-chat (Part 92 step 2). | medium |
| 99.2 | Concept neurons do not: the share of concept neurons showing in at least one answer differs by at most 15 points between Base-chat and Instruct-chat. | medium |
| 99.3 | Of the two swaps into Instruct, swapping in Base's attention removes more slot "Who are you?" shows than swapping in Base's MLPs. | low |
| 99.4 | Paired random directions show their unit's persona in at most 10% of non-human-kind pairs in every run. | medium |

Reading rule: 99.1 and 99.2 are read with a two-sided Fisher exact test on the unit counts; "held" needs the stated
direction and size. If run 1 breaks the model's answers (more than half incoherent unpushed), 99.1 and 99.2 are not tested.
