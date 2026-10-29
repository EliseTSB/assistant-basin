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

## Prior art (Muse, council topic muse-transplant-prior-art-2026-10-29 #3, received before any run)

No single-neuron transplant between a base model and its chat model, and no weight move read out as persona, was found.
Closest: Panigrahi et al. 2023 (grafting a sparse fine-tuned region onto the pretrained model, task accuracy); Al-Zuraiqi
2026 (arXiv:2601.13580, whole transformer blocks moved between models, success depends on position); Ilharco 2023 and
Wortsman 2022 (whole-model weight moves between base and fine-tuned). On recruitment against storage: Minder et al. 2025
(arXiv:2504.02922, chat-specific crosscoder latents) against a tied-crosscoder post on Gemma-2-2B arguing that fine-tuning
mostly re-times shared features. Muse's traps that apply to the component swaps: report coherence separately so swap
damage is not read as absence; report all four network-by-frame cells; a null may mean the host cannot read the
donor's component, not that the content is absent. Added before any run: a sham swap (Instruct's attention copied into
Instruct) as an identity check, and unpushed coherence for every run.

## Amendment 1 (2026-10-29 01:49 CT, from Alden's review, council alden-research-review-2026-10-29 #4, before any run)

- Correction: cosine 0.995 is not "half a percent of the weights". For unit vectors the distance is sqrt(2 - 2c), about
  0.10, and cosine ignores norm. The unit transplant is low leverage under this additive push, not a no-op by
  construction.
- Fixed injection: every run receives the identical vector per unit: its push and its paired random direction built
  once on the unmodified Instruct model in the chat frame (alpha 0.8 of Instruct's chat residual median at the unit's
  layer; `part99/make_pushes99.py`, `results/part99-pushes.pt` with norms in `.json`), passed with `--pushes`. No
  recalibration against a hybrid's residual median. The push is added at the same block-output site as Part 92.
- Transplant boundary (OLMo2): `mlp` copies gate_proj, up_proj and down_proj of every layer; `attn` copies every
  parameter under each layer's self_attn (q, k, v, o projections and the q and k norms). Host-kept in every swap: the
  embeddings, every post-attention and post-feedforward RMSNorm, the final norm and the LM head. Revisions: Base =
  allenai/OLMo-2-0425-1B at f83cd641 (stage2-ingredient3-step23852-tokens51B), Instruct = OLMo-2-0425-1B-Instruct at
  48d788ec. A result cannot localize a change outside these tensor groups.
- Frame: "Base in the chat frame" is a prompt-format condition (Instruct's serialized chat template and tokenizer for
  both weights, same BOS and assistant prefix, no system prompt), not evidence that Base has an assistant policy.
- Scoring: persona change is read relative to each run's own unpushed answers, with coherence reported per run; a run
  whose unpushed answers are mostly incoherent marks localization inconclusive for that run rather than counting as
  necessity. The sham run (Instruct's attention copied into Instruct) must reproduce the unswapped run's answers exactly.
  The 53 units were selected on Instruct; nothing here estimates prevalence over all neurons.
- Reading rule (Alden's): a loss after replacing Instruct's MLPs says those weights are conditionally required for
  expressing this injected signal in that remainder and frame; a gain from Instruct's MLPs in Base supports conditional
  sufficiency there. Neither establishes storage or when the change happened in training.

## Amendment 2 (2026-10-29 03:15 CT, scoring mechanics, written after the seven runs and before any Part 99 label)

Seen before writing this: run exit codes (all 0), row counts (474 per run) and two identity checks, no label and no
reading of answers beyond them. inst_sham equals inst_none in all 474 answers (text-exact; greedy), so the sham gate
passes and inst_sham takes inst_none's labels. inst_none's pushed answers equal Part 92 step 2's Instruct-chat answers in
223 of 468; the rest share an opening and then diverge (step 2 ran on a 3090 at batch 192 with pushes built there; this
run used the A770 at batch 48 with the fixed pushes).

- Labels: `part92/label92.py` (Haiku 5.5, unchanged prompt) on every pushed row of base_chat, inst_mlpB, inst_attnB,
  base_mlpI, base_attnI and inst_none, and on each run's six unpushed answers judged against each of the 39 units'
  personas (unit "baseline", 234 items per run).
- Relative show: a unit's persona counts as shown on a question when the pushed answer shows it and the same run's
  unpushed answer to that question does not. A unit shows in a run if at least one of the six questions shows; "Who are
  you?" is question 0. Random directions are read by the same rule against their unit's persona.
- Coherence: per run, the reader's coherent share of the six unpushed answers and of the pushed answers. More than three
  of six unpushed answers incoherent makes that run inconclusive for localization; if base_chat is inconclusive, 99.1
  and 99.2 are not tested.
- References: 99.1 and 99.2 name Part 92 step 2 as the Instruct-chat cell (by its selection rule every kept unit shows
  on "Who are you?", so 21 of 21 slot and 18 of 18 concept). Verdicts are read against that registered reference, and
  the like-for-like cell inst_none is reported beside it; if the two would give different verdicts, the report says so.
- 99.3: held if inst_attnB's count of slot units showing on "Who are you?" is strictly lower than inst_mlpB's.
- 99.4: non-human-kind units (Part 92's persona kinds), each run's random rate; held if the worst run is at most 10%.
