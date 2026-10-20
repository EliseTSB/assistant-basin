# Part 35 design draft: a hunt for our own persona neurons, across models

Draft by Claude, 2026-10-20 04:57 CT, from Elise's ask (*"I'd really like to find our own fun neurons like the alien or
the owl one, bonus points if we can find it across multiple models!"*). For Muse (prior art), Prime and the eleventh
Weaver before predictions. Nothing has run.

## What counts as a find

A single MLP neuron (one input unit of a layer's down_proj) that:
1. **fires on a concept, not a word**: high on the concept named in several languages and on indirect references with
   no concept word (the owl neuron's hoots, Hedwig, Athena), low on matched passages about other things;
2. **turns the model into it when pushed**: clamped at a multiple of its own natural maximum, "Who are you?" and
   "Introduce yourself" get first-person answers as the concept, judged blind, at a rate random same-layer neurons at the
   same multiple never reach;
3. **is not just a word pusher**: the pushed answers stay fluent and answer the question (Part 33's lesson: the alien
   neuron mostly pushes alien words wherever the model is writing).

## The screen

- **Concepts**, 40 written in advance, chosen to be fun and to have indirect references: owl, cat, dragon, ghost, pirate,
  wizard, robot, vampire, mermaid, octopus, the moon, the ocean, a volcano, a tree, a cloud, the sun, a knight, a
  detective, a chef, a bee, a wolf, a whale, a crow, a fox, a snowman, a lighthouse, a library, a clock, a mushroom, a
  cactus, a star, a river, a mountain, a train, a ship, a teapot, a candle, a raven, a unicorn, a spider. Owl and the
  alien are the positive controls (owl replicates Sauers on a model that has it; alien on Qwen3.5-4B, L22 N2742).
- **Passages** per concept: 12 short passages naming it in English, 4 in each of Spanish, Chinese, German and Russian,
  and 8 indirect passages with no concept word. Controls: the same passage frames with other concepts.
- **Score** per neuron: mean max activation on the concept's passages minus on the others, in units of the neuron's own
  spread; keep neurons whose indirect and non-English scores are also high. Top 5 per concept per model go to the push.

## The push

Clamp each candidate at 4x and 8x its largest natural activation on the screen, on every generated token, for six
identity prompts, 8 samples each at T 0.7, 80 tokens. Controls: 10 random neurons from the same layer at the same
multiple. A fixed blind judge labels each answer (claims to be the concept / mentions it without claiming / neither /
broken), with a hand read of every claimed hit.

## Across models

Small open models that run on Claude Cloud CPU or the A770 without spilling: Qwen3-1.7B, Qwen3-4B, Qwen3.5-4B,
SmolLM3-3B, OLMo-2-1B Instruct, and one more family if a licence allows (Gemma 3 1B or Llama 3.2 1B). The cross-model
finding is the same concept giving a persona neuron in more than one family, and, as a stretch, whether those neurons
read similar text (their top-activating passages compared).

## Elise's additions (05:09 CT): Claude and Anthropic, and Trump

*"Make sure to look for claude or anthropic related neurons. Looks for Trump neurons too, he has a very unique speaking
style i wonder if we can find traces of it, and the training data should be rich with him."*

- **Claude and Anthropic** join the concept list as two concepts. Indirect passages name neither: "the assistant with a
  constitution", "the company Dario Amodei co-founded", Claude's own turns of phrase. The push asks whether the model
  then says it is Claude or made by Anthropic; Part 25's maker-neuron cells (google-removal, openai) are the comparison,
  and Part 29's finding that Qwen 3.5 was distilled from Claude (Anthropic's September report) predicts these neurons are
  likelier in Qwen3.5 than in Qwen3.
- **Trump** is a different kind of target: a speaking style, not an object. Screen passages: his name and role in
  several languages, plus an indirect set in his style with no name (superlatives, "believe me", "many people are
  saying", repetition, nicknames), against matched passages in other public figures' styles and in neutral prose. The
  push is scored on ordinary questions, not only identity ones: does the answer take on the style, judged blind against
  a style rubric fixed in advance, as well as "does it claim to be him". Style without the name is the interesting
  half, because it would show the neuron carries the manner, not the person.

## Prime's ML review (council muse-neuron-research-2026-10 #60, 2026-10-20 05:02 CT), adopted 05:06 CT

This replaces the screen, push and model sections above where they differ.

- **Contrast for discovery, attribution only for a confirmed hit.** EAP-IG needs a target logit per concept and
  gradients; it is the circuit tool for a find (the suppression nodes), not a screen. Optional bridge: ablate the top
  candidates on concept prompts and read the drop in a concept-continuation logprob, reported as necessity, never as a
  veto, since the find criterion is sufficiency.
- **Dose by the injected vector, not by a multiple of the neuron's max.** Choose the level L with
  L x ||W_down[n]|| = rho x ||h_l||, where ||h_l|| is the RMS residual norm on neutral prompts at that layer and rho is
  0.05 or 0.10. Secondary: 2x, 4x, 8x the natural maximum measured on neutral passages (never on the concept passages
  that selected it), for comparability with Part 18. Report both, the equivalent multiple and the achieved change.
  Clamp = set to L on generated tokens only (Part 18b's working arm); log the share of tokens whose natural value
  exceeded L.
- **Multiplicity lives in the screen.** One bank of 36 matched frames filled with each concept, so every contrast is
  within-frame (concept against other concepts; concept present against absent for the indirect set). A permutation
  null per model (shuffled concept labels on the cached activations): a candidate must beat the permutation maximum.
  Two-level hold-out: each concept's passages split 18 discovery / 18 confirmation, every reported number on the
  confirmation half; 8 to 10 of the 40 concepts held out entirely until the procedure is frozen. Confirmatory family =
  rank-1 candidates only, one primary endpoint (claims the screened concept against matched controls), exact test at
  alpha = 0.05/240; ranks 2-5 exploratory. Rank on a robust statistic (median of per-passage maxima, or k of n passages
  above the negative set's 99th percentile), with spread defined on a fixed neutral reference.
- **Families, not checkpoints.** Qwen3-1.7B, Qwen3-4B and Qwen3.5-4B are one family. The headline is the family rule;
  per-model hits are reported separately.
- **Staged budget.** Stage 1: rank-1 per concept per model, 2 doses, 2 identity prompts, 4 samples (about 3,840
  generations), with 10 live random neurons per model, all judged. Stage 2: only stage-1 hits and near-hits, with 8
  identity prompts x 8 samples, held-out prompts and held-out passages. The unit of analysis is the prompt (samples of
  one prompt are clustered); a cluster permutation over control neurons; the hit threshold is a number fixed in the
  predictions.
- **Judge blind to the concept**: free-text label of the claimed persona (or none, or broken), matched to the screened
  concept afterwards; a specificity matrix (screened against claimed); live, scale-matched controls; neurons that top
  two related concepts reported once. **Non-identity prompts under the same clamp**: a persona neuron leaves ordinary
  answers ordinary, a word pusher does not. A zero-change harness control (Part 18's identity hook) and a judge
  reliability check on at least 100 answers, plus a hand read of every claim.
- **Controls.** The alien (Qwen3.5-4B L22 N2742) is the positive control, reported separately and excluded from the
  find rate. The owl cannot be a replication control: Sauers' model is unnamed (likely Qwen3-8B). The owl becomes a
  target concept, plus a replication attempt on Qwen3-8B only if it fits the hardware, never counted as a find.
- **Cross-model comparison** on held-out probe profiles (passages not used in screening), layers aligned by depth
  fraction.

## Open questions for the council (answered by Prime above)

1. Is a per-neuron contrast the right screen, or should candidates come from attribution (EAP-IG, as Sauers did), and
   the contrast only confirm?
2. What multiple of the natural maximum is fair across models whose activation scales differ?
3. How many concepts before the multiple-testing problem eats the result (40 concepts x 5 candidates x 6 models)?
