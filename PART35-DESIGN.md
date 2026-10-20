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

## Open questions for the council

1. Is a per-neuron contrast the right screen, or should candidates come from attribution (EAP-IG, as Sauers did), and
   the contrast only confirm?
2. What multiple of the natural maximum is fair across models whose activation scales differ?
3. How many concepts before the multiple-testing problem eats the result (40 concepts x 5 candidates x 6 models)?
