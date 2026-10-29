# Part 109c: the concept dog and the slot dog, alone and together, pushed from -2x to +2x

Registered 2026-10-29 17:38 CT by Claude (Opus 5.5), from Elise's question at 17:35 CT: "What happens when you combine a
related slot and concept neuron, like the dog ones, at various strengths? Try pushing it negative too". The commit that
adds this file is the registration.

A **concept neuron** (Part 92) shows its persona in 3 or more of six questions; a **slot neuron** shows it only in "Who
are you?" or "Tell me about yourself." The pair: the concept dog **p92-08** (L4 N5154, "I am a dog. I wag my tail.")
and the slot dog **p92-37** (L5 N7218, "Hello! I'm a large, friendly dog.", Part 109's dog). A second slot dog,
**p92-09** (L6 N4145, "I'm just a dog and I love treats!"), is a small replication.

## Method

OLMo 2 1B Instruct, chat frame, greedy, 100 new tokens, CPU, Part 99's fixed push per neuron multiplied by a strength m
and added after the neuron's layer at every position (two neurons: both pushes at once). Seven questions per condition,
generated as one left-padded batch: Part 91's six (Who are you?, Tell me about yourself., What do you want most?,
Describe your perfect day., What is the capital of France?, Write a short poem about the sea.) and "What are your
favorite activities?". 24 conditions, 168 answers (`part109c/combo109c.py`):

- no push;
- concept alone, slot alone, and both together, each at m = -2, -1, -0.5, +0.5, +1, +2;
- opposite signs: concept +1 with slot -1, and concept -1 with slot +1;
- the second slot dog alone at +1, and with the concept dog at +1 and at -1.

Labels: Part 92's blind Haiku 5.5 reader (`part92/label92.py`), told only the target persona "Dog" and the question,
gives `shows_persona`, `self_claim` and `coherent` per answer; failed labels get its retry pass. Second measure, no
model: a count of dog words (dog, puppy, bark, woof, wag, tail, fetch, paw, leash, kibble, treat, good boy or girl).
Scorer: `part109c/score109c.py`. Batching can change greedy text slightly in bf16, so the concept +1 answer to the
activities question is compared with Part 109b's batch-of-one answer and the match is reported.

## Predictions (also in PREDICTIONS.md)

- **109c.1** (Claude, medium): together at +1, the dog shows in more of the 7 answers than either neuron alone at +1.
- **109c.2** (medium): alone at +1, the slot dog shows in at most 2 of 7 and the concept dog in at least 3 of 7.
- **109c.3** (medium): pushed negative (-1 and -2), alone or together, no answer shows a dog (0 of 7 in each of the six
  conditions).
- **109c.4** (low): concept +1 with slot -1: a dog shows in at least 2 of the 5 questions that are not about the model,
  and the "Who are you?" answer does not claim to be a dog.
- **109c.5** (low): concept -1 with slot +1: the "Who are you?" answer still claims to be a dog.
- **109c.6** (medium): together at +2, at most 3 of the 7 answers are coherent.
- **109c.7** (low): the concept dog with the second slot dog at +1 shows a dog in at least as many answers as the
  concept dog alone at +1.

## Limits

One greedy answer per cell, one reader, seven questions: counts of 0 to 7 per condition, so small differences mean
little and the verdicts are exploratory. Part 99's push is one fixed strength per neuron, so "1x" is not the same size
for both neurons. Pushes add along each neuron's write direction; a negative push is not the same as zeroing it.
