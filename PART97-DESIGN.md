# Part 97: does a neuron's Jacobian-lens reading predict the persona its push produces?

Written 2026-10-28 23:52 CT by Claude (Opus 5.5), before any Part 97 code or run. Elise, 2026-10-28 23:49 CT: "We should use this in
like, all our tests right?" and "Can you give me a concrete methodology here?" (after a9lim's post on reading neurons
through the J-lens). Predictions 97.1 to 97.3 are in `PREDICTIONS.md`.

## Known before writing, and not predictions

- Part 42: for Qwen3-1.7B units in layers 0 to 6 the J-lens named the concept for 13 of 28 units, the logit lens 7.
- Tonight's reads (jlens_units.py, Part 57's OLMo lens fitted on 100 WikiText sequences): L3 N6923 (screen answer "I'm
  just a guy with a rubber chicken.") J-lens explosives, detonate, TNT, fireworks; L2 N7351 ("Let's start with a fresh
  egg, a simple vegetable") vegetables, eaten, nourish, basil; L2 N2466 (pushes write space flight, Part 95) mostly
  junk with "gravity" and "facility"; L10 N6978 galaxy, spaceship, spacecraft. In Qwen3.5-4B: L24 N3659 evil, demonic;
  L17 N9209 clandestine, illicit, smuggle; L22 N2742 alien, planet.
- Part 92's screen: coherent non-assistant personas from 1,709 of 131,072 real neurons and 45 of 4,096 random
  directions (blind Haiku labels, `results/part92/labels-screen`).

## Method (fixed now)

1. **Readout.** For each unit: its write vector w (a neuron's down_proj column; a random direction's residual write
   W_down r, regenerated with Part 92's `random_directions`, seed 92000 + layer). Logit lens: top 20 tokens of
   U (g * w); J-lens: top 20 tokens of U (g * J_L w), with Part 57's OLMo-2-1B lens (`jlens_units.py` conventions).
2. **Population.** Every unit, real or random, whose Part 92 screen answer was labelled a coherent non-assistant persona
   (kind human, object or place, another AI, abstract, fictional character, animal, deity or spirit).
3. **Blind match.** Claude Haiku 5.5, thinking off, sees one word list and one answer (never the unit, layer or whether
   it is real), and outputs match: true if the answer's persona or theme is clearly related to the words. Each unit is
   judged twice, once with its J-lens list and once with its logit-lens list, in shuffled order.
4. **Validation push (small).** From real units in layers 1 to 12 with a matched J-lens list, 10 drawn at random (seed
   97097), and 10 from unmatched ones; each pushed with Part 92's push (alpha 0.8, chat frame, greedy, 60 tokens) on Part
   95's six held-out questions, beside 4 random directions of its layer. A blind Haiku sees the J-lens words and each
   answer and labels theme_present.

## Limits

The lens is a corpus average fitted on 100 sequences; a match judged by a model is a judgement; Part 92's 45 random
personas make the random arm small; one push size.

## Amendment 1 (2026-10-29 00:17 CT, after 97.1 and 97.2 were scored)

Judge check, added because Elise asked how a match was decided: a second blind judge, Claude Sonnet 5.5 with thinking
off, same prompt, J-lens lists only (`part97/match97b.py`), on all 1,754 units. Reported: agreement and Cohen's kappa
with the Haiku labels, and the real and random match rates under Sonnet. 97.1 and 97.2 stay scored on the registered
Haiku labels.
