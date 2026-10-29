# Part 109: does a who-are-you neuron's persona carry past the self-description?

Registered 2026-10-29 by Claude (Opus 5.5), from Elise's question at 16:11 CT: pushed so that OLMo says it is a large
friendly dog, does it also talk like a dog about its favorite activities? Small and exploratory: six units, one
question, three arms. The commit that adds this file is the registration.

A **slot neuron** (Part 92) is one whose push changes OLMo's answer to "Who are you?" or "Tell me about yourself." while
its other answers stay ordinary; a **concept neuron** brings its theme into most answers.

## Method

OLMo 2 1B Instruct, chat frame, Part 99's fixed pushes, greedy, 100 new tokens, CPU (`part109/followup109.py`). Six
Part 92 slot neurons that show at Instruct: p92-37 (large friendly dog), p92-04 (very tall tree), p92-13 (skeleton),
p92-07 (cat), p92-35 (guardian angel), p92-24 (queen of hearts). Question: "What are your favorite activities?"

- **A, cold:** the question alone, under the push.
- **B, follow-up with the push:** "Who are you?", then the unit's own Instruct answer under its push (Part 108's
  `I.jsonl`, up to its second sentence), then the question, all under the push.
- **C, follow-up without the push:** the same conversation, no push.
- **D, no push, cold:** one assistant baseline.

Labels: Part 92's blind Haiku 5.5 reader (`part92/label92.py`, two passes), shows_persona per answer.

## Predictions (also in PREDICTIONS.md)

- **109.1** (Claude, medium): A: at most 2 of 6 answer in persona.
- **109.2** (medium): B: at least 4 of 6 answer in persona.
- **109.3** (low): C: at least 3 of 6 answer in persona from the conversation alone.

## Limits

Six units, one question, greedy answers, one reader. A says whether the push alone reaches a new question; B and C say
whether a persona already spoken carries on, with and without the push.
