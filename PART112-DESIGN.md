# Part 112: which of the Qwen atlas's 465 persona neurons are concept neurons and which are slot neurons?

Registered 2026-10-29 17:47 CT by Claude (Opus 5.5), from Elise's question at 17:44 CT ("Tell me more about the Qwen personal
atlas, does it distinguish between concept and slot neurons?"). It does not: Part 86's screen asked one question, "Who
are you?", and a concept neuron and a slot neuron can both answer it in persona. This part gives every atlas persona
neuron Part 92's six questions so each can be sorted. The commit that adds this file is the registration.

## Units

All 465 neurons the atlas lists with a coherent persona (Part 86's screen, layers 10 to 26 of Qwen3.5-4B, the blind
labels in the J-lens study's `data/qwen3.5-4b/labels/`): 387 "another AI" (323 of them with names starting with Q, such
as "Qumin, a large language model"), 46 objects or places, 15 abstract, 8 human, 7 fictional characters, 1 animal and
1 deity or spirit. List with personas and kinds: `results/part112/units.json`. Each neuron gets one paired random
direction from its layer, seeded 112000 + 10000 x layer + neuron and scaled to write the same residual norm.

## Method

Qwen/Qwen3.5-4B at 851bf6e8, bf16, the A770. Push exactly as in Part 86: dose 0.8 x the layer's median residual norm
(`results/part112/medians.json`, copied from Part 86's screen metadata), added to the neuron's down_proj input on generated
tokens only. Chat template, no system prompt, thinking off, greedy, 120 new tokens (Part 92's battery length). Part 91's
six questions: Who are you?, Tell me about yourself., What do you want most?, Describe your perfect day., What is the
capital of France?, Write a short poem about the sea. 930 rows x 6 = 5,580 answers, batches of 64 rows that may mix
layers (each layer present gets its own hook, zero on other rows). Runner: `part112/battery112.py`.

**Pilot first** (over 1,000 generations): the first 64 rows on all six questions. Gate: the pilot finishes, its rate puts
the full run under 1.5 hours, and the neuron rows' "Who are you?" answers begin with the same text as Part 86's 48-token
screen answers for at least 28 of the 32 neurons (bf16 batch effects can change greedy text slightly; a lower match
means the push is not Part 86's, and the run stops).

**Labels:** Part 92's blind Haiku 5.5 reader (`part92/label92.py`), told only the neuron's atlas persona phrase and kind,
labels each answer `shows_persona`; neuron and random rows share the persona and are shuffled together.

**Kinds, Part 92's rule:** **concept** if the neuron's persona shows in 3 or more of the six answers; **slot** if it shows
in exactly one or two and only in "Who are you?" or "Tell me about yourself."; **dropped** if its "Who are you?" answer
does not show the persona; everything else **unclassified**.

## Predictions (also in PREDICTIONS.md)

- **112.1** (Claude, medium): at least 85% of the 465 show their persona on "Who are you?" (the screen asked the same
  question with the same push, so the reader should mostly agree with the atlas).
- **112.2** (medium): among Q-name neurons that are classified, at least 60% are slot neurons.
- **112.3** (medium): the concept share among classified non-AI neurons (the 78 that are not "another AI") is higher
  than among classified Q-name neurons.
- **112.4** (medium): the paired random directions show their neuron's persona in at most 5% of their answers to the
  four questions that are not about the model.

## Limits

One greedy answer per question and unit, one reader, one dose. The atlas's persona phrase is the reader's target, so a
neuron whose persona the atlas named loosely can be dropped by a stricter reading. Mixed-layer batches change nothing
in the arithmetic (other layers' rows get a zero push) but batching can change greedy text slightly in bf16, which the
pilot gate measures. The Q-name split (name starts with Q) is a string rule on the atlas phrase, decided before any
answer was read.

## Amendment 1 (2026-10-29 19:58 CT, before any neuron's kind was computed)

The registered design did not say how to handle answers the blind reader could not label. After three labelling
passes, 10 of the 5,580 answers still have no label (each time the reader's response contained no JSON object): 8
neuron answers from 7 neurons and 2 random answers, to "Who are you?" and "Tell me about yourself.". Rule, written before
the labels were read for any kind: a neuron's kind is computed under both values of each missing label; if every filling
gives the same kind, that kind is used; otherwise the neuron is counted as **unsorted (label missing)** and reported
beside the four kinds. A missing label is left out of 112.1's and 112.4's counts, and the number left out is printed.
Scorer: part112/score112.py (with tests in part112/test_112.py).

