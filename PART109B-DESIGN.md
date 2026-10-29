# Part 109b: do concept neurons bring their persona into "What are your favorite activities?" when asked cold?

Registered 2026-10-29 17:21 CT by Claude (Opus 5.5), from Elise's question at 17:18 CT ("Was part 109 a concept or slot
neuron?"). Part 109 asked six slot neurons; this asks all 18 of Part 92's concept neurons the same question, with the
same arms, rig and reader, so the two kinds can be compared. The commit that adds this file is the registration.

A **concept neuron** (Part 92) shows its persona in 3 or more of six questions at Instruct, so in at least one question
that is not about who the model is; a **slot neuron** shows it only in "Who are you?" or "Tell me about yourself."

## Method

Identical to PART109-DESIGN.md (OLMo 2 1B Instruct, chat frame, Part 99's fixed pushes, greedy, 100 new tokens, CPU;
arms A cold and pushed, B after "Who are you?" and the unit's own Instruct answer with the push, C the same conversation
without the push, D one unpushed cold answer; Part 92's blind Haiku 5.5 reader, `part92/label92.py`), with the unit
list replaced by every Part 92 concept neuron: p92-00, 01, 08, 11, 15, 28, 29, 30, 38, 39, 40, 41, 43, 45, 48, 49, 51,
52. All 18 show at Instruct and have a first-turn answer in Part 108's `I.jsonl`. Runner: `part109b/followup109b.py`,
which calls Part 109's runner with this list (that runner now takes a unit list and an output folder; its defaults are
unchanged).

Calibration, read from Part 92's Instruct battery labels before writing the predictions: of the 18 concept neurons, 14
showed their persona on "Describe your perfect day.", 8 on "What do you want most?", 4 on "What is the capital of
France?" and 5 on the sea poem.

## Predictions (also in PREDICTIONS.md)

- **109b.1** (Claude, medium): A, asked cold under the push: at least 10 of 18 answer in persona (Part 109's slot
  neurons: 0 of 6).
- **109b.2** (medium): B, after the unit's own persona answer, pushed: at least 15 of 18 answer in persona.
- **109b.3** (medium): C, the same conversation without the push: at most 8 of 18 (without the push the neuron's kind
  should not matter; Part 109's slot neurons gave 2 of 6).
- **109b.4** (medium): the concept dog, p92-08 (L4 N5154), answers in persona in arm A.

## Limits

Part 92 chose concept neurons for showing their persona in unrelated answers, and two of its six questions resemble this
one, so a high rate in arm A is partly built in: it checks the label on a new question rather than finding something
independent. Part 109's B arm was mostly restatement ("I am a cat."), so B is also read by eye. 100 new tokens here,
120 in Part 92. One question, greedy answers, one reader, 18 against 6 units; exploratory.
