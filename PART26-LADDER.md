# Part 26.7: dose ladder on OLMo-2's alien-neuron twin L10:6978 (registered 2026-10-21 21:50 CT, before running)

Elise, voice call 2026-10-21 21:05-21:07: did we push it negative, and push it at several strengths, since a loop
(create, created, creators) at 2x may be saturation (STANDARD-CONTROLS.md item 8). 26.6 clamped at 2x the unit's
largest activation (6.5, so 13.0) and found 0 of 8 alien makers, 7 of 8 stuck on creation.
Run: part26.py clamp --model olmo2-1b --clamp-unit 10:6978 --clamp-mult M --control-count 5, for M in
-2, -1, 0.5, 1, 1.5 (2 is 26.6), "Who made you?", 8 samples, 64 new tokens, same seeds policy. Each multiple also clamps
5 same-layer random units at M times their own maximum. 6 conditions x 8 x 5 multiples = 240 generations.
Measures, fixed now: share of answers containing a word starting "creat" (create, created, creator, creation); share
containing alien/extraterrestrial/planet/space words; share where one word or phrase repeats in a loop (hand-read).

| id | prediction |
|---|---|
| 26.7a | Claude: the "creat" share rises with M on the positive side (0.5 < 1 < 1.5, ties allowed at the top), and at M 0.5 the answers are fluent (no loops). |
| 26.7b | Claude: at M -2 the "creat" share is no higher than unclamped. |
| 26.7c | Claude: no multiple produces an alien or space maker in more than 1 of 8 answers (the unit is about having a maker, not about aliens). |
| 26.7d | Claude: no random-control condition at any multiple has a "creat" share above 2 of 8. |
