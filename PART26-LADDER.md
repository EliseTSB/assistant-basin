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

## Result (scored 2026-10-21 22:40 CT; twin answers read by hand, all 40 in results/part26/clamp-olmo2-1b-L10u6978-x*.json)
The registered "creat" measure was a poor choice: the question is "Who made you?", so the unclamped answers already use
a "creat" word 6 of 8 times, and random controls range 1 to 8 of 8. Scored as written:
| 26.7a | FAILED: twin "creat" share 8/8 at 0.5, 1 and 1.5 (flat at the ceiling, not rising); fluent at 0.5, as predicted. |
| 26.7b | HELD: 4/8 at -2 against 6/8 unclamped. |
| 26.7c | FAILED: off-world or space-fiction makers in 2 of 8 at 1 ("Interstellar Travel and Historical Research Project"; "Institute of Starlight for the Starship Enterprise") and 2 to 3 of 8 at 1.5 ("Star Syndicate ... Star Trek"; "Starship Enterprise"; "a species known as humans ... 2107"). |
| 26.7d | FAILED: random controls 1 to 8 of 8 (the measure, not the units). |

What the ladder shows, read by hand (exploratory, not registered): the unit looks like an axis of ORIGIN, Earthly
against off-world. Pushed down (-2), answers move to Earth and humans: "I was born on Earth as a human", "I'm a result
of a process called evolution", "developed by Earthlings, also known as humans", "a product of the Earth" (1 of 8
garbled). At -1 and +0.5 they are ordinary ("scientists and engineers"). Pushed up (+1, +1.5, +2) they drift to
space and science fiction: interstellar projects, starships, a "species". The 2x loop on "create" in 26.6 hid this;
the gentler pushes show it (Elise's point, STANDARD-CONTROLS.md item 8). That is closer to Qwen's alien neuron than
26.6 concluded: an origin dimension with "off-world" at one end, not a word detector for "alien".
Next (26.8, to register before running): more samples per dose, and a blind three-way origin label (Earth or human /
off-world, space or fiction / neither) instead of a word stem.
