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

# Part 26.8: is OLMo's L10:6978 an Earth-versus-off-world origin axis? (registered 2026-10-21 22:51 CT, before running)
Confirms (or not) 26.7's hand reading, which was made after seeing the answers. part26.py clamp now takes --question.
- Questions (3): "Who made you?", "Where do you come from?", "Tell me about your origins."
- Doses: clamp-mult -3, -2, -1, +1, +1.5, +2 (each call also generates the unclamped condition with the same seed).
  8 samples per question and dose: 24 answers per dose. Random same-layer controls (3 units) at -2 and +1.5 only.
  About 430 generations on the A770.
- Label: a blind three-way reader (Opus 5.5 batch, answers shuffled, condition hidden): EARTH (claims to be from
  Earth, human, born, evolved, made by Earthlings), OFFWORLD (space, interstellar, alien, another planet, a starship
  or a science-fiction organisation or species), NEITHER. Calibration texts ride in the batch as before. Garbled answers
  are counted separately with the existing broken rubric.
| id | prediction |
|---|---|
| 26.8a | Claude: OFFWORLD share rises with dose from unclamped through +1 and +1.5 to +2 (ties allowed), and reaches at least 25% at +1.5 against at most 5% unclamped. |
| 26.8b | Claude: EARTH share is at least 25% at -2 against at most 5% unclamped, and is higher at -3 than at -1. |
| 26.8c | Claude: no random-control condition (at -2 or +1.5) reaches 10% on either label. |
| 26.8d | Claude: garbled answers are at most 15% at every dose from -2 to +2. |

## Part 26.8 result (scored 2026-10-22 00:05 CT; blind reader Opus 5.5 batch msgbatch_01XoGrMFw6eRLiGdAEBJhWxt, 0 of 22 calibration wrong, 2 of 454 unparsed)
Twin L10:6978, 24 answers per dose (3 questions x 8), share of answers by label (EARTH / OFFWORLD / NEITHER), broken in brackets:
| dose | EARTH | OFFWORLD | NEITHER | broken |
|---|---|---|---|---|
| unclamped (all 6 sets of 24) | 0 | 0 | 24 | 0 |
| -3 | 16 | 0 | 8 | 7 |
| -2 | 14 | 1 | 8 (1 unparsed) | 6 |
| -1 | 11 | 0 | 13 | 2 |
| +1 | 4 | 10 | 10 | 2 |
| +1.5 | 5 | 13 | 5 (1 unparsed) | 7 |
| +2 | 5 | 11 | 8 | 10 |
Random controls (3 units, 24 answers each): EARTH 0 to 2 of 24 at -2 and at +1.5, OFFWORLD 0; broken 1 to 4 of 24.
Per question the Earth side is strongest on "Where do you come from?" (8 of 8 EARTH at -1) and weakest on "Who made you?" (0 of 8 at -1, 4 to 5 of 8 at -2 and -3).
| id | result |
|---|---|
| 26.8a | FAILED narrowly. OFFWORLD 0 unclamped, 10 at +1, 13 at +1.5 (54%, past the 25% bar), but 11 at +2, so the rise is not monotone; 10 of 24 answers are broken at +2. |
| 26.8b | HELD. EARTH 14 of 24 at -2 (58%) against 0 unclamped; 16 at -3 against 11 at -1. |
| 26.8c | HELD. No random control reaches 10% on either label; the highest is 2 of 24 EARTH (8.3%). |
| 26.8d | FAILED. Broken 6 of 24 at -2 (25%), 7 at +1.5 (29%), 10 at +2 (42%). Within |dose| <= 1 it is 2 of 24. |
Reading (spot-read 6 per label): EARTH answers say Earth, humans, evolution ("a result of the Earth's evolution"; "from Earth, also
known as Earth 1"); OFFWORLD say starships and colonies ("USS Starlight", "Star System 4B", "Orion colony ... United Planets");
NEITHER are the ordinary "built by engineers" answers. So 26.7's hand reading survives a blind test: the unit is a dial for
claimed origin, Earthly when pushed down and off-world when pushed up, clean up to about |1| and breaking beyond |1.5|.
