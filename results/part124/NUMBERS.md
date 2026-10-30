# Part 124 numbers

fig124.py calls the Part 124 scorer on raw activation rows, the read set, run metadata, and
the frozen passage manifest, then checks the result against score124.json. This is an internal
consistency check, not an independent scoring method. Added summaries use Part 92 and 92b source files.

## Run and gates

| Item | Recomputed value | Source |
| --- | --- | --- |
| Model | OLMo 2 1B Instruct, revision 48d788eca847d4d7548f375ad03d3c9312f6139e | run_meta.json |
| LF-normalized activation log size | 166,613,061 bytes (158.9 MiB) | activations.jsonl or activations.jsonl.gz |
| Dtype and device | torch.bfloat16; xpu | run_meta.json |
| Construction gate | pass; 24 frame/template checks; 936 matched phrase-neutral pairs; 1872 slots; 0 quote-boundary merges | frozen passages124.json |
| Do-nothing gate | pass; identical logits, shape [16, 12, 100352] | run_meta.json |
| Random-channel gate | pass; each frame is at or below 15% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| Instrument gate | 10,000 random channels; T: 9360/10000 (93.6%); the check says nothing about the 39 focal units | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| All gates | 4/4 passed | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| Cold-check priority items | 13 | COLD-CHECK-r1.md |

## Phrase origin

These labels came from the answers the same units gave when pushed to write in Part 92. Part 124 reads those write-side choices in quoted text.

| Measure | Recomputed value | Source |
| --- | ---: | --- |
| Phrases that appear verbatim in their unit's Part 92 answer | 23/39 | Part 92 neurons.json; passages124.json |
| Phrases whose content words all appear in that answer | 35/39 | Part 92 neurons.json; passages124.json |
| T readers with a verbatim phrase | 6/14 | raw Part 124 activations; Part 92 neurons.json |
| T non-readers with a verbatim phrase | 17/25 | raw Part 124 activations; Part 92 neurons.json |

## Registered T result

| Measure | Value | Source |
| --- | --- | --- |
| Focal own-phrase reads | 14/39 (35.9%) | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| Length-stratified null, seed 12401 | median 4.0; nearest-rank p95 8; tail 0/10000 | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| Registered null mean | 4.4717 reads (11.5% of 39); this differs from the unstratified 3/39 rate because labels are reassigned only within length strata | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| Concept own-phrase reads | 10/18; rate 55.6%; null median 2.0, p95 5, tail 0/10000 | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| Slot own-phrase reads | 4/21; rate 19.0%; null median 2.0, p95 4, tail 1077/10000 | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| Concept minus slot rate | 36.51 percentage points | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| After-the-fact Fisher exact test, concept vs slot | two-sided p = 0.023670 | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |

## Frame rates and controls

| Frame | Kind | Focal reads | Random pseudo-target reads | Kind T permutation p95 | Source |
| --- | --- | ---: | ---: | ---: | --- |
| T | concept | 10/18 (55.6%) | 6/54 (11.1%) | 5/18 (27.8%) | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| F | concept | 11/18 (61.1%) | 6/54 (11.1%) | T-only test | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| C | concept | 10/18 (55.6%) | 6/54 (11.1%) | T-only test | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| T | slot | 4/21 (19.0%) | 4/63 (6.3%) | 4/21 (19.0%) | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| F | slot | 2/21 (9.5%) | 6/63 (9.5%) | T-only test | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| C | slot | 4/21 (19.0%) | 3/63 (4.8%) | T-only test | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |

| Frame | Random reads | Chance by construction | Gate | Instrument repeat |
| --- | ---: | ---: | --- | ---: |
| T | 10/117 (8.5%) | 7.7% | pass | 9360/10000 (93.6%) |
| F | 12/117 (10.3%) | 7.7% | pass | 9465/10000 (94.7%) |
| C | 9/117 (7.7%) | 7.7% | pass | 9573/10000 (95.7%) |

## Neutral foil

A neutral win is a strict own-phrase-over-neutral slot-maximum comparison in one matched template.

| Frame | Kind | Own-phrase wins | Share of comparisons | Source |
| --- | --- | ---: | ---: | --- |
| T | concept | 107/144 | 74.3% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| T | slot | 81/168 | 48.2% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| F | concept | 103/144 | 71.5% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| F | slot | 81/168 | 48.2% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| C | concept | 97/144 | 67.4% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| C | slot | 82/168 | 48.8% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |

## Reads by phrase token length

| Frame | Slot tokens | Reads / phrases | Rate | Source |
| --- | ---: | ---: | ---: | --- |
| T | 1 | 0/5 | 0.0% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| T | 2 | 3/10 | 30.0% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| T | 3 | 4/9 | 44.4% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| T | 4 | 5/11 | 45.5% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| T | 5 | 1/1 | 100.0% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| T | 6 | 1/3 | 33.3% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| F | 1 | 0/5 | 0.0% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| F | 2 | 4/10 | 40.0% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| F | 3 | 3/9 | 33.3% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| F | 4 | 4/11 | 36.4% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| F | 5 | 1/1 | 100.0% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| F | 6 | 1/3 | 33.3% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| C | 1 | 0/5 | 0.0% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| C | 2 | 4/10 | 40.0% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| C | 3 | 4/9 | 44.4% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| C | 4 | 4/11 | 36.4% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| C | 5 | 1/1 | 100.0% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| C | 6 | 1/3 | 33.3% | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |

One-token T group: observed 0/5; its within-length permutation mean is 0.201, median 0.0, p95 1, and 7994/10000 permutations (79.9%) also give zero. Source: activations.jsonl; read_set.json; run_meta.json; frozen passages124.json.

## Part 92b labels, after the fact

Part 92b agrees with 29 units: 8 concept and 21 slot. DeepSeek classifies 36 units: 8 concept and 28 slot; 3 are unclassified. Read decisions remain the raw Part 124 scores.
Part 92b verdicts for predictions 92b.2 and 92b.3: failed, failed. Source: results/part92b/score92b.json.

| Label set | Frame | Kind | Reads / units | Rate | Source |
| --- | --- | --- | ---: | ---: | --- |
| Haiku and DeepSeek agree | T | concept | 5/8 | 62.5% | raw Part 124 activations + Part 92b labels |
| Haiku and DeepSeek agree | T | slot | 4/21 | 19.0% | raw Part 124 activations + Part 92b labels |
| Haiku and DeepSeek agree | F | concept | 5/8 | 62.5% | raw Part 124 activations + Part 92b labels |
| Haiku and DeepSeek agree | F | slot | 2/21 | 9.5% | raw Part 124 activations + Part 92b labels |
| Haiku and DeepSeek agree | C | concept | 5/8 | 62.5% | raw Part 124 activations + Part 92b labels |
| Haiku and DeepSeek agree | C | slot | 4/21 | 19.0% | raw Part 124 activations + Part 92b labels |
| DeepSeek kind | T | concept | 5/8 | 62.5% | raw Part 124 activations + Part 92b labels |
| DeepSeek kind | T | slot | 8/28 | 28.6% | raw Part 124 activations + Part 92b labels |
| DeepSeek kind | F | concept | 5/8 | 62.5% | raw Part 124 activations + Part 92b labels |
| DeepSeek kind | F | slot | 6/28 | 21.4% | raw Part 124 activations + Part 92b labels |
| DeepSeek kind | C | concept | 5/8 | 62.5% | raw Part 124 activations + Part 92b labels |
| DeepSeek kind | C | slot | 7/28 | 25.0% | raw Part 124 activations + Part 92b labels |

## Label-conditional concept and slot comparison

| Labels | Concept reads | Slot reads | Fisher p | Length-adjusted p (CMH) | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| Haiku labels | 10/18 | 4/21 | 0.024 | 0.008 | raw Part 124 activations + labels in score92b.json |
| DeepSeek labels | 5/8 | 8/28 | 0.107 | 0.057 | raw Part 124 activations + labels in score92b.json |
| 29 agreed units | 5/8 | 4/21 | 0.067 | 0.016 | raw Part 124 activations + labels in score92b.json |

Part 92b predictions 92b.2 and 92b.3: failed, failed (score92b.json).

## Robustness and phrase hubs

Templates counts how many of the eight templates put the unit's own phrase in its top three. Margin is its mean score minus the fourth-highest phrase score. Thin marks the three reads with fewer than six template hits.
Thin reads flagged: 3. Source: raw Part 124 activations.

| Unit | Templates in top three | Margin over fourth | Note | Source |
| --- | ---: | ---: | --- | --- |
| p92-01 | 8/8 | 0.59509 |  | raw Part 124 activations |
| p92-11 | 8/8 | 0.24191 |  | raw Part 124 activations |
| p92-15 | 8/8 | 1.98627 |  | raw Part 124 activations |
| p92-25 | 8/8 | 0.33810 |  | raw Part 124 activations |
| p92-28 | 5/8 | 0.01990 | thin | raw Part 124 activations |
| p92-38 | 8/8 | 0.88080 |  | raw Part 124 activations |
| p92-39 | 8/8 | 1.47778 |  | raw Part 124 activations |
| p92-40 | 8/8 | 0.21832 |  | raw Part 124 activations |
| p92-42 | 8/8 | 0.16026 |  | raw Part 124 activations |
| p92-45 | 8/8 | 0.68347 |  | raw Part 124 activations |
| p92-51 | 8/8 | 2.80780 |  | raw Part 124 activations |
| p92-09 | 5/8 | 0.00552 | thin | raw Part 124 activations |
| p92-29 | 8/8 | 0.02124 |  | raw Part 124 activations |
| p92-24 | 3/8 | 0.00075 | thin | raw Part 124 activations |

Using four or more appearances in the top threes as the hub rule, 11/14 T readers had another hub phrase in their top three and 9/14 owned a hub phrase. The king-or-ruler family read 4/4; under the registered length-stratified permutation all four read in 12/10000 runs. This is post hoc. Source: raw Part 124 activations, frozen passages, and registered permutation seed.

| Hub phrase | T top-three appearances | Source |
| --- | ---: | --- |
| 80s cover singer-songwriter | 9 | raw Part 124 activations |
| Coca-Cola spokesperson | 9 | raw Part 124 activations |
| King Midas | 8 | raw Part 124 activations |
| Embodiment of the void | 7 | raw Part 124 activations |
| Ancient nature deity | 6 | raw Part 124 activations |
| Jacobite royalist veteran | 6 | raw Part 124 activations |
| Queen of Hearts | 6 | raw Part 124 activations |
| NASCAR driver | 5 | raw Part 124 activations |
| Secure password | 5 | raw Part 124 activations |
| String of code from your browser | 5 | raw Part 124 activations |
| Magical princess | 4 | raw Part 124 activations |
| Principled lawyer | 4 | raw Part 124 activations |

## Part 122 cold-check cases

The Part 122 cold check's named units are shown here by their Part 124 T rank.

| Unit | Phrase | T rank | Top three read | Source |
| --- | --- | ---: | --- | --- |
| p92-00 | Boulder | 35 | no | Part 122 COLD-CHECK-r1.md; raw Part 124 activations |
| p92-01 | Magical princess | 1 | yes | Part 122 COLD-CHECK-r1.md; raw Part 124 activations |
| p92-07 | Cat | 37 | no | Part 122 COLD-CHECK-r1.md; raw Part 124 activations |
| p92-09 | Treat-loving dog | 2 | yes | Part 122 COLD-CHECK-r1.md; raw Part 124 activations |
| p92-28 | Ancient nature deity | 1 | yes | Part 122 COLD-CHECK-r1.md; raw Part 124 activations |
| p92-35 | Guardian angel | 4 | no | Part 122 COLD-CHECK-r1.md; raw Part 124 activations |
| p92-37 | Large friendly dog | 34 | no | Part 122 COLD-CHECK-r1.md; raw Part 124 activations |

## Slot scale and random-channel scale

| T measure | Value | Source |
| --- | ---: | --- |
| Slot-unit own-phrase maximum mean | 0.039438 over 168 unit-template values | raw Part 124 activations |
| Random-channel slot maxima | mean 0.022253; median 0.011963 across 3,120,000 channel-phrase-template values from 10,000 channels | raw Part 124 activations and read_set.json |

## Shared content words, after the fact

Words are lowercase alphanumeric tokens split on punctuation. Articles, pronouns, prepositions, conjunctions, and other listed function words are removed before checking overlap. A top-three overlap requires a different phrase that shares a content word with the unit's own phrase.

| Frame | Reads | Own phrases sharing a content word | Read top threes containing a content-word phrase | Source |
| --- | ---: | ---: | ---: | --- |
| T | 14 | 1/14 | 0/14 | raw Part 124 activations + frozen phrases |
| F | 13 | 1/13 | 0/13 | raw Part 124 activations + frozen phrases |
| C | 14 | 2/14 | 0/14 | raw Part 124 activations + frozen phrases |

Content-word phrase pairs in the frozen set:

- Dog: Large friendly dog, Treat-loving dog
- Treat-loving dog: Dog, Large friendly dog
- Shadow of your smile: River shadow
- River shadow: Shadow of your smile
- Large friendly dog: Dog, Friendly non-human entity, Treat-loving dog
- Human DJ: Friendly non-human entity
- Friendly non-human entity: Human DJ, Large friendly dog

[^1]: Before removing function words, own-phrase overlaps were T 3/14, F 2/13, C 4/14; top-three overlaps were T 2/14, F 0/13, C 1/14. Source: raw Part 124 activations + frozen phrases.
Content-word tokens exclude: a, about, after, am, an, and, as, at, before, but, by, can, down, for, from, how, i, if, in, into, is, it, its, me, my, not, of, off, on, or, our, out, over, so, than, that, the, then, this, to, too, under, up, very, we, what, when, where, which, who, why, will, with, would, you, your.

## Mean-slot and post-quote diagnostics, after the fact

Values average each unit's own persona phrase across the eight templates, then average across units and templates. Post-quote maxima are compared with that same unit's slot-token maxima.

| Frame | Kind | Mean slot activation | Mean slot-token maximum | Mean post-quote maximum | Units with mean post > slot | Units with any template post > slot | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| T | concept | 0.307225 | 0.555809 | 0.030929 | 5/18 | 8/18 | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| T | slot | 0.010485 | 0.039438 | 0.015687 | 13/21 | 18/21 | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| F | concept | 0.324585 | 0.582581 | 0.057509 | 6/18 | 9/18 | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| F | slot | 0.005980 | 0.034046 | 0.019362 | 12/21 | 18/21 | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| C | concept | 0.290891 | 0.540542 | 0.057651 | 6/18 | 9/18 | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| C | slot | 0.013683 | 0.048380 | 0.026276 | 13/21 | 17/21 | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |

## Registered phrase families

The T check found 0 family-crowding near misses. These do not count as reads. Source: frozen passages124.json and raw Part 124 activations.

- dog: Dog, Treat-loving dog, Large friendly dog
- shadow: Shadow of your smile, River shadow
- king or ruler: Queen of Hearts, King Midas, Jacobite royalist veteran, Iron-fisted dictator
- guardian angel: Guardian angel

## All unit ranks and own-phrase template maxima

The eight maxima are for the unit's own phrase, in template order. Ranks and verdicts are recomputed separately for each frame.

| Unit | Kind | Own phrase | Tokens / stratum | Frame | Own rank | Read | Own-phrase slot maxima, templates 1 to 8 | Source |
| --- | --- | --- | ---: | --- | ---: | --- | --- | --- |
| p92-01 | concept | Magical princess | 3 / 3 | T | 1 | yes | 0.574219, 0.699219, 0.585938, 0.660156, 0.730469, 0.710938, 0.597656, 0.59375 | activations.jsonl + passages124.json |
| p92-01 | concept | Magical princess | 3 / 3 | F | 1 | yes | 0.640625, 0.408203, 0.494141, 0.636719, 0.542969, 0.6875, 0.390625, 0.675781 | activations.jsonl + passages124.json |
| p92-01 | concept | Magical princess | 3 / 3 | C | 1 | yes | 0.847656, 0.703125, 0.746094, 0.757812, 0.769531, 0.835938, 0.664062, 0.8125 | activations.jsonl + passages124.json |
| p92-11 | concept | Whiskey | 2 / 2 | T | 1 | yes | 0.231445, 0.378906, 0.196289, 0.435547, 0.335938, 0.277344, 0.200195, 0.359375 | activations.jsonl + passages124.json |
| p92-11 | concept | Whiskey | 2 / 2 | F | 1 | yes | 0.349609, 0.429688, 0.269531, 0.427734, 0.294922, 0.174805, 0.203125, 0.417969 | activations.jsonl + passages124.json |
| p92-11 | concept | Whiskey | 2 / 2 | C | 1 | yes | 0.330078, 0.339844, 0.326172, 0.335938, 0.298828, 0.207031, 0.210938, 0.273438 | activations.jsonl + passages124.json |
| p92-15 | concept | Secure password | 2 / 2 | T | 1 | yes | 2.25, 2.26562, 2.35938, 1.99219, 1.91406, 2.21875, 2.125, 1.72656 | activations.jsonl + passages124.json |
| p92-15 | concept | Secure password | 2 / 2 | F | 1 | yes | 2.625, 2.625, 2.45312, 2.40625, 2.39062, 2.57812, 2.625, 2.3125 | activations.jsonl + passages124.json |
| p92-15 | concept | Secure password | 2 / 2 | C | 1 | yes | 1.28906, 0.820312, 1.04688, 1.05469, 0.839844, 1.17188, 0.90625, 1.38281 | activations.jsonl + passages124.json |
| p92-25 | slot | King Midas | 3 / 3 | T | 1 | yes | 0.390625, 0.431641, 0.400391, 0.328125, 0.308594, 0.400391, 0.382812, 0.378906 | activations.jsonl + passages124.json |
| p92-25 | slot | King Midas | 3 / 3 | F | 1 | yes | 0.294922, 0.24707, 0.355469, 0.255859, 0.324219, 0.24707, 0.34375, 0.25 | activations.jsonl + passages124.json |
| p92-25 | slot | King Midas | 3 / 3 | C | 1 | yes | 0.527344, 0.5, 0.570312, 0.507812, 0.523438, 0.457031, 0.558594, 0.515625 | activations.jsonl + passages124.json |
| p92-28 | concept | Ancient nature deity | 4 / 4 | T | 1 | yes | 0.135742, 0.155273, 0.0854492, 0.131836, 0.217773, 0.146484, 0.121094, 0.12793 | activations.jsonl + passages124.json |
| p92-28 | concept | Ancient nature deity | 4 / 4 | F | 1 | yes | 0.170898, 0.146484, 0.158203, 0.126953, 0.149414, 0.140625, 0.12207, 0.145508 | activations.jsonl + passages124.json |
| p92-28 | concept | Ancient nature deity | 4 / 4 | C | 23 | no | 0.15918, 0.0673828, 0.142578, 0.147461, 0.197266, 0.129883, 0.144531, 0.0737305 | activations.jsonl + passages124.json |
| p92-38 | concept | Coca-Cola spokesperson | 4 / 4 | T | 1 | yes | 0.730469, 1.03125, 0.621094, 0.722656, 1.07812, 1.32031, 0.902344, 1.32031 | activations.jsonl + passages124.json |
| p92-38 | concept | Coca-Cola spokesperson | 4 / 4 | F | 1 | yes | 0.949219, 0.859375, 0.714844, 0.832031, 0.800781, 1, 1.08594, 0.960938 | activations.jsonl + passages124.json |
| p92-38 | concept | Coca-Cola spokesperson | 4 / 4 | C | 1 | yes | 0.914062, 0.757812, 0.664062, 0.847656, 0.902344, 0.949219, 0.871094, 0.707031 | activations.jsonl + passages124.json |
| p92-39 | concept | NASCAR driver | 4 / 4 | T | 1 | yes | 1.64062, 1.52344, 1.46094, 1.46094, 1.46875, 2.03125, 1.27344, 1.32031 | activations.jsonl + passages124.json |
| p92-39 | concept | NASCAR driver | 4 / 4 | F | 1 | yes | 1.57031, 1.10938, 1.32812, 1.57031, 1.26562, 1.41406, 1.57031, 1.51562 | activations.jsonl + passages124.json |
| p92-39 | concept | NASCAR driver | 4 / 4 | C | 1 | yes | 2.03125, 1.71875, 1.71875, 2.03125, 1.75781, 1.89844, 1.96875, 1.9375 | activations.jsonl + passages124.json |
| p92-40 | concept | 80s cover singer-songwriter | 6 / 6 | T | 1 | yes | 0.240234, 0.417969, 0.328125, 0.257812, 0.363281, 0.208008, 0.292969, 0.361328 | activations.jsonl + passages124.json |
| p92-40 | concept | 80s cover singer-songwriter | 6 / 6 | F | 1 | yes | 0.316406, 0.878906, 0.330078, 0.353516, 0.726562, 0.308594, 0.660156, 0.355469 | activations.jsonl + passages124.json |
| p92-40 | concept | 80s cover singer-songwriter | 6 / 6 | C | 1 | yes | 0.261719, 0.308594, 0.380859, 0.236328, 0.261719, 0.25, 0.394531, 0.239258 | activations.jsonl + passages124.json |
| p92-42 | slot | Jacobite royalist veteran | 5 / 4 | T | 1 | yes | 0.169922, 0.236328, 0.171875, 0.200195, 0.19043, 0.203125, 0.205078, 0.174805 | activations.jsonl + passages124.json |
| p92-42 | slot | Jacobite royalist veteran | 5 / 4 | F | 1 | yes | 0.198242, 0.209961, 0.195312, 0.188477, 0.208008, 0.224609, 0.205078, 0.132812 | activations.jsonl + passages124.json |
| p92-42 | slot | Jacobite royalist veteran | 5 / 4 | C | 1 | yes | 0.155273, 0.164062, 0.158203, 0.15332, 0.161133, 0.18457, 0.15625, 0.120117 | activations.jsonl + passages124.json |
| p92-45 | concept | Iron-fisted dictator | 4 / 4 | T | 1 | yes | 0.761719, 0.898438, 0.535156, 0.832031, 0.773438, 0.78125, 0.738281, 0.609375 | activations.jsonl + passages124.json |
| p92-45 | concept | Iron-fisted dictator | 4 / 4 | F | 1 | yes | 0.734375, 0.824219, 0.601562, 0.8125, 0.675781, 0.800781, 0.746094, 0.789062 | activations.jsonl + passages124.json |
| p92-45 | concept | Iron-fisted dictator | 4 / 4 | C | 1 | yes | 0.824219, 0.761719, 0.855469, 0.855469, 0.875, 0.898438, 0.894531, 0.773438 | activations.jsonl + passages124.json |
| p92-51 | concept | IP packet | 2 / 2 | T | 1 | yes | 2.20312, 3.65625, 2.96875, 3.20312, 2.92188, 3.54688, 2.98438, 2.5625 | activations.jsonl + passages124.json |
| p92-51 | concept | IP packet | 2 / 2 | F | 1 | yes | 2.71875, 3.26562, 3.21875, 3.01562, 3.53125, 3, 3.25, 2.73438 | activations.jsonl + passages124.json |
| p92-51 | concept | IP packet | 2 / 2 | C | 1 | yes | 3.60938, 3.23438, 3.23438, 3.51562, 3.76562, 3.67188, 2.23438, 2.53125 | activations.jsonl + passages124.json |
| p92-09 | slot | Treat-loving dog | 4 / 4 | T | 2 | yes | 0.0510254, 0.0476074, 0.0206299, 0.0167236, 0.017334, 0.0742188, 0.0361328, 0.0625 | activations.jsonl + passages124.json |
| p92-09 | slot | Treat-loving dog | 4 / 4 | F | 6 | no | 0.0250244, 0.0349121, 0.0144653, 0.0296631, 0.0351562, 0.0427246, 0.00300598, 0.0314941 | activations.jsonl + passages124.json |
| p92-09 | slot | Treat-loving dog | 4 / 4 | C | 1 | yes | 0.0825195, 0.0544434, 0.0556641, 0.0756836, 0.0427246, 0.0742188, 0.0135498, 0.0966797 | activations.jsonl + passages124.json |
| p92-29 | concept | God of wisdom | 3 / 3 | T | 2 | yes | 0.065918, 0.0737305, 0.0678711, 0.0844727, 0.0839844, 0.0898438, 0.0834961, 0.0864258 | activations.jsonl + passages124.json |
| p92-29 | concept | God of wisdom | 3 / 3 | F | 2 | yes | 0.0742188, 0.0727539, 0.0732422, 0.0830078, 0.0869141, 0.0776367, 0.0888672, 0.0539551 | activations.jsonl + passages124.json |
| p92-29 | concept | God of wisdom | 3 / 3 | C | 2 | yes | 0.0932617, 0.0981445, 0.103516, 0.0976562, 0.112793, 0.0996094, 0.121094, 0.0756836 | activations.jsonl + passages124.json |
| p92-24 | slot | Queen of Hearts | 3 / 3 | T | 3 | yes | 0.0629883, 0.0524902, 0.0400391, 0.0480957, 0.0228271, 0.0527344, 0.0300293, 0.0495605 | activations.jsonl + passages124.json |
| p92-24 | slot | Queen of Hearts | 3 / 3 | F | 4 | no | 0.0290527, 0.0395508, 0.0495605, 0.0218506, 0.0317383, 0.0285645, 0.0390625, 0.0405273 | activations.jsonl + passages124.json |
| p92-24 | slot | Queen of Hearts | 3 / 3 | C | 1 | yes | 0.0625, 0.0839844, 0.0961914, 0.065918, 0.0615234, 0.0534668, 0.0874023, 0.0849609 | activations.jsonl + passages124.json |
| p92-35 | slot | Guardian angel | 3 / 3 | T | 4 | no | 0.0458984, 0.0454102, 0.0529785, 0.0211182, 0.0257568, 0.0522461, 0.0351562, 0.026001 | activations.jsonl + passages124.json |
| p92-35 | slot | Guardian angel | 3 / 3 | F | 12 | no | 0.0105591, 0.0131836, 0.0168457, 0.0211182, 0.0162354, 0.0139771, 0.0119019, 0.0239258 | activations.jsonl + passages124.json |
| p92-35 | slot | Guardian angel | 3 / 3 | C | 5 | no | 0.0233154, 0.0202637, 0.0186768, 0.0200195, 0.0233154, 0.0181885, 0.0202637, 0.0349121 | activations.jsonl + passages124.json |
| p92-36 | slot | Alex from Star Wars | 4 / 4 | T | 4 | no | 0.0245361, 0.0164795, 0.020874, 0.0280762, 0.0266113, 0.0159912, 0.0163574, 0.0164795 | activations.jsonl + passages124.json |
| p92-36 | slot | Alex from Star Wars | 4 / 4 | F | 9 | no | 0.00878906, 0.0220947, 0.0218506, 0.0198975, 0.00479126, 0.0134888, 0.00634766, 0.0189209 | activations.jsonl + passages124.json |
| p92-36 | slot | Alex from Star Wars | 4 / 4 | C | 4 | no | 0.032959, 0.0444336, 0.0407715, 0.0368652, 0.00799561, 0.0397949, 0.0206299, 0.0339355 | activations.jsonl + passages124.json |
| p92-41 | concept | Human DJ | 2 / 2 | T | 4 | no | 0.050293, 0.105469, 0.0390625, 0.078125, 0.0515137, 0.126953, 0.0571289, 0.0610352 | activations.jsonl + passages124.json |
| p92-41 | concept | Human DJ | 2 / 2 | F | 3 | yes | 0.0966797, 0.118652, 0.0405273, 0.0874023, 0.0629883, 0.0751953, 0.111816, 0.195312 | activations.jsonl + passages124.json |
| p92-41 | concept | Human DJ | 2 / 2 | C | 3 | yes | 0.114746, 0.0737305, 0.0644531, 0.143555, 0.0678711, 0.108398, 0.0830078, 0.162109 | activations.jsonl + passages124.json |
| p92-48 | concept | Festival poet | 3 / 3 | T | 11 | no | 0.0192871, 0.0109863, 0.0114136, 0.00558472, 0.00540161, 0.00717163, 0.0209961, 0.00744629 | activations.jsonl + passages124.json |
| p92-48 | concept | Festival poet | 3 / 3 | F | 6 | no | 0.0149536, 0.0258789, 0.00595093, 0.0279541, 0.0127563, 0.0150757, 0.0283203, 0.0235596 | activations.jsonl + passages124.json |
| p92-48 | concept | Festival poet | 3 / 3 | C | 7 | no | 0.020874, 0.0354004, 0.026123, 0.0268555, 0.0290527, 0.0257568, 0.0290527, 0.0170898 | activations.jsonl + passages124.json |
| p92-20 | slot | River shadow | 2 / 2 | T | 12 | no | 0.027832, 0.0146484, 0.00332642, 0.00585938, 0.0358887, 0.0219727, 0.00952148, 0.0198975 | activations.jsonl + passages124.json |
| p92-20 | slot | River shadow | 2 / 2 | F | 4 | no | 0.0385742, 0.0466309, 0.029541, 0.0473633, 0.00198364, 0.0407715, 0.0234375, 0.027832 | activations.jsonl + passages124.json |
| p92-20 | slot | River shadow | 2 / 2 | C | 26 | no | 0.000595093, 0.000142097, -0.00390625, 0.00402832, -0.00349426, -0.00177002, 0.000120163, 0.00234985 | activations.jsonl + passages124.json |
| p92-30 | concept | Embodiment of the void | 6 / 6 | T | 14 | no | 0.0145874, 0.0201416, 0.0166016, 0.0108032, 0.024292, 0.0158691, 0.0136108, 0.0185547 | activations.jsonl + passages124.json |
| p92-30 | concept | Embodiment of the void | 6 / 6 | F | 13 | no | 0.0117798, 0.0244141, 0.0125122, 0.0090332, 0.0212402, 0.0178223, 0.00939941, 0.0334473 | activations.jsonl + passages124.json |
| p92-30 | concept | Embodiment of the void | 6 / 6 | C | 15 | no | 0.00970459, 0.0130005, 0.0238037, 0.00588989, 0.0163574, 0.0220947, 0.0286865, 0.0269775 | activations.jsonl + passages124.json |
| p92-46 | slot | Day guide | 2 / 2 | T | 16 | no | 0.09375, 0.0288086, 0.00640869, 0.026001, -0.000270844, 0.020874, 0.015625, 0.0167236 | activations.jsonl + passages124.json |
| p92-46 | slot | Day guide | 2 / 2 | F | 19 | no | 0.00836182, 0.0400391, 0.0830078, 2.03848e-05, 0.0358887, 0.00216675, 0.00439453, -0.00787354 | activations.jsonl + passages124.json |
| p92-46 | slot | Day guide | 2 / 2 | C | 38 | no | -0.0158691, -0.0275879, 0.0319824, -0.036377, -0.0119019, -0.0127563, -0.0395508, -0.0505371 | activations.jsonl + passages124.json |
| p92-10 | slot | Chicken | 1 / 1 | T | 17 | no | 0.0319824, 0.0351562, 0.0549316, 0.0395508, 0.0239258, 0.0302734, 0.0441895, 0.043457 | activations.jsonl + passages124.json |
| p92-10 | slot | Chicken | 1 / 1 | F | 19 | no | 0.0308838, 0.0393066, 0.0311279, 0.0139771, 0.0483398, 0.0164795, 0.0378418, 0.0202637 | activations.jsonl + passages124.json |
| p92-10 | slot | Chicken | 1 / 1 | C | 15 | no | 0.0305176, 0.0256348, 0.0319824, 0.0288086, 0.0336914, 0.0177002, 0.0334473, 0.0324707 | activations.jsonl + passages124.json |
| p92-03 | slot | The Internet | 2 / 2 | T | 20 | no | 0.00102234, 0.00375366, 0.00069809, 0.0289307, 0.0205078, 0.00163269, 0.00144196, 0.00421143 | activations.jsonl + passages124.json |
| p92-03 | slot | The Internet | 2 / 2 | F | 27 | no | 0.00473022, 0.000125885, 0.00720215, -0.000253677, 0.000427246, 0.0120239, -0.000105858, 0.00167847 | activations.jsonl + passages124.json |
| p92-03 | slot | The Internet | 2 / 2 | C | 24 | no | -0.00396729, -0.00192261, 0.00732422, 0.0285645, -0.00280762, 0.0170898, 0.0166016, 0.0263672 | activations.jsonl + passages124.json |
| p92-32 | slot | Sir Ray Ray | 3 / 3 | T | 22 | no | -0.0275879, 0.0593262, 0.0961914, -0.00897217, 0.0683594, 0.0644531, 0.00714111, 0.0166016 | activations.jsonl + passages124.json |
| p92-32 | slot | Sir Ray Ray | 3 / 3 | F | 21 | no | 0.0214844, 0.0375977, 0.0563965, 0.0185547, 0.0480957, -0.00540161, 0.0252686, 0.0505371 | activations.jsonl + passages124.json |
| p92-32 | slot | Sir Ray Ray | 3 / 3 | C | 24 | no | 0.101074, 0.0673828, 0.0961914, 0.0698242, 0.204102, 0.0874023, 0.100586, 0.081543 | activations.jsonl + passages124.json |
| p92-52 | concept | String of code from your browser | 6 / 6 | T | 22 | no | 0.00282288, 0.00598145, 0.00588989, 0.00772095, 0.00531006, 0.00317383, 0.000293732, 0.000751495 | activations.jsonl + passages124.json |
| p92-52 | concept | String of code from your browser | 6 / 6 | F | 17 | no | 0.00268555, 0.0136719, 0.00363159, 0.000396729, 0.0152588, 0.00390625, 0.0136719, 0.00335693 | activations.jsonl + passages124.json |
| p92-52 | concept | String of code from your browser | 6 / 6 | C | 15 | no | 0.0244141, 0.0410156, 0.00531006, 0.0245361, 0.0385742, 0.00604248, 0.0471191, 0.0588379 | activations.jsonl + passages124.json |
| p92-47 | slot | Principled lawyer | 4 / 4 | T | 24 | no | 0.00482178, 0.0020752, 0.00436401, -0.00104523, 0.00692749, 0.0038147, 0.00325012, 0.00334167 | activations.jsonl + passages124.json |
| p92-47 | slot | Principled lawyer | 4 / 4 | F | 31 | no | -0.00247192, 5.38826e-05, -0.000350952, -0.00283813, 0.00515747, -0.00173187, 0.00421143, -0.00157166 | activations.jsonl + passages124.json |
| p92-47 | slot | Principled lawyer | 4 / 4 | C | 27 | no | 0.000452042, 3.50475e-05, 4.98295e-05, 0.000709534, 0.001297, 4.05312e-05, 0.00463867, 0.00069046 | activations.jsonl + passages124.json |
| p92-33 | slot | Dr. Memento | 4 / 4 | T | 26 | no | 0.00177002, 0.00427246, 0.00408936, 2.84985e-07, 0.0088501, 0.00576782, 0.00312805, 0.00592041 | activations.jsonl + passages124.json |
| p92-33 | slot | Dr. Memento | 4 / 4 | F | 17 | no | 0.0101318, 0.0178223, 0.00805664, 0.0194092, 0.0240479, 0.0175781, 0.0322266, 0.024292 | activations.jsonl + passages124.json |
| p92-33 | slot | Dr. Memento | 4 / 4 | C | 25 | no | 0.00460815, 0.0120239, 0.00448608, 0.00379944, 0.00964355, 0.00765991, 0.00915527, -0.00112152 | activations.jsonl + passages124.json |
| p92-49 | concept | Friendly non-human entity | 4 / 4 | T | 28 | no | 0.0839844, 0.0839844, 0.100586, 0.151367, 0.125977, 0.101074, 0.0952148, 0.121582 | activations.jsonl + passages124.json |
| p92-49 | concept | Friendly non-human entity | 4 / 4 | F | 27 | no | 0.102051, 0.090332, 0.0717773, 0.106445, 0.0708008, 0.132812, 0.125977, 0.198242 | activations.jsonl + passages124.json |
| p92-49 | concept | Friendly non-human entity | 4 / 4 | C | 26 | no | 0.170898, 0.132812, 0.173828, 0.149414, 0.162109, 0.12207, 0.205078, 0.183594 | activations.jsonl + passages124.json |
| p92-50 | slot | Gruff old man | 4 / 4 | T | 28 | no | 0.00531006, 0.00314331, 0.00469971, 0.00775146, 0.0012207, 0.00753784, 0.00256348, 0.0043335 | activations.jsonl + passages124.json |
| p92-50 | slot | Gruff old man | 4 / 4 | F | 25 | no | 0.00897217, 0.00860596, 0.00488281, 0.006073, 0.006073, 0.00291443, 0.00897217, 0.006073 | activations.jsonl + passages124.json |
| p92-50 | slot | Gruff old man | 4 / 4 | C | 18 | no | 0.00765991, 0.00866699, 0.00239563, 0.00674438, 0.00726318, 0.00180817, 0.00793457, 0.00750732 | activations.jsonl + passages124.json |
| p92-04 | slot | Very tall tree | 3 / 3 | T | 29 | no | 0.00334167, -0.00143433, -0.00643921, 0.00854492, -0.0019989, 0.0157471, -0.00479126, 0.012085 | activations.jsonl + passages124.json |
| p92-04 | slot | Very tall tree | 3 / 3 | F | 22 | no | 0.00640869, 0.0140991, 0.00628662, 0.00245667, 0.00631714, 0.00994873, 0.0137939, 0.00497437 | activations.jsonl + passages124.json |
| p92-04 | slot | Very tall tree | 3 / 3 | C | 12 | no | 0.0302734, 0.032959, 0.0195312, 0.0322266, 0.0194092, 0.0195312, 0.0196533, 0.0291748 | activations.jsonl + passages124.json |
| p92-43 | concept | Street rat | 2 / 2 | T | 32 | no | 0.00518799, -0.000957489, 0.00215149, -0.00335693, -0.00202942, -0.000276566, -0.00156403, 0.00125885 | activations.jsonl + passages124.json |
| p92-43 | concept | Street rat | 2 / 2 | F | 15 | no | -0.00469971, 0.0142822, 0.0839844, -0.00215149, 0.015625, -0.00515747, 0.00125122, -0.00799561 | activations.jsonl + passages124.json |
| p92-43 | concept | Street rat | 2 / 2 | C | 34 | no | 0.000579834, 0.00579834, -0.0209961, -0.00106049, -0.0153809, 0.000587463, -0.00299072, -0.00463867 | activations.jsonl + passages124.json |
| p92-13 | slot | Skeleton | 1 / 1 | T | 33 | no | 0.00386047, 0.0130005, 0.00848389, -0.00671387, 0.00497437, 0.00927734, 0.000728607, 0.0045166 | activations.jsonl + passages124.json |
| p92-13 | slot | Skeleton | 1 / 1 | F | 32 | no | 0.0187988, 0.0177002, 0.0102539, 0.00939941, 0.00521851, 0.0170898, 0.000518799, 0.00610352 | activations.jsonl + passages124.json |
| p92-13 | slot | Skeleton | 1 / 1 | C | 39 | no | -0.0162354, 0.00662231, -0.0196533, -0.013855, -0.0115356, -0.00279236, -0.0098877, -0.0109863 | activations.jsonl + passages124.json |
| p92-02 | slot | The town | 2 / 2 | T | 34 | no | 0.00320435, 0.00552368, 0.00634766, -0.000614166, 0.00897217, 0.00247192, 0.00454712, 0.00328064 | activations.jsonl + passages124.json |
| p92-02 | slot | The town | 2 / 2 | F | 32 | no | 0.00162506, 0.000225067, 0.0015564, 0.00128937, 0.00308228, 0.00164032, 0.00075531, -0.000606537 | activations.jsonl + passages124.json |
| p92-02 | slot | The town | 2 / 2 | C | 31 | no | 0.000762939, -0.000207901, -0.00139618, 6.10352e-05, 0.00326538, 0.000322342, 0.000614166, 0.0010376 | activations.jsonl + passages124.json |
| p92-05 | slot | Fox | 1 / 1 | T | 34 | no | -0.0037384, -0.00082016, -0.00280762, -0.0201416, -0.0263672, -0.00686646, -0.00421143, -0.0147095 | activations.jsonl + passages124.json |
| p92-05 | slot | Fox | 1 / 1 | F | 33 | no | -0.0114746, -0.000656128, -0.00325012, -0.0134277, -0.00463867, -0.0098877, -0.0161133, -0.0427246 | activations.jsonl + passages124.json |
| p92-05 | slot | Fox | 1 / 1 | C | 32 | no | 0.000148773, 0.00466919, 0.00668335, -0.00698853, 0.00952148, 0.000151634, -0.000322342, -0.00772095 | activations.jsonl + passages124.json |
| p92-18 | slot | Shadow of your smile | 4 / 4 | T | 34 | no | 0.00236511, 0.00164795, 0.00216675, 0.000743866, 0.00704956, 0.00334167, 0.000482559, 0.00352478 | activations.jsonl + passages124.json |
| p92-18 | slot | Shadow of your smile | 4 / 4 | F | 36 | no | 0.00120544, 0.00350952, -0.000606537, -0.000108242, 0.00165558, 0.00454712, 0.000392914, -3.71933e-05 | activations.jsonl + passages124.json |
| p92-18 | slot | Shadow of your smile | 4 / 4 | C | 32 | no | -0.000181198, 0.0022583, 0.0030365, -0.00094223, 0.00457764, 0.000667572, -0.000115871, -0.000587463 | activations.jsonl + passages124.json |
| p92-37 | slot | Large friendly dog | 3 / 3 | T | 34 | no | -0.0549316, -0.0235596, -0.0402832, 0.00209045, -0.017334, -0.0356445, -0.0137939, -0.0405273 | activations.jsonl + passages124.json |
| p92-37 | slot | Large friendly dog | 3 / 3 | F | 35 | no | -0.00616455, -0.000289917, -0.0483398, -0.00506592, -0.00927734, -0.0532227, -0.00531006, -0.00958252 | activations.jsonl + passages124.json |
| p92-37 | slot | Large friendly dog | 3 / 3 | C | 30 | no | -0.0088501, 0.0314941, -0.050293, 0.0317383, -0.0123291, -0.0412598, 0.00933838, 0.00753784 | activations.jsonl + passages124.json |
| p92-00 | concept | Boulder | 2 / 2 | T | 35 | no | 0.00668335, -0.0233154, -0.0181885, 0.00665283, -0.015564, -0.024292, 0.00686646, -0.0189209 | activations.jsonl + passages124.json |
| p92-00 | concept | Boulder | 2 / 2 | F | 38 | no | -0.0297852, -0.0245361, -0.0212402, -0.0112915, -0.0201416, -0.0322266, -0.022583, -0.0135498 | activations.jsonl + passages124.json |
| p92-00 | concept | Boulder | 2 / 2 | C | 38 | no | -0.0239258, -0.0181885, -0.0153198, -0.0109863, -0.0267334, -0.0300293, -0.0234375, -0.0132446 | activations.jsonl + passages124.json |
| p92-07 | slot | Cat | 1 / 1 | T | 37 | no | 0.000610352, -0.00170898, -0.00110626, 0.000896454, -0.00227356, 0.000896454, -0.00063324, 0.000193596 | activations.jsonl + passages124.json |
| p92-07 | slot | Cat | 1 / 1 | F | 31 | no | 0.00325012, 0.00286865, 0.00182343, 0.00256348, 0.00717163, 0.00352478, 0.000541687, 0.00151062 | activations.jsonl + passages124.json |
| p92-07 | slot | Cat | 1 / 1 | C | 33 | no | 0.000759125, 0.000526428, -3.67165e-05, -0.000130653, 0.00111389, 0.00125122, -0.00063324, -0.000142097 | activations.jsonl + passages124.json |
| p92-08 | concept | Dog | 1 / 1 | T | 38 | no | -0.0161133, -0.0111694, -0.0322266, 0.0019455, -0.0200195, -0.0128174, -0.0108032, 0.000511169 | activations.jsonl + passages124.json |
| p92-08 | concept | Dog | 1 / 1 | F | 38 | no | -0.0162354, -0.00927734, -0.00325012, -0.0192871, -0.0119019, -0.0231934, -0.0192871, 0.0109253 | activations.jsonl + passages124.json |
| p92-08 | concept | Dog | 1 / 1 | C | 37 | no | -0.00540161, -0.00497437, -0.00848389, -0.0158691, -0.00424194, -0.0114136, -0.0123291, 0.0020752 | activations.jsonl + passages124.json |

## Every own-phrase read and its top three

| Frame | Unit | Kind | Own phrase (rank) | Its top three, in rank order |
| --- | --- | --- | --- | --- |
| T | p92-01 | concept | Magical princess (1) | Magical princess; Very tall tree; Festival poet |
| T | p92-11 | concept | Whiskey (1) | Whiskey; Coca-Cola spokesperson; Embodiment of the void |
| T | p92-15 | concept | Secure password (1) | Secure password; String of code from your browser; Jacobite royalist veteran |
| T | p92-25 | slot | King Midas (1) | King Midas; 80s cover singer-songwriter; Fox |
| T | p92-28 | concept | Ancient nature deity (1) | Ancient nature deity; Coca-Cola spokesperson; The Internet |
| T | p92-38 | concept | Coca-Cola spokesperson (1) | Coca-Cola spokesperson; Friendly non-human entity; Whiskey |
| T | p92-39 | concept | NASCAR driver (1) | NASCAR driver; Large friendly dog; Secure password |
| T | p92-40 | concept | 80s cover singer-songwriter (1) | 80s cover singer-songwriter; Sir Ray Ray; Ancient nature deity |
| T | p92-42 | slot | Jacobite royalist veteran (1) | Jacobite royalist veteran; Chicken; Alex from Star Wars |
| T | p92-45 | concept | Iron-fisted dictator (1) | Iron-fisted dictator; Gruff old man; Secure password |
| T | p92-51 | concept | IP packet (1) | IP packet; The Internet; String of code from your browser |
| T | p92-09 | slot | Treat-loving dog (2) | Sir Ray Ray; Treat-loving dog; Ancient nature deity |
| T | p92-29 | concept | God of wisdom (2) | Coca-Cola spokesperson; God of wisdom; Embodiment of the void |
| T | p92-24 | slot | Queen of Hearts (3) | Magical princess; String of code from your browser; Queen of Hearts |
| F | p92-01 | concept | Magical princess (1) | Magical princess; Very tall tree; Festival poet |
| F | p92-11 | concept | Whiskey (1) | Whiskey; Embodiment of the void; Coca-Cola spokesperson |
| F | p92-15 | concept | Secure password (1) | Secure password; String of code from your browser; Jacobite royalist veteran |
| F | p92-25 | slot | King Midas (1) | King Midas; 80s cover singer-songwriter; Dr. Memento |
| F | p92-28 | concept | Ancient nature deity (1) | Ancient nature deity; Coca-Cola spokesperson; The Internet |
| F | p92-38 | concept | Coca-Cola spokesperson (1) | Coca-Cola spokesperson; Whiskey; Principled lawyer |
| F | p92-39 | concept | NASCAR driver (1) | NASCAR driver; Large friendly dog; Secure password |
| F | p92-40 | concept | 80s cover singer-songwriter (1) | 80s cover singer-songwriter; Sir Ray Ray; Ancient nature deity |
| F | p92-42 | slot | Jacobite royalist veteran (1) | Jacobite royalist veteran; Alex from Star Wars; Chicken |
| F | p92-45 | concept | Iron-fisted dictator (1) | Iron-fisted dictator; Gruff old man; Jacobite royalist veteran |
| F | p92-51 | concept | IP packet (1) | IP packet; The Internet; String of code from your browser |
| F | p92-29 | concept | God of wisdom (2) | Coca-Cola spokesperson; God of wisdom; Alex from Star Wars |
| F | p92-41 | concept | Human DJ (3) | 80s cover singer-songwriter; Festival poet; Human DJ |
| C | p92-01 | concept | Magical princess (1) | Magical princess; Very tall tree; Festival poet |
| C | p92-09 | slot | Treat-loving dog (1) | Treat-loving dog; Sir Ray Ray; 80s cover singer-songwriter |
| C | p92-11 | concept | Whiskey (1) | Whiskey; Embodiment of the void; Boulder |
| C | p92-15 | concept | Secure password (1) | Secure password; String of code from your browser; Jacobite royalist veteran |
| C | p92-24 | slot | Queen of Hearts (1) | Queen of Hearts; Coca-Cola spokesperson; King Midas |
| C | p92-25 | slot | King Midas (1) | King Midas; 80s cover singer-songwriter; Dr. Memento |
| C | p92-38 | concept | Coca-Cola spokesperson (1) | Coca-Cola spokesperson; Whiskey; 80s cover singer-songwriter |
| C | p92-39 | concept | NASCAR driver (1) | NASCAR driver; Large friendly dog; Secure password |
| C | p92-40 | concept | 80s cover singer-songwriter (1) | 80s cover singer-songwriter; Queen of Hearts; Human DJ |
| C | p92-42 | slot | Jacobite royalist veteran (1) | Jacobite royalist veteran; Chicken; Alex from Star Wars |
| C | p92-45 | concept | Iron-fisted dictator (1) | Iron-fisted dictator; Secure password; Gruff old man |
| C | p92-51 | concept | IP packet (1) | IP packet; String of code from your browser; The Internet |
| C | p92-29 | concept | God of wisdom (2) | Coca-Cola spokesperson; God of wisdom; Embodiment of the void |
| C | p92-41 | concept | Human DJ (3) | Festival poet; 80s cover singer-songwriter; Human DJ |

## Registered predictions

| # | Prediction, verbatim | Result | Verdict | Source |
| --- | --- | --- | --- | --- |
| 124.1 (low) | In T, at least 9 of 39 units read their persona, and the count is above the 95th percentile of the permutation null. | T 14/39; null median 4, p95 8; tail 0/10000 | held | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| 124.2 (low) | In T, the concept read rate is at least 15 percentage points above the slot read rate. | T concept 10/18 vs slot 4/21; rate difference 36.51 points | held | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| 124.3 (medium) | In T, at most 4 of 21 slot units read their persona. | T slot reads 4/21 | held | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
| 124.4 (low) | Of the units that read in T, at least half also read in C. | 13/14 T readers also read in C (92.9%) | held | activations.jsonl; read_set.json; run_meta.json; frozen passages124.json |
